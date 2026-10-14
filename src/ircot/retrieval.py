"""BM25, exact dense search and their score-fused hybrid over one in-memory corpus.

Corpora here are at most a few tens of thousands of passages, so every retriever scores the whole
corpus per query (a sparse matrix-vector product for BM25, a dense one for embeddings) and takes
the top ``k`` exactly. There is no approximate index to tune and no recall lost to one.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from ircot.encoders import Encoder
from ircot.text import terms


@dataclass(frozen=True)
class Hit:
    doc: int
    score: float


class Retriever(Protocol):
    name: str

    def scores(self, query: str) -> np.ndarray:
        """One score per corpus passage; ``-inf`` marks passages the retriever will not return."""
        ...


def top_k(scores: np.ndarray, k: int, exclude: Collection[int] = ()) -> list[Hit]:
    """The ``k`` best passages, ties broken by lower passage id so runs are reproducible."""
    s = np.array(scores, dtype=np.float64)
    if exclude:
        s[np.fromiter(exclude, dtype=np.int64)] = -np.inf
    finite = np.flatnonzero(np.isfinite(s))
    if k <= 0 or finite.size == 0:
        return []
    k = min(k, finite.size)
    kth = np.partition(s[finite], finite.size - k)[finite.size - k]
    cand = finite[s[finite] >= kth]
    order = np.lexsort((cand, -s[cand]))[:k]
    return [Hit(int(cand[i]), float(s[cand[i]])) for i in order]


def search(retriever: Retriever, query: str, k: int, exclude: Collection[int] = ()) -> list[Hit]:
    return top_k(retriever.scores(query), k, exclude)


class BM25:
    """Okapi BM25 with the Lucene idf, ``log(1 + (N - df + 0.5) / (df + 0.5))``, which stays
    positive for terms in more than half of the passages."""

    name = "bm25"

    def __init__(self, passages: Sequence[str], k1: float = 0.9, b: float = 0.4) -> None:
        self.k1, self.b = k1, b
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        lengths = np.zeros(len(passages), dtype=np.float64)
        for doc, passage in enumerate(passages):
            counts = Counter(terms(passage))
            lengths[doc] = sum(counts.values())
            for term, tf in counts.items():
                rows.append(doc)
                cols.append(vocab.setdefault(term, len(vocab)))
                vals.append(tf)
        self.vocab = vocab
        n_docs = len(passages)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(n_docs, len(vocab)), dtype=np.float64)
        df = np.bincount(np.asarray(cols, dtype=np.int64), minlength=len(vocab))
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        avg = lengths.mean() if n_docs else 0.0
        norm = k1 * (1.0 - b + b * lengths / avg) if avg > 0 else np.full(n_docs, k1)
        data = tf.data
        row_of = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        weighted = data * (k1 + 1.0) / (data + norm[row_of]) * self.idf[tf.indices]
        self.weights = sp.csr_matrix((weighted, tf.indices, tf.indptr), shape=tf.shape).tocsc()
        self.n_docs = n_docs

    def scores(self, query: str) -> np.ndarray:
        """BM25 scores, with ``-inf`` for passages that share no term with the query."""
        out = np.full(self.n_docs, -np.inf)
        q = Counter(t for t in terms(query) if t in self.vocab)
        if not q:
            return out
        cols = np.array([self.vocab[t] for t in q], dtype=np.int64)
        qtf = np.array(list(q.values()), dtype=np.float64)
        sub = self.weights[:, cols]
        matched = np.unique(sub.indices)
        out[matched] = np.asarray(sub @ qtf).ravel()[matched]
        return out


class Dense:
    """Exact inner-product search over L2-normalised passage embeddings.

    Query embeddings are memoised by query text: every configuration starts from the same
    question, and the fixed-hop loops repeat each other's first hops. ``forget_queries`` empties
    the memo so a latency measurement pays for every encoding.
    """

    name = "dense"

    def __init__(self, embeddings: np.ndarray, encoder: Encoder) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a [passages, dim] matrix")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.encoder = encoder
        self._queries: dict[str, np.ndarray] = {}

    def scores(self, query: str) -> np.ndarray:
        q = self._queries.get(query)
        if q is None:
            q = self._queries[query] = self.encoder.encode_queries([query])[0]
        return (self.embeddings @ q).astype(np.float64)

    def forget_queries(self) -> None:
        self._queries.clear()


class Hybrid:
    """Convex combination of min-max normalised BM25 and dense scores.

    Normalisation runs over the union of each retriever's top ``depth`` passages, the set that can
    realistically end up in a result list; passages outside that union are not returned.
    """

    name = "hybrid"

    def __init__(self, bm25: BM25, dense: Dense, alpha: float = 0.5, depth: int = 100) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.bm25, self.dense, self.alpha, self.depth = bm25, dense, alpha, depth

    def scores(self, query: str) -> np.ndarray:
        return fuse(self.bm25.scores(query), self.dense.scores(query), self.alpha, self.depth)


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if hi - lo <= 0.0:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def fuse(sparse: np.ndarray, dense: np.ndarray, alpha: float, depth: int) -> np.ndarray:
    # A passage BM25 does not match at all scores at BM25's floor, which is zero.
    sparse = np.where(np.isfinite(sparse), sparse, 0.0)
    n = sparse.shape[0]
    depth = min(depth, n)
    pool = np.union1d(
        np.argpartition(-sparse, depth - 1)[:depth], np.argpartition(-dense, depth - 1)[:depth]
    )
    fused = np.full(n, -np.inf)
    fused[pool] = alpha * _minmax(dense[pool]) + (1.0 - alpha) * _minmax(sparse[pool])
    return fused
