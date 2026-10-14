from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ircot.offline import HashingEncoder
from ircot.retrieval import BM25, Dense, Hybrid, fuse, search, top_k

DOCS = [
    "Grant Green recorded for Blue Note in 1961.",
    "Steve Hillage married Miquette Giraudy.",
    "Green is an album by Steve Hillage, a guitarist from London.",
]


def bm25_by_hand(docs: list[str], query: list[str], k1: float, b: float) -> list[float]:
    from ircot.text import terms

    toks = [terms(d) for d in docs]
    avg = sum(map(len, toks)) / len(toks)
    out = []
    for t in toks:
        s = 0.0
        for q in query:
            df = sum(q in u for u in toks)
            idf = math.log(1 + (len(toks) - df + 0.5) / (df + 0.5))
            tf = t.count(q)
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avg))
        out.append(s)
    return out


def test_bm25_matches_the_formula() -> None:
    index = BM25(DOCS, k1=0.9, b=0.4)
    got = index.scores("Steve Hillage green")
    expected = bm25_by_hand(DOCS, ["steve", "hillage", "green"], 0.9, 0.4)
    np.testing.assert_allclose(got, expected)


def test_bm25_never_returns_passages_without_a_shared_term() -> None:
    index = BM25(DOCS)
    hits = search(index, "Miquette", 3)
    assert [h.doc for h in hits] == [1]
    assert search(index, "zebra", 3) == []


def test_repeated_query_terms_weigh_more() -> None:
    index = BM25(DOCS)
    once, twice = index.scores("green"), index.scores("green green")
    np.testing.assert_allclose(twice[np.isfinite(twice)], 2 * once[np.isfinite(once)])


def test_top_k_breaks_ties_by_passage_id_and_honours_exclusions() -> None:
    scores = np.array([1.0, 3.0, 3.0, -np.inf, 2.0, 3.0])
    assert [h.doc for h in top_k(scores, 3)] == [1, 2, 5]
    assert [h.doc for h in top_k(scores, 3, exclude={2})] == [1, 5, 4]
    assert [h.doc for h in top_k(scores, 10)] == [1, 2, 5, 4, 0]
    assert top_k(scores, 0) == []


@given(st.lists(st.floats(-5, 5, allow_nan=False), min_size=1, max_size=40), st.integers(1, 45))
def test_top_k_is_the_sorted_prefix(values: list[float], k: int) -> None:
    scores = np.array(values)
    hits = top_k(scores, k)
    assert len(hits) == min(k, len(values))
    expected = sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]
    assert [h.doc for h in hits] == expected


def test_dense_search_is_exact_inner_product() -> None:
    enc = HashingEncoder(dim=32)
    emb = enc.encode_passages(DOCS)
    dense = Dense(emb, enc)
    q = enc.encode_queries(["Miquette Giraudy"])[0]
    np.testing.assert_allclose(dense.scores("Miquette Giraudy"), emb @ q, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)


def test_fusion_endpoints_reduce_to_each_retriever() -> None:
    rng = np.random.default_rng(0)
    sparse, dense = rng.random(50), rng.random(50)
    only_dense = fuse(sparse, dense, alpha=1.0, depth=50)
    only_sparse = fuse(sparse, dense, alpha=0.0, depth=50)
    assert np.argsort(-only_dense).tolist() == np.argsort(-dense).tolist()
    assert np.argsort(-only_sparse).tolist() == np.argsort(-sparse).tolist()
    mixed = fuse(sparse, dense, alpha=0.5, depth=50)
    assert mixed.min() >= 0.0
    assert mixed.max() <= 1.0


def test_fusion_only_scores_the_candidate_union() -> None:
    sparse = np.array([5.0, 4.0, 0.0, 0.0, -np.inf])
    dense = np.array([0.0, 0.0, 0.0, 0.9, 0.1])
    fused = fuse(sparse, dense, alpha=0.5, depth=1)
    assert np.isfinite(fused).tolist() == [True, False, False, True, False]


def test_hybrid_rejects_bad_weights() -> None:
    enc = HashingEncoder(dim=8)
    with pytest.raises(ValueError, match="alpha"):
        Hybrid(BM25(DOCS), Dense(enc.encode_passages(DOCS), enc), alpha=1.5)
