"""The deduplicated, budgeted passage pool that a retrieval loop accumulates hop by hop."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ircot.retrieval import Hit
from ircot.text import content_terms, terms


def text_fingerprint(text: str) -> str:
    """Identity of a passage up to case, punctuation and whitespace."""
    return hashlib.blake2b(
        " ".join(terms(text, keep_stopwords=True)).encode(), digest_size=12
    ).hexdigest()


@dataclass(frozen=True)
class PassageStats:
    """Per-passage facts the loop needs on every hop, computed once per corpus."""

    tokens: np.ndarray
    fingerprints: list[str]
    term_sets: list[frozenset[str]]

    @classmethod
    def build(cls, passages: Sequence[str], count_tokens: Callable[[str], int]) -> PassageStats:
        return cls(
            tokens=np.array([count_tokens(p) for p in passages], dtype=np.int64),
            fingerprints=[text_fingerprint(p) for p in passages],
            term_sets=[frozenset(content_terms(p)) for p in passages],
        )

    def tokens_of(self, doc: int) -> int:
        return int(self.tokens[doc])

    def fingerprint_of(self, doc: int) -> str:
        return self.fingerprints[doc]


@dataclass(frozen=True)
class Entry:
    doc: int
    hop: int
    rank: int
    score: float


@dataclass(frozen=True)
class Admission:
    admitted: tuple[int, ...]
    duplicates: tuple[int, ...]
    over_budget: tuple[int, ...]


@dataclass
class PassagePool:
    """Admits retrieved passages in rank order until a passage-count or token budget is hit.

    A passage is a duplicate if its id is already pooled or its text fingerprint matches a pooled
    passage (the same paragraph filed under two titles). Admission is first-come: a later, better
    passage never evicts an earlier one, so the pool a reasoner saw at hop t is still a prefix of
    the final pool and traces stay easy to read.
    """

    tokens_of: Callable[[int], int]
    fingerprint_of: Callable[[int], str]
    max_passages: int | None = None
    max_tokens: int | None = None
    entries: list[Entry] = field(default_factory=list)
    tokens_used: int = 0
    _docs: set[int] = field(default_factory=set)
    _prints: set[str] = field(default_factory=set)

    def __contains__(self, doc: int) -> bool:
        return doc in self._docs

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def docs(self) -> list[int]:
        return [e.doc for e in self.entries]

    def admit(self, hop: int, hits: Sequence[Hit]) -> Admission:
        admitted: list[int] = []
        duplicates: list[int] = []
        over_budget: list[int] = []
        for hit in hits:
            fp = self.fingerprint_of(hit.doc)
            if hit.doc in self._docs or fp in self._prints:
                duplicates.append(hit.doc)
                continue
            cost = self.tokens_of(hit.doc)
            count_ok = self.max_passages is None or len(self.entries) < self.max_passages
            tokens_ok = self.max_tokens is None or self.tokens_used + cost <= self.max_tokens
            if not (count_ok and tokens_ok):
                over_budget.append(hit.doc)
                continue
            self.entries.append(Entry(hit.doc, hop, len(admitted), hit.score))
            self._docs.add(hit.doc)
            self._prints.add(fp)
            self.tokens_used += cost
            admitted.append(hit.doc)
        return Admission(tuple(admitted), tuple(duplicates), tuple(over_budget))

    def ranking(self) -> list[int]:
        """Round-robin over hops: every hop's best passage, then every hop's second, and so on.

        This is how a pool built by several queries is turned into one ranked list for Recall@k;
        hop order breaks ties, so the first-hop passage leads each round.
        """
        return [e.doc for e in sorted(self.entries, key=lambda e: (e.rank, e.hop))]
