"""Deterministic test doubles: scripted reasoner and reader, fixed-score retriever."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from ircot.data import Question
from ircot.reader import Span
from ircot.reasoners import NO_MARGIN, Thought


@dataclass
class ScriptedReasoner:
    """Returns a fixed sequence of thoughts, one per hop, and records what it was shown."""

    script: Sequence[Thought]
    name: str = "scripted"
    seen: list[list[int]] = field(default_factory=list)
    started: int = 0

    def start(self, question: Question) -> None:
        self.started += 1
        self.seen = []

    def step(self, hop: int, new_docs: Sequence[int]) -> Thought:
        self.seen.append(list(new_docs))
        if hop - 1 < len(self.script):
            return self.script[hop - 1]
        return Thought(None, None, "", NO_MARGIN)


def thought(
    query: str | None,
    *,
    answer: str = "",
    margin: float = 0.0,
    read: tuple[int, ...] = (),
    tokens: int = 0,
    done: bool = False,
) -> Thought:
    return Thought(query, None, answer, margin, read, tokens, done)


class TableRetriever:
    """Scores come from a per-query table; unknown queries score every passage at zero."""

    name = "table"

    def __init__(self, n_docs: int, table: Mapping[str, Sequence[float]]) -> None:
        self.n_docs = n_docs
        self.table = {q: np.asarray(v, dtype=np.float64) for q, v in table.items()}
        self.queries: list[str] = []

    def scores(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return self.table.get(query, np.zeros(self.n_docs)).copy()


class ScriptedReader:
    """Spans looked up by passage text; passages without an entry yield no span."""

    name = "scripted-reader"

    def __init__(self, spans: Mapping[str, Sequence[tuple[str, float]]]) -> None:
        self.spans = spans
        self.calls: list[tuple[str, list[str]]] = []

    def read(self, question: str, passages: Sequence[str], top_n: int = 3) -> list[list[Span]]:
        self.calls.append((question, list(passages)))
        out: list[list[Span]] = []
        for passage in passages:
            found = []
            for text, margin in self.spans.get(passage, [])[:top_n]:
                start = passage.find(text)
                found.append(Span(text, start, start + len(text), margin, 0.0))
            out.append(found)
        return out
