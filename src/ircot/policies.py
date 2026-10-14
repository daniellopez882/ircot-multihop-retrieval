"""Query formulation: how a bridge entity becomes the next hop's retrieval query.

Three policies, each a different guess at what the next sub-question is:

* ``append``   question + bridge entity: keep everything, let the retriever sort it out.
* ``residual`` bridge entity + the question words the evidence passage did not already cover,
               an approximation of "#1 >> spouse" style decompositions.
* ``sentence`` the evidence sentence the bridge was read from, the closest extractive analogue of
               IRCoT using its latest chain-of-thought sentence as the query.
"""

from __future__ import annotations

import re
from typing import Protocol

from ircot.bridge import Bridge
from ircot.text import content_terms, fold

_WORDS = re.compile(r"[0-9A-Za-zÀ-ɏ']+")


class QueryPolicy(Protocol):
    name: str

    def formulate(self, question: str, bridge: Bridge) -> str: ...


class AppendPolicy:
    name = "append"

    def formulate(self, question: str, bridge: Bridge) -> str:
        return f"{question} {bridge.entity}"


class ResidualPolicy:
    name = "residual"

    def formulate(self, question: str, bridge: Bridge) -> str:
        covered = content_terms(bridge.passage) | content_terms(bridge.entity)
        rest = [
            w
            for w in _WORDS.findall(question)
            if content_terms(w) and not content_terms(w) <= covered
        ]
        # Nothing left to ask means the passage already covered the question; fall back to it.
        tail = " ".join(rest) if rest else question
        return f"{bridge.entity} {tail}"


class SentencePolicy:
    name = "sentence"

    def formulate(self, question: str, bridge: Bridge) -> str:
        sentence = bridge.sentence
        if fold(bridge.entity) not in fold(sentence):
            sentence = f"{bridge.entity}. {sentence}"
        return sentence


POLICIES: dict[str, QueryPolicy] = {
    p.name: p for p in (AppendPolicy(), ResidualPolicy(), SentencePolicy())
}
