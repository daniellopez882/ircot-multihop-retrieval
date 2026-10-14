"""Bridge-entity extraction: pick the span that names the next hop, preferring corpus titles.

In MuSiQue and HotpotQA the passage that answers the next sub-question is usually the Wikipedia
paragraph *about* the bridge entity, so a reader span that equals another passage's title is a
much stronger next-hop signal than a span that merely scores well.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ircot.reader import Span
from ircot.text import content_terms, sentence_at, title_key

_DISAMBIGUATION = re.compile(r"\s*\([^)]*\)\s*$")


@dataclass(frozen=True)
class Bridge:
    span: str
    entity: str
    doc: int
    sentence: str
    passage: str
    margin: float
    title_match: bool


class TitleIndex:
    """Exact title lookup on the full title and on the title without its "(disambiguation)"."""

    def __init__(self, titles: Sequence[str]) -> None:
        self._full: dict[str, str] = {}
        self._base: dict[str, str] = {}
        for title in titles:
            self._full.setdefault(title_key(title), title)
            self._base.setdefault(title_key(_DISAMBIGUATION.sub("", title)), title)

    def match(self, span: str) -> str | None:
        key = title_key(span)
        if not key:
            return None
        return self._full.get(key) or self._base.get(key)


@dataclass(frozen=True)
class Candidate:
    span: Span
    doc: int
    title: str
    passage: str


def select_bridge(
    question: str,
    candidates: Iterable[Candidate],
    titles: TitleIndex,
    used: set[str],
    title_bonus: float,
) -> Bridge | None:
    """The best new bridge: reader margin plus ``title_bonus`` when the span is another passage's
    title. Spans that only repeat the question, or that were already used, are skipped."""
    q_terms = content_terms(question)
    best: tuple[float, Bridge] | None = None
    for c in candidates:
        key = title_key(c.span.text)
        span_terms = content_terms(c.span.text)
        if not key or key in used or not span_terms or span_terms <= q_terms:
            continue
        matched = titles.match(c.span.text)
        # A span naming the passage it was read from is that passage's subject, not a new hop.
        linked = matched is not None and title_key(matched) != title_key(c.title)
        entity = matched if linked and matched is not None else c.span.text
        if title_key(entity) in used:
            continue
        value = c.span.margin + (title_bonus if linked else 0.0)
        if best is None or value > best[0]:
            bridge = Bridge(
                span=c.span.text,
                entity=entity,
                doc=c.doc,
                sentence=sentence_at(c.passage, c.span.start),
                passage=c.passage,
                margin=c.span.margin,
                title_match=linked,
            )
            best = (value, bridge)
    return None if best is None else best[1]
