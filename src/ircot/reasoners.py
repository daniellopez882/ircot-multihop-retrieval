"""Reasoners: what happens between two retrieval steps of the loop.

A reasoner looks at the passages the last hop admitted and proposes the next query (or none).
Swapping the reasoner while holding the retriever, pool and budgets fixed is how the benchmark
separates the value of *iterating* from the value of *reading*:

* ``ExtractiveReasoner`` reads new passages with the SQuAD 2.0 reader, picks a bridge entity and
  turns it into a query with one of the policies. This is the offline IRCoT reasoner.
* ``TitleFeedbackReasoner`` reads nothing: the next query is the question plus the title of the
  best new passage (pseudo-relevance feedback). Iteration without a reasoner.
* ``OracleBridgeReasoner`` uses MuSiQue's gold intermediate answers as bridges, the ceiling for
  any bridge-based reasoner on that dataset.
* ``ChainOfThoughtReasoner`` asks an LLM for the next reasoning sentence and uses it as the query,
  as in the IRCoT paper. It needs an endpoint and is not part of the recorded benchmark.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ircot.bridge import Bridge, Candidate, TitleIndex, select_bridge
from ircot.data import Corpus, Question
from ircot.llm import ChatProvider
from ircot.policies import AppendPolicy, QueryPolicy
from ircot.reader import Reader, Span
from ircot.text import normalize_answer, title_key

NO_MARGIN = -30.0  # stands in for "no span read yet" in features; below any margin observed


@dataclass(frozen=True)
class Thought:
    query: str | None
    bridge: Bridge | None
    answer: str
    answer_margin: float
    passages_read: int
    llm_tokens: int = 0
    says_done: bool = False
    note: str = ""


class Reasoner(Protocol):
    name: str

    def start(self, question: Question) -> None: ...

    def step(self, hop: int, new_docs: Sequence[int]) -> Thought: ...


class ReaderCache:
    """Reader outputs keyed by (question text, passage id).

    Within one question this is what any implementation would do (never read a passage twice).
    The benchmark also shares one cache across configurations so identical (question, passage)
    pairs are computed once; latency is measured separately with a fresh cache per question.
    With ``path`` set, the cache is loaded from and saved to a JSON file.
    """

    def __init__(
        self, reader: Reader, corpus: Corpus, top_n: int = 3, path: Path | None = None
    ) -> None:
        self.reader, self.corpus, self.top_n, self.path = reader, corpus, top_n, path
        self._spans: dict[tuple[str, int], list[Span]] = {}
        self.computed = 0
        if path is not None and path.exists():
            for question, doc, spans in json.loads(path.read_text(encoding="utf-8")):
                self._spans[(question, doc)] = [Span(*s) for s in spans]

    def spans(self, question: str, docs: Sequence[int]) -> list[list[Span]]:
        missing = [d for d in dict.fromkeys(docs) if (question, d) not in self._spans]
        if missing:
            texts = [self.corpus.texts[d] for d in missing]
            read = self.reader.read(question, texts, self.top_n)
            for doc, spans in zip(missing, read, strict=True):
                self._spans[(question, doc)] = spans
            self.computed += len(missing)
        return [self._spans[(question, d)] for d in docs]

    def save(self) -> None:
        if self.path is None:
            return
        rows = [
            [q, d, [[s.text, s.start, s.end, s.score, s.null_score] for s in spans]]
            for (q, d), spans in self._spans.items()
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)


def best_answer(cache: ReaderCache, question: str, docs: Sequence[int]) -> tuple[str, float]:
    """The highest-margin span over ``docs`` (the answer step shared by every configuration)."""
    best_text, best_margin = "", NO_MARGIN
    for spans in cache.spans(question, docs):
        for span in spans:
            if span.margin > best_margin:
                best_text, best_margin = span.text, span.margin
    return best_text, best_margin


@dataclass
class ExtractiveReasoner:
    """Reads the top new passages of each hop and bridges on the best unused span.

    The newest hop's passages were retrieved for the current bridge, so their spans are tried
    first. Only when none of them qualifies does the reasoner fall back to spans from earlier
    hops, which still lets an entity read early (but not used) carry the chain forward.
    """

    cache: ReaderCache
    titles: TitleIndex
    policy: QueryPolicy
    read_per_hop: int = 3
    title_bonus: float = 4.0
    name: str = "extractive"
    _question: Question | None = None
    _used: set[str] = field(default_factory=set)
    _candidates: list[Candidate] = field(default_factory=list)
    _best: tuple[str, float] = ("", NO_MARGIN)

    def start(self, question: Question) -> None:
        self._question = question
        self._used = set()
        self._candidates = []
        self._best = ("", NO_MARGIN)

    def step(self, hop: int, new_docs: Sequence[int]) -> Thought:
        assert self._question is not None, "start() must be called before step()"
        q = self._question
        corpus = self.cache.corpus
        docs = list(new_docs[: self.read_per_hop])
        fresh: list[Candidate] = []
        for doc, spans in zip(docs, self.cache.spans(q.text, docs), strict=True):
            for span in spans:
                if span.margin > self._best[1]:
                    self._best = (span.text, span.margin)
                fresh.append(Candidate(span, doc, corpus.titles[doc], corpus.texts[doc]))
        self._candidates.extend(fresh)
        bridge = select_bridge(q.text, fresh, self.titles, self._used, self.title_bonus)
        if bridge is None:
            bridge = select_bridge(
                q.text, self._candidates, self.titles, self._used, self.title_bonus
            )
        query = None
        if bridge is not None:
            self._used.add(title_key(bridge.entity))
            query = self.policy.formulate(q.text, bridge)
        return Thought(query, bridge, self._best[0], self._best[1], len(docs))


@dataclass
class TitleFeedbackReasoner:
    corpus: Corpus
    name: str = "title-feedback"
    _question: Question | None = None
    _used: set[str] = field(default_factory=set)

    def start(self, question: Question) -> None:
        self._question = question
        self._used = set()

    def step(self, hop: int, new_docs: Sequence[int]) -> Thought:
        assert self._question is not None, "start() must be called before step()"
        for doc in new_docs:
            title = self.corpus.titles[doc]
            if title_key(title) not in self._used:
                self._used.add(title_key(title))
                bridge = Bridge(title, title, doc, title, self.corpus.texts[doc], 0.0, True)
                return Thought(f"{self._question.text} {title}", bridge, "", NO_MARGIN, 0)
        return Thought(None, None, "", NO_MARGIN, 0)


@dataclass
class OracleBridgeReasoner:
    """Gold intermediate answers as bridges: hop t queries with the answer of sub-question t."""

    policy: QueryPolicy = field(default_factory=AppendPolicy)
    name: str = "oracle-bridge"
    _question: Question | None = None

    def start(self, question: Question) -> None:
        if not question.decomposition:
            raise ValueError(f"{question.qid} has no decomposition; the oracle needs MuSiQue")
        self._question = question

    def step(self, hop: int, new_docs: Sequence[int]) -> Thought:
        assert self._question is not None, "start() must be called before step()"
        steps = self._question.decomposition
        if hop >= len(steps):
            return Thought(None, None, "", NO_MARGIN, 0)
        entity = steps[hop - 1].answer
        bridge = Bridge(entity, entity, -1, entity, "", 0.0, False)
        return Thought(self.policy.formulate(self._question.text, bridge), bridge, "", NO_MARGIN, 0)


_ANSWER = re.compile(r"answer is[:\s]+(.+?)\s*\.?\s*$", re.IGNORECASE | re.DOTALL)

COT_SYSTEM = (
    "You answer multi-hop questions one reasoning step at a time using only the given paragraphs. "
    "Reply with exactly one short sentence: the next step of the reasoning. When the paragraphs "
    "are enough to answer, reply with a sentence ending in 'So the answer is: <answer>.'"
)


@dataclass
class ChainOfThoughtReasoner:
    """IRCoT's reasoner: the newest chain-of-thought sentence becomes the next retrieval query."""

    provider: ChatProvider
    corpus: Corpus
    max_paragraphs: int = 15
    max_words_per_paragraph: int = 120
    max_new_tokens: int = 64
    name: str = "chain-of-thought"
    _question: Question | None = None
    _seen: list[int] = field(default_factory=list)
    _thoughts: list[str] = field(default_factory=list)

    def start(self, question: Question) -> None:
        self._question = question
        self._seen = []
        self._thoughts = []

    def _prompt(self) -> list[dict[str, str]]:
        assert self._question is not None
        paragraphs = []
        for doc in self._seen[-self.max_paragraphs :]:
            words = self.corpus.texts[doc].split()[: self.max_words_per_paragraph]
            paragraphs.append(f"Title: {self.corpus.titles[doc]}\n{' '.join(words)}")
        user = "\n\n".join(paragraphs) + f"\n\nQuestion: {self._question.text}\n"
        if self._thoughts:
            user += "Reasoning so far: " + " ".join(self._thoughts) + "\n"
        user += "Next sentence:"
        return [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": user}]

    def step(self, hop: int, new_docs: Sequence[int]) -> Thought:
        assert self._question is not None, "start() must be called before step()"
        self._seen.extend(new_docs)
        completion = self.provider.complete(self._prompt(), max_tokens=self.max_new_tokens)
        sentence = completion.text.strip().split("\n")[0].strip()
        if not sentence or normalize_answer(sentence) in {
            normalize_answer(t) for t in self._thoughts
        }:
            return Thought(None, None, "", NO_MARGIN, 0, completion.total_tokens, note=sentence)
        self._thoughts.append(sentence)
        found = _ANSWER.search(sentence)
        if found:
            answer = found.group(1).strip().rstrip(".")
            return Thought(None, None, answer, 0.0, 0, completion.total_tokens, True, sentence)
        return Thought(sentence, None, "", NO_MARGIN, 0, completion.total_tokens, note=sentence)
