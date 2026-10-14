"""The interleaved retrieve / reason / decide loop as an explicit state machine.

    RETRIEVE --> REASON --> DECIDE --> RETRIEVE ...
        \\                     \\
         `--> DONE (one step)   `--> DONE (budget, no new query, reasoner done, controller)

Budgets are hard limits checked before every new retrieval: the number of retrieval steps, the
pool's token budget, and (for LLM reasoners) the reasoner's own token spend. On the last allowed
step the loop skips REASON, because nothing could use the thought.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np

from ircot.controller import FEATURES, StopController
from ircot.data import Question
from ircot.pool import Admission, PassagePool, PassageStats
from ircot.reasoners import Reasoner, Thought
from ircot.retrieval import Hit, Retriever, top_k
from ircot.text import content_terms, normalize_answer


class State(Enum):
    RETRIEVE = "retrieve"
    REASON = "reason"
    DECIDE = "decide"
    DONE = "done"


@dataclass(frozen=True)
class Budget:
    max_steps: int
    k: int
    max_pool_tokens: int | None = None
    max_pool_passages: int | None = None
    max_reasoner_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.k < 1:
            raise ValueError("a loop needs at least one step retrieving at least one passage")


class StopRule(Protocol):
    name: str

    def decide(self, features: np.ndarray) -> tuple[bool, float | None]: ...


class NeverStop:
    """Run until a budget or the reasoner ends the loop (the fixed-hop baselines)."""

    name = "never"

    def decide(self, features: np.ndarray) -> tuple[bool, float | None]:
        return False, None


@dataclass
class LearnedStop:
    controller: StopController
    name: str = "learned"

    def decide(self, features: np.ndarray) -> tuple[bool, float | None]:
        p = self.controller.probability(features)
        return p >= self.controller.threshold, p


@dataclass
class HopRecord:
    hop: int
    query: str
    hits: list[tuple[int, float]]
    admitted: list[int]
    duplicates: list[int]
    over_budget: list[int]
    read: list[int] = field(default_factory=list)
    bridge: str | None = None
    bridge_title_match: bool = False
    answer: str = ""
    thought: str = ""
    features: dict[str, float] | None = None
    stop_probability: float | None = None
    decision: str = ""
    elapsed_ms: float = 0.0


@dataclass
class LoopResult:
    qid: str
    ranking: list[int]
    hops: list[HopRecord]
    stop_reason: str
    retrieval_calls: int
    passages_read: int
    reasoner_tokens: int
    pool_tokens: int
    states: list[str] = field(default_factory=list)


def stop_features(
    hop: int,
    k: int,
    thought: Thought,
    previous: Thought | None,
    admission: Admission,
    hits: list[Hit],
    pool_terms: set[str],
    question_terms: set[str],
) -> np.ndarray:
    """The controller's view of the loop after ``hop`` (order fixed by ``controller.FEATURES``)."""
    prev_margin = previous.answer_margin if previous is not None else thought.answer_margin
    stable = previous is not None and normalize_answer(previous.answer) == normalize_answer(
        thought.answer
    )
    coverage = len(question_terms & pool_terms) / len(question_terms) if question_terms else 1.0
    values = {
        "hop": float(hop),
        "answer_margin": thought.answer_margin,
        "margin_gain": thought.answer_margin - prev_margin,
        "answer_stable": float(stable),
        "bridge_found": float(thought.bridge is not None),
        "bridge_title_match": float(thought.bridge is not None and thought.bridge.title_match),
        "novelty": len(admission.admitted) / k,
        "top_score": hits[0].score if hits else 0.0,
        "question_coverage": coverage,
    }
    return np.array([values[name] for name in FEATURES])


class InterleavedLoop:
    def __init__(
        self,
        retriever: Retriever,
        budget: Budget,
        *,
        stats: PassageStats,
        reasoner: Reasoner | None = None,
        stop: StopRule | None = None,
        exclude_seen: bool = True,
    ) -> None:
        if budget.max_steps > 1 and reasoner is None:
            raise ValueError("a multi-step loop needs a reasoner to propose the next query")
        self.retriever = retriever
        self.budget = budget
        self.reasoner = reasoner
        self.stop = stop or NeverStop()
        self.stats = stats
        self.exclude_seen = exclude_seen

    def run(self, question: Question) -> LoopResult:
        b = self.budget
        pool = PassagePool(
            self.stats.tokens_of, self.stats.fingerprint_of, b.max_pool_passages, b.max_pool_tokens
        )
        if self.reasoner is not None:
            self.reasoner.start(question)
        q_terms = content_terms(question.text)
        pool_terms: set[str] = set()
        hops: list[HopRecord] = []
        states: list[str] = []
        state, query, hop = State.RETRIEVE, question.text, 0
        thought: Thought | None = None
        previous: Thought | None = None
        hits: list[Hit] = []
        admission = Admission((), (), ())
        retrieval_calls = passages_read = reasoner_tokens = 0
        stop_reason = ""
        started = time.perf_counter()

        while state is not State.DONE:
            states.append(state.value)
            if state is State.RETRIEVE:
                hop += 1
                exclude = pool.docs if self.exclude_seen else ()
                hits = top_k(self.retriever.scores(query), b.k, exclude)
                retrieval_calls += 1
                admission = pool.admit(hop, hits)
                for doc in admission.admitted:
                    pool_terms |= self.stats.term_sets[doc]
                hops.append(
                    HopRecord(
                        hop,
                        query,
                        [(h.doc, round(h.score, 6)) for h in hits],
                        list(admission.admitted),
                        list(admission.duplicates),
                        list(admission.over_budget),
                    )
                )
                if self.reasoner is None or hop >= b.max_steps:
                    state, stop_reason = State.DONE, "step-budget"
                else:
                    state = State.REASON
            elif state is State.REASON:
                assert self.reasoner is not None
                previous, thought = thought, self.reasoner.step(hop, list(admission.admitted))
                passages_read += thought.passages_read
                reasoner_tokens += thought.llm_tokens
                rec = hops[-1]
                rec.read = list(thought.read)
                rec.bridge = thought.bridge.entity if thought.bridge else None
                rec.bridge_title_match = bool(thought.bridge and thought.bridge.title_match)
                rec.answer = thought.answer
                rec.thought = thought.note
                state = State.DECIDE
            else:  # DECIDE
                assert thought is not None
                rec = hops[-1]
                x = stop_features(hop, b.k, thought, previous, admission, hits, pool_terms, q_terms)
                rec.features = {
                    name: round(float(v), 6) for name, v in zip(FEATURES, x, strict=True)
                }
                reason = self._budget_reason(pool, reasoner_tokens)
                if reason is None and thought.says_done:
                    reason = "reasoner-done"
                if reason is None and thought.query is None:
                    reason = "no-new-query"
                if reason is None:
                    stop_now, p = self.stop.decide(x)
                    rec.stop_probability = None if p is None else round(p, 6)
                    if stop_now:
                        reason = f"controller:{self.stop.name}"
                if reason is None:
                    rec.decision = "continue"
                    query = thought.query or question.text
                    state = State.RETRIEVE
                else:
                    rec.decision = f"stop:{reason}"
                    state, stop_reason = State.DONE, reason
            hops[-1].elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)

        if hops and not hops[-1].decision:
            hops[-1].decision = f"stop:{stop_reason}"
        return LoopResult(
            qid=question.qid,
            ranking=pool.ranking(),
            hops=hops,
            stop_reason=stop_reason,
            retrieval_calls=retrieval_calls,
            passages_read=passages_read,
            reasoner_tokens=reasoner_tokens,
            pool_tokens=pool.tokens_used,
            states=states,
        )

    def _budget_reason(self, pool: PassagePool, reasoner_tokens: int) -> str | None:
        b = self.budget
        if b.max_pool_tokens is not None and pool.tokens_used >= b.max_pool_tokens:
            return "token-budget"
        if b.max_pool_passages is not None and len(pool) >= b.max_pool_passages:
            return "passage-budget"
        if b.max_reasoner_tokens is not None and reasoner_tokens >= b.max_reasoner_tokens:
            return "reasoner-token-budget"
        return None
