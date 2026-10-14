"""The loop state machine driven by a scripted reasoner: hop sequences, budgets, stop rules."""

from __future__ import annotations

import numpy as np
import pytest

from fakes import ScriptedReasoner, TableRetriever, thought
from ircot.controller import FEATURES, StopController
from ircot.data import Paragraph, Question
from ircot.loop import Budget, InterleavedLoop, LearnedStop, NeverStop
from ircot.pool import PassageStats

N = 8
PASSAGES = [f"passage number {i} about topic {i % 3}" for i in range(N)]
Q = Question("q1", "test", "which topic passage", ("x",), (Paragraph("t", "p", True),), 2, "2hop")

# Each query ranks passages differently, so every hop has an unambiguous top-k.
TABLE = {
    "which topic passage": [8, 7, 6, 5, 4, 3, 2, 1],
    "hop two": [1, 2, 3, 4, 5, 6, 8, 7],
    "hop three": [9, 1, 1, 9, 1, 8, 1, 1],
}


def stats(tokens: int = 10) -> PassageStats:
    return PassageStats(
        tokens=np.full(N, tokens),
        fingerprints=[str(i) for i in range(N)],
        term_sets=[frozenset()] * N,
    )


def make(budget: Budget, script=None, stop=None, tokens: int = 10, exclude_seen: bool = True):  # type: ignore[no-untyped-def]
    retriever = TableRetriever(N, TABLE)
    reasoner = ScriptedReasoner(script) if script is not None else None
    loop = InterleavedLoop(
        retriever,
        budget,
        stats=stats(tokens),
        reasoner=reasoner,
        stop=stop,
        exclude_seen=exclude_seen,
    )
    return loop, retriever, reasoner


def test_one_shot_is_a_single_retrieve_state() -> None:
    loop, retriever, _ = make(Budget(max_steps=1, k=3))
    res = loop.run(Q)
    assert res.states == ["retrieve"]
    assert res.ranking == [0, 1, 2]
    assert res.stop_reason == "step-budget"
    assert retriever.queries == ["which topic passage"]
    assert res.hops[0].features is None


def test_fixed_three_steps_follow_the_script_and_skip_the_last_reason() -> None:
    script = [thought("hop two", read=2), thought("hop three", read=2), thought("never used")]
    loop, retriever, reasoner = make(Budget(max_steps=3, k=2), script, NeverStop())
    res = loop.run(Q)
    assert res.states == ["retrieve", "reason", "decide"] * 2 + ["retrieve"]
    assert retriever.queries == ["which topic passage", "hop two", "hop three"]
    assert [h.admitted for h in res.hops] == [[0, 1], [6, 7], [3, 5]]
    assert reasoner.seen == [[0, 1], [6, 7]]
    assert res.retrieval_calls == 3
    assert res.passages_read == 4
    assert res.ranking == [0, 6, 3, 1, 7, 5]
    assert [h.decision for h in res.hops] == ["continue", "continue", "stop:step-budget"]
    assert res.hops[0].features is not None
    assert list(res.hops[0].features) == list(FEATURES)
    assert res.hops[-1].features is None


def test_without_exclusion_a_hop_can_return_already_pooled_passages() -> None:
    loop, _, _ = make(
        Budget(max_steps=2, k=3), [thought("which topic passage")], NeverStop(), exclude_seen=False
    )
    res = loop.run(Q)
    assert res.hops[1].admitted == []
    assert res.hops[1].duplicates == [0, 1, 2]


def test_no_new_query_ends_the_loop() -> None:
    loop, retriever, _ = make(Budget(max_steps=4, k=2), [thought("hop two"), thought(None)])
    res = loop.run(Q)
    assert res.stop_reason == "no-new-query"
    assert len(retriever.queries) == 2


def test_reasoner_done_ends_the_loop() -> None:
    loop, _, _ = make(Budget(max_steps=4, k=2), [thought("hop two", answer="x", done=True)])
    res = loop.run(Q)
    assert res.stop_reason == "reasoner-done"
    assert res.hops[0].decision == "stop:reasoner-done"


def test_pool_token_budget_stops_before_the_next_retrieval() -> None:
    script = [thought("hop two"), thought("hop three"), thought("x")]
    loop, retriever, _ = make(Budget(max_steps=4, k=2, max_pool_tokens=40), script, tokens=10)
    res = loop.run(Q)
    assert res.stop_reason == "token-budget"
    assert res.pool_tokens == 40
    assert len(retriever.queries) == 2


def test_passage_budget() -> None:
    script = [thought("hop two"), thought("hop three")]
    loop, _, _ = make(Budget(max_steps=4, k=3, max_pool_passages=3), script)
    res = loop.run(Q)
    assert res.stop_reason == "passage-budget"
    assert len(res.ranking) == 3


def test_reasoner_token_budget() -> None:
    script = [thought("hop two", tokens=300), thought("hop three", tokens=300)]
    loop, retriever, _ = make(Budget(max_steps=4, k=2, max_reasoner_tokens=500), script)
    res = loop.run(Q)
    assert res.stop_reason == "reasoner-token-budget"
    assert res.reasoner_tokens == 600
    assert len(retriever.queries) == 2


def _controller(intercept: float) -> StopController:
    d = len(FEATURES)
    return StopController(FEATURES, [0.0] * d, [1.0] * d, [0.0] * d, intercept, 0.5, 1.0, 0, 0, 0)


def test_learned_stop_uses_the_controller_probability() -> None:
    stop_now = LearnedStop(_controller(intercept=3.0))
    loop, retriever, _ = make(Budget(max_steps=4, k=2), [thought("hop two")], stop_now)
    res = loop.run(Q)
    assert res.stop_reason == "controller:learned"
    assert res.hops[0].stop_probability == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-6)
    assert len(retriever.queries) == 1

    keep_going = LearnedStop(_controller(intercept=-3.0))
    script = [thought("hop two"), thought("hop three"), thought("x")]
    loop, retriever, _ = make(Budget(max_steps=3, k=2), script, keep_going)
    assert loop.run(Q).stop_reason == "step-budget"


def test_hard_budgets_win_over_the_controller() -> None:
    loop, _, _ = make(
        Budget(max_steps=4, k=2, max_pool_tokens=20),
        [thought("hop two")],
        LearnedStop(_controller(3.0)),
    )
    assert loop.run(Q).stop_reason == "token-budget"


def test_features_track_answer_stability_and_novelty() -> None:
    script = [
        thought("hop two", answer="A", margin=1.0),
        thought("hop three", answer="a", margin=3.0),
    ]
    loop, _, _ = make(Budget(max_steps=3, k=2), script, NeverStop())
    res = loop.run(Q)
    f1, f2 = res.hops[0].features, res.hops[1].features
    assert f1["hop"] == 1.0
    assert f1["margin_gain"] == 0.0
    assert f1["answer_stable"] == 0.0
    assert f2["margin_gain"] == 2.0
    assert f2["answer_stable"] == 1.0
    assert f1["novelty"] == 1.0
    assert f1["question_coverage"] == 0.0


def test_reasoner_is_restarted_per_question() -> None:
    loop, _, reasoner = make(Budget(max_steps=2, k=2), [thought("hop two")], NeverStop())
    loop.run(Q)
    loop.run(Q)
    assert reasoner.started == 2


def test_bad_budgets_and_missing_reasoner_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        Budget(max_steps=0, k=2)
    with pytest.raises(ValueError, match="needs a reasoner"):
        InterleavedLoop(TableRetriever(N, TABLE), Budget(max_steps=2, k=2), stats=stats())
