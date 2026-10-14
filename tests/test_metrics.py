"""Answer EM/F1, Recall@k, and their parity with the official MuSiQue evaluator and ir-measures."""

from __future__ import annotations

import importlib
import itertools
import json
import sys
from pathlib import Path

import ir_measures
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ircot.metrics import best_em, best_f1, complete_at_k, exact_match, recall_at_k, token_f1
from ircot.text import normalize_answer

CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "musique_answer_cases.json").read_text("utf-8")
)

answers = st.text(alphabet=st.sampled_from(list("abc the an,.-'!é 12")), max_size=24)


def test_normalisation_follows_squad_order() -> None:
    assert normalize_answer("The  Beatles!") == "beatles"
    assert normalize_answer("a-b") == "ab"  # punctuation goes first, so "a" is no article here
    assert normalize_answer("An apple, a day") == "apple day"
    assert normalize_answer("") == ""


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: f"{c['prediction']!r}")
def test_recorded_official_evaluator_outputs(case: dict) -> None:  # type: ignore[type-arg]
    assert best_em(case["prediction"], case["golds"]) == case["em"]
    assert best_f1(case["prediction"], case["golds"]) == pytest.approx(case["f1"], abs=1e-12)


@given(answers, answers)
def test_f1_is_symmetric_bounded_and_implied_by_em(a: str, b: str) -> None:
    f1 = token_f1(a, b)
    assert 0.0 <= f1 <= 1.0
    assert f1 == pytest.approx(token_f1(b, a))
    if exact_match(a, b):
        assert f1 == 1.0


@given(answers)
def test_normalisation_is_idempotent(a: str) -> None:
    assert normalize_answer(normalize_answer(a)) == normalize_answer(a)


rankings = st.lists(st.integers(0, 40), min_size=1, max_size=30, unique=True)
golds = st.sets(st.integers(0, 40), min_size=1, max_size=4)


@given(rankings, golds)
def test_recall_is_monotone_in_k(ranking: list[int], gold: set[int]) -> None:
    values = [recall_at_k(ranking, gold, k) for k in range(len(ranking) + 2)]
    assert all(a <= b for a, b in itertools.pairwise(values))
    assert values[0] == 0.0
    assert values[-1] == len(gold & set(ranking)) / len(gold)
    assert all(0.0 <= v <= 1.0 for v in values)


@given(rankings, golds, st.integers(1, 30))
def test_complete_means_full_recall(ranking: list[int], gold: set[int], k: int) -> None:
    assert complete_at_k(ranking, gold, k) == float(recall_at_k(ranking, gold, k) == 1.0)


def test_recall_needs_gold() -> None:
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k([1, 2], [], 2)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(rankings, golds), min_size=1, max_size=5), st.sampled_from([1, 2, 5, 10]))
def test_recall_matches_ir_measures(queries: list[tuple[list[int], set[int]]], k: int) -> None:
    qrels, run = [], []
    for qi, (ranking, gold) in enumerate(queries):
        qrels += [ir_measures.Qrel(f"q{qi}", f"d{d}", 1) for d in gold]
        run += [
            ir_measures.ScoredDoc(f"q{qi}", f"d{d}", float(len(ranking) - r))
            for r, d in enumerate(ranking)
        ]
    theirs = ir_measures.iter_calc([ir_measures.R @ k], qrels, run)
    by_query = {m.query_id: m.value for m in theirs}
    for qi, (ranking, gold) in enumerate(queries):
        assert recall_at_k(ranking, gold, k) == pytest.approx(by_query.get(f"q{qi}", 0.0))


def _official_answer_module():  # type: ignore[no-untyped-def]
    from ircot.paths import evaluator_dir
    from ircot.sources import evaluator_present

    if not evaluator_present():
        pytest.skip("official MuSiQue evaluator not fetched (ircot fetch --only evaluator)")
    sys.path.insert(0, str(evaluator_dir()))
    try:
        return importlib.import_module("metrics.answer")
    finally:
        sys.path.remove(str(evaluator_dir()))


@pytest.mark.integration
@settings(max_examples=300, deadline=None)
@given(answers, st.lists(answers, min_size=1, max_size=3))
def test_parity_with_the_official_answer_metric(prediction: str, gold: list[str]) -> None:
    official = _official_answer_module()
    metric = official.AnswerMetric()
    metric(prediction, gold)
    em, f1 = metric.get_metric()
    assert best_em(prediction, gold) == em
    assert best_f1(prediction, gold) == pytest.approx(f1, abs=1e-12)
