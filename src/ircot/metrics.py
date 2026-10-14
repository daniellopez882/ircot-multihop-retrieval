"""Answer EM/F1 (SQuAD normalisation, max over gold aliases) and supporting-paragraph Recall@k.

Behaviour on the edge cases follows the official MuSiQue evaluator, which tests/test_metrics.py
checks against recorded outputs and, when it has been fetched, against the evaluator itself:
an empty prediction scores F1 1 only against an empty gold, and EM/F1 take the maximum over the
answer and all of its aliases.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Sequence

from ircot.text import normalize_answer


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split() if text else []


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = _tokens(prediction), _tokens(gold)
    if not pred or not ref:
        return float(pred == ref)
    same = sum((Counter(pred) & Counter(ref)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_em(prediction: str, golds: Iterable[str]) -> float:
    return max(exact_match(prediction, g) for g in golds)


def best_f1(prediction: str, golds: Iterable[str]) -> float:
    return max(token_f1(prediction, g) for g in golds)


def recall_at_k(ranking: Sequence[int], gold: Collection[int], k: int) -> float:
    """Fraction of the gold supporting passages that appear in the first ``k`` of ``ranking``."""
    if not gold:
        raise ValueError("recall is undefined for a question without supporting passages")
    return len(set(ranking[:k]) & set(gold)) / len(gold)


def complete_at_k(ranking: Sequence[int], gold: Collection[int], k: int) -> float:
    """1 when every gold passage is in the first ``k`` (the all-or-nothing view of recall)."""
    return float(set(gold) <= set(ranking[:k]))
