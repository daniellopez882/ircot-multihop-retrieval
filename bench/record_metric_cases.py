"""Record the official MuSiQue evaluator's EM/F1 on tricky strings as a test fixture.

The default test suite cannot download the evaluator, so it checks ``ircot.metrics`` against these
recorded outputs instead; the integration suite compares against the live evaluator as well.

    uv run ircot fetch --only evaluator
    uv run python bench/record_metric_cases.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from ircot.paths import evaluator_dir
from ircot.sources import EVALUATOR_BASE, EVALUATOR_FILES, evaluator_present

CASES: list[tuple[str, list[str]]] = [
    ("Miquette Giraudy", ["Miquette Giraudy"]),
    ("miquette giraudy.", ["Miquette Giraudy"]),
    ("The Beatles", ["Beatles"]),
    ("a Beatles song", ["The Beatles"]),
    ("", ["Paris"]),
    ("Paris", [""]),
    ("", [""]),
    ("the", ["a"]),
    ("Paris, France", ["Paris"]),
    ("France Paris", ["Paris, France"]),
    ("New York City", ["New York", "NYC"]),
    ("NYC", ["New York", "NYC"]),
    ("1961", ["1961"]),
    ("in 1961", ["1961", "nineteen sixty-one"]),
    ("a-b", ["ab"]),
    ("a b", ["ab"]),
    ("Rashād", ["Rashad"]),
    ("U.S.", ["US"]),
    ("U.S. Army", ["United States Army", "US Army"]),
    ("the the the", ["the"]),
    ("cat cat dog", ["cat dog dog"]),
    ("Saint-Étienne", ["Saint Étienne"]),
    ("  spaced   out  ", ["spaced out"]),
    ("Who knows?", ["who knows"]),
    ("an apple a day", ["apple day"]),
]


def main() -> None:
    if not evaluator_present():
        raise SystemExit("fetch the evaluator first: uv run ircot fetch --only evaluator")
    sys.path.insert(0, str(evaluator_dir()))
    answer = importlib.import_module("metrics.answer")
    cases = []
    for prediction, golds in CASES:
        metric = answer.AnswerMetric()
        metric(prediction, golds)
        em, f1 = metric.get_metric()
        cases.append({"prediction": prediction, "golds": golds, "em": em, "f1": f1})
    pin = next(p for p in EVALUATOR_FILES if p.name == "metrics/answer.py")
    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "musique_answer_cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": EVALUATOR_BASE + pin.name,
        "sha256": pin.sha256,
        "recorded_with": "bench/record_metric_cases.py",
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
