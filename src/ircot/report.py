"""Markdown tables from a results JSON (what the README's results sections are pasted from)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ircot.harness import REFERENCE


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def _ci(m: Mapping[str, float]) -> str:
    return f"{_pct(m['mean'])} [{_pct(m['ci_low'])}, {_pct(m['ci_high'])}]"


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main_table(results: Mapping[str, Any]) -> str:
    head = ["configuration", "R@2", "R@5", "R@10", "EM", "F1"]
    head += ["retrieval calls", "reader passages", "p50 ms", "p95 ms"]
    lines = [_row(head), _row(["---"] * len(head))]
    for name, r in results.items():
        m, c, lat = r["metrics"], r["cost"], r.get("latency", {})
        cells = [
            name,
            _pct(m["recall@2"]["mean"]),
            _ci(m["recall@5"]),
            _pct(m["recall@10"]["mean"]),
        ]
        cells += [_pct(m["em"]["mean"]), _ci(m["f1"]), f"{c['retrieval_calls_mean']:.2f}"]
        cells += [f"{c['reader_passages_mean']:.1f}"]
        cells += [str(lat.get("p50_ms", "not measured")), str(lat.get("p95_ms", "not measured"))]
        lines.append(_row(cells))
    return "\n".join(lines)


def _diff(t: Mapping[str, float]) -> str:
    return f"{_pct(t['diff'])} [{_pct(t['ci_low'])}, {_pct(t['ci_high'])}]"


def significance_table(results: Mapping[str, Any]) -> str:
    head = ["configuration", f"ΔR@5 vs {REFERENCE}", "p", f"ΔF1 vs {REFERENCE}", "p"]
    lines = [_row(head), _row(["---"] * len(head))]
    for name, r in results.items():
        t = r.get(f"vs_{REFERENCE}")
        if t is None:
            continue
        r5, f1 = t["recall@5"], t["f1"]
        lines.append(_row([name, _diff(r5), _p(r5["p_value"]), _diff(f1), _p(f1["p_value"])]))
    return "\n".join(lines)


def breakdown_table(results: Mapping[str, Any], key: str, metric: str = "recall@5") -> str:
    groups = sorted({g for r in results.values() for g in r.get(key, {})})
    if not groups:
        return ""
    sizes = {
        g: next(r[key][g]["n"] for r in results.values() if g in r.get(key, {})) for g in groups
    }
    head = " | ".join(f"{g} (n={sizes[g]})" for g in groups)
    lines = [f"| configuration | {head} |", "|---|" + "---|" * len(groups)]
    for name, r in results.items():
        cells = " | ".join(_pct(r[key][g][metric]) if g in r.get(key, {}) else "-" for g in groups)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)


def render(payload: Mapping[str, Any]) -> str:
    out = [
        f"Results from `{payload['command']}` on {payload['date']} (git {payload['git_sha'][:12]})."
    ]
    for dataset, info in payload["datasets"].items():
        res = info["results"]
        out += [
            f"\n### {dataset}\n",
            f"Evaluation sample n={info['eval']['n']}, "
            f"corpus {info['eval']['corpus_passages']} passages.\n",
            main_table(res),
            "\n" + significance_table(res),
        ]
        key = "by_hops" if any("by_hops" in r for r in res.values()) else "by_type"
        out += [f"\nRecall@5 {key.replace('_', ' ')}:\n", breakdown_table(res, key)]
        sel = info["policy_selection"]
        out.append(
            "\nQuery-policy selection on the training sample: "
            + ", ".join(
                f"{k} {_pct(v['pool_recall'])}" for k, v in sel.items() if isinstance(v, dict)
            )
            + f" (selected: {sel['selected']})."
        )
    if payload.get("not_run"):
        out.append(
            "\nNot run: " + "; ".join(f"{n['config']} ({n['reason']})" for n in payload["not_run"])
        )
    return "\n".join(out)
