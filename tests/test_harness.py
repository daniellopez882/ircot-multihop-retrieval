"""End to end on the 30-question synthetic fixture with the offline encoder and reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ircot.harness import CONFIGS, REFERENCE, BenchConfig, build_loop, run_benchmark, train_stage
from ircot.metrics import recall_at_k
from ircot.policies import POLICIES
from ircot.reasoners import ReaderCache
from ircot.report import render


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory: pytest.TempPathFactory, offline_config: BenchConfig):  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("bench")
    out = root / "results.json"
    payload = run_benchmark(
        offline_config, "ircot bench --profile offline", out, root / "runs", lambda _: None
    )
    return payload, out, root / "runs"


def test_results_file_carries_provenance(offline_run) -> None:  # type: ignore[no-untyped-def]
    payload, out, _ = offline_run
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(payload))
    for key in ("machine", "date", "git_sha", "command", "config", "config_hash", "not_run"):
        assert key in payload
    assert payload["machine"]["cpu"]
    assert "numpy" in payload["machine"]["packages"]
    assert payload["profile"] == "offline"
    assert payload["not_run"][0]["status"] == "not run"


def test_every_configuration_reports_metrics_costs_and_tests(offline_run) -> None:  # type: ignore[no-untyped-def]
    payload, _, _ = offline_run
    results = payload["datasets"]["synthetic"]["results"]
    assert set(results) == {c.name for c in CONFIGS}
    for name, r in results.items():
        assert r["n"] == 30
        for m in ("recall@2", "recall@5", "recall@10", "em", "f1", "complete@10"):
            assert (
                r["metrics"][m]["ci_low"] <= r["metrics"][m]["mean"] <= r["metrics"][m]["ci_high"]
            )
        assert r["latency"]["n"] == 3
        assert set(r["by_hops"]) == {"2", "3", "4"}
        assert (f"vs_{REFERENCE}" in r) == (name != REFERENCE)
    assert results["oneshot-bm25"]["cost"]["retrieval_calls_mean"] == 1.0
    assert results["loop3-extractive"]["cost"]["retrieval_calls_mean"] == 3.0


def test_iteration_helps_on_the_synthetic_world(offline_run) -> None:  # type: ignore[no-untyped-def]
    results = offline_run[0]["datasets"]["synthetic"]["results"]
    r5 = {name: r["metrics"]["recall@5"]["mean"] for name, r in results.items()}
    assert r5["loop2-extractive"] > r5["oneshot-hybrid"]
    assert r5["loop-oracle-bridge"] >= r5["loop2-extractive"]


def test_traces_and_manifest_are_written(offline_run) -> None:  # type: ignore[no-untyped-def]
    payload, _, runs = offline_run
    (run_dir,) = list(runs.iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_hash"] == payload["config_hash"]
    assert "eval" in manifest["datasets"]["synthetic"]
    trace = (
        (run_dir / "synthetic" / "ircot-extractive.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(trace) == 30
    first = json.loads(trace[0])
    assert first["hops"][0]["query"]
    assert first["hops"][-1]["decision"].startswith("stop:")


def test_report_renders(offline_run) -> None:  # type: ignore[no-untyped-def]
    text = render(offline_run[0])
    assert "| ircot-extractive |" in text
    assert "Not run: ircot-cot-llm" in text


def test_training_stage_is_deterministic(world, offline_models, offline_config) -> None:  # type: ignore[no-untyped-def]
    from ircot.harness import Workspace

    ws = Workspace.build(world[1], offline_models, offline_config, lambda _: None)
    a = train_stage(ws, offline_models, offline_config, lambda _: None)
    b = train_stage(ws, offline_models, offline_config, lambda _: None)
    assert a.alpha == b.alpha
    assert a.alpha in offline_config.hybrid_alphas
    assert a.policy.name == b.policy.name
    assert a.controller.to_json() == b.controller.to_json()
    assert a.report == b.report
    assert set(a.report) == {"alpha_selection", "policy_selection", "stop_controller"}


def test_loops_share_the_hop_one_retrieval_with_one_shot(
    eval_ws, offline_models, offline_config
) -> None:  # type: ignore[no-untyped-def]
    cache = ReaderCache(offline_models.reader, eval_ws.corpus)
    specs = {c.name: c for c in CONFIGS}
    one = build_loop(
        specs["oneshot-hybrid"], eval_ws, cache, POLICIES["append"], None, offline_config
    )
    two = build_loop(
        specs["loop2-extractive"], eval_ws, cache, POLICIES["append"], None, offline_config
    )
    for q in eval_ws.questions[:10]:
        a, b = one.run(q), two.run(q)
        assert b.hops[0].admitted == a.ranking[: offline_config.k_hop]
        gold = eval_ws.corpus.gold[q.qid]
        assert recall_at_k(b.ranking, gold, len(b.ranking)) >= recall_at_k(
            a.ranking, gold, offline_config.k_hop
        )


def test_learned_stop_needs_a_controller(eval_ws, offline_models, offline_config) -> None:  # type: ignore[no-untyped-def]
    cache = ReaderCache(offline_models.reader, eval_ws.corpus)
    spec = next(c for c in CONFIGS if c.stop == "learned")
    with pytest.raises(ValueError, match="stop controller"):
        build_loop(spec, eval_ws, cache, POLICIES["append"], None, offline_config)


def test_unknown_profile_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="profile"):
        run_benchmark(BenchConfig(profile="gpu"), "x", tmp_path / "r.json", None, lambda _: None)


def test_reader_passages_count_each_passage_once(offline_run) -> None:  # type: ignore[no-untyped-def]
    payload, _, runs = offline_run
    results = payload["datasets"]["synthetic"]["results"]
    cfg = BenchConfig(**{k: v for k, v in payload["config"].items() if k != "datasets"})
    # One shot reads only the answer step's passages; the title-feedback loop reads nothing else.
    assert results["oneshot-hybrid"]["cost"]["reader_passages_mean"] == cfg.answer_depth
    assert results["loop2-title-feedback"]["cost"]["reader_passages_mean"] == cfg.answer_depth
    (run_dir,) = list(runs.iterdir())
    for line in (run_dir / "synthetic" / "loop2-extractive.jsonl").read_text("utf-8").splitlines():
        record = json.loads(line)
        read = {d for h in record["hops"] for d in h["read"]}
        assert read <= set(record["ranking"])
        assert len(read) <= cfg.read_per_hop
    # Two steps read three hop-one passages, which the round-robin top five already contains.
    assert results["loop2-extractive"]["cost"]["reader_passages_mean"] == cfg.answer_depth
