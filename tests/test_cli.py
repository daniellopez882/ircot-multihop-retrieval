from __future__ import annotations

import inspect
import json
from pathlib import Path

from typer.testing import CliRunner

from ircot.cli import app, bench
from ircot.harness import BenchConfig

runner = CliRunner()


def test_demo_runs_offline() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    assert "demo ok" in result.output
    assert "ircot-extractive" in result.output


def test_info_names_the_data_dir(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRCOT_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert str(tmp_path) in result.output
    assert "musique    missing" in result.output


def test_offline_bench_and_report(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRCOT_DATA_DIR", str(tmp_path / "data"))
    out = tmp_path / "smoke.json"
    args = ["bench", "--profile", "offline", "--n", "30", "--train-n", "30", "--latency-n", "2"]
    result = runner.invoke(app, [*args, "--out", str(out), "--no-traces"])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["datasets"]["synthetic"]["eval"]["n"] == 30
    assert not (tmp_path / "data" / "runs").exists()
    report = runner.invoke(app, ["report", str(out)])
    assert report.exit_code == 0
    assert "### synthetic" in report.output


def test_fetch_rejects_unknown_targets() -> None:
    result = runner.invoke(app, ["fetch", "--only", "everything"])
    assert result.exit_code != 0


def test_bench_defaults_match_the_benchmark_config() -> None:
    # The CLI is how the published numbers are produced, so its defaults are the documented ones.
    params = inspect.signature(bench).parameters
    cfg = BenchConfig()
    assert params["n"].default == cfg.n_eval
    assert params["train_n"].default == cfg.n_train
    assert params["latency_n"].default == cfg.latency_n
    assert params["seed"].default == cfg.seed
