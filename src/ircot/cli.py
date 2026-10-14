"""Command line: fetch data and models, run the benchmark, render reports, inspect one question."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Annotated

import typer

from ircot import paths

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help=__doc__, pretty_exceptions_show_locals=False
)


def _say(line: str) -> None:
    # Windows consoles default to a legacy code page; titles such as "Rashād" must not crash a run.
    typer.echo(
        line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8"
        )
    )


@app.command()
def fetch(
    only: Annotated[
        str, typer.Option(help="all, data, models or evaluator (the official MuSiQue scorer)")
    ] = "all",
) -> None:
    """Download and verify MuSiQue, HotpotQA, both models and the official evaluator."""
    from ircot import sources

    if only not in {"all", "data", "models", "evaluator"}:
        raise typer.BadParameter("expected all, data, models or evaluator")
    _say(f"data dir  {paths.data_dir()}")
    if only in {"all", "data"}:
        sources.fetch_musique(_say)
        sources.fetch_hotpotqa(_say)
    if only in {"all", "models"}:
        sources.fetch_models(_say)
    if only in {"all", "evaluator"}:
        sources.fetch_evaluator(_say)


@app.command()
def info() -> None:
    """Show where data lives and which assets are present."""
    from ircot import sources

    _say(f"data dir   {paths.data_dir()}  (set {paths.ENV_VAR} to move it)")
    _say(f"musique    {'present' if sources.musique_path().exists() else 'missing'}")
    _say(f"hotpotqa   {'present' if sources.hotpotqa_path().exists() else 'missing'}")
    _say(f"reader     {'present' if sources.reader_onnx_path().exists() else 'missing'}")
    for model in (sources.BGE_SMALL, sources.MINILM_SQUAD2):
        try:
            sources.model_files(model)
            state = "cached"
        except sources.MissingAssetError:
            state = "missing"
        _say(f"model      {model.repo}: {state}")
    _say(f"evaluator  {'present' if sources.evaluator_present() else 'missing'}")


@app.command()
def bench(
    out: Annotated[Path, typer.Option(help="results JSON to write")] = Path(
        "bench/results/full.json"
    ),
    profile: Annotated[
        str, typer.Option(help="models (real data and models) or offline (synthetic, CI)")
    ] = "models",
    datasets: Annotated[
        str, typer.Option(help="comma-separated: musique, hotpotqa (synthetic for offline)")
    ] = "",
    n: Annotated[int, typer.Option(help="evaluation questions per dataset")] = 1000,
    train_n: Annotated[
        int, typer.Option(help="training questions per dataset (policy + stop controller)")
    ] = 200,
    latency_n: Annotated[
        int, typer.Option(help="questions re-timed with a cold reader cache")
    ] = 30,
    seed: Annotated[int, typer.Option(help="sampling seed")] = 2026,
    traces: Annotated[
        bool, typer.Option(help="write per-question JSONL traces under the data dir")
    ] = True,
) -> None:
    """Run every configuration and write one results JSON (plus traces and a run manifest)."""
    from ircot.harness import BenchConfig, run_benchmark

    chosen = tuple(d.strip() for d in datasets.split(",") if d.strip()) or (
        ("synthetic",) if profile == "offline" else ("musique", "hotpotqa")
    )
    cfg = BenchConfig(
        profile=profile, datasets=chosen, n_eval=n, n_train=train_n, latency_n=latency_n, seed=seed
    )
    command = "ircot bench " + " ".join(shlex.quote(a) for a in sys.argv[2:])
    run_benchmark(cfg, command.strip(), out, paths.runs_dir() if traces else None, _say)


@app.command()
def report(results: Path) -> None:
    """Print the Markdown tables for a results JSON."""
    from ircot.report import render

    _say(render(json.loads(results.read_text(encoding="utf-8"))))


@app.command()
def ask(
    question_id: Annotated[
        str, typer.Argument(help="a question id from the evaluation sample")
    ] = "",
    dataset: Annotated[str, typer.Option(help="musique or hotpotqa")] = "musique",
    n: Annotated[int, typer.Option(help="size of the sample whose corpus is searched")] = 1000,
    seed: Annotated[int, typer.Option(help="sampling seed (match the benchmark's)")] = 2026,
    results: Annotated[
        Path | None,
        typer.Option(help="results JSON to take the hybrid weight, policy and controller from"),
    ] = None,
) -> None:
    """Run the extractive loop on one question and print every hop.

    Without ``--results`` this is the fixed 3-step loop with the append policy; with it, the
    learned-stop loop using exactly the choices the benchmark made on its training sample.
    """
    from ircot.controller import StopController
    from ircot.data import split_sample
    from ircot.harness import (
        CONFIGS,
        BenchConfig,
        Workspace,
        build_loop,
        load_models,
        load_questions,
    )
    from ircot.policies import POLICIES
    from ircot.reasoners import best_answer

    cfg = BenchConfig(datasets=(dataset,), n_eval=n, n_train=0, seed=seed)
    models = load_models("models")
    sample, _ = split_sample(load_questions(dataset), n, 0, seed)
    found = [x for x in sample if x.qid == question_id] if question_id else sample[:1]
    if not found:
        raise typer.BadParameter(f"{question_id} is not in the {n}-question {dataset} sample")
    q = found[0]
    ws = Workspace.build(sample, models, cfg, _say)
    policy, controller, spec_name = POLICIES["append"], None, "loop3-extractive"
    if results is not None:
        info = json.loads(results.read_text(encoding="utf-8"))["datasets"][dataset]
        ws.use_alpha(info["alpha_selection"]["selected"], cfg.hybrid_depth)
        policy = POLICIES[info["policy_selection"]["selected"]]
        fields = StopController.__dataclass_fields__
        controller = StopController.from_json(
            {k: v for k, v in info["stop_controller"].items() if k in fields}
        )
        spec_name = "ircot-extractive"
    spec = next(s for s in CONFIGS if s.name == spec_name)
    cache = ws.reader_cache(models)
    result = build_loop(spec, ws, cache, policy, controller, cfg).run(q)
    answer, _ = best_answer(cache, q.text, result.ranking[: cfg.answer_depth])
    gold = set(ws.corpus.gold[q.qid])
    _say(f"{q.qid}: {q.text}\ngold answer: {q.answer}\n")
    for h in result.hops:
        _say(f"hop {h.hop}  query: {h.query}")
        for doc in h.admitted:
            mark = "*" if doc in gold else " "
            _say(f"   {mark} [{doc}] {ws.corpus.titles[doc]}")
        if h.bridge:
            _say(f"   bridge: {h.bridge} (title match: {h.bridge_title_match})")
        p_stop = f" (p_stop={h.stop_probability})" if h.stop_probability is not None else ""
        _say(f"   decision: {h.decision}{p_stop}")
    _say(f"\nanswer: {answer}   (* = gold supporting passage, policy {policy.name}, {spec.name})")


@app.command()
def demo() -> None:
    """Offline run on the synthetic world (no downloads, no models); the container smoke test."""
    from ircot.data import split_sample
    from ircot.harness import (
        CONFIGS,
        BenchConfig,
        Workspace,
        build_loop,
        load_models,
        load_questions,
        train_stage,
    )
    from ircot.metrics import recall_at_k

    cfg = BenchConfig(profile="offline", datasets=("synthetic",), n_eval=30, n_train=30)
    models = load_models("offline")
    eval_q, train_q = split_sample(load_questions("synthetic"), 30, 30, cfg.seed)
    trained = train_stage(Workspace.build(train_q, models, cfg, _say), models, cfg, _say)
    ws = Workspace.build(eval_q, models, cfg, _say)
    ws.use_alpha(trained.alpha, cfg.hybrid_depth)
    cache = ws.reader_cache(models)
    specs = {s.name: s for s in CONFIGS}
    for spec in CONFIGS:
        loop = build_loop(spec, ws, cache, trained.policy, trained.controller, cfg)
        recall = [recall_at_k(loop.run(q).ranking, ws.corpus.gold[q.qid], 5) for q in eval_q]
        _say(f"{spec.name:<22} R@5 {sum(recall) / len(recall):.3f}")
    q = next(x for x in eval_q if x.hops >= 3)
    loop = build_loop(specs["ircot-extractive"], ws, cache, trained.policy, trained.controller, cfg)
    _say(f"\n{q.text}")
    for h in loop.run(q).hops:
        _say(f"  hop {h.hop}: {h.query!r} -> bridge {h.bridge!r}, {h.decision}")
    _say(f"stop controller: {trained.controller.train_rows} rows, policy {trained.policy.name}")
    _say("demo ok (offline stand-ins on a synthetic world; not model results)")


if __name__ == "__main__":
    app()
