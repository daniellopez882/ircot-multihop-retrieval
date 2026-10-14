"""The benchmark harness: one command from raw data to a results JSON plus per-hop traces.

For each dataset the harness

1. draws two disjoint fixed-seed samples (evaluation, stop-controller training) and builds a
   union-of-paragraphs corpus for each,
2. on the training sample only: picks the hybrid retriever's dense weight, runs the extractive
   loop with each query policy, keeps the policy with the best pool recall, and fits the stop
   controller on that policy's per-hop traces,
3. runs every configuration over the evaluation sample with one shared reader cache,
4. times a subset again with cold caches per question, so latency includes every model call,
5. adds bootstrap CIs and paired permutation tests against one-shot hybrid retrieval.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ircot.bridge import TitleIndex
from ircot.controller import FEATURES, StopController
from ircot.data import (
    Corpus,
    Question,
    ids_digest,
    load_hotpotqa,
    load_musique,
    parse_musique,
    split_sample,
)
from ircot.encoders import Encoder, cached_passage_embeddings
from ircot.loop import Budget, InterleavedLoop, LearnedStop, LoopResult, NeverStop, StopRule
from ircot.manifest import config_hash, git_sha, local_date, local_timestamp, machine_info
from ircot.metrics import best_em, best_f1, complete_at_k, recall_at_k
from ircot.policies import POLICIES, QueryPolicy
from ircot.pool import PassageStats
from ircot.reader import Reader
from ircot.reasoners import (
    ExtractiveReasoner,
    OracleBridgeReasoner,
    ReaderCache,
    Reasoner,
    TitleFeedbackReasoner,
    best_answer,
)
from ircot.retrieval import BM25, Dense, Hybrid, Retriever, top_k
from ircot.stats import bootstrap_ci, paired_permutation_test

Log = Callable[[str], None]
REFERENCE = "oneshot-hybrid"
METRICS = ("recall@2", "recall@5", "recall@10", "em", "f1")


@dataclass(frozen=True)
class BenchConfig:
    profile: str = "models"
    datasets: tuple[str, ...] = ("musique", "hotpotqa")
    n_eval: int = 1000
    n_train: int = 200
    latency_n: int = 30
    seed: int = 2026
    k_oneshot: int = 10
    k_hop: int = 5
    read_per_hop: int = 3
    answer_depth: int = 5
    ircot_max_steps: int = 4
    pool_token_budget: int = 3000
    title_bonus: float = 4.0
    bm25_k1: float = 0.9
    bm25_b: float = 0.4
    # Dense weight candidates for the hybrid, tried on the training sample; the first wins ties.
    hybrid_alphas: tuple[float, ...] = (0.5, 0.25, 0.75)
    hybrid_depth: int = 100
    controller_l2: float = 1.0
    controller_threshold: float = 0.5
    bootstrap_resamples: int = 2000
    ci_level: float = 0.95
    permutations: int = 10000


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    retriever: str
    steps: int
    reasoner: str | None = None
    stop: str = "never"
    only: tuple[str, ...] = ()
    description: str = ""


CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec("oneshot-bm25", "bm25", 1, description="question as the only query, BM25"),
    ConfigSpec("oneshot-dense", "dense", 1, description="question as the only query, bge-small"),
    ConfigSpec("oneshot-hybrid", "hybrid", 1, description="question as the only query, fused"),
    ConfigSpec(
        "loop2-title-feedback",
        "hybrid",
        2,
        "title-feedback",
        description="2 steps, next query = question + best new title; no reader",
    ),
    ConfigSpec(
        "loop2-extractive", "hybrid", 2, "extractive", description="2 fixed steps, reader bridge"
    ),
    ConfigSpec(
        "loop3-extractive", "hybrid", 3, "extractive", description="3 fixed steps, reader bridge"
    ),
    ConfigSpec(
        "ircot-extractive",
        "hybrid",
        4,
        "extractive",
        "learned",
        description="up to 4 steps, reader bridge, learned stop",
    ),
    ConfigSpec(
        "loop-oracle-bridge",
        "hybrid",
        4,
        "oracle-bridge",
        only=("musique", "synthetic"),
        description="gold intermediate answers as bridges (reasoner ceiling)",
    ),
)

NOT_RUN = (
    {
        "config": "ircot-cot-llm",
        "status": "not run",
        "reason": (
            "the chain-of-thought reasoner needs an OpenAI-compatible LLM endpoint; it is "
            "implemented (ircot.reasoners.ChainOfThoughtReasoner) but no LLM was run for "
            "these results"
        ),
    },
)


@dataclass
class Models:
    encoder: Encoder
    reader: Reader
    count_tokens: Callable[[str], int]
    info: dict[str, object]
    cache_root: Path | None


def load_models(profile: str) -> Models:
    if profile == "offline":
        from ircot.offline import HashingEncoder, LexicalReader
        from ircot.text import word_count

        info: dict[str, object] = {
            "encoder": "hashing-64 (offline stand-in, not a model)",
            "reader": "lexical capitalised-span reader (offline stand-in, not a model)",
            "token_counter": "whitespace words",
        }
        return Models(HashingEncoder(), LexicalReader(), word_count, info, None)
    if profile != "models":
        raise ValueError(f"unknown profile {profile!r} (expected 'models' or 'offline')")
    from tokenizers import Tokenizer

    from ircot import sources
    from ircot.encoders import OnnxBgeEncoder
    from ircot.paths import cache_dir
    from ircot.reader import OnnxSquadReader

    bge = sources.model_files(sources.BGE_SMALL)
    reader_files = sources.model_files(sources.MINILM_SQUAD2)
    onnx_path = sources.reader_onnx_path()
    if not onnx_path.exists():
        raise sources.MissingAssetError(f"{onnx_path} is missing; run `ircot fetch`")
    onnx_sha = sources.sha256_file(onnx_path)
    counter = Tokenizer.from_file(str(bge["tokenizer.json"]))
    counter.no_truncation()

    def count_tokens(text: str) -> int:
        return len(counter.encode(text, add_special_tokens=False).ids)

    info = {
        "encoder": {
            "repo": sources.BGE_SMALL.repo,
            "revision": sources.BGE_SMALL.revision,
            "file": "onnx/model.onnx",
            "sha256": sources.BGE_SMALL.pin("onnx/model.onnx").sha256,
            "license": sources.BGE_SMALL.license,
        },
        "reader": {
            "repo": sources.MINILM_SQUAD2.repo,
            "revision": sources.MINILM_SQUAD2.revision,
            "weights_sha256": sources.MINILM_SQUAD2.pin("model.safetensors").sha256,
            "onnx_sha256": onnx_sha,
            "license": sources.MINILM_SQUAD2.license,
            "export": "ircot.onnx_bert (graph built from safetensors, opset 17)",
        },
        "token_counter": "bge-small WordPiece tokens",
    }
    return Models(
        OnnxBgeEncoder(bge["onnx/model.onnx"], bge["tokenizer.json"]),
        # The reader's name carries its graph hash, so cached spans never outlive a re-export.
        OnnxSquadReader(
            onnx_path, reader_files["vocab.txt"], name=f"minilm-squad2-{onnx_sha[:12]}"
        ),
        count_tokens,
        info,
        cache_dir(),
    )


def load_questions(dataset: str) -> list[Question]:
    if dataset == "synthetic":
        from ircot.offline import synthetic_world

        return [parse_musique(r) for r in synthetic_world(seed=7, n_questions=60)]
    from ircot import sources

    if dataset == "musique":
        path = sources.musique_path()
        if not path.exists():
            raise sources.MissingAssetError(f"{path} is missing; run `ircot fetch`")
        return load_musique(path)
    if dataset == "hotpotqa":
        path = sources.hotpotqa_path()
        if not path.exists():
            raise sources.MissingAssetError(f"{path} is missing; run `ircot fetch`")
        return load_hotpotqa(path)
    raise ValueError(f"unknown dataset {dataset!r}")


@dataclass
class Workspace:
    """One sample's corpus with every index and per-passage fact the loops need."""

    questions: list[Question]
    corpus: Corpus
    stats: PassageStats
    retrievers: dict[str, Retriever]
    titles: TitleIndex

    @classmethod
    def build(
        cls, questions: list[Question], models: Models, cfg: BenchConfig, log: Log
    ) -> Workspace:
        corpus = Corpus.from_questions(questions)
        passages = [corpus.passage(d) for d in range(len(corpus))]
        stats = PassageStats.build(passages, models.count_tokens)
        bm25 = BM25(passages, cfg.bm25_k1, cfg.bm25_b)
        embeddings = cached_passage_embeddings(models.encoder, passages, models.cache_root, log)
        dense = Dense(embeddings, models.encoder)
        hybrid = Hybrid(bm25, dense, cfg.hybrid_alphas[0], cfg.hybrid_depth)
        return cls(
            questions,
            corpus,
            stats,
            {"bm25": bm25, "dense": dense, "hybrid": hybrid},
            TitleIndex(corpus.titles),
        )

    def use_alpha(self, alpha: float, depth: int) -> None:
        bm25, dense = self.retrievers["bm25"], self.retrievers["dense"]
        assert isinstance(bm25, BM25)
        assert isinstance(dense, Dense)
        self.retrievers["hybrid"] = Hybrid(bm25, dense, alpha, depth)

    def reader_cache(self, models: Models) -> ReaderCache:
        """A reader cache for this corpus, saved next to the embeddings when there is a cache
        directory, so an interrupted benchmark does not re-read passages it already read."""
        path = None
        if models.cache_root is not None:
            name = f"reader-{models.reader.name}-{self.corpus.digest()[:20]}.json"
            path = models.cache_root / name
        return ReaderCache(models.reader, self.corpus, path=path)


def _reasoner(
    kind: str | None, ws: Workspace, cache: ReaderCache, policy: QueryPolicy, cfg: BenchConfig
) -> Reasoner | None:
    if kind is None:
        return None
    if kind == "extractive":
        return ExtractiveReasoner(cache, ws.titles, policy, cfg.read_per_hop, cfg.title_bonus)
    if kind == "title-feedback":
        return TitleFeedbackReasoner(ws.corpus)
    if kind == "oracle-bridge":
        return OracleBridgeReasoner()
    raise ValueError(f"unknown reasoner {kind!r}")


def build_loop(
    spec: ConfigSpec,
    ws: Workspace,
    cache: ReaderCache,
    policy: QueryPolicy,
    controller: StopController | None,
    cfg: BenchConfig,
) -> InterleavedLoop:
    k = cfg.k_oneshot if spec.steps == 1 else cfg.k_hop
    stop: StopRule = NeverStop()
    if spec.stop == "learned":
        if controller is None:
            raise ValueError(f"{spec.name} needs a trained stop controller")
        stop = LearnedStop(controller)
    budget = Budget(
        max_steps=spec.steps,
        k=k,
        max_pool_tokens=None if spec.steps == 1 else cfg.pool_token_budget,
    )
    return InterleavedLoop(
        ws.retrievers[spec.retriever],
        budget,
        stats=ws.stats,
        reasoner=_reasoner(spec.reasoner, ws, cache, policy, cfg),
        stop=stop,
    )


def score_question(
    q: Question, result: LoopResult, ws: Workspace, cache: ReaderCache, cfg: BenchConfig
) -> dict[str, object]:
    gold = ws.corpus.gold[q.qid]
    answer, margin = best_answer(cache, q.text, result.ranking[: cfg.answer_depth])
    return {
        "qid": q.qid,
        "hops": q.hops,
        "kind": q.kind,
        "gold": list(gold),
        "ranking": result.ranking[:20],
        "recall@2": recall_at_k(result.ranking, gold, 2),
        "recall@5": recall_at_k(result.ranking, gold, 5),
        "recall@10": recall_at_k(result.ranking, gold, 10),
        "recall@pool": recall_at_k(result.ranking, gold, len(result.ranking)),
        "complete@10": complete_at_k(result.ranking, gold, 10),
        "pool_size": len(result.ranking),
        "pool_tokens": result.pool_tokens,
        "answer": answer,
        "answer_margin": round(margin, 4),
        "em": best_em(answer, q.answers),
        "f1": best_f1(answer, q.answers),
        "steps": len(result.hops),
        "retrieval_calls": result.retrieval_calls,
        "reader_passages": result.passages_read + min(cfg.answer_depth, len(result.ranking)),
        "stop_reason": result.stop_reason,
    }


def trace_record(result: LoopResult, scored: dict[str, object]) -> dict[str, object]:
    return {
        "qid": result.qid,
        "stop_reason": result.stop_reason,
        "states": result.states,
        "hops": [asdict(h) for h in result.hops],
        "ranking": result.ranking,
        "answer": scored["answer"],
        "em": scored["em"],
        "f1": scored["f1"],
        "recall@5": scored["recall@5"],
    }


def controller_rows(
    results: Iterable[LoopResult], gold: dict[str, tuple[int, ...]]
) -> tuple[np.ndarray, np.ndarray]:
    """Features at every decision point, labelled 1 when later hops add no gold passage."""
    X: list[list[float]] = []
    y: list[int] = []
    for res in results:
        g = set(gold[res.qid])
        cumulative: list[set[int]] = []
        seen: set[int] = set()
        for h in res.hops:
            seen |= set(h.admitted)
            cumulative.append(set(seen))
        final = len(g & cumulative[-1]) if cumulative else 0
        for i, h in enumerate(res.hops):
            if h.features is None:
                continue
            X.append([h.features[name] for name in FEATURES])
            y.append(int(len(g & cumulative[i]) == final))
    return np.asarray(X, dtype=np.float64).reshape(-1, len(FEATURES)), np.asarray(y, dtype=np.int64)


@dataclass
class TrainOutcome:
    alpha: float
    policy: QueryPolicy
    controller: StopController
    report: dict[str, object]


def _mean(values: Iterable[float]) -> float:
    return round(float(np.mean(list(values))), 4)


def train_stage(ws: Workspace, models: Models, cfg: BenchConfig, log: Log) -> TrainOutcome:
    """Every choice the evaluation depends on, made on the training sample only."""
    gold = ws.corpus.gold
    alpha_scores: dict[str, float] = {}
    for alpha in cfg.hybrid_alphas:
        ws.use_alpha(alpha, cfg.hybrid_depth)
        hybrid = ws.retrievers["hybrid"]
        alpha_scores[str(alpha)] = _mean(
            recall_at_k([h.doc for h in top_k(hybrid.scores(q.text), 5)], gold[q.qid], 5)
            for q in ws.questions
        )
    order = list(cfg.hybrid_alphas)
    alpha = max(order, key=lambda a: (alpha_scores[str(a)], -order.index(a)))
    ws.use_alpha(alpha, cfg.hybrid_depth)
    log(f"  hybrid dense weight {alpha} (one-shot R@5 by weight: {alpha_scores})")

    cache = ws.reader_cache(models)
    spec = ConfigSpec("train-probe", "hybrid", cfg.ircot_max_steps, "extractive")
    selection: dict[str, dict[str, float]] = {}
    runs: dict[str, list[LoopResult]] = {}
    for name, policy in POLICIES.items():
        loop = build_loop(spec, ws, cache, policy, None, cfg)
        runs[name] = [loop.run(q) for q in ws.questions]
        cache.save()
        selection[name] = {
            "pool_recall": _mean(
                recall_at_k(r.ranking, gold[r.qid], len(r.ranking)) for r in runs[name]
            ),
            "recall@5": _mean(recall_at_k(r.ranking, gold[r.qid], 5) for r in runs[name]),
            "mean_pool_size": _mean(len(r.ranking) for r in runs[name]),
        }
        log(f"  policy {name:<9} pool recall {selection[name]['pool_recall']:.3f}")
    names = list(POLICIES)
    chosen = max(names, key=lambda n: (selection[n]["pool_recall"], -names.index(n)))
    X, y = controller_rows(runs[chosen], gold)
    controller = StopController.fit(X, y, l2=cfg.controller_l2, threshold=cfg.controller_threshold)
    predicted = np.array([controller.probability(row) >= controller.threshold for row in X])
    accuracy = _mean((predicted == y.astype(bool)).astype(float))
    log(f"  controller rows {len(y)}, positive rate {y.mean():.3f}, train accuracy {accuracy}")
    report: dict[str, object] = {
        "alpha_selection": {
            "recall@5_by_alpha": alpha_scores,
            "selected": alpha,
            "selected_by": "highest one-shot hybrid Recall@5 on the training sample",
        },
        "policy_selection": {
            **selection,
            "selected": chosen,
            "selected_by": f"highest pool recall of the {cfg.ircot_max_steps}-step loop "
            "on the training sample",
        },
        "stop_controller": {
            **controller.to_json(),
            "train_accuracy": accuracy,
            "train_positive_rate": _mean(y.astype(float)),
            "policy": chosen,
            "label": "1 if no later hop of the training loop adds a gold passage",
        },
    }
    return TrainOutcome(alpha, POLICIES[chosen], controller, report)


def summarise(
    rows: list[dict[str, object]], reference: list[dict[str, object]] | None, cfg: BenchConfig
) -> dict[str, object]:
    out: dict[str, object] = {"n": len(rows)}
    metrics: dict[str, object] = {}
    for m in (*METRICS, "recall@pool", "complete@10"):
        values = [float(r[m]) for r in rows]  # type: ignore[arg-type]
        ci = bootstrap_ci(
            values, resamples=cfg.bootstrap_resamples, level=cfg.ci_level, seed=cfg.seed
        )
        metrics[m] = {
            "mean": round(ci.mean, 4),
            "ci_low": round(ci.low, 4),
            "ci_high": round(ci.high, 4),
        }
    out["metrics"] = metrics
    out["cost"] = {
        "retrieval_calls_mean": round(float(np.mean([r["retrieval_calls"] for r in rows])), 3),
        "reader_passages_mean": round(float(np.mean([r["reader_passages"] for r in rows])), 3),
        "steps_mean": round(float(np.mean([r["steps"] for r in rows])), 3),
        "pool_size_mean": round(float(np.mean([r["pool_size"] for r in rows])), 3),
        "pool_tokens_mean": round(float(np.mean([r["pool_tokens"] for r in rows])), 1),
    }
    out["stop_reasons"] = dict(Counter(str(r["stop_reason"]) for r in rows))
    if reference is not None:
        tests: dict[str, object] = {}
        for m in METRICS:
            a = np.array([float(r[m]) for r in rows])  # type: ignore[arg-type]
            b = np.array([float(r[m]) for r in reference])  # type: ignore[arg-type]
            diff = bootstrap_ci(
                a - b, resamples=cfg.bootstrap_resamples, level=cfg.ci_level, seed=cfg.seed
            )
            p = paired_permutation_test(a, b, permutations=cfg.permutations, seed=cfg.seed)
            tests[m] = {
                "diff": round(diff.mean, 4),
                "ci_low": round(diff.low, 4),
                "ci_high": round(diff.high, 4),
                "p_value": round(p, 5),
            }
        out[f"vs_{REFERENCE}"] = tests
    return out


def breakdown(rows: list[dict[str, object]], key: str) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return {
        g: {
            "n": len(items),
            **{m: round(float(np.mean([float(r[m]) for r in items])), 4) for m in METRICS},  # type: ignore[arg-type]
        }
        for g, items in sorted(groups.items())
    }


def _percentile(values: Sequence[float], q: float) -> float:
    return round(float(np.percentile(np.asarray(values), q)), 2)


def latency_stage(
    ws: Workspace,
    specs: Sequence[ConfigSpec],
    models: Models,
    trained: TrainOutcome,
    cfg: BenchConfig,
) -> dict[str, dict[str, object]]:
    """Wall time per question with cold caches: retrieval, reasoning and answer reading."""
    questions = ws.questions[: cfg.latency_n]
    dense = ws.retrievers["dense"]
    assert isinstance(dense, Dense)
    out: dict[str, dict[str, object]] = {}
    for spec in specs:
        times: list[float] = []
        for q in questions:
            dense.forget_queries()
            cache = ReaderCache(models.reader, ws.corpus)
            loop = build_loop(spec, ws, cache, trained.policy, trained.controller, cfg)
            t0 = time.perf_counter()
            result = loop.run(q)
            best_answer(cache, q.text, result.ranking[: cfg.answer_depth])
            times.append((time.perf_counter() - t0) * 1000.0)
        out[spec.name] = {
            "n": len(times),
            "p50_ms": _percentile(times, 50),
            "p95_ms": _percentile(times, 95),
        }
    return out


def run_dataset(
    dataset: str, models: Models, cfg: BenchConfig, log: Log, traces: Path | None
) -> dict[str, object]:
    questions = load_questions(dataset)
    eval_q, train_q = split_sample(questions, cfg.n_eval, cfg.n_train, cfg.seed)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    log(f"[{dataset}] building training workspace ({len(train_q)} questions)")
    train_ws = Workspace.build(train_q, models, cfg, log)
    log(f"[{dataset}] building evaluation workspace ({len(eval_q)} questions)")
    eval_ws = Workspace.build(eval_q, models, cfg, log)
    timings["index_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    log(f"[{dataset}] choosing the hybrid weight and query policy, fitting the stop controller")
    trained = train_stage(train_ws, models, cfg, log)
    eval_ws.use_alpha(trained.alpha, cfg.hybrid_depth)
    timings["train_s"] = round(time.perf_counter() - t0, 1)

    specs = [s for s in CONFIGS if not s.only or dataset in s.only]
    cache = eval_ws.reader_cache(models)
    per_question: dict[str, list[dict[str, object]]] = {}
    t0 = time.perf_counter()
    for spec in specs:
        loop = build_loop(spec, eval_ws, cache, trained.policy, trained.controller, cfg)
        rows: list[dict[str, object]] = []
        sink = None
        if traces is not None:
            (traces / dataset).mkdir(parents=True, exist_ok=True)
            sink = (traces / dataset / f"{spec.name}.jsonl").open("w", encoding="utf-8")
        try:
            for i, q in enumerate(eval_q, start=1):
                result = loop.run(q)
                scored = score_question(q, result, eval_ws, cache, cfg)
                rows.append(scored)
                if sink is not None:
                    sink.write(json.dumps(trace_record(result, scored), ensure_ascii=False) + "\n")
                if i % 250 == 0 and i < len(eval_q):
                    elapsed = time.perf_counter() - t0
                    log(f"[{dataset}] {spec.name}: {i}/{len(eval_q)} questions, {elapsed:.0f} s")
        finally:
            if sink is not None:
                sink.close()
        per_question[spec.name] = rows
        cache.save()
        log(
            f"[{dataset}] {spec.name:<21} R@5 {np.mean([r['recall@5'] for r in rows]):.3f}  "
            f"F1 {np.mean([r['f1'] for r in rows]):.3f}  (reader passages so far {cache.computed})"
        )
    timings["eval_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    latency = latency_stage(eval_ws, specs, models, trained, cfg) if cfg.latency_n > 0 else {}
    timings["latency_s"] = round(time.perf_counter() - t0, 1)

    reference = per_question[REFERENCE]
    results: dict[str, object] = {}
    for spec in specs:
        rows = per_question[spec.name]
        entry = summarise(rows, None if spec.name == REFERENCE else reference, cfg)
        entry["description"] = spec.description
        entry["latency"] = latency.get(spec.name, {"status": "not measured"})
        entry["by_hops" if dataset != "hotpotqa" else "by_type"] = breakdown(
            rows, "hops" if dataset != "hotpotqa" else "kind"
        )
        results[spec.name] = entry

    summary = {
        "eval": {
            "n": len(eval_q),
            "ids_sha256": ids_digest(eval_q),
            "corpus_passages": len(eval_ws.corpus),
            "corpus_sha256": eval_ws.corpus.digest(),
            "composition": dict(Counter(q.kind for q in eval_q)),
        },
        "train": {
            "n": len(train_q),
            "ids_sha256": ids_digest(train_q),
            "corpus_passages": len(train_ws.corpus),
            "corpus_sha256": train_ws.corpus.digest(),
        },
        **trained.report,
        "results": results,
        "reader_passages_computed": cache.computed,
        "timings_s": timings,
    }
    return summary


def run_benchmark(
    cfg: BenchConfig, command: str, out: Path, traces_root: Path | None, log: Log = print
) -> dict[str, object]:
    started = time.perf_counter()
    models = load_models(cfg.profile)
    cfg_dict = asdict(cfg)
    digest = config_hash(cfg_dict)
    run_id = f"{local_date()}-{digest[:10]}"
    traces = traces_root / run_id if traces_root is not None else None
    datasets: dict[str, object] = {}
    checksums: dict[str, dict[str, object]] = {}
    if cfg.profile == "models":
        from ircot.sources import dataset_checksums

        checksums = dataset_checksums()
    for dataset in cfg.datasets:
        summary = run_dataset(dataset, models, cfg, log, traces)
        source = checksums.get(dataset, {"source": "ircot.offline.synthetic_world(seed=7)"})
        datasets[dataset] = source | summary

    payload: dict[str, object] = {
        "benchmark": "ircot-multihop-retrieval",
        "profile": cfg.profile,
        "date": local_date(),
        "timestamp": local_timestamp(),
        "git_sha": git_sha(),
        "command": command,
        "machine": machine_info(),
        "config": cfg_dict,
        "config_hash": digest,
        "configs": {s.name: asdict(s) for s in CONFIGS},
        "models": models.info,
        "datasets": datasets,
        "not_run": list(NOT_RUN),
        "wall_time_s": round(time.perf_counter() - started, 1),
    }
    if cfg.profile == "offline":
        payload["note"] = (
            "offline stand-ins on a synthetic world; the numbers say nothing about real models"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if traces is not None:
        traces.mkdir(parents=True, exist_ok=True)
        manifest = {
            k: payload[k]
            for k in (
                "date",
                "timestamp",
                "git_sha",
                "command",
                "config",
                "config_hash",
                "machine",
                "models",
            )
        }
        manifest["datasets"] = {
            name: {
                k: v
                for k, v in info.items()
                if k in ("file", "sha256", "bytes", "source", "eval", "train")
            }  # type: ignore[union-attr]
            for name, info in datasets.items()
        }
        manifest["results_file"] = str(out)
        (traces / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log(f"traces    {traces}")
    log(f"results   {out} ({payload['wall_time_s']} s)")
    return payload
