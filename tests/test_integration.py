"""Checks against the real datasets, models and evaluator. Skipped unless `ircot fetch` ran."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ircot import sources

pytestmark = pytest.mark.integration


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} not fetched (run `ircot fetch`)")
    return path


@pytest.fixture(scope="module")
def reader():  # type: ignore[no-untyped-def]
    from ircot.reader import OnnxSquadReader

    onnx_path = _require(sources.reader_onnx_path())
    try:
        vocab = sources.model_files(sources.MINILM_SQUAD2)["vocab.txt"]
    except sources.MissingAssetError:
        pytest.skip("reader not cached")
    return OnnxSquadReader(onnx_path, vocab)


def test_reader_answers_a_reading_comprehension_question(reader) -> None:  # type: ignore[no-untyped-def]
    passage = (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built it."
    )
    (spans,) = reader.read("Who is the Eiffel Tower named after?", [passage])
    assert "Gustave Eiffel" in spans[0].text
    assert spans[0].margin > 0
    assert passage[spans[0].start : spans[0].end] == spans[0].text


def test_reader_windows_find_an_answer_past_the_first_window(reader) -> None:  # type: ignore[no-untyped-def]
    filler = "The weather that day was unremarkable and nothing else happened. " * 40
    passage = filler + "The treaty was signed by Queen Matilda in 1141."
    (spans,) = reader.read("Who signed the treaty?", [passage])
    assert any("Matilda" in s.text for s in spans)


def test_bge_embeddings_are_normalised_and_rank_the_relevant_passage_first() -> None:
    from ircot.encoders import OnnxBgeEncoder

    try:
        files = sources.model_files(sources.BGE_SMALL)
    except sources.MissingAssetError:
        pytest.skip("bge-small not cached")
    enc = OnnxBgeEncoder(files["onnx/model.onnx"], files["tokenizer.json"])
    passages = ["Paris is the capital of France.", "Bananas are rich in potassium."]
    emb = enc.encode_passages(passages)
    q = enc.encode_queries(["What is the capital of France?"])[0]
    assert emb.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-5)
    assert int(np.argmax(emb @ q)) == 0


def test_datasets_load_with_the_documented_sizes() -> None:
    from ircot.data import load_hotpotqa, load_musique

    musique = load_musique(_require(sources.musique_path()))
    hotpot = load_hotpotqa(_require(sources.hotpotqa_path()))
    assert len(musique) == 2417
    assert len(hotpot) == 7405
    assert {q.hops for q in musique} == {2, 3, 4}
    assert all(sum(p.supporting for p in q.paragraphs) == q.hops for q in musique)
    assert {q.kind for q in hotpot} == {"bridge", "comparison"}


def test_answer_metric_matches_the_official_evaluator_on_real_predictions(
    tmp_path: Path, reader
) -> None:  # type: ignore[no-untyped-def]
    from ircot.data import load_musique
    from ircot.metrics import best_em, best_f1
    from ircot.paths import evaluator_dir

    if not sources.evaluator_present():
        pytest.skip("official evaluator not fetched")
    raw = [
        json.loads(line)
        for line in _require(sources.musique_path()).read_text("utf-8").splitlines()[:40]
    ]
    questions = load_musique(sources.musique_path())[:40]
    preds = []
    ours_em, ours_f1 = [], []
    for record, q in zip(raw, questions, strict=True):
        texts = [p.text for p in q.paragraphs[:4]]
        spans = [s for per in reader.read(q.text, texts) for s in per]
        answer = max(spans, key=lambda s: s.margin).text if spans else ""
        ours_em.append(best_em(answer, q.answers))
        ours_f1.append(best_f1(answer, q.answers))
        preds.append(
            {
                "id": record["id"],
                "predicted_answer": answer,
                "predicted_support_idxs": [0, 1],
                "predicted_answerable": True,
            }
        )
    gold_path, pred_path = tmp_path / "gold.jsonl", tmp_path / "pred.jsonl"
    gold_path.write_text("\n".join(json.dumps(r) for r in raw) + "\n", encoding="utf-8")
    pred_path.write_text("\n".join(json.dumps(p) for p in preds) + "\n", encoding="utf-8")
    sys.path.insert(0, str(evaluator_dir()))
    try:
        spec = importlib.util.spec_from_file_location(
            "musique_eval", evaluator_dir() / "evaluate_v1.0.py"
        )
        assert spec is not None
        assert spec.loader is not None
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        result = official.evaluate(str(pred_path), str(gold_path))
    finally:
        sys.path.remove(str(evaluator_dir()))
    assert result["answer_em"] == pytest.approx(round(float(np.mean(ours_em)), 3), abs=1e-3)
    assert result["answer_f1"] == pytest.approx(round(float(np.mean(ours_f1)), 3), abs=1e-3)
