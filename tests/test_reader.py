"""The hand-built ONNX BERT graph against a NumPy reference, and span decoding."""

from __future__ import annotations

import itertools

import numpy as np
import onnxruntime as ort
import pytest
from scipy.special import erf

from ircot.onnx_bert import BertShape, build_qa_graph
from ircot.reader import Span, _merge_windows, _top_spans

SHAPE = BertShape(hidden=16, layers=2, heads=4, intermediate=32, eps=1e-12)
VOCAB, POSITIONS = 40, 32


def random_weights(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, f = SHAPE.hidden, SHAPE.intermediate

    def r(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.3).astype(np.float32)

    w = {
        "bert.embeddings.word_embeddings.weight": r(VOCAB, h),
        "bert.embeddings.position_embeddings.weight": r(POSITIONS, h),
        "bert.embeddings.token_type_embeddings.weight": r(2, h),
        "bert.embeddings.LayerNorm.weight": 1 + r(h),
        "bert.embeddings.LayerNorm.bias": r(h),
        "qa_outputs.weight": r(2, h),
        "qa_outputs.bias": r(2),
    }
    for i in range(SHAPE.layers):
        p = f"bert.encoder.layer.{i}."
        for kind in ("query", "key", "value"):
            w[f"{p}attention.self.{kind}.weight"] = r(h, h)
            w[f"{p}attention.self.{kind}.bias"] = r(h)
        w[f"{p}attention.output.dense.weight"] = r(h, h)
        w[f"{p}attention.output.dense.bias"] = r(h)
        w[f"{p}attention.output.LayerNorm.weight"] = 1 + r(h)
        w[f"{p}attention.output.LayerNorm.bias"] = r(h)
        w[f"{p}intermediate.dense.weight"] = r(f, h)
        w[f"{p}intermediate.dense.bias"] = r(f)
        w[f"{p}output.dense.weight"] = r(h, f)
        w[f"{p}output.dense.bias"] = r(h)
        w[f"{p}output.LayerNorm.weight"] = 1 + r(h)
        w[f"{p}output.LayerNorm.bias"] = r(h)
    return w


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + SHAPE.eps) * g + b


def reference_forward(
    w: dict[str, np.ndarray], ids: np.ndarray, mask: np.ndarray, types: np.ndarray
):  # type: ignore[no-untyped-def]
    """The textbook BERT-for-QA forward pass, written independently of the ONNX builder."""
    e = "bert.embeddings."
    x = w[e + "word_embeddings.weight"][ids] + w[e + "position_embeddings.weight"][: ids.shape[1]]
    x = layer_norm(
        x + w[e + "token_type_embeddings.weight"][types],
        w[e + "LayerNorm.weight"],
        w[e + "LayerNorm.bias"],
    )
    bsz, seq, h = x.shape
    nh, hd = SHAPE.heads, h // SHAPE.heads
    add_mask = (1.0 - mask[:, None, None, :]) * -10000.0
    for i in range(SHAPE.layers):
        p = f"bert.encoder.layer.{i}."

        def lin(t: np.ndarray, name: str, p: str = p) -> np.ndarray:
            return t @ w[p + name + ".weight"].T + w[p + name + ".bias"]

        def heads(t: np.ndarray) -> np.ndarray:
            return t.reshape(bsz, seq, nh, hd).transpose(0, 2, 1, 3)

        q, k, v = (heads(lin(x, f"attention.self.{n}")) for n in ("query", "key", "value"))
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + add_mask
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(bsz, seq, h)
        x = layer_norm(
            x + lin(ctx, "attention.output.dense"),
            w[p + "attention.output.LayerNorm.weight"],
            w[p + "attention.output.LayerNorm.bias"],
        )
        inter = lin(x, "intermediate.dense")
        inter = 0.5 * inter * (1 + erf(inter / np.sqrt(2)))
        x = layer_norm(
            x + lin(inter, "output.dense"),
            w[p + "output.LayerNorm.weight"],
            w[p + "output.LayerNorm.bias"],
        )
    logits = x @ w["qa_outputs.weight"].T + w["qa_outputs.bias"]
    return logits[..., 0], logits[..., 1]


def test_onnx_graph_matches_the_numpy_reference() -> None:
    w = random_weights()
    model = build_qa_graph(w, SHAPE)
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    ids = rng.integers(0, VOCAB, size=(3, 11)).astype(np.int64)
    mask = np.ones_like(ids)
    mask[1, 8:] = 0
    mask[2, 5:] = 0
    types = np.zeros_like(ids)
    types[:, 4:] = 1
    start, end = session.run(
        ["start_logits", "end_logits"],
        {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
    )
    ref_start, ref_end = reference_forward(w, ids, mask.astype(np.float32), types)
    np.testing.assert_allclose(start, ref_start, atol=2e-4)
    np.testing.assert_allclose(end, ref_end, atol=2e-4)


def test_padding_does_not_change_the_unpadded_logits() -> None:
    w = random_weights(seed=5)
    session = ort.InferenceSession(
        build_qa_graph(w, SHAPE).SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ids = np.array([[2, 7, 9, 3, 11, 12]], dtype=np.int64)
    feeds = {
        "input_ids": ids,
        "attention_mask": np.ones_like(ids),
        "token_type_ids": np.zeros_like(ids),
    }
    short = session.run(["start_logits"], feeds)[0]
    padded_ids = np.concatenate([ids, np.zeros((1, 4), np.int64)], axis=1)
    mask = np.concatenate([np.ones_like(ids), np.zeros((1, 4), np.int64)], axis=1)
    feeds = {
        "input_ids": padded_ids,
        "attention_mask": mask,
        "token_type_ids": np.zeros_like(padded_ids),
    }
    long = session.run(["start_logits"], feeds)[0]
    np.testing.assert_allclose(long[:, :6], short, atol=1e-4)


PASSAGE = "Green is an album by Steve Hillage"
# token offsets for: [CLS] q [SEP] green is an album by steve hill ##age [SEP]
OFFSETS = [
    (0, 0),
    (0, 0),
    (0, 0),
    (0, 5),
    (6, 8),
    (9, 11),
    (12, 17),
    (18, 20),
    (21, 26),
    (27, 31),
    (31, 34),
    (0, 0),
]
VALID = np.array([False, False, False] + [True] * 8 + [False])


def test_span_decoding_respects_passage_tokens_and_length() -> None:
    start = np.full(12, -5.0)
    end = np.full(12, -5.0)
    start[0], end[0] = 1.0, 1.0  # no-answer
    start[8], end[10] = 6.0, 5.0  # "Steve Hillage"
    start[1], end[1] = 50.0, 50.0  # a question token must never win
    spans = _top_spans(start, end, VALID, OFFSETS, PASSAGE, top_n=2, max_tokens=5)
    assert spans[0].text == "Steve Hillage"
    assert spans[0].margin == pytest.approx(11.0 - 2.0)
    assert all(s.start >= 0 and PASSAGE[s.start : s.end] == s.text for s in spans)
    assert not any(a.start < b.end and b.start < a.end for a, b in itertools.pairwise(spans))
    short = _top_spans(start, end, VALID, OFFSETS, PASSAGE, top_n=1, max_tokens=2)
    assert short[0].text != "Steve Hillage"


def test_windows_merge_with_the_smallest_null_score() -> None:
    a = Span("Steve Hillage", 21, 34, 4.0, 3.0)
    b = Span("Hillage", 27, 34, 3.0, -1.0)
    c = Span("Green", 0, 5, 2.0, -1.0)
    merged = _merge_windows([a, b, c], top_n=3)
    assert [s.text for s in merged] == ["Steve Hillage", "Green"]
    assert all(s.null_score == -1.0 for s in merged)
    assert _merge_windows([], 3) == []
