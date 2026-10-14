"""Build an ONNX graph for a BERT extractive-QA model directly from its safetensors weights.

The reader checkpoint (deepset/minilm-uncased-squad2) is published as PyTorch/safetensors only.
Exporting it the usual way needs torch, which is far larger than everything else this project
installs, so the graph is written op by op with ``onnx.helper`` instead (see ADR 0002). The graph
is the standard post-LayerNorm BERT encoder with exact erf GELU and a linear span head.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17  # first opset with a native LayerNormalization


@dataclass(frozen=True)
class BertShape:
    hidden: int
    layers: int
    heads: int
    intermediate: int
    eps: float

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> BertShape:
        return cls(
            hidden=int(config["hidden_size"]),  # type: ignore[call-overload]
            layers=int(config["num_hidden_layers"]),  # type: ignore[call-overload]
            heads=int(config["num_attention_heads"]),  # type: ignore[call-overload]
            intermediate=int(config["intermediate_size"]),  # type: ignore[call-overload]
            eps=float(config.get("layer_norm_eps", 1e-12)),  # type: ignore[arg-type]
        )


class _Graph:
    def __init__(self) -> None:
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def name(self, stem: str) -> str:
        self._n += 1
        return f"{stem}_{self._n}"

    def const(self, stem: str, value: np.ndarray) -> str:
        name = self.name(stem)
        self.inits.append(numpy_helper.from_array(value, name))
        return name

    def op(self, op_type: str, inputs: list[str], stem: str, **attrs: object) -> str:
        out = self.name(stem)
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out

    def linear(self, x: str, weight: np.ndarray, bias: np.ndarray, stem: str) -> str:
        # torch Linear stores [out, in]; MatMul wants [in, out].
        w = self.const(f"{stem}_w", np.ascontiguousarray(weight.T.astype(np.float32)))
        b = self.const(f"{stem}_b", bias.astype(np.float32))
        return self.op("Add", [self.op("MatMul", [x, w], f"{stem}_mm"), b], stem)

    def layer_norm(self, x: str, scale: np.ndarray, bias: np.ndarray, eps: float, stem: str) -> str:
        s = self.const(f"{stem}_g", scale.astype(np.float32))
        b = self.const(f"{stem}_b", bias.astype(np.float32))
        return self.op("LayerNormalization", [x, s, b], stem, axis=-1, epsilon=eps)


def build_qa_graph(weights: Mapping[str, np.ndarray], shape: BertShape) -> onnx.ModelProto:
    """ONNX model with inputs ``input_ids``, ``attention_mask``, ``token_type_ids`` (int64, [B, S])
    and outputs ``start_logits``, ``end_logits`` (float32, [B, S])."""
    g = _Graph()
    head_dim = shape.hidden // shape.heads
    p = "bert."

    tokens = g.op(
        "Gather",
        [g.const("word_emb", weights[f"{p}embeddings.word_embeddings.weight"]), "input_ids"],
        "tok",
    )
    seq_len = g.op(
        "Gather",
        [g.op("Shape", ["input_ids"], "shape"), g.const("one", np.array(1, np.int64))],
        "seq",
    )
    positions = g.op(
        "Range",
        [g.const("zero", np.array(0, np.int64)), seq_len, g.const("step", np.array(1, np.int64))],
        "pos_ids",
    )
    pos = g.op(
        "Gather",
        [g.const("pos_emb", weights[f"{p}embeddings.position_embeddings.weight"]), positions],
        "pos",
    )
    types = g.op(
        "Gather",
        [
            g.const("type_emb", weights[f"{p}embeddings.token_type_embeddings.weight"]),
            "token_type_ids",
        ],
        "typ",
    )
    x = g.op("Add", [g.op("Add", [tokens, pos], "emb"), types], "emb")
    x = g.layer_norm(
        x,
        weights[f"{p}embeddings.LayerNorm.weight"],
        weights[f"{p}embeddings.LayerNorm.bias"],
        shape.eps,
        "emb_ln",
    )

    # Additive mask [B, 1, 1, S]: 0 where attended, -10000 on padding (the BERT convention).
    mask = g.op("Cast", ["attention_mask"], "mask_f", to=TensorProto.FLOAT)
    mask = g.op("Sub", [g.const("ones", np.array(1.0, np.float32)), mask], "mask_inv")
    mask = g.op("Mul", [mask, g.const("neg", np.array(-10000.0, np.float32))], "mask_neg")
    mask = g.op("Unsqueeze", [mask, g.const("axes", np.array([1, 2], np.int64))], "mask4d")

    split_heads = g.const("split", np.array([0, 0, shape.heads, head_dim], np.int64))
    merge_heads = g.const("merge", np.array([0, 0, shape.hidden], np.int64))
    scale = g.const("scale", np.array(1.0 / math.sqrt(head_dim), np.float32))
    half = g.const("half", np.array(0.5, np.float32))
    one = g.const("one_f", np.array(1.0, np.float32))
    inv_sqrt2 = g.const("inv_sqrt2", np.array(1.0 / math.sqrt(2.0), np.float32))

    for layer in range(shape.layers):
        lp = f"{p}encoder.layer.{layer}."

        heads: dict[str, str] = {}
        for kind, perm in (("query", [0, 2, 1, 3]), ("key", [0, 2, 3, 1]), ("value", [0, 2, 1, 3])):
            y = g.linear(
                x,
                weights[f"{lp}attention.self.{kind}.weight"],
                weights[f"{lp}attention.self.{kind}.bias"],
                f"l{layer}_{kind}",
            )
            heads[kind] = g.op(
                "Transpose", [g.op("Reshape", [y, split_heads], "heads")], "t", perm=perm
            )
        q, k, v = heads["query"], heads["key"], heads["value"]
        scores = g.op(
            "Add", [g.op("Mul", [g.op("MatMul", [q, k], "qk"), scale], "scaled"), mask], "masked"
        )
        probs = g.op("Softmax", [scores], "probs", axis=-1)
        ctx = g.op("Transpose", [g.op("MatMul", [probs, v], "ctx")], "ctx_t", perm=[0, 2, 1, 3])
        ctx = g.op("Reshape", [ctx, merge_heads], "ctx_m")
        attn = g.linear(
            ctx,
            weights[f"{lp}attention.output.dense.weight"],
            weights[f"{lp}attention.output.dense.bias"],
            f"l{layer}_attn_out",
        )
        x = g.layer_norm(
            g.op("Add", [attn, x], "res1"),
            weights[f"{lp}attention.output.LayerNorm.weight"],
            weights[f"{lp}attention.output.LayerNorm.bias"],
            shape.eps,
            f"l{layer}_ln1",
        )
        inter = g.linear(
            x,
            weights[f"{lp}intermediate.dense.weight"],
            weights[f"{lp}intermediate.dense.bias"],
            f"l{layer}_inter",
        )
        erf = g.op("Erf", [g.op("Mul", [inter, inv_sqrt2], "gelu_in")], "erf")
        gelu = g.op(
            "Mul", [g.op("Mul", [inter, half], "gelu_h"), g.op("Add", [erf, one], "gelu_e")], "gelu"
        )
        out = g.linear(
            gelu,
            weights[f"{lp}output.dense.weight"],
            weights[f"{lp}output.dense.bias"],
            f"l{layer}_out",
        )
        x = g.layer_norm(
            g.op("Add", [out, x], "res2"),
            weights[f"{lp}output.LayerNorm.weight"],
            weights[f"{lp}output.LayerNorm.bias"],
            shape.eps,
            f"l{layer}_ln2",
        )

    logits = g.linear(x, weights["qa_outputs.weight"], weights["qa_outputs.bias"], "qa")
    sizes = g.const("sizes", np.array([1, 1], np.int64))
    last = g.const("last", np.array([-1], np.int64))
    g.nodes.append(helper.make_node("Split", [logits, sizes], ["start3", "end3"], axis=-1))
    g.nodes.append(helper.make_node("Squeeze", ["start3", last], ["start_logits"]))
    g.nodes.append(helper.make_node("Squeeze", ["end3", last], ["end_logits"]))

    dims = ["batch", "sequence"]
    graph = helper.make_graph(
        g.nodes,
        "bert_squad_reader",
        [
            helper.make_tensor_value_info(n, TensorProto.INT64, dims)
            for n in ("input_ids", "attention_mask", "token_type_ids")
        ],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, dims)
            for n in ("start_logits", "end_logits")
        ],
        initializer=g.inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="ircot"
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def export_squad_reader(safetensors_path: Path, config_path: Path, dest: Path) -> None:
    from safetensors.numpy import load_file

    weights = load_file(str(safetensors_path))
    shape = BertShape.from_config(json.loads(config_path.read_text(encoding="utf-8")))
    model = build_qa_graph(weights, shape)
    tmp = dest.with_suffix(".onnx.part")
    onnx.save_model(model, str(tmp))
    tmp.replace(dest)
