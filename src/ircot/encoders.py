"""Sentence encoders for dense retrieval: the bge-small ONNX model and its on-disk cache."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ircot.sessions import cpu_session

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    name: str

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class OnnxBgeEncoder:
    """bge-small-en-v1.5: CLS pooling, L2 normalisation, instruction prefix on queries only."""

    def __init__(
        self,
        model_path: Path,
        tokenizer_path: Path,
        *,
        max_length: int = 256,
        batch_size: int = 32,
        threads: int | None = None,
        name: str = "bge-small-en-v1.5",
    ) -> None:
        from tokenizers import Tokenizer

        self.name = name
        self.batch_size = batch_size
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.no_padding()
        self.session = cpu_session(model_path, threads)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(list(texts))
        # Length-sorted batches keep padding (wasted compute) small.
        order = np.argsort([len(e.ids) for e in encodings], kind="stable")
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for lo in range(0, len(order), self.batch_size):
            idx = order[lo : lo + self.batch_size]
            width = max(len(encodings[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), np.int64)
            mask = np.zeros_like(ids)
            for r, i in enumerate(idx):
                n = len(encodings[i].ids)
                ids[r, :n] = encodings[i].ids
                mask[r, :n] = 1
            (hidden,) = self.session.run(
                ["last_hidden_state"],
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)},
            )
            out[idx] = hidden[:, 0, :]
        return l2_normalise(out)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([BGE_QUERY_PREFIX + t for t in texts])

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


def cached_passage_embeddings(
    encoder: Encoder,
    passages: Sequence[str],
    cache_root: Path | None,
    log: Callable[[str], None] = print,
) -> np.ndarray:
    """Encode ``passages`` once per (encoder, exact passage list); later runs load the matrix."""
    if cache_root is None:
        return encoder.encode_passages(passages)
    h = hashlib.sha256(encoder.name.encode("utf-8"))
    for p in passages:
        h.update(p.encode("utf-8") + b"\x1e")
    path = cache_root / f"emb-{encoder.name}-{h.hexdigest()[:20]}.npy"
    if path.exists():
        log(f"embeddings  cached {path.name}")
        return np.load(path)
    cache_root.mkdir(parents=True, exist_ok=True)
    log(f"embeddings  encoding {len(passages)} passages with {encoder.name}")
    matrix = encoder.encode_passages(passages)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, matrix)
    tmp.replace(path)
    return matrix
