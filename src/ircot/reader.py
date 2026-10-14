"""Extractive span reader: the reasoner's eyes in the offline IRCoT loop.

``OnnxSquadReader`` runs the SQuAD 2.0 MiniLM reader (exported by ``ircot.onnx_bert``) over
(question, passage) pairs. Long passages are split into overlapping windows, spans never cross the
question or special tokens, and every span keeps its character offsets so the loop can recover the
evidence sentence and match the span against corpus titles.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    score: float
    null_score: float

    @property
    def margin(self) -> float:
        """How much the reader prefers this span over answering "no answer"."""
        return self.score - self.null_score


class Reader(Protocol):
    name: str

    def read(self, question: str, passages: Sequence[str], top_n: int = 3) -> list[list[Span]]:
        """Up to ``top_n`` non-overlapping spans per passage, best first."""
        ...


def _top_spans(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    valid: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    passage: str,
    top_n: int,
    max_tokens: int,
    candidates: int = 20,
) -> list[Span]:
    null = float(start_logits[0] + end_logits[0])
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    starts = idx[np.argsort(-start_logits[idx], kind="stable")[:candidates]]
    ends = idx[np.argsort(-end_logits[idx], kind="stable")[:candidates]]
    scored: list[tuple[float, int, int]] = [
        (float(start_logits[s] + end_logits[e]), int(s), int(e))
        for s in starts
        for e in ends
        if s <= e < s + max_tokens
    ]
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    chosen: list[Span] = []
    taken: list[tuple[int, int]] = []
    for score, s, e in scored:
        lo, hi = offsets[s][0], offsets[e][1]
        if hi <= lo or any(lo < b and a < hi for a, b in taken):
            continue
        text = passage[lo:hi].strip()
        if not text:
            continue
        chosen.append(Span(text, lo, hi, score, null))
        taken.append((lo, hi))
        if len(chosen) == top_n:
            break
    return chosen


class OnnxSquadReader:
    """SQuAD 2.0 span reader on onnxruntime with WordPiece tokenisation from ``vocab.txt``."""

    def __init__(
        self,
        model_path: Path,
        vocab_path: Path,
        *,
        max_length: int = 320,
        stride: int = 96,
        max_answer_tokens: int = 24,
        batch_size: int = 16,
        threads: int | None = None,
        name: str = "minilm-uncased-squad2",
    ) -> None:
        import onnxruntime as ort
        from tokenizers import BertWordPieceTokenizer

        self.name = name
        self.max_answer_tokens = max_answer_tokens
        self.batch_size = batch_size
        self.tokenizer = BertWordPieceTokenizer(str(vocab_path), lowercase=True)
        self.tokenizer.enable_truncation(max_length, stride=stride, strategy="only_second")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or os.cpu_count() or 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )

    def read(self, question: str, passages: Sequence[str], top_n: int = 3) -> list[list[Span]]:
        windows: list[tuple[int, object]] = []
        for i, passage in enumerate(passages):
            enc = self.tokenizer.encode(question, passage)
            windows.extend((i, w) for w in [enc, *enc.overflowing])
        best: list[list[Span]] = [[] for _ in passages]
        order = sorted(range(len(windows)), key=lambda j: len(windows[j][1].ids))  # type: ignore[attr-defined]
        for lo in range(0, len(order), self.batch_size):
            chunk = [windows[j] for j in order[lo : lo + self.batch_size]]
            width = max(len(w.ids) for _, w in chunk)  # type: ignore[attr-defined]
            ids = np.zeros((len(chunk), width), np.int64)
            types = np.zeros_like(ids)
            mask = np.zeros_like(ids)
            for r, (_, w) in enumerate(chunk):
                n = len(w.ids)  # type: ignore[attr-defined]
                ids[r, :n] = w.ids  # type: ignore[attr-defined]
                types[r, :n] = w.type_ids  # type: ignore[attr-defined]
                mask[r, :n] = 1
            start, end = self.session.run(
                ["start_logits", "end_logits"],
                {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
            )
            for r, (pi, w) in enumerate(chunk):
                seq = np.array([s == 1 for s in w.sequence_ids] + [False] * (width - len(w.ids)))  # type: ignore[attr-defined]
                spans = _top_spans(
                    start[r],
                    end[r],
                    seq,
                    w.offsets,  # type: ignore[attr-defined]
                    passages[pi],
                    top_n,
                    self.max_answer_tokens,
                )
                best[pi].extend(spans)
        return [_merge_windows(spans, top_n) for spans in best]


def _merge_windows(spans: list[Span], top_n: int) -> list[Span]:
    """Keep the best non-overlapping spans across a passage's windows.

    Each window has its own no-answer score; the passage-level null is the smallest one, as in the
    reference SQuAD 2.0 post-processing (a passage is answerable if any window says so).
    """
    if not spans:
        return []
    null = min(s.null_score for s in spans)
    out: list[Span] = []
    for span in sorted(spans, key=lambda s: (-s.score, s.start)):
        if any(span.start < o.end and o.start < span.end for o in out):
            continue
        out.append(Span(span.text, span.start, span.end, span.score, null))
        if len(out) == top_n:
            break
    return out
