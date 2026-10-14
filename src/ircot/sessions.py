"""onnxruntime CPU sessions shared by the encoder and the reader."""

from __future__ import annotations

import os
from pathlib import Path

import onnxruntime as ort


def cpu_session(model_path: Path, threads: int | None = None) -> ort.InferenceSession:
    """A CPU session using every core, with worker-thread spinning turned off.

    Spinning workers busy-wait between operators. When other processes share the CPU they lose
    their time slices while spinning, and on the development machine that made the reader
    roughly twice as slow; with spinning off, idle workers sleep instead.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads or os.cpu_count() or 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
