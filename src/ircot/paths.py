"""Where datasets, models, caches and run traces live.

Nothing here points at a machine-specific location: the root comes from ``IRCOT_DATA_DIR`` and
falls back to ``~/.cache/ircot-multihop-retrieval``. Pretrained models go through the Hugging
Face cache (``HF_HOME``) so one copy can be shared with other tools on the same machine.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "IRCOT_DATA_DIR"


def data_dir() -> Path:
    configured = os.environ.get(ENV_VAR)
    root = Path(configured) if configured else Path.home() / ".cache" / "ircot-multihop-retrieval"
    return root.expanduser()


def raw_dir() -> Path:
    return data_dir() / "raw"


def models_dir() -> Path:
    return data_dir() / "models"


def evaluator_dir() -> Path:
    return data_dir() / "evaluator" / "musique"


def cache_dir() -> Path:
    """Embedding matrices keyed by corpus and model hash; safe to delete at any time."""
    return data_dir() / "cache"


def runs_dir() -> Path:
    """Per-question JSONL traces and run manifests written by the benchmark harness."""
    return data_dir() / "runs"
