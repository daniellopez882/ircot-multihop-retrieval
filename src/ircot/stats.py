"""Uncertainty for per-question metrics: percentile bootstrap CIs and paired sign-flip tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 2000, level: float = 0.95, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for the mean of ``values``."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("bootstrap needs at least one value")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return Interval(
        float(x.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))
    )


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], *, permutations: int = 10000, seed: int = 0
) -> float:
    """Two-sided p-value for "mean(a - b) == 0" by randomly flipping the sign of each difference.

    Uses the (count + 1) / (permutations + 1) estimate, so the p-value is never exactly zero.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0:
        raise ValueError("the test needs at least one pair")
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for lo in range(0, permutations, 1000):
        n = min(1000, permutations - lo)
        signs = rng.choice((-1.0, 1.0), size=(n, d.size))
        hits += int(np.sum(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))
    return (hits + 1) / (permutations + 1)
