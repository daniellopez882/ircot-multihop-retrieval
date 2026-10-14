"""The learned stop controller: an L2-regularised logistic regression on per-hop loop signals.

After each hop the loop computes a small feature vector (how confident the reader is, whether the
answer moved, whether the bridge linked to a title, how much the hop added...). The controller
estimates the probability that stopping now loses no gold passage that further hops would have
found, and the loop stops when that probability crosses a threshold.

The model is fitted with Newton's method (iteratively reweighted least squares) on standardised
features. There is no randomness anywhere in the fit, so the same traces give the same weights.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FEATURES: tuple[str, ...] = (
    "hop",
    "answer_margin",
    "margin_gain",
    "answer_stable",
    "bridge_found",
    "bridge_title_match",
    "novelty",
    "top_score",
    "question_coverage",
)


def fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-10
) -> tuple[np.ndarray, float, int]:
    """Minimise mean log-loss + ``l2 / (2n) * ||w||^2`` (intercept unpenalised).

    That objective equals scikit-learn's ``LogisticRegression(C=1/l2)`` up to a factor ``n``, which
    tests/test_controller.py uses as the parity reference. Returns (weights, intercept, iterations).
    """
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    penalty = np.full(d + 1, l2)
    penalty[-1] = 0.0
    iterations = 0
    for iterations in range(1, max_iter + 1):  # noqa: B007 - the count is returned
        z = A @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
        grad = A.T @ (p - y) + penalty * theta
        weights = p * (1.0 - p)
        hess = (A * weights[:, None]).T @ A + np.diag(penalty) + 1e-12 * np.eye(d + 1)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if float(np.max(np.abs(step))) < tol:
            break
    return theta[:-1], float(theta[-1]), iterations


@dataclass
class StopController:
    features: tuple[str, ...]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    threshold: float
    l2: float
    train_rows: int
    train_positive: int
    iterations: int

    @classmethod
    def fit(
        cls, X: np.ndarray, y: np.ndarray, *, l2: float = 1.0, threshold: float = 0.5
    ) -> StopController:
        if X.ndim != 2 or X.shape[1] != len(FEATURES):
            raise ValueError(f"expected [rows, {len(FEATURES)}] features, got {X.shape}")
        if X.shape[0] == 0:
            raise ValueError("stop-controller training data is empty")
        mean = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0
        if len(np.unique(y)) < 2:
            # One label only (tiny samples): the unpenalised intercept would diverge, so fall back
            # to the add-one-half smoothed base rate and ignore the features.
            positive = float(y.sum())
            intercept = float(np.log((positive + 0.5) / (len(y) - positive + 0.5)))
            coef, iterations = np.zeros(len(FEATURES)), 0
        else:
            coef, intercept, iterations = fit_logistic((X - mean) / scale, y.astype(np.float64), l2)
        return cls(
            features=FEATURES,
            mean=mean.tolist(),
            scale=scale.tolist(),
            coef=coef.tolist(),
            intercept=intercept,
            threshold=threshold,
            l2=l2,
            train_rows=int(X.shape[0]),
            train_positive=int(y.sum()),
            iterations=iterations,
        )

    def probability(self, x: Sequence[float]) -> float:
        z = (np.asarray(x, dtype=np.float64) - np.asarray(self.mean)) / np.asarray(self.scale)
        logit = float(z @ np.asarray(self.coef) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -35.0, 35.0))))

    def to_json(self) -> dict[str, object]:
        out = asdict(self)
        out["features"] = list(self.features)
        return out

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> StopController:
        data = dict(payload)
        data["features"] = tuple(data["features"])  # type: ignore[arg-type]
        if data["features"] != FEATURES:
            raise ValueError("controller was trained on a different feature set")
        return cls(**data)  # type: ignore[arg-type]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> StopController:
        return cls.from_json(json.loads(path.read_text(encoding="utf-8")))
