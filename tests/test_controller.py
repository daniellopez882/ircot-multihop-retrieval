from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ircot.controller import FEATURES, StopController, fit_logistic
from ircot.harness import controller_rows
from ircot.loop import HopRecord, LoopResult


def synthetic_rows(n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES)))
    w = np.linspace(-1.5, 1.5, len(FEATURES))
    p = 1 / (1 + np.exp(-(X @ w + 0.3)))
    return X, (rng.random(n) < p).astype(np.int64)


def test_training_is_deterministic() -> None:
    X, y = synthetic_rows()
    a = StopController.fit(X, y, l2=2.0)
    b = StopController.fit(X.copy(), y.copy(), l2=2.0)
    assert a.to_json() == b.to_json()


@pytest.mark.parametrize("l2", [0.1, 1.0, 10.0])
def test_newton_fit_matches_scikit_learn(l2: float) -> None:
    X, y = synthetic_rows()
    coef, intercept, iterations = fit_logistic(X, y.astype(float), l2=l2)
    ref = LogisticRegression(C=1.0 / l2, tol=1e-12, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(coef, ref.coef_[0], atol=1e-5)
    assert intercept == pytest.approx(ref.intercept_[0], abs=1e-5)
    assert iterations < 50


def test_probability_and_round_trip() -> None:
    X, y = synthetic_rows()
    model = StopController.fit(X, y)
    clone = StopController.from_json(model.to_json())
    for row in X[:20]:
        assert clone.probability(row) == model.probability(row)
        assert 0.0 < model.probability(row) < 1.0
    assert model.train_rows == len(y)
    assert model.train_positive == int(y.sum())


def test_constant_features_do_not_break_standardisation() -> None:
    X, y = synthetic_rows()
    X[:, 0] = 4.0
    model = StopController.fit(X, y)
    assert model.scale[0] == 1.0
    assert np.isfinite(model.probability(X[0]))


def test_single_class_and_wrong_shapes_are_rejected() -> None:
    X, y = synthetic_rows()
    with pytest.raises(ValueError, match="both stop and continue"):
        StopController.fit(X, np.ones_like(y))
    with pytest.raises(ValueError, match="features"):
        StopController.fit(X[:, :3], y)
    payload = StopController.fit(X, y).to_json()
    payload["features"] = ["hop"]
    with pytest.raises(ValueError, match="different feature set"):
        StopController.from_json(payload)


def _hop(hop: int, admitted: list[int], with_features: bool) -> HopRecord:
    feats = {name: float(hop) for name in FEATURES} if with_features else None
    return HopRecord(hop, f"q{hop}", [], admitted, [], [], features=feats)


def test_labels_say_whether_later_hops_add_gold() -> None:
    hops = [
        _hop(1, [0, 1], True),
        _hop(2, [2, 3], True),
        _hop(3, [4, 5], True),
        _hop(4, [6], False),
    ]
    res = LoopResult("q", [0, 1, 2, 3, 4, 5, 6], hops, "step-budget", 4, 0, 0, 0)
    X, y = controller_rows([res], {"q": (1, 4)})
    assert X.shape == (3, len(FEATURES))
    # gold 1 arrives at hop 1, gold 4 at hop 3: stopping is safe only from hop 3 on.
    assert y.tolist() == [0, 0, 1]
    _, y_never = controller_rows([res], {"q": (9,)})
    assert y_never.tolist() == [1, 1, 1]
