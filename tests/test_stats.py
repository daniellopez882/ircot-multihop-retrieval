from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats as sps

from ircot.stats import bootstrap_ci, paired_permutation_test


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=60))
def test_bootstrap_interval_brackets_the_mean(values: list[float]) -> None:
    ci = bootstrap_ci(values, resamples=300, seed=1)
    assert ci.low - 1e-9 <= ci.mean <= ci.high + 1e-9
    assert ci.mean == pytest.approx(float(np.mean(values)))


def test_bootstrap_is_seeded() -> None:
    values = np.random.default_rng(0).random(50)
    assert bootstrap_ci(values, seed=4) == bootstrap_ci(values, seed=4)


def test_identical_systems_are_not_different() -> None:
    a = np.random.default_rng(0).random(30)
    assert paired_permutation_test(a, a, permutations=999) == 1.0


def test_p_value_matches_the_exact_sign_flip_test() -> None:
    rng = np.random.default_rng(7)
    a = rng.random(10)
    b = a - rng.normal(0.15, 0.2, size=10)
    exact = sps.permutation_test(
        (a, b),
        lambda x, y: np.mean(x - y),
        permutation_type="samples",
        n_resamples=np.inf,
        alternative="two-sided",
    ).pvalue
    ours = paired_permutation_test(a, b, permutations=20000, seed=3)
    assert ours == pytest.approx(exact, abs=0.01)


def test_clear_differences_get_small_p_values() -> None:
    a = np.ones(200)
    b = np.zeros(200)
    assert paired_permutation_test(a, b, permutations=2000) == pytest.approx(1 / 2001)
