from __future__ import annotations

import pytest

from ircot.data import Question, parse_musique, split_sample
from ircot.harness import BenchConfig, Models, Workspace, load_models
from ircot.offline import synthetic_world
from ircot.pool import PassageStats
from ircot.text import word_count


@pytest.fixture(scope="session")
def offline_models() -> Models:
    return load_models("offline")


@pytest.fixture(scope="session")
def offline_config() -> BenchConfig:
    return BenchConfig(
        profile="offline",
        datasets=("synthetic",),
        n_eval=30,
        n_train=30,
        latency_n=3,
        bootstrap_resamples=200,
        permutations=500,
    )


@pytest.fixture(scope="session")
def world() -> tuple[list[Question], list[Question]]:
    """The 30-question CI fixture (evaluation) and a disjoint 30-question training sample."""
    questions = [parse_musique(r) for r in synthetic_world(seed=7, n_questions=60)]
    return split_sample(questions, 30, 30, seed=2026)


@pytest.fixture(scope="session")
def eval_ws(world, offline_models, offline_config) -> Workspace:  # type: ignore[no-untyped-def]
    return Workspace.build(world[0], offline_models, offline_config, lambda _: None)


@pytest.fixture
def word_stats():  # type: ignore[no-untyped-def]
    def build(passages: list[str]) -> PassageStats:
        return PassageStats.build(passages, word_count)

    return build
