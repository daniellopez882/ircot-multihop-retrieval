from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ircot.pool import PassagePool, text_fingerprint
from ircot.retrieval import Hit


def make_pool(tokens: list[int], prints: list[str] | None = None, **budget: int) -> PassagePool:
    fps = prints or [str(i) for i in range(len(tokens))]
    return PassagePool(lambda d: tokens[d], lambda d: fps[d], **budget)


def hits(*docs: int) -> list[Hit]:
    return [Hit(d, 1.0 / (i + 1)) for i, d in enumerate(docs)]


def test_duplicates_by_id_and_by_text_fingerprint_are_skipped() -> None:
    pool = make_pool([10] * 5, prints=["a", "b", "a", "c", "d"])
    first = pool.admit(1, hits(0, 1))
    second = pool.admit(2, hits(1, 2, 3))
    assert first.admitted == (0, 1)
    assert second.admitted == (3,)
    assert second.duplicates == (1, 2)
    assert pool.docs == [0, 1, 3]


def test_fingerprint_ignores_case_punctuation_and_spacing() -> None:
    assert text_fingerprint("Green is an album.") == text_fingerprint("green  is an ALBUM")
    assert text_fingerprint("Green is an album.") != text_fingerprint("Green is a song.")


def test_token_budget_rejects_what_does_not_fit_but_keeps_going() -> None:
    pool = make_pool([60, 50, 30, 10], max_tokens=100)
    adm = pool.admit(1, hits(0, 1, 2, 3))
    assert adm.admitted == (0, 2, 3)
    assert adm.over_budget == (1,)
    assert pool.tokens_used == 100


def test_passage_budget() -> None:
    pool = make_pool([1] * 6, max_passages=4)
    pool.admit(1, hits(0, 1, 2))
    adm = pool.admit(2, hits(3, 4, 5))
    assert adm.admitted == (3,)
    assert adm.over_budget == (4, 5)
    assert len(pool) == 4


def test_ranking_is_round_robin_over_hops() -> None:
    pool = make_pool([1] * 9)
    pool.admit(1, hits(0, 1, 2))
    pool.admit(2, hits(3, 4))
    pool.admit(3, hits(5, 6, 7, 8))
    assert pool.ranking() == [0, 3, 5, 1, 4, 6, 2, 7, 8]


@given(
    st.lists(st.integers(1, 80), min_size=12, max_size=12),
    st.lists(st.lists(st.integers(0, 11), max_size=6), max_size=5),
    st.integers(1, 400),
)
def test_budget_holds_and_ranking_is_a_permutation(
    tokens: list[int], hops: list[list[int]], budget: int
) -> None:
    pool = make_pool(tokens, max_tokens=budget)
    for h, docs in enumerate(hops, start=1):
        pool.admit(h, hits(*docs))
        assert pool.tokens_used <= budget
    assert sorted(pool.ranking()) == sorted(pool.docs)
    assert len(set(pool.docs)) == len(pool.docs)
    assert pool.tokens_used == sum(tokens[d] for d in pool.docs)
