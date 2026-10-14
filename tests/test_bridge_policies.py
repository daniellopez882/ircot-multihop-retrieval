from __future__ import annotations

from ircot.bridge import Bridge, Candidate, TitleIndex, select_bridge
from ircot.policies import POLICIES
from ircot.reader import Span

TITLES = [
    "Steve Hillage",
    "Green (Steve Hillage album)",
    "Miquette Giraudy",
    "Green",
    "The Beatles",
]
QUESTION = "Who is the spouse of the Green performer?"
PASSAGE = (
    "Green is the fourth studio album by English progressive rock musician Steve Hillage. "
    "It was released in spring 1978."
)


def span(text: str, margin: float, passage: str = PASSAGE) -> Span:
    start = passage.find(text)
    return Span(text, start, start + len(text), margin, 0.0)


def cand(text: str, margin: float, title: str = "Green (Steve Hillage album)") -> Candidate:
    return Candidate(span(text, margin), 1, title, PASSAGE)


def test_title_index_matches_full_titles_and_bare_titles() -> None:
    index = TitleIndex(TITLES)
    assert index.match("steve hillage") == "Steve Hillage"
    assert index.match("Beatles") == "The Beatles"
    assert index.match("Green") == "Green"
    assert index.match("Steve") is None
    assert index.match("...") is None


def test_disambiguated_titles_match_on_their_base_name() -> None:
    index = TitleIndex(["Titanic (1997 film)"])
    assert index.match("Titanic") == "Titanic (1997 film)"


def test_a_title_linked_span_beats_a_higher_scoring_plain_span() -> None:
    index = TitleIndex(TITLES)
    bridge = select_bridge(
        QUESTION, [cand("spring 1978", 3.0), cand("Steve Hillage", 0.5)], index, set(), 4.0
    )
    assert bridge is not None
    assert bridge.entity == "Steve Hillage"
    assert bridge.title_match
    assert bridge.sentence.startswith("Green is the fourth studio album")


def test_spans_repeating_the_question_or_already_used_are_skipped() -> None:
    index = TitleIndex(TITLES)
    assert select_bridge(QUESTION, [cand("Green", 9.0)], index, set(), 4.0) is None
    assert (
        select_bridge(QUESTION, [cand("Steve Hillage", 1.0)], index, {"steve hillage"}, 4.0) is None
    )


def test_a_span_naming_its_own_passage_gets_no_title_bonus() -> None:
    index = TitleIndex(TITLES)
    own = Candidate(span("Steve Hillage", 1.0), 0, "Steve Hillage", PASSAGE)
    other = cand("spring 1978", 2.0)
    bridge = select_bridge(QUESTION, [own, other], index, set(), 4.0)
    assert bridge is not None
    assert bridge.entity == "spring 1978"


def _bridge() -> Bridge:
    sentence = (
        "Green is the fourth studio album by English progressive rock musician Steve Hillage."
    )
    return Bridge("Steve Hillage", "Steve Hillage", 1, sentence, PASSAGE, 1.0, True)


def test_append_policy() -> None:
    assert POLICIES["append"].formulate(QUESTION, _bridge()) == f"{QUESTION} Steve Hillage"


def test_residual_policy_keeps_only_uncovered_question_words() -> None:
    query = POLICIES["residual"].formulate(QUESTION, _bridge())
    assert query == "Steve Hillage spouse performer"


def test_residual_policy_falls_back_to_the_question() -> None:
    covered = Bridge("X", "X", 1, "s", "spouse Green performer", 1.0, False)
    assert POLICIES["residual"].formulate(QUESTION, covered) == f"X {QUESTION}"


def test_sentence_policy_uses_the_evidence_sentence() -> None:
    query = POLICIES["sentence"].formulate(QUESTION, _bridge())
    assert query == _bridge().sentence
    linked = Bridge("Hillage", "Steve Hillage", 1, "It was by Hillage.", PASSAGE, 1.0, True)
    assert POLICIES["sentence"].formulate(QUESTION, linked) == "Steve Hillage. It was by Hillage."
