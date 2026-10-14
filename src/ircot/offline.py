"""Offline stand-ins: a hashing encoder, a lexical span reader and a synthetic multi-hop world.

These let the whole pipeline (fetch excluded) run with no model and no download: the default
test suite, CI and the container smoke run all use them. They are deliberately simple and their
numbers are never reported as model results; the ``offline`` profile labels its output as such.

The synthetic world is a small fictional encyclopedia (films, studios, people, cities,
countries) with 2-, 3- and 4-hop questions in MuSiQue's JSON format, so the real loader parses it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ircot.encoders import l2_normalise
from ircot.reader import Span
from ircot.text import STOPWORDS, content_terms, sentence_bounds, terms


class HashingEncoder:
    """Bag-of-terms embedding: each term maps to a fixed pseudo-random vector (sha256-seeded)."""

    def __init__(self, dim: int = 64, name: str = "hashing-64") -> None:
        self.dim = dim
        self.name = name
        self._vectors: dict[str, np.ndarray] = {}

    def _vector(self, term: str) -> np.ndarray:
        vec = self._vectors.get(term)
        if vec is None:
            seed = int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "little")
            vec = self._vectors[term] = np.random.default_rng(seed).standard_normal(self.dim)
        return vec

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, text in enumerate(texts):
            for term in terms(text):
                out[i] += self._vector(term)
        return l2_normalise(out)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)


def _capitalised_runs(text: str) -> list[tuple[int, int]]:
    """Character ranges of maximal runs of capitalised words or numbers ("Kestrel Point")."""
    runs: list[tuple[int, int]] = []
    start = end = -1
    pos = 0
    for word in text.split(" "):
        stripped = word.strip(".,;:!?\"'()")
        lead = word.find(stripped) if stripped else 0
        w_start, w_end = pos + lead, pos + lead + len(stripped)
        is_name = bool(stripped) and (stripped[0].isupper() or stripped.isdigit())
        if is_name and stripped.lower() not in STOPWORDS:
            if start < 0:
                start = w_start
            end = w_end
            if word != stripped and not word.endswith("'s"):
                runs.append((start, end))
                start = -1
        elif start >= 0:
            runs.append((start, end))
            start = -1
        pos += len(word) + 1
    if start >= 0:
        runs.append((start, end))
    return runs


def _stems(text: str) -> set[str]:
    # Five-letter prefixes are a crude stemmer: "founder" and "founded" both become "found".
    return {t[:5] for t in content_terms(text)}


class LexicalReader:
    """Scores capitalised spans by how many question stems share their sentence.

    The no-answer score is a constant 1.0, so a span needs at least two overlapping question
    stems to have a positive margin; spans repeating question words are skipped.
    """

    name = "lexical"

    def read(self, question: str, passages: Sequence[str], top_n: int = 3) -> list[list[Span]]:
        q_terms = content_terms(question)
        q_stems = _stems(question)
        out: list[list[Span]] = []
        for passage in passages:
            spans: list[Span] = []
            for a, b in sentence_bounds(passage):
                overlap = len(q_stems & _stems(passage[a:b]))
                for lo, hi in _capitalised_runs(passage[a:b]):
                    text = passage[a + lo : a + hi]
                    if content_terms(text) <= q_terms:
                        continue
                    # Later spans in a sentence lose ties, which favours objects over subjects.
                    spans.append(Span(text, a + lo, a + hi, overlap - 0.001 * lo, 1.0))
            spans.sort(key=lambda s: (-s.score, s.start))
            unique = list({s.text: s for s in reversed(spans)}.values())[::-1]
            out.append(unique[:top_n])
        return out


_FIRST = (
    "Adela Brin Corvin Dalia Edmund Fenna Gideon Hester Ivo Juno Kasimir Lorna Matthias Nerys "
    "Osric Petra Quentin Rosalind Silas Thora Ulric Vesna Wendel Yara"
).split()
_LAST = (
    "Ashgrove Blackwood Carrow Dunmore Ellery Fairweather Gallard Harrowby Ingram Jessop Kingsley "
    "Larkspur Merriweather Northcote Oakhurst Pemberton Quill Ravensworth Stroud Thorne"
).split()
_COUNTRIES = ("Vandoria", "Ostrelia", "Kelmark", "Surova", "Tarnland", "Elbonia")
_CITY_A = "Kestrel Marlow Hollin Brask Cindral Dorrow Fenwick Galway Harlan".split()
_CITY_B = ("Point", "Bay", "Ford", "Haven", "Crossing", "Falls")
_STUDIO_A = (
    "Bluefin Copperleaf Driftwood Emberline Foxglove Granite Halcyon Ironbark Juniper Kingfisher"
).split()
_FILM_A = "Silent Glass Hollow Crimson Distant Forgotten Iron Painted Velvet Winter".split()
_FILM_B = "Orchard Harbor Meridian Lantern Archive Garden Tide Signal Compass Quarry".split()


def synthetic_world(
    seed: int = 7, n_questions: int = 30, paragraphs_per_question: int = 10
) -> list[dict]:  # type: ignore[type-arg]
    """MuSiQue-format records over one fictional encyclopedia, fully determined by ``seed``."""
    rng = np.random.default_rng(seed)
    countries = list(_COUNTRIES)
    cities = [f"{a} {b}" for a in _CITY_A for b in _CITY_B]
    rng.shuffle(cities)
    cities = cities[:18]
    people = [f"{f} {last}" for f in _FIRST for last in _LAST]
    rng.shuffle(people)
    people = people[:44]
    studios = [f"{a} Pictures" for a in _STUDIO_A]
    films = [f"The {a} {b}" for a in _FILM_A for b in _FILM_B]
    rng.shuffle(films)
    films = films[:n_questions]

    country_of = {c: countries[i % len(countries)] for i, c in enumerate(cities)}
    born_in = {p: cities[int(rng.integers(len(cities)))] for p in people}
    spouse_of = {people[i]: people[i + 1] for i in range(0, len(people) - 1, 2)}
    spouse_of |= {b: a for a, b in list(spouse_of.items())}
    founder_of = {s: people[int(rng.integers(len(people)))] for s in studios}
    hq_of = {s: cities[int(rng.integers(len(cities)))] for s in studios}
    director_of = {f: people[int(rng.integers(len(people)))] for f in films}
    studio_of = {f: studios[int(rng.integers(len(studios)))] for f in films}

    paragraph: dict[str, str] = {}
    for c in countries:
        capital = next(city for city in cities if country_of[city] == c)
        paragraph[c] = (
            f"{c} is a country whose capital is {capital}. It is known for its mountain railways."
        )
    for city in cities:
        paragraph[city] = (
            f"{city} is a city in the country of {country_of[city]}. "
            f"It has a population of about {int(rng.integers(20, 900))} thousand people."
        )
    for p in people:
        paragraph[p] = (
            f"{p} is a writer and filmmaker. {p} was born in {born_in[p]}. "
            f"The spouse of {p} is {spouse_of.get(p, 'not recorded')}."
        )
    for s in studios:
        paragraph[s] = (
            f"{s} is a film studio. {s} was founded by {founder_of[s]} in "
            f"{int(rng.integers(1920, 1990))}. Its offices are in {hq_of[s]}."
        )
    for f in films:
        paragraph[f] = (
            f"{f} is a {int(rng.integers(1950, 2020))} drama film. The director of {f} was "
            f"{director_of[f]}. It was produced by the studio {studio_of[f]}."
        )

    records = []
    titles = list(paragraph)
    for i, film in enumerate(films):
        director, studio = director_of[film], studio_of[film]
        founder = founder_of[studio]
        kind = i % 5
        if kind == 0:
            text = f"Who is the spouse of the director of {film}?"
            chain = [(f"{film} >> director", director), ("#1 >> spouse", spouse_of[director])]
            support = [film, director]
        elif kind == 1:
            text = f"Who founded the studio that produced {film}?"
            chain = [(f"{film} >> studio", studio), ("#1 >> founded by", founder)]
            support = [film, studio]
        elif kind == 2:
            city = born_in[director]
            text = f"In which country was the director of {film} born?"
            chain = [
                (f"{film} >> director", director),
                ("#1 >> place of birth", city),
                ("#2 >> country", country_of[city]),
            ]
            support = [film, director, city]
        elif kind == 3:
            text = f"Who is the spouse of the founder of the studio that produced {film}?"
            chain = [
                (f"{film} >> studio", studio),
                ("#1 >> founded by", founder),
                ("#2 >> spouse", spouse_of[founder]),
            ]
            support = [film, studio, founder]
        else:
            city = born_in[founder]
            text = f"In which country was the founder of the studio that produced {film} born?"
            chain = [
                (f"{film} >> studio", studio),
                ("#1 >> founded by", founder),
                ("#2 >> place of birth", city),
                ("#3 >> country", country_of[city]),
            ]
            support = [film, studio, founder, city]
        support = list(dict.fromkeys(support))
        others = [t for t in titles if t not in support]
        picks = rng.choice(len(others), size=paragraphs_per_question - len(support), replace=False)
        chosen = support + [others[j] for j in sorted(picks.tolist())]
        order = rng.permutation(len(chosen)).tolist()
        paragraphs = [
            {
                "idx": pos,
                "title": chosen[j],
                "paragraph_text": paragraph[chosen[j]],
                "is_supporting": chosen[j] in support,
            }
            for pos, j in enumerate(order)
        ]
        hops = len(chain)
        records.append(
            {
                "id": f"{hops}hop__synthetic{seed}_{i}",
                "paragraphs": paragraphs,
                "question": text,
                "question_decomposition": [
                    {"id": k, "question": q, "answer": a, "paragraph_support_idx": None}
                    for k, (q, a) in enumerate(chain)
                ],
                "answer": chain[-1][1],
                "answer_aliases": [],
                "answerable": True,
            }
        )
    return records
