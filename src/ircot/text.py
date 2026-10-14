"""Text normalisation shared by BM25, title matching, query policies and the answer metrics."""

from __future__ import annotations

import re
import string
import unicodedata

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WORD = re.compile(r"[0-9a-z]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")

# A short list on purpose: multi-hop questions carry their meaning in relation words ("spouse",
# "founded", "born") that longer IR stopword lists sometimes drop.
STOPWORDS = frozenset(
    """a an and are as at be been but by did do does for from had has have he her his how i if in
    into is it its of on or she that the their them they this to was were what when where which
    who whom whose why will with would""".split()
)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, drop punctuation and articles, collapse whitespace.

    The order of operations matches the official MuSiQue/SQuAD evaluators exactly; it matters for
    strings such as "a-b" where punctuation removal creates or removes an article boundary.
    """
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCT)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def fold(text: str) -> str:
    """Lowercase and strip accents so "Rashād" and "Rashad" index to the same term."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def terms(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Index/query terms: folded alphanumeric runs, stopwords removed unless asked to keep them."""
    words = _WORD.findall(fold(text))
    if keep_stopwords:
        return words
    return [w for w in words if w not in STOPWORDS]


def content_terms(text: str) -> set[str]:
    return set(terms(text))


def title_key(text: str) -> str:
    """Key for exact title matching: folded, punctuation-free, article-free."""
    return normalize_answer(fold(text))


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Character ranges of the sentences in ``text`` (a deliberately simple splitter)."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        bounds.append((start, m.start()))
        start = m.end()
    if start < len(text):
        bounds.append((start, len(text)))
    return [(a, b) for a, b in bounds if text[a:b].strip()]


def sentence_at(text: str, offset: int) -> str:
    """The sentence of ``text`` that contains character ``offset``."""
    for a, b in sentence_bounds(text):
        if a <= offset < b:
            return text[a:b].strip()
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())
