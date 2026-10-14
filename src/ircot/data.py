"""MuSiQue and HotpotQA questions, fixed-seed samples, and the union-of-paragraphs corpus.

Each benchmark sample retrieves over one shared corpus built from the paragraphs of all of its
questions (deduplicated), not over the 10-20 paragraphs attached to a single question. That turns
the distractor setting into a small open-retrieval problem where a hop can actually go wrong.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Paragraph:
    title: str
    text: str
    supporting: bool


@dataclass(frozen=True)
class SubQuestion:
    """One step of a MuSiQue decomposition, e.g. ("#1 >> spouse", "Miquette Giraudy")."""

    question: str
    answer: str


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    text: str
    answers: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]
    hops: int
    kind: str
    decomposition: tuple[SubQuestion, ...] = ()

    @property
    def answer(self) -> str:
        return self.answers[0]


def _musique_kind(qid: str) -> tuple[int, str]:
    kind = qid.split("__", 1)[0]
    return int(kind[0]), kind


def parse_musique(record: dict) -> Question:  # type: ignore[type-arg]
    hops, kind = _musique_kind(record["id"])
    return Question(
        qid=record["id"],
        dataset="musique",
        text=record["question"],
        answers=(record["answer"], *record.get("answer_aliases", [])),
        paragraphs=tuple(
            Paragraph(p["title"], p["paragraph_text"], bool(p["is_supporting"]))
            for p in record["paragraphs"]
        ),
        hops=hops,
        kind=kind,
        decomposition=tuple(
            SubQuestion(step["question"], step["answer"])
            for step in record.get("question_decomposition", [])
        ),
    )


def load_musique(path: Path) -> list[Question]:
    with path.open(encoding="utf-8") as fh:
        return [parse_musique(json.loads(line)) for line in fh if line.strip()]


def parse_hotpotqa(record: dict) -> Question:  # type: ignore[type-arg]
    support = set(record["supporting_facts"]["title"])
    titles = record["context"]["title"]
    sentences = record["context"]["sentences"]
    return Question(
        qid=record["id"],
        dataset="hotpotqa",
        text=record["question"],
        answers=(record["answer"],),
        paragraphs=tuple(
            Paragraph(title, "".join(sents).strip(), title in support)
            for title, sents in zip(titles, sentences, strict=True)
        ),
        hops=2,
        kind=record["type"],
    )


def load_hotpotqa(path: Path) -> list[Question]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path, columns=["id", "question", "answer", "type", "supporting_facts", "context"]
    )
    return [parse_hotpotqa(row) for row in table.to_pylist()]


def split_sample(
    questions: Sequence[Question], n_eval: int, n_train: int, seed: int
) -> tuple[list[Question], list[Question]]:
    """Two disjoint fixed-seed samples: the evaluation set and the stop-controller training set.

    Both keep dataset order so traces line up with the source file.
    """
    if n_eval + n_train > len(questions):
        raise ValueError(f"asked for {n_eval}+{n_train} questions, only {len(questions)} exist")
    order = np.random.default_rng(seed).permutation(len(questions))
    eval_idx = sorted(order[:n_eval].tolist())
    train_idx = sorted(order[n_eval : n_eval + n_train].tolist())
    return [questions[i] for i in eval_idx], [questions[i] for i in train_idx]


def ids_digest(questions: Iterable[Question]) -> str:
    return hashlib.sha256("\n".join(q.qid for q in questions).encode("utf-8")).hexdigest()


@dataclass
class Corpus:
    """Deduplicated passages plus, per question, the corpus ids of its supporting passages."""

    titles: list[str]
    texts: list[str]
    gold: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.texts)

    def passage(self, doc: int) -> str:
        """What the encoders and BM25 index: title and paragraph, as the retrievers see them."""
        return f"{self.titles[doc]}. {self.texts[doc]}"

    def digest(self) -> str:
        h = hashlib.sha256()
        for title, text in zip(self.titles, self.texts, strict=True):
            h.update(title.encode("utf-8") + b"\x1f" + text.encode("utf-8") + b"\x1e")
        return h.hexdigest()

    @classmethod
    def from_questions(cls, questions: Sequence[Question]) -> Corpus:
        titles: list[str] = []
        texts: list[str] = []
        index: dict[tuple[str, str], int] = {}
        gold: dict[str, tuple[int, ...]] = {}
        for q in questions:
            support: list[int] = []
            for p in q.paragraphs:
                key = (p.title, p.text)
                doc = index.get(key)
                if doc is None:
                    doc = index[key] = len(texts)
                    titles.append(p.title)
                    texts.append(p.text)
                if p.supporting and doc not in support:
                    support.append(doc)
            gold[q.qid] = tuple(support)
        return cls(titles, texts, gold)
