from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ircot.data import Corpus, ids_digest, load_hotpotqa, load_musique, split_sample
from ircot.offline import synthetic_world

HOTPOT_ROW = {
    "id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "type": "comparison",
    "level": "hard",
    "supporting_facts": {"title": ["Scott Derrickson", "Ed Wood"], "sent_id": [0, 0]},
    "context": {
        "title": ["Ed Wood (film)", "Scott Derrickson", "Ed Wood"],
        "sentences": [
            ["Ed Wood is a 1994 film.", " It stars Johnny Depp."],
            ["Scott Derrickson is an American director."],
            ["Ed Wood was an American filmmaker."],
        ],
    },
}


def test_musique_loader_reads_the_official_format(tmp_path: Path) -> None:
    records = synthetic_world(seed=7, n_questions=10)
    path = tmp_path / "musique.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    questions = load_musique(path)
    assert len(questions) == 10
    q = questions[0]
    assert q.dataset == "musique"
    assert q.hops == len(q.decomposition) == 2
    assert q.kind == "2hop"
    assert q.answer == q.decomposition[-1].answer
    assert sum(p.supporting for p in q.paragraphs) == 2
    assert len(q.paragraphs) == 10


def test_hotpotqa_loader_reads_the_hugging_face_parquet(tmp_path: Path) -> None:
    path = tmp_path / "hotpot.parquet"
    pq.write_table(pa.Table.from_pylist([HOTPOT_ROW]), path)
    (q,) = load_hotpotqa(path)
    assert q.kind == "comparison"
    assert q.hops == 2
    assert q.answers == ("yes",)
    assert [p.title for p in q.paragraphs if p.supporting] == ["Scott Derrickson", "Ed Wood"]
    assert q.paragraphs[0].text == "Ed Wood is a 1994 film. It stars Johnny Depp."


def test_samples_are_disjoint_deterministic_and_in_source_order() -> None:
    from ircot.data import parse_musique

    questions = [parse_musique(r) for r in synthetic_world(seed=7, n_questions=60)]
    ev, tr = split_sample(questions, 20, 15, seed=1)
    ev2, tr2 = split_sample(questions, 20, 15, seed=1)
    assert ids_digest(ev) == ids_digest(ev2)
    assert ids_digest(tr) == ids_digest(tr2)
    assert not {q.qid for q in ev} & {q.qid for q in tr}
    position = {q.qid: i for i, q in enumerate(questions)}
    assert [position[q.qid] for q in ev] == sorted(position[q.qid] for q in ev)
    assert ids_digest(split_sample(questions, 20, 15, seed=2)[0]) != ids_digest(ev)
    with pytest.raises(ValueError, match="only 60"):
        split_sample(questions, 50, 15, seed=1)


def test_corpus_is_the_deduplicated_union_with_gold_ids(world) -> None:  # type: ignore[no-untyped-def]
    questions = world[0]
    corpus = Corpus.from_questions(questions)
    total = sum(len(q.paragraphs) for q in questions)
    assert len(corpus) < total  # shared paragraphs collapse
    assert len(set(zip(corpus.titles, corpus.texts, strict=True))) == len(corpus)
    for q in questions:
        gold = corpus.gold[q.qid]
        assert len(gold) == sum(p.supporting for p in q.paragraphs)
        supporting = {(p.title, p.text) for p in q.paragraphs if p.supporting}
        assert {(corpus.titles[d], corpus.texts[d]) for d in gold} == supporting
    assert corpus.digest() == Corpus.from_questions(questions).digest()
    assert corpus.passage(0).startswith(corpus.titles[0] + ". ")
