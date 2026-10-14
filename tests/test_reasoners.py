from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from fakes import ScriptedReader
from ircot.bridge import TitleIndex
from ircot.data import Corpus, Paragraph, Question, SubQuestion
from ircot.llm import OpenAICompatibleProvider, ScriptedProvider
from ircot.policies import POLICIES
from ircot.reasoners import (
    ChainOfThoughtReasoner,
    ExtractiveReasoner,
    OracleBridgeReasoner,
    ReaderCache,
    TitleFeedbackReasoner,
    best_answer,
)

TITLES = ["Green (album)", "Steve Hillage", "Miquette Giraudy", "Grant Green"]
TEXTS = [
    "Green is an album by Steve Hillage, recorded with Grant Green on one track.",
    "Steve Hillage is an English guitarist. His spouse is Miquette Giraudy.",
    "Miquette Giraudy is a French keyboard player.",
    "Grant Green was a jazz guitarist.",
]
CORPUS = Corpus(TITLES, TEXTS, {"q": (0, 1)})
QUESTION = Question(
    "q",
    "musique",
    "Who is the spouse of the Green performer?",
    ("Miquette Giraudy",),
    tuple(Paragraph(t, x, i < 2) for i, (t, x) in enumerate(zip(TITLES, TEXTS, strict=True))),
    2,
    "2hop",
    (
        SubQuestion("Green >> performer", "Steve Hillage"),
        SubQuestion("#1 >> spouse", "Miquette Giraudy"),
    ),
)
READER = ScriptedReader(
    {
        TEXTS[0]: [("Steve Hillage", 2.0), ("Grant Green", 2.5)],
        TEXTS[1]: [("Miquette Giraudy", 5.0)],
        TEXTS[3]: [("jazz guitarist", 1.0)],
    }
)


def extractive(read_per_hop: int = 2) -> tuple[ExtractiveReasoner, ReaderCache]:
    cache = ReaderCache(READER, CORPUS)
    titles = TitleIndex(TITLES)
    return ExtractiveReasoner(
        cache, titles, POLICIES["append"], read_per_hop, title_bonus=1.0
    ), cache


def test_extractive_reasoner_bridges_on_the_best_title_linked_span() -> None:
    reasoner, _ = extractive()
    reasoner.start(QUESTION)
    t = reasoner.step(1, [0])
    assert t.bridge is not None
    assert t.bridge.entity == "Grant Green"  # 2.5 + bonus beats 2.0 + bonus
    assert t.query == f"{QUESTION.text} Grant Green"
    assert t.answer == "Grant Green"
    assert t.passages_read == 1


def test_newest_hop_candidates_come_first_then_earlier_ones() -> None:
    reasoner, _ = extractive()
    reasoner.start(QUESTION)
    reasoner.step(1, [0])
    t2 = reasoner.step(2, [3])  # the only new span is not a title, so it still wins
    assert t2.bridge is not None
    assert t2.bridge.entity == "jazz guitarist"
    t3 = reasoner.step(3, [])  # nothing new: fall back to the unused hop-1 span
    assert t3.bridge is not None
    assert t3.bridge.entity == "Steve Hillage"
    assert reasoner.step(4, []).query is None


def test_reader_cache_reads_each_pair_once_and_reads_at_most_k_per_hop() -> None:
    reasoner, cache = extractive(read_per_hop=1)
    reasoner.start(QUESTION)
    t = reasoner.step(1, [1, 0])
    assert t.passages_read == 1
    assert cache.computed == 1
    answer, margin = best_answer(cache, QUESTION.text, [1, 0, 1])
    assert (answer, margin) == ("Miquette Giraudy", 5.0)
    assert cache.computed == 2


def test_title_feedback_uses_the_best_unused_title() -> None:
    reasoner = TitleFeedbackReasoner(CORPUS)
    reasoner.start(QUESTION)
    assert reasoner.step(1, [2, 3]).query == f"{QUESTION.text} Miquette Giraudy"
    assert reasoner.step(2, [2, 3]).query == f"{QUESTION.text} Grant Green"
    assert reasoner.step(3, [2, 3]).query is None


def test_oracle_bridge_walks_the_gold_decomposition() -> None:
    reasoner = OracleBridgeReasoner()
    reasoner.start(QUESTION)
    assert reasoner.step(1, []).query == f"{QUESTION.text} Steve Hillage"
    assert reasoner.step(2, []).query is None
    with pytest.raises(ValueError, match="decomposition"):
        reasoner.start(Question("h", "hotpotqa", "q?", ("a",), (), 2, "bridge"))


def test_chain_of_thought_reasoner_queries_with_its_newest_sentence() -> None:
    replies = iter(
        [
            "Green was performed by Steve Hillage.",
            "Steve Hillage is married to Miquette Giraudy. So the answer is: Miquette Giraudy.",
        ]
    )
    provider = ScriptedProvider(lambda _: next(replies))
    reasoner = ChainOfThoughtReasoner(provider, CORPUS)
    reasoner.start(QUESTION)
    t1 = reasoner.step(1, [0])
    assert t1.query == "Green was performed by Steve Hillage."
    assert t1.llm_tokens > 0
    assert not t1.says_done
    t2 = reasoner.step(2, [1])
    assert t2.says_done
    assert t2.answer == "Miquette Giraudy"
    prompt = provider.calls[-1][1]["content"]
    assert "Title: Steve Hillage" in prompt
    assert "Reasoning so far: Green was performed by Steve Hillage." in prompt


def test_chain_of_thought_stops_when_it_repeats_itself() -> None:
    reasoner = ChainOfThoughtReasoner(ScriptedProvider(lambda _: "Green is an album."), CORPUS)
    reasoner.start(QUESTION)
    assert reasoner.step(1, [0]).query == "Green is an album."
    assert reasoner.step(2, [1]).query is None


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []  # noqa: RUF012 - shared with the test on purpose

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
        reply = {
            "choices": [
                {"message": {"role": "assistant", "content": " Steve Hillage made Green. "}}
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 5},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


def test_openai_compatible_adapter_round_trip() -> None:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}/v1"
        provider = OpenAICompatibleProvider(base, "local-model", api_key="test-key")
        out = provider.complete([{"role": "user", "content": "hi"}], max_tokens=16, stop=["\n"])
    finally:
        server.shutdown()
        thread.join()
    assert out.text == "Steve Hillage made Green."
    assert (out.prompt_tokens, out.completion_tokens, out.total_tokens) == (42, 5, 47)
    sent = _Handler.seen[-1]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == "Bearer test-key"
    assert sent["temperature"] == 0
    assert sent["model"] == "local-model"
    assert sent["stop"] == ["\n"]
