"""A minimal chat-completion provider interface for the chain-of-thought reasoner.

``OpenAICompatibleProvider`` speaks the ``/chat/completions`` protocol that OpenAI, Ollama,
vLLM and LM Studio all serve, over the standard library's HTTP client. ``ScriptedProvider`` is the
deterministic stand-in used by the tests; no benchmark number in this repository comes from an LLM.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

Message = dict[str, str]


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatProvider(Protocol):
    name: str

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int, stop: Sequence[str] = ()
    ) -> Completion: ...


class OpenAICompatibleProvider:
    """Greedy (temperature 0) completions from any OpenAI-compatible endpoint.

    ``base_url`` is the API root, e.g. ``http://localhost:11434/v1`` for Ollama. The key is read
    from ``IRCOT_LLM_API_KEY`` when not passed; local servers usually accept any value.
    """

    def __init__(
        self, base_url: str, model: str, *, api_key: str | None = None, timeout: float = 120.0
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai-compatible:{model}"
        self.api_key = api_key if api_key is not None else os.environ.get("IRCOT_LLM_API_KEY", "")
        self.timeout = timeout

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int, stop: Sequence[str] = ()
    ) -> Completion:
        body: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if stop:
            body["stop"] = list(stop)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"] or ""
        usage = payload.get("usage") or {}
        return Completion(
            text=text.strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )


class ScriptedProvider:
    """Answers from a function of the messages; token counts are whitespace word counts."""

    name = "scripted"

    def __init__(self, respond: Callable[[Sequence[Message]], str]) -> None:
        self.respond = respond
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int, stop: Sequence[str] = ()
    ) -> Completion:
        self.calls.append(list(messages))
        text = self.respond(messages)
        words = text.split()[:max_tokens]
        prompt = sum(len(m["content"].split()) for m in messages)
        return Completion(" ".join(words), prompt, len(words))
