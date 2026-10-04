"""One small interface for talking to language models.

Why this file exists
--------------------
Every later lab (retrieval, evals, agents, MCP) needs to call a model. If each of them
called Ollama directly, we could never swap the server (mlx_lm.server in lab 10) or run
tests in CI, where no model is installed. So the rest of Docent only ever sees
`LLMClient`, and this file holds the two things that fulfil it:

* `OpenAICompatibleLLM` - the real thing. Talks to any server that speaks the OpenAI
  chat-completions API. Ollama serves one at http://localhost:11434/v1.
* `FakeLLM` - a stand-in for tests. Returns scripted replies, instantly, every time.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict

from openai import OpenAI

# Where Ollama's OpenAI-compatible API lives. An environment variable overrides it, so
# pointing Docent at another server never needs a code change.
DEFAULT_BASE_URL = "http://localhost:11434/v1"


# ---- the data that flows in and out ------------------------------------------------


class Message(TypedDict):
    """One turn of a conversation, in the shape every chat API uses.

    A TypedDict is just a plain dict at runtime (so it goes straight into the openai
    SDK), but the type checker knows which keys it must have.
    """

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    """How many tokens the call cost: what we sent in (prompt) and what came back."""

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Reply:
    """The full answer from `chat()`."""

    text: str
    model: str
    usage: Usage | None  # None if the server didn't report usage


@dataclass(frozen=True)
class Chunk:
    """One piece of a streamed answer.

    While the answer is being generated, chunks carry `text`. The very last chunk
    carries `usage` instead (token counts are only known once generation stops).
    """

    text: str = ""
    usage: Usage | None = None


# ---- the interface -----------------------------------------------------------------


class LLMClient(Protocol):
    """Anything with these two methods counts as an LLM client.

    `Protocol` means "structural typing": FakeLLM doesn't inherit from this class, it
    simply has the same methods, and the type checker accepts it. That keeps test doubles
    free of any dependency on the real backend.
    """

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Reply: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Iterator[Chunk]: ...


# ---- the real backend --------------------------------------------------------------


class OpenAICompatibleLLM:
    """Talks to any OpenAI-compatible server: Ollama now, mlx_lm.server later.

    Limitation worth knowing: the /v1 API has no way to set Ollama's context window
    (num_ctx). That's why we pin it in models/*.Modelfile instead (next step).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str = "ollama",  # Ollama ignores the key, but the SDK insists on one
        client: OpenAI | None = None,  # tests pass a client with a fake network
    ) -> None:
        base_url = base_url or os.environ.get("DOCENT_BASE_URL", DEFAULT_BASE_URL)
        self._client = client or OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Reply:
        resp = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            seed=seed,  # same seed + same temperature + same model = repeatable sampling
        )
        usage = None
        if resp.usage is not None:
            usage = Usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        # `content` can be None (e.g. a pure tool call), so fall back to "".
        return Reply(text=resp.choices[0].message.content or "", model=resp.model, usage=usage)

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Iterator[Chunk]:
        # With stream=True the server sends Server-Sent Events: many small JSON
        # messages, each holding a few characters ("deltas"). include_usage asks for
        # one extra message at the end that has the token counts and no text.
        events = self._client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            seed=seed,
            stream=True,
            stream_options={"include_usage": True},
        )
        for event in events:
            if event.choices and event.choices[0].delta.content:
                yield Chunk(text=event.choices[0].delta.content)
            if event.usage is not None:
                yield Chunk(usage=Usage(event.usage.prompt_tokens, event.usage.completion_tokens))


# ---- the test double ---------------------------------------------------------------


@dataclass
class FakeLLM:
    """Returns scripted replies in order. Used by unit tests and CI.

    It also records every call in `calls`, so a test can check *what we sent*, e.g.
    "did the RAG prompt include the retrieved passage?" (lab 3 will rely on this).
    """

    replies: list[str]
    calls: list[dict] = field(default_factory=list)

    def _next(self, messages: Sequence[Message], model: str, temperature: float) -> str:
        self.calls.append({"messages": list(messages), "model": model, "temperature": temperature})
        if not self.replies:
            # Failing loudly beats returning "" - a silent empty answer hides test bugs.
            raise RuntimeError("FakeLLM ran out of scripted replies")
        return self.replies.pop(0)

    @staticmethod
    def _usage(messages: Sequence[Message], text: str) -> Usage:
        # Fake "tokens" = words. Wrong in absolute terms, but stable and easy to assert.
        return Usage(sum(len(m["content"].split()) for m in messages), len(text.split()))

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Reply:
        text = self._next(messages, model, temperature)
        return Reply(text=text, model=model, usage=self._usage(messages, text))

    def stream(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Iterator[Chunk]:
        text = self._next(messages, model, temperature)
        # One chunk per word, keeping the spaces, so "".join(chunks) == text exactly.
        words = text.split(" ")
        for i, word in enumerate(words):
            yield Chunk(text=word if i == len(words) - 1 else word + " ")
        yield Chunk(usage=self._usage(messages, text))
