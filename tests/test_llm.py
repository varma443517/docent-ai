"""Tests for docent.llm. None of them needs Ollama or the internet.

The OpenAICompatibleLLM tests use httpx.MockTransport: the real openai SDK builds the
real HTTP request, and a tiny function plays the server. So we test our code *and*
our assumptions about the wire format, with no model installed.
"""

import json

import httpx
import pytest
from openai import OpenAI

from docent.llm import Chunk, FakeLLM, OpenAICompatibleLLM, Usage

HELLO = [{"role": "user", "content": "say hello"}]


# ---- FakeLLM -----------------------------------------------------------------------


def test_fake_returns_replies_in_order_and_records_calls():
    llm = FakeLLM(replies=["first", "second"])
    assert llm.chat(HELLO, model="m").text == "first"
    assert llm.chat(HELLO, model="m", temperature=0.7).text == "second"
    assert [c["temperature"] for c in llm.calls] == [0.0, 0.7]


def test_fake_fails_loudly_when_out_of_replies():
    with pytest.raises(RuntimeError, match="ran out"):
        FakeLLM(replies=[]).chat(HELLO, model="m")


def test_fake_stream_rebuilds_the_exact_text_then_reports_usage():
    chunks = list(FakeLLM(replies=["hello there world"]).stream(HELLO, model="m"))
    assert "".join(c.text for c in chunks) == "hello there world"
    assert chunks[-1].usage == Usage(prompt_tokens=2, completion_tokens=3)


# ---- OpenAICompatibleLLM against a pretend server ------------------------------------


def llm_with_server(handler) -> OpenAICompatibleLLM:
    """Build the real backend, but route its HTTP traffic to `handler`."""
    http = httpx.Client(transport=httpx.MockTransport(handler))
    sdk = OpenAI(base_url="http://fake/v1", api_key="x", http_client=http)
    return OpenAICompatibleLLM(client=sdk)


def test_chat_sends_model_and_seed_and_parses_reply():
    seen = {}

    def server(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "created": 0,
                "model": "docent-qwen3-8b",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hello!"},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            },
        )

    reply = llm_with_server(server).chat(HELLO, model="docent-qwen3-8b", seed=42)
    assert reply.text == "Hello!"
    assert reply.usage == Usage(11, 3)
    assert seen["model"] == "docent-qwen3-8b" and seen["seed"] == 42


def test_stream_yields_text_then_a_final_usage_chunk():
    def event(delta: dict, usage: dict | None = None) -> str:
        body = {
            "id": "1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": delta}] if delta else [],
            "usage": usage,
        }
        return f"data: {json.dumps(body)}\n\n"

    sse = (
        event({"role": "assistant", "content": "Hel"})
        + event({"content": "lo"})
        + event({}, usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7})
        + "data: [DONE]\n\n"
    )

    def server(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream_options"] == {"include_usage": True}
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    chunks = list(llm_with_server(server).stream(HELLO, model="m"))
    assert chunks == [Chunk(text="Hel"), Chunk(text="lo"), Chunk(usage=Usage(5, 2))]
