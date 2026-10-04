"""CLI tests: the real `docent ask` command, with FakeLLM standing in for Ollama."""

from typer.testing import CliRunner

from docent import cli
from docent.llm import FakeLLM


def test_ask_streams_answer_to_stdout_and_stats_to_stderr(monkeypatch):
    fake = FakeLLM(replies=["A Pod is the smallest deployable unit."])
    # monkeypatch swaps make_llm for this test only, then puts the original back.
    monkeypatch.setattr(cli, "make_llm", lambda: fake)

    result = CliRunner().invoke(cli.app, ["ask", "What is a Pod?", "--seed", "1"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "A Pod is the smallest deployable unit."
    assert "4 prompt + 7 completion tokens" in result.stderr  # FakeLLM counts words
    assert "first token" in result.stderr


def test_ask_sends_the_question_model_and_temperature(monkeypatch):
    fake = FakeLLM(replies=["ok"])
    monkeypatch.setattr(cli, "make_llm", lambda: fake)

    CliRunner().invoke(cli.app, ["ask", "hi", "--model", "docent-qwen3-8b", "--temperature", "0.7"])

    call = fake.calls[0]
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["model"] == "docent-qwen3-8b"
    assert call["temperature"] == 0.7
