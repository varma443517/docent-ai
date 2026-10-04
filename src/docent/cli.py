"""Command-line entry point (`docent ...`).

    docent version
    docent ask "What is a Pod?" --model docent-qwen3-8b --temperature 0.7 --seed 1

Design rule: the ANSWER goes to stdout, the STATS go to stderr. So
`docent ask "..." > answer.txt` saves only the answer, and you still see the stats.
"""

import time

import typer

from docent import __version__
from docent.llm import LLMClient, Message, OpenAICompatibleLLM

app = typer.Typer(help="Docent: ask questions about the Kubernetes docs, locally.")

# Always a pinned alias from models/*.Modelfile, never a raw tag, so num_ctx is known.
# The small model is the default because it answers fastest.
DEFAULT_MODEL = "docent-llama3.2-3b"


def make_llm() -> LLMClient:
    """The one place that decides which backend `ask` talks to.

    Tests replace this function with one that returns a FakeLLM (see tests/test_cli.py),
    so the CLI is tested end to end without Ollama.
    """
    return OpenAICompatibleLLM()


@app.callback()
def main() -> None:
    """A callback makes `docent` a command *group*.

    Without it, Typer treats a one-command app as that command, and `docent version`
    fails with "unexpected extra argument". With it, subcommands work from day one.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question, in quotes."),
    model: str = typer.Option(DEFAULT_MODEL, help="A pinned alias built by `make models`."),
    temperature: float = typer.Option(0.0, help="0 = most likely word; higher = more varied."),
    seed: int | None = typer.Option(None, help="Fix the random sampling so runs can repeat."),
) -> None:
    """Ask a question and stream the answer as it is generated."""
    messages: list[Message] = [{"role": "user", "content": question}]

    # perf_counter is a stopwatch: precise, and unaffected by the wall clock changing.
    start = time.perf_counter()
    first_token_at: float | None = None
    usage = None

    for chunk in make_llm().stream(messages, model=model, temperature=temperature, seed=seed):
        if chunk.text:
            if first_token_at is None:
                first_token_at = time.perf_counter()  # the moment the user sees something
            # nl=False: no newline after each piece, so the pieces join into one answer.
            typer.echo(chunk.text, nl=False)
        if chunk.usage:
            usage = chunk.usage  # arrives last, once generation has stopped

    total = time.perf_counter() - start
    typer.echo()  # finish the answer's line

    ttft = f"{first_token_at - start:.2f}s" if first_token_at else "n/a"
    tokens = (
        f"{usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens"
        if usage
        else "token counts not reported"
    )
    # err=True sends this to stderr, keeping stdout clean for the answer alone.
    typer.echo(f"[{model} · {tokens} · first token {ttft} · total {total:.2f}s]", err=True)
