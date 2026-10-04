"""How many tokens does the same text cost in each model?

Run:  make bench   (or: uv run python bench/tokens.py)
Out:  a table on screen + bench/results/tokens.csv

How it counts
-------------
There is no "tokenize" call in Ollama's API, so we use a trick: send the text to the
native /api/generate endpoint and ask for just ONE output token. The reply's
`prompt_eval_count` says how many tokens the model read. Two settings make that number
mean "tokens in this text" and nothing else:

* raw=True       - skip the chat template (the hidden <|user|>-style wrapper, ~16-24 tokens)
* num_predict=1  - generate one token and stop; we only want the reading cost

Even in raw mode the model adds special tokens of its own (e.g. a <begin_of_text> marker).
We measure them once per model with the prompt "a", which is exactly one token in both
vocabularies, and subtract them.

The prompt cache doesn't fool this: prompt_eval_count is the full count even when most of
the prompt came from the cache (that's prompt_eval_cached_count, a separate field).
"""

import csv
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434"
HERE = Path(__file__).parent
TEXTS = HERE / "texts"
OUT = HERE / "results" / "tokens.csv"
# Every pinned alias built by `make models`, so new Modelfiles are picked up automatically.
MODELS = sorted(p.stem for p in (HERE.parent / "models").glob("*.Modelfile"))


def tokens_read(model: str, text: str) -> int:
    """Raw token count for `text`, special tokens included."""
    r = httpx.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": text,
            "raw": True,
            "stream": False,
            "think": False,  # a thinking model would otherwise "think" before its 1 token
            "options": {"num_predict": 1},
        },
        timeout=120,  # the first call also loads the model into memory
    )
    r.raise_for_status()
    return r.json()["prompt_eval_count"]


def main() -> None:
    texts = {p.name: p.read_text(encoding="utf-8").strip() for p in sorted(TEXTS.iterdir())}
    rows = []
    for model in MODELS:
        special = tokens_read(model, "a") - 1  # whatever the model adds on its own
        for name, text in texts.items():
            tokens = tokens_read(model, text) - special
            rows.append(
                {
                    "model": model,
                    "text": name,
                    "chars": len(text),
                    "tokens": tokens,
                    # Bigger = the tokenizer packs this kind of text more efficiently.
                    "chars_per_token": round(len(text) / tokens, 2),
                }
            )

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # A Markdown table (text down the side, one column per model), ready for the README.
    print("| text | chars | " + " | ".join(f"{m} tokens (chars/token)" for m in MODELS) + " |")
    print("|---|---:|" + "---:|" * len(MODELS))
    for name, text in texts.items():
        cells = [
            f"{r['tokens']} ({r['chars_per_token']})"
            for m in MODELS
            for r in rows
            if r["model"] == m and r["text"] == name
        ]
        print(f"| {name} | {len(text)} | " + " | ".join(cells) + " |")
    print(f"\nwrote {OUT.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
