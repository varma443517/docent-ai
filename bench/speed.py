"""An honest prefill / decode speed benchmark for the pinned local models.

Run:  make bench   (or: uv run --extra bench python bench/speed.py)
Out:  a summary table on screen, bench/results/speed.csv and docs/img/speed.png

Two phases, two speeds
----------------------
* PREFILL - the model reads the whole prompt at once (in parallel). Its speed is
  prompt tokens / prefill time, and it decides the time to first token (TTFT).
* DECODE  - the model writes the answer one token at a time. Its speed is
  output tokens / decode time, and it decides how fast the words scroll.

What would make the numbers lie, and what we do about it
--------------------------------------------------------
1. Model loading: the first call loads weights into memory -> one warm-up run is discarded.
2. The prompt cache: an identical prompt prefix is reused, so prefill is skipped ->
   every prompt STARTS with a unique random run id, and the script fails if a measured
   run read more than the fixed chat-template header from the cache.
3. Hidden settings: Ollama picks num_ctx and thinking for you -> we pass both explicitly.
4. Noise: 5 measured runs per setting, and we report the median.
One deliberately cached run is kept in the CSV (cached=True) to show what the cache hides.
"""

import csv
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx
from tokens import MODELS, tokens_read  # sibling script: same models, same counting trick

OLLAMA = "http://localhost:11434"
HERE = Path(__file__).parent
CSV_OUT = HERE / "results" / "speed.csv"
PNG_OUT = HERE.parent / "docs" / "img" / "speed.png"

NUM_CTX = 8192  # same as the Modelfiles; passed explicitly so nothing is left to defaults
SIZES = [100, 1000, 4000]  # target prompt sizes, in THIS model's tokens
RUNS = 5  # measured runs per (model, size), after one warm-up
MAX_NEW = 128  # cap the answer length so decode is timed over similar spans
QUESTION = "\n\nSummarise the text above in three short bullet points."

# Filler that looks like what Docent will really read: docs prose plus a manifest.
TEXTS = HERE / "texts"
FILLER = (TEXTS / "prose.txt").read_text() + "\n" + (TEXTS / "manifest.yaml").read_text()


def build_prompt(model: str, target: int) -> str:
    """Repeat the filler until the prompt is ~`target` tokens in this model's tokenizer."""
    unit = tokens_read(model, FILLER)
    if target <= unit:  # small prompts: cut the filler by characters, proportionally
        return FILLER[: int(len(FILLER) * target / unit)] + QUESTION
    return FILLER * round(target / unit) + QUESTION


def generate(model: str, prompt: str) -> dict:
    """One streamed call. Returns Ollama's timing fields plus our own TTFT stopwatch."""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "think": False,  # qwen3 would otherwise think first; llama ignores it
        "options": {"num_ctx": NUM_CTX, "num_predict": MAX_NEW, "temperature": 0, "seed": 1},
    }
    start = time.perf_counter()
    ttft = None
    with httpx.stream("POST", f"{OLLAMA}/api/generate", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():  # one JSON object per line
            msg = json.loads(line)
            if ttft is None and msg.get("response"):
                ttft = time.perf_counter() - start
            if msg.get("done"):
                msg["ttft_s"] = ttft
                return msg
    raise RuntimeError("stream ended without a final 'done' message")


def row(model: str, size: int, run: int, r: dict, cached_run: bool) -> dict:
    """Turn Ollama's raw timings (nanoseconds) into the numbers we care about."""
    cached = r.get("prompt_eval_cached_count", 0)
    prefilled = r["prompt_eval_count"] - cached  # tokens actually computed, not reused
    return {
        "model": model,
        "target_tokens": size,
        "run": run,
        "cached": cached_run,
        "prompt_tokens": r["prompt_eval_count"],
        "cached_tokens": cached,
        "ttft_s": round(r["ttft_s"], 3),
        "prefill_tok_s": round(prefilled / (r["prompt_eval_duration"] / 1e9), 1),
        "decode_tok_s": round(r["eval_count"] / (r["eval_duration"] / 1e9), 1),
        "output_tokens": r["eval_count"],
    }


def header_tokens(model: str) -> int:
    """How many tokens of the chat template come BEFORE our text (always cacheable).

    Two prompts that differ from their very first character can only share the
    template header, so the second one's cached count is exactly that header.
    """
    generate(model, "A" + QUESTION)
    return generate(model, "B" + QUESTION).get("prompt_eval_cached_count", 0)


def load_rows() -> list[dict]:
    """Read speed.csv back (to redraw the chart without re-running the benchmark)."""
    with CSV_OUT.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["target_tokens"] = int(r["target_tokens"])
        r["cached"] = r["cached"] == "True"
        for k in ("prompt_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s"):
            r[k] = float(r[k])
    return rows


def main() -> None:
    if "--replot" in sys.argv:  # `python bench/speed.py --replot`: chart only, from the CSV
        rows = load_rows()
        print_summary(rows)
        plot(rows)
        return
    rows = []
    for model in MODELS:
        header = header_tokens(model)
        print(f"{model}: chat-template header = {header} tokens")
        for size in SIZES:
            body = build_prompt(model, size)
            for run in range(RUNS + 1):  # run 0 is the warm-up
                # The unique id is the VERY FIRST thing: a prefix cache can only reuse a
                # shared start. (A fixed label like "[run <id>]" before it would be shared:
                # that's 3 leaked tokens per run, caught by the check below.)
                r = generate(model, f"{uuid.uuid4().hex} (run id)\n{body}")
                if run == 0:
                    continue
                # A random id may share a token or two with the previous id by chance.
                if r.get("prompt_eval_cached_count", 0) > header + 2:
                    raise SystemExit(
                        f"cache leak: {model} size {size} run {run} reused "
                        f"{r['prompt_eval_cached_count']} tokens (header is {header})"
                    )
                rows.append(row(model, size, run, r, cached_run=False))
            print(f"  ~{size} tokens: done")
        # For contrast: send the SAME prompt twice; the second is served from the cache.
        same = f"{uuid.uuid4().hex} (run id)\n{body}"
        generate(model, same)
        rows.append(row(model, SIZES[-1], 0, generate(model, same), cached_run=True))

    CSV_OUT.parent.mkdir(exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print_summary(rows)
    plot(rows)
    print(f"\nwrote {CSV_OUT.relative_to(HERE.parent)} and {PNG_OUT.relative_to(HERE.parent)}")


def medians(rows: list[dict], model: str, size: int) -> dict:
    """Median of the measured (uncached) runs for one model at one prompt size."""
    sel = [
        r for r in rows if r["model"] == model and r["target_tokens"] == size and not r["cached"]
    ]
    keys = ("prompt_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s")
    return {k: statistics.median(r[k] for r in sel) for k in keys}


def print_summary(rows: list[dict]) -> None:
    print("\n| model | prompt tokens | TTFT (s) | prefill tok/s | decode tok/s |")
    print("|---|---:|---:|---:|---:|")
    for model in MODELS:
        for size in SIZES:
            m = medians(rows, model, size)
            print(
                f"| {model} | {m['prompt_tokens']:.0f} | {m['ttft_s']:.2f} "
                f"| {m['prefill_tok_s']:.0f} | {m['decode_tok_s']:.0f} |"
            )
    for r in (r for r in rows if r["cached"]):
        print(
            f"| {r['model']} CACHED | {r['prompt_tokens']:.0f} ({r['cached_tokens']} reused) "
            f"| {r['ttft_s']:.2f} | (skipped) | {r['decode_tok_s']:.0f} |"
        )


def plot(rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        ("ttft_s", "Time to first token (s)"),
        ("prefill_tok_s", "Prefill (tokens/s)"),
        ("decode_tok_s", "Decode (tokens/s)"),
    ]
    for ax, (key, title) in zip(axes, panels, strict=True):
        for model in MODELS:
            pts = [medians(rows, model, s) for s in SIZES]
            xs, ys = [p["prompt_tokens"] for p in pts], [p[key] for p in pts]
            ax.plot(xs, ys, marker="o", label=model)
        ax.set_xscale("log")
        # Plain tick labels at our three sizes (the log scale's default is "2x10^2" clutter).
        ax.set_xticks(SIZES, [f"~{s:,}" for s in SIZES])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)  # start at zero so differences aren't exaggerated
        ax.set_xlabel("prompt tokens")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"Ollama on M5 Max · num_ctx {NUM_CTX} · prompt cache defeated · median of {RUNS}")
    fig.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=150)


if __name__ == "__main__":
    main()
