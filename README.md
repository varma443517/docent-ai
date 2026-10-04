# docent-ai

A local-first assistant that answers questions about the Kubernetes docs, built in public one lab a week.
Everything runs on my laptop (Ollama on an M5 Max); tests run in CI with no model at all.

```bash
make install   # uv sync
make test      # unit tests, no model needed
make models    # build pinned Ollama model aliases
make bench     # token table + speed benchmark
```

## Week 1 — Hello, local LLM
_In progress: a typed client, a streaming CLI and an honest prefill/decode benchmark._

**Same text, different token bills** (`bench/tokens.py`, raw mode, special tokens subtracted):

| text | chars | llama3.2:3b tokens | qwen3:8b tokens |
|---|---:|---:|---:|
| English prose | 411 | 81 | 81 |
| Python code | 372 | 92 | 92 |
| Kubernetes YAML | 486 | 136 | 147 |
| JSON | 210 | 80 | 84 |
| `terminationGracePeriodSeconds` | 29 | 4 | 4 |
| Telugu sentence | 91 | 162 | 136 |

English packs ~5 characters per token, YAML ~3.5, Telugu ~0.6, so "4k tokens" is a different
amount of text for every model and language.
