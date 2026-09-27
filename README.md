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
