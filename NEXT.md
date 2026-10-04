# Next

**Week 1 is done**: client, CLI, pinned models, token counter, speed benchmark.

Open threads for later weeks:
- `docent ask` hides qwen3's thinking: the /v1 API returns it in a separate field. Decide: show it dimmed, or turn thinking off by default.
- Nothing checks prompt size before sending. Once retrieval adds documents (week 3), count tokens first and refuse or trim, so the window never overflows silently.
- The benchmark covers 2 models. Stretch: add gpt-oss:20b (mixture of experts), compare q4_K_M vs q8_0 quantisation, and test sampling variety at temperature 0 / 0.7 / 1.2.
- Turn this skeleton into a GitHub template repo for the other tracks.
