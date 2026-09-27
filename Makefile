# One command per job, so CI, the README and future-you all run things the same way.
.PHONY: install lint fmt test models bench

install:          ## create .venv with every dependency (dev + bench extras)
	uv sync --all-extras

lint:             ## style + likely-bug checks; CI fails if this fails
	uv run ruff check .
	uv run ruff format --check .

fmt:              ## auto-fix what ruff can fix
	uv run ruff check --fix .
	uv run ruff format .

test:             ## unit tests: run WITHOUT any model installed (they use FakeLLM)
	uv run pytest -q

models:           ## build the pinned model aliases from models/*.Modelfile (Session 2)
	@for f in models/*.Modelfile; do n=$$(basename $$f .Modelfile); echo "ollama create $$n"; ollama create $$n -f $$f; done

bench:            ## token table + speed benchmark (Sessions 3-4), needs Ollama running
	uv run --extra bench python bench/tokens.py
	uv run --extra bench python bench/speed.py
