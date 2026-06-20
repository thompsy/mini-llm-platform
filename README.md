# Mini LLM Platform

A small, self-hosted LLM learning project — built to sharpen Python and build hands-on LLM intuition.

## Roadmap

- [x] **M1 — Inference API.** Typed FastAPI `/chat` endpoint (request → response) wrapping a local model via Ollama; pydantic schemas; capture request latency; tests with the model client mocked.
- [ ] **M2 — RAG.** Ingest + chunk + embed a document corpus; store vectors (pgvector / Chroma); retrieve top-k; answer grounded with citations.
- [ ] **M3 — Tracing.** Record each LLM + retrieval call as a span (tokens, latency, cost) to SQLite; inspectable traces.
- [ ] **M4 — Eval harness.** Golden Q&A set; score with exact-match + LLM-as-judge; CLI + report; flag regressions across prompt/model changes.
- [ ] **Stretch.** Streaming responses (SSE) + TTFT metric; reasoning agent with tool use (ReAct + function calling); distillation toy; full `docker-compose` (app + Postgres/pgvector).

## Tech stack

- **Language:** Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- **API:** FastAPI + uvicorn, pydantic / pydantic-settings, httpx
- **LLM backend:** [Ollama](https://ollama.com) (local model)
- **Quality:** ruff (lint + format), mypy, pre-commit

## Getting started

### Prerequisites

- Python 3.12 (see `.python-version`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

```bash
# Install dependencies into a local .venv
uv sync

# Install git hooks (ruff lint + format on commit)
uv run pre-commit install
```

### Run

```bash
# Start the API (host/port from .env; defaults to http://127.0.0.1:8000)
uv run python -m app.main
```

With [Ollama](https://ollama.com) running, from another terminal:

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### Development

```bash
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy src          # type-check
uv run pytest            # tests
```

## Project structure

```
.
├── src/app/
│   ├── main.py              # FastAPI app + lifespan + entry point
│   ├── config.py            # settings (pydantic-settings)
│   ├── models.py            # request/response schemas
│   ├── api/routes.py        # /chat, /health endpoints
│   └── llm/client.py        # async Ollama client
├── pyproject.toml           # deps + ruff/mypy config
├── .pre-commit-config.yaml
├── .env.example
└── README.md
```
