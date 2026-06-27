# Mini LLM Platform

A small, self-hosted LLM learning project — built to sharpen Python and build hands-on LLM intuition.

## Roadmap

- [x] **M1 — Inference API.** Typed FastAPI `/chat` endpoint (request → response) wrapping a local model via Ollama; pydantic schemas; capture request latency; tests with the model client mocked.
- [x] **M2 — RAG.** Ingest + chunk + embed a document corpus; store vectors (Chroma); retrieve top-k; answer grounded with citations.
- [ ] **M3 — Tracing.** Record each LLM + retrieval call as a span (tokens, latency, cost) to SQLite; inspectable traces.
- [ ] **M4 — Eval harness.** Golden Q&A set; score with exact-match + LLM-as-judge; CLI + report; flag regressions across prompt/model changes.
- [ ] **Stretch.** Streaming responses (SSE) + TTFT metric; reasoning agent with tool use (ReAct + function calling); distillation toy; full `docker-compose` (app + Postgres/pgvector).

## Follow-up concepts & projects

Ideas to extend this project and deepen understanding — roughly easiest to hardest.

> See [`concepts.md`](concepts.md) for running notes on the ideas behind this
> project (embeddings, cosine similarity, and more),
> [`reading.md`](reading.md) for books on designing these kinds of systems, and
> [`learning-roadmap.md`](learning-roadmap.md) for how the skills map to roles
> (AI engineer, FDE, ML engineer, …).

### Build your own models

The long-term goal: replace the off-the-shelf Ollama models with ones built from
scratch, then serve them through this same platform (the `OllamaClient` /
`OllamaEmbedder` abstractions make the backend swappable).

- **Toy embedding model.** Train a small encoder that maps text → a fixed-length
  vector. Start with averaged word embeddings (word2vec/GloVe-style) or a tiny
  transformer encoder trained with a contrastive objective (similar texts close,
  dissimilar far). Expose it behind the same interface as `OllamaEmbedder` and
  point `APP_EMBED_MODEL` at it. _Concept: representation learning, contrastive
  loss, cosine geometry._
- **Toy generation model.** Train a small character- or token-level transformer
  decoder (a "nanoGPT"-scale model) on a modest corpus. Wrap it to match the
  `OllamaClient.chat` interface so `/chat` and `/rag` work unchanged.
  _Concept: autoregressive next-token prediction, sampling/temperature._
- **Swap them into RAG.** Once both exist, run the full M2 pipeline end-to-end on
  your own models and compare retrieval/answer quality against the Ollama models.

### Agentic patterns

- **ReAct agent with tool use.** Implement the Reason + Act loop: the model emits `Thought → Action → Observation` in a cycle until it can answer. Wire the existing RAG pipeline as a `rag_search` tool alongside trivials like `calculator` and `get_date`. Add a `/agent` endpoint alongside `/chat` and `/rag`. _Concepts: tool schemas, structured output parsing, how the model decides when to call vs. answer, context growth over turns._
- **Iterative / self-correcting RAG.** After initial retrieval, ask the model whether the context is sufficient; if not, have it reformulate the query and retrieve again (up to N rounds). Extends M2 without a major architecture change. _Concepts: query rewriting, conditional retrieval loops, cost/latency vs. answer-quality tradeoff._
- **Planner / executor split.** A two-agent pipeline: a Planner decomposes a goal into a JSON step plan; an Executor runs each step using RAG, chat, or other tools and feeds results back. Naturally pairs with M3 tracing (one span per step). _Concepts: orchestration, structured outputs, how plan/execute failures propagate._
- **LLM-as-judge eval agent.** A meta-agent that runs the M4 golden Q&A set, grades each answer with the model on a structured rubric, and writes a regression report. _Concepts: LLM-as-judge, self-serving bias, rubric design, calibration against human labels._

### Concepts worth a deeper dive

- **Embeddings:** why encoder (bidirectional) vs. decoder (causal); dimensionality
  vs. quality; how distance metrics (cosine vs. dot vs. L2) change retrieval.
- **Chunking strategies:** fixed-window vs. semantic/sentence-aware chunking; how
  size/overlap affect recall and answer grounding.
- **Tokenization:** BPE vs. WordPiece; how token budgets bound chunk size and
  context windows.
- **Evaluation:** measuring retrieval quality (recall@k, MRR) separately from
  answer quality — connects directly to M4.

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

## RAG (M2)

Answer questions grounded in a local document corpus (`data/`), citing the
sources used. The pipeline embeds each chunk, retrieves the most similar chunks
for a question, and asks the model to answer **only** from those sources.

> Requires an embedding model in addition to the chat model. `make setup` pulls
> both; otherwise run `ollama pull nomic-embed-text`.

### Ingest

Indexing is an offline step — run it once, and again whenever `data/` changes:

```bash
make ingest                     # ingest data/ into the vector store (.chroma)
make ingest DATA=path/to/docs   # ingest a different directory

# verbose: per-chunk previews + embedding dimensions
uv run python -m app.rag.ingest -v data
```

### Query

```bash
# A) API already running (e.g. `make run` in another terminal):
make rag PROMPT="Who created Python?"

# B) one-shot: start the API, ask, then stop it:
make rag-once PROMPT="What is cosine similarity?"

# or raw curl (top_k / min_score are optional overrides):
curl http://127.0.0.1:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{"question": "Who created Python?", "top_k": 4, "min_score": 0.5}'
```

The answer cites sources inline as `[1]`, `[2]`, … and the response lists each
citation (source + similarity score). If no chunk clears `min_score`, the answer
is "I don't know"; if nothing has been ingested yet, it says to run `make ingest`.

### Configuration

RAG and logging settings (prefixed `APP_`, set via `.env` — see `.env.example`):

| Variable               | Default            | Meaning                                    |
| ---------------------- | ------------------ | ------------------------------------------ |
| `APP_EMBED_MODEL`      | `nomic-embed-text` | Ollama model used to embed text            |
| `APP_VECTOR_STORE_DIR` | `.chroma`          | where the Chroma index is persisted        |
| `APP_RAG_TOP_K`        | `4`                | chunks retrieved per query                 |
| `APP_RAG_MIN_SCORE`    | `0.5`              | drop retrieved chunks below this score     |
| `APP_CHUNK_SIZE`       | `200`              | max words per chunk                        |
| `APP_CHUNK_OVERLAP`    | `40`               | words shared between adjacent chunks        |
| `APP_LOG_LEVEL`        | `INFO`             | logging level (`DEBUG` for per-chunk logs) |

## Project structure

```
.
├── src/app/
│   ├── main.py              # FastAPI app + lifespan + entry point
│   ├── config.py            # settings (pydantic-settings)
│   ├── models.py            # request/response schemas (chat + rag)
│   ├── logging_config.py    # central logging setup
│   ├── api/routes.py        # /chat, /rag, /health endpoints
│   ├── llm/
│   │   ├── client.py        # async Ollama chat client
│   │   └── embeddings.py    # async Ollama embeddings client
│   └── rag/
│       ├── chunking.py      # deterministic word-based chunker
│       ├── store.py         # Chroma vector store (VectorStore protocol)
│       ├── ingest.py        # offline ingest CLI
│       └── pipeline.py      # query-time RAG (retrieve + ground + cite)
├── data/                    # sample corpus (.md)
├── tests/                   # pytest suite
├── Makefile                 # setup/lint/test/run/chat/ingest/rag targets
├── concepts.md              # notes on the ideas behind the project
├── pyproject.toml           # deps + ruff/mypy config
├── .pre-commit-config.yaml
├── .env.example
└── README.md
```
