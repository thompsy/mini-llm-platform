# Mini LLM Platform

A small, self-hosted LLM learning project — built to sharpen Python and build hands-on LLM intuition.

## Roadmap

- [x] **M1 — Inference API.** Typed FastAPI `/chat` endpoint (request → response) wrapping a local model via Ollama; pydantic schemas; capture request latency; tests with the model client mocked.
- [x] **M2 — RAG.** Ingest + chunk + embed a document corpus; store vectors (Chroma); retrieve top-k; answer grounded with citations.
- [x] **M3 — Tracing.** Record each LLM + retrieval call as a span (tokens, latency) to SQLite; inspectable traces via `/traces`.
- [x] **M4 — Eval harness.** Golden Q&A set; score with exact-match + recall@k + LLM-as-judge; CLI + report; flag regressions across prompt/model changes.
- [ ] **M5 — Agent.** Text-based ReAct loop (Thought → Action → Observation) with tool use over the existing corpus; `/agent` endpoint + CLI; each step/tool traced (M3) and scoreable (M4). See [Agent (M5 — planned)](#agent-m5--planned).
- [ ] **Stretch.** Streaming responses (SSE) + TTFT metric; native tool-calling (vs text ReAct); distillation toy; full `docker-compose` (app + Postgres/pgvector).

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
| `APP_TRACE_STORE_PATH` | `traces.db`        | SQLite file where request traces are stored |
| `APP_JUDGE_MODEL`      | `model`            | model used by the LLM-as-judge (defaults to chat model) |
| `APP_EVAL_REGRESSION_THRESHOLD` | `0.05`    | min mean-score drop vs baseline to flag a regression |
| `APP_LOG_LEVEL`        | `INFO`             | logging level (`DEBUG` for per-chunk logs) |

## Tracing (M3)

Every `/chat` and `/rag` request is recorded as a **trace**: a timed tree of
**spans** (one per embed / retrieve / chat step), with metadata like the model
used and output tokens. A middleware opens a trace per request, and the
instrumented pipeline records spans into it via a `ContextVar` — so the code
doing the work never has to pass a trace around. Finished traces are persisted to
SQLite (`traces.db` by default; see `APP_TRACE_STORE_PATH`), including requests
that fail mid-flight.

Inspect them with the API running (`make run` in another terminal):

```bash
make traces                 # list recent traces (LIMIT=20 by default)
make traces LIMIT=50        # show more
make trace ID=<trace_id>    # one trace with its spans (get an id from `make traces`)
```

Or via raw HTTP:

```bash
curl 'http://127.0.0.1:8000/traces?limit=20'   # newest first; id, route, duration, span_count
curl http://127.0.0.1:8000/traces/<trace_id>   # full detail incl. each span + metadata
```

## Evaluation (M4)

A golden-set eval harness measures answer quality so changes (prompt, model,
chunking, `top_k`, …) can be judged objectively instead of by vibes, and silent
regressions get caught. The golden set (`evals/golden.json`) is a list of
questions, each with a reference answer and the source files that *should* be
retrieved. Each question is run through the RAG pipeline and scored three ways:

- **exact-match** — is the (normalised) reference contained in the answer? A
  strict, deterministic baseline.
- **recall@k** — fraction of expected sources actually retrieved. Separates
  *retrieval* quality from *answer* quality.
- **LLM-as-judge** — a model grades the answer against the reference as
  `CORRECT` / `PARTIAL` / `INCORRECT` (→ `1.0` / `0.5` / `0.0`), catching correct
  paraphrases that exact-match misses.

Each eval item runs inside its own trace (`eval:<id>`), so a run is inspectable
via `make traces` just like a live request.

> Needs Ollama running and an ingested corpus (`make ingest`).

```bash
make eval                                   # run the golden set, print a score table
make eval OUTPUT=report.json                # also save the full report as JSON
make eval BASELINE=report.json              # flag regressions vs a saved report
make eval GOLDEN=path/to/golden.json        # use a different golden set
```

`make eval BASELINE=…` exits non-zero if any scorer's mean drops more than
`APP_EVAL_REGRESSION_THRESHOLD` below the baseline — usable as a CI gate. Note
the judge is an LLM and so non-deterministic run-to-run; trust aggregate trends
over any single number, and grow the golden set as the corpus grows.

## Agent (M5 — planned)

> **Status: planned, not yet built.** This section is the design; the milestone
> is tracked in the roadmap above.

An agent answers a question by **reasoning and acting in a loop** rather than in
a single shot: it thinks, picks a tool, observes the result, and repeats until it
can answer. M5 adds a **text-based [ReAct](https://arxiv.org/abs/2210.03629)**
(Reason + Act) agent over the existing document corpus.

### The loop

The model emits `Thought → Action → Action Input`; the runner executes the named
tool and appends an `Observation`; the (growing) transcript is resent each turn
until the model emits a `Final Answer` or a step cap is hit:

```
Thought: I should look this up in the documents.
Action: rag_search
Action Input: who created Python?
Observation: Guido van Rossum created Python. [data/python.md]
Thought: I have the answer.
Final Answer: Python was created by Guido van Rossum.
```

### Tools (general-purpose, domain-neutral)

- **`rag_search`** — the existing retrieval + grounding pipeline; the agent's main tool.
- **`calculator`** — safe arithmetic (AST-parsed, not `eval`).
- **`get_date`** — the current date.

### Design

- New package `src/app/agent/`: `tools.py` (a `Tool` protocol + the three tools),
  `react.py` (pure prompt builder + step parser), `runner.py` (`run_agent` loop),
  and a `__main__.py` CLI. A `POST /agent` endpoint and `make agent` mirror the
  existing `/rag` surface.
- **Action mechanism:** text-based ReAct parsed with a forgiving parser
  (case-insensitive, ignores hallucinated observations; an unparseable reply gets
  a format-reminder observation rather than crashing). Chosen for being
  backend-agnostic and for teaching structured-output parsing; native
  tool-calling is a later stretch. Small local models adhere to the format
  imperfectly — mitigated with few tools, a low step cap, and `temperature=0`.
- **Tracing (M3):** each iteration is an `agent_step` span and each tool call a
  `tool:<name>` span, so an `/agent` trace shows the whole reasoning tree
  (including the RAG pipeline's own nested spans) via `make traces`.
- **Eval (M4):** the harness can route golden questions through the agent instead
  of plain RAG, scoring with the same metrics plus agent signals (e.g. step count).

### Phasing

1. **M5.1** — tool layer (`Tool` protocol + the three tools) + tests.
2. **M5.2** — ReAct prompt + parser + `run_agent` loop + per-step tracing + tests.
3. **M5.3** — `/agent` endpoint + CLI + `make agent` + tests.
4. **M5.4** — eval-the-agent integration + docs.

## Project structure

```
.
├── src/app/
│   ├── main.py              # FastAPI app + lifespan + entry point
│   ├── config.py            # settings (pydantic-settings)
│   ├── models.py            # request/response schemas (chat + rag + traces)
│   ├── logging_config.py    # central logging setup
│   ├── tracing.py           # Trace/Span core + per-request ContextVar
│   ├── tracing_store.py     # SQLite trace store (TraceStore protocol)
│   ├── api/routes.py        # /chat, /rag, /health, /traces endpoints
│   ├── llm/
│   │   ├── client.py        # async Ollama chat client
│   │   └── embeddings.py    # async Ollama embeddings client
│   ├── rag/
│   │   ├── chunking.py      # deterministic word-based chunker
│   │   ├── store.py         # Chroma vector store (VectorStore protocol)
│   │   ├── ingest.py        # offline ingest CLI
│   │   └── pipeline.py      # query-time RAG (retrieve + ground + cite)
│   └── evals/
│       ├── dataset.py       # load/validate the golden set
│       ├── scorers.py       # exact-match, recall@k, LLM-as-judge
│       ├── runner.py        # run golden set through RAG + score (traced)
│       ├── report.py        # console table, JSON I/O, baseline comparison
│       └── __main__.py      # eval CLI (python -m app.evals)
├── data/                    # sample corpus (.md)
├── evals/golden.json        # golden Q&A set for the eval harness
├── tests/                   # pytest suite
├── Makefile                 # setup/lint/test/run/chat/ingest/rag/traces/eval targets
├── concepts.md              # notes on the ideas behind the project
├── pyproject.toml           # deps + ruff/mypy config
├── .pre-commit-config.yaml
├── .env.example
└── README.md
```
