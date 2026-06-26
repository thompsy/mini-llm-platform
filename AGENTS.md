# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

A small, self-hosted LLM learning project: a typed FastAPI service over a local
Ollama model, plus a RAG pipeline (Chroma). It's built milestone-by-milestone
(see the Roadmap in `README.md`); M1 (inference) and M2 (RAG) are done, M3
(tracing) is in progress.

## Version control: this repo uses Jujutsu (jj), not git directly

The repo is a **colocated jj + git** checkout: jj (`jj 0.32.x`) is the working
VCS, with git as its storage backend (`.git` is present but managed by jj). Use
`jj` commands, not `git`, for everyday work:

- There is **no staging area / index**. The working copy is itself a commit
  (`@`); edits to already-tracked files are auto-snapshotted on every `jj`
  command, so `jj st` / `jj diff` show current changes without an `add` step.
- **Tracking new files is explicit here.** This repo sets
  `snapshot.auto-track = none()`, so newly created files are *not* picked up
  automatically — they appear under `Untracked paths` (`?`) in `jj st` and stay
  out of every commit until you track them. Run:
  - `jj file track <path>...` — track specific new files (e.g.
    `jj file track AGENTS.md src/app/tracing.py`).
  - `jj file track .` — track everything not ignored under the current dir.
  - `.gitignore` is honoured, so ignored files won't be tracked even with `.`.
  - (Auto-tracking is the jj default; it's only disabled in this repo via the
    `none()` setting above. Don't change that setting unless asked.)
- Common equivalents: `jj st` (status), `jj diff`, `jj log`, `jj describe -m`
  (set the current commit's message), `jj new` (start the next change), `jj bookmark`
  (jj's branches).
- Prefer not to invoke `git` mutating commands here — they can desync jj's view
  of the working copy. Read-only `git` inspection is fine.
- Only create or describe commits (`jj commit` / `jj describe`) when the user
  explicitly asks; the auto-snapshot updates the working-copy commit but won't
  finalize history on its own.

## Commands

Everything goes through the `Makefile` (run `make help` for the full list). Under
the hood it's `uv`, so `uv run <tool>` works too.

```bash
make setup            # uv sync + pre-commit install + .env + pull Ollama models
make check            # lint + typecheck (what CI/pre-commit gate on)
make test             # uv run pytest
make run              # start the API (uvicorn, reload on)
make ingest           # index data/ into the Chroma store (offline step)
make rag-once PROMPT="..."   # start API, ask one RAG question, stop API
```

Single test / focused runs:

```bash
uv run pytest tests/test_pipeline.py            # one file
uv run pytest tests/test_pipeline.py::test_name # one test
uv run pytest -k "citation"                      # by keyword
uv run mypy src                                  # typecheck (src only)
uv run ruff check . && uv run ruff format .      # lint + format
```

The test suite runs fully offline — Ollama is **not** required for `make test`
(the model client is mocked). `make run`, `make ingest`, and the `rag`/`chat`
targets *do* need a running Ollama with the chat + embed models pulled.

## Architecture

Source lives under `src/app/` (installed as the package `app` via hatchling, so
imports are `app.x` and mypy/pytest run against `src`/`tests`). Three layers:

- **API** (`api/routes.py`, `main.py`, `models.py`) — FastAPI routes `/chat`,
  `/rag`, `/health`. Long-lived clients (`OllamaClient`, `OllamaEmbedder`,
  `ChromaStore`) are created once in the `lifespan` handler, stashed on
  `app.state`, and injected into routes via `Depends(get_*)`. `models.py` holds
  the pydantic request/response schemas (the wire contract); internal results
  are plain dataclasses.
- **LLM backend** (`llm/client.py`, `llm/embeddings.py`) — thin async `httpx`
  wrappers over Ollama's `/api/chat` and embeddings endpoints. Backend errors
  are normalized to `OllamaError` / `EmbeddingError`, which routes translate to
  HTTP 503.
- **RAG** (`rag/`) — split into an *offline* indexing half (`chunking.py` →
  `ingest.py` → `store.py`) run via CLI, and a *query-time* half (`pipeline.py`)
  hit by the `/rag` route. `store.py` uses cosine similarity and returns
  `score = 1 - distance` (higher = more similar).

### Key conventions to follow

- **Swappable backends via `Protocol`.** The app depends on `VectorStore`
  (`rag/store.py`), `Embedder`, and `ChatClient` (`rag/pipeline.py`) structural
  protocols, never on Chroma/Ollama concretely. This is deliberate — the
  long-term goal is to swap in home-grown models/stores. Preserve it: new code
  should depend on the protocol, and tests substitute fakes that satisfy it.
- **Dependency injection over globals.** Async orchestrators (`answer_question`,
  `ingest`) take their dependencies (embedder, store, client, models) as
  keyword-only arguments rather than reaching for `get_settings()`. Settings are
  resolved at the edge (routes / CLI `_run`) and passed down.
- **Pure functions for prompt/text logic.** Prompt assembly (`build_messages`,
  `_build_context`, `_to_citations`) and chunking are I/O-free so they're unit
  tested directly without mocks.
- **Config** is a single pydantic-settings `Settings` (`config.py`), all vars
  prefixed `APP_`, accessed through the `lru_cache`d `get_settings()`. Add new
  config there with a default; document it in the README table.

### Tracing (M3, in progress)

`tracing.py` defines `Trace`/`Span` and a `record_span()` context manager backed
by a `ContextVar`. `record_span` is a **no-op that yields a detached span when no
trace is active** (tests, CLI), so it can be sprinkled into pipeline/LLM code
without those call sites needing a trace. Wiring it into the request lifecycle
and persisting to SQLite is the remaining M3 work.

## Testing notes

- `asyncio_mode = "auto"` (pyproject) — `async def test_*` needs no decorator.
- Route tests use FastAPI's `TestClient` plus `app.dependency_overrides` to swap
  in fakes (see `FakeClient` in `tests/test_api.py`); always clear overrides in
  the fixture teardown.
- One test file per module (`test_<module>.py`), mirroring `src/app/`.
