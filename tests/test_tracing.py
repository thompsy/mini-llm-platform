"""Tests for the tracing core: Span, Trace, ContextVar, record_span."""

import asyncio
import logging
import sqlite3

from fastapi.testclient import TestClient

from app.api.routes import get_client
from app.llm.client import ChatResult
from app.main import app
from app.rag.pipeline import answer_question
from app.rag.store import Retrieved
from app.tracing import Trace, _current_trace, record_span, start_trace
from app.tracing_store import SQLiteTraceStore


def test_span_duration_while_open():
    trace = Trace(route="/test")
    span = trace.start_span("op")
    assert span.duration_ms >= 0
    assert span.finished_at is None


def test_span_duration_after_finish():
    trace = Trace(route="/test")
    span = trace.start_span("op")
    span.finish()
    assert span.finished_at is not None
    assert span.duration_ms >= 0


def test_trace_collects_spans():
    trace = Trace(route="/test")
    trace.start_span("a")
    trace.start_span("b")
    assert [s.name for s in trace.spans] == ["a", "b"]


def test_record_span_attaches_to_current_trace():
    trace = Trace(route="/test")
    token = _current_trace.set(trace)
    try:
        with record_span("embed_query") as span:
            span.metadata["tokens"] = 5
    finally:
        _current_trace.reset(token)

    assert len(trace.spans) == 1
    assert trace.spans[0].name == "embed_query"
    assert trace.spans[0].metadata["tokens"] == 5
    assert trace.spans[0].finished_at is not None


def test_record_span_noop_without_trace():
    # No ContextVar set — should not raise, span is detached
    with record_span("orphan") as span:
        span.metadata["x"] = 1
    assert span.name == "orphan"


def test_record_span_finishes_on_exception():
    trace = Trace(route="/test")
    token = _current_trace.set(trace)
    try:
        with record_span("failing_op"):
            raise ValueError("boom")
    except ValueError:
        pass
    finally:
        _current_trace.reset(token)

    assert trace.spans[0].finished_at is not None


def test_contextvars_isolated_across_tasks():
    """Concurrent tasks each see their own trace."""
    results: dict[str, str | None] = {}

    async def run():
        async def task(name: str, route: str) -> None:
            trace = Trace(route=route)
            token = _current_trace.set(trace)
            try:
                await asyncio.sleep(0)  # yield to let other task run
                results[name] = (
                    _current_trace.get().route if _current_trace.get() else None
                )
            finally:
                _current_trace.reset(token)

        await asyncio.gather(task("t1", "/chat"), task("t2", "/rag"))

    asyncio.run(run())
    assert results["t1"] == "/chat"
    assert results["t2"] == "/rag"


# --- Wiring: pipeline instrumentation + request middleware (M3 steps 1-2) ---


class _FakeEmbedder:
    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def add(self, **kwargs: object) -> None:  # pragma: no cover - unused here
        raise NotImplementedError

    def count(self) -> int:
        return len(self._hits)

    def query(self, *, embedding: list[float], top_k: int) -> list[Retrieved]:
        return self._hits


class _FakeChat:
    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content="answer", model=model, output_tokens=7)


async def test_pipeline_records_spans_under_active_trace() -> None:
    hits = [Retrieved(text="doc", metadata={"source": "s"}, score=0.9)]
    with start_trace("/rag") as trace:
        await answer_question(
            "q",
            embedder=_FakeEmbedder(),
            store=_FakeStore(hits),
            chat_client=_FakeChat(),
            embed_model="e",
            chat_model="c",
            temperature=0.0,
            top_k=4,
            min_score=0.5,
        )

    assert [s.name for s in trace.spans] == ["embed_query", "retrieve", "chat"]
    assert all(s.finished_at is not None for s in trace.spans)
    chat_span = trace.spans[-1]
    assert chat_span.metadata["model"] == "c"
    assert chat_span.metadata["output_tokens"] == 7


def _capture_main_logger(caplog) -> logging.Logger:
    """Attach caplog's handler directly to the app.main logger.

    The app's ``setup_logging`` runs ``logging.basicConfig(force=True)`` during
    TestClient startup, which strips pytest's capture handler off the *root*
    logger. Attaching it to the ``app.main`` logger itself survives that, since
    ``basicConfig`` only touches the root.
    """
    main_logger = logging.getLogger("app.main")
    main_logger.addHandler(caplog.handler)
    main_logger.setLevel(logging.INFO)
    return main_logger


def test_middleware_creates_and_logs_trace(caplog) -> None:
    """The chat span recorded inside the endpoint reaches the middleware's trace.

    This also proves the ContextVar set in the middleware propagates into the
    endpoint task; if it didn't, the trace would have no spans and nothing logs.
    """
    app.dependency_overrides[get_client] = lambda: _FakeChat()
    main_logger = _capture_main_logger(caplog)
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
            )
    finally:
        main_logger.removeHandler(caplog.handler)
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert any("trace /chat" in r.message for r in caplog.records)


def test_middleware_skips_logging_when_no_spans(caplog) -> None:
    main_logger = _capture_main_logger(caplog)
    try:
        with TestClient(app) as client:
            resp = client.get("/health")
    finally:
        main_logger.removeHandler(caplog.handler)

    assert resp.status_code == 200
    # The per-request trace log starts with "trace /<route>"; assert none fired
    # (distinct from the lifespan's "trace store: ..." startup line).
    assert not any(r.message.startswith("trace /") for r in caplog.records)


def test_middleware_persists_trace(tmp_path) -> None:
    """End-to-end: a request's trace is written to the trace store."""
    db_path = str(tmp_path / "traces.db")
    store = SQLiteTraceStore(path=db_path)
    app.dependency_overrides[get_client] = lambda: _FakeChat()
    try:
        with TestClient(app) as client:
            # Replace the store the lifespan opened with our temp one; the
            # middleware reads request.app.state.trace_store per request.
            app.state.trace_store = store
            resp = client.post(
                "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
            )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200

    conn = sqlite3.connect(db_path)
    try:
        routes = conn.execute("SELECT route FROM traces").fetchall()
        span_names = conn.execute("SELECT name FROM spans").fetchall()
    finally:
        conn.close()

    assert routes == [("/chat",)]
    assert span_names == [("chat",)]


def _seed_trace(store: SQLiteTraceStore) -> Trace:
    trace = Trace(route="/rag")
    span = trace.start_span("chat")
    span.metadata["output_tokens"] = 7
    span.finish()
    trace.finish()
    store.save(trace)
    return trace


def test_traces_endpoints_list_and_detail(tmp_path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    trace = _seed_trace(store)

    with TestClient(app) as client:
        app.state.trace_store = store
        list_resp = client.get("/traces")
        detail_resp = client.get(f"/traces/{trace.id}")
        missing_resp = client.get("/traces/nope")

    assert list_resp.status_code == 200
    summaries = list_resp.json()
    assert len(summaries) == 1
    assert summaries[0]["route"] == "/rag"
    assert summaries[0]["span_count"] == 1

    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == trace.id
    assert detail["spans"][0]["name"] == "chat"
    assert detail["spans"][0]["metadata"]["output_tokens"] == 7

    assert missing_resp.status_code == 404
