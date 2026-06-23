"""Tests for the tracing core: Span, Trace, ContextVar, record_span."""

import asyncio

from app.tracing import Trace, _current_trace, record_span


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
                results[name] = _current_trace.get().route if _current_trace.get() else None
            finally:
                _current_trace.reset(token)

        await asyncio.gather(task("t1", "/chat"), task("t2", "/rag"))

    asyncio.run(run())
    assert results["t1"] == "/chat"
    assert results["t2"] == "/rag"
