"""Tests for SQLite trace persistence (SQLiteTraceStore)."""

import json
import sqlite3
from pathlib import Path

from app.tracing import Trace
from app.tracing_store import SQLiteTraceStore


def _make_trace() -> Trace:
    trace = Trace(route="/rag")
    embed = trace.start_span("embed_query")
    embed.metadata["model"] = "e"
    embed.finish()
    chat = trace.start_span("chat")
    chat.metadata["output_tokens"] = 7
    chat.finish()
    trace.finish()
    return trace


def test_save_persists_trace_and_spans(tmp_path: Path) -> None:
    path = str(tmp_path / "traces.db")
    store = SQLiteTraceStore(path=path)
    trace = _make_trace()
    store.save(trace)
    store.close()

    conn = sqlite3.connect(path)
    try:
        traces = conn.execute("SELECT id, route, duration_ms FROM traces").fetchall()
        spans = conn.execute(
            "SELECT name, metadata FROM spans WHERE trace_id = ? ORDER BY id",
            (trace.id,),
        ).fetchall()
    finally:
        conn.close()

    assert len(traces) == 1
    assert traces[0][0] == trace.id
    assert traces[0][1] == "/rag"
    assert traces[0][2] >= 0

    assert [s[0] for s in spans] == ["embed_query", "chat"]
    assert json.loads(spans[0][1]) == {"model": "e"}
    assert json.loads(spans[1][1]) == {"output_tokens": 7}


def test_save_multiple_traces_accumulate(tmp_path: Path) -> None:
    path = str(tmp_path / "traces.db")
    store = SQLiteTraceStore(path=path)
    store.save(_make_trace())
    store.save(_make_trace())
    store.close()

    conn = sqlite3.connect(path)
    try:
        (traces,) = conn.execute("SELECT COUNT(*) FROM traces").fetchone()
        (spans,) = conn.execute("SELECT COUNT(*) FROM spans").fetchone()
    finally:
        conn.close()

    assert traces == 2
    assert spans == 4


def test_save_trace_without_spans(tmp_path: Path) -> None:
    path = str(tmp_path / "traces.db")
    store = SQLiteTraceStore(path=path)
    trace = Trace(route="/health")
    trace.finish()
    store.save(trace)  # should not raise
    store.close()

    conn = sqlite3.connect(path)
    try:
        (count,) = conn.execute("SELECT COUNT(*) FROM traces").fetchone()
    finally:
        conn.close()

    assert count == 1


def test_get_trace_round_trip(tmp_path: Path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    trace = _make_trace()
    store.save(trace)

    fetched = store.get_trace(trace.id)
    store.close()

    assert fetched is not None
    assert fetched.id == trace.id
    assert fetched.route == "/rag"
    assert [s.name for s in fetched.spans] == ["embed_query", "chat"]
    assert fetched.spans[0].metadata == {"model": "e"}
    assert fetched.spans[1].metadata == {"output_tokens": 7}


def test_get_trace_returns_none_when_missing(tmp_path: Path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    assert store.get_trace("does-not-exist") is None
    store.close()


def test_list_traces_newest_first_with_span_count(tmp_path: Path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    older = _make_trace()  # 2 spans
    newer = Trace(route="/chat")
    newer.start_span("chat").finish()  # 1 span
    newer.finish()
    store.save(older)
    store.save(newer)

    summaries = store.list_traces(limit=10)
    store.close()

    # Newest first; created_at is stamped at save time, so newer saved last.
    assert [s.id for s in summaries] == [newer.id, older.id]
    assert summaries[0].span_count == 1
    assert summaries[1].span_count == 2


def test_list_traces_respects_limit(tmp_path: Path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    for _ in range(3):
        store.save(_make_trace())

    summaries = store.list_traces(limit=2)
    store.close()

    assert len(summaries) == 2
