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
