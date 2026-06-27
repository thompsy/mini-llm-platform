"""Persistence for traces: write finished traces and their spans to SQLite.

Mirrors the ``VectorStore`` pattern in ``app.rag.store`` — the rest of the app
depends on the ``TraceStore`` Protocol, not on SQLite directly, so the backend
stays swappable and tracing can be tested in isolation.

``save`` is the write side (called from the request middleware); ``list_traces``
and ``get_trace`` are the read side (served by the /traces endpoints).
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Protocol

from app.tracing import Trace

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TraceSummary:
    """A trace without its spans — for the list view."""

    id: str
    route: str
    duration_ms: float
    created_at: float
    span_count: int


@dataclass(frozen=True, slots=True)
class StoredSpan:
    """A span as read back from storage."""

    name: str
    duration_ms: float
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class StoredTrace:
    """A trace plus its spans — for the detail view."""

    id: str
    route: str
    duration_ms: float
    created_at: float
    spans: list[StoredSpan]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id          TEXT PRIMARY KEY,
    route       TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL REFERENCES traces(id),
    name        TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    metadata    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id);
"""


class TraceStore(Protocol):
    """Structural interface for persisting and reading traces."""

    def save(self, trace: Trace) -> None: ...

    def list_traces(self, *, limit: int) -> list[TraceSummary]: ...

    def get_trace(self, trace_id: str) -> StoredTrace | None: ...


class SQLiteTraceStore:
    """A persistent, SQLite-backed trace store."""

    def __init__(self, *, path: str) -> None:
        # check_same_thread=False: the API may persist from a worker thread that
        # differs from the one that opened the connection. Writes are short and
        # serialised by the single connection, which is enough for this app.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save(self, trace: Trace) -> None:
        """Persist a finished trace and its spans in one transaction.

        Span metadata (arbitrary keys like model/tokens) is stored as a JSON blob
        rather than fixed columns, so new span fields need no schema change.
        """
        with self._conn:  # commits on success, rolls back on exception
            self._conn.execute(
                "INSERT INTO traces (id, route, duration_ms, created_at) "
                "VALUES (?, ?, ?, ?)",
                (trace.id, trace.route, trace.duration_ms, time.time()),
            )
            self._conn.executemany(
                "INSERT INTO spans (trace_id, name, duration_ms, metadata) "
                "VALUES (?, ?, ?, ?)",
                [
                    (trace.id, span.name, span.duration_ms, json.dumps(span.metadata))
                    for span in trace.spans
                ],
            )

    def list_traces(self, *, limit: int = 50) -> list[TraceSummary]:
        """Return the most recent traces (newest first), without their spans."""
        rows = self._conn.execute(
            "SELECT t.id, t.route, t.duration_ms, t.created_at, COUNT(s.id) "
            "FROM traces t LEFT JOIN spans s ON s.trace_id = t.id "
            "GROUP BY t.id "
            "ORDER BY t.created_at DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            TraceSummary(
                id=row[0],
                route=row[1],
                duration_ms=row[2],
                created_at=row[3],
                span_count=row[4],
            )
            for row in rows
        ]

    def get_trace(self, trace_id: str) -> StoredTrace | None:
        """Return one trace with its spans, or None if no such trace exists."""
        row = self._conn.execute(
            "SELECT id, route, duration_ms, created_at FROM traces WHERE id = ?",
            (trace_id,),
        ).fetchone()
        if row is None:
            return None

        span_rows = self._conn.execute(
            "SELECT name, duration_ms, metadata FROM spans "
            "WHERE trace_id = ? ORDER BY id",
            (trace_id,),
        ).fetchall()
        spans = [
            StoredSpan(name=sr[0], duration_ms=sr[1], metadata=json.loads(sr[2]))
            for sr in span_rows
        ]
        return StoredTrace(
            id=row[0],
            route=row[1],
            duration_ms=row[2],
            created_at=row[3],
            spans=spans,
        )

    def close(self) -> None:
        self._conn.close()
