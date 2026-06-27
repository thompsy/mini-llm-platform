"""Persistence for traces: write finished traces and their spans to SQLite.

Mirrors the ``VectorStore`` pattern in ``app.rag.store`` — the rest of the app
depends on the ``TraceStore`` Protocol, not on SQLite directly, so the backend
stays swappable and the persistence step can be tested in isolation.

Only ``save`` lives here for now; reading traces back (for an inspection endpoint
or CLI) is the next M3 step.
"""

import json
import logging
import sqlite3
import time
from typing import Protocol

from app.tracing import Trace

logger = logging.getLogger(__name__)

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
    """Structural interface for persisting traces."""

    def save(self, trace: Trace) -> None: ...


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

    def close(self) -> None:
        self._conn.close()
