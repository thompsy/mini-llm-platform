"""Request tracing: spans, traces, and async-safe context propagation.

Each request creates a Trace that collects Spans — named, timed operations.
The current trace is stored in a ContextVar so any code in the call stack can
record a span without needing a trace parameter passed explicitly.

Usage:

    with record_span("embed_query") as span:
        result = await embedder.embed(texts)
        span.metadata["tokens"] = len(texts)
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import monotonic
from typing import Generator


@dataclass
class Span:
    name: str
    started_at: float = field(default_factory=monotonic)
    finished_at: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def finish(self) -> None:
        self.finished_at = monotonic()

    @property
    def duration_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else monotonic()
        return (end - self.started_at) * 1000


@dataclass
class Trace:
    route: str
    started_at: float = field(default_factory=monotonic)
    finished_at: float | None = None
    spans: list[Span] = field(default_factory=list)

    def start_span(self, name: str) -> Span:
        span = Span(name=name)
        self.spans.append(span)
        return span

    def finish(self) -> None:
        self.finished_at = monotonic()

    @property
    def duration_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else monotonic()
        return (end - self.started_at) * 1000


_current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


@contextmanager
def record_span(name: str) -> Generator[Span, None, None]:
    """Record a named span on the current trace, if one is active.

    If no trace is active (e.g. during tests or CLI use), this is a no-op and
    yields a detached Span that is never attached to any trace.
    """
    trace = _current_trace.get()
    span = trace.start_span(name) if trace is not None else Span(name=name)
    try:
        yield span
    finally:
        span.finish()
