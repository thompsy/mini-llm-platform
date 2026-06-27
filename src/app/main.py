import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from app.api.routes import router
from app.config import get_settings
from app.llm.client import OllamaClient
from app.llm.embeddings import OllamaEmbedder
from app.logging_config import setup_logging
from app.rag.store import ChromaStore
from app.tracing import start_trace

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("starting Mini LLM Platform (model=%s)", settings.model)
    app.state.client = OllamaClient(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    app.state.embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    app.state.store = ChromaStore(path=settings.vector_store_dir)
    logger.info("vector store dir: %s", settings.vector_store_dir)
    try:
        yield
    finally:
        logger.info("shutting down")
        await app.state.client.aclose()
        await app.state.embedder.aclose()


app = FastAPI(title="Mini LLM Platform", lifespan=lifespan)


@app.middleware("http")
async def trace_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Wrap each request in a Trace so nested ``record_span`` calls are collected.

    The trace is logged on completion; persisting it to SQLite is the remaining
    M3 step. Requests that record no spans (e.g. /health) are not logged.
    """
    with start_trace(request.url.path) as trace:
        response = await call_next(request)

    if trace.spans:
        logger.info(
            "trace %s: %.1fms (%d span(s))",
            trace.route,
            trace.duration_ms,
            len(trace.spans),
        )
        for span in trace.spans:
            logger.debug(
                "  span %s: %.1fms %s", span.name, span.duration_ms, span.metadata
            )
    return response


app.include_router(router)


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    main()
