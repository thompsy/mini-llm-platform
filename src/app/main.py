import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.llm.client import OllamaClient
from app.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("starting Mini LLM Platform (model=%s)", settings.model)
    app.state.client = OllamaClient(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    try:
        yield
    finally:
        logger.info("shutting down")
        await app.state.client.aclose()


app = FastAPI(title="Mini LLM Platform", lifespan=lifespan)
app.include_router(router)


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    main()
