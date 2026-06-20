from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.llm.client import OllamaClient


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.client = OllamaClient(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(title="Mini LLM Platform", lifespan=lifespan)
app.include_router(router)


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)


if __name__ == "__main__":
    main()
