import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.llm.client import OllamaClient, OllamaError
from app.models import ChatMetrics, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_client(request: Request) -> OllamaClient:
    client: OllamaClient = request.app.state.client
    return client


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[OllamaClient, Depends(get_client)],
) -> ChatResponse:
    model = payload.model or settings.model
    temperature = (
        payload.temperature
        if payload.temperature is not None
        else settings.default_temperature
    )
    logger.debug(
        "chat request: %d message(s), model=%s, temperature=%s",
        len(payload.messages),
        model,
        temperature,
    )

    start = time.perf_counter()
    try:
        result = await client.chat(
            messages=payload.messages, model=model, temperature=temperature
        )
    except OllamaError as exc:
        logger.warning("chat failed (model=%s): %s", model, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    total_seconds = time.perf_counter() - start

    logger.info(
        "chat ok: model=%s, %.3fs, output_tokens=%s",
        result.model,
        total_seconds,
        result.output_tokens,
    )
    return ChatResponse(
        content=result.content,
        model=result.model,
        metrics=ChatMetrics(
            total_seconds=total_seconds, output_tokens=result.output_tokens
        ),
    )
