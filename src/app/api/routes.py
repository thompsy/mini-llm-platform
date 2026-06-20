import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.llm.client import OllamaClient, OllamaError
from app.models import ChatMetrics, ChatRequest, ChatResponse

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

    start = time.perf_counter()
    try:
        result = await client.chat(
            messages=payload.messages, model=model, temperature=temperature
        )
    except OllamaError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    total_seconds = time.perf_counter() - start

    return ChatResponse(
        content=result.content,
        model=result.model,
        metrics=ChatMetrics(
            total_seconds=total_seconds, output_tokens=result.output_tokens
        ),
    )
