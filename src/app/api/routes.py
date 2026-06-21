import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.llm.client import OllamaClient, OllamaError
from app.llm.embeddings import EmbeddingError, OllamaEmbedder
from app.models import (
    ChatMetrics,
    ChatRequest,
    ChatResponse,
    CitationModel,
    RagRequest,
    RagResponse,
)
from app.rag.pipeline import answer_question
from app.rag.store import ChromaStore

logger = logging.getLogger(__name__)

router = APIRouter()


def get_client(request: Request) -> OllamaClient:
    client: OllamaClient = request.app.state.client
    return client


def get_embedder(request: Request) -> OllamaEmbedder:
    embedder: OllamaEmbedder = request.app.state.embedder
    return embedder


def get_store(request: Request) -> ChromaStore:
    store: ChromaStore = request.app.state.store
    return store


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


@router.post("/rag", response_model=RagResponse)
async def rag(
    payload: RagRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[OllamaEmbedder, Depends(get_embedder)],
    store: Annotated[ChromaStore, Depends(get_store)],
    client: Annotated[OllamaClient, Depends(get_client)],
) -> RagResponse:
    top_k = payload.top_k or settings.rag_top_k
    min_score = (
        payload.min_score if payload.min_score is not None else settings.rag_min_score
    )
    logger.debug(
        "rag request: %d chars, top_k=%d, min_score=%s",
        len(payload.question),
        top_k,
        min_score,
    )

    start = time.perf_counter()
    try:
        result = await answer_question(
            payload.question,
            embedder=embedder,
            store=store,
            chat_client=client,
            embed_model=settings.embed_model,
            chat_model=settings.model,
            temperature=settings.default_temperature,
            top_k=top_k,
            min_score=min_score,
        )
    except (OllamaError, EmbeddingError) as exc:
        logger.warning("rag failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    total_seconds = time.perf_counter() - start

    logger.info(
        "rag ok: %d citation(s), %.3fs, output_tokens=%s",
        len(result.citations),
        total_seconds,
        result.output_tokens,
    )
    return RagResponse(
        answer=result.answer,
        citations=[
            CitationModel(
                index=c.index, source=c.source, score=c.score, snippet=c.snippet
            )
            for c in result.citations
        ],
        metrics=ChatMetrics(
            total_seconds=total_seconds, output_tokens=result.output_tokens
        ),
    )
