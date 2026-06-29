import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.agent.runner import run_agent
from app.agent.tools import build_tools
from app.config import Settings, get_settings
from app.llm.client import OllamaClient, OllamaError
from app.llm.embeddings import EmbeddingError, OllamaEmbedder
from app.models import (
    AgentRequest,
    AgentResponse,
    AgentStepModel,
    ChatMetrics,
    ChatRequest,
    ChatResponse,
    CitationModel,
    RagRequest,
    RagResponse,
    SpanModel,
    TraceDetailModel,
    TraceSummaryModel,
)
from app.rag.pipeline import answer_question
from app.rag.store import ChromaStore
from app.tracing import record_span
from app.tracing_store import SQLiteTraceStore

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


def get_trace_store(request: Request) -> SQLiteTraceStore:
    trace_store: SQLiteTraceStore = request.app.state.trace_store
    return trace_store


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
        with record_span("chat") as span:
            result = await client.chat(
                messages=payload.messages, model=model, temperature=temperature
            )
            span.metadata["model"] = model
            span.metadata["output_tokens"] = result.output_tokens
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


@router.get("/traces", response_model=list[TraceSummaryModel])
async def list_traces(
    trace_store: Annotated[SQLiteTraceStore, Depends(get_trace_store)],
    limit: int = 50,
) -> list[TraceSummaryModel]:
    summaries = trace_store.list_traces(limit=limit)
    return [
        TraceSummaryModel(
            id=s.id,
            route=s.route,
            duration_ms=s.duration_ms,
            created_at=s.created_at,
            span_count=s.span_count,
        )
        for s in summaries
    ]


@router.get("/traces/{trace_id}", response_model=TraceDetailModel)
async def get_trace(
    trace_id: str,
    trace_store: Annotated[SQLiteTraceStore, Depends(get_trace_store)],
) -> TraceDetailModel:
    trace = trace_store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"trace {trace_id!r} not found",
        )
    return TraceDetailModel(
        id=trace.id,
        route=trace.route,
        duration_ms=trace.duration_ms,
        created_at=trace.created_at,
        spans=[
            SpanModel(name=sp.name, duration_ms=sp.duration_ms, metadata=sp.metadata)
            for sp in trace.spans
        ],
    )


@router.post("/agent", response_model=AgentResponse)
async def agent(
    payload: AgentRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: Annotated[OllamaEmbedder, Depends(get_embedder)],
    store: Annotated[ChromaStore, Depends(get_store)],
    client: Annotated[OllamaClient, Depends(get_client)],
) -> AgentResponse:
    max_steps = payload.max_steps or settings.agent_max_steps
    tools = build_tools(
        embedder=embedder,
        store=store,
        chat_client=client,
        embed_model=settings.embed_model,
        chat_model=settings.model,
        temperature=settings.default_temperature,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )
    logger.debug(
        "agent request: %d chars, max_steps=%d", len(payload.question), max_steps
    )

    try:
        result = await run_agent(
            payload.question,
            chat_client=client,
            model=settings.model,
            tools=tools,
            max_steps=max_steps,
        )
    except (OllamaError, EmbeddingError) as exc:
        logger.warning("agent failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    logger.info(
        "agent ok: %d step(s), stopped=%s", len(result.steps), result.stopped_reason
    )
    return AgentResponse(
        answer=result.answer,
        steps=[
            AgentStepModel(
                thought=s.thought,
                action=s.action,
                action_input=s.action_input,
                observation=s.observation,
            )
            for s in result.steps
        ],
        stopped_reason=result.stopped_reason,
    )
