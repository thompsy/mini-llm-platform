from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ChatMetrics(BaseModel):
    total_seconds: float
    output_tokens: int | None = None


class ChatResponse(BaseModel):
    content: str
    model: str
    metrics: ChatMetrics | None = None


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=2.0)


class CitationModel(BaseModel):
    index: int
    source: str
    score: float
    snippet: str


class RagResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    metrics: ChatMetrics | None = None


class TraceSummaryModel(BaseModel):
    id: str
    route: str
    duration_ms: float
    created_at: float
    span_count: int


class SpanModel(BaseModel):
    name: str
    duration_ms: float
    metadata: dict[str, Any]


class TraceDetailModel(BaseModel):
    id: str
    route: str
    duration_ms: float
    created_at: float
    spans: list[SpanModel]
