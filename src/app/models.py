from enum import StrEnum

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
