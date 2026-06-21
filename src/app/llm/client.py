import logging
from dataclasses import dataclass

import httpx

from app.models import ChatMessage

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when the Ollama backend cannot fulfil a request."""


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    model: str
    output_tokens: int | None


class OllamaClient:
    """Thin async client for Ollama's /api/chat endpoint (non-streaming)."""

    def __init__(
        self,
        base_url: str,
        timeout: float,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, transport=transport
        )

    async def chat(
        self,
        *,
        messages: list[ChatMessage],
        model: str,
        temperature: float,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": [
                {"role": m.role.value, "content": m.content} for m in messages
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        logger.debug("calling Ollama /api/chat (model=%s)", model)
        try:
            response = await self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise OllamaError(
                    f"Model {model!r} not available — run `ollama pull {model}`"
                ) from exc
            logger.debug("Ollama HTTP error: %s", exc)
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            logger.debug("Ollama transport error: %s", exc)
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, ValueError) as exc:
            raise OllamaError(f"Unexpected Ollama response: {exc}") from exc

        return ChatResult(
            content=content,
            model=data.get("model", model),
            output_tokens=data.get("eval_count"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
