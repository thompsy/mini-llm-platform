from collections.abc import Callable

import httpx
import pytest

from app.llm.client import OllamaClient, OllamaError
from app.models import ChatMessage, Role


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaClient:
    return OllamaClient(
        base_url="http://test",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )


async def test_chat_parses_successful_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {"role": "assistant", "content": "hello"},
                "eval_count": 7,
            },
        )

    client = _make_client(handler)
    try:
        result = await client.chat(
            messages=[ChatMessage(role=Role.USER, content="hi")],
            model="test-model",
            temperature=0.0,
        )
    finally:
        await client.aclose()

    assert result.content == "hello"
    assert result.model == "test-model"
    assert result.output_tokens == 7


async def test_chat_wraps_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _make_client(handler)
    try:
        with pytest.raises(OllamaError):
            await client.chat(
                messages=[ChatMessage(role=Role.USER, content="hi")],
                model="m",
                temperature=0.0,
            )
    finally:
        await client.aclose()


async def test_chat_wraps_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    client = _make_client(handler)
    try:
        with pytest.raises(OllamaError):
            await client.chat(
                messages=[ChatMessage(role=Role.USER, content="hi")],
                model="m",
                temperature=0.0,
            )
    finally:
        await client.aclose()
