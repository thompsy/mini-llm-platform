"""Tests for the /agent endpoint."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_client, get_embedder, get_store
from app.llm.client import ChatResult, OllamaError
from app.main import app


class _ScriptedChat:
    def __init__(self, replies: list[str], error: Exception | None = None) -> None:
        self._replies = list(replies)
        self._error = error

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        if self._error is not None:
            raise self._error
        return ChatResult(content=self._replies.pop(0), model=model, output_tokens=None)


class _FakeEmbedder:
    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [[0.1] for _ in texts]


class _FakeStore:
    def add(self, **kwargs: object) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def count(self) -> int:
        return 0

    def query(self, *, embedding: list[float], top_k: int) -> list:
        return []


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_embedder] = lambda: _FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: _FakeStore()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_agent_endpoint_uses_tool_then_answers(client: TestClient) -> None:
    # calculator path needs no embedder/store, so the scripted chat fully drives it.
    app.dependency_overrides[get_client] = lambda: _ScriptedChat(
        [
            "Thought: compute it\nAction: calculator\nAction Input: 2 + 2",
            "Thought: done\nFinal Answer: 4",
        ]
    )

    resp = client.post("/agent", json={"question": "What is 2 + 2?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "4"
    assert body["stopped_reason"] == "final_answer"
    assert body["steps"][0]["action"] == "calculator"
    assert body["steps"][0]["observation"] == "4"


def test_agent_endpoint_returns_503_on_backend_error(client: TestClient) -> None:
    app.dependency_overrides[get_client] = lambda: _ScriptedChat(
        [], error=OllamaError("ollama down")
    )

    resp = client.post("/agent", json={"question": "anything"})

    assert resp.status_code == 503


def test_agent_endpoint_validates_empty_question(client: TestClient) -> None:
    app.dependency_overrides[get_client] = lambda: _ScriptedChat([])
    resp = client.post("/agent", json={"question": ""})
    assert resp.status_code == 422
