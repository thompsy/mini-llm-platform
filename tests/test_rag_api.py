from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_client, get_embedder, get_store
from app.config import Settings, get_settings
from app.llm.client import ChatResult, OllamaError
from app.llm.embeddings import EmbeddingError
from app.main import app
from app.rag.store import Retrieved


class FakeEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        if self.error is not None:
            raise self.error
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    """Returns canned Retrieved hits; records the top_k it was queried with."""

    def __init__(self, hits: list[Retrieved]) -> None:
        self.hits = hits
        self.top_k_seen: int | None = None

    def query(self, *, embedding: list[float], top_k: int) -> list[Retrieved]:
        self.top_k_seen = top_k
        return self.hits[:top_k]


class FakeChatClient:
    def __init__(
        self, content: str = "grounded [1]", error: Exception | None = None
    ) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        self.calls.append({"model": model, "temperature": temperature})
        if self.error is not None:
            raise self.error
        return ChatResult(content=self.content, model=model, output_tokens=11)


def _hit(source: str, score: float, text: str = "some text") -> Retrieved:
    return Retrieved(text=text, metadata={"source": source}, score=score)


@pytest.fixture
def test_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _wire(
    *,
    store: FakeStore,
    chat: FakeChatClient,
    embedder: FakeEmbedder | None = None,
) -> None:
    app.dependency_overrides[get_embedder] = lambda: embedder or FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_client] = lambda: chat


def test_rag_success_returns_answer_and_citations(test_client: TestClient) -> None:
    store = FakeStore([_hit("python.md", 0.8), _hit("rag.md", 0.6)])
    chat = FakeChatClient(content="Python was made by Guido [1]")
    _wire(store=store, chat=chat)

    resp = test_client.post("/rag", json={"question": "who made python?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Python was made by Guido [1]"
    assert len(body["citations"]) == 2
    assert body["citations"][0]["source"] == "python.md"
    assert body["metrics"]["output_tokens"] == 11
    assert body["metrics"]["total_seconds"] >= 0


def test_rag_uses_settings_defaults_for_top_k(test_client: TestClient) -> None:
    store = FakeStore([_hit("a.md", 0.9)])
    _wire(store=store, chat=FakeChatClient())
    app.dependency_overrides[get_settings] = lambda: Settings(rag_top_k=7)

    resp = test_client.post("/rag", json={"question": "hi"})

    assert resp.status_code == 200
    assert store.top_k_seen == 7


def test_rag_request_overrides_top_k(test_client: TestClient) -> None:
    store = FakeStore([_hit("a.md", 0.9), _hit("b.md", 0.8), _hit("c.md", 0.7)])
    _wire(store=store, chat=FakeChatClient())

    resp = test_client.post("/rag", json={"question": "hi", "top_k": 2})

    assert resp.status_code == 200
    assert store.top_k_seen == 2


def test_rag_min_score_filters_weak_citations(test_client: TestClient) -> None:
    store = FakeStore([_hit("strong.md", 0.9), _hit("weak.md", 0.3)])
    chat = FakeChatClient()
    _wire(store=store, chat=chat)

    resp = test_client.post("/rag", json={"question": "hi", "min_score": 0.5})

    assert resp.status_code == 200
    citations = resp.json()["citations"]
    assert [c["source"] for c in citations] == ["strong.md"]


def test_rag_all_below_min_score_returns_dont_know(test_client: TestClient) -> None:
    store = FakeStore([_hit("weak.md", 0.2)])
    chat = FakeChatClient()
    _wire(store=store, chat=chat)

    resp = test_client.post("/rag", json={"question": "hi", "min_score": 0.5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"] == []
    assert "don't know" in body["answer"]
    assert chat.calls == []  # no LLM call when nothing clears the threshold


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "hi", "top_k": 0},
        {"question": "hi", "top_k": 99},
        {"question": "hi", "min_score": -1},
    ],
)
def test_rag_rejects_invalid_input(
    test_client: TestClient, payload: dict[str, object]
) -> None:
    _wire(store=FakeStore([_hit("a.md", 0.9)]), chat=FakeChatClient())
    resp = test_client.post("/rag", json=payload)
    assert resp.status_code == 422


def test_rag_embedding_error_returns_503(test_client: TestClient) -> None:
    _wire(
        store=FakeStore([]),
        chat=FakeChatClient(),
        embedder=FakeEmbedder(error=EmbeddingError("embedder down")),
    )

    resp = test_client.post("/rag", json={"question": "hi"})

    assert resp.status_code == 503
    assert "embedder down" in resp.json()["detail"]


def test_rag_ollama_error_returns_503(test_client: TestClient) -> None:
    store = FakeStore([_hit("a.md", 0.9)])
    _wire(store=store, chat=FakeChatClient(error=OllamaError("backend down")))

    resp = test_client.post("/rag", json={"question": "hi"})

    assert resp.status_code == 503
    assert "backend down" in resp.json()["detail"]
