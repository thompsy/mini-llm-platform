import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_client
from app.config import Settings, get_settings
from app.llm.client import ChatResult, OllamaError
from app.main import app


class FakeClient:
    """Stands in for OllamaClient so route tests run offline."""

    def __init__(
        self, result: ChatResult | None = None, error: Exception | None = None
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        self.calls.append({"model": model, "temperature": temperature})
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@pytest.fixture
def test_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _use_client(fake: FakeClient) -> None:
    app.dependency_overrides[get_client] = lambda: fake


def test_health(test_client: TestClient) -> None:
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_success(test_client: TestClient) -> None:
    fake = FakeClient(result=ChatResult(content="hi there", model="m", output_tokens=5))
    _use_client(fake)

    resp = test_client.post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "hi there"
    assert body["model"] == "m"
    assert body["metrics"]["output_tokens"] == 5
    assert body["metrics"]["total_seconds"] >= 0


def test_chat_uses_settings_defaults(test_client: TestClient) -> None:
    fake = FakeClient(
        result=ChatResult(content="x", model="default-model", output_tokens=1)
    )
    _use_client(fake)
    app.dependency_overrides[get_settings] = lambda: Settings(
        model="default-model", default_temperature=0.3
    )

    resp = test_client.post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert resp.status_code == 200
    assert fake.calls[0] == {"model": "default-model", "temperature": 0.3}


def test_chat_request_overrides_settings(test_client: TestClient) -> None:
    fake = FakeClient(result=ChatResult(content="x", model="x", output_tokens=1))
    _use_client(fake)
    app.dependency_overrides[get_settings] = lambda: Settings(
        model="default-model", default_temperature=0.7
    )

    resp = test_client.post(
        "/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "override-model",
            "temperature": 0.0,
        },
    )

    assert resp.status_code == 200
    # 0.0 must survive resolution (the `is not None` guard, not `or`).
    assert fake.calls[0] == {"model": "override-model", "temperature": 0.0}


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "hi"}], "temperature": 5},
        {"messages": [{"role": "user", "content": ""}]},
    ],
)
def test_chat_rejects_invalid_input(
    test_client: TestClient, payload: dict[str, object]
) -> None:
    resp = test_client.post("/chat", json=payload)
    assert resp.status_code == 422


def test_chat_ollama_error_returns_503(test_client: TestClient) -> None:
    _use_client(FakeClient(error=OllamaError("backend down")))

    resp = test_client.post(
        "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )

    assert resp.status_code == 503
    assert "backend down" in resp.json()["detail"]


def test_chat_logs_success_at_info(
    test_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _use_client(FakeClient(result=ChatResult(content="x", model="m", output_tokens=3)))

    with caplog.at_level(logging.INFO, logger="app.api.routes"):
        test_client.post(
            "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )

    info = [r.message for r in caplog.records if r.levelno == logging.INFO]
    assert any("chat ok" in m and "model=m" in m for m in info)


def test_chat_logs_warning_on_error(
    test_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    _use_client(FakeClient(error=OllamaError("backend down")))

    with caplog.at_level(logging.WARNING, logger="app.api.routes"):
        test_client.post(
            "/chat", json={"messages": [{"role": "user", "content": "hi"}]}
        )

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("chat failed" in m for m in warnings)
