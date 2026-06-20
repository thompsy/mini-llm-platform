from collections.abc import Callable

import httpx
import pytest

from app.llm.embeddings import EmbeddingError, OllamaEmbedder


def _make_embedder(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaEmbedder:
    return OllamaEmbedder(
        base_url="http://test",
        timeout=5.0,
        transport=httpx.MockTransport(handler),
    )


async def test_embed_returns_vectors_in_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    embedder = _make_embedder(handler)
    try:
        vectors = await embedder.embed(texts=["a", "b"], model="m")
    finally:
        await embedder.aclose()

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_empty_input_raises() -> None:
    embedder = _make_embedder(lambda request: httpx.Response(200, json={}))
    try:
        with pytest.raises(EmbeddingError, match="must not be empty"):
            await embedder.embed(texts=[], model="m")
    finally:
        await embedder.aclose()


async def test_embed_wraps_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    embedder = _make_embedder(handler)
    try:
        with pytest.raises(EmbeddingError):
            await embedder.embed(texts=["a"], model="m")
    finally:
        await embedder.aclose()


async def test_embed_404_suggests_pull() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    embedder = _make_embedder(handler)
    try:
        with pytest.raises(EmbeddingError, match="ollama pull"):
            await embedder.embed(texts=["a"], model="missing-model")
    finally:
        await embedder.aclose()


async def test_embed_wraps_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    embedder = _make_embedder(handler)
    try:
        with pytest.raises(EmbeddingError):
            await embedder.embed(texts=["a"], model="m")
    finally:
        await embedder.aclose()


async def test_embed_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Asked for two, backend returns one.
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    embedder = _make_embedder(handler)
    try:
        with pytest.raises(EmbeddingError, match="Expected 2 embeddings, got 1"):
            await embedder.embed(texts=["a", "b"], model="m")
    finally:
        await embedder.aclose()
