import logging

import httpx

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when the Ollama embeddings backend cannot fulfil a request."""


class OllamaEmbedder:
    """Thin async client for Ollama's /api/embed endpoint.

    Mirrors ``OllamaClient``: same constructor shape (injectable transport for
    offline tests) and the same error-wrapping conventions. Turns text into
    fixed-length vectors so chunks and queries can be compared by similarity.
    """

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

    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        """Embed a batch of texts, preserving input order.

        Args:
            texts: Non-empty list of strings to embed.
            model: The embedding model name (e.g. ``nomic-embed-text``).

        Returns:
            One embedding vector per input text, in the same order.
        """
        if not texts:
            raise EmbeddingError("texts must not be empty")

        logger.debug("embedding %d text(s) with model %r", len(texts), model)
        payload = {"model": model, "input": texts}
        try:
            response = await self._client.post("/api/embed", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise EmbeddingError(
                    f"Model {model!r} not available — run `ollama pull {model}`"
                ) from exc
            raise EmbeddingError(f"Ollama embed request failed: {exc}") from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embed request failed: {exc}") from exc

        try:
            data = response.json()
            embeddings = data["embeddings"]
        except (KeyError, ValueError) as exc:
            raise EmbeddingError(f"Unexpected Ollama embed response: {exc}") from exc

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )
        dim = len(embeddings[0]) if embeddings else 0
        logger.debug("received %d vector(s) of dim %d", len(embeddings), dim)
        return [list(vector) for vector in embeddings]

    async def aclose(self) -> None:
        await self._client.aclose()
