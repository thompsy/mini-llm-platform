"""Vector storage and similarity search for RAG, backed by Chroma.

A vector store keeps embedding vectors (one per chunk) plus their source text
and metadata, and answers "which stored chunks are most similar to this query
vector?". Similarity is measured by cosine distance, so retrieval depends on the
*direction* of a vector (its meaning) rather than its magnitude.
"""

from dataclasses import dataclass
from typing import Protocol, cast

import chromadb
from chromadb.api.types import Embeddings, Metadata

# Chroma collection names must be 3-512 chars; this is the default corpus.
DEFAULT_COLLECTION = "documents"


@dataclass(frozen=True, slots=True)
class Retrieved:
    """A single search hit: the chunk text, its metadata, and similarity score."""

    text: str
    metadata: dict[str, str]
    score: float


class VectorStore(Protocol):
    """Structural interface for a vector store.

    The rest of the app depends on this Protocol, not on Chroma directly, so the
    backend can be swapped (e.g. for a home-grown store) without touching ingest
    or the RAG pipeline.
    """

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None: ...

    def query(self, *, embedding: list[float], top_k: int) -> list[Retrieved]: ...

    def count(self) -> int: ...


class ChromaStore:
    """A persistent Chroma-backed vector store using cosine similarity."""

    def __init__(self, *, path: str, collection_name: str = DEFAULT_COLLECTION) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        # upsert (not add) so re-ingesting the same ids updates in place rather
        # than erroring or duplicating.
        self._collection.upsert(
            ids=ids,
            embeddings=cast(Embeddings, embeddings),
            documents=documents,
            metadatas=cast(list[Metadata], metadatas),
        )

    def query(self, *, embedding: list[float], top_k: int) -> list[Retrieved]:
        result = self._collection.query(
            query_embeddings=cast(Embeddings, [embedding]),
            n_results=top_k,
        )
        # Chroma nests results one level per query embedding; we sent one.
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]

        retrieved: list[Retrieved] = []
        for text, metadata, distance in zip(
            documents, metadatas, distances, strict=True
        ):
            retrieved.append(
                Retrieved(
                    text=text,
                    metadata={str(k): str(v) for k, v in (metadata or {}).items()},
                    # Cosine distance in [0, 2]; convert so higher = more similar.
                    score=1.0 - float(distance),
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()
