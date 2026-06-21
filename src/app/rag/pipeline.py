"""The query-time half of RAG: retrieve relevant chunks and answer with citations.

Given a question, this module embeds it, retrieves the most similar chunks from
the vector store, builds a grounded prompt that instructs the model to answer
*only* from those numbered sources, and calls the chat model. The result carries
the answer plus machine-readable citations so each claim can be traced back to a
source.

Prompt construction is kept as pure functions (no I/O) so it can be unit-tested
directly; ``answer_question`` is the async orchestrator and takes its
dependencies by injection, mirroring ``app.rag.ingest``.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from app.llm.client import ChatResult
from app.models import ChatMessage, Role
from app.rag.store import Retrieved, VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the numbered "
    "sources below. Cite the sources you use inline with bracketed numbers like "
    "[1]. If the sources do not contain the answer, say you don't know."
)

# Message returned (without calling the LLM) when retrieval finds nothing.
_NO_CONTEXT_ANSWER = "I don't know — no relevant sources were found."

# Message returned when the store has no documents at all (vs. no good match).
_EMPTY_STORE_ANSWER = (
    "No documents have been ingested yet — run ingestion first (e.g. `make ingest`)."
)

# Length of the citation preview snippet.
_SNIPPET_CHARS = 160


@dataclass(frozen=True, slots=True)
class Citation:
    """Provenance for one numbered source the answer may have used."""

    index: int
    source: str
    score: float
    snippet: str


@dataclass(frozen=True, slots=True)
class RagResult:
    """A grounded answer plus the sources it was given and token usage."""

    answer: str
    citations: list[Citation]
    output_tokens: int | None


class Embedder(Protocol):
    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]: ...


class ChatClient(Protocol):
    async def chat(
        self, *, messages: list[ChatMessage], model: str, temperature: float
    ) -> ChatResult: ...


def _build_context(retrieved: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered source block ([1], [2], ...)."""
    return "\n\n".join(f"[{i + 1}] {hit.text}" for i, hit in enumerate(retrieved))


def _to_citations(retrieved: list[Retrieved]) -> list[Citation]:
    """Map each retrieved chunk to a Citation, one per chunk (no dedupe)."""
    citations: list[Citation] = []
    for i, hit in enumerate(retrieved):
        snippet = hit.text[:_SNIPPET_CHARS].replace("\n", " ")
        if len(hit.text) > _SNIPPET_CHARS:
            snippet += "..."
        citations.append(
            Citation(
                index=i + 1,
                source=hit.metadata.get("source", "unknown"),
                score=hit.score,
                snippet=snippet,
            )
        )
    return citations


def build_messages(question: str, retrieved: list[Retrieved]) -> list[ChatMessage]:
    """Assemble the grounded chat prompt from the question and retrieved chunks."""
    context = _build_context(retrieved)
    user_content = f"Sources:\n{context}\n\nQuestion: {question}"
    return [
        ChatMessage(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=Role.USER, content=user_content),
    ]


async def answer_question(
    question: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    chat_client: ChatClient,
    embed_model: str,
    chat_model: str,
    temperature: float,
    top_k: int,
    min_score: float,
) -> RagResult:
    """Embed the question, retrieve top-k chunks, and answer grounded in them.

    Retrieved chunks scoring below ``min_score`` are dropped before building the
    prompt, so the model only sees (and we only cite) genuinely similar sources.
    """
    logger.info("answering question (top_k=%d, min_score=%.2f)", top_k, min_score)

    if store.count() == 0:
        # Distinguish "nothing ingested yet" from "ingested but no good match".
        logger.info("vector store is empty; no documents ingested")
        return RagResult(answer=_EMPTY_STORE_ANSWER, citations=[], output_tokens=None)

    query_vectors = await embedder.embed(texts=[question], model=embed_model)
    retrieved = store.query(embedding=query_vectors[0], top_k=top_k)

    kept = [hit for hit in retrieved if hit.score >= min_score]
    dropped = len(retrieved) - len(kept)
    logger.info(
        "retrieved %d chunk(s), kept %d (dropped %d below min_score)",
        len(retrieved),
        len(kept),
        dropped,
    )

    if not kept:
        # Nothing relevant enough — don't waste an LLM call; return "don't know".
        return RagResult(answer=_NO_CONTEXT_ANSWER, citations=[], output_tokens=None)

    citations = _to_citations(kept)
    for c in citations:
        logger.debug("  [%d] %s (score=%.3f)", c.index, c.source, c.score)

    messages = build_messages(question, kept)
    result = await chat_client.chat(
        messages=messages, model=chat_model, temperature=temperature
    )
    return RagResult(
        answer=result.content,
        citations=citations,
        output_tokens=result.output_tokens,
    )
