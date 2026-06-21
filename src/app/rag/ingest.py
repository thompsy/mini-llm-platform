"""Offline ingestion: read documents, chunk, embed, and store their vectors.

This wires together the three building blocks of the indexing half of RAG:
chunking (app.rag.chunking), embedding (app.llm.embeddings) and storage
(app.rag.store). It is meant to be run as a one-shot CLI before serving queries:

    uv run python -m app.rag.ingest docs/

Chunks are stored under stable ids ("{source}:{index}") so re-running ingestion
upserts in place rather than creating duplicates.
"""

import argparse
import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.llm.embeddings import OllamaEmbedder
from app.logging_config import setup_logging
from app.rag.chunking import chunk_text
from app.rag.store import ChromaStore, VectorStore

logger = logging.getLogger(__name__)

# Length of the chunk text preview emitted at DEBUG level.
_PREVIEW_CHARS = 80

# Only plain-text corpora for M2.
SUPPORTED_SUFFIXES = (".md", ".txt")


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Summary of an ingestion run."""

    files: int
    chunks: int


def _iter_files(paths: Iterable[Path]) -> list[Path]:
    """Expand the given paths into a sorted, deterministic list of files.

    Directories are walked recursively for supported file types; explicit file
    paths are kept as-is. Sorting makes ingestion order deterministic.
    """
    found: set[Path] = set()
    for path in paths:
        if path.is_dir():
            for suffix in SUPPORTED_SUFFIXES:
                found.update(path.rglob(f"*{suffix}"))
        elif path.is_file():
            found.add(path)
    return sorted(found)


async def ingest(
    paths: Iterable[Path],
    *,
    embedder: OllamaEmbedder,
    store: VectorStore,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    """Read, chunk, embed, and upsert every supported file under ``paths``."""
    files = _iter_files(paths)
    logger.info("ingesting %d file(s)", len(files))

    total_chunks = 0
    for file in files:
        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            logger.info("skipping %s (no chunks)", file)
            continue

        logger.info("ingesting %s -> %d chunk(s)", file, len(chunks))
        for index, chunk in enumerate(chunks):
            preview = chunk[:_PREVIEW_CHARS].replace("\n", " ")
            logger.debug(
                "  chunk %d (%d words): %s%s",
                index,
                len(chunk.split()),
                preview,
                "..." if len(chunk) > _PREVIEW_CHARS else "",
            )

        embeddings = await embedder.embed(texts=chunks, model=model)
        source = str(file)
        ids = [f"{source}:{index}" for index in range(len(chunks))]
        metadatas = [{"source": source, "chunk": str(i)} for i in range(len(chunks))]

        store.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    logger.info("ingested %d chunk(s) from %d file(s)", total_chunks, len(files))
    return IngestResult(files=len(files), chunks=total_chunks)


async def _run(paths: list[Path]) -> IngestResult:
    settings = get_settings()
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    store = ChromaStore(path=settings.vector_store_dir)
    try:
        return await ingest(
            paths,
            embedder=embedder,
            store=store,
            model=settings.embed_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    finally:
        await embedder.aclose()


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to ingest (.md/.txt).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (per-chunk previews), overriding APP_LOG_LEVEL.",
    )
    args = parser.parse_args()

    level = "DEBUG" if args.verbose else get_settings().log_level
    setup_logging(level)

    asyncio.run(_run(args.paths))


if __name__ == "__main__":
    _main()
