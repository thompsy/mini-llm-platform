"""Deterministic, word-based text chunking for RAG ingestion.

Splitting a document into smaller overlapping passages ("chunks") is the first
step of RAG. Chunks are what we embed and retrieve, so they need to be small
enough to embed cleanly and focused enough to retrieve precisely, while overlap
keeps ideas that straddle a boundary intact in at least one chunk.
"""


def chunk_text(text: str, *, size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping, word-based chunks.

    Args:
        text: The source text. Whitespace is normalised; empty input yields ``[]``.
        size: Maximum number of words per chunk. Must be > 0.
        overlap: Number of words shared between adjacent chunks. Must satisfy
            ``0 <= overlap < size`` (overlap < size guarantees forward progress).

    Returns:
        A list of chunk strings, in document order. The final chunk may be
        shorter than ``size``.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + size]
        chunks.append(" ".join(chunk_words))
        # Once a window reaches the end of the document, stop: any further
        # windows would only repeat the tail.
        if start + size >= len(words):
            break
    return chunks
