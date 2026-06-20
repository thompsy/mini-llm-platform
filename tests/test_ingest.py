from pathlib import Path

from app.rag.ingest import ingest
from app.rag.store import ChromaStore


class FakeEmbedder:
    """Returns a deterministic vector per text; records how it was called."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        self.calls.append(list(texts))
        # One distinct 1-D vector per text; values don't matter for these tests.
        return [[float(i + 1)] for i in range(len(texts))]


def _store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(path=str(tmp_path / "chroma"), collection_name="ingest-test")


def _ingest(corpus: Path, store: ChromaStore, embedder: FakeEmbedder):  # type: ignore[no-untyped-def]
    return ingest(
        [corpus],
        embedder=embedder,  # type: ignore[arg-type]
        store=store,
        model="fake",
        chunk_size=3,
        chunk_overlap=0,
    )


async def test_ingest_reads_chunks_and_stores(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("one two three four five six")  # -> 2 chunks

    store = _store(tmp_path)
    embedder = FakeEmbedder()

    result = await _ingest(corpus, store, embedder)

    assert result.files == 1
    assert result.chunks == 2
    assert store.count() == 2


async def test_ingest_sets_source_metadata(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "a.md"
    doc.write_text("alpha beta gamma")

    store = _store(tmp_path)
    await _ingest(corpus, store, FakeEmbedder())

    hits = store.query(embedding=[1.0], top_k=1)
    assert hits[0].metadata["source"] == str(doc)
    assert hits[0].metadata["chunk"] == "0"


async def test_ingest_empty_corpus_is_noop(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()  # no files

    store = _store(tmp_path)
    result = await _ingest(corpus, store, FakeEmbedder())

    assert result == result.__class__(files=0, chunks=0)
    assert store.count() == 0


async def test_ingest_skips_empty_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "empty.md").write_text("   \n\t ")
    (corpus / "real.md").write_text("alpha beta")

    store = _store(tmp_path)
    embedder = FakeEmbedder()
    result = await _ingest(corpus, store, embedder)

    # Empty file contributes no chunks, and isn't sent to the embedder.
    assert result.files == 2
    assert result.chunks == 1
    assert embedder.calls == [["alpha beta"]]


async def test_reingest_upserts_in_place(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("alpha beta gamma")

    store = _store(tmp_path)
    await _ingest(corpus, store, FakeEmbedder())
    first = store.count()

    # Re-ingest the identical corpus: stable ids -> upsert, no duplication.
    await _ingest(corpus, store, FakeEmbedder())

    assert store.count() == first


async def test_ingest_only_picks_supported_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.md").write_text("alpha beta")
    (corpus / "notes.txt").write_text("gamma delta")
    (corpus / "image.png").write_text("not text")

    store = _store(tmp_path)
    result = await _ingest(corpus, store, FakeEmbedder())

    assert result.files == 2  # .md and .txt only
