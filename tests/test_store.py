from pathlib import Path

from app.rag.store import ChromaStore, Retrieved


def _store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(path=str(tmp_path), collection_name="test-corpus")


def _seed(store: ChromaStore) -> None:
    # Three orthogonal-ish 2D vectors so "nearest" is unambiguous.
    store.add(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        documents=["doc a", "doc b", "doc c"],
        metadatas=[{"source": "x"}, {"source": "y"}, {"source": "z"}],
    )


def test_query_returns_nearest_first_with_metadata(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)

    hits = store.query(embedding=[1.0, 0.0], top_k=3)

    assert isinstance(hits[0], Retrieved)
    assert hits[0].text == "doc a"
    assert hits[0].metadata == {"source": "x"}
    # Exact match -> cosine distance 0 -> score 1.0.
    assert hits[0].score == 1.0


def test_results_sorted_by_descending_score(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)

    hits = store.query(embedding=[1.0, 0.0], top_k=3)

    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_results(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)

    hits = store.query(embedding=[1.0, 0.0], top_k=2)

    assert len(hits) == 2


def test_count_reflects_added_documents(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.count() == 0
    _seed(store)
    assert store.count() == 3


def test_upsert_same_id_does_not_duplicate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed(store)

    # Re-add id "a" with new content; count must stay 3, content must update.
    store.add(
        ids=["a"],
        embeddings=[[1.0, 0.0]],
        documents=["doc a v2"],
        metadatas=[{"source": "x2"}],
    )

    assert store.count() == 3
    hits = store.query(embedding=[1.0, 0.0], top_k=1)
    assert hits[0].text == "doc a v2"
    assert hits[0].metadata == {"source": "x2"}
