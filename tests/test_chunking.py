import pytest

from app.rag.chunking import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("", size=10, overlap=2) == []
    assert chunk_text("   \n\t ", size=10, overlap=2) == []


def test_text_shorter_than_size_is_one_chunk() -> None:
    assert chunk_text("one two three", size=10, overlap=2) == ["one two three"]


def test_exact_multiple_no_overlap() -> None:
    text = "a b c d e f"
    assert chunk_text(text, size=2, overlap=0) == ["a b", "c d", "e f"]


def test_overlap_shares_words_between_chunks() -> None:
    text = "a b c d e"
    # size=3, overlap=1 -> step=2: [a b c], [c d e]
    assert chunk_text(text, size=3, overlap=1) == ["a b c", "c d e"]


def test_remainder_tail_is_shorter_final_chunk() -> None:
    text = "a b c d e f"
    # size=3, overlap=1 -> step=2: [a b c], [c d e], start=4 -> [e f] reaches end.
    assert chunk_text(text, size=3, overlap=1) == ["a b c", "c d e", "e f"]


def test_whitespace_is_normalised() -> None:
    assert chunk_text("a    b\n\nc", size=2, overlap=0) == ["a b", "c"]


def test_no_window_repeats_only_the_tail() -> None:
    # When the window already covers the end, we stop rather than emit a chunk
    # that is a pure suffix of the previous one.
    text = "a b c d"
    # size=3, overlap=1 -> step=2: [a b c], start=2 -> [c d] reaches end, stop.
    assert chunk_text(text, size=3, overlap=1) == ["a b c", "c d"]


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_size_raises(size: int) -> None:
    with pytest.raises(ValueError, match="size must be > 0"):
        chunk_text("a b c", size=size, overlap=0)


@pytest.mark.parametrize("overlap", [-1, 3, 4])
def test_invalid_overlap_raises(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("a b c", size=3, overlap=overlap)
