"""Tests for loading and validating the golden Q&A set."""

import json
from pathlib import Path

import pytest

from app.evals.dataset import GoldenItem, GoldenSetError, load_golden_set


def _write(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_valid_set(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "golden.json",
        [
            {
                "id": "a",
                "question": "Q1?",
                "reference_answer": "A1",
                "expected_sources": ["data/x.md"],
                "notes": "n",
            },
            {"id": "b", "question": "Q2?", "reference_answer": "A2"},
        ],
    )

    items = load_golden_set(path)

    assert items == [
        GoldenItem(
            id="a",
            question="Q1?",
            reference_answer="A1",
            expected_sources=["data/x.md"],
            notes="n",
        ),
        GoldenItem(id="b", question="Q2?", reference_answer="A2"),
    ]
    # Optional fields default to None when absent.
    assert items[1].expected_sources is None
    assert items[1].notes is None


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="not found"):
        load_golden_set(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GoldenSetError, match="not valid JSON"):
        load_golden_set(path)


def test_not_a_list_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "golden.json", {"id": "a"})
    with pytest.raises(GoldenSetError, match="must be a JSON array"):
        load_golden_set(path)


def test_non_object_item_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "golden.json", ["just a string"])
    with pytest.raises(GoldenSetError, match="must be an object"):
        load_golden_set(path)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "golden.json", [{"id": "a", "question": "Q?"}])
    with pytest.raises(GoldenSetError, match="reference_answer"):
        load_golden_set(path)


def test_duplicate_id_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "golden.json",
        [
            {"id": "dup", "question": "Q1?", "reference_answer": "A1"},
            {"id": "dup", "question": "Q2?", "reference_answer": "A2"},
        ],
    )
    with pytest.raises(GoldenSetError, match="duplicate item id"):
        load_golden_set(path)


def test_empty_set_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "golden.json", [])
    with pytest.raises(GoldenSetError, match="empty"):
        load_golden_set(path)


def test_seed_golden_set_is_valid() -> None:
    """The committed evals/golden.json loads and points at real source files."""
    items = load_golden_set(Path("evals/golden.json"))

    assert len(items) >= 5
    assert len({item.id for item in items}) == len(items)  # ids unique
    for item in items:
        for source in item.expected_sources or []:
            assert Path(source).exists(), f"{item.id}: missing source {source}"
