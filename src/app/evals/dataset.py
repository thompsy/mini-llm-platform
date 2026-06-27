"""Load the golden Q&A set used by the evaluation harness (M4).

The golden set is a JSON array of question/answer items kept under version
control (``evals/golden.json`` by default). Each item is the ground truth one
eval run scores against: the question to ask, a reference answer, and optionally
the source files that *should* be retrieved (used for the recall@k metric).

Loading is deliberately strict — a malformed or duplicate-id golden set should
fail loudly at the start of a run, not produce a misleading score.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# Fields every golden item must define; the rest are optional.
_REQUIRED_FIELDS = ("id", "question", "reference_answer")


@dataclass(frozen=True, slots=True)
class GoldenItem:
    """One ground-truth Q&A the harness scores answers against."""

    id: str
    question: str
    reference_answer: str
    expected_sources: list[str] | None = None
    notes: str | None = None


class GoldenSetError(Exception):
    """Raised when the golden set file is missing or malformed."""


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Read and validate the golden set at ``path``.

    Raises :class:`GoldenSetError` if the file is missing, is not a JSON array,
    contains non-object items, is missing a required field, has duplicate ids,
    or is empty.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenSetError(f"golden set not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenSetError(f"golden set is not valid JSON ({path}): {exc}") from exc

    if not isinstance(raw, list):
        raise GoldenSetError(
            f"golden set must be a JSON array, got {type(raw).__name__}"
        )

    items: list[GoldenItem] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise GoldenSetError(
                f"item {index} must be an object, got {type(entry).__name__}"
            )
        missing = [field for field in _REQUIRED_FIELDS if field not in entry]
        if missing:
            raise GoldenSetError(
                f"item {index} is missing required field(s): {', '.join(missing)}"
            )
        item_id = str(entry["id"])
        if item_id in seen_ids:
            raise GoldenSetError(f"duplicate item id: {item_id!r}")
        seen_ids.add(item_id)
        items.append(
            GoldenItem(
                id=item_id,
                question=str(entry["question"]),
                reference_answer=str(entry["reference_answer"]),
                expected_sources=entry.get("expected_sources"),
                notes=entry.get("notes"),
            )
        )

    if not items:
        raise GoldenSetError(f"golden set is empty: {path}")

    return items
