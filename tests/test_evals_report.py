"""Tests for eval report rendering, JSON round-trip, and baseline comparison."""

from pathlib import Path

from app.evals.report import (
    compare_to_baseline,
    read_report,
    render_console,
    write_report,
)
from app.evals.runner import EvalItemResult, EvalReport, ScoreResult


def _report(exact: float = 1.0, recall: float = 1.0, judge: float = 1.0) -> EvalReport:
    results = [
        EvalItemResult(
            id="py",
            question="Who created Python?",
            answer="Guido van Rossum",
            scores=[
                ScoreResult("exact_match", exact),
                ScoreResult("recall@k", recall),
                ScoreResult("judge", judge, detail="CORRECT"),
            ],
        )
    ]
    return EvalReport(
        results=results,
        aggregates={"exact_match": exact, "recall@k": recall, "judge": judge},
    )


def test_render_console_includes_ids_and_mean() -> None:
    rendered = render_console(_report())
    assert "py" in rendered
    assert "mean" in rendered
    assert "exact_match" in rendered


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_report(_report(exact=0.5, recall=1.0, judge=0.5), path)

    data = read_report(path)
    assert data["aggregates"] == {"exact_match": 0.5, "recall@k": 1.0, "judge": 0.5}
    assert data["results"][0]["id"] == "py"
    assert data["results"][0]["scores"][2]["detail"] == "CORRECT"


def test_compare_flags_regression() -> None:
    baseline = {"exact_match": 1.0, "recall@k": 1.0, "judge": 1.0}
    current = _report(exact=0.5, recall=1.0, judge=1.0)  # exact dropped 0.5

    regressions = compare_to_baseline(current, baseline, threshold=0.05)

    assert len(regressions) == 1
    assert regressions[0].scorer == "exact_match"
    assert regressions[0].baseline == 1.0
    assert regressions[0].current == 0.5


def test_compare_ignores_small_drop_within_threshold() -> None:
    baseline = {"exact_match": 1.0, "recall@k": 1.0, "judge": 1.0}
    current = _report(exact=0.97, recall=1.0, judge=1.0)  # within 0.05

    assert compare_to_baseline(current, baseline, threshold=0.05) == []


def test_compare_ignores_improvement_and_unknown_scorers() -> None:
    baseline = {"exact_match": 0.5}  # only one scorer present
    current = _report(exact=1.0, recall=1.0, judge=1.0)  # improved; others absent

    assert compare_to_baseline(current, baseline, threshold=0.05) == []
