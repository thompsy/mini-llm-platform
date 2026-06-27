"""Render, persist, and compare eval reports.

Three concerns:

- ``render_console`` — a human-readable table of per-item scores plus the means.
- ``write_report`` / ``read_report`` — persist a report as JSON and read it back,
  so a run can serve as a baseline for a later one.
- ``compare_to_baseline`` — flag scorers whose mean dropped beyond a threshold,
  which the CLI turns into a non-zero exit (a CI gate against regressions).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.evals.runner import EvalReport

# Width of each score column in the console table.
_COL = 12


@dataclass(frozen=True, slots=True)
class Regression:
    scorer: str
    baseline: float
    current: float


def render_console(report: EvalReport) -> str:
    """Render the report as a fixed-width table: one row per item, then means."""
    scorers = list(report.aggregates)
    id_width = max([len("id"), *(len(r.id) for r in report.results)])

    def row(label: str, cells: list[str]) -> str:
        body = "  ".join(f"{c:>{_COL}}" for c in cells)
        return f"{label:<{id_width}}  {body}"

    header = row("id", scorers)
    lines = [header, "-" * len(header)]
    for result in report.results:
        by_scorer = {s.scorer: s.score for s in result.scores}
        cells = [f"{by_scorer.get(s, float('nan')):.2f}" for s in scorers]
        lines.append(row(result.id, cells))
    lines.append("-" * len(header))
    lines.append(row("mean", [f"{report.aggregates[s]:.2f}" for s in scorers]))
    return "\n".join(lines)


def write_report(report: EvalReport, path: Path) -> None:
    """Serialise the full report (results + aggregates) to JSON."""
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def read_report(path: Path) -> dict[str, Any]:
    """Load a previously written report as a plain dict (for baseline diffing)."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def compare_to_baseline(
    report: EvalReport, baseline_aggregates: dict[str, float], threshold: float
) -> list[Regression]:
    """Return scorers whose mean dropped more than ``threshold`` vs the baseline.

    Scorers absent from the baseline are skipped (nothing to compare against).
    """
    regressions: list[Regression] = []
    for scorer, current in report.aggregates.items():
        baseline = baseline_aggregates.get(scorer)
        if baseline is None:
            continue
        if current < baseline - threshold:
            regressions.append(
                Regression(scorer=scorer, baseline=baseline, current=current)
            )
    return regressions
