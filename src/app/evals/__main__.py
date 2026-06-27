"""CLI for the evaluation harness: run the golden set and report scores.

    uv run python -m app.evals                       # default evals/golden.json
    uv run python -m app.evals path/to/golden.json
    uv run python -m app.evals --output report.json  # save a report
    uv run python -m app.evals --baseline report.json  # flag regressions

Like ``app.rag.ingest``, this wires up real dependencies and runs once. It needs
Ollama running (both the chat and embed models) and an ingested corpus. Exit
codes: 0 ok, 1 regression vs baseline, 2 bad golden set.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.evals.dataset import GoldenSetError, load_golden_set
from app.evals.report import (
    compare_to_baseline,
    read_report,
    render_console,
    write_report,
)
from app.evals.runner import run_evals
from app.llm.client import OllamaClient
from app.llm.embeddings import OllamaEmbedder
from app.logging_config import setup_logging
from app.rag.store import ChromaStore
from app.tracing_store import SQLiteTraceStore

logger = logging.getLogger(__name__)


async def _run(golden_path: Path, output: Path | None, baseline: Path | None) -> int:
    settings = get_settings()
    items = load_golden_set(golden_path)

    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    client = OllamaClient(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    store = ChromaStore(path=settings.vector_store_dir)
    trace_store = SQLiteTraceStore(path=settings.trace_store_path)
    judge_model = settings.judge_model or settings.model

    try:
        report = await run_evals(
            items,
            embedder=embedder,
            store=store,
            chat_client=client,
            judge_client=client,
            embed_model=settings.embed_model,
            chat_model=settings.model,
            judge_model=judge_model,
            temperature=settings.default_temperature,
            top_k=settings.rag_top_k,
            min_score=settings.rag_min_score,
            trace_store=trace_store,
        )
    finally:
        await embedder.aclose()
        await client.aclose()
        trace_store.close()

    print(render_console(report))

    if output is not None:
        write_report(report, output)
        logger.info("wrote report to %s", output)

    if baseline is not None:
        baseline_aggregates = read_report(baseline).get("aggregates", {})
        regressions = compare_to_baseline(
            report, baseline_aggregates, settings.eval_regression_threshold
        )
        if regressions:
            print("\nREGRESSIONS:")
            for reg in regressions:
                print(f"  {reg.scorer}: {reg.baseline:.2f} -> {reg.current:.2f}")
            return 1
        print("\nNo regressions vs baseline.")

    return 0


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation harness.")
    parser.add_argument(
        "golden",
        nargs="?",
        default=Path("evals/golden.json"),
        type=Path,
        help="Golden set JSON file (default: evals/golden.json).",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the full report to this JSON file."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Compare against this report and flag regressions.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else get_settings().log_level)

    try:
        exit_code = asyncio.run(_run(args.golden, args.output, args.baseline))
    except GoldenSetError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    sys.exit(exit_code)


if __name__ == "__main__":
    _main()
