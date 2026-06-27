"""The eval orchestrator: run the golden set through RAG and score each answer.

For each golden item this runs the existing RAG pipeline, then applies the three
scorers (exact-match, recall@k, LLM-as-judge) and collects the results into an
:class:`EvalReport`. Dependencies are injected, mirroring ``answer_question``.

Each item runs inside its own trace (``eval:<id>``), so an eval run is
inspectable via the same trace store / ``make traces`` as live requests — a low
score comes with the latency and token spans that explain it.
"""

import logging
from dataclasses import dataclass

from app.evals.dataset import GoldenItem
from app.evals.scorers import exact_match, judge_score, recall_at_k
from app.rag.pipeline import ChatClient, Embedder, answer_question
from app.rag.store import VectorStore
from app.tracing import record_span, start_trace
from app.tracing_store import TraceStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    scorer: str
    score: float
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EvalItemResult:
    id: str
    question: str
    answer: str
    scores: list[ScoreResult]


@dataclass(frozen=True, slots=True)
class EvalReport:
    results: list[EvalItemResult]
    aggregates: dict[str, float]  # mean score per scorer


def _aggregate(results: list[EvalItemResult]) -> dict[str, float]:
    """Mean score per scorer across all items (insertion order preserved)."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for score in result.scores:
            sums[score.scorer] = sums.get(score.scorer, 0.0) + score.score
            counts[score.scorer] = counts.get(score.scorer, 0) + 1
    return {scorer: sums[scorer] / counts[scorer] for scorer in sums}


async def run_evals(
    items: list[GoldenItem],
    *,
    embedder: Embedder,
    store: VectorStore,
    chat_client: ChatClient,
    judge_client: ChatClient,
    embed_model: str,
    chat_model: str,
    judge_model: str,
    temperature: float,
    top_k: int,
    min_score: float,
    trace_store: TraceStore | None = None,
) -> EvalReport:
    """Run every golden item through RAG, score it, and build a report.

    ``chat_client``/``judge_client`` are separate so the judge can use a
    different model (or backend) than the system under test; the CLI passes the
    same client with ``judge_model`` differing.
    """
    logger.info("running %d eval item(s)", len(items))
    results: list[EvalItemResult] = []

    for item in items:
        with start_trace(f"eval:{item.id}") as trace:
            rag = await answer_question(
                item.question,
                embedder=embedder,
                store=store,
                chat_client=chat_client,
                embed_model=embed_model,
                chat_model=chat_model,
                temperature=temperature,
                top_k=top_k,
                min_score=min_score,
            )
            with record_span("judge") as span:
                judge_value, verdict = await judge_score(
                    item.question,
                    item.reference_answer,
                    rag.answer,
                    chat_client=judge_client,
                    model=judge_model,
                )
                span.metadata["verdict"] = verdict

        if trace_store is not None and trace.spans:
            trace_store.save(trace)

        sources = [c.source for c in rag.citations]
        scores = [
            ScoreResult("exact_match", exact_match(item.reference_answer, rag.answer)),
            ScoreResult("recall@k", recall_at_k(item.expected_sources, sources)),
            ScoreResult("judge", judge_value, detail=verdict),
        ]
        logger.info(
            "  %s: exact=%.0f recall=%.2f judge=%s",
            item.id,
            scores[0].score,
            scores[1].score,
            verdict,
        )
        results.append(
            EvalItemResult(
                id=item.id,
                question=item.question,
                answer=rag.answer,
                scores=scores,
            )
        )

    return EvalReport(results=results, aggregates=_aggregate(results))
