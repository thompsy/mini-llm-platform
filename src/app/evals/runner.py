"""The eval orchestrator: run the golden set through an answer producer and score.

The thing that produces an answer is injected as an ``AnswerFn`` — so the same
harness can score the plain RAG pipeline *or* the ReAct agent (or anything else).
An ``AnswerFn`` returns the answer plus whatever is needed for the mode-specific
scorers: ``sources`` (for recall@k, RAG mode) and/or ``steps`` (agent mode).

Scores applied:

- ``exact_match`` and ``judge`` — always (answer quality).
- ``recall@k`` — only when ``sources`` is provided (it measures the retrieval
  step, which only exists as a discrete step in the RAG pipeline; in agent mode
  retrieval is the agent's own decision, so it is omitted).

Each item runs inside its own trace (``eval:<id>``); the injected producer's own
spans (RAG's embed/retrieve/chat, or the agent's steps) nest inside it.
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.agent.runner import run_agent
from app.agent.tools import Tool
from app.evals.dataset import GoldenItem
from app.evals.scorers import exact_match, judge_score, recall_at_k
from app.rag.pipeline import ChatClient, Embedder, answer_question
from app.rag.store import VectorStore
from app.tracing import record_span, start_trace
from app.tracing_store import TraceStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerOutput:
    """What an answer producer returns for one question."""

    answer: str
    sources: list[str] | None = None  # provided => recall@k is scored (RAG mode)
    steps: int | None = None  # agent step count, logged as an operational signal


# An answer producer: question -> answer (+ scoring inputs).
AnswerFn = Callable[[str], Awaitable[AnswerOutput]]


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
    answer_fn: AnswerFn,
    judge_client: ChatClient,
    judge_model: str,
    trace_store: TraceStore | None = None,
) -> EvalReport:
    """Run every golden item through ``answer_fn``, score it, and build a report."""
    logger.info("running %d eval item(s)", len(items))
    results: list[EvalItemResult] = []

    for item in items:
        with start_trace(f"eval:{item.id}") as trace:
            output = await answer_fn(item.question)
            with record_span("judge") as span:
                judge_value, verdict = await judge_score(
                    item.question,
                    item.reference_answer,
                    output.answer,
                    chat_client=judge_client,
                    model=judge_model,
                )
                span.metadata["verdict"] = verdict

        if trace_store is not None and trace.spans:
            trace_store.save(trace)

        scores = [
            ScoreResult(
                "exact_match", exact_match(item.reference_answer, output.answer)
            )
        ]
        if output.sources is not None:
            scores.append(
                ScoreResult(
                    "recall@k", recall_at_k(item.expected_sources, output.sources)
                )
            )
        scores.append(ScoreResult("judge", judge_value, detail=verdict))

        steps_note = f", steps={output.steps}" if output.steps is not None else ""
        logger.info(
            "  %s: exact=%.0f judge=%s%s",
            item.id,
            scores[0].score,
            verdict,
            steps_note,
        )
        results.append(
            EvalItemResult(
                id=item.id,
                question=item.question,
                answer=output.answer,
                scores=scores,
            )
        )

    return EvalReport(results=results, aggregates=_aggregate(results))


def rag_answer_fn(
    *,
    embedder: Embedder,
    store: VectorStore,
    chat_client: ChatClient,
    embed_model: str,
    chat_model: str,
    temperature: float,
    top_k: int,
    min_score: float,
) -> AnswerFn:
    """An answer producer that runs the plain RAG pipeline (reports sources)."""

    async def answer(question: str) -> AnswerOutput:
        rag = await answer_question(
            question,
            embedder=embedder,
            store=store,
            chat_client=chat_client,
            embed_model=embed_model,
            chat_model=chat_model,
            temperature=temperature,
            top_k=top_k,
            min_score=min_score,
        )
        return AnswerOutput(
            answer=rag.answer, sources=[c.source for c in rag.citations]
        )

    return answer


def agent_answer_fn(
    *,
    chat_client: ChatClient,
    model: str,
    tools: dict[str, Tool],
    max_steps: int,
    temperature: float = 0.0,
) -> AnswerFn:
    """An answer producer that runs the ReAct agent (reports step count)."""

    async def answer(question: str) -> AnswerOutput:
        result = await run_agent(
            question,
            chat_client=chat_client,
            model=model,
            tools=tools,
            max_steps=max_steps,
            temperature=temperature,
        )
        return AnswerOutput(answer=result.answer, sources=None, steps=len(result.steps))

    return answer
