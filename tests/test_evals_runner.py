"""Tests for the eval runner (injected answer fn, scoring, tracing)."""

from pathlib import Path

from app.agent.tools import CalculatorTool
from app.evals.dataset import GoldenItem
from app.evals.runner import (
    AnswerOutput,
    agent_answer_fn,
    rag_answer_fn,
    run_evals,
)
from app.llm.client import ChatResult
from app.rag.store import Retrieved
from app.tracing_store import SQLiteTraceStore


class _FakeEmbedder:
    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def add(self, **kwargs: object) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def count(self) -> int:
        return len(self._hits)

    def query(self, *, embedding: list[float], top_k: int) -> list[Retrieved]:
        return self._hits


class _FakeChat:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content=self.answer, model=model, output_tokens=3)


class _FakeJudge:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content=self.verdict, model=model, output_tokens=1)


class _ScriptedChat:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content=self._replies.pop(0), model=model, output_tokens=None)


def _item() -> GoldenItem:
    return GoldenItem(
        id="py",
        question="Who created Python?",
        reference_answer="Guido van Rossum",
        expected_sources=["data/python.md"],
    )


def _rag_fn(answer: str, hits: list[Retrieved]):
    return rag_answer_fn(
        embedder=_FakeEmbedder(),
        store=_FakeStore(hits),
        chat_client=_FakeChat(answer),
        embed_model="e",
        chat_model="c",
        temperature=0.0,
        top_k=4,
        min_score=0.5,
    )


async def test_rag_mode_scores_all_three() -> None:
    hits = [Retrieved(text="...", metadata={"source": "data/python.md"}, score=0.9)]
    report = await run_evals(
        [_item()],
        answer_fn=_rag_fn("Python was created by Guido van Rossum.", hits),
        judge_client=_FakeJudge("CORRECT"),
        judge_model="j",
    )

    by = {s.scorer: s.score for s in report.results[0].scores}
    assert by == {"exact_match": 1.0, "recall@k": 1.0, "judge": 1.0}


async def test_rag_mode_recall_misses_wrong_source() -> None:
    hits = [Retrieved(text="...", metadata={"source": "data/rag.md"}, score=0.9)]
    report = await run_evals(
        [_item()],
        answer_fn=_rag_fn("Guido van Rossum", hits),
        judge_client=_FakeJudge("CORRECT"),
        judge_model="j",
    )
    by = {s.scorer: s.score for s in report.results[0].scores}
    assert by["recall@k"] == 0.0


async def test_agent_mode_skips_recall_and_records_steps() -> None:
    # Scripted agent: compute with calculator, then answer. No recall@k expected.
    answer_fn = agent_answer_fn(
        chat_client=_ScriptedChat(
            [
                "Thought: compute\nAction: calculator\nAction Input: 1 + 1",
                "Thought: done\nFinal Answer: Guido van Rossum",
            ]
        ),
        model="m",
        tools={"calculator": CalculatorTool()},
        max_steps=5,
    )
    report = await run_evals(
        [_item()],
        answer_fn=answer_fn,
        judge_client=_FakeJudge("CORRECT"),
        judge_model="j",
    )

    scorers = {s.scorer for s in report.results[0].scores}
    assert scorers == {"exact_match", "judge"}  # no recall@k in agent mode
    assert "recall@k" not in report.aggregates


async def test_persists_a_trace_per_item(tmp_path: Path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    hits = [Retrieved(text="...", metadata={"source": "data/python.md"}, score=0.9)]
    await run_evals(
        [_item()],
        answer_fn=_rag_fn("Guido van Rossum", hits),
        judge_client=_FakeJudge("CORRECT"),
        judge_model="j",
        trace_store=store,
    )

    summaries = store.list_traces(limit=10)
    store.close()

    assert len(summaries) == 1
    assert summaries[0].route == "eval:py"
    # embed_query, retrieve, chat (from rag_answer_fn) + judge (from run_evals)
    assert summaries[0].span_count == 4


def test_answer_output_defaults() -> None:
    out = AnswerOutput(answer="hi")
    assert out.sources is None
    assert out.steps is None
