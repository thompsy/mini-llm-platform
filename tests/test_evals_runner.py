"""Tests for the eval runner (orchestration + scoring + tracing)."""

from pathlib import Path

from app.evals.dataset import GoldenItem
from app.evals.runner import run_evals
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
    """Returns a fixed RAG answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content=self.answer, model=model, output_tokens=3)


class _FakeJudge:
    """Returns a fixed verdict regardless of input."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content=self.verdict, model=model, output_tokens=1)


def _item() -> GoldenItem:
    return GoldenItem(
        id="py",
        question="Who created Python?",
        reference_answer="Guido van Rossum",
        expected_sources=["data/python.md"],
    )


async def _run_one(
    *,
    answer: str = "Python was created by Guido van Rossum.",
    verdict: str = "CORRECT",
    hits: list[Retrieved] | None = None,
    trace_store: SQLiteTraceStore | None = None,
):
    if hits is None:
        hits = [Retrieved(text="...", metadata={"source": "data/python.md"}, score=0.9)]
    return await run_evals(
        [_item()],
        embedder=_FakeEmbedder(),
        store=_FakeStore(hits),
        chat_client=_FakeChat(answer),
        judge_client=_FakeJudge(verdict),
        embed_model="e",
        chat_model="c",
        judge_model="j",
        temperature=0.0,
        top_k=4,
        min_score=0.5,
        trace_store=trace_store,
    )


async def test_run_evals_scores_all_three() -> None:
    report = await _run_one()

    assert len(report.results) == 1
    result = report.results[0]
    by = {s.scorer: s for s in result.scores}
    assert by["exact_match"].score == 1.0
    assert by["recall@k"].score == 1.0
    assert by["judge"].score == 1.0
    assert by["judge"].detail == "CORRECT"
    assert report.aggregates == {"exact_match": 1.0, "recall@k": 1.0, "judge": 1.0}


async def test_run_evals_reflects_bad_answer() -> None:
    report = await _run_one(answer="It was Linus Torvalds.", verdict="INCORRECT")

    by = {s.scorer: s.score for s in report.results[0].scores}
    assert by["exact_match"] == 0.0  # reference not contained
    assert by["recall@k"] == 1.0  # retrieval still correct
    assert by["judge"] == 0.0


async def test_run_evals_recall_misses_wrong_source() -> None:
    hits = [Retrieved(text="...", metadata={"source": "data/rag.md"}, score=0.9)]
    report = await _run_one(hits=hits)

    by = {s.scorer: s.score for s in report.results[0].scores}
    assert by["recall@k"] == 0.0  # expected data/python.md, retrieved data/rag.md


async def test_run_evals_persists_a_trace_per_item(tmp_path: Path) -> None:
    store = SQLiteTraceStore(path=str(tmp_path / "traces.db"))
    await _run_one(trace_store=store)

    summaries = store.list_traces(limit=10)
    store.close()

    assert len(summaries) == 1
    assert summaries[0].route == "eval:py"
    # embed_query, retrieve, chat (from the pipeline) + judge (from the runner)
    assert summaries[0].span_count == 4
