"""Tests for the agent's tools: calculator, get_date, rag_search, registry."""

from datetime import date

from app.agent.tools import (
    CalculatorTool,
    GetDateTool,
    RagSearchTool,
    build_tools,
    safe_eval,
)
from app.llm.client import ChatResult
from app.rag.store import Retrieved


# --- calculator ---------------------------------------------------------------


def test_safe_eval_arithmetic_and_precedence() -> None:
    assert safe_eval("2 * (3 + 4)") == 14
    assert safe_eval("10 / 4") == 2.5
    assert safe_eval("2 ** 3") == 8
    assert safe_eval("-5 + 2") == -3


def test_safe_eval_rejects_code() -> None:
    for hostile in ["__import__('os')", "len([1,2])", "x + 1", "True + 1"]:
        try:
            safe_eval(hostile)
        except (ValueError, SyntaxError):
            continue
        raise AssertionError(f"expected {hostile!r} to be rejected")


async def test_calculator_tool_returns_clean_numbers() -> None:
    calc = CalculatorTool()
    assert await calc.run("2 * (3 + 4)") == "14"  # whole number, no .0
    assert await calc.run("10 / 4") == "2.5"


async def test_calculator_tool_reports_errors_without_raising() -> None:
    calc = CalculatorTool()
    assert (await calc.run("1 / 0")).startswith("Error:")
    assert (await calc.run("nonsense(")).startswith("Error:")


# --- get_date -----------------------------------------------------------------


async def test_get_date_tool() -> None:
    assert await GetDateTool().run("") == date.today().isoformat()


# --- rag_search ---------------------------------------------------------------


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
    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        return ChatResult(content="Guido van Rossum.", model=model, output_tokens=3)


def _rag_tool(hits: list[Retrieved]) -> RagSearchTool:
    return RagSearchTool(
        embedder=_FakeEmbedder(),
        store=_FakeStore(hits),
        chat_client=_FakeChat(),
        embed_model="e",
        chat_model="c",
        temperature=0.0,
        top_k=4,
        min_score=0.5,
    )


async def test_rag_search_includes_answer_and_sources() -> None:
    hits = [Retrieved(text="...", metadata={"source": "data/python.md"}, score=0.9)]
    observation = await _rag_tool(hits).run("Who created Python?")
    assert "Guido van Rossum." in observation
    assert "data/python.md" in observation


async def test_rag_search_no_citations_returns_plain_answer() -> None:
    observation = await _rag_tool([]).run("Who created Python?")
    assert "sources:" not in observation


# --- registry -----------------------------------------------------------------


def test_build_tools_registry() -> None:
    tools = build_tools(
        embedder=_FakeEmbedder(),
        store=_FakeStore([]),
        chat_client=_FakeChat(),
        embed_model="e",
        chat_model="c",
        temperature=0.0,
        top_k=4,
        min_score=0.5,
    )
    assert set(tools) == {"rag_search", "calculator", "get_date"}
    assert tools["calculator"].name == "calculator"
