"""Tools the ReAct agent can call (M5).

A tool is anything with a ``name``, a ``description`` (shown to the model so it
knows when to use it), and an async ``run(arg)`` that takes the action input
string and returns an observation string. The agent depends on the ``Tool``
Protocol, not concrete tools, so the set is easy to extend or swap.

Three general-purpose tools to start:

- ``rag_search`` — the existing retrieval + grounding pipeline (the main tool).
- ``calculator`` — safe arithmetic (AST-parsed, never ``eval``).
- ``get_date`` — today's date.
"""

import ast
import operator
from datetime import date
from typing import Callable, Protocol

from app.rag.pipeline import ChatClient, Embedder, answer_question
from app.rag.store import VectorStore


class Tool(Protocol):
    name: str
    description: str

    async def run(self, arg: str) -> str: ...


# --- calculator ---------------------------------------------------------------

# Only these node/operator types are allowed; everything else (names, calls,
# attribute access, …) is rejected, so the calculator can't execute code.
_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        # bool is a subclass of int; exclude it so "True + 1" isn't arithmetic.
        if isinstance(node.value, bool):
            raise ValueError("booleans are not numbers")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def safe_eval(expression: str) -> float:
    """Evaluate a basic arithmetic expression without executing arbitrary code."""
    return _eval_node(ast.parse(expression, mode="eval").body)


class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'. "
        "Input: the expression to evaluate."
    )

    async def run(self, arg: str) -> str:
        try:
            result = safe_eval(arg)
        except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
            return f"Error: {exc}"
        # Render whole numbers without a trailing .0 for cleaner observations.
        return str(int(result)) if result == int(result) else str(result)


# --- get_date -----------------------------------------------------------------


class GetDateTool:
    name = "get_date"
    description = "Get today's date in YYYY-MM-DD format. Input: ignored."

    async def run(self, arg: str) -> str:
        return date.today().isoformat()


# --- rag_search ---------------------------------------------------------------


class RagSearchTool:
    name = "rag_search"
    description = (
        "Search the document corpus and return a grounded answer with its "
        "sources. Input: a natural-language search query or question."
    )

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        chat_client: ChatClient,
        embed_model: str,
        chat_model: str,
        temperature: float,
        top_k: int,
        min_score: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chat_client = chat_client
        self._embed_model = embed_model
        self._chat_model = chat_model
        self._temperature = temperature
        self._top_k = top_k
        self._min_score = min_score

    async def run(self, arg: str) -> str:
        result = await answer_question(
            arg,
            embedder=self._embedder,
            store=self._store,
            chat_client=self._chat_client,
            embed_model=self._embed_model,
            chat_model=self._chat_model,
            temperature=self._temperature,
            top_k=self._top_k,
            min_score=self._min_score,
        )
        if not result.citations:
            return result.answer
        sources = ", ".join(sorted({c.source for c in result.citations}))
        return f"{result.answer}\n(sources: {sources})"


def build_tools(
    *,
    embedder: Embedder,
    store: VectorStore,
    chat_client: ChatClient,
    embed_model: str,
    chat_model: str,
    temperature: float,
    top_k: int,
    min_score: float,
) -> dict[str, Tool]:
    """Build the default tool registry, keyed by tool name."""
    tools: list[Tool] = [
        RagSearchTool(
            embedder=embedder,
            store=store,
            chat_client=chat_client,
            embed_model=embed_model,
            chat_model=chat_model,
            temperature=temperature,
            top_k=top_k,
            min_score=min_score,
        ),
        CalculatorTool(),
        GetDateTool(),
    ]
    return {tool.name: tool for tool in tools}
