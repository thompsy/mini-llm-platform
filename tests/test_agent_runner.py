"""Tests for the ReAct loop (run_agent): tool calls, recovery, tracing, limits."""

from app.agent.runner import run_agent
from app.llm.client import ChatResult
from app.tracing import Trace, _current_trace


class _ScriptedChat:
    """Returns a pre-scripted sequence of replies, one per chat() call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.call_count = 0

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        self.call_count += 1
        return ChatResult(content=self._replies.pop(0), model=model, output_tokens=None)


class _FakeTool:
    name = "rag_search"
    description = "search the docs"

    def __init__(self, result: str) -> None:
        self.result = result
        self.inputs: list[str] = []

    async def run(self, arg: str) -> str:
        self.inputs.append(arg)
        return self.result


async def test_agent_uses_tool_then_answers() -> None:
    tool = _FakeTool("Guido van Rossum. (sources: data/python.md)")
    chat = _ScriptedChat(
        [
            "Thought: search\nAction: rag_search\nAction Input: who created python",
            "Thought: got it\nFinal Answer: Guido van Rossum.",
        ]
    )

    result = await run_agent(
        "Who created Python?",
        chat_client=chat,
        model="m",
        tools={tool.name: tool},
        max_steps=5,
    )

    assert result.answer == "Guido van Rossum."
    assert result.stopped_reason == "final_answer"
    assert tool.inputs == ["who created python"]
    assert len(result.steps) == 1
    assert result.steps[0].action == "rag_search"


async def test_agent_handles_unknown_tool() -> None:
    chat = _ScriptedChat(
        [
            "Action: nonexistent\nAction Input: x",
            "Final Answer: done",
        ]
    )
    result = await run_agent("Q?", chat_client=chat, model="m", tools={}, max_steps=5)
    assert "Unknown tool 'nonexistent'" in result.steps[0].observation
    assert result.answer == "done"


async def test_agent_recovers_from_unparseable_reply() -> None:
    chat = _ScriptedChat(
        [
            "I have no idea what format to use",
            "Final Answer: recovered",
        ]
    )
    result = await run_agent("Q?", chat_client=chat, model="m", tools={}, max_steps=5)
    assert result.answer == "recovered"
    assert "could not parse" in result.steps[0].observation.lower()


async def test_agent_stops_at_max_steps() -> None:
    # Always emits an action, never a final answer.
    chat = _ScriptedChat(["Action: rag_search\nAction Input: loop"] * 3)
    tool = _FakeTool("still going")
    result = await run_agent(
        "Q?", chat_client=chat, model="m", tools={tool.name: tool}, max_steps=3
    )
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 3
    assert chat.call_count == 3


async def test_agent_records_spans_on_active_trace() -> None:
    tool = _FakeTool("answer")
    chat = _ScriptedChat(
        [
            "Action: rag_search\nAction Input: q",
            "Final Answer: done",
        ]
    )
    trace = Trace(route="/agent")
    token = _current_trace.set(trace)
    try:
        await run_agent(
            "Q?", chat_client=chat, model="m", tools={tool.name: tool}, max_steps=5
        )
    finally:
        _current_trace.reset(token)

    span_names = [s.name for s in trace.spans]
    assert "agent_step" in span_names
    assert "tool:rag_search" in span_names
