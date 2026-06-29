"""Tests for the ReAct prompt builder and output parser (pure)."""

from app.agent.react import (
    AgentAction,
    FinalAnswer,
    build_agent_messages,
    parse_step,
)
from app.agent.tools import CalculatorTool, GetDateTool
from app.models import Role


def _tools() -> dict:
    return {"calculator": CalculatorTool(), "get_date": GetDateTool()}


def test_build_agent_messages_lists_tools_and_question() -> None:
    messages = build_agent_messages("What is 2+2?", _tools())
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    system = messages[0].content
    assert "calculator:" in system
    assert "get_date:" in system
    assert "Action Input:" in system
    assert "Final Answer:" in system
    assert messages[1].content == "Question: What is 2+2?"


def test_parse_action() -> None:
    text = (
        "Thought: I should search.\nAction: rag_search\nAction Input: who made python"
    )
    parsed = parse_step(text)
    assert parsed == AgentAction(
        thought="I should search.", tool="rag_search", tool_input="who made python"
    )


def test_parse_final_answer() -> None:
    text = "Thought: I know this.\nFinal Answer: Guido van Rossum."
    assert parse_step(text) == FinalAnswer(
        thought="I know this.", answer="Guido van Rossum."
    )


def test_parse_ignores_hallucinated_observation() -> None:
    text = "Action: calculator\nAction Input: 2 + 2\nObservation: 4"
    parsed = parse_step(text)
    assert isinstance(parsed, AgentAction)
    assert parsed.tool_input == "2 + 2"  # stops before the hallucinated Observation


def test_parse_is_case_insensitive() -> None:
    parsed = parse_step("action: get_date\naction input: today")
    assert isinstance(parsed, AgentAction)
    assert parsed.tool == "get_date"


def test_parse_final_answer_takes_priority() -> None:
    # If both appear, Final Answer wins (it is terminal).
    text = "Action: calculator\nAction Input: 2+2\nFinal Answer: 4"
    assert parse_step(text) == FinalAnswer(thought="", answer="4")


def test_parse_strips_wrapping_quotes_from_input() -> None:
    # Models often quote tool args; the calculator must receive 2 + 4, not '2 + 4'.
    for quoted in ["'2 + 4'", '"2 + 4"', "`2 + 4`"]:
        parsed = parse_step(f"Action: calculator\nAction Input: {quoted}")
        assert isinstance(parsed, AgentAction)
        assert parsed.tool_input == "2 + 4"


def test_parse_keeps_internal_quotes() -> None:
    # Only wrapping pairs are stripped; internal quotes are preserved.
    parsed = parse_step('Action: rag_search\nAction Input: who said "hello"?')
    assert isinstance(parsed, AgentAction)
    assert parsed.tool_input == 'who said "hello"?'


def test_parse_unparseable_returns_none() -> None:
    assert parse_step("I'm not sure what to do here.") is None
