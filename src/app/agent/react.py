"""The ReAct prompt and output parser (pure — no I/O, unit-testable).

The model is asked to emit one step at a time in a fixed text format:

    Thought: <reasoning>
    Action: <tool name>
    Action Input: <input to the tool>

…to which the runner replies with ``Observation: <result>``; or, to finish:

    Thought: <reasoning>
    Final Answer: <answer>

``parse_step`` turns a model reply into an ``AgentAction`` or ``FinalAnswer``.
Parsing is deliberately forgiving (case-insensitive, tolerant of extra prose,
ignores any ``Observation:`` the model hallucinates) because small models follow
the format imperfectly; an unparseable reply returns ``None`` so the runner can
nudge the model rather than crash.
"""

import re
from dataclasses import dataclass

from app.agent.tools import Tool
from app.models import ChatMessage, Role


@dataclass(frozen=True, slots=True)
class AgentAction:
    """A decision to call a tool."""

    thought: str
    tool: str
    tool_input: str


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """A decision to stop and answer."""

    thought: str
    answer: str


_SYSTEM_TEMPLATE = """\
You are an agent that answers the user's question by reasoning step by step and \
using tools.

You have access to the following tools:
{tools}

Use exactly this format, one step at a time:

Thought: your reasoning about what to do next
Action: the tool to use, exactly one of [{tool_names}]
Action Input: the input to the tool

You will then be shown:

Observation: the result of the tool

Repeat Thought/Action/Action Input as many times as needed. When you have enough \
information, respond with:

Thought: your reasoning
Final Answer: your answer to the question"""


def build_agent_messages(question: str, tools: dict[str, Tool]) -> list[ChatMessage]:
    """Build the initial ReAct conversation: system instructions + the question."""
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools.values())
    system = _SYSTEM_TEMPLATE.format(tools=tool_lines, tool_names=", ".join(tools))
    return [
        ChatMessage(role=Role.SYSTEM, content=system),
        ChatMessage(role=Role.USER, content=f"Question: {question}"),
    ]


def _extract_thought(text: str) -> str:
    match = re.search(
        r"thought:\s*(.*?)(?:\n\s*(?:action|final answer)\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _strip_quotes(text: str) -> str:
    """Remove one pair of wrapping quotes, e.g. "'2 + 4'" -> "2 + 4".

    Models commonly quote tool arguments as if calling a function. We only strip
    when the quote appears exactly twice (the two ends), so inputs with internal
    quotes are left untouched.
    """
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in "'\"`"
        and text.count(text[0]) == 2
    ):
        return text[1:-1]
    return text


def parse_step(text: str) -> AgentAction | FinalAnswer | None:
    """Parse a model reply into an action or a final answer; None if unparseable.

    ``Final Answer`` takes priority (it is terminal). Otherwise an ``Action`` +
    ``Action Input`` pair is required; the input runs until a hallucinated
    ``Observation:`` or end of text.
    """
    final = re.search(r"final answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if final:
        return FinalAnswer(
            thought=_extract_thought(text), answer=final.group(1).strip()
        )

    action = re.search(r"action:\s*(.+)", text, re.IGNORECASE)
    action_input = re.search(
        r"action input:\s*(.*?)(?:\n\s*observation\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if action and action_input:
        return AgentAction(
            thought=_extract_thought(text),
            tool=action.group(1).strip(),
            tool_input=_strip_quotes(action_input.group(1).strip()),
        )
    return None
