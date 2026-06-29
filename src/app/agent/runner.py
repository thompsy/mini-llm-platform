"""The ReAct loop: drive the model through Thought → Action → Observation.

``run_agent`` repeatedly asks the model for the next step, executes the chosen
tool, feeds the observation back, and loops until the model gives a final answer
or a step cap is hit. Each iteration is an ``agent_step`` span and each tool call
a ``tool:<name>`` span, so a run is fully inspectable via the trace store.

Robustness for small models: an unparseable reply is answered with a format
reminder (costing a step) rather than crashing, and an unknown tool name becomes
an observation listing the valid tools — both let the model recover in-loop.
"""

import logging
from dataclasses import dataclass

from app.agent.react import AgentAction, FinalAnswer, build_agent_messages, parse_step
from app.agent.tools import Tool
from app.models import ChatMessage, Role
from app.rag.pipeline import ChatClient
from app.tracing import record_span

logger = logging.getLogger(__name__)

_FORMAT_REMINDER = (
    "I could not parse your reply. Respond using exactly:\n"
    "Thought: ...\nAction: <tool>\nAction Input: ...\n"
    "or, to finish:\nThought: ...\nFinal Answer: ..."
)

_MAX_STEPS_ANSWER = "I could not reach a final answer within the step limit."


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One iteration of the loop: a tool call (or a recovery nudge) + its result."""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass(frozen=True, slots=True)
class AgentResult:
    answer: str
    steps: list[AgentStep]
    stopped_reason: str  # "final_answer" | "max_steps"


async def run_agent(
    question: str,
    *,
    chat_client: ChatClient,
    model: str,
    tools: dict[str, Tool],
    max_steps: int = 5,
    temperature: float = 0.0,
) -> AgentResult:
    """Run the ReAct loop for one question and return the answer + step trace."""
    logger.info("agent: %r (max_steps=%d)", question, max_steps)
    messages = build_agent_messages(question, tools)
    steps: list[AgentStep] = []

    for _ in range(max_steps):
        with record_span("agent_step") as span:
            reply = await chat_client.chat(
                messages=messages, model=model, temperature=temperature
            )
            parsed = parse_step(reply.content)

            if isinstance(parsed, FinalAnswer):
                span.metadata["action"] = "final_answer"
                logger.info("agent: final answer after %d step(s)", len(steps))
                return AgentResult(
                    answer=parsed.answer, steps=steps, stopped_reason="final_answer"
                )

            if parsed is None:
                observation = _FORMAT_REMINDER
                thought, action, action_input = "", "", ""
                span.metadata["action"] = "unparseable"
            else:
                thought, action, action_input = (
                    parsed.thought,
                    parsed.tool,
                    parsed.tool_input,
                )
                observation = await _run_tool(parsed, tools)
                span.metadata["action"] = action

        steps.append(
            AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
        )
        _append_turn(messages, reply.content, observation)

    logger.info("agent: hit step limit (%d)", max_steps)
    return AgentResult(
        answer=_MAX_STEPS_ANSWER, steps=steps, stopped_reason="max_steps"
    )


async def _run_tool(action: AgentAction, tools: dict[str, Tool]) -> str:
    """Execute the chosen tool, or return a recoverable 'unknown tool' message."""
    tool = tools.get(action.tool)
    if tool is None:
        return f"Unknown tool '{action.tool}'. Available tools: {', '.join(tools)}."
    with record_span(f"tool:{action.tool}") as span:
        span.metadata["input"] = action.tool_input
        return await tool.run(action.tool_input)


def _append_turn(messages: list[ChatMessage], reply: str, observation: str) -> None:
    """Append the model's step and the resulting observation to the transcript."""
    messages.append(
        ChatMessage(role=Role.ASSISTANT, content=reply.strip() or "(no output)")
    )
    messages.append(ChatMessage(role=Role.USER, content=f"Observation: {observation}"))
