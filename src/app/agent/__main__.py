"""CLI for the ReAct agent: ask a question and print its reasoning trace.

    uv run python -m app.agent "What is 2 + 2, and who created Python?"

Like the other CLIs, this wires up real dependencies and runs once. It needs
Ollama running and an ingested corpus (`make ingest`). The run is recorded as a
trace (route ``agent:cli``) and persisted, so it shows up in `make traces`.
"""

import argparse
import asyncio
import logging

from app.agent.runner import run_agent
from app.agent.tools import build_tools
from app.config import get_settings
from app.llm.client import OllamaClient
from app.llm.embeddings import OllamaEmbedder
from app.logging_config import setup_logging
from app.rag.store import ChromaStore
from app.tracing import start_trace
from app.tracing_store import SQLiteTraceStore

logger = logging.getLogger(__name__)


async def _run(question: str, max_steps: int) -> None:
    settings = get_settings()
    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    client = OllamaClient(
        base_url=settings.ollama_base_url, timeout=settings.request_timeout
    )
    store = ChromaStore(path=settings.vector_store_dir)
    trace_store = SQLiteTraceStore(path=settings.trace_store_path)
    tools = build_tools(
        embedder=embedder,
        store=store,
        chat_client=client,
        embed_model=settings.embed_model,
        chat_model=settings.model,
        temperature=settings.default_temperature,
        top_k=settings.rag_top_k,
        min_score=settings.rag_min_score,
    )

    try:
        with start_trace("agent:cli") as trace:
            result = await run_agent(
                question,
                chat_client=client,
                model=settings.model,
                tools=tools,
                max_steps=max_steps,
            )
        if trace.spans:
            trace_store.save(trace)
    finally:
        await embedder.aclose()
        await client.aclose()
        trace_store.close()

    for i, step in enumerate(result.steps, start=1):
        print(f"--- step {i} ---")
        if step.thought:
            print(f"Thought: {step.thought}")
        if step.action:
            print(f"Action: {step.action}({step.action_input})")
        print(f"Observation: {step.observation}")
    print(f"\nAnswer: {result.answer}")
    print(f"(stopped: {result.stopped_reason}, trace {trace.id})")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ask the ReAct agent a question.")
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override the step cap."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging("DEBUG" if args.verbose else settings.log_level)
    max_steps = args.max_steps or settings.agent_max_steps

    asyncio.run(_run(args.question, max_steps))


if __name__ == "__main__":
    _main()
