from pathlib import Path

from app.llm.client import ChatResult
from app.models import ChatMessage, Role
from app.rag.pipeline import (
    SYSTEM_PROMPT,
    Citation,
    _build_context,
    _to_citations,
    answer_question,
    build_messages,
)
from app.rag.store import ChromaStore, Retrieved


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, *, texts: list[str], model: str) -> list[list[float]]:
        self.calls += 1
        # One vector per text; value irrelevant since the store is seeded to match.
        return [[1.0, 0.0] for _ in texts]


class FakeChatClient:
    def __init__(self, content: str = "grounded answer [1]") -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    async def chat(
        self, *, messages: list[ChatMessage], model: str, temperature: float
    ) -> ChatResult:
        self.calls.append(
            {"messages": messages, "model": model, "temperature": temperature}
        )
        return ChatResult(content=self.content, model=model, output_tokens=9)


def _seeded_store(tmp_path: Path) -> ChromaStore:
    store = ChromaStore(path=str(tmp_path / "chroma"), collection_name="pipeline-test")
    store.add(
        ids=["doc.md:0", "doc.md:1", "other.md:0"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=["the cat sat", "the dog sat", "the stock rose"],
        metadatas=[
            {"source": "doc.md", "chunk": "0"},
            {"source": "doc.md", "chunk": "1"},
            {"source": "other.md", "chunk": "0"},
        ],
    )
    return store


# --- pure helpers -----------------------------------------------------------


def test_build_context_numbers_sources() -> None:
    retrieved = [
        Retrieved(text="first", metadata={}, score=1.0),
        Retrieved(text="second", metadata={}, score=0.5),
    ]
    assert _build_context(retrieved) == "[1] first\n\n[2] second"


def test_build_messages_has_grounding_and_question() -> None:
    retrieved = [Retrieved(text="ctx", metadata={"source": "a.md"}, score=1.0)]
    messages = build_messages("what?", retrieved)

    assert messages[0].role == Role.SYSTEM
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].role == Role.USER
    assert "[1] ctx" in messages[1].content
    assert "Question: what?" in messages[1].content


def test_to_citations_maps_index_source_and_truncates() -> None:
    long_text = "x" * 300
    retrieved = [
        Retrieved(text="short", metadata={"source": "a.md"}, score=0.9),
        Retrieved(text=long_text, metadata={"source": "b.md"}, score=0.4),
    ]
    citations = _to_citations(retrieved)

    assert citations[0] == Citation(index=1, source="a.md", score=0.9, snippet="short")
    assert citations[1].index == 2
    assert citations[1].source == "b.md"
    assert citations[1].snippet.endswith("...")
    assert len(citations[1].snippet) == 160 + 3  # snippet chars + ellipsis


# --- orchestrator -----------------------------------------------------------


async def test_answer_question_happy_path(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    chat = FakeChatClient(content="cats sit [1]")

    result = await answer_question(
        "what sat?",
        embedder=FakeEmbedder(),
        store=store,
        chat_client=chat,
        embed_model="embed",
        chat_model="chat-model",
        temperature=0.2,
        top_k=2,
        min_score=0.0,
    )

    assert result.answer == "cats sit [1]"
    assert result.output_tokens == 9
    assert len(result.citations) == 2
    # Nearest to [1,0] is "the cat sat" from doc.md.
    assert result.citations[0].source == "doc.md"
    # The chat client received the resolved model + temperature.
    assert chat.calls[0]["model"] == "chat-model"
    assert chat.calls[0]["temperature"] == 0.2


async def test_answer_question_passes_top_k(tmp_path: Path) -> None:
    store = _seeded_store(tmp_path)
    chat = FakeChatClient()

    result = await answer_question(
        "what sat?",
        embedder=FakeEmbedder(),
        store=store,
        chat_client=chat,
        embed_model="embed",
        chat_model="chat-model",
        temperature=0.0,
        top_k=1,
        min_score=0.0,
    )

    assert len(result.citations) == 1


async def test_answer_question_empty_store_message_skips_embed_and_llm(
    tmp_path: Path,
) -> None:
    empty_store = ChromaStore(path=str(tmp_path / "chroma"), collection_name="empty2")
    embedder = FakeEmbedder()
    chat = FakeChatClient()

    result = await answer_question(
        "anything?",
        embedder=embedder,
        store=empty_store,
        chat_client=chat,
        embed_model="embed",
        chat_model="chat-model",
        temperature=0.0,
        top_k=4,
        min_score=0.5,
    )

    assert result.citations == []
    assert "make ingest" in result.answer  # distinct empty-store message
    assert embedder.calls == 0  # short-circuits before embedding
    assert chat.calls == []  # and before the LLM
