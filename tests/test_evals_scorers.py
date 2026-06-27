"""Tests for the evaluation scorers: exact-match, recall@k, and the LLM judge."""

from app.evals.scorers import (
    build_judge_messages,
    exact_match,
    judge_score,
    parse_verdict,
    recall_at_k,
)
from app.llm.client import ChatResult
from app.models import Role


# --- exact_match ---------------------------------------------------------------


def test_exact_match_contained_ignoring_case_and_punctuation() -> None:
    assert (
        exact_match("Guido van Rossum", "Python was created by guido van rossum.")
        == 1.0
    )


def test_exact_match_identical() -> None:
    assert exact_match("1991", "1991") == 1.0


def test_exact_match_no_match() -> None:
    assert exact_match("Guido van Rossum", "It was Linus Torvalds.") == 0.0


def test_exact_match_empty_reference_is_zero() -> None:
    assert exact_match("", "anything") == 0.0


# --- recall_at_k ---------------------------------------------------------------


def test_recall_full_hit() -> None:
    assert recall_at_k(["data/python.md"], ["data/python.md", "data/rag.md"]) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(["a.md", "b.md"], ["a.md", "z.md"]) == 0.5


def test_recall_miss() -> None:
    assert recall_at_k(["a.md"], ["b.md"]) == 0.0


def test_recall_no_expected_is_vacuously_one() -> None:
    assert recall_at_k(None, ["a.md"]) == 1.0
    assert recall_at_k([], ["a.md"]) == 1.0


def test_recall_dedupes_expected() -> None:
    assert recall_at_k(["a.md", "a.md"], ["a.md"]) == 1.0


# --- parse_verdict -------------------------------------------------------------


def test_parse_verdict_correct() -> None:
    assert parse_verdict("CORRECT") == (1.0, "CORRECT")


def test_parse_verdict_partial() -> None:
    assert parse_verdict("PARTIAL") == (0.5, "PARTIAL")


def test_parse_verdict_incorrect_not_read_as_correct() -> None:
    # "CORRECT" is a substring of "INCORRECT"; whole-word matching must win.
    assert parse_verdict("INCORRECT") == (0.0, "INCORRECT")


def test_parse_verdict_within_a_sentence() -> None:
    assert parse_verdict("The answer is correct.") == (1.0, "CORRECT")


def test_parse_verdict_unparseable_fails_closed() -> None:
    assert parse_verdict("I'm not sure about this one") == (0.0, "UNKNOWN")


# --- build_judge_messages ------------------------------------------------------


def test_build_judge_messages_shape() -> None:
    messages = build_judge_messages("Q?", "ref", "cand")
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    user = messages[1].content
    assert "Q?" in user
    assert "ref" in user
    assert "cand" in user


# --- judge_score ---------------------------------------------------------------


class _FakeJudge:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, object]] = []

    async def chat(
        self, *, messages: object, model: str, temperature: float
    ) -> ChatResult:
        self.calls.append({"model": model, "temperature": temperature})
        return ChatResult(content=self.reply, model=model, output_tokens=None)


async def test_judge_score_maps_verdict() -> None:
    judge = _FakeJudge("PARTIAL — it's incomplete")
    score, label = await judge_score(
        "Q?", "ref", "cand", chat_client=judge, model="judge-model"
    )
    assert (score, label) == (0.5, "PARTIAL")
    assert judge.calls[0] == {"model": "judge-model", "temperature": 0.0}


async def test_judge_score_malformed_reply_fails_closed() -> None:
    judge = _FakeJudge("hmm, hard to say")
    score, label = await judge_score("Q?", "ref", "cand", chat_client=judge, model="m")
    assert (score, label) == (0.0, "UNKNOWN")
