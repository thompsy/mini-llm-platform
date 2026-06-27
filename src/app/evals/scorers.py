"""Scoring functions for the evaluation harness (M4).

Three scorers, each returning a score in ``[0.0, 1.0]``:

- ``exact_match`` — deterministic, no model: is the (normalised) reference answer
  contained in the candidate? A cheap, strict baseline for factual questions.
- ``recall_at_k`` — retrieval quality: what fraction of the expected source files
  were actually retrieved? Separates retrieval quality from answer quality.
- ``judge_score`` — LLM-as-judge: a model grades the candidate against the
  reference, returning CORRECT / PARTIAL / INCORRECT -> 1.0 / 0.5 / 0.0.

Prompt construction and verdict parsing are pure functions so they can be
unit-tested without a model, mirroring ``app.rag.pipeline``.
"""

import re
import string

from app.models import ChatMessage, Role
from app.rag.pipeline import ChatClient

# Map the judge's one-word verdict to a score.
_VERDICT_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a candidate answer matches a reference answer to a "
    "question. Judge the meaning, not the wording. Respond with exactly one word: "
    "CORRECT if the candidate conveys the same key information as the reference, "
    "PARTIAL if it is partly correct or incomplete, or INCORRECT if it is wrong "
    "or unrelated."
)

# Translation table that drops ASCII punctuation, built once.
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for lenient matching."""
    return " ".join(text.lower().translate(_PUNCTUATION).split())


def exact_match(reference: str, candidate: str) -> float:
    """1.0 if the normalised reference is contained in the normalised candidate.

    Containment (not equality) so a correct fact embedded in a fuller sentence
    still counts — e.g. reference "Guido van Rossum" inside "Python was created
    by Guido van Rossum". Still strict: paraphrases of the reference won't match,
    which is what the LLM judge is for.
    """
    ref = _normalize(reference)
    if not ref:
        return 0.0
    return 1.0 if ref in _normalize(candidate) else 0.0


def recall_at_k(
    expected_sources: list[str] | None, retrieved_sources: list[str]
) -> float:
    """Fraction of expected source files that appear among the retrieved ones.

    Returns 1.0 vacuously when an item declares no expected sources (there is
    nothing to miss), so it never penalises items that opt out of this metric.
    """
    expected = set(expected_sources or [])
    if not expected:
        return 1.0
    retrieved = set(retrieved_sources)
    hits = sum(1 for source in expected if source in retrieved)
    return hits / len(expected)


def build_judge_messages(
    question: str, reference: str, candidate: str
) -> list[ChatMessage]:
    """Assemble the grader prompt (pure, no I/O)."""
    user_content = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Candidate answer: {candidate}\n\n"
        "Verdict (CORRECT, PARTIAL, or INCORRECT):"
    )
    return [
        ChatMessage(role=Role.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
        ChatMessage(role=Role.USER, content=user_content),
    ]


def parse_verdict(text: str) -> tuple[float, str]:
    """Map a judge reply to ``(score, verdict_label)``.

    Scans for the first whole word matching a known verdict. Matching whole
    words (not substrings) avoids "INCORRECT" being read as "CORRECT". An
    unparseable reply yields ``(0.0, "UNKNOWN")`` so a confused judge fails
    closed rather than silently scoring well.
    """
    for token in re.findall(r"[A-Z]+", text.upper()):
        if token in _VERDICT_SCORES:
            return _VERDICT_SCORES[token], token
    return 0.0, "UNKNOWN"


async def judge_score(
    question: str,
    reference: str,
    candidate: str,
    *,
    chat_client: ChatClient,
    model: str,
) -> tuple[float, str]:
    """Grade the candidate with the model; returns ``(score, verdict_label)``.

    Temperature is 0 so grading is as deterministic as the backend allows.
    """
    messages = build_judge_messages(question, reference, candidate)
    result = await chat_client.chat(messages=messages, model=model, temperature=0.0)
    return parse_verdict(result.content)
