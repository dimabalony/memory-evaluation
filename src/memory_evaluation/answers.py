"""Canonical pure answer requests and lexical scoring; no backend dependency."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from memory_evaluation.eval.fixtures import ASSISTANT, HISTORY_WINDOW
from memory_evaluation.eval.matching import text_contains

ABSTENTION_MARKERS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "i don't have",
    "i do not have",
    "no information",
    "not sure",
    "no record",
    "can't find",
    "cannot find",
    "не знаю",
    "нет информации",
)
_VERDICT = re.compile(r"\bVERDICT:\s*(correct|incorrect|abstained)\b", re.IGNORECASE)


@dataclass(frozen=True)
class AnswerRequest:
    """Everything the host inference port is given for one question in one arm."""

    fixture_id: str
    question_id: str
    arm: str
    question: str
    asker: str | None
    history: tuple[dict[str, str], ...]
    memory_context: str
    retrieved_trace_ids: tuple[str, ...]


@dataclass(frozen=True)
class JudgeRequest:
    """What a judge sees: the question, the answer, and the labelled expectation."""

    question: str
    answer: str
    expected: dict[str, Any]


AnswerFn = Callable[[AnswerRequest], Awaitable[str | None]]
JudgeFn = Callable[[JudgeRequest], Awaitable[str | None]]


def parse_verdict(text: str | None) -> str | None:
    """``correct`` / ``incorrect`` / ``abstained`` from ``VERDICT: <x>``, or None when it
    is missing or ambiguous. None makes the row invalid; it is never read as incorrect."""
    if not isinstance(text, str):
        return None
    found = {m.lower() for m in _VERDICT.findall(text)}
    return found.pop() if len(found) == 1 else None


def is_abstention(answer: str) -> bool:
    # A marker followed by a factual guess is not a refusal. Deliberately accept
    # only a small, explicit grammar rather than guessing semantic abstention.
    folded = answer.replace("\u2019", "'").replace("\u02bc", "'").casefold().strip()
    return (
        re.fullmatch(
            r"(?:i (?:don't|do not) know(?: (?:that|this|the answer))?"
            r"|i (?:don't|do not) have (?:that information|information|a record)"
            r"|(?:i )?(?:can't|cannot) find (?:that|this|the answer)"
            r"|(?:i am |i'm )?not sure|no information|no record|не знаю|нет информации)[.!?\s]*",
            folded,
        )
        is not None
    )


def check_answer(question: dict[str, Any], answer: str) -> dict[str, Any]:
    """Deterministic grade of one answer against its labelled expectation."""
    if not isinstance(answer, str) or not answer.strip():
        return {"expected_hit": False, "prohibited_hits": [], "abstained": False, "verdict": "invalid"}
    folded = answer.casefold()
    abstained = is_abstention(answer)
    # Deliberately bounded lexical scopes: punctuation and a new coordinated
    # predicate separate assertions, while "not Leeds or York" stays together.
    segments = re.split(
        r"[.!?;,\n\uff0c\u3002\uff01\uff1f\uff1b]+|\s+(?:but|however)\s+|\s+and\s+(?=(?:does|do|did|has|have|is|are|was|were|will|can|must|should)\b)",
        answer.casefold(),
        flags=re.I,
    )
    negation = re.compile(
        r"\b(?:not|never|no|isn't|isn\u2019t|doesn't|doesn\u2019t|wasn't|wasn\u2019t|не|нет|non|pas|jamais|ne)\b"
        r"|\bn['\u2019](?=\w)",
        re.I,
    )

    def has_negation(text: str) -> bool:
        return bool(negation.search(text) or re.search(r"不|没|並非|并非|ではない|じゃない|ません|ない", text))

    def polarities(segment: str, value: str, *, allow_possessive: bool = True) -> list[bool]:
        """Negation before/inside a bounded match, plus explicit following denials."""
        result = []
        value = value.casefold()
        for match in re.finditer(re.escape(value), segment):
            context = segment[max(0, match.start() - 1) : min(len(segment), match.end() + 1)]
            if not text_contains(context, value):
                continue
            prefix = segment[: match.end()]
            suffix = segment[match.end() :]
            if not allow_possessive and re.match(r"^(?:['\u2019\u02bc]s\b|['\u2019\u02bc](?=\s|$))", suffix):
                continue
            following_denial = re.match(
                r"^\s*(?:(?:is|are|was|were)\s+(?:not|never|no)\b|(?:isn't|isn\u2019t|wasn't|wasn\u2019t)\b|ではない|じゃない|ません|ない)",
                suffix,
            )
            result.append(has_negation(prefix) or bool(following_denial))
        return result

    def affirmative(segment: str, value: str, *, allow_possessive: bool = True) -> bool:
        return any(not negative for negative in polarities(segment, value, allow_possessive=allow_possessive))

    prohibited = [
        p for p in question.get("prohibited") or [] if any(has_negation(p) in polarities(s, p) for s in segments)
    ]
    wrong_subjects = question.get("wrong_subjects") or []
    any_wrong_subject = any(affirmative(s, name) for s in segments for name in wrong_subjects)

    def supports(value: str, negative_expected: bool) -> bool:
        for segment in segments:
            if negative_expected not in polarities(segment, value):
                continue
            if any(affirmative(segment, name) for name in wrong_subjects):
                continue
            # An unnamed/pronominal answer refers to the question's subject unless
            # an explicitly wrong named subject is asserted elsewhere. This also
            # keeps comma-separated appositions from laundering a wrong subject:
            # an expected name used only as a possessive is not that assertion's subject.
            if (
                question.get("expect_subject")
                and not affirmative(segment, question["expect_subject"], allow_possessive=False)
                and any_wrong_subject
            ):
                continue
            return True
        return False

    if question.get("abstain"):
        expected_hit = None
        correct = abstained and not prohibited
    else:
        checks = []
        if question.get("expect_any"):
            checks.append(any(supports(e, has_negation(e)) for e in question["expect_any"]))
        if question.get("expect_all"):
            negative_all = any(has_negation(e) for e in question["expect_all"])
            checks.append(all(supports(e, negative_all) for e in question["expect_all"]))
        if question.get("expect_exact") is not None:
            checks.append(" ".join(folded.split()) == " ".join(str(question["expect_exact"]).casefold().split()))
        expected_hit = bool(checks) and all(checks)
        normalized_answer = answer.replace("\u2019", "'").replace("\u02bc", "'")
        uncertainty = any(text_contains(normalized_answer, marker) for marker in ABSTENTION_MARKERS)
        correct = expected_hit and not prohibited and not uncertainty
    return {
        "expected_hit": expected_hit,
        "prohibited_hits": prohibited,
        "abstained": abstained,
        "verdict": "correct" if correct else "incorrect",
        "evidence": {
            "expectation": _expectation(question),
            "method": "conservative lexical checks; not semantic correctness",
        },
    }


def _history(fixture: dict[str, Any], question: dict[str, Any]) -> tuple[dict[str, str], ...]:
    if question.get("history", "full") == "none":
        return ()
    participants = {p["key"]: p for p in fixture.get("participants") or []}
    window = fixture["messages"][-HISTORY_WINDOW:]
    return tuple(
        {
            "role": "assistant" if m["from"] == ASSISTANT else "user",
            "sender": "assistant" if m["from"] == ASSISTANT else participants[m["from"]]["name"],
            "text": m["text"],
        }
        for m in window
    )


def _expectation(question: dict[str, Any]) -> dict[str, Any]:
    keys = ("expect_any", "expect_all", "expect_exact", "abstain", "prohibited", "expect_subject", "wrong_subjects")
    return {k: question[k] for k in keys if k in question}
