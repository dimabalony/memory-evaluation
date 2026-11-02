"""One comparison runner for independently supplied memory backends.

This report has a new identity. Historical SDK Layer A/B reports and their
matched no_memory arm remain available unchanged under memory_evaluation.eval.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any

from memory_evaluation.answers import (
    AnswerFn,
    AnswerRequest,
    JudgeFn,
    JudgeRequest,
    _history,
    check_answer,
    parse_verdict,
)
from memory_evaluation.contracts import Backend, RunConfig
from memory_evaluation.eval.fixtures import FixtureSet


def sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


async def evaluate(
    fixture_set: FixtureSet,
    backend: Backend,
    *,
    answer_fn: AnswerFn,
    config: RunConfig,
    judge_fn: JudgeFn | None = None,
) -> dict[str, Any]:
    """Record all requested rows; failures and unsupported instruments never score zero."""
    descriptor = backend.describe()
    if not descriptor.name or not descriptor.version:
        raise ValueError("backend name/version are required")
    if (judge_fn is None) != (config.judge is None):
        raise ValueError("judge callable and identity must be supplied together")
    if not config.source_revision or not config.answerer or datetime.fromisoformat(config.clock).tzinfo is None:
        raise ValueError("source revision, answerer identity and timezone-aware clock are required")
    if len({f["id"] for f in fixture_set.fixtures}) != len(fixture_set.fixtures):
        raise ValueError("duplicate fixture identity")
    rows = []
    for fixture in fixture_set.fixtures:
        questions = (fixture.get("expected") or {}).get("questions") or []
        if len({q["id"] for q in questions}) != len(questions):
            raise ValueError("duplicate question identity")
        pending = [{"fixture_id": fixture["id"], "question_id": q["id"], "status": "unrun"} for q in questions]
        if "read_to_answer" not in descriptor.capabilities:
            for row in pending:
                row.update(status="unmeasured", reason="backend does not support read_to_answer")
            rows.extend(pending)
            continue
        try:
            async with backend.open_fixture(json.loads(json.dumps(fixture)), config) as session:
                baseline = await session.state_receipt()
                if not isinstance(baseline, str) or not baseline:
                    raise ValueError("backend state receipt must be a nonempty string")
                participants = {p["key"]: p for p in fixture.get("participants") or []}
                for question, row in zip(questions, pending, strict=True):
                    try:
                        asker = participants.get(question.get("asker"))
                        request = AnswerRequest(
                            fixture["id"],
                            question["id"],
                            descriptor.name,
                            question["question"],
                            asker["name"] if asker else None,
                            _history(fixture, question),
                            "",
                            (),
                        )
                        input_hash = sha256([request.question, request.asker, list(request.history)])
                        retrieved = await session.retrieve(request, json.loads(json.dumps(question)))
                        if not isinstance(retrieved.context, str) or not all(
                            isinstance(i, str) for i in retrieved.trace_ids
                        ):
                            raise ValueError("malformed retrieval")
                        if await session.state_receipt() != baseline:
                            raise ValueError("retrieval mutated backend state")
                        request = replace(
                            request, memory_context=retrieved.context, retrieved_trace_ids=retrieved.trace_ids
                        )
                        answer = await answer_fn(request)
                        if input_hash != sha256([request.question, request.asker, list(request.history)]):
                            raise ValueError("answerer mutated common input")
                        if not isinstance(answer, str) or not answer.strip():
                            raise ValueError("empty or malformed answer")
                        grade = check_answer(question, answer)
                        if judge_fn:
                            verdict = parse_verdict(
                                await judge_fn(JudgeRequest(question["question"], answer, question))
                            )
                            if verdict is None:
                                raise ValueError("judge verdict did not parse")
                            grade["judge_verdict"] = verdict
                        row.update(
                            status="valid",
                            answer=answer,
                            **grade,
                            input_sha256=input_hash,
                            history_sha256=sha256(list(request.history)),
                            memory_sha256=sha256(retrieved.context),
                            retrieved=list(retrieved.trace_ids),
                            state_receipt_sha256=sha256(baseline),
                        )
                    except Exception as exc:
                        row.update(status="invalid", reason=type(exc).__name__)
                if await session.state_receipt() != baseline:
                    raise ValueError("backend state mutated during answers")
        except Exception as exc:
            for row in pending:
                if row["status"] != "unrun":
                    row.update(status="invalid", reason=type(exc).__name__)
                else:
                    row["reason"] = type(exc).__name__
        rows.extend(pending)
    valid = [r for r in rows if r["status"] == "valid"]
    return {
        "format": "memory-backend-comparison/v1",
        "backend": asdict(descriptor),
        "run": asdict(config),
        "fixture_set_sha256": fixture_set.sha256,
        "fixture_ids": [f["id"] for f in fixture_set.fixtures],
        "denominators": {s: sum(r["status"] == s for r in rows) for s in ("valid", "invalid", "unrun", "unmeasured")},
        "accuracy": sum(r["verdict"] == "correct" for r in valid) / len(valid) if valid else None,
        "write_readback": {
            "status": "unmeasured",
            "reason": "use the SDK lifecycle instrument"
            if "sdk_lifecycle" in descriptor.capabilities
            else "unsupported by backend",
        },
        "usage": {"provider_calls": None, "cost": None, "status": "unknown; supplied callables own usage accounting"},
        "rows": rows,
    }
