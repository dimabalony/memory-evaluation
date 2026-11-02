"""Layer A: read to answer over the replayed memory (DB-840, DB-411).

Each fixture is first replayed exactly as Layer B replays it (same disposable
session, same recorded extraction), so the memory every question reads is the
memory the conversation wrote. Then, for every ``expected.questions`` entry:

1. each retrieval arm runs recognition with ``record_access=False`` and pinned
   weights: last-message uses the question; windowed uses production segmented
   user-history cues plus the question; no-memory makes no retrieval call;
2. the INJECTED answer function (the host's inference port) is asked the
   question once per arm. The arms differ only in the memory they get:
   ``memory`` gets the recognition output, ``no_memory`` gets none. History,
   question and asker are byte-identical across arms (hashes are recorded);
3. the returned answer itself is scored: deterministic contains/exact checks
   against the expected answer, prohibited-content checks, and abstention when
   the answer is not in memory. An optional judge function grades the same
   answer; a verdict that does not parse makes the row INVALID, never
   "incorrect". A missing or failed answer is also INVALID.

This layer measures nothing about model quality unless a live answer function
ran; with a fake answerer its scoreboard is ``wiring-only``. The answer prompt
here is a minimal stand-in, not the production prompt builder (DB-745).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.host_calls import utc_now
from kai_memory.retrieval.memory_priming import MemoryPrimingService
from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS
from kai_memory import ports
from memory_evaluation.eval.disposable_db import DisposableDB
from memory_evaluation.eval.extractors import LIVE_MODEL, CallBudget, RecordedExtractor
from memory_evaluation.eval.fixtures import ASSISTANT, HISTORY_WINDOW, FixtureSet
from memory_evaluation.eval.matching import text_contains
from memory_evaluation.eval.providers import evaluation_embeddings
from memory_evaluation.eval.runner import Extractor, _chat_context, _repo_relative, _run_fixture, store_fingerprint
from kai_memory.retrieval.iterative_retrieval import WindowCues, SegmentCue, build_segment_cues
from kai_memory.retrieval.memory_observation import MemoryObservation, MemoryDecision
from kai_memory.store.embeddings import generate_embedding

ANSWER_SCOREBOARD_FORMAT = "kai-memory-answer-scoreboard/v1"
ARMS: tuple[str, ...] = ("memory", "no_memory")
M4_ARMS: tuple[str, ...] = ("no_memory", "last_message", "windowed")
KNOWN_ARMS = frozenset((*ARMS, *M4_ARMS))
PAIR_HASHES = (
    "input_sha256",
    "history_sha256",
    "store_sha256",
    "stageC_policy_sha256",
    "model_sha256",
    "prompt_sha256",
)
# Messages of history an answer sees for a "full"-history question: the production
# conversation window. A "none"-history question is asked in a later, fresh chat.
DEFAULT_ANSWER_K = 8
# Case-insensitive phrases that mark an answer as an abstention.
from memory_evaluation.answers import (
    ABSTENTION_MARKERS, AnswerRequest, JudgeRequest, AnswerFn, JudgeFn,
    parse_verdict, is_abstention, check_answer, _history, _expectation,
)

def _describe(fn: Any, fallback: str) -> dict[str, Any]:
    describe = getattr(fn, "describe", None)
    return describe() if callable(describe) else {"name": fallback, "quality": "unknown"}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()



async def run_read_to_answer(
    fixture_set: FixtureSet,
    db: DisposableDB,
    *,
    answer_fn: AnswerFn,
    judge_fn: JudgeFn | None = None,
    extractor: Extractor | None = None,
    arms: tuple[str, ...] = ARMS,
    report_root: Path | None = None,
    fixture_clock: Callable[[dict[str, Any]], datetime] | None = None,
    embedding_adapter: Any = None,
    embedding_descriptor: dict[str, Any] | None = None,
    answer_factory: Callable[[AsyncSession, dict[str, Any]], AnswerFn] | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Replay each fixture, then ask its questions; return the raw report
    (score it with ``build_answer_scoreboard``; ``build_scoreboard`` reads its Layer B part)."""
    unknown = set(arms) - KNOWN_ARMS
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be nonempty and unique")
    if unknown:
        raise ValueError(f"unknown arms {sorted(unknown)}; known: {sorted(KNOWN_ARMS)}")
    extractor = extractor if extractor is not None else RecordedExtractor()

    paired = bool(set(arms) & {"last_message", "windowed"})
    model = _describe(answer_fn, "injected")
    model_identity = {key: model.get(key, "unknown") for key in ("name", "model", "temperature", "max_tokens")}
    policy = {"StageC": "disabled; one final answer, no tool/thinking retrieval", "record_access": False}

    async def ask(
        session: AsyncSession, fixture: dict[str, Any], result: dict[str, Any], last_sender: dict[str, Any] | None
    ) -> None:
        bound_answer = answer_factory(session, fixture) if answer_factory is not None else answer_fn
        participants = {p["key"]: p for p in fixture.get("participants") or []}
        rows: list[dict[str, Any]] = []
        result["answers"] = rows
        if fail_fast:
            expected_probes = {p["id"] for p in (fixture.get("expected") or {}).get("readback") or []}
            probes = result.get("readback") or []
            if (
                any(cp.get("store_errors") or cp.get("error") for cp in result.get("checkpoints") or [])
                or {p.get("id") for p in probes} != expected_probes
                or any(p.get("error") for p in probes)
            ):
                raise RuntimeError("fail-fast: fixture extraction or readback failed before answers")
        baseline = await store_fingerprint(session) if paired else None
        for question in (fixture.get("expected") or {}).get("questions") or []:
            asker = participants.get(question.get("asker") or "")
            chat_ctx = _chat_context(fixture, asker or last_sender)
            history = _history(fixture, question)
            for arm in arms:
                request = AnswerRequest(
                    fixture_id=fixture["id"],
                    question_id=question["id"],
                    arm=arm,
                    question=question["question"],
                    asker=asker["name"] if asker else None,
                    history=tuple(dict(m) for m in history),
                    memory_context="",
                    retrieved_trace_ids=(),
                )
                row: dict[str, Any] = {
                    "question_id": question["id"],
                    "arm": arm,
                    "history_sha256": _sha(list(history)),
                    "input_sha256": _sha([request.question, request.asker, list(history)]),
                    "memory_sha256": _sha(""),
                    "retrieved": [],
                    "answer": None,
                    "error": None,
                    "judge_raw": None,
                    "judge_error": None,
                    "availability": question.get("availability", "unknown"),
                    "distance_user_turns": question.get("distance_user_turns"),
                    "split": fixture.get("split", "unknown"),
                    "timing": {
                        "extraction_s": sum(cp.get("extraction_s", 0) for cp in result.get("checkpoints", [])),
                        "retrieval_s": None,
                        "projection_s": None,
                        "answer_s": None,
                        "failure_s": None,
                    },
                    "usage": {"input_tokens": None, "output_tokens": None, "status": "unknown"},
                }
                if paired:
                    row.update(
                        {
                            "paired_metadata": "m4/v1",
                            "store_sha256": _sha(baseline),
                            "stageC_policy_sha256": _sha(policy),
                            "model_sha256": _sha(model_identity)
                            if model_identity["model"] not in {None, "", "unknown"}
                            else None,
                            "prompt_sha256": _sha(ANSWER_PROMPT),
                            "stageC_policy": policy,
                        }
                    )
                arm_started = retrieval_started = time.monotonic()
                phase = "retrieval"
                projection = None
                try:
                    formatted = ""
                    ids: list[Any] = []
                    if arm != "no_memory":
                        decision = MemoryDecision(
                            MemoryObservation(model="eval", scope="fixture", input_ids=[]),
                            "eval",
                            "initial",
                            "recognition",
                            "question",
                            [],
                        )
                        priming = MemoryPrimingService(
                            session,
                            weights_override=DEFAULT_RETRIEVAL_WEIGHTS,
                            **({"observation": decision} if paired else {}),
                        )
                        kwargs: dict[str, Any] = {}
                        if arm == "windowed":
                            cues = await _window_cues(history, question["question"])
                            kwargs["window_cues"] = cues
                            row["window_cues"] = [{"text": cue.text, "weight": cue.weight} for cue in cues.segments]
                            row["is_windowed"] = cues.is_windowed
                        formatted, ids = await priming.prime(
                            question["question"],
                            limit=question.get("k", DEFAULT_ANSWER_K),
                            chat_ctx=chat_ctx,
                            record_access=False,
                            **kwargs,
                        )
                        if paired and decision.issues:
                            raise RuntimeError(f"retrieval incomplete: {decision.issues}")
                    row["timing"]["retrieval_s"] = time.monotonic() - retrieval_started
                    request = AnswerRequest(
                        **{
                            **request.__dict__,
                            "memory_context": formatted,
                            "retrieved_trace_ids": tuple(str(i) for i in ids),
                        }
                    )
                    row.update(
                        {
                            "memory_sha256": _sha(formatted),
                            "retrieved": list(request.retrieved_trace_ids),
                            "memory_context": formatted,
                            "history": copy.deepcopy(list(history)),
                            "prompt": render_answer_request(request),
                        }
                    )
                    if paired and await store_fingerprint(session) != baseline:
                        raise RuntimeError("store mutated during paired retrieval")
                    phase = "projection"
                    projection_started = time.monotonic()
                    evidence = getattr(bound_answer, "evidence", None)
                    if callable(evidence):
                        projection = evidence(request)
                        if inspect.isawaitable(projection):
                            projection = await projection
                        if not isinstance(projection, dict):
                            raise ValueError("answer projection evidence must be a dict")
                        row["projection_evidence"] = copy.deepcopy(projection)
                        if "common_context_sha256" in projection:
                            row["prompt_sha256"] = projection["common_context_sha256"]
                        for key in ("model_sha256", "stageC_policy_sha256"):
                            if key in projection:
                                row[key] = projection[key]
                    if paired and (
                        _sha(list(request.history)) != row["history_sha256"]
                        or _sha([request.question, request.asker, list(request.history)]) != row["input_sha256"]
                    ):
                        raise RuntimeError("paired model input mutated during projection")
                    row["timing"]["projection_s"] = time.monotonic() - projection_started
                    phase = "answer"
                    started = time.monotonic()
                    try:
                        row["answer"] = await bound_answer(request)
                    finally:
                        row["timing"]["answer_s"] = time.monotonic() - started
                    if paired and isinstance(projection, dict) and _sha(projection) != _sha(row["projection_evidence"]):
                        raise RuntimeError("paired projected payload mutated during transport")
                    if paired and (
                        _sha(list(request.history)) != row["history_sha256"]
                        or _sha([request.question, request.asker, list(request.history)]) != row["input_sha256"]
                    ):
                        raise RuntimeError("paired model input mutated during answer")
                except Exception as exc:
                    if row["timing"]["retrieval_s"] is None:
                        row["timing"]["retrieval_s"] = time.monotonic() - retrieval_started
                    if phase == "projection":
                        row["timing"]["projection_s"] = time.monotonic() - projection_started
                    row["timing"]["failure_s"] = time.monotonic() - arm_started
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                if paired:
                    after = await store_fingerprint(session)
                    row["store_after_sha256"] = _sha(after)
                    if after != baseline:
                        row["error"] = "store mutated during paired evaluation"
                if (
                    judge_fn is not None
                    and not row["error"]
                    and isinstance(row["answer"], str)
                    and row["answer"].strip()
                ):
                    try:
                        raw = await judge_fn(
                            JudgeRequest(question["question"], row["answer"], copy.deepcopy(_expectation(question)))
                        )
                        row["judge_raw"] = (raw or "")[:500]
                    except Exception as exc:
                        row["judge_error"] = f"{type(exc).__name__}: {exc}"[:300]
                rows.append(row)
                if fail_fast and (
                    row["error"]
                    or row["judge_error"]
                    or not isinstance(row["answer"], str)
                    or not row["answer"].strip()
                ):
                    raise RuntimeError("fail-fast: answer arm failed; further calls refused")
        if paired and await store_fingerprint(session) != baseline:
            for row in rows:
                row["error"] = "store mutated during paired evaluation"

    results: list[dict[str, Any]] = []
    with evaluation_embeddings(embedding_adapter, embedding_descriptor) as embedding_metadata:
        embedder = embedding_metadata["name"]
        for fixture in fixture_set.fixtures:
            now = (
                fixture_clock(fixture)
                if fixture_clock is not None
                else (
                    datetime.fromisoformat(fixture["start_at"])
                    + timedelta(minutes=max(m["minute"] for m in fixture["messages"]) + 1)
                    if paired
                    else None
                )
            )
            with ports.scoped({"clock": lambda now=now: now}, inherit=True) if now is not None else nullcontext():
                result = await _run_fixture(
                    fixture, db, extractor, then=ask, **({"fail_fast": True} if fail_fast else {})
                )
                results.append(result)
            if fail_fast and (
                result.get("status") != "ok"
                or any(cp.get("store_errors") or cp.get("error") for cp in result.get("checkpoints") or [])
                or any(a.get("error") or a.get("judge_error") for a in result.get("answers") or [])
                or any(p.get("error") for p in result.get("readback") or [])
            ):
                break
    return {
        "fixture_set": {
            "path": _repo_relative(fixture_set.path, report_root),
            "sha256": fixture_set.sha256,
            "n": len(fixture_set.fixtures),
        },
        "extractor": extractor.describe(),
        "answerer": _describe(answer_fn, "injected"),
        "judge": _describe(judge_fn, "injected") if judge_fn is not None else None,
        "arms": list(arms),
        "fail_fast": fail_fast,
        "stopped_early": fail_fast
        and (len(results) < len(fixture_set.fixtures) or any(r.get("status") != "ok" for r in results)),
        "embedder": embedder,
        "embedding_metadata": embedding_metadata,
        "results": results,
        "mutation_check": db.mutation_check(),
    }


def _invalid(fixture_id: str, question: dict[str, Any], arm: str, reason: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "question_id": question["id"],
        "arm": arm,
        "status": "invalid",
        "reason": reason,
        "expectation": _expectation(question),
        "axes": list(question.get("axes") or []),
    }


def _score_row(fixture_id: str, question: dict[str, Any], row: dict[str, Any], judged: bool) -> dict[str, Any]:
    if row.get("paired_metadata") and (
        row.get("paired_metadata") != "m4/v1"
        or not isinstance(row.get("store_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", row["store_sha256"]) is None
        or row["store_sha256"] != row.get("store_after_sha256")
    ):
        return _invalid(fixture_id, question, row["arm"], "missing or mismatched paired store receipt")
    if row.get("error"):
        return _invalid(fixture_id, question, row["arm"], f"answer failed: {row['error']}")
    if not isinstance(row.get("answer"), str):
        return _invalid(fixture_id, question, row["arm"], "malformed answer")
    if not row["answer"].strip():
        return _invalid(fixture_id, question, row["arm"], "empty answer")
    out = {
        "fixture_id": fixture_id,
        "question_id": question["id"],
        "arm": row["arm"],
        "status": "ok",
        "abstain_expected": bool(question.get("abstain")),
        "axes": list(question.get("axes") or []),
        "answer": row["answer"],
        "input_sha256": row.get("input_sha256"),
        "history_sha256": row.get("history_sha256"),
        "memory_sha256": row.get("memory_sha256"),
        "retrieved": row.get("retrieved", []),
        **check_answer(question, row["answer"]),
        "judge_verdict": None,
        **{
            k: row[k]
            for k in (
                "paired_metadata",
                *PAIR_HASHES,
                "timing",
                "usage",
                "availability",
                "distance_user_turns",
                "split",
                "store_after_sha256",
            )
            if k in row
        },
    }
    if judged:
        if row.get("judge_error"):
            return _invalid(fixture_id, question, row["arm"], f"judge failed: {row['judge_error']}")
        verdict = parse_verdict(row.get("judge_raw"))
        if verdict is None:
            return _invalid(fixture_id, question, row["arm"], "judge verdict did not parse")
        out["judge_verdict"] = verdict
        out["judge_correct"] = verdict == "correct" or (verdict == "abstained" and out["abstain_expected"])
    return out


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _arm_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["status"] == "ok"]
    answerable = [r for r in valid if not r["abstain_expected"]]
    abstain = [r for r in valid if r["abstain_expected"]]
    judged = [r for r in valid if r.get("judge_verdict")]
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "n_invalid": len(rows) - len(valid),
        "denominators": {
            "accuracy": len(valid),
            "answer_accuracy": len(answerable),
            "abstention_accuracy": len(abstain),
            "abstention_rate": len(valid),
            "judge_accuracy": len(judged),
            "judge_agreement": len(judged),
        },
        "accuracy": _ratio(sum(r["verdict"] == "correct" for r in valid), len(valid)),
        "answer_accuracy": _ratio(sum(r["verdict"] == "correct" for r in answerable), len(answerable)),
        "abstention_accuracy": _ratio(sum(r["verdict"] == "correct" for r in abstain), len(abstain)),
        "false_abstentions": sum(r["abstained"] for r in answerable),
        "abstention_rate": _ratio(sum(r["abstained"] for r in valid), len(valid)),
        "prohibited_rows": sum(bool(r["prohibited_hits"]) for r in valid),
        "judge_accuracy": _ratio(sum(r["judge_correct"] for r in judged), len(judged)),
        "judge_agreement": _ratio(sum(r["judge_correct"] == (r["verdict"] == "correct") for r in judged), len(judged)),
    }


ANSWER_GAPS: tuple[dict[str, str], ...] = (
    {
        "instrument": "production prompt path",
        "reason": "answers come from the injected answer function with a stand-in prompt, not Kai's prompt builder "
        "and provider projection (DB-745)",
    },
    {"instrument": "windowed retrieval arm", "reason": "only no-memory and last-message recognition arms run"},
    {
        "instrument": "judge calibration",
        "reason": "any judge here is uncalibrated; kappa >= 0.6 is required before gating",
    },
    {"instrument": "live answer baseline", "reason": "not run; needs operator authorization and explicit caps"},
)


def build_answer_scoreboard(
    report: dict[str, Any], fixture_set: FixtureSet, *, source_sha: str | None = None
) -> dict[str, Any]:
    """The Layer A scoreboard for one ``run_read_to_answer`` report."""
    arms = report.get("arms")
    if not isinstance(arms, (list, tuple)) or not arms or len(set(arms)) != len(arms) or set(arms) - KNOWN_ARMS:
        raise ValueError("arms must be nonempty, unique and known")
    if len({r["id"] for r in report["results"]}) != len(report["results"]):
        raise ValueError("duplicate fixture results")
    judged = report.get("judge") is not None
    rows: list[dict[str, Any]] = []
    by_id = {r["id"]: r for r in report["results"]}
    if set(by_id) - {f["id"] for f in fixture_set.fixtures}:
        raise ValueError("unknown fixture results")
    for fixture in fixture_set.fixtures:
        result = by_id.get(fixture["id"], {"status": "error", "error": "not run"})
        questions = (fixture.get("expected") or {}).get("questions") or []
        raw_answers = result.get("answers") or []
        answers = {(a["question_id"], a["arm"]): a for a in raw_answers}
        if len(answers) != len(raw_answers):
            raise ValueError("duplicate answer rows")
        known_questions = {q["id"] for q in questions}
        if any(q not in known_questions or arm not in arms for q, arm in answers):
            raise ValueError("unmatched question or arm")
        for question in questions:
            for arm in report["arms"]:
                store_failures = [e for cp in result.get("checkpoints") or [] for e in cp.get("store_errors") or []]
                if store_failures:
                    rows.append(_invalid(fixture["id"], question, arm, "checkpoint store errors"))
                elif result["status"] != "ok":
                    rows.append(_invalid(fixture["id"], question, arm, f"fixture errored: {result.get('error')}"))
                elif (question["id"], arm) not in answers:
                    rows.append(_invalid(fixture["id"], question, arm, "not asked"))
                else:
                    rows.append(_score_row(fixture["id"], question, answers[(question["id"], arm)], judged))

    lookup = {
        (f["id"], q["id"]): (f, q)
        for f in fixture_set.fixtures
        for q in (f.get("expected") or {}).get("questions") or []
    }
    for scored in rows:
        f, q = lookup[(scored["fixture_id"], scored["question_id"])]
        scored.update(
            {
                "availability": q.get("availability", "unknown"),
                "distance_user_turns": q.get("distance_user_turns"),
                "split": f.get("split", "unknown"),
            }
        )
    per_arm = {arm: _arm_aggregate([r for r in rows if r["arm"] == arm]) for arm in report["arms"]}
    benefit = None
    if {"memory", "no_memory"} <= set(report["arms"]):
        keyed = {(r["fixture_id"], r["question_id"], r["arm"]): r for r in rows if r["status"] == "ok"}
        pairs = [
            (keyed[(f, q, "memory")], keyed[(f, q, "no_memory")])
            for (f, q, arm) in keyed
            if arm == "memory" and (f, q, "no_memory") in keyed
        ]

        def comparable(pair: tuple[dict[str, Any], dict[str, Any]]) -> bool:
            return comparable_pair(*pair)

        invalid_pairs = sum(not comparable(pair) for pair in pairs)
        pairs = [pair for pair in pairs if comparable(pair)]
        benefit = {
            "paired_questions": len(pairs),
            "memory_correct": sum(m["verdict"] == "correct" for m, _ in pairs),
            "no_memory_correct": sum(n["verdict"] == "correct" for _, n in pairs),
        }
    if benefit is not None and invalid_pairs:
        benefit["invalid_pairs"] = invalid_pairs
    axes: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["status"] == "ok" and r["arm"] == "memory":
            for axis in r["axes"]:
                tally = axes.setdefault(axis, {"n": 0, "correct": 0})
                tally["n"] += 1
                tally["correct"] += r["verdict"] == "correct"

    from memory_evaluation.eval.diagnostics import paired_comparisons

    comparisons = paired_comparisons(rows, list(arms))
    answerer = report["answerer"]
    return {
        "header": {
            "format": ANSWER_SCOREBOARD_FORMAT,
            "generated_at": utc_now().isoformat(),
            "source_sha": source_sha or "unknown",
            "fixture_set": report["fixture_set"],
            "fixture_provenance": [
                {
                    "id": f["id"],
                    "source": f["_provenance"]["source"],
                    "content_sha256": f["_provenance"]["content_sha256"],
                }
                for f in fixture_set.fixtures
            ],
            "memory_written_by": report["extractor"],
            "answerer": answerer,
            "judge": report.get("judge"),
            "quality": answerer.get("quality", "unknown"),
            "arms": report["arms"],
            "history_window": HISTORY_WINDOW,
            "abstention_markers": list(ABSTENTION_MARKERS),
            "embedder": report["embedder"],
            "retrieval_weights": "code defaults (DEFAULT_RETRIEVAL_WEIGHTS, pinned)",
        },
        "mutation_check": report["mutation_check"],
        "aggregate": {
            "per_arm": per_arm,
            "memory_benefit": benefit,
            "by_axis_memory_arm": axes,
            "paired_comparisons": comparisons,
        },
        "rows": rows,
        "gaps": [
            dict(g) for g in ANSWER_GAPS if not (g["instrument"] == "windowed retrieval arm" and "windowed" in arms)
        ],
    }


ANSWER_PROMPT = """\
You are Kai, a personal assistant with long-term memory. Answer the question below.

Conversation so far (may be empty):
{history}

What you remember (may be incomplete or empty):
{memory}

{asker} asks: {question}

Answer in one or two sentences. Use only the conversation and what you remember. \
If neither tells you, say "I don't know"."""


class LiveAnswerer:
    """Answers through the host inference port under a call cap and a dollar cap.

    Only ``scripts/run_memory_eval.py --layer a --live-answers`` constructs one,
    on demand. It is never run in CI or tests."""

    def __init__(
        self,
        model: str,
        *,
        max_calls: int,
        max_usd: float,
        usd_per_mtok: tuple[float, float] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens if max_tokens is not None else (400 if model == LIVE_MODEL else 300)
        self.budget = CallBudget.for_model(model, max_calls=max_calls, max_usd=max_usd, usd_per_mtok=usd_per_mtok)

    def describe(self) -> dict[str, Any]:
        return {"name": "live", "model": self.model, "quality": "model-and-memory", **self.budget.describe()}

    async def __call__(self, request: AnswerRequest) -> str | None:
        prompt = render_answer_request(request)
        reserved = self.budget.reserve(len(prompt), self.max_tokens)
        provider = ports.get_provider_for_model(ports.get_settings(), self.model)
        kwargs = {"thinking_mode": "disabled"} if self.model == LIVE_MODEL else {}
        answer = await ports.call_llm_text(provider, prompt, max_tokens=self.max_tokens, **kwargs)
        self.budget.spend(len(prompt), len(answer or ""), reserved=reserved)
        return cast("str | None", answer)


def render_answer_request(request: AnswerRequest) -> str:
    return ANSWER_PROMPT.format(
        history="\n".join(f"[{m['sender']}]: {m['text']}" for m in request.history) or "(none)",
        memory=request.memory_context or "(nothing)",
        asker=request.asker or "The user",
        question=request.question,
    )


async def _window_cues(history: tuple[dict[str, str], ...], question: str) -> WindowCues:
    """Use production segmentation directly, without flag caches or fail-open fallback."""
    from kai_memory.retrieval.memory_priming import _derive_embedding_query

    texts = [m["text"] for m in history if m["role"] == "user"] + [question]
    raw = build_segment_cues(texts) if len(texts) > 1 else [(_derive_embedding_query(question), 1.0)]
    segments = [SegmentCue(text, weight, await generate_embedding(text)) for text, weight in raw]
    return WindowCues(segments=segments, is_windowed=len(texts) > 1)


def comparable_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    m4 = left.get("paired_metadata") or right.get("paired_metadata")
    if m4 and (
        left.get("paired_metadata") != "m4/v1"
        or right.get("paired_metadata") != "m4/v1"
        or any(row.get("store_sha256") != row.get("store_after_sha256") for row in (left, right))
    ):
        return False
    keys = PAIR_HASHES if m4 else PAIR_HASHES[:2]
    return all(
        isinstance(left.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", left[k]) is not None and left[k] == right.get(k)
        for k in keys
    )
