"""Layer B: replay fixtures through the real persistence and L2 path, then read back (DB-840).

Per fixture, inside one ``DisposableDB.session()`` (always rolled back):

1. seed ``pre_state`` (known participants, labelled traces with their entities);
2. ``ConversationManager.save_message`` every message, stamped with the fixture's
   clock so order and rendered timestamps are deterministic;
3. at each checkpoint run ``ExtractionService.extract_after_cursor`` from the
   cursor the previous checkpoint returned (the production DB-794 path, called
   directly instead of after the five-minute timer), with the extractor injected
   through ``llm_text_fn``;
4. snapshot traces, facts, entities, links, relationships, invalidations and
   social rows;
5. ask every read-after-write probe through ``MemoryPrimingService.prime`` with
   ``record_access=False``.

A fixture that raises is reported ``status: error`` with its message; the
scoreboard treats it as invalid, never as a zero.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from kai_memory.storage.models.entity_relationship import EntityRelationship
from kai_memory.storage.models.memory_entity import MemoryEntity
from kai_memory.storage.models.memory_entity_link import MemoryEntityLink
from kai_memory.storage.models.memory_trace import MemoryTrace
from kai_memory.storage.models.social_context import SocialContext, SocialExperience
from kai_memory.extraction.memory_extraction import ExtractionService
from kai_memory.retrieval.memory_priming import MemoryPrimingService
from kai_memory.retrieval.memory_observation import MemoryDecision, MemoryObservation
from kai_memory.retrieval.iterative_retrieval import RECALL_MARKER
from kai_memory.store.cognitive_memory import CognitiveMemoryService
from kai_memory.store.cognitive_memory_constants import DEFAULT_RETRIEVAL_WEIGHTS
from kai_memory import ports
from kai_memory.host_calls import parse_json_object_response
from memory_evaluation.eval.disposable_db import DisposableDB
from memory_evaluation.eval.extractors import ExtractionCall, RecordedExtractor
from memory_evaluation.eval.fixtures import ASSISTANT, PRODUCTION_PRIME_LIMIT, TRACE_REF, FixtureSet
from memory_evaluation.eval.matching import prohibited_matches, write_matches
from memory_evaluation.eval.providers import evaluation_embeddings
from kai_memory.extraction_cursor import ExtractionCursor
from kai_memory.subject_attribution import attribute_fact_subjects

# Readback primes as production does (fixtures.PRODUCTION_PRIME_LIMIT; a lower k makes
# the entity path's equal-salience cutoff arbitrary, so the fixture loader refuses it).
DEFAULT_READBACK_K = PRODUCTION_PRIME_LIMIT


class Extractor(Protocol):
    calls: list[Any]

    def describe(self) -> dict[str, Any]: ...
    def arm(self, checkpoint: str, response: str | None) -> None: ...
    def consumed(self) -> bool: ...
    async def __call__(self, provider: Any, prompt: str, max_tokens: int = ..., span: Any = ...) -> str | None: ...


class UnresolvedTraceRef(RuntimeError):
    """A recorded response points at a trace label that no write resolved to."""


async def run_write_readback(
    fixture_set: FixtureSet,
    db: DisposableDB,
    *,
    extractor: Extractor | None = None,
    report_root: Path | None = None,
    fixture_clock: Callable[[dict[str, Any]], datetime] | None = None,
    embedding_adapter: Any = None,
    embedding_descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay every fixture on ``db`` and return the raw run report (score it with ``build_scoreboard``)."""
    extractor = extractor if extractor is not None else RecordedExtractor()
    results: list[dict[str, Any]] = []
    with evaluation_embeddings(embedding_adapter, embedding_descriptor) as embedding_metadata:
        embedder = embedding_metadata["name"]
        for fixture in fixture_set.fixtures:
            now = fixture_clock(fixture) if fixture_clock is not None else None
            with ports.scoped({"clock": lambda now=now: now}, inherit=True) if now is not None else nullcontext():
                results.append(await _run_fixture(fixture, db, extractor))
    return {
        "fixture_set": {
            "path": _repo_relative(fixture_set.path, report_root),
            "sha256": fixture_set.sha256,
            "n": len(fixture_set.fixtures),
        },
        "extractor": extractor.describe(),
        "embedder": embedder,
        "embedding_metadata": embedding_metadata,
        "results": results,
        "mutation_check": db.mutation_check(),
    }


def _repo_relative(path: str, report_root: Path | None = None) -> str:
    """Preserve caller paths unless the host supplies an explicit report root."""
    if report_root is None:
        return path
    try:
        return Path(path).resolve().relative_to(report_root.resolve()).as_posix()
    except ValueError:
        return path


# A second layer that runs on the replayed state, inside the same disposable session
# (Layer A asks its questions here): (session, fixture, result, last sender).
AfterReplay = Callable[[AsyncSession, dict[str, Any], dict[str, Any], dict[str, Any] | None], Awaitable[None]]


async def _run_fixture(
    fixture: dict[str, Any],
    db: DisposableDB,
    extractor: Extractor,
    then: AfterReplay | None = None,
    *,
    fail_fast: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": fixture["id"],
        "fail_fast": fail_fast,
        "status": "ok",
        "error": None,
        "labels": {},
        "checkpoints": [],
        "prompt_checks": [],
        "snapshot": None,
        "readback": [],
        "messages_saved": 0,
    }
    started = time.monotonic()
    calls_before = len(extractor.calls)
    try:
        async with db.session() as session:
            with _pinned_admins(fixture):
                await _replay(session, fixture, extractor, result, then)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    result["prompt_checks"] = _prompt_checks(fixture, extractor.calls[calls_before:])
    result["duration_s"] = round(time.monotonic() - started, 3)
    return result


@contextmanager
def _pinned_admins(fixture: dict[str, Any]) -> Iterator[None]:
    """Pin the admin set to the fixture's ``admin`` participants (P-1 first-party durability)."""
    settings = ports.get_settings()
    original = settings.telegram_admin_user_ids
    settings.telegram_admin_user_ids = ",".join(
        str(p["telegram_user_id"]) for p in fixture.get("participants") or [] if p.get("admin")
    )
    try:
        yield
    finally:
        settings.telegram_admin_user_ids = original


def _chat_context(fixture: dict[str, Any], sender: dict[str, Any] | None) -> Any:
    chat = fixture["chat"]
    return ports.ChatContext(
        chat_id=chat["chat_id"],
        topic_id=chat.get("topic_id"),
        venue="telegram",
        is_group=bool(chat.get("is_group")),
        chat_title=chat.get("title"),
        chat_type="supergroup" if chat.get("is_group") else "private",
        sender_user_id=sender["telegram_user_id"] if sender else None,
        sender_name=sender["name"] if sender else None,
        sender_username=sender.get("username") if sender else None,
    )


async def _replay(
    session: AsyncSession,
    fixture: dict[str, Any],
    extractor: Extractor,
    result: dict[str, Any],
    then: AfterReplay | None = None,
) -> None:
    participants = {p["key"]: p for p in fixture.get("participants") or []}
    start_at = datetime.fromisoformat(fixture["start_at"])
    labels: dict[str, str] = result["labels"]
    pre_ids = await _seed_pre_state(session, fixture, participants, start_at, labels)
    await session.commit()

    conv = ports.ConversationManager(session)
    checkpoints = {cp["after"]: (n, cp) for n, cp in enumerate(fixture["checkpoints"], start=1)}
    messages = fixture["messages"]
    cursor: ExtractionCursor | None = None
    last_sender: dict[str, Any] | None = None
    window_start = 0
    for idx, msg in enumerate(messages, start=1):
        sender = None if msg["from"] == ASSISTANT else participants[msg["from"]]
        saved = await conv.save_message(
            fixture["chat"]["chat_id"],
            fixture["chat"].get("topic_id"),
            role="assistant" if sender is None else "user",
            content=msg["text"],
            sender_user_id=sender["telegram_user_id"] if sender else None,
            sender_name=sender["name"] if sender else None,
            venue="telegram",
        )
        saved.created_at = start_at + timedelta(minutes=msg["minute"], microseconds=idx)
        await session.flush()
        result["messages_saved"] = idx
        if sender is not None:
            last_sender = sender
        if idx not in checkpoints:
            continue
        n, cp = checkpoints[idx]
        cursor = await _run_checkpoint(
            session, fixture, extractor, result, n, cp, cursor, _chat_context(fixture, last_sender)
        )
        result["checkpoints"][-1]["window"] = [window_start + 1, idx]
        window_start = idx
        _bind_labels(fixture, await _trace_rows(session), pre_ids, labels)

    result["snapshot"] = await _snapshot(session, pre_ids, labels)
    result["readback"] = await _readback(
        session, fixture, labels, _chat_context(fixture, last_sender), fail_fast=result.get("fail_fast", False)
    )
    if then is not None:
        await then(session, fixture, result, last_sender)


async def _run_checkpoint(
    session: AsyncSession,
    fixture: dict[str, Any],
    extractor: Extractor,
    result: dict[str, Any],
    n: int,
    cp: dict[str, Any],
    cursor: ExtractionCursor | None,
    chat_ctx: Any,
) -> ExtractionCursor | None:
    completions: list[ExtractionCall] = []
    name = f"cp{n}"
    extractor.arm(name, _render_response(cp["response"], result["labels"], name))

    async def inference(provider: Any, prompt: str, **kwargs: Any) -> str | None:
        # Capture what the service actually receives, independently of an adapter's
        # own call log (which may record a different underlying response).
        attempt: dict[str, Any] = {
            "checkpoint": name,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response": None,
            "response_sha256": None,
            "error": None,
            "duration_s": None,
            "usage": {"input_tokens": None, "output_tokens": None, "status": "unknown"},
        }
        result.setdefault("extraction_calls", []).append(attempt)
        started = time.monotonic()
        try:
            raw = await extractor(provider, prompt, **kwargs)
            completions.append(ExtractionCall(name, prompt, attempt["prompt_sha256"], raw))
            attempt["response"] = raw
            if not isinstance(raw, str) or not raw.strip():
                attempt["error"] = "empty or malformed completion"
            attempt["response_sha256"] = hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw, str) else None
            return raw
        except BaseException as exc:
            attempt["error"] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            attempt["duration_s"] = time.monotonic() - started

    svc = ExtractionService(session, llm_text_fn=inference)
    # Same shape as the L2 trigger: the run is one savepoint, probed before release.
    extraction_started = time.monotonic()
    checkpoint_attempt = {
        "name": name,
        "after": cp["after"],
        "status": "running",
        "duration_s": None,
        "error": None,
        "stats": None,
    }
    result.setdefault("checkpoint_attempts", []).append(checkpoint_attempt)
    try:
        async with session.begin_nested():
            stats, cursor = await svc.extract_after_cursor(chat_ctx, cursor)
            expected_messages = cp["after"] - (result["checkpoints"][-1]["after"] if result["checkpoints"] else 0)
            _validate_checkpoint(stats, cursor, expected_messages, completions)
            await session.execute(select(1))
        await session.commit()
        if not extractor.consumed():
            raise RuntimeError(f"{name}: extraction made no model call (window too short or empty?)")
        checkpoint_attempt["status"] = "ok"
        checkpoint_attempt["stats"] = stats
    except BaseException as exc:
        checkpoint_attempt["status"] = "error"
        checkpoint_attempt["error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        checkpoint_attempt["duration_s"] = time.monotonic() - extraction_started
    result["checkpoints"].append(
        {
            "name": name,
            "extraction_s": time.monotonic() - extraction_started,
            "after": cp["after"],
            "stats": {k: v for k, v in stats.items() if isinstance(v, int | bool)},
            "store_errors": list(svc.store_errors),
            "completion_evidence": [
                {
                    "prompt": call.prompt,
                    "prompt_sha256": call.prompt_sha256,
                    "response": call.response,
                    "response_sha256": hashlib.sha256(call.response.encode()).hexdigest()
                    if isinstance(call.response, str)
                    else None,
                }
                for call in completions
            ],
        }
    )
    return cursor


def _validate_checkpoint(stats: dict[str, Any], cursor: Any, expected_messages: int, calls: list[Any]) -> None:
    """Require completed extraction, not merely an inference call or a moved cursor."""
    if (
        stats.get("messages_read") != expected_messages
        or stats.get("cursor_advanced") is not True
        or cursor is None
        or stats.get("backlog_remaining")
        or stats.get("skipped_after_retries")
        or stats.get("store_failures")
        or not calls
    ):
        raise RuntimeError("checkpoint extraction incomplete or failed")
    for call in calls:
        if not isinstance(call.response, str) or not call.response.strip():
            raise RuntimeError("checkpoint extraction has no completion")
        parsed = parse_json_object_response(call.response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("memories"), list):
            raise RuntimeError("checkpoint extraction has invalid completion schema")
        for key in ("memories", "social_experiences", "turn_outcomes", "behavior_rules", "rule_compliance"):
            if key in parsed and (
                not isinstance(parsed[key], list) or any(not isinstance(v, dict) for v in parsed[key])
            ):
                raise RuntimeError("checkpoint extraction has invalid completion schema")


def _render_response(response: dict[str, Any], labels: dict[str, str], checkpoint: str) -> str:
    def _sub(match: Any) -> str:
        label = match.group(1)
        if label not in labels:
            raise UnresolvedTraceRef(f"{checkpoint}: trace label {label!r} did not resolve to a stored trace")
        return labels[label]

    return TRACE_REF.sub(_sub, json.dumps(response, ensure_ascii=False, indent=2))


async def _seed_pre_state(
    session: AsyncSession,
    fixture: dict[str, Any],
    participants: dict[str, dict[str, Any]],
    start_at: datetime,
    labels: dict[str, str],
) -> set[str]:
    chat = fixture["chat"]
    pre_state = fixture.get("pre_state") or {}
    known = [participants[k] for k in pre_state.get("known_participants") or []]
    if known:
        session.add(
            SocialContext(
                chat_id=chat["chat_id"],
                topic_id=chat.get("topic_id"),
                context_type="group" if chat.get("is_group") else "personal",
                context_name=chat.get("title"),
                participants=[
                    {
                        "telegram_user_id": p["telegram_user_id"],
                        "name": p["name"],
                        "username": p.get("username"),
                        "last_seen_at": start_at.isoformat(),
                    }
                    for p in known
                ],
                last_active_at=start_at,
            )
        )
        await session.flush()

    svc = CognitiveMemoryService(session)
    pre_ids: set[str] = set()
    for spec in pre_state.get("traces") or []:
        entities = []
        for e in spec.get("entities") or []:
            participant = participants.get(e.get("participant") or "")
            entities.append(
                await svc.find_or_create_entity(
                    e["name"],
                    e.get("type", "person"),
                    telegram_user_id=participant["telegram_user_id"] if participant else None,
                )
            )
        facts = json.loads(json.dumps(spec.get("facts") or []))
        attribute_fact_subjects(facts, entities)
        trace = await svc.store_trace(
            content=spec["content"],
            memory_type=spec.get("memory_type", "fact"),
            significance=spec.get("significance", 3),
            source_type="conversation",
            source_chat_id=chat["chat_id"],
            source_topic_id=chat.get("topic_id"),
            typed_metadata={"schema_version": 3, "facts": facts, "entities": spec.get("entities") or []},
            source_path="l2_inline",
        )
        trace.created_at = start_at - timedelta(days=spec.get("age_days", 30))
        for entity in entities:
            await svc.link_trace_to_entity(trace.id, entity.id, "about")
        labels[spec["label"]] = str(trace.id)
        pre_ids.add(str(trace.id))
    await session.flush()
    return pre_ids


async def _trace_rows(session: AsyncSession) -> list[MemoryTrace]:
    stmt = (
        select(MemoryTrace).order_by(MemoryTrace.created_at, MemoryTrace.id).execution_options(populate_existing=True)
    )
    return list((await session.execute(stmt)).scalars())


def _bind_labels(fixture: dict[str, Any], traces: list[MemoryTrace], pre_ids: set[str], labels: dict[str, str]) -> None:
    """Bind each still-unbound expected write to the first stored trace it matches.

    A trace that hits a prohibited pattern never satisfies an expected write, so a
    stored echo cannot also be counted as a capture."""
    expected = fixture.get("expected") or {}
    prohibited = expected.get("prohibited") or []
    written = [
        view
        for view in (_trace_view(t) for t in traces if str(t.id) not in pre_ids)
        if not any(prohibited_matches(spec, view) for spec in prohibited)
    ]
    for write in expected.get("writes") or []:
        if write["label"] in labels:
            continue
        for view in written:
            if write_matches(write, view):
                labels[write["label"]] = view["id"]
                break


def _trace_view(t: MemoryTrace) -> dict[str, Any]:
    tm = t.typed_metadata if isinstance(t.typed_metadata, dict) else {}
    facts = [
        {k: f.get(k) for k in ("subject", "subject_id", "field", "value", "polarity", "slot", "durability")}
        for f in tm.get("facts") or []
        if isinstance(f, dict)
    ]
    return {
        "id": str(t.id),
        "content": t.content,
        "memory_type": t.memory_type,
        "significance": t.significance,
        "salience": round(float(t.salience or 0.0), 4),
        "durability": tm.get("durability"),
        "facts": facts,
        "invalidated": t.invalidated_at is not None,
        "invalidated_by": str(t.invalidated_by) if t.invalidated_by else None,
        "invalidation_source": t.invalidation_source,
        "source_chat_id": t.source_chat_id,
    }


async def _snapshot(session: AsyncSession, pre_ids: set[str], labels: dict[str, str]) -> dict[str, Any]:
    by_id: dict[str, list[str]] = {}
    for label, trace_id in labels.items():
        by_id.setdefault(trace_id, []).append(label)
    traces = []
    for t in await _trace_rows(session):
        view = _trace_view(t)
        view["pre_state"] = view["id"] in pre_ids
        view["labels"] = sorted(by_id.get(view["id"], []))
        traces.append(view)

    entities = (
        await session.execute(
            select(MemoryEntity)
            .order_by(MemoryEntity.created_at, MemoryEntity.id)
            .execution_options(populate_existing=True)
        )
    ).scalars()
    links = (await session.execute(select(MemoryEntityLink))).scalars()
    relationships = (await session.execute(select(EntityRelationship))).scalars()
    experiences = (await session.execute(select(SocialExperience))).scalars()
    contexts = (await session.execute(select(SocialContext).execution_options(populate_existing=True))).scalars()
    return {
        "traces": traces,
        "entities": [
            {
                "id": str(e.id),
                "name": e.name,
                "type": e.entity_type,
                "aliases": sorted(e.aliases or []),
                "telegram_user_id": e.telegram_user_id,
                "merged_into_id": str(e.merged_into_id) if e.merged_into_id else None,
            }
            for e in entities
        ],
        "links": sorted([str(lk.memory_id), str(lk.entity_id), lk.role] for lk in links),
        "relationships": sorted(
            [str(r.from_entity_id), str(r.to_entity_id), r.relationship_type] for r in relationships
        ),
        "social_experiences": sorted(x.experience for x in experiences),
        "social_participants": sorted(
            str(p.get("telegram_user_id")) for c in contexts for p in (c.participants or []) if isinstance(p, dict)
        ),
    }


async def _readback(
    session: AsyncSession,
    fixture: dict[str, Any],
    labels: dict[str, str],
    chat_ctx: Any,
    *,
    fail_fast: bool = False,
) -> list[dict[str, Any]]:
    label_of = {trace_id: label for label, trace_id in labels.items()}
    out = []
    for probe in (fixture.get("expected") or {}).get("readback") or []:
        k = probe.get("k", DEFAULT_READBACK_K)
        # Pinned code-default weights: the process-wide weights cache is neither read
        # nor written, so a run is reproducible and leaves no cached state behind.
        decision = MemoryDecision(
            MemoryObservation(model="eval", scope="fixture", input_ids=[]),
            "readback",
            "initial",
            "recognition",
            "probe",
            [],
        )
        priming = MemoryPrimingService(
            session, weights_override=DEFAULT_RETRIEVAL_WEIGHTS, **({"observation": decision} if fail_fast else {})
        )
        try:
            formatted, ids = await priming.prime(probe["cue"], limit=k, chat_ctx=chat_ctx, record_access=False)
            if fail_fast and decision.issues:
                raise RuntimeError(f"readback incomplete: {decision.issues}")
        except Exception as exc:
            if not fail_fast:
                raise
            out.append({"id": probe["id"], "error": f"{type(exc).__name__}: {exc}"[:300]})
            break
        out.append(
            {
                "id": probe["id"],
                "k": k,
                "retrieved": [str(i) for i in ids],
                "retrieved_labels": [label_of.get(str(i)) for i in ids],
                "formatted": formatted,
            }
        )
    return out


def _prompt_checks(fixture: dict[str, Any], calls: list[Any]) -> list[dict[str, Any]]:
    """Echo containment at the prompt: texts that must never reach the extractor's input.

    ``prompt_must_contain`` is the positive control: text from the same message that
    MUST arrive, so an absent echo means it was stripped, not that the message was lost."""
    prompts = [c.prompt.casefold() for c in calls]
    expected = fixture.get("expected") or {}
    checks = [
        {"id": spec["id"], "kind": spec.get("kind", "echo"), "expect": "absent", "found": False, "text": spec["text"]}
        for spec in expected.get("prompt_must_not_contain") or []
    ]
    checks += [
        {"id": spec["id"], "kind": "control", "expect": "present", "found": False, "text": spec["text"]}
        for spec in expected.get("prompt_must_contain") or []
    ]
    checks.append(
        {"id": "recall_marker", "kind": "recalled_echo", "expect": "absent", "found": False, "text": RECALL_MARKER}
    )
    for check in checks:
        needle = check.pop("text").casefold()
        check["found"] = any(needle in p for p in prompts)
    return checks


async def store_fingerprint(session: AsyncSession) -> dict[str, tuple[int, str]]:
    """Full same-transaction table fingerprints detect read-side writes, including access counters."""
    await session.flush()
    tables = (
        await session.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"))
    ).scalars()
    out = {}
    for table in list(tables):
        quoted = '"' + table.replace('"', '""') + '"'
        row = (
            await session.execute(
                text(f"SELECT count(*), coalesce(string_agg(t::text, '|' ORDER BY t::text), '') FROM {quoted} AS t")
            )
        ).one()
        out[table] = (int(row[0]), hashlib.sha256(str(row[1]).encode()).hexdigest())
    return out
