"""Fixture format ``kai-memory-lifecycle/v1`` and its loader (DB-840).

One fixture is one conversation replayed through the real persistence and L2 path:

* ``chat`` and ``participants`` (each with a Telegram-style sender id);
* ``pre_state``: memory that exists before the conversation (labelled traces,
  participants already known to the chat);
* ``messages`` in order, each from a participant key or ``"assistant"``;
* ``checkpoints``: after which message L2 runs, and the extraction response
  recorded for that run. String values may carry ``{{trace:<label>}}``, replaced
  at run time by the id of the trace that label resolved to;
* ``expected``: labelled writes, prohibited writes, supersessions, entity dedup,
  entity bindings and read-after-write probes (Layer B), and ``questions`` with
  expected, prohibited and abstention outcomes (Layer A). These labels are the
  truth; a recorded response is only a stand-in for the extractor and may be wrong.

Every fixture carries ``_provenance`` (source, author, approved_by, version,
content_sha256, pii). The loader refuses a fixture without it, a stale
``content_sha256``, and any non-synthetic fixture without ``approved_by``:
production data stays out of git unless it is sanitized AND approved (DB-411).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = "kai-memory-lifecycle/v1"
PROVENANCE_SOURCES = frozenset({"synthetic", "sanitized", "production-derived"})
_PROVENANCE_KEYS = ("source", "author", "approved_by", "version", "content_sha256", "pii")
AVAILABILITIES = frozenset({"memory_only", "history_only", "both_agree", "history_conflicts_stale_memory", "nowhere"})
SPLITS = frozenset({"tuning", "held_out"})
HISTORY_WINDOW = 30
ASSISTANT = "assistant"
# The ten DB-411 lifecycle axes a fixture may declare.
AXES = frozenset(
    {
        "continuation_anaphora",
        "topic_switch_return",
        "overlapping_names",
        "corrections_contradictions",
        "durable_vs_transient",
        "stale_facts_updates",
        "assistant_paraphrase_echo",
        "distractor_topics",
        "group_sender_attribution",
        "abstention_control",
    }
)
TRACE_REF = re.compile(r"\{\{trace:([A-Za-z0-9_\-]+)\}\}")
# The limit production priming runs with. A readback probe below it would cut the
# entity-linked path (ORDER BY salience DESC LIMIT k, no tie-breaker) inside a run of
# equal-salience traces, so which trace survives would be arbitrary between runs.
PRODUCTION_PRIME_LIMIT = 8


class FixtureError(ValueError):
    """A fixture file the loader refuses."""


@dataclass(frozen=True)
class FixtureSet:
    """The loaded fixtures of one directory, in file-name order."""

    path: str
    fixtures: list[dict[str, Any]]
    sha256: str


def content_sha256(fixture: dict[str, Any]) -> str:
    """Hash of the fixture as canonical JSON, without its own ``content_sha256``."""
    body = json.loads(json.dumps(fixture))
    body.get("_provenance", {}).pop("content_sha256", None)
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stamp_content_sha256(path: str | Path) -> str:
    """Rewrite a fixture file's ``_provenance.content_sha256`` (for authors)."""
    p = Path(path)
    fixture = json.loads(p.read_text(encoding="utf-8"))
    fixture.setdefault("_provenance", {})["content_sha256"] = content_sha256(fixture)
    p.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(fixture["_provenance"]["content_sha256"])


def provenance_issues(fixture: dict[str, Any]) -> list[str]:
    """Why the loader would refuse this fixture's ``_provenance`` (empty: accepted)."""
    prov = fixture.get("_provenance")
    if not isinstance(prov, dict):
        return ["missing _provenance"]
    issues = [f"_provenance.{k} missing" for k in _PROVENANCE_KEYS if k not in prov]
    source = prov.get("source")
    if source not in PROVENANCE_SOURCES:
        issues.append(f"_provenance.source must be one of {sorted(PROVENANCE_SOURCES)}, got {source!r}")
    elif source != "synthetic" and not prov.get("approved_by"):
        issues.append(f"_provenance.source={source} requires approved_by")
    if not isinstance(prov.get("author"), str) or not prov.get("author"):
        issues.append("_provenance.author must be a non-empty string")
    if not isinstance(prov.get("version"), int):
        issues.append("_provenance.version must be an integer")
    if "content_sha256" in prov and prov["content_sha256"] != content_sha256(fixture):
        issues.append("_provenance.content_sha256 does not match the content (re-stamp after review)")
    return issues


def structure_issues(fixture: dict[str, Any]) -> list[str]:
    """Structural problems that would make a run meaningless (empty: well-formed)."""
    issues: list[str] = []
    if fixture.get("format") != FORMAT:
        issues.append(f"format must be {FORMAT!r}")
    if not fixture.get("id"):
        issues.append("id missing")
    unknown_axes = set(fixture.get("axes") or []) - AXES
    if unknown_axes:
        issues.append(f"unknown axes {sorted(unknown_axes)}")
    if "split" in fixture and fixture["split"] not in SPLITS:
        issues.append("split must be tuning or held_out")
    chat = fixture.get("chat")
    if not isinstance(chat, dict) or not isinstance(chat.get("chat_id"), int):
        issues.append("chat.chat_id must be an integer")
    # Replay derives every created_at from start_at; a naive value would be persisted
    # into timestamptz columns and compared against aware cursor timestamps.
    start_at = fixture.get("start_at")
    try:
        parsed_start = datetime.fromisoformat(start_at) if isinstance(start_at, str) else None
    except ValueError:
        parsed_start = None
    if parsed_start is None or parsed_start.tzinfo is None:
        issues.append("start_at must be an ISO-8601 timestamp with a UTC offset")

    participants = fixture.get("participants") or []
    keys = [p.get("key") for p in participants]
    ids = [p.get("telegram_user_id") for p in participants]
    if len(set(keys)) != len(keys) or ASSISTANT in keys:
        issues.append("participant keys must be unique and not 'assistant'")
    if len(set(ids)) != len(ids) or not all(isinstance(i, int) for i in ids):
        issues.append("participant telegram_user_id values must be unique integers")
    known = set(keys)

    pre_state = fixture.get("pre_state") or {}
    for key in pre_state.get("known_participants") or []:
        if key not in known:
            issues.append(f"pre_state.known_participants: unknown participant {key!r}")

    messages = fixture.get("messages") or []
    if not messages:
        issues.append("messages empty")
    for i, msg in enumerate(messages, start=1):
        if msg.get("from") != ASSISTANT and msg.get("from") not in known:
            issues.append(f"messages[{i}].from {msg.get('from')!r} is not a participant or 'assistant'")
        if not isinstance(msg.get("text"), str) or not msg["text"].strip():
            issues.append(f"messages[{i}].text empty")
    minutes = [m.get("minute") for m in messages]
    if not all(isinstance(m, int) for m in minutes) or minutes != sorted(minutes):
        issues.append("messages[].minute must be non-decreasing integers")

    labels: list[str] = [t.get("label") for t in pre_state.get("traces") or []]
    expected = fixture.get("expected") or {}
    labels += [w.get("label") for w in expected.get("writes") or []]
    if len(set(labels)) != len(labels) or not all(labels):
        issues.append("trace labels (pre_state.traces + expected.writes) must be unique and non-empty")
    label_set = set(labels)

    last = 0
    for c, cp in enumerate(fixture.get("checkpoints") or [], start=1):
        after = cp.get("after")
        if not isinstance(after, int) or not last < after <= len(messages):
            issues.append(f"checkpoints[{c}].after must increase within 1..{len(messages)}")
        else:
            last = after
        response = cp.get("response")
        if not isinstance(response, dict) or not isinstance(response.get("memories"), list):
            issues.append(f"checkpoints[{c}].response must be an object with a memories list")
            continue
        for ref in TRACE_REF.findall(json.dumps(response)):
            if ref not in label_set:
                issues.append(f"checkpoints[{c}] references unknown trace label {ref!r}")
    if not fixture.get("checkpoints"):
        issues.append("checkpoints empty")

    for s in expected.get("supersessions") or []:
        for side in ("old", "new"):
            if s.get(side) not in label_set:
                issues.append(f"supersession {side} label {s.get(side)!r} unknown")
    for label in expected.get("must_stay_valid") or []:
        if label not in label_set:
            issues.append(f"must_stay_valid label {label!r} unknown")
    for probe in expected.get("readback") or []:
        for label in [*(probe.get("expect_labels") or []), *(probe.get("forbid_labels") or [])]:
            if label not in label_set:
                issues.append(f"readback {probe.get('id')!r} label {label!r} unknown")
        if not probe.get("cue"):
            issues.append(f"readback {probe.get('id')!r} has no cue")
        if probe.get("k", PRODUCTION_PRIME_LIMIT) < PRODUCTION_PRIME_LIMIT:
            issues.append(
                f"readback {probe.get('id')!r}: k below the production priming limit {PRODUCTION_PRIME_LIMIT}"
            )
    questions = expected.get("questions") or []
    issues += _question_issues(questions, known)
    for q in questions:
        if "availability" not in q:
            if "split" in fixture:
                issues.append(f"question {q.get('id')!r}: split fixtures require availability metadata")
            continue
        prefix = f"question {q.get('id')!r}"
        availability = q["availability"]
        if availability not in AVAILABILITIES:
            issues.append(f"{prefix}: unknown availability")
        distance = q.get("distance_user_turns")
        if type(distance) is not int or distance < 0:
            issues.append(f"{prefix}: distance_user_turns must be a nonnegative integer")
        evidence = q.get("evidence_message_indices")
        if not isinstance(evidence, list) or any(type(i) is not int or not 1 <= i <= len(messages) for i in evidence):
            issues.append(f"{prefix}: evidence_message_indices must index messages (1-based)")
        elif evidence:
            actual = sum(m["from"] != ASSISTANT for m in messages[max(evidence) :])
            if distance != actual:
                issues.append(f"{prefix}: distance_user_turns disagrees with evidence")
        memory_labels = q.get("memory_labels")
        if not isinstance(memory_labels, list) or any(
            not isinstance(label, str) or label not in label_set for label in memory_labels
        ):
            issues.append(f"{prefix}: memory_labels must reference known traces")
        if availability == "memory_only" and q.get("history", "full") != "none":
            issues.append(f"{prefix}: memory_only requires history none")
        history_available = availability in {"history_only", "both_agree", "history_conflicts_stale_memory"}
        if history_available and q.get("history", "full") != "full":
            issues.append(f"{prefix}: history availability requires history full")
        if availability == "nowhere" and (evidence or memory_labels or not q.get("abstain")):
            issues.append(f"{prefix}: nowhere requires abstention without evidence or memory")
        if history_available and not evidence:
            issues.append(f"{prefix}: history availability requires message evidence")
        if (
            history_available
            and isinstance(evidence, list)
            and any(type(i) is int and i <= len(messages) - HISTORY_WINDOW for i in evidence)
        ):
            issues.append(f"{prefix}: history evidence is outside the delivered final {HISTORY_WINDOW} messages")
        if availability == "nowhere" and distance != 0:
            issues.append(f"{prefix}: nowhere requires distance zero")
        if availability == "history_only" and memory_labels:
            issues.append(f"{prefix}: history_only cannot name memory labels")
        if availability in {"memory_only", "both_agree", "history_conflicts_stale_memory"} and not memory_labels:
            issues.append(f"{prefix}: memory availability requires memory labels")
    for w in expected.get("writes") or []:
        if not (w.get("content_contains") or w.get("fact")):
            issues.append(f"write {w.get('label')!r} needs content_contains or fact")
        if w.get("subject_participant") is not None and w["subject_participant"] not in known:
            issues.append(f"write {w.get('label')!r}: unknown subject_participant")
    return issues


def _question_issues(questions: list[dict[str, Any]], participants: set[str]) -> list[str]:
    """Layer A questions: each is answerable (an expected answer) or an abstention, never both."""
    issues: list[str] = []
    ids = [q.get("id") for q in questions]
    if len(set(ids)) != len(ids) or not all(ids):
        issues.append("question ids must be unique and non-empty")
    for q in questions:
        qid = q.get("id")
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            issues.append(f"question {qid!r} has no question text")
        if q.get("asker") is not None and q["asker"] not in participants:
            issues.append(f"question {qid!r}: unknown asker {q['asker']!r}")
        if q.get("history", "full") not in ("full", "none"):
            issues.append(f"question {qid!r}: history must be 'full' or 'none'")
        unknown_axes = set(q.get("axes") or []) - AXES
        if unknown_axes:
            issues.append(f"question {qid!r}: unknown axes {sorted(unknown_axes)}")
        expects = bool(q.get("expect_any") or q.get("expect_all") or q.get("expect_exact") is not None)
        if bool(q.get("abstain")) == expects:
            issues.append(f"question {qid!r} needs exactly one of an expected answer or abstain: true")
    return issues


def load_fixture(path: str | Path) -> dict[str, Any]:
    """Load one fixture file, refusing bad provenance or structure."""
    p = Path(path)
    try:
        fixture = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{p.name}: not JSON ({exc})") from None
    if not isinstance(fixture, dict):
        raise FixtureError(f"{p.name}: a fixture is a JSON object")
    issues = provenance_issues(fixture) + structure_issues(fixture)
    if issues:
        raise FixtureError(f"{p.name}: " + "; ".join(issues))
    return fixture


def load_fixture_set(path: str | Path, *, ids: list[str] | None = None) -> FixtureSet:
    """Load every ``*.json`` fixture under ``path`` (or the named ``ids``), in file order."""
    root = Path(path)
    files = sorted(root.glob("*.json")) if root.is_dir() else [root]
    if not files:
        raise FixtureError(f"no fixtures under {root}")
    fixtures = [load_fixture(f) for f in files]
    seen: dict[str, str] = {}
    for file, fixture in zip(files, fixtures, strict=True):
        fixture_id = fixture["id"]
        if fixture_id in seen:
            raise FixtureError(f"duplicate fixture id {fixture_id!r} in {seen[fixture_id]} and {file.name}")
        seen[fixture_id] = file.name
    if ids is not None:
        wanted = set(ids)
        missing = wanted - {f["id"] for f in fixtures}
        if missing:
            raise FixtureError(f"unknown fixture ids {sorted(missing)}")
        fixtures = [f for f in fixtures if f["id"] in wanted]
    set_sha = hashlib.sha256(
        "\n".join(f"{f['id']}:{f['_provenance']['content_sha256']}" for f in fixtures).encode("utf-8")
    ).hexdigest()
    return FixtureSet(path=str(root), fixtures=fixtures, sha256=set_sha)


def sets_missing_provenance(fixtures_root: str | Path, *, exclude: frozenset[str] = frozenset()) -> list[str]:
    """Fixture-set directories under ``fixtures_root`` where no JSON file declares ``_provenance``.

    The ratchet test pins today's list (the legacy sets): a new set without
    provenance fails it, and a legacy set that gains provenance must leave it.
    """
    missing: list[str] = []
    for directory in sorted(p for p in Path(fixtures_root).iterdir() if p.is_dir() and not p.name.startswith("_")):
        if directory.name in exclude:
            continue
        declared = False
        for f in directory.rglob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and isinstance(data.get("_provenance"), dict):
                declared = True
                break
        if not declared:
            missing.append(directory.name)
    return missing


def select_fixture_subset(examples: list[dict[str, Any]], n: int | None, seed: int | None) -> list[dict[str, Any]]:
    """Preserve the existing runner's seeded sample and original fixture ordering."""
    import random

    if n is None or n >= len(examples):
        return list(examples)
    indexed = list(enumerate(examples))
    chosen = random.Random(seed if seed is not None else 0).sample(indexed, n)
    chosen.sort(key=lambda pair: pair[0])
    return [example for _index, example in chosen]
