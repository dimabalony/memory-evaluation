"""Score a write-to-readback run into a scoreboard (DB-840).

Pure: reads the runner's report and the fixtures, touches nothing else. One row
per fixture and an aggregate over the valid ones. A fixture whose run errored is
``invalid`` and stays out of every aggregate: an error is not a zero.

``quality`` says what the numbers can claim. With the recorded extractor the
model's judgment is fixed, so the scoreboard measures the pipeline around it
(persistence, attribution, slot validation, durability stamping, dedup,
supersession gates, echo containment, cursor coverage, readback): "pipeline-only".
"""

from __future__ import annotations

import hashlib
from typing import Any

from kai_memory.host_calls import utc_now
from kai_memory.extraction.memory_extraction import EXTRACTION_PROMPT
from kai_memory.canonicalization import same_canonical_entity
from memory_evaluation.eval.fixtures import FixtureSet
from memory_evaluation.eval.matching import matching_fact, prohibited_matches, text_contains

SCOREBOARD_FORMAT = "kai-memory-scoreboard/v1"
ECHO_KINDS = frozenset({"assistant_echo", "recalled_echo"})

# Every §6 instrument this run does not measure (design 2026-10-09, §6 table).
GAPS: tuple[dict[str, str], ...] = (
    {
        "instrument": "read-to-answer (Layer A)",
        "component": "Recognition/retrieval, Mid-turn recall",
        "reason": "answer correctness needs the DB-745 turn harness and a judge whose unparsed verdict is invalid",
    },
    {"instrument": "consolidation-invariants", "component": "Consolidation", "reason": "L3 is not replayed"},
    {
        "instrument": "relationship-allowlist precision/recall, mixed-script fork rate",
        "component": "Entities/relationships",
        "reason": "relationship edges are written by L3, which is not replayed",
    },
    {"instrument": "rule-efficacy", "component": "Procedural learning", "reason": "procedural extraction is off"},
    {
        "instrument": "steer-rejection attribution, verdict-anchor fidelity",
        "component": "Attribution",
        "reason": "no native steering in this layer",
    },
    {"instrument": "scope-leak-rate", "component": "Scope", "reason": "one chat per fixture; no cross-scope reads"},
    {"instrument": "retention-curve", "component": "Forgetting", "reason": "no decay or pruning replay"},
    {"instrument": "native-vs-legacy parity", "component": "Native vs legacy", "reason": "needs the recognition hook"},
    {
        "instrument": "persona composition, rule serving scope",
        "component": "Recognition/retrieval",
        "reason": "not exercised by write-to-readback",
    },
)
_RECORDED_GAP = {
    "instrument": "extractor quality",
    "component": "Extraction",
    "reason": "recorded responses fix the model's output; run --live for extractor-and-pipeline evidence",
}


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _entity_names(entity: dict[str, Any]) -> list[str]:
    return [entity["name"], *(entity.get("aliases") or [])]


def _names_match(expected: str, entity: dict[str, Any]) -> bool:
    return any(same_canonical_entity(expected, n) for n in _entity_names(entity))


def score_fixture(fixture: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """One scoreboard row. An errored run is ``invalid`` and carries no metrics."""
    if result.get("status") != "ok":
        return {"id": fixture["id"], "status": "invalid", "error": result.get("error")}

    failures = [e for cp in result.get("checkpoints") or [] for e in cp.get("store_errors") or []]
    if failures:
        return {"id": fixture["id"], "status": "invalid", "error": "checkpoint store errors", "errors": failures}

    expected = fixture.get("expected") or {}
    snapshot = result["snapshot"]
    labels: dict[str, str] = result["labels"]
    traces = {t["id"]: t for t in snapshot["traces"]}
    written_views = [t for t in snapshot["traces"] if not t["pre_state"]]
    entities = [e for e in snapshot["entities"] if not e.get("merged_into_id")]
    entity_by_id = {e["id"]: e for e in snapshot["entities"]}
    participants = {p["key"]: p for p in fixture.get("participants") or []}
    writes = expected.get("writes") or []

    written = [w["label"] for w in writes if w["label"] in labels]
    omitted = [w["label"] for w in writes if w["label"] not in labels]

    prohibited_hits = []
    for spec in expected.get("prohibited") or []:
        for view in written_views:
            if prohibited_matches(spec, view):
                kind = spec.get("kind", "prohibited")
                prohibited_hits.append(
                    {"id": spec["id"], "kind": kind, "trace_id": view["id"], "content": view["content"]}
                )
    hit_ids = {h["trace_id"] for h in prohibited_hits}
    matched_ids = {labels[w] for w in written}
    unexpected = [
        {"trace_id": v["id"], "content": v["content"]}
        for v in written_views
        if v["id"] not in matched_ids and v["id"] not in hit_ids
    ]

    attribution: dict[str, Any] = {"checked": 0, "correct": 0, "misses": []}
    durability: dict[str, Any] = {"checked": 0, "correct": 0, "misses": []}
    for w in writes:
        view = traces.get(labels.get(w["label"], ""))
        if view is None:
            continue
        fact = matching_fact(w, view) if w.get("fact") else None
        subject = w.get("subject")
        participant = participants.get(w.get("subject_participant") or "")
        if subject or participant:
            attribution["checked"] += 1
            entity = entity_by_id.get(str((fact or {}).get("subject_id")))
            ok = (
                entity is not None
                and (not subject or _names_match(subject, entity))
                and (not participant or entity.get("telegram_user_id") == participant["telegram_user_id"])
            )
            if ok:
                attribution["correct"] += 1
            else:
                attribution["misses"].append({"label": w["label"], "subject_id": (fact or {}).get("subject_id")})
        if "durable" in w:
            durability["checked"] += 1
            got = (fact or {}).get("durability") if w.get("fact") else view.get("durability")
            want = "durable" if w["durable"] else ("transient" if w.get("fact") else None)
            if got == want:
                durability["correct"] += 1
            else:
                durability["misses"].append({"label": w["label"], "expected": want, "got": got})
    for spec in expected.get("entities") or []:
        attribution["checked"] += 1
        participant = participants.get(spec.get("participant") or "")
        found = [e for e in entities if e["type"] == spec.get("type", "person") and _names_match(spec["name"], e)]
        # An explicit "participant": null means the entity must NOT be bound to any sender.
        want_id = participant["telegram_user_id"] if participant else None
        ok = len(found) == 1 and ("participant" not in spec or found[0].get("telegram_user_id") == want_id)
        if ok:
            attribution["correct"] += 1
        else:
            attribution["misses"].append({"entity": spec["name"], "rows": len(found)})

    dedup: dict[str, Any] = {"checked": 0, "met": 0, "misses": []}
    for spec in expected.get("dedup") or []:
        dedup["checked"] += 1
        rows = [
            e
            for e in entities
            if e["type"] == spec.get("type", "person") and any(_names_match(n, e) for n in spec["names"])
        ]
        if 1 <= len(rows) <= spec.get("max_rows", 1):
            dedup["met"] += 1
        else:
            dedup["misses"].append({"id": spec["id"], "rows": [e["name"] for e in rows]})

    supersession: dict[str, Any] = {"checked": 0, "met": 0, "misses": []}
    for spec in expected.get("supersessions") or []:
        supersession["checked"] += 1
        old, new = traces.get(labels.get(spec["old"], "")), labels.get(spec["new"])
        if old is not None and new is not None and old["invalidated"] and old["invalidated_by"] == new:
            supersession["met"] += 1
        else:
            supersession["misses"].append({"old": spec["old"], "new": spec["new"]})
    for label in expected.get("must_stay_valid") or []:
        supersession["checked"] += 1
        view = traces.get(labels.get(label, ""))
        if view is not None and not view["invalidated"]:
            supersession["met"] += 1
        else:
            supersession["misses"].append({"must_stay_valid": label})

    echo_leaks = [{"id": h["id"], "where": "store"} for h in prohibited_hits if h["kind"] in ECHO_KINDS]
    prompt_checks = result.get("prompt_checks") or []
    echo_leaks += [{"id": c["id"], "where": "prompt"} for c in prompt_checks if c["expect"] == "absent" and c["found"]]
    controls = [c for c in prompt_checks if c["expect"] == "present"]

    probes_out = []
    by_probe = {r["id"]: r for r in result.get("readback") or []}
    for probe in expected.get("readback") or []:
        got = by_probe.get(probe["id"])
        if got is None or got.get("error"):
            probes_out.append(
                {"id": probe["id"], "status": "invalid", "pass": False, "error": (got or {}).get("error", "not run")}
            )
            continue
        ranks = {
            label: (got["retrieved"].index(labels[label]) + 1 if labels.get(label) in got["retrieved"] else None)
            for label in probe.get("expect_labels") or []
        }
        forbidden = [label for label in probe.get("forbid_labels") or [] if labels.get(label) in got["retrieved"]]
        forbidden += [t for t in probe.get("forbid_contains") or [] if text_contains(got["formatted"], t)]
        passed = all(r is not None for r in ranks.values()) and not forbidden
        probes_out.append({"id": probe["id"], "pass": passed, "expect_ranks": ranks, "forbidden_hits": forbidden})

    messages_read = sum(int(cp["stats"].get("messages_read", 0)) for cp in result["checkpoints"])
    extraction_horizon = (
        result["checkpoints"][-1].get("after", result["messages_saved"]) if result["checkpoints"] else 0
    )
    return {
        "id": fixture["id"],
        "status": "ok",
        "messages": len(fixture["messages"]),
        "checkpoints": len(result["checkpoints"]),
        "n_expected": len(writes),
        "n_written_traces": len(written_views),
        "written": written,
        "omitted": omitted,
        "capture_recall": _ratio(len(written), len(writes)),
        "prohibited_hits": prohibited_hits,
        "unexpected": unexpected,
        "attribution": attribution,
        "durability": durability,
        "dedup": dedup,
        "supersession": supersession,
        "echo_leaks": echo_leaks,
        "prompt_controls": {
            "checked": len(controls),
            "met": sum(c["found"] for c in controls),
            "misses": [c["id"] for c in controls if not c["found"]],
        },
        "readback": {
            "checked": sum(p.get("status") != "invalid" for p in probes_out),
            "invalid": sum(p.get("status") == "invalid" for p in probes_out),
            "passed": sum(p["pass"] for p in probes_out),
            "probes": probes_out,
        },
        "extraction_coverage": _ratio(messages_read, extraction_horizon),
        "extraction_evidence": {
            "messages_read": messages_read,
            "eligible_messages": extraction_horizon,
            "retained_history_messages": result["messages_saved"],
            "intentionally_unextracted": result["messages_saved"] - extraction_horizon,
            "last_checkpoint": extraction_horizon,
        },
        "errors": [e for cp in result["checkpoints"] for e in cp.get("store_errors") or []],
        "duration_s": result.get("duration_s"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["status"] == "ok"]

    def total(key: str, sub: str) -> int:
        return sum(int(r[key][sub]) for r in valid)

    coverage = [r["extraction_coverage"] for r in valid if r["extraction_coverage"] is not None]
    n_hits = sum(len(r["prohibited_hits"]) for r in valid)
    # A trace that matches two prohibited specs is still one fabricated trace.
    n_fabricated = sum(len({h["trace_id"] for h in r["prohibited_hits"]}) for r in valid)
    return {
        "n_fixtures": len(rows),
        "n_valid": len(valid),
        "n_invalid": len(rows) - len(valid),
        "n_invalid_readback": sum(r["readback"].get("invalid", 0) for r in valid),
        "denominators": {
            "capture_recall": sum(r["n_expected"] for r in valid),
            "fabrication_rate": sum(r["n_written_traces"] for r in valid),
            "attribution_accuracy": total("attribution", "checked"),
            "durability_accuracy": total("durability", "checked"),
            "dedup_accuracy": total("dedup", "checked"),
            "supersession_accuracy": total("supersession", "checked"),
            "readback_accuracy": total("readback", "checked"),
        },
        "capture_recall": _ratio(sum(len(r["written"]) for r in valid), sum(r["n_expected"] for r in valid)),
        "omitted": sum(len(r["omitted"]) for r in valid),
        "prohibited_hits": n_hits,
        "fabrication_rate": _ratio(n_fabricated, sum(r["n_written_traces"] for r in valid)),
        "unexpected": sum(len(r["unexpected"]) for r in valid),
        "attribution_accuracy": _ratio(total("attribution", "correct"), total("attribution", "checked")),
        "durability_accuracy": _ratio(total("durability", "correct"), total("durability", "checked")),
        "dedup_accuracy": _ratio(total("dedup", "met"), total("dedup", "checked")),
        "supersession_accuracy": _ratio(total("supersession", "met"), total("supersession", "checked")),
        "echo_leaks": sum(len(r["echo_leaks"]) for r in valid),
        "prompt_control_misses": sum(len(r["prompt_controls"]["misses"]) for r in valid),
        "readback_accuracy": _ratio(total("readback", "passed"), total("readback", "checked")),
        "extraction_coverage_min": min(coverage) if coverage else None,
        "errors": sum(len(r["errors"]) for r in valid),
    }


def build_scoreboard(
    report: dict[str, Any], fixture_set: FixtureSet, *, source_sha: str | None = None
) -> dict[str, Any]:
    """The scoreboard for one ``run_write_readback`` report over ``fixture_set``."""
    by_id = {f["id"]: f for f in fixture_set.fixtures}
    results = {r["id"]: r for r in report["results"]}
    if len(results) != len(report["results"]) or set(results) - set(by_id):
        raise ValueError("duplicate or unknown fixture results")
    rows = [
        score_fixture(f, results.get(f["id"], {"status": "error", "error": "not run"})) for f in fixture_set.fixtures
    ]
    extractor = report["extractor"]
    gaps = [dict(g) for g in GAPS]
    if extractor.get("name") == "recorded":
        gaps.append(dict(_RECORDED_GAP))
    return {
        "header": {
            "format": SCOREBOARD_FORMAT,
            "generated_at": utc_now().isoformat(),
            "source_sha": source_sha or "unknown",
            "fixture_set": report["fixture_set"],
            "fixture_provenance": [
                {
                    "id": f["id"],
                    "source": f["_provenance"]["source"],
                    "approved_by": f["_provenance"]["approved_by"],
                    "version": f["_provenance"]["version"],
                    "content_sha256": f["_provenance"]["content_sha256"],
                }
                for f in fixture_set.fixtures
            ],
            "extractor": extractor,
            "quality": extractor.get("quality"),
            "prompt_sha256": hashlib.sha256(EXTRACTION_PROMPT.encode("utf-8")).hexdigest(),
            "embedder": report["embedder"],
            "judge": None,
            "cognitive_config": "code defaults (the disposable database has no cognitive_config rows)",
            "retrieval_weights": "code defaults (DEFAULT_RETRIEVAL_WEIGHTS, pinned for readback)",
        },
        "mutation_check": report["mutation_check"],
        "aggregate": _aggregate(rows),
        "fixtures": rows,
        "gaps": gaps,
    }
