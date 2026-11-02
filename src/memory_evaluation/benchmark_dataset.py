"""Benchmark dataset lifecycle — pure manifest/batch logic (Sub-design C INC 2).

The benchmark dataset is 5 active batches of 20 fixtures each: one permanent
**anchor** (frozen, never retired — the comparability spine + long-horizon recall
test) plus up to four **rolling** batches cut from recent prod, each living ~5
months. Monthly a new rolling batch is cut and the oldest rolling batch is
retired (moved to archive, never deleted). The active set is pinned by
``manifest.json`` in S3.

This module is pure (no S3/DB I/O): the CLI/harness read/write S3 and call these
helpers. Storage layout (relative to the S3 base prefix):

    benchmark/manifest.json
    benchmark/batches/<batch_id>.fixtures.json     # Part A: {query, gold_trace_ids}
    benchmark/batches/<batch_id>.gold.jsonl        # Part B: {query, answer, provenance, status}
    benchmark/candidate_pools/<git_sha>/<batch_id>.pool.json
    benchmark/results/recall/<date>.json
    benchmark/results/eval/<date>.json
    benchmark/archive/<batch_id>.*                 # retired rolling batches

``batch_id`` = ``"<kind>-<YYYY-MM-DD>"`` (e.g. ``anchor-2026-05-02`` /
``rolling-2026-06-06``).
"""

from __future__ import annotations

from typing import Any

BENCHMARK_PREFIX = "benchmark"
MAX_ROLLING = 4

ANCHOR = "anchor"
ROLLING = "rolling"


# ── batch id parsing ────────────────────────────────────────────────────────


def batch_kind(batch_id: str) -> str:
    """``"anchor"`` or ``"rolling"`` — the prefix before the date."""
    return batch_id.split("-", 1)[0]


def batch_date(batch_id: str) -> str:
    """The ISO date portion of the id (sortable as a string)."""
    return batch_id.split("-", 1)[1]


# ── manifest lifecycle ──────────────────────────────────────────────────────


def cut_manifest(
    manifest: dict[str, Any],
    new_batch_id: str,
    *,
    at: str,
    max_rolling: int = MAX_ROLLING,
) -> tuple[dict[str, Any], str | None]:
    """Add ``new_batch_id`` (a rolling batch) to the active set; if that exceeds
    ``max_rolling`` rolling batches, retire the OLDEST rolling batch (by date).

    The anchor is never retired. Returns ``(new_manifest, retired_batch_id)``
    where ``retired_batch_id`` is None when nothing was retired. The input
    manifest is not mutated; a ``cut`` history event is appended.
    """
    if batch_kind(new_batch_id) != ROLLING:
        raise ValueError(f"cut_manifest only adds rolling batches (the anchor is never cut), got {new_batch_id!r}")
    active_existing = manifest.get("active_batches", [])
    if new_batch_id in active_existing:
        raise ValueError(f"batch {new_batch_id!r} is already active")

    active = [*active_existing, new_batch_id]
    rolling_oldest_first = sorted((b for b in active if batch_kind(b) == ROLLING), key=batch_date)

    retired: str | None = None
    while sum(1 for b in active if batch_kind(b) == ROLLING) > max_rolling:
        retired = rolling_oldest_first.pop(0)
        active.remove(retired)

    new_manifest = {
        **manifest,
        "active_batches": active,
        "updated_at": at,
        "history": [
            *manifest.get("history", []),
            {"event": "cut", "batch_id": new_batch_id, "retired": retired, "at": at},
        ],
    }
    return new_manifest, retired


# ── scoring comparability (the swap-night rule) ─────────────────────────────


def is_gateable(batch_id: str, prior_baseline_batches: set[str]) -> bool:
    """A batch is regression-GATEABLE only once it has ≥1 prior night's
    baseline. A freshly cut batch is report-only until its next night."""
    return batch_id in set(prior_baseline_batches)


def is_rebaseline(current_active: set[str], baseline_active: set[str]) -> bool:
    """True when the active batch set changed vs the baseline (a swap night) —
    the aggregate will step on composition change, so the harness logs a
    re-baseline event rather than alarming."""
    return set(current_active) != set(baseline_active)


# ── S3 keys (never git) ──────────────────────────────────────────────────────


def _join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p.strip("/"))


def manifest_key(base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "manifest.json")


def batch_fixtures_key(batch_id: str, base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "batches", f"{batch_id}.fixtures.json")


def batch_gold_key(batch_id: str, base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "batches", f"{batch_id}.gold.jsonl")


def candidate_pool_key(git_sha: str, batch_id: str, base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "candidate_pools", git_sha, f"{batch_id}.pool.json")


def result_key(name: str, date_str: str, base_prefix: str = "") -> str:
    """Canonical results key for any benchmark: ``benchmark/results/<name>/<date>.json``.

    Generalizes ``recall_result_key`` / ``eval_result_key`` so the framework
    runner can archive any registered benchmark on the same layout (design §3.6).
    """
    return _join(base_prefix, BENCHMARK_PREFIX, "results", name, f"{date_str}.json")


def recall_result_key(date_str: str, base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "results", "recall", f"{date_str}.json")


def eval_result_key(date_str: str, base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "results", "eval", f"{date_str}.json")


def archive_key(filename: str, base_prefix: str = "") -> str:
    return _join(base_prefix, BENCHMARK_PREFIX, "archive", filename)


def gold_qa_key(date_str: str, base_prefix: str = "") -> str:
    """Part B: the date-versioned mined gold Q→A set (provisional → approved)."""
    return _join(base_prefix, BENCHMARK_PREFIX, "gold_qa", f"{date_str}.jsonl")


def eval_key(date_str: str, base_prefix: str = "") -> str:
    """Part B: the date-versioned end-to-end eval results (on/off ablation)."""
    return _join(base_prefix, BENCHMARK_PREFIX, "eval", f"{date_str}.json")


def frozen_gold_key(code_sha: str, base_prefix: str = "") -> str:
    """Part B: the gold set frozen at a commit SHA — the immutable question set the
    eval replays so re-runs after each C increment are comparable."""
    return _join(base_prefix, BENCHMARK_PREFIX, "gold_qa", "frozen", f"{code_sha}.jsonl")
