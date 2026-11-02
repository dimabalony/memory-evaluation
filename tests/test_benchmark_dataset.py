"""Benchmark dataset lifecycle — manifest / batch / gate pure core (INC 2).

Implements the agreed dataset-lifecycle design: 5 active batches (1 frozen
anchor + ≤4 rolling), monthly cut-one / retire-oldest-rolling, S3-versioned via
manifest.json, per-batch gateable-vs-report-only scoring. Pure logic only —
no S3/DB I/O (that lives in the CLI/harness).
"""

from __future__ import annotations

import pytest

from memory_evaluation.benchmark_dataset import (
    archive_key,
    batch_date,
    batch_fixtures_key,
    batch_gold_key,
    batch_kind,
    candidate_pool_key,
    cut_manifest,
    is_gateable,
    is_rebaseline,
    manifest_key,
    recall_result_key,
)


def _manifest(active):
    """Test helper: manifest."""
    return {
        "version": 1,
        "anchor": "anchor-2026-05-02",
        "active_batches": active,
        "history": [],
    }


# ── batch id parsing ────────────────────────────────────────────────────────


def test_batch_kind_and_date():
    """Batch kind and date."""
    assert batch_kind("anchor-2026-05-02") == "anchor"
    assert batch_kind("rolling-2026-06-06") == "rolling"
    assert batch_date("rolling-2026-06-06") == "2026-06-06"


# ── cut_manifest (the monthly refresh: add new rolling, retire oldest) ───────


def test_cut_below_cap_adds_without_retiring():
    """Cut below cap adds without retiring."""
    m = _manifest(["anchor-2026-05-02", "rolling-2026-06-06"])
    new, retired = cut_manifest(m, "rolling-2026-07-06", at="2026-07-06T00:00:00Z")
    assert retired is None
    assert new["active_batches"] == ["anchor-2026-05-02", "rolling-2026-06-06", "rolling-2026-07-06"]
    assert new["history"][-1] == {
        "event": "cut",
        "batch_id": "rolling-2026-07-06",
        "retired": None,
        "at": "2026-07-06T00:00:00Z",
    }


def test_cut_at_cap_retires_oldest_rolling_only():
    """Cut at cap retires oldest rolling only."""
    m = _manifest(
        [
            "anchor-2026-05-02",
            "rolling-2026-02-06",  # oldest rolling
            "rolling-2026-03-06",
            "rolling-2026-04-06",
            "rolling-2026-05-06",
        ]
    )
    new, retired = cut_manifest(m, "rolling-2026-06-06", at="2026-06-06T00:00:00Z")
    assert retired == "rolling-2026-02-06"
    assert "rolling-2026-02-06" not in new["active_batches"]
    assert "rolling-2026-06-06" in new["active_batches"]
    # exactly 5 active, anchor preserved, 4 rolling
    assert len(new["active_batches"]) == 5
    assert "anchor-2026-05-02" in new["active_batches"]
    assert sum(1 for b in new["active_batches"] if batch_kind(b) == "rolling") == 4


def test_cut_never_retires_anchor():
    """Cut never retires anchor."""
    m = _manifest(
        [
            "anchor-2026-05-02",
            "rolling-2026-03-06",
            "rolling-2026-04-06",
            "rolling-2026-05-06",
            "rolling-2026-06-06",
        ]
    )
    new, retired = cut_manifest(m, "rolling-2026-07-06", at="2026-07-06T00:00:00Z")
    assert batch_kind(retired) == "rolling"
    assert "anchor-2026-05-02" in new["active_batches"]


def test_cut_rejects_duplicate_batch():
    """Cut rejects duplicate batch."""
    m = _manifest(["anchor-2026-05-02", "rolling-2026-06-06"])
    with pytest.raises(ValueError):
        cut_manifest(m, "rolling-2026-06-06", at="2026-06-06T00:00:00Z")


def test_cut_rejects_non_rolling_batch():
    """The anchor is never cut/retired; cut_manifest only adds rolling batches."""
    m = _manifest(["anchor-2026-05-02"])
    with pytest.raises(ValueError):
        cut_manifest(m, "anchor-2026-07-06", at="2026-07-06T00:00:00Z")


def test_cut_does_not_mutate_input_manifest():
    """Cut does not mutate input manifest."""
    m = _manifest(["anchor-2026-05-02", "rolling-2026-06-06"])
    cut_manifest(m, "rolling-2026-07-06", at="2026-07-06T00:00:00Z")
    assert m["active_batches"] == ["anchor-2026-05-02", "rolling-2026-06-06"]
    assert m["history"] == []


# ── gateability + re-baseline (the swap-night comparability rule) ────────────


def test_batch_gateable_only_with_prior_baseline():
    """Batch gateable only with prior baseline."""
    assert is_gateable("rolling-2026-06-06", prior_baseline_batches={"anchor-2026-05-02"}) is False
    assert is_gateable("anchor-2026-05-02", prior_baseline_batches={"anchor-2026-05-02"}) is True


def test_is_rebaseline_when_active_set_changed():
    """Is rebaseline when active set changed."""
    base = {"anchor-2026-05-02", "rolling-2026-05-06"}
    assert is_rebaseline({"anchor-2026-05-02", "rolling-2026-06-06"}, base) is True
    assert is_rebaseline({"anchor-2026-05-02", "rolling-2026-05-06"}, base) is False


# ── S3 keys (never git) ──────────────────────────────────────────────────────


def test_s3_key_builders():
    """S3 key builders."""
    assert manifest_key() == "benchmark/manifest.json"
    assert batch_fixtures_key("anchor-2026-05-02") == "benchmark/batches/anchor-2026-05-02.fixtures.json"
    assert batch_gold_key("rolling-2026-06-06") == "benchmark/batches/rolling-2026-06-06.gold.jsonl"
    assert (
        candidate_pool_key("abc1234", "rolling-2026-06-06")
        == "benchmark/candidate_pools/abc1234/rolling-2026-06-06.pool.json"
    )
    assert recall_result_key("2026-06-06") == "benchmark/results/recall/2026-06-06.json"
    assert archive_key("rolling-2026-02-06.fixtures.json") == "benchmark/archive/rolling-2026-02-06.fixtures.json"


def test_s3_keys_honor_base_prefix():
    """S3 keys honor base prefix."""
    assert manifest_key(base_prefix="ai/workforce") == "ai/workforce/benchmark/manifest.json"


def test_part_b_gold_and_eval_keys():
    """Part b gold and eval keys."""
    from memory_evaluation.benchmark_dataset import eval_key, gold_qa_key

    assert gold_qa_key("2026-06-07") == "benchmark/gold_qa/2026-06-07.jsonl"
    assert eval_key("2026-06-07") == "benchmark/eval/2026-06-07.json"
    assert gold_qa_key("2026-06-07", base_prefix="p") == "p/benchmark/gold_qa/2026-06-07.jsonl"


def test_frozen_gold_key():
    """Frozen gold key."""
    from memory_evaluation.benchmark_dataset import frozen_gold_key

    assert frozen_gold_key("abc1234") == "benchmark/gold_qa/frozen/abc1234.jsonl"
    assert frozen_gold_key("abc1234", base_prefix="p") == "p/benchmark/gold_qa/frozen/abc1234.jsonl"
