"""Repeat diagnostics use explicit valid denominators and report-only statistics."""

from copy import deepcopy

import pytest

from memory_evaluation.eval.answering import PAIR_HASHES
from memory_evaluation.eval.diagnostics import summarize_repeats
from memory_evaluation.eval.fixtures import FixtureSet


def fixture_set():
    return FixtureSet(
        path="p",
        sha256="c",
        fixtures=[
            {
                "id": "f",
                "split": "held_out",
                "_provenance": {"source": "synthetic", "content_sha256": "c"},
                "expected": {
                    "questions": [
                        {
                            "id": "q",
                            "question": "Where?",
                            "expect_any": ["Leeds"],
                            "availability": "memory_only",
                            "distance_user_turns": 8,
                        }
                    ]
                },
            }
        ],
    )


def report(answer="Leeds"):
    rows = [
        {
            "question_id": "q",
            "arm": arm,
            "answer": answer,
            "error": None,
            "judge_error": None,
            "judge_raw": None,
            "paired_metadata": "m4/v1",
            "store_after_sha256": "a" * 64,
            "timing": {"extraction_s": 0.1, "retrieval_s": 0.2, "answer_s": 0.3, "projection_s": 0.1},
            **{key: "a" * 64 for key in PAIR_HASHES},
        }
        for arm in ("no_memory", "last_message", "windowed")
    ]
    return {
        "arms": ["no_memory", "last_message", "windowed"],
        "judge": None,
        "answerer": {"name": "fake", "quality": "wiring-only"},
        "extractor": {},
        "embedder": "fake",
        "fixture_set": {"sha256": "c"},
        "mutation_check": {},
        "results": [{"id": "f", "status": "ok", "answers": rows}],
    }


def test_three_repeat_mean_std_range_latency_and_valid_denominators():
    raw = [report("Leeds"), report("York"), report("")]
    diagnostics = summarize_repeats(raw, fixture_set())
    arm = diagnostics["per_arm"]["last_message"]
    assert diagnostics["report_only"] and diagnostics["repeat_target_met"]
    assert (arm["n"], arm["n_valid"], arm["n_invalid"]) == (3, 2, 1)
    assert arm["correctness"] == {"n": 2, "mean": 0.5, "std": 0.5, "min": 0.0, "max": 1.0}
    assert arm["latency"]["answer_s"]["p95"] == 0.3
    assert arm["latency"]["projection_s"]["p95"] == 0.1
    assert arm["by_label"]["availability"]["memory_only"]["n_invalid"] == 1
    assert diagnostics["paired"][-1]["no_memory__last_message"]["n_valid"] == 0


def test_repeats_reject_mismatched_arm_manifest():
    raw = report()
    other = deepcopy(raw)
    other["arms"] = ["no_memory"]
    other["results"][0]["answers"] = other["results"][0]["answers"][:1]
    with pytest.raises(ValueError, match="manifests"):
        summarize_repeats([raw, other], fixture_set())


@pytest.mark.parametrize("change", ["fixture", "extractor", "embedder", "model", "prompt", "policy"])
def test_repeat_aggregation_rejects_different_experiments(change):
    raw, other = report(), report()
    if change == "fixture":
        other["fixture_set"]["sha256"] = "other"
    elif change in {"extractor", "embedder"}:
        other[change] = {"name": "other", "model": "other"}
    elif change == "model":
        other["answerer"]["model"] = "other"
    else:
        key = "prompt_sha256" if change == "prompt" else "stageC_policy_sha256"
        other["results"][0]["answers"][0][key] = "b" * 64
    with pytest.raises(ValueError, match="repeat"):
        summarize_repeats([raw, other], fixture_set())


def test_repeat_store_changes_are_allowed_but_budget_consumption_is_not_configuration():
    raw, other = report(), report()
    raw["extractor"] = {"name": "live", "model": "fake", "calls": 1, "spent_usd_estimate": 0.01}
    other["extractor"] = {"name": "live", "model": "fake", "calls": 2, "spent_usd_estimate": 0.02}
    for row in other["results"][0]["answers"]:
        row["store_sha256"] = row["store_after_sha256"] = "b" * 64
    assert summarize_repeats([raw, other], fixture_set())["repeat_count"] == 2


def test_extraction_latency_weights_fixtures_once_despite_unequal_question_counts():
    fixtures = fixture_set()
    second = deepcopy(fixtures.fixtures[0])
    second["id"] = "g"
    second["expected"]["questions"] = [
        {**deepcopy(second["expected"]["questions"][0]), "id": f"q{i}"} for i in range(9)
    ]
    fixtures.fixtures.append(second)
    raw = report()
    first = raw["results"][0]
    first["checkpoints"] = [{"extraction_s": 0.4}, {"extraction_s": 0.6}]
    other = deepcopy(first)
    other["id"] = "g"
    other["checkpoints"] = [{"extraction_s": 9.0}]
    other["answers"] = [{**deepcopy(row), "question_id": f"q{i}"} for i in range(9) for row in first["answers"]]
    raw["results"].append(other)
    for result in raw["results"]:
        for row in result["answers"]:
            row["timing"]["extraction_s"] = 999.0  # Row copies are not extraction observations.
    diagnostics = summarize_repeats([raw], fixtures)
    assert diagnostics["latency"]["extraction_s"] == {
        "n_measured": 2,
        "n_missing": 0,
        "p50": 5.0,
        "p95": pytest.approx(8.6),
    }
    assert all("extraction_s" not in arm["latency"] for arm in diagnostics["per_arm"].values())
    assert diagnostics["per_arm"]["windowed"]["latency"]["answer_s"]["n_measured"] == 10


@pytest.mark.parametrize(
    "checkpoints",
    [
        None,
        [],
        [{}],
        [{"extraction_s": None}],
        [{"extraction_s": "1"}],
        [{"extraction_s": True}],
        [{"extraction_s": -1}],
        [{"extraction_s": float("nan")}],
        [{"extraction_s": float("inf")}],
        [{"extraction_s": 1.0}, {}],
        [{"extraction_s": 1.0, "store_errors": ["failed"]}],
        [{"extraction_s": 1.0, "status": "error"}],
    ],
)
def test_missing_or_invalid_checkpoint_duration_is_missing_not_zero(checkpoints):
    raw = report()
    raw["results"][0]["checkpoints"] = checkpoints
    diagnostics = summarize_repeats([raw], fixture_set())
    assert diagnostics["latency"]["extraction_s"] == {
        "n_measured": 0,
        "n_missing": 1,
        "p50": None,
        "p95": None,
    }


def test_extraction_latency_counts_failed_and_absent_fixtures_as_missing_but_keeps_explicit_zero():
    zero, failed, missing = report(), report(), report()
    zero["results"][0]["checkpoints"] = [{"extraction_s": 0.0}]
    failed["results"][0].update(status="error", checkpoints=[{"extraction_s": 10.0}])
    missing["results"] = []
    diagnostics = summarize_repeats([zero, failed, missing], fixture_set())
    assert diagnostics["latency"]["extraction_s"] == {
        "n_measured": 1,
        "n_missing": 2,
        "p50": 0.0,
        "p95": 0.0,
    }
