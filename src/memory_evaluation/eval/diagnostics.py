"""Report-only repeat diagnostics; lexical correctness is not semantic calibration."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from memory_evaluation.eval.answering import build_answer_scoreboard, comparable_pair
from memory_evaluation.eval.fixtures import FixtureSet


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "std": math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)),
        "min": min(values),
        "max": max(values),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def paired_comparisons(rows: list[dict[str, Any]], arms: list[str]) -> dict[str, Any]:
    """Compare only the same fixture/question with matching required evidence hashes."""
    keyed = {(r["fixture_id"], r["question_id"], r["arm"]): r for r in rows}
    cases = {(f, q) for f, q, _ in keyed}
    out = {}
    for left, right in combinations(arms, 2):
        pairs = []
        for f, q in cases:
            a, b = keyed.get((f, q, left)), keyed.get((f, q, right))
            if a and b and a["status"] == b["status"] == "ok" and comparable_pair(a, b):
                pairs.append((a, b))
        out[f"{left}__{right}"] = {
            "n_expected": len(cases),
            "n_valid": len(pairs),
            "n_invalid": len(cases) - len(pairs),
            "left_correct": sum(a["verdict"] == "correct" for a, _ in pairs),
            "right_correct": sum(b["verdict"] == "correct" for _, b in pairs),
            "correctness_delta": (
                sum((b["verdict"] == "correct") - (a["verdict"] == "correct") for a, b in pairs) / len(pairs)
            )
            if pairs
            else None,
        }
    return out


def summarize_repeats(reports: list[dict[str, Any]], fixture_set: FixtureSet) -> dict[str, Any]:
    """Aggregate three requested repeats without turning an offline score into a gate."""
    _validate_repeats(reports, fixture_set)
    boards = [build_answer_scoreboard(report, fixture_set) for report in reports]
    arms = list(reports[0]["arms"]) if reports else []
    if any(list(report["arms"]) != arms for report in reports):
        raise ValueError("repeat arm manifests differ")
    per_arm = {}
    for arm in arms:
        aggregates = [board["aggregate"]["per_arm"][arm] for board in boards]
        rows = [row for board in boards for row in board["rows"] if row["arm"] == arm]
        latency = {}
        for phase in ("retrieval_s", "projection_s", "answer_s"):
            values = [row.get("timing", {}).get(phase) for row in rows if row["status"] == "ok"]
            measured = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
            latency[phase] = {
                "n_measured": len(measured),
                "n_missing": len(rows) - len(measured),
                "p50": _percentile(measured, 0.5),
                "p95": _percentile(measured, 0.95),
            }
        groups = {}
        for key in ("availability", "distance_user_turns", "split"):
            tallies: dict[str, dict[str, int]] = {}
            for row in rows:
                value = str(row.get(key, "unknown"))
                bucket = tallies.setdefault(value, {"n": 0, "n_valid": 0, "n_invalid": 0, "correct": 0})
                bucket["n"] += 1
                bucket["n_valid"] += row["status"] == "ok"
                bucket["n_invalid"] += row["status"] != "ok"
                bucket["correct"] += row.get("verdict") == "correct" and row["status"] == "ok"
            groups[key] = tallies
        per_arm[arm] = {
            "n": sum(a["n"] for a in aggregates),
            "n_valid": sum(a["n_valid"] for a in aggregates),
            "n_invalid": sum(a["n_invalid"] for a in aggregates),
            "correctness": _distribution([a["accuracy"] for a in aggregates if a["accuracy"] is not None]),
            "abstention": _distribution(
                [a["abstention_accuracy"] for a in aggregates if a["abstention_accuracy"] is not None]
            ),
            "abstention_rate": _distribution(
                [a["abstention_rate"] for a in aggregates if a["abstention_rate"] is not None]
            ),
            "latency": latency,
            "by_label": groups,
        }
    return {
        "format": "kai-memory-repeat-diagnostics/v1",
        "report_only": True,
        "repeat_target": 3,
        "repeat_count": len(reports),
        "repeat_target_met": len(reports) == 3,
        "per_arm": per_arm,
        "latency": {"extraction_s": _extraction_latency(reports, fixture_set)},
        "paired": [paired_comparisons(board["rows"], arms) for board in boards],
        "limits": [
            "conservative lexical correctness; no semantic calibration",
            "token usage may be unknown",
            "retrieval/projection/answer latency uses valid measured rows; extraction uses successful fixtures once",
            "repeats are not independent human evidence",
        ],
    }


def _validate_repeats(reports: list[dict[str, Any]], fixture_set: FixtureSet) -> None:
    """A repeat varies stochastic output, not the experimental treatment."""
    if any(report.get("fixture_set", {}).get("sha256") != fixture_set.sha256 for report in reports):
        raise ValueError("repeat fixture hashes differ or are missing")

    def stable(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in {"calls", "spent_usd_estimate"}}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value

    def config(report: dict[str, Any]) -> dict[str, Any]:
        return {
            key: stable(report.get(key)) for key in ("extractor", "answerer", "embedding_metadata", "embedder", "judge")
        }

    if reports and any(config(report) != config(reports[0]) for report in reports[1:]):
        raise ValueError("repeat extractor/embedding/model configuration differs")
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for report in reports:
        for fixture in report["results"]:
            for row in fixture.get("answers") or []:
                key = (fixture["id"], row["question_id"], row["arm"])
                identity = {
                    k: row.get(k)
                    for k in (
                        "model_sha256",
                        "prompt_sha256",
                        "stageC_policy_sha256",
                        "input_sha256",
                        "history_sha256",
                        "paired_metadata",
                    )
                }
                if key in observed and identity != observed[key]:
                    raise ValueError("repeat model/prompt/policy/input metadata differs")
                observed[key] = identity


def _extraction_latency(reports: list[dict[str, Any]], fixture_set: FixtureSet) -> dict[str, Any]:
    """One checkpoint-duration total per successful fixture, never per question or arm."""
    measured: list[float] = []
    for report in reports:
        by_id = {result["id"]: result for result in report["results"]}
        for fixture in fixture_set.fixtures:
            result = by_id.get(fixture["id"], {})
            checkpoints = result.get("checkpoints")
            if result.get("status") != "ok" or not isinstance(checkpoints, list) or not checkpoints:
                continue
            if "checkpoints" in fixture and len(checkpoints) != len(fixture["checkpoints"]):
                continue
            durations: list[float] = []
            for checkpoint in checkpoints:
                value = checkpoint.get("extraction_s")
                if (
                    checkpoint.get("status") not in (None, "ok")
                    or checkpoint.get("error")
                    or checkpoint.get("store_errors")
                    or type(value) not in (int, float)
                ):
                    break
                try:
                    duration = float(value)
                except (ValueError, OverflowError):
                    break
                if not math.isfinite(duration) or duration < 0:
                    break
                durations.append(duration)
            if len(durations) == len(checkpoints) and math.isfinite(sum(durations)):
                measured.append(sum(durations))
    return {
        "n_measured": len(measured),
        "n_missing": len(reports) * len(fixture_set.fixtures) - len(measured),
        "p50": _percentile(measured, 0.5),
        "p95": _percentile(measured, 0.95),
    }
