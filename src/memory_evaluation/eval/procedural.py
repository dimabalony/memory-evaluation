"""Offline preparation for DB-883; no provider, storage, or semantic scoring calls."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from memory_evaluation.eval.fixtures import content_sha256, provenance_issues

FORMAT = "kai-memory-procedural-preparation/v1"


def prepare_serving_pairs(fixture: dict[str, Any]) -> dict[str, Any]:
    """Freeze independent ON/OFF answer inputs; all observations remain unrun.

    Teaching and superseded text stay outside both answer inputs. ON differs
    only by the current procedural rule. Each turn is an independent probe so
    an earlier generated answer cannot contaminate the other arm's history.
    This prepares an experiment, not a runnable provider or an efficacy result.
    """
    errors = provenance_issues(fixture)
    if fixture.get("format") != FORMAT:
        errors.append("unsupported procedural preparation format")
    provenance = fixture.get("_provenance")
    if not isinstance(provenance, dict) or provenance.get("source") != "synthetic":
        errors.append("only synthetic preparation is supported")
    for key in ("repeats", "hosts", "cases", "model_settings", "calibration"):
        if key not in fixture:
            errors.append(f"missing required field: {key}")
    if errors:
        raise ValueError("; ".join(errors))
    repeats = fixture["repeats"]
    if type(repeats) is not int or repeats < 2:
        raise ValueError("at least two repeats are required to measure variance")
    hosts = fixture["hosts"]
    cases = fixture["cases"]
    if not hosts or len(hosts) != len(set(hosts)) or not cases:
        raise ValueError("nonempty unique hosts and cases required")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case id")
    pairs = []
    for case in cases:
        turns = case["turns"]
        window = case["violation_window"]
        expected_ids = list(range(window["start_turn"], window["end_turn"] + 1))
        if not expected_ids or [turn["id"] for turn in turns] != expected_ids:
            raise ValueError("violation window must cover exactly the ordered probe turns")
        if not case["rule"].strip() or any(type(turn["applicable"]) is not bool for turn in turns):
            raise ValueError("rule and explicit applicability labels required")
        for host in hosts:
            for repeat in range(repeats):
                for turn in turns:
                    common = {
                        "host": host,
                        "case_id": case["id"],
                        "repeat": repeat,
                        "turn_id": turn["id"],
                        "input": turn["input"],
                        "history": [],
                        "model_settings": fixture["model_settings"],
                        "violation_window": window,
                    }
                    # JSON roundtrip gives arms independent nested inputs and pins bytes.
                    encoded = json.dumps(common, sort_keys=True, ensure_ascii=False)
                    arms = []
                    for enabled in (False, True):
                        arm = json.loads(encoded)
                        arm.update(
                            serving="on" if enabled else "off",
                            procedural_rules=[case["rule"]] if enabled else [],
                            status="unrun",
                        )
                        arms.append(arm)
                    pairs.append(
                        {
                            "common_input_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                            "applicable": turn["applicable"],
                            "arms": arms,
                        }
                    )
    observations = len(pairs) * 2
    return {
        "format": FORMAT,
        "fixture_sha256": content_sha256(fixture),
        "quality": "offline-preparation-only",
        "semantic_acceptance": "unmeasured",
        "calibration": fixture["calibration"],
        "pairs": pairs,
        "denominators": {
            "eligible_pairs": len(pairs),
            "applicable_pairs": sum(pair["applicable"] for pair in pairs),
            "not_applicable_pairs": sum(not pair["applicable"] for pair in pairs),
            "valid_pairs": 0,
            "invalid_pairs": 0,
            "unrun_pairs": len(pairs),
            "compliance_observations": 0,
        },
        "compliance_delta": None,
        "repeat_variance": {"n_measured": 0, "std": None},
        "latency": {"n_measured": 0, "n_missing": observations, "p50_s": None, "p95_s": None},
        "cost": {"provider_calls": 0, "preparation_usd": 0, "efficacy_usd": None},
        "gaps": [
            "No provider answers, extraction, embeddings, or procedural capture executed.",
            "Serving selection and attribution are separate contract tests; these are planned answer inputs.",
            "Model/account/budget require a fresh approved bounded pilot; pending model is not runnable.",
            "Calibration, observed applicability/compliance, repeat variance, latency and efficacy cost unmeasured.",
        ],
    }
