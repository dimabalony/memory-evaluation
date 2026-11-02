"""DB-883 preparation cannot masquerade as provider efficacy evidence."""

import json
import re
from pathlib import Path

import pytest

from memory_evaluation.eval.fixtures import content_sha256
from memory_evaluation.eval.procedural import prepare_serving_pairs

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "tests/fixtures/memory_m5/efficacy_preparation.json").read_text())


def test_matched_pairs_have_only_the_serving_intervention_and_fixed_windows():
    report = prepare_serving_pairs(fixture())
    assert len(report["pairs"]) == 54  # 3 cases x 3 turns x 3 repeats x 2 hosts
    for pair in report["pairs"]:
        off, on = pair["arms"]
        assert off["procedural_rules"] == [] and len(on["procedural_rules"]) == 1
        assert off["serving"] == "off" and on["serving"] == "on"
        assert {k: v for k, v in off.items() if k not in {"serving", "procedural_rules"}} == {
            k: v for k, v in on.items() if k not in {"serving", "procedural_rules"}
        }
        assert off["violation_window"] == {"start_turn": 1, "end_turn": 3}
        assert off["history"] == on["history"] == []
        assert "superseded_rule" not in on
    report["pairs"][0]["arms"][0]["model_settings"]["max_tokens"] = 1
    assert report["pairs"][0]["arms"][1]["model_settings"]["max_tokens"] == 256


def test_preparation_preserves_unrun_denominators_and_unknown_measurements():
    report = prepare_serving_pairs(fixture())
    assert report["denominators"] == {
        "eligible_pairs": 54,
        "applicable_pairs": 36,
        "not_applicable_pairs": 18,
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "unrun_pairs": 54,
        "compliance_observations": 0,
    }
    assert report["semantic_acceptance"] == "unmeasured" and report["quality"] == "offline-preparation-only"
    assert report["compliance_delta"] is None
    assert report["repeat_variance"] == {"n_measured": 0, "std": None}
    assert report["latency"] == {"n_measured": 0, "n_missing": 108, "p50_s": None, "p95_s": None}
    assert report["cost"] == {"provider_calls": 0, "preparation_usd": 0, "efficacy_usd": None}
    assert all(arm["status"] == "unrun" for pair in report["pairs"] for arm in pair["arms"])


@pytest.mark.parametrize(
    "change,reason",
    [
        ("hash", "_provenance.content_sha256 does not match the content (re-stamp after review)"),
        ("window", "violation window must cover exactly the ordered probe turns"),
        ("duplicate", "duplicate case id"),
        ("applicability", "rule and explicit applicability labels required"),
        ("repeat", "at least two repeats are required to measure variance"),
        ("repeat_bool", "at least two repeats are required to measure variance"),
        ("repeat_float", "at least two repeats are required to measure variance"),
        ("format", "unsupported procedural preparation format"),
        ("source", "only synthetic preparation is supported"),
        ("empty_hosts", "nonempty unique hosts and cases required"),
        ("duplicate_hosts", "nonempty unique hosts and cases required"),
        ("empty_cases", "nonempty unique hosts and cases required"),
        ("blank_rule", "rule and explicit applicability labels required"),
    ],
)
def test_malformed_preparation_is_rejected(change, reason):
    data = fixture()
    if change == "hash":
        data["cases"][0]["rule"] += " changed"
    else:
        if change == "window":
            data["cases"][0]["violation_window"]["end_turn"] = 4
        elif change == "duplicate":
            data["cases"].append(data["cases"][0])
        elif change == "applicability":
            data["cases"][0]["turns"][0]["applicable"] = "yes"
        elif change.startswith("repeat"):
            data["repeats"] = {"repeat": 1, "repeat_bool": True, "repeat_float": 3.0}[change]
        elif change == "format":
            data["format"] = "unsupported"
        elif change == "source":
            data["_provenance"].update(source="sanitized", approved_by="synthetic test reviewer")
        elif change == "empty_hosts":
            data["hosts"] = []
        elif change == "duplicate_hosts":
            data["hosts"] = ["atlas", "atlas"]
        elif change == "empty_cases":
            data["cases"] = []
        elif change == "blank_rule":
            data["cases"][0]["rule"] = "   "
        data["_provenance"]["content_sha256"] = content_sha256(data)
    with pytest.raises(ValueError, match=f"^{re.escape(reason)}$"):
        prepare_serving_pairs(data)


@pytest.mark.parametrize("provenance", [None, [], ["synthetic"]])
def test_malformed_provenance_preserves_aggregated_validation_errors(provenance):
    data = fixture()
    data["_provenance"] = provenance
    data["format"] = "unsupported"
    # An invalid provenance container cannot hold a restamped hash.
    reason = "missing _provenance; unsupported procedural preparation format; only synthetic preparation is supported"
    with pytest.raises(ValueError, match=f"^{re.escape(reason)}$"):
        prepare_serving_pairs(data)


@pytest.mark.parametrize("key", ["repeats", "hosts", "cases", "model_settings", "calibration"])
def test_missing_required_fixture_field_has_validation_error(key):
    data = fixture()
    del data[key]
    data["_provenance"]["content_sha256"] = content_sha256(data)
    with pytest.raises(ValueError, match=f"^missing required field: {key}$"):
        prepare_serving_pairs(data)
