"""Dedup-precision metric + fixture (Sub-design C INC 2, A2 / design §3 Instrument 2).

PRECISION-gated: a false merge (collapsing two distinct people) is the
catastrophic error, so the gate trips on a precision drop — not recall. The
metric is pure (a predict function is supplied; the real canonicalization lands
with INC 5). The fixture is hand-labeled seed data from the deferred cluster.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from memory_evaluation.dedup_metric import (
    dedup_precision_recall_f1,
    entity_eligibility_accuracy,
    load_dedup_pairs,
)

FIXTURE = Path(__file__).parent / "fixtures" / "priming_benchmark" / "dedup_pairs.json"


def test_load_dedup_pairs_preserves_unicode_under_ascii_locale(tmp_path):
    fixture = tmp_path / "dedup_pairs.json"
    data = {
        "merge_pairs": [{"surface_a": "Miyuki", "surface_b": "美雪", "should_merge": True}],
        "entity_eligibility": [{"surface": "美雪", "should_be_entity": True}],
    }
    fixture.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8=0",
            "-c",
            "import json, locale, sys; "
            "from memory_evaluation.dedup_metric import load_dedup_pairs; "
            "locale.setlocale(locale.LC_CTYPE, 'C'); "
            "assert not sys.flags.utf8_mode; "
            "assert locale.getencoding().lower() not in ('utf-8', 'utf8'); "
            "print(json.dumps(load_dedup_pairs(sys.argv[1])))",
            str(fixture),
        ],
        cwd=tmp_path,
        env={**os.environ, "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == data


def test_precision_recall_f1_counts_false_merge_as_fp():
    pairs = [
        {"surface_a": "Katya", "surface_b": "Катя", "should_merge": True},
        {"surface_a": "Pavel", "surface_b": "Павел", "should_merge": True},
        {"surface_a": "Kai", "surface_b": "Kai", "should_merge": False},  # agent vs person
        {"surface_a": "Misha", "surface_b": "Olga", "should_merge": False},
    ]

    # Predictor that merges the first true pair (TP) AND wrongly merges Kai (FP).
    def predict(pair):
        return pair["surface_a"] in {"Katya", "Kai"}

    res = dedup_precision_recall_f1(pairs, predict)
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 1 and res["tn"] == 1
    assert res["precision"] == 0.5  # 1 / (1 + 1 false merge)
    assert res["recall"] == 0.5  # 1 / (1 + 1 missed)
    assert round(res["f1"], 4) == 0.5


def test_perfect_predictor_is_precision_1():
    pairs = [
        {"surface_a": "alpha", "surface_b": "alpha-alias", "should_merge": True},
        {"surface_a": "x", "surface_b": "y", "should_merge": False},
    ]
    res = dedup_precision_recall_f1(pairs, lambda p: p["should_merge"])
    assert res["precision"] == 1.0 and res["recall"] == 1.0 and res["f1"] == 1.0


def test_entity_eligibility_accuracy():
    rows = [
        {"surface": "access_token", "should_be_entity": False},
        {"surface": "Misha", "should_be_entity": True},
    ]
    # Predictor that correctly rejects access_token, accepts Misha.
    res = entity_eligibility_accuracy(rows, lambda s: s != "access_token")
    assert res["accuracy"] == 1.0
    assert res["n"] == 2


# ── the committed fixture itself ─────────────────────────────────────────────


def test_fixture_loads_and_has_seeded_cluster_rows():
    data = load_dedup_pairs(FIXTURE)
    merge = data["merge_pairs"]
    elig = data["entity_eligibility"]
    assert len(merge) >= 20, "design asks for ~30-60 rows; at least a solid seed"

    def _has_merge(a, b, should):
        return any({p["surface_a"], p["surface_b"]} == {a, b} and p["should_merge"] is should for p in merge)

    # Cyrillic-alias dedup (should merge).
    assert _has_merge("Katya", "Катя", True)
    # Kai agent-vs-person must NEVER merge (the entity-typing cluster).
    assert any(p["should_merge"] is False and "Kai" in {p["surface_a"], p["surface_b"]} for p in merge)
    # access_token is not an entity (credential-term cluster).
    assert any(r["surface"] == "access_token" and r["should_be_entity"] is False for r in elig)

    # Every row is well-typed.
    for p in merge:
        assert isinstance(p["should_merge"], bool)
        assert p["surface_a"] and p["surface_b"]
    for r in elig:
        assert isinstance(r["should_be_entity"], bool)


def test_fixture_is_valid_json():
    json.loads(FIXTURE.read_text())
