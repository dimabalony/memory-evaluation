"""PRECISION gate for the cross-script canonicalization function.

Design §3 Instrument 2: the dedup-precision metric is PRECISION-gated — a false
merge (collapsing two distinct people) is the catastrophic error. This wires the
REAL canonicalization (`same_canonical_entity`) as the predictor and asserts it
makes ZERO false merges over the hand-labeled fixture, including hardened
cross-script near-miss negatives.

Recall is intentionally NOT gated (under-merge is acceptable; a missed merge is
recoverable, a false merge is not).
"""

from __future__ import annotations

from pathlib import Path

from kai_memory.canonicalization import same_canonical_entity
from memory_evaluation.dedup_metric import (
    dedup_precision_recall_f1,
    load_dedup_pairs,
)

FIXTURE = Path(__file__).parent / "fixtures" / "priming_benchmark" / "dedup_pairs.json"


def _predict_merge(pair: dict) -> bool:
    return same_canonical_entity(
        pair["surface_a"],
        pair["surface_b"],
        pair.get("type_a"),
        pair.get("type_b"),
    )


def test_canonicalization_predictor_is_precision_1_zero_false_merge():
    """Hard gate: precision == 1.0 (no false merge) over the whole fixture."""
    data = load_dedup_pairs(FIXTURE)
    res = dedup_precision_recall_f1(data["merge_pairs"], _predict_merge)
    assert res["fp"] == 0, f"FALSE MERGE(S) present — catastrophic: {res}"
    assert res["precision"] == 1.0, res


def test_clean_cross_script_positives_are_recalled():
    """Sanity: the function DOES merge the clean transliteration class (it is
    not a trivially-precise no-op that merges nothing)."""
    data = load_dedup_pairs(FIXTURE)
    res = dedup_precision_recall_f1(data["merge_pairs"], _predict_merge)
    assert res["tp"] >= 5, f"expected the clean-translit positives to merge: {res}"


def test_fixture_contains_hardened_cross_script_negatives():
    """The hardened near-miss negatives that pin the scope boundary must be in
    the committed fixture (otherwise the gate can't catch a regression)."""
    data = load_dedup_pairs(FIXTURE)
    merge = data["merge_pairs"]

    def _has_negative(a, b):
        return any({p["surface_a"], p["surface_b"]} == {a, b} and p["should_merge"] is False for p in merge)

    # Diminutive↔formal and spelling/ending variants: OUT of scope, must NOT merge.
    assert _has_negative("Миша", "Михаил"), "diminutive↔formal guard missing"
    assert _has_negative("Петрав", "Petrov"), "spelling-variant guard missing"
    # Cross-script distinct people whose romanizations look similar.
    assert _has_negative("Алёна", "Елена")
    assert _has_negative("Олег", "Олеся")
    assert _has_negative("Pavel", "Павло")


def test_fixture_contains_clean_cross_script_positive():
    """The clean cross-script person merge the function is built to recover."""
    data = load_dedup_pairs(FIXTURE)
    assert any(
        {p["surface_a"], p["surface_b"]} == {"Миша", "Misha"} and p["should_merge"] is True for p in data["merge_pairs"]
    )
