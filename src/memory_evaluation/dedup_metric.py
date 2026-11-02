"""Dedup-precision metric (Sub-design C INC 2 / design §3 Instrument 2).

Measures canonicalization quality on a hand-labeled fixture. **PRECISION-gated**:
a false merge (collapsing two distinct people, e.g. typing Kai as a person and
merging him into person-Kai) is the catastrophic error, so the gate trips on a
precision drop — not recall (a missed merge is recoverable).

Pure: the ``predict_merge`` / ``predict_is_entity`` functions (the real
canonicalization) are supplied by the caller and land with INC 5. INC 2 fixes
the metric + fixture shape.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_dedup_pairs(path: str | Path) -> dict[str, Any]:
    """Load the dedup fixture (``merge_pairs`` + ``entity_eligibility``)."""
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def dedup_precision_recall_f1(
    merge_pairs: list[dict[str, Any]],
    predict_merge: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Precision / recall / F1 of the merge decision over labeled pairs.

    A predicted merge that contradicts ``should_merge=False`` is a **false
    merge** (``fp``) — the catastrophic, gated error.

    Edge-case conventions (degenerate denominators): with zero predicted merges
    ``precision = 1.0`` (no false merges); with zero gold-positive pairs
    (``tp + fn == 0``) ``recall = 1.0`` (nothing to recall ⇒ vacuously complete).
    Since the gate is precision-only, the vacuous recall never affects pass/fail.
    """
    tp = fp = fn = tn = 0
    for pair in merge_pairs:
        predicted = bool(predict_merge(pair))
        gold = bool(pair["should_merge"])
        if predicted and gold:
            tp += 1
        elif predicted and not gold:
            fp += 1
        elif not predicted and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": len(merge_pairs),
    }


def entity_eligibility_accuracy(
    rows: list[dict[str, Any]],
    predict_is_entity: Callable[[str], bool],
) -> dict[str, Any]:
    """Accuracy of the entity-eligibility decision (e.g. ``access_token`` must
    be rejected as a credential term, not minted as an entity)."""
    correct = sum(1 for row in rows if bool(predict_is_entity(row["surface"])) == bool(row["should_be_entity"]))
    n = len(rows)
    return {"accuracy": correct / n if n else 1.0, "correct": correct, "n": n}
