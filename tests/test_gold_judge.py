"""Gold eval judge — pure core (Sub-design C INC 2 Part B, STEP 2).

Cross-provider (Gemini) judging of memory-ON vs memory-OFF answers against
human-validated gold, with position-swap averaging and metric aggregation. These
are the pure pieces (prompt construction, verdict parsing, swap resolution,
metrics); the LLM/DB/S3 orchestration runs in prod (scripts/run_gold_eval.py).
"""

from __future__ import annotations

import pytest

from memory_evaluation.gold_judge import (
    aggregate,
    build_pairwise_prompt,
    build_pointwise_prompt,
    paired_bootstrap_ci,
    parse_pairwise_verdict,
    parse_pointwise_verdict,
    swap_winner,
)


# ── pointwise prompt ──────────────────────────────────────────────────────────


def test_pointwise_prompt_fact_embeds_qa_and_asks_for_json_verdict():
    """Pointwise prompt fact embeds qa and asks for json verdict."""
    p = build_pointwise_prompt(
        "What is the user's last name?", "Riverstone", "Their last name is Riverstone.", is_abstention=False
    )
    assert "What is the user's last name?" in p
    assert "Riverstone" in p
    assert "Their last name is Riverstone." in p
    low = p.lower()
    assert "correct" in low and "faithful" in low and "json" in low


def test_pointwise_prompt_abstention_asks_whether_model_declined():
    """Pointwise prompt abstention asks whether model declined."""
    p = build_pointwise_prompt("What is the doctor's name?", "", "I don't have that in memory.", is_abstention=True)
    low = p.lower()
    # an abstention item is graded on whether the model correctly DECLINED
    assert "abstain" in low or "decline" in low
    assert "json" in low


# ── pointwise verdict parsing ─────────────────────────────────────────────────


def test_parse_pointwise_verdict_plain_and_fenced():
    """Parse pointwise verdict plain and fenced."""
    assert parse_pointwise_verdict('{"correct": true, "faithful": true, "abstained": false}') == {
        "correct": True,
        "faithful": True,
        "abstained": False,
    }
    fenced = parse_pointwise_verdict('```json\n{"correct": false, "faithful": true, "abstained": true}\n```')
    assert fenced == {"correct": False, "faithful": True, "abstained": True}


def test_parse_pointwise_verdict_malformed_returns_none():
    """Parse pointwise verdict malformed returns none."""
    assert parse_pointwise_verdict("not json at all") is None
    assert parse_pointwise_verdict("") is None
    # missing required keys → None (don't silently default a score)
    assert parse_pointwise_verdict('{"correct": true}') is None


def test_parse_pointwise_verdict_coerces_string_booleans():
    """String-encoded JSON booleans must coerce correctly — bool('false') is True
    in Python, so a judge emitting the string 'false' must not read as a pass."""
    v = parse_pointwise_verdict('{"correct": "false", "faithful": "true", "abstained": "false"}')
    assert v == {"correct": False, "faithful": True, "abstained": False}


# ── pairwise prompt + verdict ─────────────────────────────────────────────────


def test_pairwise_prompt_embeds_gold_and_both_answers():
    """Pairwise prompt embeds gold and both answers."""
    p = build_pairwise_prompt("Q?", "GOLD-ANSWER", "answer one", "answer two")
    assert "GOLD-ANSWER" in p and "answer one" in p and "answer two" in p
    low = p.lower()
    assert "json" in low and ("winner" in low or "better" in low)


def test_parse_pairwise_verdict():
    """Parse pairwise verdict."""
    assert parse_pairwise_verdict('{"winner": "A"}') == "A"
    assert parse_pairwise_verdict('{"winner": "B"}') == "B"
    assert parse_pairwise_verdict('{"winner": "tie"}') == "tie"
    assert parse_pairwise_verdict("garbage") is None


# ── position-swap resolution ──────────────────────────────────────────────────


def test_swap_winner_agreement_disagreement_and_ties():
    # first call: ON is answer A; second call: ON is answer B (swapped)
    """Swap winner agreement disagreement and ties."""
    assert swap_winner("A", "B") == "on"  # ON wins under both orderings
    assert swap_winner("B", "A") == "off"  # OFF wins under both
    assert swap_winner("A", "A") == "tie"  # contradictory (position bias) → tie
    assert swap_winner("tie", "B") == "on"  # one tie + one ON vote
    assert swap_winner("A", "tie") == "on"
    assert swap_winner("tie", "tie") == "tie"


# ── metric aggregation ────────────────────────────────────────────────────────


def _row(category, should_abstain, on, off, pairwise):
    """Test helper: row."""
    return {"category": category, "should_abstain": should_abstain, "on": on, "off": off, "pairwise": pairwise}


_CORRECT = {"correct": True, "faithful": True, "abstained": False}
_WRONG = {"correct": False, "faithful": True, "abstained": False}
_ABST_OK = {"correct": True, "faithful": True, "abstained": True}
_ABST_FABRICATED = {"correct": False, "faithful": False, "abstained": False}


def test_aggregate_headline_delta_abstention_accuracy_and_pairwise():
    """Aggregate headline delta abstention accuracy and pairwise."""
    rows = [
        _row("fact", False, _CORRECT, _WRONG, "on"),  # memory helped answer
        _row("fact", False, _CORRECT, _CORRECT, "tie"),  # both right
        _row("abstention", True, _ABST_OK, _ABST_FABRICATED, "on"),  # ON declined, OFF hallucinated
    ]
    m = aggregate(rows)
    assert m["n"] == 3
    # overall correctness ON=3/3, OFF=1/3 → delta 2/3
    assert m["correctness"]["on"] == pytest.approx(1.0)
    assert m["correctness"]["off"] == pytest.approx(1 / 3)
    assert m["correctness"]["delta"] == pytest.approx(2 / 3)
    # abstention accuracy is computed over abstention items only
    assert m["abstention_accuracy"]["on"] == pytest.approx(1.0)
    assert m["abstention_accuracy"]["off"] == pytest.approx(0.0)
    assert m["abstention_accuracy"]["n"] == 1
    # fact correctness over fact items only (ON 2/2, OFF 1/2)
    assert m["fact_correctness"]["on"] == pytest.approx(1.0)
    assert m["fact_correctness"]["off"] == pytest.approx(0.5)
    # faithfulness over all (OFF fabricated once → 2/3)
    assert m["faithfulness"]["off"] == pytest.approx(2 / 3)
    # pairwise: 2 ON wins, 0 OFF, 1 tie → winrate (2 + 0.5)/3
    assert m["pairwise"]["on_wins"] == 2
    assert m["pairwise"]["ties"] == 1
    assert m["pairwise"]["on_winrate"] == pytest.approx(2.5 / 3)
    # headline delta carries a paired-bootstrap CI bracketing the point estimate
    lo, hi = m["correctness"]["delta_ci"]
    assert lo <= m["correctness"]["delta"] <= hi
    assert isinstance(m["correctness"]["delta_resolves"], bool)


def test_aggregate_empty_is_safe():
    """Aggregate empty is safe."""
    m = aggregate([])
    assert m["n"] == 0
    assert m["correctness"]["delta"] == 0
    assert m["correctness"]["delta_ci"] == [0.0, 0.0]
    assert m["correctness"]["delta_resolves"] is False
    assert m["abstention_accuracy"]["n"] == 0


# ── paired bootstrap CI on the headline delta (Part A n-too-small lesson) ─────


def test_paired_bootstrap_ci_clear_positive_signal_resolves():
    """Paired bootstrap ci clear positive signal resolves."""
    ci = paired_bootstrap_ci([1.0, 1.0, 1.0, 1.0])
    assert ci["mean"] == pytest.approx(1.0)
    assert ci["ci_lo"] == pytest.approx(1.0)
    assert ci["excludes_zero"] is True  # two-sided: distinguishable from zero
    assert ci["excludes_zero_positive"] is True  # and the gain is positive


def test_paired_bootstrap_ci_no_signal_does_not_resolve():
    """Paired bootstrap ci no signal does not resolve."""
    ci = paired_bootstrap_ci([0.0, 0.0, 0.0, 0.0])
    assert ci["mean"] == 0.0
    assert ci["excludes_zero"] is False
    assert ci["excludes_zero_positive"] is False


def test_paired_bootstrap_ci_empty_is_safe():
    """Paired bootstrap ci empty is safe."""
    ci = paired_bootstrap_ci([])
    assert ci["n"] == 0
    assert ci["excludes_zero"] is False


def test_paired_bootstrap_ci_is_deterministic():
    """Paired bootstrap ci is deterministic."""
    deltas = [1.0, 0.0, 1.0, -1.0, 1.0, 0.0]
    assert paired_bootstrap_ci(deltas, seed=7) == paired_bootstrap_ci(deltas, seed=7)
