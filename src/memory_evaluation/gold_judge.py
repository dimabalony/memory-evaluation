"""Gold eval judge — pure core (Sub-design C INC 2 Part B, STEP 2).

The end-to-end memory benchmark answers each human-validated gold question twice —
memory-ON (retrieve → context → production model) and memory-OFF (model answers
cold) — then a CROSS-PROVIDER judge (Gemini; NEVER Opus/Sonnet, to avoid
self-preference bias since the answerer IS Opus/Sonnet) scores them. This module
is the pure core: judge-prompt construction, verdict parsing, position-swap
resolution, and metric aggregation. The LLM/DB/S3 orchestration lives in
``scripts/run_gold_eval.py`` and runs in prod.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

# Cross-provider judge model. MUST NOT be an Anthropic model — the answers being
# judged are produced by the production Opus/Sonnet chat model, so judging with
# the same family would bias toward it.
JUDGE_MODEL = "gemini-3-flash-preview"


# ── pointwise judging (score one answer against gold) ─────────────────────────

_POINTWISE_FACT = """\
You are a strict, impartial grader for a personal-memory benchmark. Compare the
MODEL ANSWER to the GOLD ANSWER for the question and grade it.

QUESTION: {question}
GOLD ANSWER: {gold}
MODEL ANSWER: {answer}

Grade three booleans:
- "correct": true if the MODEL ANSWER conveys the same key fact as the GOLD
  ANSWER (semantic match — ignore wording and harmless extra detail). false if it
  contradicts the gold, misses the key fact, or declines to answer.
- "faithful": true if the MODEL ANSWER does not assert fabricated or unsupported
  specifics beyond answering the question. false if it hallucinates details.
- "abstained": true if the MODEL ANSWER declines, says it does not know, or says
  it has no such information instead of giving an answer.

Return ONLY a JSON object: {{"correct": <bool>, "faithful": <bool>, "abstained": <bool>}}
"""

_POINTWISE_ABSTENTION = """\
You are a strict, impartial grader for a personal-memory benchmark. This is an
ABSTENTION item: the answer is NOT present in the assistant's memory, so the
CORRECT behavior is for the model to DECLINE / abstain rather than guess.

QUESTION: {question}
MODEL ANSWER: {answer}

Grade three booleans:
- "abstained": true if the MODEL ANSWER declines, says it does not know, or says
  it has no such information.
- "correct": true ONLY if the model abstained (declining is the right answer
  here); false if it fabricated or guessed a specific answer.
- "faithful": true if the model did not assert fabricated specifics.

Return ONLY a JSON object: {{"correct": <bool>, "faithful": <bool>, "abstained": <bool>}}
"""


def build_pointwise_prompt(question: str, gold_answer: str, model_answer: str, *, is_abstention: bool) -> str:
    """Judge prompt scoring a single answer's correctness/faithfulness/abstention.

    Abstention items are graded inverted: correct == the model declined.
    """
    if is_abstention:
        return _POINTWISE_ABSTENTION.format(question=question, answer=model_answer)
    return _POINTWISE_FACT.format(question=question, gold=gold_answer, answer=model_answer)


def _as_bool(value: Any) -> bool:
    """Coerce a verdict field to bool, handling string-encoded JSON booleans —
    ``bool("false")`` is True in Python, so a judge that emits the string
    ``"false"`` must not be read as a pass."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return bool(value)


def parse_pointwise_verdict(raw: str) -> dict[str, bool] | None:
    """Parse a pointwise verdict into ``{correct, faithful, abstained}`` bools, or
    None if malformed / missing any field (never silently default a score)."""
    obj = _parse_json_object(raw)
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in ("correct", "faithful", "abstained")):
        return None
    return {k: _as_bool(obj[k]) for k in ("correct", "faithful", "abstained")}


# ── pairwise judging (ON vs OFF head-to-head) ─────────────────────────────────

_PAIRWISE = """\
You are a strict, impartial judge for a personal-memory benchmark. Two assistants
answered the same QUESTION. Decide which answer better matches the GOLD ANSWER in
correctness and faithfulness.

QUESTION: {question}
GOLD ANSWER: {gold}

ANSWER A: {answer_a}
ANSWER B: {answer_b}

Pick the answer that more correctly and faithfully matches the GOLD ANSWER. If
they are equivalent in quality, choose "tie".

Return ONLY a JSON object: {{"winner": "A" | "B" | "tie"}}
"""


def build_pairwise_prompt(question: str, gold_answer: str, answer_a: str, answer_b: str) -> str:
    """Judge prompt picking the better of two answers (A / B / tie)."""
    return _PAIRWISE.format(question=question, gold=gold_answer, answer_a=answer_a, answer_b=answer_b)


def parse_pairwise_verdict(raw: str) -> str | None:
    """Parse a pairwise verdict into ``"A"`` / ``"B"`` / ``"tie"`` (or None)."""
    obj = _parse_json_object(raw)
    if not isinstance(obj, dict):
        return None
    winner = str(obj.get("winner", "")).strip().lower()
    return {"a": "A", "b": "B", "tie": "tie"}.get(winner)


def swap_winner(verdict_on_as_a: str, verdict_on_as_b: str) -> str:
    """Resolve a position-swapped pair of pairwise verdicts into ``"on"`` /
    ``"off"`` / ``"tie"``.

    ``verdict_on_as_a`` is the judge's pick when memory-ON was answer A;
    ``verdict_on_as_b`` is the pick when ON was answer B (the swapped run).
    Disagreement between the two orderings (pure position bias) resolves to a tie.
    """
    vote_a = {"A": "on", "B": "off", "tie": "tie"}.get(verdict_on_as_a, "tie")
    vote_b = {"A": "off", "B": "on", "tie": "tie"}.get(verdict_on_as_b, "tie")
    votes = [vote_a, vote_b]
    on, off = votes.count("on"), votes.count("off")
    if on > off:
        return "on"
    if off > on:
        return "off"
    return "tie"


# ── metric aggregation ────────────────────────────────────────────────────────


def _rate(rows: list[dict[str, Any]], side: str, key: str) -> float | None:
    """Fraction of ``rows`` whose ``side`` verdict ``key`` is true (None when empty)."""
    if not rows:
        return None
    return sum(1.0 if r[side][key] else 0.0 for r in rows) / len(rows)


def paired_bootstrap_ci(
    deltas: list[float], n_resamples: int = 2000, seed: int = 0, ci: float = 0.95
) -> dict[str, Any]:
    """Paired-bootstrap CI of the mean per-question delta (deterministic, seeded).

    Mirrors the Part A harness so the headline ON-OFF delta ships with an interval,
    not a bare point estimate (the n-too-small guard). ``excludes_zero`` is the
    two-sided RESOLVES signal — the CI is distinguishable from zero in either
    direction; ``excludes_zero_positive`` additionally requires a positive gain.
    """
    n = len(deltas)
    if n == 0:
        return {
            "mean": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "excludes_zero": False,
            "excludes_zero_positive": False,
            "n": 0,
            "n_resamples": 0,
        }
    mean = sum(deltas) / n
    if n_resamples <= 0:
        return {
            "mean": mean,
            "ci_lo": mean,
            "ci_hi": mean,
            "excludes_zero": mean != 0.0,
            "excludes_zero_positive": mean > 0.0,
            "n": n,
            "n_resamples": 0,
        }
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(total / n)
    means.sort()
    lo_idx = max(0, min(int((1 - ci) / 2 * n_resamples), n_resamples - 1))
    hi_idx = max(0, min(int((1 + ci) / 2 * n_resamples) - 1, n_resamples - 1))
    ci_lo, ci_hi = means[lo_idx], means[hi_idx]
    return {
        "mean": mean,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "excludes_zero": ci_lo > 0.0 or ci_hi < 0.0,
        "excludes_zero_positive": ci_lo > 0.0,
        "n": n,
        "n_resamples": n_resamples,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question results into headline metrics.

    Each row: ``{category, should_abstain, on: {correct,faithful,abstained},
    off: {...}, pairwise: "on"|"off"|"tie"}``. The headline is the memory-ON minus
    memory-OFF correctness delta; abstention accuracy and fact correctness are
    sliced by item type, and the pairwise win-rate counts ties as half.
    """
    n = len(rows)
    abst = [r for r in rows if r["should_abstain"]]
    facts = [r for r in rows if not r["should_abstain"]]
    on_corr = _rate(rows, "on", "correct") or 0.0
    off_corr = _rate(rows, "off", "correct") or 0.0
    # paired per-question deltas (ON minus OFF correctness) → bootstrap CI on the headline
    delta_pairs = [(1.0 if r["on"]["correct"] else 0.0) - (1.0 if r["off"]["correct"] else 0.0) for r in rows]
    delta_ci = paired_bootstrap_ci(delta_pairs)
    pw = [r["pairwise"] for r in rows]
    on_wins, off_wins, ties = pw.count("on"), pw.count("off"), pw.count("tie")
    return {
        "n": n,
        "correctness": {
            "on": on_corr,
            "off": off_corr,
            "delta": on_corr - off_corr,
            "delta_ci": [delta_ci["ci_lo"], delta_ci["ci_hi"]],
            "delta_resolves": delta_ci["excludes_zero"],
            "delta_excludes_zero_positive": delta_ci["excludes_zero_positive"],
        },
        "faithfulness": {"on": _rate(rows, "on", "faithful") or 0.0, "off": _rate(rows, "off", "faithful") or 0.0},
        "abstention_accuracy": {
            "on": _rate(abst, "on", "correct"),
            "off": _rate(abst, "off", "correct"),
            "n": len(abst),
        },
        "fact_correctness": {
            "on": _rate(facts, "on", "correct"),
            "off": _rate(facts, "off", "correct"),
            "n": len(facts),
        },
        "pairwise": {
            "on_wins": on_wins,
            "off_wins": off_wins,
            "ties": ties,
            "on_winrate": (on_wins + 0.5 * ties) / n if n else 0.0,
        },
    }


# ── JSON object extraction (tolerant of fences / surrounding prose) ───────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    """Parse the first JSON object from an LLM response, tolerating markdown
    fences and surrounding prose (None if none found)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
