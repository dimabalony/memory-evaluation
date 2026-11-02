"""Gold Q→A eval — pure core (Sub-design C INC 2 Part B).

Schema/validation + transcript-text extraction + mined-candidate parsing for the
end-to-end memory-quality benchmark. Gold is mined from REAL transcripts with
provenance (source message ids), ships ``provisional``, and only counts once a
HUMAN flips it to ``approved`` — answers are never invented. Pure: no DB/LLM/S3.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

GOLD_STATUSES = frozenset({"provisional", "approved", "rejected"})
GOLD_CATEGORIES = frozenset({"fact", "knowledge_update", "contradiction", "abstention"})

# Fields that define the eval-relevant gold CONTENT; mutable metadata (mined_at,
# evidence, status, source_*) is excluded so a freeze identity is stable across
# re-curation that doesn't change a question or its answer.
_FROZEN_FIELDS = ("id", "question", "answer", "category", "should_abstain", "provenance")


# ── gold row schema/validation ───────────────────────────────────────────────


def validate_gold_row(row: Any) -> list[str]:
    """Return a list of human-readable problems with a gold row (empty = valid).

    Non-abstention rows MUST carry a non-empty answer AND non-empty provenance
    (the grounding contract). Abstention rows (``should_abstain`` / category
    ``abstention``) intentionally have no provenance — the point is the answer is
    NOT in memory and Kai should decline.
    """
    if not isinstance(row, dict):
        return ["row is not a dict"]
    errors: list[str] = []
    for field in ("id", "question", "category", "status"):
        if not row.get(field):
            errors.append(f"missing {field}")
    status = row.get("status")
    if status and status not in GOLD_STATUSES:
        errors.append(f"invalid status {status!r}")
    category = row.get("category")
    if category and category not in GOLD_CATEGORIES:
        errors.append(f"invalid category {category!r}")
    is_abstention = bool(row.get("should_abstain")) or category == "abstention"
    if not is_abstention:
        if not row.get("answer"):
            errors.append("non-abstention row needs a non-empty answer")
        if not row.get("provenance"):
            errors.append("non-abstention row needs non-empty provenance (source message ids)")
    return errors


def count_by_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count gold rows grouped by their ``status`` field."""
    counts: dict[str, int] = {}
    for row in rows:
        key = row.get("status", "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def approved_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only ``approved`` rows count toward the benchmark (human-validated gold)."""
    return [row for row in rows if row.get("status") == "approved"]


# ── transcript extraction ────────────────────────────────────────────────────


def message_text(msg: dict[str, Any]) -> str:
    """Concatenate the text of a message's content blocks (newline-joined)."""
    blocks = msg.get("content_blocks") or []
    return "\n".join(b["text"] for b in blocks if isinstance(b, dict) and b.get("text"))


def format_window(messages: list[dict[str, Any]]) -> str:
    """Render a transcript window for the mining prompt, labelling each line with
    a short message id (for provenance) + role/sender."""
    lines: list[str] = []
    for msg in messages:
        short_id = str(msg.get("id", ""))[:8]
        role = msg.get("role", "")
        sender = msg.get("sender_name")
        who = f"{role}/{sender}" if sender else f"{role}"
        lines.append(f"[{short_id}] {who}: {message_text(msg)}")
    return "\n".join(lines)


# ── mined-candidate parsing + grounding ──────────────────────────────────────


def parse_mined_candidates(
    raw: list[Any],
    window_ids: set[str],
    *,
    mined_at: str,
) -> list[dict[str, Any]]:
    """Normalise raw LLM-proposed candidates into provisional gold rows, keeping
    only the ones that are GROUNDED: every non-abstention candidate's provenance
    must be non-empty AND a subset of the window's message ids, with a non-empty
    answer. Abstention candidates are kept without provenance. Malformed rows are
    dropped (never invented into existence).
    """
    window = {str(x) for x in window_ids}
    out: list[dict[str, Any]] = []
    for candidate in raw:
        if not isinstance(candidate, dict):
            continue
        question = (candidate.get("question") or "").strip()
        if not question:
            continue
        category = candidate.get("category") or "fact"
        provenance = [str(p) for p in (candidate.get("provenance") or [])]
        is_abstention = bool(candidate.get("should_abstain")) or category == "abstention"
        if not is_abstention:
            if not provenance or not all(p in window for p in provenance):
                continue
            if not (candidate.get("answer") or "").strip():
                continue
        out.append(
            {
                "id": candidate.get("id") or "",
                "question": question,
                # Abstention rows are "answer not in memory → decline": force empty
                # answer/provenance even if the model supplied them, so a should_abstain
                # row can never carry an answer or citation.
                "answer": "" if is_abstention else (candidate.get("answer") or ""),
                "category": category,
                "provenance": [] if is_abstention else provenance,
                "should_abstain": is_abstention,
                "evidence": candidate.get("evidence") or "",
                "status": "provisional",
                "mined_at": mined_at,
            }
        )
    return out


# ── freeze-at-SHA (B3): pin the gold so eval re-runs are comparable ───────────


def canonical_gold_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Deterministic byte representation of the eval-relevant gold content (sorted
    by id, canonical JSON over ``_FROZEN_FIELDS`` only). Stable across metadata
    churn — the basis for the freeze content hash."""
    canon = sorted(({k: r.get(k) for k in _FROZEN_FIELDS} for r in rows), key=lambda r: r.get("id") or "")
    return json.dumps(canon, sort_keys=True, ensure_ascii=False).encode("utf-8")


def gold_content_sha256(rows: list[dict[str, Any]]) -> str:
    """SHA-256 of the canonical gold content — the freeze identity / integrity check."""
    return hashlib.sha256(canonical_gold_bytes(rows)).hexdigest()


def build_frozen_pointer(
    rows: list[dict[str, Any]], *, code_sha: str, frozen_at: str, base_prefix: str = ""
) -> dict[str, Any]:
    """PII-free pointer to a frozen gold set (safe to commit to git): the commit
    SHA, the S3 key holding the content, the row count, a content hash for
    verification, and the question ids — but NO question/answer text."""
    from memory_evaluation.benchmark_dataset import frozen_gold_key

    return {
        "code_sha": code_sha,
        "frozen_at": frozen_at,
        "n": len(rows),
        "s3_key": frozen_gold_key(code_sha, base_prefix),
        "content_sha256": gold_content_sha256(rows),
        "question_ids": sorted(r["id"] for r in rows),
    }
