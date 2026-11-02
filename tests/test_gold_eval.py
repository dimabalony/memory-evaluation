"""Gold Q→A eval — pure core (Sub-design C INC 2 Part B, B1/B2).

The end-to-end memory-quality benchmark. Gold is mined from REAL transcripts
with provenance, ships ``provisional``, and only counts once a HUMAN flips it to
``approved`` (never invented). These pure helpers — schema/validation, transcript
text extraction, and mined-candidate parsing/grounding — carry no DB/LLM/S3.
"""

from __future__ import annotations

import json

from memory_evaluation.gold_eval import (
    GOLD_STATUSES,
    approved_rows,
    count_by_status,
    format_window,
    message_text,
    parse_mined_candidates,
    validate_gold_row,
)


def _row(**kw):
    """Test helper: row."""
    base = {
        "id": "gold-0001",
        "question": "What is Misha's role at Examplecorp?",
        "answer": "Founder",
        "category": "fact",
        "provenance": ["m1"],
        "should_abstain": False,
        "status": "provisional",
    }
    base.update(kw)
    return base


# ── validation ───────────────────────────────────────────────────────────────


def test_valid_fact_row_has_no_errors():
    """Valid fact row has no errors."""
    assert validate_gold_row(_row()) == []


def test_fact_row_requires_nonempty_provenance_and_answer():
    """Fact row requires nonempty provenance and answer."""
    assert "provenance" in " ".join(validate_gold_row(_row(provenance=[])))
    assert "answer" in " ".join(validate_gold_row(_row(answer="")))


def test_abstention_row_allows_empty_provenance():
    """Abstention row allows empty provenance."""
    row = _row(category="abstention", should_abstain=True, answer="not in memory — should decline", provenance=[])
    assert validate_gold_row(row) == []


def test_bad_status_and_category_flagged():
    """Bad status and category flagged."""
    assert any("status" in e for e in validate_gold_row(_row(status="approvedd")))
    assert any("category" in e for e in validate_gold_row(_row(category="banana")))


def test_status_set():
    """Status set."""
    assert GOLD_STATUSES == {"provisional", "approved", "rejected"}


def test_count_and_approved():
    """Count and approved."""
    rows = [_row(status="approved"), _row(status="provisional"), _row(status="approved"), _row(status="rejected")]
    assert count_by_status(rows) == {"approved": 2, "provisional": 1, "rejected": 1}
    assert len(approved_rows(rows)) == 2


# ── transcript extraction ────────────────────────────────────────────────────


def _msg(mid, role, *texts, sender=None):
    """Test helper: msg."""
    return {
        "id": mid,
        "role": role,
        "sender_name": sender,
        "content_blocks": [{"text": t} for t in texts],
    }


def test_message_text_joins_text_blocks():
    """Message text joins text blocks."""
    msg = {"content_blocks": [{"text": "hello"}, {"text": "Photo:", "media_ref": {"x": 1}}, {"text": "world"}]}
    assert message_text(msg) == "hello\nPhoto:\nworld"


def test_message_text_handles_empty_blocks():
    """Message text handles empty blocks."""
    assert message_text({"content_blocks": []}) == ""


def test_format_window_labels_role_and_short_id():
    """Format window labels role and short id."""
    msgs = [_msg("abcdef12-0000", "user", "hi", sender="Misha"), _msg("99887766-1111", "assistant", "hello")]
    out = format_window(msgs)
    assert "abcdef12" in out and "99887766" in out
    assert "user" in out and "assistant" in out
    assert "Misha" in out


# ── mined-candidate parsing + grounding ──────────────────────────────────────


def test_parse_keeps_only_grounded_candidates():
    """Parse keeps only grounded candidates."""
    window_ids = {"m1", "m2", "m3"}
    raw = [
        {"question": "Q1", "answer": "A1", "category": "fact", "provenance": ["m1"], "evidence": "e1"},
        {"question": "Q2", "answer": "A2", "category": "fact", "provenance": ["m9"], "evidence": "e2"},  # ungrounded
    ]
    out = parse_mined_candidates(raw, window_ids, mined_at="2026-06-07T00:00:00Z")
    assert len(out) == 1
    assert out[0]["question"] == "Q1"
    assert out[0]["status"] == "provisional"
    assert out[0]["mined_at"] == "2026-06-07T00:00:00Z"


def test_parse_allows_abstention_without_provenance():
    """Parse allows abstention without provenance."""
    raw = [{"question": "Qx", "answer": "decline", "category": "abstention", "should_abstain": True, "provenance": []}]
    out = parse_mined_candidates(raw, {"m1"}, mined_at="2026-06-07T00:00:00Z")
    assert len(out) == 1 and out[0]["category"] == "abstention"


def test_parse_drops_malformed_rows():
    """Parse drops malformed rows."""
    raw = [{"question": "", "answer": "A", "category": "fact", "provenance": ["m1"]}, "not-a-dict"]
    assert parse_mined_candidates(raw, {"m1"}, mined_at="2026-06-07T00:00:00Z") == []


def test_parse_abstention_forces_empty_answer_and_provenance():
    # a should_abstain candidate must NEVER carry a model-supplied answer/citation
    """Parse abstention forces empty answer and provenance."""
    raw = [
        {
            "question": "What is X's password?",
            "answer": "hunter2",
            "provenance": ["m1"],
            "should_abstain": True,
            "category": "abstention",
        }
    ]
    out = parse_mined_candidates(raw, {"m1"}, mined_at="2026-06-07T00:00:00Z")
    assert len(out) == 1
    assert out[0]["should_abstain"] is True
    assert out[0]["answer"] == ""
    assert out[0]["provenance"] == []


# ── freeze-at-SHA (B3): canonical content hash + PII-free pointer ──────────────


def _frow(rid, ans, **extra):
    """Test helper: frow."""
    return {
        "id": rid,
        "question": "q",
        "answer": ans,
        "category": "fact",
        "should_abstain": False,
        "provenance": ["m1"],
        **extra,
    }


def test_gold_content_sha256_is_canonical_and_content_sensitive():
    """Gold content sha256 is canonical and content sensitive."""
    from memory_evaluation.gold_eval import gold_content_sha256

    a = [_frow("g2", "a2"), _frow("g1", "a1")]
    assert gold_content_sha256(a) == gold_content_sha256(list(reversed(a)))  # order-independent (sorted by id)
    assert len(gold_content_sha256(a)) == 64
    changed = [_frow("g1", "DIFFERENT"), _frow("g2", "a2")]
    assert gold_content_sha256(changed) != gold_content_sha256(a)  # answer change → different hash
    # metadata-only differences (mined_at, evidence, status) do NOT change the freeze identity
    meta = [_frow("g2", "a2", mined_at="X", evidence="Y", status="approved"), _frow("g1", "a1", mined_at="Z")]
    assert gold_content_sha256(meta) == gold_content_sha256(a)


def test_build_frozen_pointer_is_pii_free():
    """Build frozen pointer is pii free."""
    from memory_evaluation.gold_eval import build_frozen_pointer

    rows = [_frow("gold-0002", "Riverstone"), _frow("gold-0001", "secret answer")]
    p = build_frozen_pointer(rows, code_sha="abc1234", frozen_at="2026-06-13T00:00:00Z")
    assert p["code_sha"] == "abc1234"
    assert p["n"] == 2
    assert p["s3_key"] == "benchmark/gold_qa/frozen/abc1234.jsonl"
    assert p["question_ids"] == ["gold-0001", "gold-0002"]  # sorted ids only
    assert len(p["content_sha256"]) == 64
    # PII-free: only ids + hashes — no answer/question text leaks into the git pointer
    assert set(p) == {"code_sha", "frozen_at", "n", "s3_key", "content_sha256", "question_ids"}
    assert "secret answer" not in json.dumps(p) and "Riverstone" not in json.dumps(p)
