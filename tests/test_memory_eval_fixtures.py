"""Lifecycle fixture format, provenance refusal and the legacy-set ratchet (DB-840)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from memory_test_parsing import parse_json_object_response
from memory_evaluation.eval import load_fixture_set
from memory_evaluation.eval.fixtures import (
    AXES,
    FORMAT,
    TRACE_REF,
    FixtureError,
    content_sha256,
    sets_missing_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/memory_lifecycle/v1"

# The ratchet: fixture sets that predate mandatory provenance (design 2026-10-09
# §6 rule 4). This list may only shrink. A set that gains `_provenance` must be
# removed from it; a new set must declare `_provenance` from the start.
LEGACY_SETS_WITHOUT_PROVENANCE = [
    "c_eprobe",
    "capture",
    "elephant",
    "gold_eval",
    "grace",
    "priming_benchmark",
    "procedural",
    "store_extraction",
]




def test_committed_lifecycle_set_meets_the_db840_shape():
    fixture_set = load_fixture_set(FIXTURES)
    fixtures = fixture_set.fixtures
    assert len(fixtures) >= 12
    sizes = [len(f["messages"]) for f in fixtures]
    assert all(24 <= n <= 36 for n in sizes), sizes
    assert sum(n >= 30 for n in sizes) >= 6, sizes
    assert set().union(*(f["axes"] for f in fixtures)) == AXES
    assert {f["_provenance"]["source"] for f in fixtures} == {"synthetic"}
    assert len({f["id"] for f in fixtures}) == len(fixtures)
    assert {f["format"] for f in fixtures} == {FORMAT}


def test_every_recorded_response_parses_through_the_real_parser():
    """Recorded responses are rendered and parsed exactly as the extraction path does;
    the integration smoke then proves the validators and store accept every item."""
    for fixture in load_fixture_set(FIXTURES).fixtures:
        for cp in fixture["checkpoints"]:
            rendered = TRACE_REF.sub(lambda _m: str(uuid.uuid4()), json.dumps(cp["response"], ensure_ascii=False))
            parsed = parse_json_object_response(rendered)
            assert len(parsed["memories"]) == len(cp["response"]["memories"]), fixture["id"]
            for item in parsed["memories"]:
                assert item["content"] and item["memory_type"] and isinstance(item["entities"], list)


def _minimal(**provenance) -> dict:
    fixture = {
        "format": FORMAT,
        "_provenance": {
            "source": "synthetic",
            "author": "test",
            "approved_by": None,
            "version": 1,
            "content_sha256": "",
            "pii": "none",
            **provenance,
        },
        "id": "tiny",
        "axes": ["durable_vs_transient"],
        "chat": {"chat_id": 1, "topic_id": None, "is_group": False},
        "start_at": "2026-01-01T00:00:00+00:00",
        "participants": [{"key": "p", "telegram_user_id": 1, "name": "Pat Example"}],
        "messages": [{"from": "p", "minute": 0, "text": "I work at Example Co"}],
        "checkpoints": [{"after": 1, "response": {"memories": [], "social_experiences": []}}],
        "expected": {"writes": [{"label": "w", "content_contains": ["Example Co"]}]},
    }
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    return fixture


def _write(tmp_path: Path, fixture: dict) -> Path:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(fixture))
    return tmp_path


def test_loader_accepts_synthetic_without_approval(tmp_path):
    assert load_fixture_set(_write(tmp_path, _minimal())).fixtures[0]["id"] == "tiny"


def test_loader_refuses_a_fixture_without_provenance(tmp_path):
    fixture = _minimal()
    del fixture["_provenance"]
    with pytest.raises(FixtureError, match="missing _provenance"):
        load_fixture_set(_write(tmp_path, fixture))


@pytest.mark.parametrize("source", ["production-derived", "sanitized"])
def test_loader_refuses_non_synthetic_without_approval(tmp_path, source):
    with pytest.raises(FixtureError, match=f"source={source} requires approved_by"):
        load_fixture_set(_write(tmp_path, _minimal(source=source)))


def test_loader_accepts_production_derived_once_approved(tmp_path):
    fixture = _minimal(source="production-derived", approved_by="reviewer")
    assert load_fixture_set(_write(tmp_path, fixture)).fixtures


def test_loader_refuses_an_edit_without_a_re_stamp(tmp_path):
    fixture = _minimal()
    fixture["messages"][0]["text"] = "I work at Other Co"
    with pytest.raises(FixtureError, match="content_sha256 does not match"):
        load_fixture_set(_write(tmp_path, fixture))


def test_loader_refuses_an_unknown_trace_ref(tmp_path):
    fixture = _minimal()
    fixture["checkpoints"][0]["response"]["memories"] = [{"content": "x", "contradicts_memory_id": "{{trace:nope}}"}]
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match="unknown trace label 'nope'"):
        load_fixture_set(_write(tmp_path, fixture))


def test_loader_refuses_a_checkpoint_past_the_last_message(tmp_path):
    fixture = _minimal()
    fixture["checkpoints"][0]["after"] = 2
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match="checkpoints"):
        load_fixture_set(_write(tmp_path, fixture))


def test_loader_refuses_an_unknown_sender(tmp_path):
    fixture = _minimal()
    fixture["messages"][0]["from"] = "stranger"
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match="not a participant"):
        load_fixture_set(_write(tmp_path, fixture))


def test_subset_by_id_and_set_hash(tmp_path):
    full = load_fixture_set(FIXTURES)
    one = load_fixture_set(FIXTURES, ids=["lc-02-private-family-health"])
    assert [f["id"] for f in one.fixtures] == ["lc-02-private-family-health"]
    assert one.sha256 != full.sha256
    with pytest.raises(FixtureError, match="unknown fixture ids"):
        load_fixture_set(FIXTURES, ids=["nope"])


@pytest.mark.parametrize("start_at", [None, "2026-01-01T00:00:00", "not a time"])
def test_loader_refuses_a_missing_or_naive_start_at(tmp_path, start_at):
    fixture = _minimal()
    if start_at is None:
        del fixture["start_at"]
    else:
        fixture["start_at"] = start_at
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match="start_at must be an ISO-8601 timestamp with a UTC offset"):
        load_fixture_set(_write(tmp_path, fixture))


def test_loader_refuses_a_readback_cutoff_below_the_production_limit(tmp_path):
    fixture = _minimal()
    fixture["expected"]["readback"] = [{"id": "rb", "cue": "where does Pat work", "k": 3, "expect_labels": ["w"]}]
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match="below the production priming limit"):
        load_fixture_set(_write(tmp_path, fixture))


@pytest.mark.parametrize("ids", [None, ["tiny"], ["other"], []])
def test_duplicate_fixture_ids_are_rejected_before_filtering(tmp_path, ids):
    duplicate = _minimal()
    for filename in ("first.json", "second.json"):
        (tmp_path / filename).write_text(json.dumps(duplicate))
    other = _minimal()
    other["id"] = "other"
    other["_provenance"]["content_sha256"] = content_sha256(other)
    (tmp_path / "third.json").write_text(json.dumps(other))
    with pytest.raises(FixtureError, match=r"duplicate fixture id 'tiny'.*first\.json.*second\.json"):
        load_fixture_set(tmp_path, ids=ids)


def test_distinct_fixture_ids_preserve_file_order_and_filtering(tmp_path):
    for filename, fixture_id in (("b.json", "second"), ("a.json", "first")):
        fixture = _minimal()
        fixture["id"] = fixture_id
        fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
        (tmp_path / filename).write_text(json.dumps(fixture))
    assert [f["id"] for f in load_fixture_set(tmp_path).fixtures] == ["first", "second"]
    assert [f["id"] for f in load_fixture_set(tmp_path, ids=["second"]).fixtures] == ["second"]


def test_corpus_has_complete_availability_and_split_labels():
    from memory_evaluation.eval.fixtures import AVAILABILITIES

    fixtures = load_fixture_set(FIXTURES).fixtures
    assert {f["split"] for f in fixtures} == {"tuning", "held_out"}
    for split in ("tuning", "held_out"):
        selected = [f for f in fixtures if f["split"] == split]
        assert len(selected) >= 4
        assert {q["availability"] for f in selected for q in f["expected"]["questions"]} == AVAILABILITIES
    for fixture in fixtures:
        assert fixture["_provenance"]["license"] == "private-unlicensed"
        assert fixture["_provenance"]["approved_by"] is None
        assert fixture["expected"]["writes"] and fixture["expected"]["prohibited"]
        for question in fixture["expected"]["questions"]:
            assert {
                "availability",
                "distance_user_turns",
                "evidence_message_indices",
                "memory_labels",
            } <= question.keys()


def test_original_fixture_and_question_ids_survive():
    expected = {
        "lc-01-flatmates-group": {
            "q_teo_employer",
            "q_sister_vet",
            "q_bram_hometown",
            "q_mila_allergy",
            "q_derby_abstain",
        },
        "lc-02-private-family-health": {
            "q_birthday",
            "q_moving",
            "q_mum_surgery",
            "q_screen_breaks",
            "q_hanne_language",
        },
        "lc-03-studio-team-topic": {"q_launch_date", "q_helsinki", "q_lunch_abstain"},
        "lc-04-cross-script-household": {
            "q_yaroslav_city",
            "q_vera_pharmacy",
            "q_yara_job",
            "q_dentist_later",
            "q_dentist_same_chat",
        },
    }
    fixtures = {f["id"]: f for f in load_fixture_set(FIXTURES).fixtures}
    for fixture_id, question_ids in expected.items():
        assert {q["id"] for q in fixtures[fixture_id]["expected"]["questions"]} == question_ids


@pytest.mark.parametrize(
    "patch,match",
    [
        ({"availability": "invented"}, "unknown availability"),
        ({"distance_user_turns": -1}, "distance_user_turns"),
        ({"distance_user_turns": True}, "distance_user_turns"),
        ({"memory_labels": ["absent"]}, "memory_labels"),
        ({"evidence_message_indices": [99]}, "evidence_message_indices"),
        ({"distance_user_turns": 3}, "disagrees with evidence"),
        (
            {"availability": "history_only", "history": "full", "memory_labels": [], "evidence_message_indices": []},
            "history availability requires message evidence",
        ),
        ({"memory_labels": []}, "memory availability requires memory labels"),
        ({"history": "full"}, "memory_only requires"),
        ({"availability": "nowhere", "abstain": True}, "nowhere requires"),
    ],
)
def test_loader_refuses_invalid_availability_metadata(tmp_path, patch, match):
    fixture = _minimal()
    question = {
        "id": "q",
        "question": "Where does Pat work?",
        "expect_any": ["Example Co"],
        "history": "none",
        "availability": "memory_only",
        "distance_user_turns": 0,
        "evidence_message_indices": [1],
        "memory_labels": ["w"],
    }
    question.update(patch)
    fixture["expected"]["questions"] = [question]
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match=match):
        load_fixture_set(_write(tmp_path, fixture))


def test_history_conflict_is_deliberately_unextracted_and_prohibits_old_answer():
    for fixture in load_fixture_set(FIXTURES).fixtures:
        for question in fixture["expected"]["questions"]:
            if question["availability"] != "history_conflicts_stale_memory":
                continue
            assert min(question["evidence_message_indices"]) > max(cp["after"] for cp in fixture["checkpoints"])
            assert question["prohibited"]
            assert question["memory_labels"]
            assert question["history"] == "full"


def test_history_availability_refuses_evidence_outside_delivered_window(tmp_path):
    fixture = _minimal()
    fixture["messages"] = [
        {"from": "p", "minute": i, "text": "I work at Example Co" if i == 0 else "A different topic"} for i in range(31)
    ]
    fixture["expected"]["questions"] = [
        {
            "id": "q",
            "question": "Where does Pat work?",
            "expect_any": ["Example Co"],
            "history": "full",
            "availability": "both_agree",
            "distance_user_turns": 30,
            "evidence_message_indices": [1],
            "memory_labels": ["w"],
        }
    ]
    fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
    with pytest.raises(FixtureError, match="outside the delivered final 30 messages"):
        load_fixture_set(_write(tmp_path, fixture))


def test_new_corpus_attributes_recorded_facts_and_authorship_honestly():
    from memory_evaluation.eval.matching import matching_fact

    fixtures = load_fixture_set(FIXTURES).fixtures[4:]
    for fixture in fixtures:
        assert fixture["_provenance"]["author"] == "DB-840 synthetic corpus"
        recorded = [memory for cp in fixture["checkpoints"] for memory in cp["response"]["memories"]]
        for write in fixture["expected"]["writes"]:
            if write.get("subject"):
                assert write.get("fact"), (fixture["id"], write["label"])
                matches = [matching_fact(write, memory) for memory in recorded]
                assert any(fact and fact["subject"] == write["subject"] for fact in matches)
        for question in fixture["expected"]["questions"]:
            # Mechanism-specific probes must not claim the entire DB-411 suite.
            assert 0 < len(question["axes"]) <= 4, (fixture["id"], question["id"])
            assert set(question["axes"]) <= set(fixture["axes"])


def test_new_held_out_conversations_have_distinct_structural_shapes():
    fixtures = load_fixture_set(FIXTURES).fixtures[8:]
    shapes = {
        (len(f["messages"]), len(f["participants"]), f["chat"]["is_group"], len(f["checkpoints"])) for f in fixtures
    }
    assert len(shapes) == 4
    assert any(not f["chat"]["is_group"] for f in fixtures)
    assert any(len(f["participants"]) >= 3 for f in fixtures)
    assert any(len(f["checkpoints"]) >= 2 for f in fixtures)
    assert sum(bool(f.get("pre_state")) for f in fixtures) >= 2
    role_conflicts = [
        q
        for f in fixtures
        for q in f["expected"]["questions"]
        if q["availability"] == "history_conflicts_stale_memory" and "pre_role" in q["memory_labels"]
    ]
    assert len(role_conflicts) >= 3


def test_cross_script_annotations_use_complete_grounded_words():
    from memory_evaluation.eval.matching import write_matches

    fixture = load_fixture_set(FIXTURES, ids=["lc-04-cross-script-household"]).fixtures[0]
    write = next(w for w in fixture["expected"]["writes"] if w["label"] == "yara_neighbour")
    assert write["fact"]["value_contains"] == "пекарня"
    recorded = [memory for cp in fixture["checkpoints"] for memory in cp["response"]["memories"]]
    assert any(write_matches(write, memory) for memory in recorded)
    questions = {q["id"]: q for q in fixture["expected"]["questions"]}
    assert questions["q_yaroslav_city"]["expect_any"] == ["Самара", "Самаре", "Самару", "Samara"]
    assert questions["q_yara_job"]["expect_any"] == ["пекарня", "пекарне", "пекарню", "bakery"]
    # Stems are not standalone words under the Unicode-aware lexical scorer.
    for question in questions.values():
        assert not ({"Самар", "пекарн"} & set(question.get("expect_any", []) + question.get("prohibited", [])))


@pytest.mark.parametrize("stored_durability,correct", [("transient", 1), ("durable", 0)])
def test_cycling_stable_preference_obeys_curated_retention_policy(stored_durability, correct):
    from memory_evaluation.eval.scoring import score_fixture

    fixture = load_fixture_set(FIXTURES, ids=["lc-07-cycling-club"]).fixtures[0]
    write = next(w for w in fixture["expected"]["writes"] if w["label"] == "stable_preference")
    recorded = fixture["checkpoints"][0]["response"]["memories"][0]
    assert write["durable"] is False
    assert recorded["facts"][0]["slot"] == "stable_preferences"
    assert recorded["facts"][0]["durability"] == "transient"
    # Long-term preference wording does not grant a curated durable retention floor.
    probe = {**fixture, "expected": {"writes": [write]}}
    owner = fixture["participants"][0]
    result = {
        "status": "ok",
        "labels": {"stable_preference": "t-preference"},
        "snapshot": {
            "traces": [
                {
                    "id": "t-preference",
                    "content": recorded["content"],
                    "pre_state": False,
                    "invalidated": False,
                    "invalidated_by": None,
                    "durability": stored_durability,
                    "facts": [{**recorded["facts"][0], "subject_id": "e-mei", "durability": stored_durability}],
                }
            ],
            "entities": [
                {
                    "id": "e-mei",
                    "name": owner["name"],
                    "type": "person",
                    "aliases": [],
                    "telegram_user_id": owner["telegram_user_id"],
                }
            ],
        },
        "checkpoints": [{"after": 16, "stats": {"messages_read": 16}, "store_errors": []}],
        "messages_saved": len(fixture["messages"]),
        "prompt_checks": [],
        "readback": [],
    }
    row = score_fixture(probe, result)
    assert row["status"] == "ok"
    assert row["written"] == ["stable_preference"]
    assert row["durability"]["checked"] == 1
    assert row["durability"]["correct"] == correct
    assert len(row["durability"]["misses"]) == 1 - correct


def test_new_durable_named_writes_have_durable_recorded_facts():
    from memory_evaluation.eval.matching import matching_fact

    for fixture in load_fixture_set(FIXTURES).fixtures[4:]:
        for write in fixture["expected"]["writes"]:
            if write["label"] != "durable":
                continue
            facts = [
                matching_fact(write, memory)
                for checkpoint in fixture["checkpoints"]
                for memory in checkpoint["response"]["memories"]
            ]
            assert any(fact and fact["durability"] == "durable" for fact in facts), fixture["id"]
