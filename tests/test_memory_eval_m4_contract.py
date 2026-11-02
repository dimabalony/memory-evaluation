"""Offline adversarial paired-answer contracts (DB-840)."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from memory_evaluation.eval import answering
from memory_evaluation.eval.fixtures import FixtureSet


def fixtures():
    return FixtureSet(
        path="p",
        sha256="s",
        fixtures=[
            {
                "id": "f",
                "_provenance": {"source": "synthetic", "content_sha256": "x"},
                "participants": [{"key": "p", "name": "Pat", "telegram_user_id": 7}],
                "chat": {"chat_id": 1},
                "messages": [{"from": "p", "text": "original"}],
                "expected": {"questions": [{"id": "q", "question": "Where?", "expect_any": ["Leeds"]}]},
            }
        ],
    )


def report():
    return {
        "fixture_set": {"path": "p", "sha256": "s", "n": 1},
        "extractor": {"name": "recorded"},
        "answerer": {"quality": "wiring-only"},
        "judge": None,
        "arms": ["memory", "no_memory"],
        "embedder": "fake",
        "mutation_check": {},
        "results": [
            {
                "id": "f",
                "status": "ok",
                "answers": [
                    {
                        "question_id": "q",
                        "arm": arm,
                        "answer": "Leeds",
                        "error": None,
                        "judge_raw": None,
                        "judge_error": None,
                        "input_sha256": "a" * 64,
                        "history_sha256": "b" * 64,
                    }
                    for arm in answering.ARMS
                ],
            }
        ],
    }


def _assert_valid_pair(raw):
    board = answering.build_answer_scoreboard(raw, fixtures())
    assert board["aggregate"]["memory_benefit"] == {
        "paired_questions": 1,
        "memory_correct": 1,
        "no_memory_correct": 1,
    }


def test_failed_arm_does_not_enter_paired_denominator():
    raw = report()
    _assert_valid_pair(raw)
    raw["results"][0]["answers"][1]["error"] = "synthetic failure"
    board = answering.build_answer_scoreboard(raw, fixtures())
    assert board["aggregate"]["memory_benefit"]["paired_questions"] == 0
    assert board["aggregate"]["per_arm"]["no_memory"]["accuracy"] is None
    assert board["aggregate"]["per_arm"]["memory"]["accuracy"] == 1


def test_missing_fixture_preserves_invalid_denominator():
    raw = report()
    _assert_valid_pair(raw)
    raw["results"] = []
    board = answering.build_answer_scoreboard(raw, fixtures())
    assert len(board["rows"]) == 2
    assert all(row["status"] == "invalid" for row in board["rows"])
    assert board["aggregate"]["memory_benefit"]["paired_questions"] == 0


@pytest.mark.parametrize("arms", [[], ["memory", "memory"], ["unknown"]])
def test_malformed_arm_manifest_cannot_silently_score(arms):
    raw = report()
    raw["arms"] = arms
    with pytest.raises(ValueError):
        answering.build_answer_scoreboard(raw, fixtures())


@pytest.mark.parametrize("duplicate", ["fixture", "answer"])
def test_duplicate_report_records_cannot_silently_score(duplicate):
    raw = report()
    rows = raw["results"] if duplicate == "fixture" else raw["results"][0]["answers"]
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        answering.build_answer_scoreboard(raw, fixtures())


@pytest.mark.parametrize("field", ["input_sha256", "history_sha256"])
def test_changed_pair_input_cannot_count_as_memory_benefit(field):
    raw = report()
    _assert_valid_pair(raw)
    raw["results"][0]["answers"][1][field] = "c" * 64
    board = answering.build_answer_scoreboard(raw, fixtures())
    assert board["aggregate"]["memory_benefit"]["paired_questions"] == 0


@pytest.mark.asyncio
async def test_nested_model_input_mutation_cannot_change_other_arm_or_labels(monkeypatch):
    corpus = fixtures()
    original = deepcopy(corpus.fixtures)
    seen = []

    class Priming:
        def __init__(self, *args, **kwargs):
            pass

        async def prime(self, *args, **kwargs):
            return "memory", ["trace"]

    async def replay(fixture, db, extractor, then):
        result = {"id": fixture["id"], "status": "ok"}
        await then(object(), fixture, result, None)
        return result

    async def answer(request):
        seen.append(deepcopy(request.history))
        request.history[0]["text"] = "corrupted"
        return "Leeds"

    async def judge(request):
        request.expected["expect_any"].append("corrupted")
        return "VERDICT: correct"

    monkeypatch.setattr(answering, "MemoryPrimingService", Priming)
    monkeypatch.setattr(answering, "_run_fixture", replay)
    monkeypatch.setattr(answering, "_chat_context", lambda *args: object())
    raw = await answering.run_read_to_answer(
        corpus, SimpleNamespace(mutation_check=lambda: {}), answer_fn=answer, judge_fn=judge
    )
    assert seen[0] == seen[1]
    assert corpus.fixtures == original
    rows = raw["results"][0]["answers"]
    assert rows[0]["input_sha256"] == rows[1]["input_sha256"]


@pytest.mark.asyncio
async def test_failed_replay_releases_session_restores_admins_and_next_fixture(monkeypatch):
    from contextlib import asynccontextmanager

    from memory_evaluation.eval import runner

    settings = SimpleNamespace(telegram_admin_user_ids="42")
    exits = []

    class DB:
        @asynccontextmanager
        async def session(self):
            try:
                yield object()
            finally:
                exits.append(True)

    async def replay(session, fixture, extractor, result, then):
        assert settings.telegram_admin_user_ids == "7"
        if fixture["id"] == "bad":
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner, "_replay", replay)
    extractor = SimpleNamespace(calls=[])
    base = {"participants": [{"telegram_user_id": 7, "admin": True}]}
    # Dynamic ports resolve through __getattr__; monkeypatch would restore the
    # resolved fallback as a real module attribute and bypass later scopes.
    with runner.ports.scoped({"get_settings": lambda: settings}):
        bad = await runner._run_fixture({**base, "id": "bad"}, DB(), extractor)
        good = await runner._run_fixture({**base, "id": "good"}, DB(), extractor)
    assert bad["status"] == "error" and good["status"] == "ok"
    assert exits == [True, True]
    assert settings.telegram_admin_user_ids == "42"
