"""Layer A scoring, verdict parsing and the capped live answerer (DB-840). No database, no model."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from memory_evaluation.eval import build_answer_scoreboard, load_fixture_set
from memory_evaluation.eval.answering import (
    AnswerRequest,
    LiveAnswerer,
    check_answer,
    is_abstention,
    parse_verdict,
)
from memory_evaluation.eval.extractors import BudgetExceeded
from memory_evaluation.eval.fixtures import FixtureError, FixtureSet, content_sha256
from datetime import UTC

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/memory_lifecycle/v1"


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("VERDICT: correct", "correct"),
        ("thinking...\nverdict: Incorrect", "incorrect"),
        ("VERDICT: abstained", "abstained"),
        ("VERDICT: correct\nVERDICT: correct", "correct"),
        ("VERDICT: correct\nVERDICT: incorrect", None),
        ("looks right to me", None),
        ("VERDICT: maybe", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_verdict_never_guesses(text, verdict):
    assert parse_verdict(text) == verdict


def test_check_answer_expected_prohibited_and_abstention():
    q = {"expect_any": ["Quillmoor"], "prohibited": ["Larkspur"]}
    assert check_answer(q, "She works at Quillmoor Labs.")["verdict"] == "correct"
    graded = check_answer(q, "Quillmoor Labs, after Larkspur.")
    assert graded["verdict"] == "incorrect" and graded["prohibited_hits"] == ["Larkspur"]
    assert check_answer(q, "I don't know.")["abstained"] is True
    assert check_answer({"expect_all": ["9", "November"]}, "9 November")["verdict"] == "correct"
    assert check_answer({"expect_exact": "Rotterdam"}, "  rotterdam ")["verdict"] == "correct"
    assert check_answer({"expect_exact": "Rotterdam"}, "Rotterdam, I think")["verdict"] == "incorrect"
    abstain = {"abstain": True, "prohibited": ["blood type is"]}
    assert check_answer(abstain, "I don't know that.")["verdict"] == "correct"
    assert check_answer(abstain, "His blood type is O.")["verdict"] == "incorrect"
    assert is_abstention("Не знаю.") and not is_abstention("Rotterdam")  # noqa: RUF001
    # A typographic apostrophe, as live models often write it, still reads as abstaining.
    assert is_abstention("I don\u2019t know.") and is_abstention("I can\u2019t find that.")


def _fixture() -> dict:
    return {
        "id": "f",
        "_provenance": {"source": "synthetic", "content_sha256": "x"},
        "expected": {
            "questions": [
                {"id": "q1", "question": "Where?", "expect_any": ["Leeds"], "axes": ["durable_vs_transient"]},
                {"id": "q2", "question": "Blood type?", "abstain": True, "axes": ["abstention_control"]},
            ]
        },
    }


def _row(qid: str, arm: str, answer: str | None, **kw) -> dict:
    return {
        "question_id": qid,
        "arm": arm,
        "answer": answer,
        "error": None,
        "judge_raw": None,
        "judge_error": None,
        "input_sha256": "a" * 64,
        "history_sha256": "b" * 64,
        **kw,
    }


def _report(answers: list[dict], *, status: str = "ok", judge: dict | None = None) -> dict:
    return {
        "fixture_set": {"path": "p", "sha256": "s", "n": 1},
        "extractor": {"name": "recorded", "quality": "pipeline-only"},
        "answerer": {"name": "fake", "quality": "wiring-only"},
        "judge": judge,
        "arms": ["memory", "no_memory"],
        "embedder": "token-hash-v1",
        "mutation_check": {"residual_rows": 0},
        "results": [{"id": "f", "status": status, "error": "boom" if status != "ok" else None, "answers": answers}],
    }


def test_scoreboard_grades_returned_answers_and_counts_memory_benefit():
    fixture_set = FixtureSet(path="p", fixtures=[_fixture()], sha256="s")
    report = _report(
        [
            _row("q1", "memory", "She lives in Leeds."),
            _row("q1", "no_memory", "I don't know."),
            _row("q2", "memory", "I don't know."),
            _row("q2", "no_memory", "I do not know."),
        ]
    )
    board = build_answer_scoreboard(report, fixture_set, source_sha="abc")
    memory = board["aggregate"]["per_arm"]["memory"]
    assert (memory["accuracy"], memory["answer_accuracy"], memory["abstention_accuracy"]) == (1.0, 1.0, 1.0)
    no_memory = board["aggregate"]["per_arm"]["no_memory"]
    assert (no_memory["answer_accuracy"], no_memory["false_abstentions"]) == (0.0, 1)
    assert board["aggregate"]["memory_benefit"] == {"paired_questions": 2, "memory_correct": 2, "no_memory_correct": 1}
    assert board["header"]["quality"] == "wiring-only" and board["header"]["judge"] is None


def test_errored_fixture_empty_answer_and_bad_judge_rows_are_invalid():
    fixture_set = FixtureSet(path="p", fixtures=[_fixture()], sha256="s")
    errored = build_answer_scoreboard(_report([], status="error"), fixture_set, source_sha="a")
    assert {r["status"] for r in errored["rows"]} == {"invalid"}
    assert errored["aggregate"]["per_arm"]["memory"]["accuracy"] is None  # not a zero

    judged = _report(
        [
            _row("q1", "memory", "Leeds", judge_raw="VERDICT: correct"),
            _row("q1", "no_memory", "", judge_raw=None),
            _row("q2", "memory", "I don't know.", judge_raw="no verdict here"),
            _row("q2", "no_memory", "I don't know.", judge_error="TimeoutError: slow"),
        ],
        judge={"name": "fake-judge"},
    )
    rows = {
        (r["question_id"], r["arm"]): r for r in build_answer_scoreboard(judged, fixture_set, source_sha="a")["rows"]
    }
    assert rows[("q1", "memory")]["judge_correct"] is True
    assert rows[("q1", "no_memory")]["reason"] == "empty answer"
    assert rows[("q2", "memory")]["reason"] == "judge verdict did not parse"
    assert rows[("q2", "no_memory")]["reason"].startswith("judge failed")


@pytest.fixture
def fake_inference(monkeypatch):
    seen: list[str] = []

    async def fake_call(provider, prompt, max_tokens=2048, span=None, **_):
        seen.append(prompt)
        return "She lives in Leeds."

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake_call)
    monkeypatch.setattr("kai_memory.ports.get_provider_for_model", lambda settings, model: object())
    return seen


def _request(**kw) -> AnswerRequest:
    base = {
        "fixture_id": "f",
        "question_id": "q",
        "arm": "memory",
        "question": "Where does Ilse live?",
        "asker": "Pat",
        "history": ({"role": "user", "sender": "Pat", "text": "hi"},),
        "memory_context": "## Active Associations\n- [fact] Ilse lives in Leeds",
        "retrieved_trace_ids": ("t1",),
    }
    return AnswerRequest(**{**base, **kw})


def test_live_answerer_prompt_and_call_cap(fake_inference):
    answerer = LiveAnswerer("haiku-4.5", max_calls=1, max_usd=1.0)
    assert asyncio.run(answerer(_request())) == "She lives in Leeds."
    assert "Ilse lives in Leeds" in fake_inference[0] and "Pat asks: Where does Ilse live?" in fake_inference[0]
    with pytest.raises(BudgetExceeded, match="call cap"):
        asyncio.run(answerer(_request()))
    assert answerer.describe()["calls"] == 1


def test_live_answerer_needs_known_prices_or_explicit_ones(fake_inference):
    with pytest.raises(ValueError, match="no list price"):
        LiveAnswerer("some-unpriced-model", max_calls=1, max_usd=1.0)
    LiveAnswerer("some-unpriced-model", max_calls=1, max_usd=1.0, usd_per_mtok=(3.0, 15.0))
    tiny = LiveAnswerer("haiku-4.5", max_calls=5, max_usd=0.000001)
    with pytest.raises(BudgetExceeded, match="could cost"):
        asyncio.run(tiny(_request()))
    assert fake_inference == []


def test_loader_refuses_a_question_that_is_both_or_neither(tmp_path):
    fixture = json.loads((FIXTURES / "01-flatmates-group.json").read_text())
    for bad in (
        {"id": "qx", "question": "?", "abstain": True, "expect_any": ["a"]},
        {"id": "qx", "question": "?"},
    ):
        fixture["expected"]["questions"] = [bad]
        fixture["_provenance"]["content_sha256"] = content_sha256(fixture)
        (tmp_path / "f.json").write_text(json.dumps(fixture))
        with pytest.raises(FixtureError, match="exactly one of an expected answer or abstain"):
            load_fixture_set(tmp_path)


def test_committed_questions_cover_every_axis():
    from memory_evaluation.eval.fixtures import AXES

    questions = [q for f in load_fixture_set(FIXTURES).fixtures for q in f["expected"]["questions"]]
    assert set().union(*(q["axes"] for q in questions)) == AXES
    assert sum(bool(q.get("abstain")) for q in questions) >= 3


def test_live_haiku55_answerer_disables_thinking_and_retains_text_headroom(monkeypatch):
    from memory_evaluation.eval.extractors import LIVE_MODEL

    seen = []

    async def fake_call(provider, prompt, **kwargs):
        seen.append(kwargs)
        return "She lives in Leeds."

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake_call)
    monkeypatch.setattr("kai_memory.ports.get_provider_for_model", lambda settings, model: object())
    answerer = LiveAnswerer(LIVE_MODEL, max_calls=1, max_usd=1)
    assert asyncio.run(answerer(_request())) == "She lives in Leeds."
    assert seen == [{"max_tokens": 400, "thinking_mode": "disabled"}]
    assert LiveAnswerer(LIVE_MODEL, max_calls=1, max_usd=1, max_tokens=300).max_tokens == 300


@pytest.mark.parametrize("answer", ["Not Leeds.", "Leedsville", "I don't know, perhaps Leeds."])
def test_positive_answer_rejects_fragments_negation_and_uncertainty(answer):
    assert check_answer({"expect_any": ["Leeds"]}, answer)["verdict"] == "incorrect"


def test_subject_label_requires_value_in_subject_clause():
    question = {"expect_any": ["Leeds"], "expect_subject": "Ilse", "wrong_subjects": ["Pat"]}
    assert check_answer(question, "Pat lives in Leeds. Ilse lives elsewhere.")["verdict"] == "incorrect"
    assert check_answer(question, "Ilse lives in Leeds.")["verdict"] == "correct"


@pytest.mark.parametrize("answer", [42, [], {}, "   "])
def test_malformed_answers_are_invalid_and_excluded(answer):
    fixture_set = FixtureSet(path="p", fixtures=[_fixture()], sha256="s")
    board = build_answer_scoreboard(_report([_row("q1", "memory", answer)]), fixture_set)
    aggregate = board["aggregate"]["per_arm"]["memory"]
    assert aggregate["n_valid"] == 0
    assert aggregate["denominators"]["accuracy"] == 0
    assert aggregate["accuracy"] is None


def test_abstention_with_prohibited_answer_never_passes():
    question = {"abstain": True, "prohibited": ["O positive"]}
    assert check_answer(question, "I don't know, but probably O positive.")["verdict"] == "incorrect"


@pytest.mark.parametrize(
    ("question", "answer", "verdict"),
    [
        ({"expect_all": ["not", "Leeds"]}, "Ilse does not live in Leeds.", "correct"),
        ({"expect_all": ["not", "Leeds"]}, "Ilse lives in Leeds.", "incorrect"),
        ({"expect_any": ["17 June"]}, "Her birthday is 17 June.", "correct"),
        ({"expect_any": ["17 June"]}, "Her birthday is not 17 June.", "incorrect"),
        ({"abstain": True}, "I do not know, but the answer is 17 June.", "incorrect"),
        ({"abstain": True}, "I don't know. The answer is 17 June.", "incorrect"),
        ({"abstain": True}, "I do not know the answer.", "correct"),
        (
            {"expect_any": ["Leeds"], "expect_subject": "Ilse", "wrong_subjects": ["Pat"]},
            "Ilse lives in Leeds.",
            "correct",
        ),
    ],
)
def test_frozen_conservative_positive_and_negative_answer_cases(question, answer, verdict):
    assert check_answer(question, answer)["verdict"] == verdict


@pytest.mark.parametrize("hashes", [{}, {"input_sha256": "a" * 64}, {"input_sha256": "x", "history_sha256": "x"}])
def test_missing_or_malformed_pair_hashes_never_measure_memory_benefit(hashes):
    answers = [_row("q1", arm, "Leeds") for arm in ("memory", "no_memory")]
    for row in answers:
        row.pop("input_sha256")
        row.pop("history_sha256")
        row.update(hashes)
    board = build_answer_scoreboard(_report(answers), FixtureSet(path="p", fixtures=[_fixture()], sha256="s"))
    assert board["aggregate"]["memory_benefit"]["paired_questions"] == 0
    assert board["aggregate"]["memory_benefit"]["invalid_pairs"] == 1


def test_answer_layer_checkpoint_store_errors_are_invalid():
    report = _report([_row("q1", "memory", "Leeds")])
    report["results"][0]["checkpoints"] = [{"store_errors": ["failed write"]}]
    board = build_answer_scoreboard(report, FixtureSet(path="p", fixtures=[_fixture()], sha256="s"))
    assert all(row["status"] == "invalid" for row in board["rows"])
    assert board["aggregate"]["per_arm"]["memory"]["accuracy"] is None


@pytest.mark.asyncio
async def test_answer_fixture_clock_is_pinned_during_replay(monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from kai_memory import ports
    from memory_evaluation.eval import answering

    now = datetime(2026, 1, 2, tzinfo=UTC)
    seen = []

    async def replay(fixture, db, extractor, then):
        from kai_memory.host_calls import utc_now

        seen.append(utc_now())
        return {"id": fixture["id"], "status": "ok"}

    monkeypatch.setattr(answering, "_run_fixture", replay)
    with ports.scoped({}, inherit=True):
        await answering.run_read_to_answer(
            FixtureSet(path="p", fixtures=[_fixture()], sha256="s"),
            SimpleNamespace(mutation_check=lambda: {}),
            answer_fn=None,
            fixture_clock=lambda f: now,
        )
    assert seen == [now]


@pytest.mark.parametrize(
    ("question", "answer", "verdict"),
    [
        ({"expect_any": ["not planning to relocate", "holiday"]}, "It is just a holiday.", "correct"),
        ({"expect_any": ["not planning to relocate", "holiday"]}, "She is not planning to relocate.", "correct"),
        ({"expect_any": ["not planning to relocate", "holiday"]}, "She is planning to relocate.", "incorrect"),
        ({"expect_any": ["not planning to relocate", "holiday"]}, "It is not a holiday.", "incorrect"),
        ({"expect_all": ["not", "Leeds"]}, "Ilse does not live in Leeds.", "correct"),
        ({"expect_all": ["not", "Leeds"]}, "Ilse lives in Leeds.", "incorrect"),
        ({"expect_any": ["吃素"]}, "她长期吃素", "correct"),
        ({"expect_any": ["吃素"]}, "她不吃素", "incorrect"),
        ({"expect_any": ["吃素"]}, "她没吃素", "incorrect"),
        ({"expect_any": ["不吃素"]}, "她不吃素", "correct"),
        ({"expect_any": ["素食者"], "expect_subject": "Mei Tan"}, "Mei Tan是素食者", "correct"),
        ({"expect_any": ["素食者"], "expect_subject": "Mei Tan"}, "Mei Tan不是素食者", "incorrect"),
        ({"expect_any": ["菜食主義者"]}, "彼女は菜食主義者です", "correct"),
        ({"expect_any": ["菜食主義者"]}, "彼女は菜食主義者ではない", "incorrect"),
    ],
)
def test_alternative_polarity_and_unspaced_han_kana_controls(question, answer, verdict):
    assert check_answer(question, answer)["verdict"] == verdict


@pytest.mark.parametrize(
    ("question", "answer", "verdict"),
    [
        (
            {"expect_any": ["vegetarian"], "expect_subject": "Mei Tan", "wrong_subjects": ["May Tan"]},
            "Mei Tan is vegetarian and does not eat fish or meat.",
            "correct",
        ),
        (
            {"expect_any": ["吃素"], "expect_subject": "Mei Tan", "wrong_subjects": ["May Tan"]},
            "她吃素\uff0c不吃鱼或肉",
            "correct",
        ),
        ({"expect_any": ["Dara Iliev"]}, "Dara Iliev, not Daria Iliev.", "correct"),
        ({"expect_any": ["Ravi"]}, "Ravi, not Ravina.", "correct"),
        ({"expect_any": ["operates the kiln"]}, "Ravi operates the kiln and has no ownership stake.", "correct"),
        ({"expect_any": ["lactose"]}, "He is lactose intolerant, not vegan.", "correct"),
        ({"expect_any": ["Leeds"]}, "Not Leeds or York.", "incorrect"),
        ({"expect_any": ["York"]}, "Not Leeds or York.", "incorrect"),
        ({"expect_any": ["allergic"], "expect_subject": "Dara"}, "Dara is not allergic.", "incorrect"),
        ({"expect_any": ["Dara"]}, "Dara is not allergic.", "incorrect"),
        ({"expect_any": ["Dara"]}, "not Dara, Daria", "incorrect"),
        ({"expect_all": ["not", "Leeds"]}, "Ilse does not live in Leeds.", "correct"),
        (
            {"expect_any": ["vegetarian"], "expect_subject": "Mei Tan", "wrong_subjects": ["May Tan"]},
            "May Tan, a rider, is vegetarian.",
            "incorrect",
        ),
        (
            {"expect_any": ["vegetarian"], "expect_subject": "Mei Tan", "wrong_subjects": ["May Tan"]},
            "Mei Tan is vegetarian, not May Tan.",
            "correct",
        ),
        ({"expect_any": ["吃素"]}, "她不吃素\uff0c不吃鱼或肉", "incorrect"),
    ],
)
def test_local_polarity_scopes_and_subject_contrasts(question, answer, verdict):
    assert check_answer(question, answer)["verdict"] == verdict


@pytest.mark.parametrize(
    ("fixture_name", "question_id", "answer"),
    [
        ("07-cycling-club.json", "q_memory", "Mei Tan is vegetarian and does not eat fish or meat."),
        ("07-cycling-club.json", "q_memory", "她吃素\uff0c不吃鱼或肉"),
        ("09-theatre-props.json", "q_agree", "Dara Iliev, not Daria Iliev."),
        ("12-ceramics-studio.json", "q_agree", "Ravi, not Ravina."),
        ("12-ceramics-studio.json", "q_correction", "Ravi operates the kiln and has no ownership stake."),
        ("10-bakery-team.json", "q_correction", "He is lactose intolerant, not vegan."),
    ],
)
def test_actual_fixture_questions_accept_supported_local_contrasts(fixture_name, question_id, answer):
    fixture = json.loads((FIXTURES / fixture_name).read_text())
    question = next(q for q in fixture["expected"]["questions"] if q["id"] == question_id)
    assert check_answer(question, answer)["verdict"] == "correct"


def test_explicit_forbidden_fact_denial_does_not_hide_a_different_forbidden_assertion():
    assert (
        check_answer({"expect_any": ["lactose"], "prohibited": ["vegan"]}, "He is lactose intolerant, not vegan.")[
            "verdict"
        ]
        == "correct"
    )
    assert (
        check_answer({"expect_any": ["Leeds"], "prohibited": ["not vegetarian"]}, "Leeds. She is not vegetarian.")[
            "verdict"
        ]
        == "incorrect"
    )


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        ("Il est intolérant au lactose.", "correct"),
        ("Il n'est pas intolérant au lactose.", "incorrect"),
        ("Il n\u2019est pas intolérant au lactose.", "incorrect"),
        ("Il ne peut consommer de lactose.", "incorrect"),
        ("Il ne consomme jamais de lactose.", "incorrect"),
        ("Il n'ose consommer de lactose.", "incorrect"),
        ("Non au lactose.", "incorrect"),
        ("Pas de lactose.", "incorrect"),
        ("Jamais de lactose.", "incorrect"),
        ("Il passe au lactose.", "correct"),
        ("Pascal préfère le lactose.", "correct"),
        ("Le nonogramme mentionne le lactose.", "correct"),
        ("Nemo préfère le lactose.", "correct"),
    ],
)
def test_bounded_french_negation_preserves_positive_word_fragments(answer, verdict):
    assert check_answer({"expect_any": ["lactose"]}, answer)["verdict"] == verdict


@pytest.mark.parametrize(
    ("question", "answer", "verdict"),
    [
        ({"expect_any": ["pas de lactose", "yaourt"]}, "Il préfère le yaourt.", "correct"),
        ({"expect_all": ["pas", "lactose"]}, "Il ne consomme pas de lactose.", "correct"),
        ({"expect_any": ["Leeds"]}, "Not Leeds.", "incorrect"),
        ({"expect_any": ["Leeds"]}, "Ilse lives in Leeds.", "correct"),
        ({"expect_any": ["Leeds"]}, "Она не живёт в Leeds.", "incorrect"),
        ({"expect_any": ["吃素"]}, "她不吃素", "incorrect"),
    ],
)
def test_french_controls_preserve_per_alternative_shared_and_other_language_polarity(question, answer, verdict):
    assert check_answer(question, answer)["verdict"] == verdict


@pytest.mark.parametrize(
    ("question_id", "answer", "verdict"),
    [
        ("q_date", "Le 3 mars 2026, et non le 28 février 2026.", "correct"),
        ("q_date", "Le 3 mars 2026, pas le 28 février 2026.", "correct"),
        ("q_date", "Le 28 février 2026.", "incorrect"),
        ("q_correction", "Il n'est pas intolérant au lactose.", "incorrect"),
        ("q_correction", "Il n\u2019est pas intolérant au lactose.", "incorrect"),
        ("q_correction", "Il est intolérant au lactose, et non végétalien.", "correct"),
    ],
)
def test_actual_bakery_questions_use_bounded_french_negation(question_id, answer, verdict):
    fixture = json.loads((FIXTURES / "10-bakery-team.json").read_text())
    question = next(q for q in fixture["expected"]["questions"] if q["id"] == question_id)
    assert check_answer(question, answer)["verdict"] == verdict


@pytest.mark.parametrize(
    ("filename", "subject", "wrong_subject", "correct", "contrast", "wrong_possessive"),
    [
        (
            "09-theatre-props.json",
            "Dara Iliev",
            "Daria Iliev",
            "Dara Iliev has the latex allergy.",
            "Dara Iliev, not Daria Iliev, has the latex allergy.",
            "Daria Iliev, Dara Iliev{apostrophe}s colleague, has the latex allergy.",
        ),
        (
            "10-bakery-team.json",
            "Ivo Marin",
            "Ivo Celik",
            "Ivo Marin signed the apprenticeship at Grain Bleu.",
            "Ivo Marin, not Ivo Celik, signed it.",
            "Ivo Celik, Ivo Marin{apostrophe}s cousin, signed it.",
        ),
        (
            "12-ceramics-studio.json",
            "Ravi Sen",
            "Ravina Sen",
            "Ravi Sen grew up in Mallowford.",
            "Ravi Sen, not Ravina Sen, grew up in Mallowford.",
            "Ravina Sen, Ravi Sen{apostrophe}s studio colleague, grew up in Mallowford.",
        ),
    ],
)
@pytest.mark.parametrize("apostrophe", ["'", "\u2019"])
def test_actual_overlapping_name_probes_reject_possessive_subject_distractors(
    filename, subject, wrong_subject, correct, contrast, wrong_possessive, apostrophe
):
    fixture = load_fixture_set(FIXTURES / filename).fixtures[0]
    question = next(q for q in fixture["expected"]["questions"] if q["id"] == "q_agree")
    assert question["expect_subject"] == subject
    assert question["wrong_subjects"] == [wrong_subject]
    assert check_answer(question, correct)["verdict"] == "correct"
    assert check_answer(question, contrast)["verdict"] == "correct"
    assert check_answer(question, wrong_possessive.format(apostrophe=apostrophe))["verdict"] == "incorrect"


@pytest.mark.parametrize("answer", ["He signed at Grain Bleu.", "Ivo Marin's apprenticeship is at Grain Bleu."])
def test_possessive_guard_preserves_unnamed_and_owned_fact_answers_without_wrong_subject(answer):
    question = {"expect_any": ["Grain Bleu"], "expect_subject": "Ivo Marin", "wrong_subjects": ["Ivo Celik"]}
    assert check_answer(question, answer)["verdict"] == "correct"
