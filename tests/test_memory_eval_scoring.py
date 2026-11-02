"""Scoreboard scoring over a hand-built run result (DB-840). Pure: no database."""

from __future__ import annotations


import pytest

from memory_evaluation.eval import build_scoreboard
from memory_evaluation.eval.fixtures import FixtureSet
from memory_evaluation.eval.scoring import GAPS, score_fixture
from kai_memory import ports
from kai_memory.sdk import _parse_object

PAT, LEE = "e-pat", "e-lee"


def _fixture() -> dict:
    return {
        "id": "f",
        "_provenance": {"source": "synthetic", "approved_by": None, "version": 1, "content_sha256": "x"},
        "participants": [{"key": "p_pat", "telegram_user_id": 11, "name": "Pat Example"}],
        "messages": [{"from": "p_pat", "minute": 0, "text": "..."}] * 4,
        "expected": {
            "writes": [
                {"label": "job", "fact": {"value_contains": "Example Co"}, "subject": "Pat Example", "durable": True},
                {"label": "city", "fact": {"value_contains": "Leeds"}, "subject": "Pat Example", "durable": True},
                {"label": "pet", "content_contains": ["parrot"]},
            ],
            "prohibited": [{"id": "echo", "kind": "assistant_echo", "content_contains": ["moving abroad"]}],
            "supersessions": [{"old": "old_job", "new": "job"}],
            "must_stay_valid": ["keep"],
            "dedup": [{"id": "pat_once", "names": ["Pat Example"], "type": "person"}],
            "entities": [{"name": "Lee Other", "type": "person", "participant": None}],
            "readback": [
                {"id": "rb_job", "cue": "where does Pat work", "expect_labels": ["job"], "forbid_labels": ["old_job"]},
                {"id": "rb_echo", "cue": "is Pat moving", "expect_labels": [], "forbid_contains": ["moving abroad"]},
            ],
        },
    }


def _trace(tid: str, content: str, facts=(), **kw) -> dict:
    view = {"id": tid, "content": content, "facts": list(facts), "pre_state": False, "invalidated": False}
    view.update({"invalidated_by": None, "durability": None, **kw})
    return view


def _result() -> dict:
    traces = [
        _trace("t-old", "Pat worked at Old Co", pre_state=True, invalidated=True, invalidated_by="t-job"),
        _trace("t-keep", "Pat likes tea", pre_state=True),
        _trace(
            "t-job", "Pat works at Example Co", [{"value": "Example Co", "subject_id": PAT, "durability": "durable"}]
        ),
        # City captured, but attributed to the wrong person and stamped transient.
        _trace("t-city", "Pat moved to Leeds", [{"value": "Leeds", "subject_id": LEE, "durability": "transient"}]),
        _trace("t-echo", "Pat is moving abroad next year"),
        _trace("t-noise", "It rained all day"),
    ]
    return {
        "status": "ok",
        "labels": {"old_job": "t-old", "keep": "t-keep", "job": "t-job", "city": "t-city"},
        "snapshot": {
            "traces": traces,
            "entities": [
                {"id": PAT, "name": "Pat Example", "type": "person", "aliases": [], "telegram_user_id": 11},
                {"id": LEE, "name": "Lee Other", "type": "person", "aliases": [], "telegram_user_id": None},
            ],
        },
        "checkpoints": [{"stats": {"messages_read": 3}, "store_errors": []}],
        "messages_saved": 4,
        "prompt_checks": [
            {"id": "recall_marker", "kind": "recalled_echo", "expect": "absent", "found": True},
            {"id": "control", "kind": "control", "expect": "present", "found": True},
        ],
        "readback": [
            {"id": "rb_job", "retrieved": ["t-job", "t-old"], "formatted": "Pat works at Example Co"},
            {"id": "rb_echo", "retrieved": ["t-echo"], "formatted": "Pat is moving abroad next year"},
        ],
    }


def test_score_fixture_reports_every_kind_of_finding():
    row = score_fixture(_fixture(), _result())
    assert row["written"] == ["job", "city"] and row["omitted"] == ["pet"]
    assert row["capture_recall"] == 0.6667
    assert [h["id"] for h in row["prohibited_hits"]] == ["echo"]
    assert [u["trace_id"] for u in row["unexpected"]] == ["t-noise"]
    assert row["attribution"]["checked"] == 3 and row["attribution"]["correct"] == 2
    assert row["attribution"]["misses"] == [{"label": "city", "subject_id": LEE}]
    assert row["durability"]["misses"] == [{"label": "city", "expected": "durable", "got": "transient"}]
    assert row["dedup"]["met"] == 1
    assert row["supersession"] == {"checked": 2, "met": 2, "misses": []}
    assert {(e["id"], e["where"]) for e in row["echo_leaks"]} == {("echo", "store"), ("recall_marker", "prompt")}
    probes = {p["id"]: p for p in row["readback"]["probes"]}
    assert probes["rb_job"] == {
        "id": "rb_job",
        "pass": False,
        "expect_ranks": {"job": 1},
        "forbidden_hits": ["old_job"],
    }
    assert probes["rb_echo"]["forbidden_hits"] == ["moving abroad"]
    assert row["extraction_coverage"] == 0.75
    assert row["prompt_controls"] == {"checked": 1, "met": 1, "misses": []}


def test_an_errored_run_is_invalid_and_stays_out_of_the_aggregate():
    fixture_set = FixtureSet(path="p", fixtures=[_fixture(), {**_fixture(), "id": "g"}], sha256="s")
    report = {
        "fixture_set": {"path": "p", "sha256": "s", "n": 2},
        "extractor": {"name": "recorded", "quality": "pipeline-only"},
        "embedder": "token-hash-v1",
        "mutation_check": {"residual_rows": 0, "changed_tables": []},
        "results": [_result(), {"id": "g", "status": "error", "error": "boom", "labels": {}}],
    }
    report["results"][0]["id"] = "f"
    board = build_scoreboard(report, fixture_set, source_sha="abc")
    assert board["fixtures"][1] == {"id": "g", "status": "invalid", "error": "boom"}
    agg = board["aggregate"]
    assert (agg["n_valid"], agg["n_invalid"]) == (1, 1)
    assert agg["capture_recall"] == 0.6667  # the invalid fixture is not a zero
    assert agg["fabrication_rate"] == 0.25  # 1 prohibited hit over 4 written traces
    assert board["header"]["source_sha"] == "abc" and board["header"]["judge"] is None
    assert len(board["header"]["prompt_sha256"]) == 64
    assert [g["instrument"] for g in board["gaps"]] == [g["instrument"] for g in GAPS] + ["extractor quality"]


def test_a_trace_matching_two_prohibited_specs_counts_once_in_the_fabrication_rate():
    fixture = _fixture()
    fixture["expected"]["prohibited"] = [
        *fixture["expected"]["prohibited"],
        {"id": "echo_again", "kind": "assistant_echo", "content_contains": ["moving abroad"]},
    ]
    report = {
        "fixture_set": {"path": "p", "sha256": "s", "n": 1},
        "extractor": {"name": "recorded", "quality": "pipeline-only"},
        "embedder": "token-hash-v1",
        "mutation_check": {"residual_rows": 0, "changed_tables": []},
        "results": [{**_result(), "id": "f"}],
    }
    agg = build_scoreboard(report, FixtureSet(path="p", fixtures=[fixture], sha256="s"), source_sha="abc")["aggregate"]
    assert agg["prohibited_hits"] == 2  # one trace, two specs
    assert agg["fabrication_rate"] == 0.25  # still 1 fabricated trace over 4 written


def test_live_runs_do_not_carry_the_recorded_extractor_gap():
    fixture_set = FixtureSet(path="p", fixtures=[_fixture()], sha256="s")
    result = {**_result(), "id": "f"}
    report = {
        "fixture_set": {},
        "extractor": {"name": "live", "quality": "extractor-and-pipeline"},
        "embedder": "token-hash-v1",
        "mutation_check": {},
        "results": [result],
    }
    board = build_scoreboard(report, fixture_set, source_sha="abc")
    assert all(g["instrument"] != "extractor quality" for g in board["gaps"])
    assert board["header"]["quality"] == "extractor-and-pipeline"


def test_report_paths_require_an_explicit_host_root(tmp_path):
    from memory_evaluation.eval.runner import _repo_relative

    path = str(tmp_path / "fixtures" / "tiny.json")
    assert _repo_relative(path) == path
    assert _repo_relative(path, tmp_path) == "fixtures/tiny.json"
    assert _repo_relative(path, tmp_path / "elsewhere") == path


def test_checkpoint_store_failure_is_invalid():
    result = _result()
    result["checkpoints"][0]["store_errors"] = ["write failed"]
    row = score_fixture(_fixture(), result)
    assert row["status"] == "invalid"
    assert row["errors"] == ["write failed"]


def test_missing_negative_readback_probe_cannot_pass():
    result = _result()
    result["readback"] = []
    row = score_fixture(_fixture(), result)
    assert row["readback"]["checked"] == 0
    assert row["readback"]["passed"] == 0
    assert row["readback"]["invalid"] == 2


def test_fact_values_match_whole_tokens():
    from memory_evaluation.eval.matching import matching_fact, text_contains

    assert matching_fact({"fact": {"value_contains": "Leeds"}}, {"facts": [{"value": "Leedsville"}]}) is None
    assert not text_contains("19 November", "9")
    assert text_contains("9 November", "9")


def test_invalid_readback_is_visible_in_aggregate():
    result = _result()
    result["id"] = "f"
    result["readback"] = []
    report = {
        "results": [result],
        "extractor": {"name": "recorded"},
        "fixture_set": {},
        "embedder": "fake",
        "mutation_check": {},
    }
    board = build_scoreboard(report, FixtureSet(path="p", fixtures=[_fixture()], sha256="s"))
    assert board["aggregate"]["n_invalid_readback"] == 2
    assert board["aggregate"]["readback_accuracy"] is None
    assert board["aggregate"]["denominators"]["readback_accuracy"] == 0


@pytest.mark.parametrize("response", ['{"memories": []}', '```json\n{"memories": []}\n```'])
def test_checkpoint_accepts_successful_empty_extraction(response):
    from types import SimpleNamespace
    from memory_evaluation.eval.runner import _validate_checkpoint

    with ports.scoped({"parse_json_object_response": _parse_object}, inherit=True):
        _validate_checkpoint(
            {"messages_read": 4, "cursor_advanced": True}, object(), 4, [SimpleNamespace(response=response)]
        )


@pytest.mark.parametrize(
    "change,response,error_type,message",
    [
        ({"messages_read": 3}, '{"memories": []}', RuntimeError, "^checkpoint extraction incomplete or failed$"),
        ({"backlog_remaining": True}, '{"memories": []}', RuntimeError, "^checkpoint extraction incomplete or failed$"),
        (
            {"skipped_after_retries": ["m"]},
            '{"memories": []}',
            RuntimeError,
            "^checkpoint extraction incomplete or failed$",
        ),
        ({"cursor_advanced": False}, '{"memories": []}', RuntimeError, "^checkpoint extraction incomplete or failed$"),
        ({"store_failures": 1}, '{"memories": []}', RuntimeError, "^checkpoint extraction incomplete or failed$"),
        ({}, "{}", RuntimeError, "^checkpoint extraction has invalid completion schema$"),
        ({}, "not json", RuntimeError, "^checkpoint extraction has invalid completion schema$"),
        ({}, '{"memories": "failed"}', RuntimeError, "^checkpoint extraction has invalid completion schema$"),
    ],
)
def test_checkpoint_rejects_incomplete_or_malformed_extraction(change, response, error_type, message):
    from types import SimpleNamespace
    from memory_evaluation.eval.runner import _validate_checkpoint

    stats = {"messages_read": 4, "cursor_advanced": True, **change}
    with (
        ports.scoped({"parse_json_object_response": _parse_object}, inherit=True),
        pytest.raises(error_type, match=message) as error,
    ):
        _validate_checkpoint(stats, object(), 4, [SimpleNamespace(response=response)])
    assert type(error.value) is error_type


def test_extraction_coverage_separates_intentional_tail_from_skipped_rows():
    result = _result()
    result["messages_saved"] = 30
    result["checkpoints"] = [{"after": 16, "stats": {"messages_read": 16}, "store_errors": []}]
    row = score_fixture(_fixture(), result)
    assert row["extraction_coverage"] == 1.0
    assert row["extraction_evidence"] == {
        "messages_read": 16,
        "eligible_messages": 16,
        "retained_history_messages": 30,
        "intentionally_unextracted": 14,
        "last_checkpoint": 16,
    }
    result["checkpoints"][0]["stats"]["messages_read"] = 15
    assert score_fixture(_fixture(), result)["extraction_coverage"] == 0.9375


@pytest.mark.asyncio
async def test_checkpoint_validates_actual_completion_not_adapter_log(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from memory_evaluation.eval import runner

    class MisleadingExtractor:
        calls = [SimpleNamespace(response='{"memories": []}')]

        def arm(self, *args):
            pass

        def consumed(self):
            return True

        async def __call__(self, *args, **kwargs):
            return "{}"

    class Service:
        store_errors = []

        def __init__(self, session, llm_text_fn):
            self.inference = llm_text_fn

        async def extract_after_cursor(self, *args):
            await self.inference(None, "synthetic prompt")
            return {"messages_read": 4, "cursor_advanced": True}, object()

    @asynccontextmanager
    async def transaction():
        yield

    session = SimpleNamespace(begin_nested=transaction, execute=AsyncMock(), commit=AsyncMock())
    monkeypatch.setattr(runner, "ExtractionService", Service)
    with (
        ports.scoped({"parse_json_object_response": _parse_object}, inherit=True),
        pytest.raises(RuntimeError, match=r"^checkpoint extraction has invalid completion schema$"),
    ):
        await runner._run_checkpoint(
            session,
            {},
            MisleadingExtractor(),
            {"labels": {}, "checkpoints": []},
            1,
            {"after": 4, "response": {"memories": []}},
            None,
            object(),
        )
    session.commit.assert_not_awaited()


@pytest.mark.parametrize(
    ("text", "needle", "expected"),
    [
        ("她长期吃素", "吃素", True),
        ("Mei Tan是素食者", "Mei Tan", True),
        ("彼女はベジタリアンです", "ベジタリアン", True),
        ("彼女は菜食主義者です", "菜食主義者", True),
        ("Leedsville", "Leeds", False),
        ("Ярослав", "Яр", False),
        ("19 November", "9", False),
        ("채식주의자", "채식", False),
        ("Mei Tan은 채식주의자", "Mei Tan", False),
    ],
)
def test_matching_han_kana_edges_preserve_latin_cyrillic_and_hangul_boundaries(text, needle, expected):
    from memory_evaluation.eval.matching import text_contains

    assert text_contains(text, needle) is expected
