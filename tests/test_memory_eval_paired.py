"""Offline paired harness evidence; fake transport proves wiring, never model quality."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from kai_memory import ports
from memory_evaluation.eval import answering
from memory_evaluation.eval.fixtures import FixtureSet
from memory_evaluation.eval.providers import TokenHashEmbedder, evaluation_embeddings


def corpus(history="full"):
    return FixtureSet(
        path="synthetic",
        sha256="c",
        fixtures=[
            {
                "id": "f",
                "start_at": "2026-01-01T00:00:00+00:00",
                "split": "held_out",
                "_provenance": {"source": "synthetic", "content_sha256": "c"},
                "participants": [{"key": "p", "name": "Pat", "telegram_user_id": 1}],
                "messages": [{"from": "p", "minute": 0, "text": "Ilse moved to Leeds."}],
                "expected": {
                    "questions": [
                        {
                            "id": "q",
                            "question": "Where does Ilse live?",
                            "asker": "p",
                            "history": history,
                            "expect_any": ["Leeds"],
                            "availability": "both_agree",
                            "distance_user_turns": 0,
                        }
                    ]
                },
            }
        ],
    )


class Answer:
    def describe(self):
        return {"name": "fake", "model": "fake-model", "quality": "wiring-only"}

    async def __call__(self, request):
        return "Leeds"


@pytest.fixture
def harness(monkeypatch):
    state = {"writes": 1}
    calls = []

    async def fingerprint(session):
        return deepcopy(state)

    async def replay(fixture, db, extractor, then):
        result = {
            "id": fixture["id"],
            "status": "ok",
            "snapshot": {"written": "Leeds"},
            "checkpoints": [{"extraction_s": 0.2}],
        }
        await then(object(), fixture, result, None)
        return result

    class Priming:
        def __init__(self, session, **kwargs):
            self.observation = kwargs.get("observation")

        async def prime(self, text, **kwargs):
            calls.append((text, kwargs))
            return "Ilse lives in Leeds", ["t1"]

    monkeypatch.setattr(answering, "store_fingerprint", fingerprint)
    monkeypatch.setattr(answering, "_run_fixture", replay)
    monkeypatch.setattr(answering, "_chat_context", lambda *args: object())
    monkeypatch.setattr(answering, "MemoryPrimingService", Priming)
    return state, calls


async def run(corpus_set, **kwargs):
    return await answering.run_read_to_answer(
        corpus_set, SimpleNamespace(mutation_check=lambda: {}), answer_fn=Answer(), arms=answering.M4_ARMS, **kwargs
    )


@pytest.mark.asyncio
async def test_three_arms_share_all_nonretrieval_inputs_and_actual_production_cues(harness):
    _, calls = harness
    raw = await run(corpus())
    rows = raw["results"][0]["answers"]
    assert [r["arm"] for r in rows] == list(answering.M4_ARMS)
    assert rows[0]["retrieved"] == [] and rows[0]["memory_context"] == ""
    assert all(answering.comparable_pair(rows[0], row) for row in rows[1:])
    assert len(calls) == 2 and all(not kw["record_access"] for _, kw in calls)
    cues = calls[1][1]["window_cues"]
    from kai_memory.retrieval.iterative_retrieval import build_segment_cues

    assert [(cue.text, cue.weight) for cue in cues.segments] == build_segment_cues(
        ["Ilse moved to Leeds.", "Where does Ilse live?"]
    )
    assert cues.is_windowed
    assert all(row["timing"]["answer_s"] is not None for row in rows)
    assert all(row["usage"]["status"] == "unknown" for row in rows)
    board = answering.build_answer_scoreboard(raw, corpus())
    assert all(pair["n_valid"] == 1 for pair in board["aggregate"]["paired_comparisons"].values())


@pytest.mark.asyncio
async def test_no_history_window_is_honestly_single_cue(harness):
    raw = await run(corpus("none"))
    row = raw["results"][0]["answers"][-1]
    assert row["is_windowed"] is False
    assert row["window_cues"] == [{"text": "Where does Ilse live?", "weight": 1.0}]


@pytest.mark.asyncio
async def test_real_injected_adapter_is_used_and_restored(harness):
    class Adapter(TokenHashEmbedder):
        seen = []

        async def embed(self, texts, **kwargs):
            self.seen.extend(texts)
            return await super().embed(texts, **kwargs)

    adapter = Adapter()
    previous = object()
    with ports.scoped({"embedder": previous}, inherit=True):
        raw = await run(
            corpus(), embedding_adapter=adapter, embedding_descriptor={"name": "fake-real-transport", "kind": "real"}
        )
        assert ports.embedder is previous
    assert adapter.seen and raw["embedding_metadata"]["kind"] == "real"


@pytest.mark.asyncio
async def test_actual_store_mutation_invalidates_every_arm(harness):
    state, _ = harness

    class MutatingAnswer(Answer):
        async def __call__(self, request):
            state["writes"] += 1
            return "Leeds"

    raw = await answering.run_read_to_answer(
        corpus(), SimpleNamespace(mutation_check=lambda: {}), answer_fn=MutatingAnswer(), arms=answering.M4_ARMS
    )
    assert all(row["error"] for row in raw["results"][0]["answers"])
    assert all(row["status"] == "invalid" for row in answering.build_answer_scoreboard(raw, corpus())["rows"])


@pytest.mark.asyncio
async def test_window_embedding_failure_invalidates_arm_without_fallback(harness):
    class Broken:
        async def embed(self, *args, **kwargs):
            raise RuntimeError("synthetic embedding failure")

    raw = await run(corpus(), embedding_adapter=Broken(), embedding_descriptor={"name": "broken", "kind": "real"})
    rows = raw["results"][0]["answers"]
    assert rows[-1]["answer"] is None and rows[-1]["error"]
    assert rows[0]["error"] is None


@pytest.mark.parametrize("field", answering.PAIR_HASHES)
def test_missing_m4_evidence_never_compares(field):
    row = {key: "a" * 64 for key in answering.PAIR_HASHES}
    row["paired_metadata"] = "m4/v1"
    missing = dict(row)
    missing.pop(field)
    assert not answering.comparable_pair(row, missing)


def test_injected_adapter_requires_truthful_explicit_descriptor():
    with pytest.raises(ValueError, match="descriptor"):
        with evaluation_embeddings(TokenHashEmbedder()):
            pass


@pytest.mark.asyncio
async def test_private_answer_factory_supplies_projection_evidence_without_overwriting_store(harness):
    seen = []

    class Bound(Answer):
        async def evidence(self, request):
            return {
                "common_context_sha256": "d" * 64,
                "model_sha256": "e" * 64,
                "stageC_policy_sha256": "f" * 64,
                "messages": [{"text": request.memory_context}],
                "store_sha256": "forged",
            }

    def factory(session, fixture):
        seen.append((session, fixture["id"]))
        return Bound()

    raw = await run(corpus(), answer_factory=factory)
    assert len(seen) == 1
    rows = raw["results"][0]["answers"]
    assert all(row["prompt_sha256"] == "d" * 64 for row in rows)
    assert all(row["store_sha256"] != "forged" for row in rows)
    assert all(answering.comparable_pair(rows[0], row) for row in rows[1:])


@pytest.mark.asyncio
async def test_unknown_answer_model_does_not_claim_paired_comparability(harness):
    async def unknown_model(request):
        return "Leeds"

    raw = await answering.run_read_to_answer(
        corpus(), SimpleNamespace(mutation_check=lambda: {}), answer_fn=unknown_model, arms=answering.M4_ARMS
    )
    board = answering.build_answer_scoreboard(raw, corpus())
    assert all(value["n_valid"] == 0 for value in board["aggregate"]["paired_comparisons"].values())


@pytest.mark.parametrize("change", ["missing_after", "changed_after", "missing_metadata", "unknown_metadata"])
def test_m4_comparison_requires_both_store_validation_receipts(change):
    row = {key: "a" * 64 for key in answering.PAIR_HASHES}
    row.update({"paired_metadata": "m4/v1", "store_after_sha256": "a" * 64})
    assert answering.comparable_pair(row, row)
    other = dict(row)
    if change == "missing_after":
        other.pop("store_after_sha256")
    elif change == "changed_after":
        other["store_after_sha256"] = "b" * 64
    elif change == "missing_metadata":
        other.pop("paired_metadata")
    else:
        other["paired_metadata"] = "m4/unknown"
    assert not answering.comparable_pair(row, other)


@pytest.mark.asyncio
async def test_answer_failure_keeps_retrieval_timing_separate(harness, monkeypatch):
    import time

    original_monotonic = time.monotonic
    ticks = iter(range(1000))
    monkeypatch.setattr(answering, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    assert time.monotonic is original_monotonic

    class Failing(Answer):
        async def __call__(self, request):
            raise RuntimeError("synthetic failed answer")

    raw = await answering.run_read_to_answer(
        corpus(), SimpleNamespace(mutation_check=lambda: {}), answer_fn=Failing(), arms=answering.M4_ARMS
    )
    for row in raw["results"][0]["answers"]:
        assert row["error"]
        assert row["timing"]["answer_s"] is not None
        assert row["timing"]["failure_s"] > row["timing"]["retrieval_s"]


@pytest.mark.asyncio
@pytest.mark.parametrize("completion", [None, "raise"])
async def test_failed_extraction_preserves_attempt_error_prompt_and_timings(monkeypatch, completion):
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock
    from memory_evaluation.eval import runner

    class Extractor:
        calls = []

        def arm(self, *args):
            pass

        async def __call__(self, *args, **kwargs):
            if completion == "raise":
                raise RuntimeError("synthetic inference failure")
            return None

    class Service:
        def __init__(self, session, llm_text_fn):
            self.call = llm_text_fn

        async def extract_after_cursor(self, *args):
            await self.call(None, "synthetic extraction prompt")
            raise RuntimeError("no completion")

    @asynccontextmanager
    async def transaction():
        yield

    monkeypatch.setattr(runner, "ExtractionService", Service)
    result = {"labels": {}, "checkpoints": []}
    session = SimpleNamespace(begin_nested=transaction, execute=AsyncMock(), commit=AsyncMock())
    with pytest.raises(RuntimeError):
        await runner._run_checkpoint(
            session, {}, Extractor(), result, 1, {"after": 4, "response": {"memories": []}}, None, object()
        )
    call = result["extraction_calls"][0]
    assert call["prompt"] == "synthetic extraction prompt" and len(call["prompt_sha256"]) == 64
    assert call["error"] and call["response"] is None and call["duration_s"] is not None
    assert result["checkpoint_attempts"][0]["status"] == "error"
    assert result["checkpoint_attempts"][0]["duration_s"] is not None
    session.commit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["projection", "answer"])
async def test_mutated_m4_model_input_is_invalid_even_when_other_arms_are_isolated(harness, phase):
    class Mutating(Answer):
        async def evidence(self, request):
            if phase == "projection":
                request.history[0]["text"] = "forged history"
            return {}

        async def __call__(self, request):
            if phase == "answer":
                request.history[0]["text"] = "forged history"
            return "Leeds"

    raw = await answering.run_read_to_answer(
        corpus(), SimpleNamespace(mutation_check=lambda: {}), answer_fn=Mutating(), arms=answering.M4_ARMS
    )
    assert all("input mutated" in row["error"] for row in raw["results"][0]["answers"])
    assert all(row["status"] == "invalid" for row in answering.build_answer_scoreboard(raw, corpus())["rows"])


@pytest.mark.asyncio
async def test_mutable_projection_evidence_is_copied_before_answer_transport(harness):
    class Mutating(Answer):
        payload = {"messages": [{"text": "original"}]}

        async def evidence(self, request):
            return self.payload

        async def __call__(self, request):
            self.payload["messages"][0]["text"] = "changed"
            return "Leeds"

    raw = await answering.run_read_to_answer(
        corpus(), SimpleNamespace(mutation_check=lambda: {}), answer_fn=Mutating(), arms=answering.M4_ARMS
    )
    assert raw["results"][0]["answers"][0]["projection_evidence"]["messages"][0]["text"] == "original"
    assert "projected payload mutated" in raw["results"][0]["answers"][0]["error"]


@pytest.mark.asyncio
async def test_raw_m4_store_mismatch_invalidates_quality_not_only_pairing(harness):
    raw = await run(corpus())
    raw["results"][0]["answers"][0]["store_after_sha256"] = "b" * 64
    board = answering.build_answer_scoreboard(raw, corpus())
    assert board["rows"][0]["status"] == "invalid"
    assert board["aggregate"]["per_arm"]["no_memory"]["n_valid"] == 0
    assert board["aggregate"]["per_arm"]["no_memory"]["accuracy"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_fast", [False, True])
async def test_opt_in_fail_fast_stops_after_failed_arm_and_preserves_partial_evidence(harness, monkeypatch, fail_fast):
    selected = corpus()
    second = deepcopy(selected.fixtures[0])
    second["id"] = "second"
    selected.fixtures.append(second)
    calls = []

    class FailedAnswer(Answer):
        async def __call__(self, request):
            calls.append(request.fixture_id)
            raise RuntimeError("synthetic answer failure")

    async def replay(fixture, db, extractor, then, **kwargs):
        result = {
            "id": fixture["id"],
            "status": "ok",
            "checkpoints": [],
            "readback": [],
            "extraction_calls": [{"prompt": "preserved extraction evidence"}],
        }
        try:
            await then(object(), fixture, result, None)
        except Exception as error:
            result.update(status="error", error=str(error))
        return result

    monkeypatch.setattr(answering, "_run_fixture", replay)
    raw = await answering.run_read_to_answer(
        selected,
        SimpleNamespace(mutation_check=lambda: {}),
        answer_fn=FailedAnswer(),
        arms=answering.M4_ARMS,
        fail_fast=fail_fast,
    )
    assert len(calls) == (1 if fail_fast else 6)
    assert len(raw["results"]) == (1 if fail_fast else 2)
    assert raw["results"][0]["answers"][0]["error"]
    assert raw["results"][0]["extraction_calls"][0]["prompt"] == "preserved extraction evidence"
    assert raw["stopped_early"] is fail_fast
