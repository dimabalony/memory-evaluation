"""Recorded and live extractors, and the ExtractionService ``llm_text_fn`` seam (DB-840).

No test here reaches a model: the live extractor's caps are exercised against a
fake ``call_llm_text``.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from kai_memory.extraction.memory_extraction import ExtractionService
from memory_evaluation.eval.extractors import (
    BudgetExceeded,
    LiveExtractor,
    RecordedExtractor,
    UnexpectedExtractionCall,
)


def test_recorded_extractor_serves_one_response_per_checkpoint():
    ex = RecordedExtractor()
    with pytest.raises(UnexpectedExtractionCall):
        asyncio.run(ex(None, "prompt"))
    ex.arm("cp1", '{"memories": []}')
    assert not ex.consumed()
    assert asyncio.run(ex(None, "the prompt", max_tokens=8192, span=None)) == '{"memories": []}'
    assert ex.consumed()
    assert ex.calls[0].checkpoint == "cp1" and len(ex.calls[0].prompt_sha256) == 64
    with pytest.raises(UnexpectedExtractionCall):
        asyncio.run(ex(None, "a second call in the same checkpoint"))
    assert ex.describe()["quality"] == "pipeline-only"


def test_live_extractor_requires_positive_caps():
    with pytest.raises(ValueError):
        LiveExtractor(max_calls=0, max_usd=1.0)
    with pytest.raises(ValueError):
        LiveExtractor(max_calls=1, max_usd=0)


@pytest.fixture
def fake_model(monkeypatch):
    """Stand-in for the host's call_llm_text, which ports resolves at call time."""
    seen: list[dict] = []

    async def fake(provider, prompt, max_tokens=2048, span=None, **_):
        seen.append({"prompt": prompt, "max_tokens": max_tokens})
        return '{"memories": [], "social_experiences": []}'

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake)
    return seen


def test_live_extractor_enforces_the_call_cap(fake_model):
    ex = LiveExtractor(max_calls=2, max_usd=10.0)
    for n in range(2):
        ex.arm(f"cp{n}", None)
        asyncio.run(ex(None, "short prompt", max_tokens=100))
    with pytest.raises(BudgetExceeded, match="call cap"):
        asyncio.run(ex(None, "short prompt", max_tokens=100))
    assert len(fake_model) == 2
    assert ex.budget.spent_usd > 0 and ex.budget.calls == 2


def test_live_extractor_refuses_a_call_that_could_pass_the_dollar_cap(fake_model):
    ex = LiveExtractor(max_calls=10, max_usd=0.001)
    # 8192 output tokens exceed this cap even at Haiku 5.5's lower price.
    with pytest.raises(BudgetExceeded, match="could cost"):
        asyncio.run(ex(None, "prompt", max_tokens=8192))
    assert fake_model == []


def test_extraction_service_uses_the_injected_llm_text_fn():
    seen: list[dict] = []

    async def fake(provider, prompt, max_tokens=2048, span=None, **kwargs):
        seen.append({"prompt": prompt, "max_tokens": max_tokens, **kwargs})
        return json.dumps({"memories": [{"content": "Pat works at Example Co"}], "social_experiences": []})

    svc = ExtractionService(MagicMock(), llm_text_fn=fake)
    out = asyncio.run(svc._call_haiku("[user (Pat)]: I work at Example Co", "(none)", "(no existing memories)"))
    assert [m["content"] for m in out["memories"]] == ["Pat works at Example Co"]
    assert seen[0]["max_tokens"] == 11000
    assert seen[0]["thinking_mode"] == "disabled"
    assert "I work at Example Co" in seen[0]["prompt"]


def test_extraction_service_default_reads_call_llm_text_at_call_time(monkeypatch):
    """The default helper is looked up per call and missing completion fails the run."""
    calls: list[str] = []

    async def patched(provider, prompt, max_tokens=2048, span=None, **kwargs):
        calls.append(prompt)
        return None

    svc = ExtractionService(MagicMock())
    monkeypatch.setattr("kai_memory.extraction.memory_extraction.get_settings", lambda: object())
    monkeypatch.setattr(
        "kai_memory.extraction.memory_extraction.get_provider_for_model", lambda settings, model: object()
    )
    monkeypatch.setattr("kai_memory.extraction.memory_extraction.start_generation_span", lambda *args: None)
    monkeypatch.setattr("kai_memory.extraction.memory_extraction.call_llm_text", patched)
    with pytest.raises(RuntimeError, match="memory extraction returned no completion"):
        asyncio.run(svc._call_haiku("conv", "(none)", "(none)"))
    assert len(calls) == 1


def test_a_live_call_that_raises_stays_charged_at_its_worst_case(monkeypatch):
    """A provider call can consume tokens and then fail (timeout, dropped stream); the
    dollar cap must not forget it, or repeated failures could run past --max-usd."""

    async def failing(provider, prompt, max_tokens=2048, span=None, **_):
        raise TimeoutError("stream dropped")

    monkeypatch.setattr("kai_memory.ports.call_llm_text", failing)
    ex = LiveExtractor(max_calls=10, max_usd=10.0)
    ex.arm("cp0", None)
    with pytest.raises(TimeoutError):
        asyncio.run(ex(None, "prompt", max_tokens=100))
    worst_case = ex.budget.spent_usd
    assert worst_case > 0 and ex.budget.calls == 1


def test_a_live_call_that_returns_is_settled_to_its_estimate(fake_model):
    ex = LiveExtractor(max_calls=10, max_usd=10.0)
    ex.arm("cp0", None)
    asyncio.run(ex(None, "prompt", max_tokens=8000))
    # Settled to prompt + the short response, far below the 8000-token worst case.
    assert 0 < ex.budget.spent_usd < ex.budget._usd(len("prompt"), 8000 * 4) / 10


def test_haiku55_budget_covers_costlier_fallback_and_new_tokenizer():
    from memory_evaluation.eval.extractors import CallBudget, LIVE_MODEL

    budget = CallBudget.for_model(LIVE_MODEL, max_calls=1, max_usd=1)
    # Both prompt tiers reserve at the higher Haiku 4.5 fallback rate.
    assert budget._usd(300_000, 0) == pytest.approx(0.1)
    assert budget._usd(300_003, 3_000) == pytest.approx(0.100001 + 0.005)
    assert budget.reserve(0, 1000) == pytest.approx(0.005)
    custom = CallBudget.for_model(LIVE_MODEL, max_calls=1, max_usd=1, usd_per_mtok=(3, 15))
    assert custom.reserve(0, 1000) == pytest.approx(0.015)
    legacy = CallBudget.for_model("haiku-4.5", max_calls=1, max_usd=1)
    assert legacy._usd(400, 400) == pytest.approx(0.0006)


def test_live_extractor_forwards_explicit_thinking_mode(monkeypatch):
    seen = []

    async def fake(provider, prompt, **kwargs):
        seen.append(kwargs)
        return "{}"

    monkeypatch.setattr("kai_memory.ports.call_llm_text", fake)
    ex = LiveExtractor(max_calls=1, max_usd=1)
    asyncio.run(ex(None, "prompt", max_tokens=11000, thinking_mode="disabled", task_name="memory/l2-extraction"))
    assert seen == [
        {"max_tokens": 11000, "span": None, "thinking_mode": "disabled", "task_name": "memory/l2-extraction"}
    ]
