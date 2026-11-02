"""Fixture selection must not change another replay's temporal reference frame."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from kai_memory import ports
from memory_evaluation.eval import runner

HOST_NOW = datetime(2030, 1, 1, tzinfo=UTC)
FIXTURES = {
    "early": {"id": "early", "end": datetime(2026, 3, 2, 10, 8, tzinfo=UTC)},
    "late": {"id": "late", "end": datetime(2026, 6, 20, 17, 47, tzinfo=UTC)},
}


def fixture_set(ids):
    return SimpleNamespace(fixtures=[FIXTURES[key] for key in ids], path="synthetic", sha256="synthetic")


@pytest.mark.parametrize("ids", [("early",), ("early", "late"), ("late", "early")])
async def test_each_fixture_keeps_its_clock_under_subset_and_reordering(monkeypatch, ids):
    observed = {}
    settings = object()

    async def replay(fixture, _db, _extractor):
        assert ports.get_settings() is settings
        observed[fixture["id"]] = ports.clock()
        return {"id": fixture["id"]}

    monkeypatch.setattr(runner, "_run_fixture", replay)
    db = SimpleNamespace(mutation_check=lambda: {})
    with ports.scoped({"clock": lambda: HOST_NOW, "get_settings": lambda: settings}):
        await runner.run_write_readback(fixture_set(ids), db, fixture_clock=lambda fixture: fixture["end"])
        assert ports.clock() == HOST_NOW
    assert observed == {key: FIXTURES[key]["end"] for key in ids}


async def test_legacy_runner_without_fixture_clock_keeps_host_clock(monkeypatch):
    observed = []

    async def replay(fixture, _db, _extractor):
        observed.append(ports.clock())
        return {"id": fixture["id"]}

    monkeypatch.setattr(runner, "_run_fixture", replay)
    db = SimpleNamespace(mutation_check=lambda: {})
    with ports.scoped({"clock": lambda: HOST_NOW}):
        await runner.run_write_readback(fixture_set(("early", "late")), db)
        assert ports.clock() == HOST_NOW
    assert observed == [HOST_NOW, HOST_NOW]


async def test_replay_exception_restores_the_callers_clock(monkeypatch):
    async def replay(_fixture, _db, _extractor):
        assert ports.clock() == FIXTURES["early"]["end"]
        raise RuntimeError("synthetic replay failure")

    monkeypatch.setattr(runner, "_run_fixture", replay)
    db = SimpleNamespace(mutation_check=lambda: {})
    with ports.scoped({"clock": lambda: HOST_NOW}):
        with pytest.raises(RuntimeError, match="synthetic replay failure"):
            await runner.run_write_readback(fixture_set(("early",)), db, fixture_clock=lambda fixture: fixture["end"])
        assert ports.clock() == HOST_NOW
