"""Explicit PostgreSQL acceptance; never skip when requested."""

import hashlib
import os
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from memory_evaluation.backends.sdk import SDKBackend
from memory_evaluation.contracts import RunConfig
from memory_evaluation.eval import DisposableDB, build_scoreboard, load_fixture_set, run_write_readback
from memory_evaluation.eval.__main__ import fixture_end
from memory_evaluation.eval.standalone import HISTORY, recorded_host
from memory_evaluation.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
async def database():
    url = os.environ.get("EVAL_DATABASE_URL")
    if not url:
        pytest.fail("EVAL_DATABASE_URL must identify a dedicated empty local pgvector database")
    DisposableDB(url, production_url=None)  # Validate localhost before any connection.
    admin = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://"))
    name = "evaluation_" + uuid.uuid4().hex
    await admin.execute(f'CREATE DATABASE "{name}"')
    parts = urlsplit(url)
    try:
        yield DisposableDB(urlunsplit(parts._replace(path="/" + name)), production_url=None)
    finally:
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


async def test_preserved_lifecycle_replay(database):
    fixtures = load_fixture_set(ROOT / "tests/fixtures/memory_lifecycle/v1")
    db = database
    with recorded_host(max(map(fixture_end, fixtures.fixtures))):
        await db.create_schema(HISTORY)
        raw = await run_write_readback(fixtures, db, fixture_clock=fixture_end)
        board = build_scoreboard(raw, fixtures)
    assert board["aggregate"]["n_invalid"] == 0
    assert len(raw["results"]) == 12
    assert all(row["status"] == "ok" for row in raw["results"])
    assert db.mutation_check()["residual_rows"] == 0
    assert not db.mutation_check()["changed_tables"]


async def test_sdk_uses_same_neutral_runner(database):
    fixtures = load_fixture_set(ROOT / "tests/fixtures/memory_lifecycle/v1")
    fixtures.fixtures[:] = fixtures.fixtures[:1]
    db = database
    now = fixture_end(fixtures.fixtures[0])
    with recorded_host(now):
        await db.create_schema(HISTORY)
    # Caller pins the actual wheel, never a synthetic identity.
    wheel = Path(os.environ["EVAL_SDK_WHEEL"])
    backend = SDKBackend(db, artifact_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())

    async def answer(request):
        return request.memory_context or "I don't know."

    report = await evaluate(
        fixtures,
        backend,
        answer_fn=answer,
        config=RunConfig("integration", now.isoformat(), 42, {"model": "recorded-context", "prompt_version": "1"}),
    )
    assert report["rows"]
    assert report["denominators"]["invalid"] == 0
    assert report["denominators"]["unrun"] == 0
    assert report["denominators"]["valid"] == len(report["rows"])
    assert db.mutation_check()["residual_rows"] == 0
    assert not db.mutation_check()["changed_tables"]
