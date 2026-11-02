"""Explicit PostgreSQL acceptance; never skip when requested."""

import hashlib
import importlib.util
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from kai_memory.retrieval.memory_priming import MemoryPrimingService
from memory_evaluation.backends.sdk import SDKBackend
from memory_evaluation.contracts import RunConfig
from memory_evaluation.eval import DisposableDB, build_scoreboard, load_fixture_set, run_write_readback
from memory_evaluation.eval.__main__ import fixture_end
from memory_evaluation.eval.standalone import HISTORY, recorded_host
from memory_evaluation.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def empty_database(url):
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


@pytest.fixture
async def database():
    async with empty_database(os.environ.get("EVAL_DATABASE_URL")) as db:
        yield db


class IsolatedFixtureDB(DisposableDB):
    """Release instrument v2: each fixture receives a fresh physical index."""

    @asynccontextmanager
    async def session(self):
        async with empty_database(self.url) as child:
            await child.create_schema(HISTORY)
            try:
                async with child.session() as session:
                    yield session
            finally:
                self.mutation_checks.extend(child.mutation_checks)


async def test_preserved_lifecycle_replay(database, monkeypatch):
    fixtures = load_fixture_set(ROOT / "tests/fixtures/memory_lifecycle/v1")
    db = IsolatedFixtureDB(database.url, production_url=None)
    scores = []

    async def capture(self, *args, **kwargs):
        # prime() delegates to this public method; retain its exact outputs.
        formatted, ids, ranked = await self.prime_with_traces(*args, **kwargs)
        scores.append({
            "cue": args[0],
            "ranked": [
                {"trace_id": str(trace.id), "content": trace.content, "score": score, "components": components}
                for trace, score, components in ranked
            ],
        })
        return formatted, ids

    monkeypatch.setattr(MemoryPrimingService, "prime", capture)
    with recorded_host(max(map(fixture_end, fixtures.fixtures))):
        await db.create_schema(HISTORY)
        raw = await run_write_readback(fixtures, db, fixture_clock=fixture_end, report_root=ROOT)
        board = build_scoreboard(raw, fixtures)
    assert board["aggregate"]["n_invalid"] == 0
    assert len(raw["results"]) == 12
    assert all(row["status"] == "ok" for row in raw["results"])
    assert db.mutation_check()["residual_rows"] == 0
    assert not db.mutation_check()["changed_tables"]
    spec = importlib.util.spec_from_file_location("scoreboard_projection", ROOT / "scripts/scoreboard_projection.py")
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    baseline = json.loads((ROOT / "tests/baselines/lifecycle-v1.json").read_text())
    contract_path = ROOT / "tests/baselines/lifecycle-isolated-v2.json"
    contract = json.loads(contract_path.read_text())
    baseline_hash = hashlib.sha256((ROOT / "tests/baselines/lifecycle-v1.json").read_bytes()).hexdigest()
    assert baseline_hash == contract["historical_baseline_sha256"]
    comparison = projection.compare_isolated(board, baseline, contract)
    # Rank 1 must be measured with both semantic candidates, not accepted by a tolerance.
    durable = next(row for row in scores if row["cue"] == "What stable role or preference did Mei Tan state?")
    labels = raw["results"][6]["labels"]
    components = {row["trace_id"]: row for row in durable["ranked"]}
    preference = components[labels["stable_preference"]]
    date = components[labels["date_v1"]]
    semantic_complete = (
        preference["components"].get("semantic", 0) > date["components"].get("semantic", 0) > 0
        and preference["score"] > date["score"]
    )
    reports = Path(os.environ.get("REPORTS", ROOT / "build/proof"))
    reports.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("lifecycle-raw.json", raw),
        ("lifecycle-scoreboard.json", board),
        ("lifecycle-mutation.json", db.mutation_check()),
        ("lifecycle-retrieval-scores.json", {"instrument": contract["format"], "readbacks": scores}),
        (
            "lifecycle-parity.json",
            {
                **comparison,
                "semantic_candidates_complete": semantic_complete,
                "instrument_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                "baseline_sha256": hashlib.sha256(
                    (ROOT / "tests/baselines/lifecycle-v1.json").read_bytes()
                ).hexdigest(),
                "baseline_source_revision": baseline["source_revision"],
                "aggregate": board["aggregate"],
            },
        ),
    ):
        (reports / name).write_text(json.dumps(value, indent=2, default=str) + "\n")
    assert semantic_complete
    assert comparison["isolated_baseline_passed"]
    assert not comparison["historical_exact_parity"]
    assert comparison["historical_differences"] == contract["differences"]


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
