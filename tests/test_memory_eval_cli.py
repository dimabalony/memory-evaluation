"""Portable recorded CLI contract; no database or paid provider is opened."""

import asyncio
import random
from types import SimpleNamespace

import pytest

from memory_evaluation.eval.fixtures import select_fixture_subset
from memory_evaluation.eval.__main__ import build_parser, run
from memory_evaluation.eval.disposable_db import UnsafeDatabaseError


@pytest.mark.parametrize("n,seed", [(None, None), (2, None), (2, 7), (4, None), (9, None), (0, None)])
def test_subset_preserves_existing_runner_selection(n, seed):
    examples = [{"id": str(i)} for i in range(4)]
    if n is None or n >= len(examples):
        expected = examples
    else:
        chosen = random.Random(seed if seed is not None else 0).sample(list(enumerate(examples)), n)
        expected = [row for _, row in sorted(chosen)]
    assert select_fixture_subset(examples, n, seed) == expected


def test_cli_refuses_configured_database_before_connecting(monkeypatch):
    from memory_evaluation.eval import __main__ as cli

    monkeypatch.setattr(cli, "load_fixture_set", lambda *a, **k: SimpleNamespace(fixtures=[]))
    monkeypatch.setenv("DATABASE_URL", "postgresql://other:pw@localhost:5433/eval")
    args = build_parser().parse_args(["--fixtures", "unused", "--db-url", "postgresql://eval:eval@127.0.0.1:5433/eval"])
    with pytest.raises(UnsafeDatabaseError, match="configured application"):
        asyncio.run(run(args))


@pytest.mark.parametrize("rollback_fails", [False, True])
async def test_recorded_host_explicitly_recovers_failed_sessions(rollback_fails):
    from datetime import UTC, datetime
    from unittest.mock import AsyncMock
    from memory_evaluation.eval.standalone import recorded_host

    session = SimpleNamespace(
        rollback=AsyncMock(side_effect=RuntimeError("synthetic failed rollback") if rollback_fails else None),
        invalidate=AsyncMock(),
    )
    with recorded_host(datetime(2026, 1, 1, tzinfo=UTC)) as bindings:
        await bindings["recover_session"](session)
    session.rollback.assert_awaited_once()
    assert session.invalidate.await_count == int(rollback_fails)


@pytest.mark.parametrize("subset", [0, -1])
@pytest.mark.parametrize("validate_only", [False, True])
def test_cli_rejects_nonpositive_subset_before_loading_or_database(monkeypatch, subset, validate_only):
    from unittest.mock import Mock
    from memory_evaluation.eval import __main__ as cli

    loader = Mock(side_effect=AssertionError("fixtures must not be loaded"))
    database = Mock(side_effect=AssertionError("database must not be constructed"))
    monkeypatch.setattr(cli, "load_fixture_set", loader)
    monkeypatch.setattr(cli, "DisposableDB", database)
    arguments = ["--fixtures", "unused", "--subset", str(subset)]
    if validate_only:
        arguments.append("--validate-only")
    with pytest.raises(ValueError, match=r"^--subset must be at least 1$"):
        asyncio.run(run(build_parser().parse_args(arguments)))
    loader.assert_not_called()
    database.assert_not_called()


def test_cli_valid_subset_keeps_seeded_selection(monkeypatch, capsys):
    import json
    from pathlib import Path
    from unittest.mock import Mock
    from memory_evaluation.eval import __main__ as cli

    database = Mock(side_effect=AssertionError("validate-only must not construct database"))
    monkeypatch.setattr(cli, "DisposableDB", database)
    fixtures = Path(__file__).parent / "fixtures/memory_lifecycle/v1"
    args = build_parser().parse_args(["--fixtures", str(fixtures), "--subset", "1", "--seed", "7", "--validate-only"])
    assert asyncio.run(run(args)) == 0
    assert json.loads(capsys.readouterr().out)["ids"] == ["lc-06-choir-rehearsal"]
    database.assert_not_called()


@pytest.mark.asyncio
async def test_cli_invalid_readback_returns_failure_without_invalidating_quality(monkeypatch):
    from contextlib import nullcontext
    from unittest.mock import AsyncMock
    from memory_evaluation.eval import __main__ as cli

    fixture = {"start_at": "2026-01-01T00:00:00+00:00", "messages": [{"minute": 0}]}
    monkeypatch.setattr(cli, "load_fixture_set", lambda *a, **k: SimpleNamespace(fixtures=[fixture]))
    monkeypatch.setattr(cli, "DisposableDB", lambda *a, **k: object())
    monkeypatch.setattr(cli, "recorded_host", lambda *a: nullcontext())
    monkeypatch.setattr(cli, "run_write_readback", AsyncMock(return_value={}))
    monkeypatch.setattr(
        cli,
        "build_scoreboard",
        lambda *a, **k: {
            "aggregate": {"n_invalid": 0, "n_invalid_readback": 1},
            "mutation_check": {"residual_rows": 0, "changed_tables": []},
        },
    )
    args = build_parser().parse_args(["--fixtures", "unused", "--db-url", "unused"])
    assert await cli.run(args) == 1
