"""Recorded-host recovery keeps failure diagnostics private and best effort."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kai_memory import ports
from memory_evaluation.eval import standalone


class PrivateRollbackError(RuntimeError):
    pass


class PrivateInvalidateError(RuntimeError):
    pass


@pytest.mark.asyncio
async def test_successful_rollback_needs_no_invalidation_or_diagnostic(caplog):
    session = SimpleNamespace(rollback=AsyncMock(), invalidate=AsyncMock())
    with standalone.recorded_host(datetime(2030, 1, 1, tzinfo=UTC)):
        await ports.recover_session(session)
    session.rollback.assert_awaited_once_with()
    session.invalidate.assert_not_awaited()
    assert not caplog.records


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate_fails", [False, True])
async def test_recovery_logs_only_exception_types_and_preserves_operation_error(caplog, invalidate_fails):
    private_text = "PRIVATE_RECORD_SENTINEL SELECT hidden_payload FROM synthetic_table"
    session = SimpleNamespace(
        rollback=AsyncMock(side_effect=PrivateRollbackError(private_text)),
        invalidate=AsyncMock(side_effect=PrivateInvalidateError(private_text) if invalidate_fails else None),
    )
    original = RuntimeError("original synthetic operation failed")
    with caplog.at_level(logging.WARNING, logger=standalone.__name__):
        with pytest.raises(RuntimeError) as raised:
            with standalone.recorded_host(datetime(2030, 1, 1, tzinfo=UTC)):
                try:
                    raise original
                except RuntimeError:
                    await ports.recover_session(session)
                    raise
    assert raised.value is original
    session.rollback.assert_awaited_once_with()
    session.invalidate.assert_awaited_once_with()
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Recorded evaluation session rollback failed: PrivateRollbackError"] + (
        ["Recorded evaluation session invalidate failed: PrivateInvalidateError"] if invalidate_fails else []
    )
    assert private_text not in caplog.text
    assert all(not record.exc_info and not record.stack_info for record in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate_fails", [False, True])
async def test_logging_failure_cannot_skip_invalidation_or_replace_operation_error(caplog, invalidate_fails):
    attempts = []

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            attempts.append(record)
            raise RuntimeError("synthetic log handler failed")

    handler = BrokenHandler()
    session = SimpleNamespace(
        rollback=AsyncMock(side_effect=PrivateRollbackError("private synthetic rollback text")),
        invalidate=AsyncMock(
            side_effect=PrivateInvalidateError("private synthetic invalidation text") if invalidate_fails else None
        ),
    )
    original = RuntimeError("original synthetic operation failed")
    standalone.logger.addHandler(handler)
    try:
        with caplog.at_level(logging.WARNING, logger=standalone.__name__):
            with pytest.raises(RuntimeError) as raised:
                with standalone.recorded_host(datetime(2030, 1, 1, tzinfo=UTC)):
                    try:
                        raise original
                    except RuntimeError:
                        await ports.recover_session(session)
                        raise
    finally:
        standalone.logger.removeHandler(handler)
    assert raised.value is original
    assert len(attempts) == 1 + int(invalidate_fails)
    session.rollback.assert_awaited_once_with()
    session.invalidate.assert_awaited_once_with()
