"""A disposable database boundary that proves it left nothing behind (DB-840).

Every write of an eval run happens on ONE connection inside an outer transaction
that is always rolled back. The run's session joins that transaction in savepoint
mode, so the extraction code's own ``begin_nested``/``rollback``/``commit`` stay
inside it. Four guards keep a run off any database that matters:

1. URL guard: localhost only; ``*.rds.amazonaws.com`` and the configured
   application database URL are refused by name.
2. Emptiness guard: ``conversation_messages`` and ``memory_traces`` must have no
   rows before seeding, so the run can never sit on real memory.
3. Stray-session tripwire: scoped ``SessionLocal`` and ``session_scope`` adapters
   raise without loading or mutating host modules, so code
   that opens its own committing session fails loudly instead of writing.
4. Mutation check: every table's row count and content checksum is read on a
   fresh connection before the outer transaction and again after its rollback.
   ``mutation_check.residual_rows`` is the summed row-count difference; any table
   whose checksum moved is listed in ``changed_tables``.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import Table, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from kai_memory import ports
from kai_memory.storage.disposable import (
    LOCAL_HOSTS as LOCAL_HOSTS,
    UnsafeDatabaseError as UnsafeDatabaseError,
    check_url_is_local as check_url_is_local,
)

# Tables that must be empty before a run seeds anything.
EMPTY_BEFORE_SEEDING = ("conversation_messages", "memory_traces")


class StraySessionError(RuntimeError):
    """Code under evaluation opened its own session instead of the eval's."""


def _refusing_session_factory(*args: Any, **kwargs: Any) -> Any:
    raise StraySessionError(
        "session_scope()/SessionLocal was called during a memory eval run; that session would commit "
        "outside the disposable transaction"
    )


def _async_url(url: str) -> str:
    return url.replace("postgresql://", "postgresql+asyncpg://", 1) if url.startswith("postgresql://") else url


class DisposableDB:
    """A local database the eval may write to, inside a transaction it always rolls back.

    ``production_url`` defaults to the configured application ``DATABASE_URL``;
    pass ``None`` only when there is none to protect.
    """

    _UNSET: Any = object()

    def __init__(self, url: str, *, production_url: str | None = _UNSET) -> None:
        if production_url is DisposableDB._UNSET:
            production_url = ports.get_settings().database_url or None
        check_url_is_local(url, production_url=production_url)
        self.url = _async_url(url)
        self.mutation_checks: list[dict[str, Any]] = []

    def _engine(self) -> AsyncEngine:
        return create_async_engine(self.url, poolclass=NullPool)

    async def create_schema(self, history_table: Table) -> None:
        """Bootstrap fresh storage with the evaluation host's explicit history table."""
        from memory_evaluation.eval._models import metadata
        from kai_memory.storage import bootstrap_disposable

        engine = self._engine()
        try:
            await bootstrap_disposable(engine, schema=metadata(history_table))
        finally:
            await engine.dispose()

    async def _fingerprint(self) -> dict[str, tuple[int, str]]:
        engine = self._engine()
        try:
            async with engine.connect() as conn:
                tables = (
                    await conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"))
                ).scalars()
                out: dict[str, tuple[int, str]] = {}
                for table in list(tables):
                    quoted = '"' + table.replace('"', '""') + '"'
                    row = (
                        await conn.execute(
                            text(
                                f"SELECT count(*), coalesce(string_agg(t::text, '|' ORDER BY t::text), '') "
                                f"FROM {quoted} AS t"
                            )
                        )
                    ).one()
                    out[table] = (int(row[0]), hashlib.sha256(str(row[1]).encode("utf-8")).hexdigest())
                return out
        finally:
            await engine.dispose()

    async def _require_empty(self) -> None:
        """Refuse a populated database before reading any of it: the fingerprint below
        aggregates every row of every table, which on a restored real dump is exactly
        the read the guard exists to prevent."""
        engine = self._engine()
        try:
            async with engine.connect() as conn:
                for table in EMPTY_BEFORE_SEEDING:
                    rows = await conn.scalar(text(f"SELECT count(*) FROM {table}"))
                    if rows:
                        raise UnsafeDatabaseError(f"{table} has {rows} rows; the eval only runs on an empty database")
        finally:
            await engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session whose every write is rolled back, then record the mutation check."""
        await self._require_empty()
        before = await self._fingerprint()
        engine = self._engine()
        try:
            async with engine.connect() as conn:
                outer = await conn.begin()
                try:
                    for table in EMPTY_BEFORE_SEEDING:
                        rows = await conn.scalar(text(f"SELECT count(*) FROM {table}"))
                        if rows:
                            raise UnsafeDatabaseError(
                                f"{table} has {rows} rows; the eval only runs on an empty database"
                            )
                    session = AsyncSession(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
                    with ports.scoped(
                        {"SessionLocal": _refusing_session_factory, "session_scope": _refusing_session_factory},
                        inherit=True,
                    ):
                        try:
                            yield session
                        finally:
                            await session.close()
                finally:
                    await outer.rollback()
        finally:
            await engine.dispose()
            # Recorded even when the run raised: a failed run must still prove it left nothing.
            self.mutation_checks.append(_compare(before, await self._fingerprint()))

    def mutation_check(self) -> dict[str, Any]:
        """Every run's check, summed: ``residual_rows`` must be 0 and ``changed_tables`` empty."""
        changed = sorted({t for check in self.mutation_checks for t in check["changed_tables"]})
        return {
            "runs_checked": len(self.mutation_checks),
            "tables_checked": max((c["tables_checked"] for c in self.mutation_checks), default=0),
            "residual_rows": sum(c["residual_rows"] for c in self.mutation_checks),
            "changed_tables": changed,
            "method": "per-table count + content sha256 on a fresh connection, before vs after rollback",
        }


def _compare(before: dict[str, tuple[int, str]], after: dict[str, tuple[int, str]]) -> dict[str, Any]:
    tables = sorted(set(before) | set(after))
    residual = sum(abs(after.get(t, (0, ""))[0] - before.get(t, (0, ""))[0]) for t in tables)
    changed = [t for t in tables if before.get(t) != after.get(t)]
    return {"tables_checked": len(tables), "residual_rows": residual, "changed_tables": changed}
