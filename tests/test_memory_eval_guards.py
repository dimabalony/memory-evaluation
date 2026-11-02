"""DisposableDB URL guard: localhost only, never RDS, never the app's own DATABASE_URL (DB-840)."""

from __future__ import annotations

import pytest

from kai_memory import ports
from memory_evaluation.eval import DisposableDB
from memory_evaluation.eval.disposable_db import UnsafeDatabaseError, check_url_is_local

LOCAL = "postgresql+asyncpg://eval:eval@localhost:5433/memory_eval"


@pytest.mark.parametrize(
    "url",
    [
        LOCAL,
        "postgresql+asyncpg://eval:eval@127.0.0.1:5433/memory_eval",
        "postgresql://eval:eval@[::1]:5433/memory_eval",
    ],
)
def test_local_urls_are_accepted(url):
    check_url_is_local(url, production_url=None)


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("postgresql+asyncpg://u:p@kai.abc123.ap-southeast-1.rds.amazonaws.com:5432/kai", "RDS host"),
        ("postgresql+asyncpg://u:p@db.internal:5432/kai", "non-local host"),
        ("postgresql+asyncpg://u:p@10.0.0.5:5432/kai", "non-local host"),
        ("postgresql+asyncpg://u:p@/kai", "non-local host"),
        ("not a url", "unparseable"),
        ("postgresql+asyncpg://u:p@localhost/kai?host=prod.example.com", "connection-target parameters"),
        ("postgresql+asyncpg://u:p@localhost/kai?port=6543", "connection-target parameters"),
        ("postgresql://u:p@127.0.0.1:5433/kai?HostAddr=10.0.0.5", "connection-target parameters"),
    ],
)
def test_non_local_urls_are_refused(url, reason):
    with pytest.raises(UnsafeDatabaseError, match=reason):
        check_url_is_local(url, production_url=None)


def test_the_configured_application_database_is_refused_even_on_localhost(monkeypatch):
    monkeypatch.setattr(ports.get_settings(), "database_url", "postgresql://kai:pw@localhost:5433/memory_eval")
    with pytest.raises(UnsafeDatabaseError, match="configured application DATABASE_URL"):
        DisposableDB(LOCAL)
    # A different database on the same local server is fine.
    DisposableDB("postgresql+asyncpg://eval:eval@localhost:5433/another_db")


@pytest.mark.parametrize(
    ("url", "production"),
    [
        ("postgresql+asyncpg://e:e@127.0.0.1:5432/kai", "postgresql://kai:pw@localhost/kai"),
        ("postgresql+asyncpg://e:e@localhost/kai", "postgresql://kai:pw@[::1]:5432/kai"),
        ("postgresql+asyncpg://e:e@LOCALHOST:5433/kai", "postgresql://kai:pw@127.0.0.1:5433/kai"),
    ],
)
def test_the_application_database_is_refused_under_any_local_spelling(url, production):
    with pytest.raises(UnsafeDatabaseError, match="configured application DATABASE_URL"):
        check_url_is_local(url, production_url=production)


def test_async_driver_is_filled_in():
    assert DisposableDB("postgresql://e:e@localhost:5433/x", production_url=None).url.startswith(
        "postgresql+asyncpg://"
    )


@pytest.mark.parametrize("driver,parameter", [("asyncpg", "database"), ("psycopg2", "dbname")])
def test_query_database_override_is_refused_before_connecting(driver, parameter):
    from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
    from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
    from sqlalchemy.engine import make_url

    url = f"postgresql+{driver}://eval:eval@localhost:5433/memory_eval?{parameter}=application"
    dialect = PGDialect_asyncpg() if driver == "asyncpg" else PGDialect_psycopg2()
    _, kwargs = dialect.create_connect_args(make_url(url))
    # Real dialect translation proves the query defeats a path-only comparison;
    # no engine or network connection is created.
    assert kwargs[parameter] == "application"
    with pytest.raises(UnsafeDatabaseError, match="connection-target parameters"):
        DisposableDB(url, production_url="postgresql://eval:eval@localhost:5433/application")


@pytest.mark.parametrize("parameter", ["database", "dbname", "DATABASE", "DbName"])
def test_database_target_queries_are_refused_without_a_production_url(parameter):
    with pytest.raises(UnsafeDatabaseError, match="connection-target parameters"):
        DisposableDB(f"{LOCAL}?{parameter}=other", production_url=None)
