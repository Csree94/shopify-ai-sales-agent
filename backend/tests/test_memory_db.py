"""Phase M1 tests: Neon PostgreSQL conversation memory — ALL OFFLINE.

No test touches a database, a network, or real credentials: Neon settings
come from monkeypatched env dicts, psycopg.connect is stubbed, PostgresSaver
methods are patched at class level, and the unconfigured path is asserted to
be the exact unchanged InMemorySaver behavior.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

import app.memory_db as memory_db
from app.config import ConfigurationError, load_neon_settings

# Synthetic DSN for tests only — no real host, user, or secret. Must avoid
# the word 'placeholder' so the real placeholder-rejection logic is exercised.
VALID_DSN = "postgres://checkpoint-user:checkpoint-secret@ep-nothing-000000.neon.example/db"


# -- config.py: NeonSettings validation + secret hygiene ----------------------


def test_load_neon_settings_accepts_valid_dsn(monkeypatch):
    settings = load_neon_settings({"NEON_DATABASE_URL": VALID_DSN})
    assert settings is not None
    # The DSN is available to the factory but never echoed.
    assert settings.dsn == VALID_DSN


def test_load_neon_settings_unset_returns_none(monkeypatch):
    assert load_neon_settings({}) is None
    assert load_neon_settings({"NEON_DATABASE_URL": ""}) is None


def test_load_neon_settings_rejects_placeholder(monkeypatch):
    assert (
        load_neon_settings({"NEON_DATABASE_URL": "placeholder_get_from_neon_dashboard"})
        is None
    )


def test_load_neon_settings_rejects_non_postgres_scheme():
    with pytest.raises(ConfigurationError):
        load_neon_settings({"NEON_DATABASE_URL": "mysql://user:pass@host/db"})


def test_neon_settings_repr_never_exposes_dsn():
    settings = load_neon_settings({"NEON_DATABASE_URL": VALID_DSN})
    assert settings is not None
    assert VALID_DSN not in repr(settings)
    assert VALID_DSN not in str(settings)
    assert "<redacted>" in repr(settings)


# -- memory_db.py: factory + lifecycle (all stubbed psycopg) -------------------


class _StubConnection:
    """Stands in for psycopg.connect(); records kwargs, does no database I/O."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_get_checkpointer_unconfigured_returns_inmemory(monkeypatch):
    """Without NEON_DATABASE_URL the exact existing behavior is preserved."""
    monkeypatch.setattr(memory_db, "load_neon_settings", lambda: None)
    checkpointer = memory_db.get_checkpointer()
    assert isinstance(checkpointer, InMemorySaver)
    assert isinstance(checkpointer, BaseCheckpointSaver)


def test_unconfigured_first_requests_race_and_get_one_saver(monkeypatch):
    """Approved Phase M1 follow-up: concurrent first requests in unconfigured
    (in-memory) mode all receive the SAME cached saver instance — decided
    under the same lock as the Postgres path, so state never diverges.
    """
    import threading

    monkeypatch.setattr(memory_db, "load_neon_settings", lambda: None)
    memory_db.reset_postgres_singleton()
    try:
        results: list = []

        def worker():
            results.append(memory_db.get_checkpointer())

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 6
        first = results[0]
        assert all(result is first for result in results), (
            "concurrent first requests built divergent saver instances"
        )
    finally:
        memory_db.reset_postgres_singleton()


def test_get_checkpointer_configured_builds_postgres_saver(monkeypatch):
    """Configured mode: one psycopg connection with the source-verified
    options (autocommit, prepare threshold 0, dict rows), a PostgresSaver
    over it, and exactly one idempotent setup() run."""
    connect_calls: list[dict] = []
    setup_calls: list[int] = []

    def fake_connect(dsn, **kwargs):
        connect_calls.append({"dsn": dsn, **kwargs})
        return _StubConnection()

    monkeypatch.setattr(memory_db.psycopg, "connect", fake_connect)
    monkeypatch.setattr(
        memory_db, "load_neon_settings", lambda: _FakeSettings(VALID_DSN)
    )
    monkeypatch.setattr(PostgresSaver, "setup", lambda self: setup_calls.append(1))
    memory_db.reset_postgres_singleton()
    try:
        checkpointer = memory_db.get_checkpointer()
        assert isinstance(checkpointer, PostgresSaver)
        assert len(connect_calls) == 1  # ONE long-lived connection, cached
        assert connect_calls[0]["dsn"] == VALID_DSN
        assert connect_calls[0]["autocommit"] is True
        assert connect_calls[0]["prepare_threshold"] == 0
        assert setup_calls == [1]  # idempotent schema setup ran once

        # The singleton is reused across calls — never rebuilt per request.
        again = memory_db.get_checkpointer()
        assert again is checkpointer
        assert len(connect_calls) == 1
        assert setup_calls == [1]
    finally:
        monkeypatch.setattr(PostgresSaver, "setup", PostgresSaver.setup)
        memory_db.reset_postgres_singleton()


def test_unreachable_database_raises_memory_error_not_silent_fallback(monkeypatch):
    """A configured-but-unreachable store must NEVER silently downgrade to
    in-memory (that would invisibly disable persistence)."""
    def rejecting_connect(dsn, **kwargs):
        raise ConnectionError("cannot reach host")

    monkeypatch.setattr(memory_db.psycopg, "connect", rejecting_connect)
    monkeypatch.setattr(
        memory_db, "load_neon_settings", lambda: _FakeSettings(VALID_DSN)
    )
    memory_db.reset_postgres_singleton()
    try:
        with pytest.raises(memory_db.MemoryDependencyError):
            memory_db.get_checkpointer()
        # And the failure detail never leaks the DSN.
        with pytest.raises(memory_db.MemoryDependencyError) as excinfo:
            memory_db.get_checkpointer()
        assert VALID_DSN not in str(excinfo.value)
    finally:
        memory_db.reset_postgres_singleton()


class _FakeSettings:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
