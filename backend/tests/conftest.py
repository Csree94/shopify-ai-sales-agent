"""Shared pytest fixtures for the backend suite — TEST-ONLY isolation policy.

Approved Phase M1 follow-up: the suite must never depend on the local
environment. Before this fixture, any test that did not explicitly pin the
checkpoint store consulted the real loader (`load_neon_settings`); once a
real `NEON_DATABASE_URL` exists in the local .env, stray tests could open a
REAL Neon connection and the cached Postgres saver leaked across tests
(albumin collection order), breaking the "unconfigured → in-memory" test.
Now every test runs in in-memory mode by default; the dedicated fake-Neon
tests override this fixture explicitly with their own stubs.
"""

from __future__ import annotations

import pytest

import app.memory_db as memory_db
from app.telegram.service import reset_session_locks
from app.telegram.dedup import reset_duplicate_guard


@pytest.fixture(autouse=True)
def pin_telegram_disabled(monkeypatch):
    """Run every test with Telegram DISABLED, independent of the local env.

    Mirrors the M1 checkpointer-isolation policy: the lifespan reads the real
    environment (and the git-ignored .env files) when a TestClient enters its
    context; a stray real TELEGRAM_BOT_TOKEN would otherwise construct a real
    TelegramClient during tests. Explicit EMPTY values win over .env files
    (process environment beats dotenv here, same as the runtime override
    rule). Tests that need an enabled integration set the variables again in
    their own fixture — the later monkeypatch set wins within the scope.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "")


@pytest.fixture(autouse=True)
def reset_dedup_guard():
    """Clear the per-process duplicate-update registry around every test."""
    reset_duplicate_guard()
    try:
        yield
    finally:
        reset_duplicate_guard()


@pytest.fixture(autouse=True)
def reset_session_lock_map():
    """Clear the per-process session-lock map around every test.

    asyncio.Lock objects bind to the event loop that first WAITS on them;
    tests run per-scenario loops (asyncio.run), so a lock created under one
    loop would raise 'bound to a different event loop' under the next.
    Production never needs this: one uvicorn loop for the whole process.
    Mirrors the reset_dedup_guard / reset_postgres_singleton policy.
    """
    reset_session_locks()
    try:
        yield
    finally:
        reset_session_locks()


@pytest.fixture(autouse=True)
def pin_in_memory_checkpointer(monkeypatch):
    """Run each test independent of the real Neon environment.

    - `memory_db.load_neon_settings` returns None for every test → no real
      DSN read, no real connection, in-memory store, in every env and order.
    - The cached singleton is cleared before and after each test so no saver
      leaks across tests.
    Tests that need a store (the stubbed fake-Neon factory test) opt out by
    monkeypatching `memory_db.load_neon_settings` themselves inside the test
    (later patch wins within the same monkeypatch scope).
    """
    monkeypatch.setattr(memory_db, "load_neon_settings", lambda: None)
    memory_db.reset_postgres_singleton()
    try:
        yield
    finally:
        memory_db.reset_postgres_singleton()
