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
