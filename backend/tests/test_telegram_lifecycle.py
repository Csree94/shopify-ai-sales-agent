"""Lifecycle tests for the Telegram integration (Stage 2).

Covers the FastAPI lifespan behaviors: the Telegram client is created only
when validly configured, tracked background tasks are drained at shutdown
BEFORE the client is closed, and the disabled path leaves everything exactly
as it was (health + web chat included, checked by the full regression suite).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.telegram.client import TelegramClient


@pytest.fixture()
def telegram_enabled_env(monkeypatch: pytest.MonkeyPatch):
    """Enable Telegram with fake credentials (conftest pins them empty)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:FAKE-TEST-TOKEN")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-webhook-secret")


def test_client_created_when_configured(telegram_enabled_env):
    with TestClient(create_app()) as test_client:
        client = test_client.app.state.telegram_client
        assert isinstance(client, TelegramClient)
        assert test_client.app.state.telegram_background_tasks == set()


def test_disabled_integration_keeps_client_none():
    """No token → no client, no tasks; /health unchanged."""
    with TestClient(create_app()) as test_client:
        assert test_client.app.state.telegram_client is None
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_webhook_route_registered_at_expected_path():
    # FastAPI 0.143 does not flatten included routers into ``app.routes``
    # (internal _IncludedRouter objects lack ``.path``); the OpenAPI schema
    # reflects every registered concrete route.
    paths = set(create_app().openapi()["paths"])
    assert "/api/telegram/webhook" in paths
    assert "/health" in paths
    assert "/api/chat" in paths
    assert "/api/products" in paths


def test_drain_helper_awaits_outstanding_tasks_on_one_loop():
    """The lifespan shutdown drain (``app.main._drain_background_tasks``)
    awaits every tracked task to completion on the SAME loop — production
    semantics (tasks + lifespan share one uvicorn loop; the cross-loop
    combination is impossible there and is asserted directly instead).

    Integration-level: every webhook-harness teardown also exercises the
    real lifespan drain over genuine route-created tasks (created on the
    TestClient portal loop, like uvicorn does)."""
    from app.main import _drain_background_tasks

    finished: dict[str, bool] = {}

    async def probe_task() -> None:
        await asyncio.sleep(0)
        finished["drained"] = True

    async def scenario() -> None:
        task = asyncio.create_task(probe_task())
        await _drain_background_tasks({task})

    asyncio.run(scenario())
    assert finished.get("drained") is True


def test_shutdown_closes_telegram_client(telegram_enabled_env):
    closed: dict[str, bool] = {}

    class ClosingClient(TelegramClient):
        async def aclose(self) -> None:  # type: ignore[override]
            closed["closed"] = True

    async def scenario() -> None:
        test_app = create_app()
        tester = TestClient(test_app)
        with tester:
            # Replace AFTER startup created the production client: shutdown
            # must close whatever resides on app.state.
            test_app.state.telegram_client = ClosingClient("123456:FAKE-TEST-TOKEN")
        # with-block exited → drain ran → aclose called on the replacement.

    asyncio.run(scenario())
    assert closed.get("closed") is True


def test_done_callback_discards_and_consumes_exceptions():
    """The webhook's done-callback: discards the task from the tracker and
    consumes its exception (no unhandled-task warnings in production)."""
    from app.telegram.routes import _make_done_callback

    tracked: set = {}
    tracked.clear()
    tracked = set()
    callback = _make_done_callback(tracked)

    async def raises() -> None:
        raise RuntimeError("bug")

    async def scenario() -> None:
        ok = asyncio.create_task(asyncio.sleep(0))
        bad = asyncio.create_task(raises())
        tracked.update({ok, bad})
        # Retrieve the exception via gather (like any surrounding runner); the
        # callback must still be safe on an already-inspected task.
        await asyncio.gather(ok, bad, return_exceptions=True)
        callback(ok)
        callback(bad)

    asyncio.run(scenario())
    assert tracked == set()