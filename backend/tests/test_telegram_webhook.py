"""Tests for the Telegram webhook route (Stage 2).

All flows are offline: the Telegram client is a fake, the agent is patched
at the ``app.telegram.service.run_chat`` seam, and the background task is
captured by patching ``asyncio.create_task`` and drained with ``asyncio.run``
(fully deterministic; no cross-loop task juggling). The checkpointer stays
in-memory via the conftest fixtures.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

SECRET = "test-webhook-secret"
TOKEN_PART = "123456"


def make_update(
    *,
    update_id: int = 5001,
    text: str = "what hats do you have",
    chat_id: int = 555_001,
    user_id: int = 42,
    chat_type: str = "private",
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 10,
            "from": {"id": user_id, "first_name": "Ada", "username": "ada_shop"},
            "chat": {"id": chat_id, "type": chat_type},
            "date": 1_790_000_000,
            "text": text,
        },
    }


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))

    async def aclose(self) -> None:
        return None


def drain(harness: dict[str, Any]) -> None:
    """Await the captured background coroutine (if any) deterministically."""
    coro = harness["captured"].pop("coro", None)
    if coro is not None:
        asyncio.run(coro)


def post_update(
    client: TestClient,
    payload: dict[str, Any],
    *,
    secret: str | None = SECRET,
) -> Any:
    headers: dict[str, str] = {}
    if secret is not None:
        headers["X-Telegram-Bot-Api-Secret-Token"] = secret
    return client.post("/api/telegram/webhook", json=payload, headers=headers)


@pytest.fixture()
def telegram_env(monkeypatch: pytest.MonkeyPatch):
    """Enable Telegram with fake credentials (conftest pins them empty first; the
    later per-fixture monkeypatch set wins within this fixture's scope)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", f"{TOKEN_PART}:FAKE-TEST-TOKEN")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", SECRET)


@pytest.fixture()
def harness(telegram_env, monkeypatch: pytest.MonkeyPatch):
    """TestClient with the lifespan running + captured background coroutine.

    The route schedules processing via ``asyncio.create_task``; the patch
    captures the coroutine so each test can await it (or assert it was never
    scheduled) without crossing event loops.
    """
    captured: dict[str, Any] = {}

    def fake_create_task(coro: Any, *args: Any, **kwargs: Any) -> Any:
        """Capture ONLY the webhook's turn coroutine (name ``_process``);
        hand everything else (LangGraph internals, TestClient machinery) to
        the REAL ``create_task`` — with the full signature: LangGraph calls
        ``create_task(coro, context=...)``.

        The captured coro is NOT scheduled; a real pending no-op task with
        the same lifecycle takes its place in the tracker so shutdown's
        real ``asyncio.gather`` always receives proper Futures.
        """
        if getattr(coro, "__name__", "") == "_process":
            captured["coro"] = coro

            async def _noop_pending() -> None:
                await asyncio.sleep(1.0)  # stays pending through the request

            return asyncio.get_running_loop().create_task(_noop_pending())
        return asyncio.get_running_loop().create_task(coro, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    run_chat_calls: list[dict[str, Any]] = []

    def fake_run_chat(message: str, session_id: str) -> Any:
        run_chat_calls.append({"message": message, "session_id": session_id})
        reply = f"reply:{message}"
        return MagicMock(
            reply=reply, session_id=session_id, tools_used=[], model_used="gemini", fallback_used=False
        )

    monkeypatch.setattr("app.telegram.service.run_chat", fake_run_chat)

    app = create_app()
    fake_telegram = FakeTelegram()
    with TestClient(app) as test_client:
        test_client.app.state.telegram_client = fake_telegram
        harness_data: dict[str, Any] = {
            "client": test_client,
            "telegram": fake_telegram,
            "captured": captured,
            "run_chat_calls": run_chat_calls,
        }
        yield harness_data

    leftover = captured.pop("coro", None)
    if leftover is not None:
        asyncio.run(leftover)


def test_disabled_integration_answers_503():
    """No telegram_client on app.state → controlled 503 (never a crash)."""
    with TestClient(create_app()) as test_client:
        response = post_update(test_client, make_update())
        assert response.status_code == 503
        assert "TELEGRAM_BOT_TOKEN" not in response.text


def test_missing_webhook_secret_header_rejected(telegram_env):
    """Secret configured but header absent → 401 (never silently skipped)."""
    with TestClient(create_app()) as test_client:
        test_client.app.state.telegram_client = FakeTelegram()
        response = post_update(test_client, make_update(), secret=None)
        assert response.status_code == 401


def test_wrong_webhook_secret_rejected(telegram_env):
    with TestClient(create_app()) as test_client:
        test_client.app.state.telegram_client = FakeTelegram()
        response = post_update(test_client, make_update(), secret="wrong-value")
        assert response.status_code == 401


def test_correct_webhook_secret_accepted(harness: dict[str, Any]):
    response = post_update(harness["client"], make_update())
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    drain(harness)


def test_malformed_body_acked(harness: dict[str, Any]):
    response = harness["client"].post(
        "/api/telegram/webhook",
        content=b"not-json{{",
        headers={
            "Content-Type": "application/json",
            "X-Telegram-Bot-Api-Secret-Token": SECRET,
        },
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert "coro" not in harness["captured"]


def test_unparsable_update_acked(harness: dict[str, Any]):
    response = post_update(harness["client"], {"foo": "bar"})
    assert response.status_code == 200
    assert "coro" not in harness["captured"]


def test_non_text_update_acked_and_skipped(harness: dict[str, Any]):
    update = make_update(update_id=6002)
    del update["message"]["text"]
    update["message"]["photo"] = [{"file_id": "abc"}]
    response = post_update(harness["client"], update)
    assert response.status_code == 200
    assert "coro" not in harness["captured"]


def test_edited_message_is_skipped(harness: dict[str, Any]):
    payload = make_update(update_id=6003)
    payload["edited_message"] = payload.pop("message")
    response = post_update(harness["client"], payload)
    assert response.status_code == 200
    assert "coro" not in harness["captured"]


def test_group_message_is_skipped(harness: dict[str, Any]):
    response = post_update(harness["client"], make_update(update_id=6004, chat_type="group"))
    assert response.status_code == 200
    assert "coro" not in harness["captured"]


def test_blank_text_message_is_skipped(harness: dict[str, Any]):
    response = post_update(harness["client"], make_update(update_id=6005, text="   "))
    assert response.status_code == 200
    assert "coro" not in harness["captured"]


def test_valid_message_reaches_agent_and_chat(harness: dict[str, Any]):
    """One accepted update → exactly one run_chat + reply to the chat."""
    response = post_update(harness["client"], make_update())
    drain(harness)

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    calls = harness["run_chat_calls"]
    assert len(calls) == 1
    assert calls[0]["message"] == "what hats do you have"
    assert calls[0]["session_id"] == "tg:42"
    sent = harness["telegram"].sent
    assert len(sent) == 1
    sent_chat_id, sent_text = sent[0]
    assert sent_chat_id == 555_001
    assert sent_text == "reply:what hats do you have"


def test_pending_task_is_tracked_on_app_state(harness: dict[str, Any]):
    """A scheduled turn sits on ``app.state.telegram_background_tasks``
    (strong reference). Discard-on-completion is unit-tested in
    ``test_telegram_lifecycle.py`` (``_make_done_callback``)."""
    post_update(harness["client"], make_update())
    tracker = harness["client"].app.state.telegram_background_tasks
    assert len(tracker) == 1
    drain(harness)


def test_secrets_and_token_never_leak_into_ack(harness: dict[str, Any]):
    response = post_update(harness["client"], make_update())
    drain(harness)
    body = response.text + str(harness["run_chat_calls"])
    assert "FAKE-TEST-TOKEN" not in body
    assert SECRET not in body
