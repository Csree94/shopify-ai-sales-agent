"""Tests for the Telegram→agent turn pipeline (Stage 2).

The agent is patched at the ``app.telegram.service.run_chat`` seam; the
Telegram client is a fake; the checkpointer stays in-memory (conftest). Verifies: one agent turn per accepted update, the exact ``tg:{user_id}``
session id, reply delivery, one friendly error message on any agent
failure, and no exception ever escaping the pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.telegram.schemas import InboundMessage, TelegramUser
from app.telegram.service import (
    FRIENDLY_ERROR_MESSAGE,
    process_inbound,
    telegram_session_id,
)


def make_inbound(
    *,
    user_id: int = 42,
    update_id: int = 7001,
    chat_id: int = 555_001,
    text: str = "what hats do you have",
) -> InboundMessage:
    return InboundMessage(
        chat_id=chat_id,
        user=TelegramUser(id=user_id, username="ada_shop", first_name="Ada"),
        text=text,
        message_id=update_id + 10,
        update_id=update_id,
    )


class FakeTelegram:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[int, str]] = []
        self._fail = fail

    async def send_message(self, chat_id: int, text: str) -> None:
        if self._fail:
            raise RuntimeError("delivery failed")
        self.sent.append((chat_id, text))

    async def aclose(self) -> None:
        return None


def patch_run_chat(monkeypatch: pytest.MonkeyPatch, behavior: Any) -> list[dict[str, Any]]:
    """Replace the service's run_chat seam; returns the call record list."""
    calls: list[dict[str, Any]] = []

    def fake_run_chat(message: str, session_id: str) -> Any:
        calls.append({"message": message, "session_id": session_id})
        return behavior(message, session_id)

    monkeypatch.setattr("app.telegram.service.run_chat", fake_run_chat)
    return calls


def test_telegram_session_id_format():
    assert telegram_session_id(42) == "tg:42"


def test_reply_delivered_to_originating_chat(monkeypatch: pytest.MonkeyPatch):
    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(), telegram=telegram))

    assert len(calls) == 1
    assert calls[0]["session_id"] == "tg:42"
    assert telegram.sent == [(555_001, "We have hats!")]


def test_agent_failure_sends_friendly_error(monkeypatch: pytest.MonkeyPatch):
    from app.agent import ChatAgentError

    def behavior(message: str, session_id: str) -> Any:
        raise ChatAgentError("Gemini failed and no grounded answer...", fallback_attempted=True)

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(update_id=7002), telegram=telegram))

    assert telegram.sent == [(555_001, FRIENDLY_ERROR_MESSAGE)]


def test_shopify_failure_sends_friendly_error_without_details(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.shopify_client import ShopifyAuthError

    def behavior(message: str, session_id: str) -> Any:
        raise ShopifyAuthError("401 from store at https://user:pw@host/… secret=x")

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(update_id=7003), telegram=telegram))

    (chat_id, text) = telegram.sent[0]
    assert text == FRIENDLY_ERROR_MESSAGE
    assert "secret=" not in text and "https" not in text


def test_unexpected_exception_still_friendly(monkeypatch: pytest.MonkeyPatch):
    def behavior(message: str, session_id: str) -> Any:
        raise ZeroDivisionError("bug")

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(update_id=7004), telegram=telegram))
    assert telegram.sent == [(555_001, FRIENDLY_ERROR_MESSAGE)]


def test_empty_reply_sends_nothing(monkeypatch: pytest.MonkeyPatch):
    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="", session_id=session_id)

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(update_id=7005), telegram=telegram))
    assert telegram.sent == []


def test_duplicate_update_runs_no_second_turn(monkeypatch: pytest.MonkeyPatch):
    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    inbound = make_inbound(update_id=7006)

    asyncio.run(process_inbound(inbound, telegram=telegram))
    asyncio.run(process_inbound(inbound, telegram=telegram))  # redelivery

    assert len(calls) == 1
    assert len(telegram.sent) == 1


def test_ledger_failure_sends_friendly_error_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch,
):
    guard = MagicMock()
    guard.try_claim = MagicMock(side_effect=RuntimeError("ledger down"))

    async def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("agent must not run when the ledger is unavailable")

    calls = patch_run_chat(monkeypatch, forbidden)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(update_id=7007), telegram=telegram, guard=guard))

    assert calls == []
    assert telegram.sent == [(555_001, FRIENDLY_ERROR_MESSAGE)]


def test_concurrent_duplicate_deliveries_run_one_turn(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two deliveries of the same update racing → exactly one agent turn."""

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    inbound = make_inbound(update_id=7008)

    async def scenario() -> None:
        await asyncio.gather(
            process_inbound(inbound, telegram=telegram),
            process_inbound(inbound, telegram=telegram),
        )

    asyncio.run(scenario())
    assert len(calls) == 1
    assert len(telegram.sent) == 1


def test_pipeline_never_raises_even_when_delivery_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram(fail=True)
    asyncio.run(process_inbound(make_inbound(update_id=7009), telegram=telegram))
    assert telegram.sent == []  # failure swallowed, no exception escaped


def test_two_updates_from_same_user_serialize(monkeypatch: pytest.MonkeyPatch):
    """Both updates are processed (in turn) under the per-session lock."""

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply=f"reply:{message}", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()

    async def scenario() -> None:
        first = make_inbound(update_id=7010, text="first question")
        second = make_inbound(update_id=7011, text="second question")
        await asyncio.gather(
            process_inbound(first, telegram=telegram),
            process_inbound(second, telegram=telegram),
        )

    asyncio.run(scenario())
    assert len(calls) == 2
    # Lock acquisition order between the two racing tasks is not fixed —
    # assert on the SET of delivered replies, plus one per update, not on
    # delivery order.
    assert sorted(text for _, text in telegram.sent) == sorted([
        "reply:first question",
        "reply:second question",
    ])


def test_concurrent_different_updates_both_durably_claimed(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression guard (review fix 1): two DIFFERENT update ids from one
    user racing → each runs exactly one turn, and BOTH ids end up durably
    recorded in the dedup ledger. This is exactly the shape the pre-fix
    code could break: the claims raced their ledger checkpoint writes on the
    same ``tg-dedup:{user_id}`` thread and the second write clobbered the
    first (in-process registry only) so a later redelivery-after-restart
    could double-process the loser. The claim must share the turn's
    per-session lock so the ledger writes serialize per user."""

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply=f"reply:{message}", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    first = make_inbound(update_id=7030, text="first question")
    second = make_inbound(update_id=7031, text="second question")

    async def scenario() -> None:
        await asyncio.gather(
            process_inbound(first, telegram=telegram),
            process_inbound(second, telegram=telegram),
        )

    asyncio.run(scenario())

    # Exactly one turn per update (no duplicates, no losses).
    assert len(calls) == 2
    assert len(telegram.sent) == 2

    # Both ids are durably recorded: a FRESH guard (simulated restart) must
    # treat both as already claimed. Reading the ledger directly also pins
    # the durable contract a restart depends on.
    from app.telegram.dedup import build_claim_graph, dedup_thread_id

    async def read_seen(user_id: int) -> list[int]:
        graph = build_claim_graph()
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": dedup_thread_id(user_id)}}
        )
        return (snapshot.values or {}).get("seen_update_ids") or []

    assert 7030 in asyncio.run(read_seen(42))
    assert 7031 in asyncio.run(read_seen(42))

    from app.telegram.dedup import DuplicateGuard

    async def fresh_guard_sees_both() -> tuple[bool, bool]:
        guard = DuplicateGuard()
        first_again = await guard.try_claim(42, 7030)
        second_again = await guard.try_claim(42, 7031)
        return first_again, second_again

    assert asyncio.run(fresh_guard_sees_both()) == (False, False)


def test_repeated_delivery_after_restart_guard_suppressed(
    monkeypatch: pytest.MonkeyPatch,
):
    """A redelivery reaching a fresh process (simulated restart) must not
    run a second turn — the durable ledger, not the in-process registry,
    must suppress it (follows directly from the fix-1 contract)."""

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    inbound = make_inbound(update_id=7032)

    asyncio.run(process_inbound(inbound, telegram=telegram))
    # Simulate a restart: reset_duplicate_guard (conftest autouse also does
    # this around each test) and reprocess the same update.
    from app.telegram.dedup import reset_duplicate_guard

    reset_duplicate_guard()
    asyncio.run(process_inbound(inbound, telegram=telegram))

    assert len(calls) == 1
    assert len(telegram.sent) == 1


def test_full_pipeline_runs_agent_with_sync_only_saver(
    monkeypatch: pytest.MonkeyPatch,
):
    """PRODUCTION-INCIDENT REGRESSION (service level): with a saver shaped
    exactly like production's sync ``PostgresSaver`` (async methods raise
    ``NotImplementedError``), the full pipeline must STILL claim durably,
    run the agent, and deliver the reply — the pre-fix code answered every
    message with the friendly error because ``try_claim`` awaited the
    unsupported async checkpoint API."""
    import app.memory_db as memory_db
    from test_telegram_synconly_checkpoint import SyncOnlyPostgresStyleSaver

    monkeypatch.setattr(memory_db, "checkpointer_factory", lambda: SyncOnlyPostgresStyleSaver())

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    calls = patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    asyncio.run(process_inbound(make_inbound(update_id=7900), telegram=telegram))

    assert len(calls) == 1
    assert telegram.sent == [(555_001, "We have hats!")]


def test_dedup_failure_log_names_stage_and_type_not_content(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Sanitized logging: failure stage + exception TYPE, no exception
    messages, no customer message contents, no credentials."""
    import logging as _logging

    guard = MagicMock()
    guard.try_claim = MagicMock(
        side_effect=RuntimeError(
            "ledger exploded: postgres://user:supersecret@db.example/store"
        )
    )
    patch_run_chat(monkeypatch, lambda m, s: MagicMock(reply="x", session_id=s))
    telegram = FakeTelegram()

    with caplog.at_level(_logging.ERROR, logger="app.telegram"):
        asyncio.run(
            process_inbound(
                make_inbound(update_id=7950, text="my hat size question"),
                telegram=telegram,
                guard=guard,
            )
        )

    text = caplog.text
    assert "stage=dedup_claim" in text
    assert "exception_type=RuntimeError" in text
    assert "update_id=7950" in text
    # Never logged: exception message/DSN, customer message content, secret.
    assert "supersecret" not in text
    assert "ledger exploded" not in text
    assert "my hat size question" not in text
    # Friendly message still the only customer-facing response.
    assert telegram.sent == [(555_001, FRIENDLY_ERROR_MESSAGE)]


def test_unexpected_agent_failure_logs_redacted_traceback(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Unexpected exceptions get a traceback in the log — with DSNs and
    bot-token-shaped credentials redacted, and customer text absent."""
    import logging as _logging

    class Boom(RuntimeError):
        pass

    def behavior(message: str, session_id: str) -> Any:
        raise Boom(
            f"unexpected: postgres://u:pw@db/x bot999:deadbeefdeadbeefdeadbeef"
        )

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()

    with caplog.at_level(_logging.ERROR, logger="app.telegram"):
        asyncio.run(
            process_inbound(
                make_inbound(update_id=7951, text="tell me about mugs"),
                telegram=telegram,
            )
        )

    text = caplog.text
    assert "stage=agent_turn" in text
    assert "exception_type=Boom" in text
    assert "Traceback (most recent call last)" in text
    # Redactions: DSN and bot-token-shaped credentials never survive.
    assert "postgres://" not in text
    assert "bot999:" not in text
    assert "<redacted-dsn>" in text
    assert "bot<redacted-token>" in text
    # Customer message content is never logged.
    assert "tell me about mugs" not in text


def test_deliver_reply_failure_logged_at_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    import logging as _logging

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram(fail=True)

    with caplog.at_level(_logging.WARNING, logger="app.telegram"):
        asyncio.run(process_inbound(make_inbound(update_id=7952), telegram=telegram))

    text = caplog.text
    assert "stage=deliver_reply" in text
    assert "exception_type=RuntimeError" in text
    assert "delivery failed" not in text  # message never logged, type only


def test_pipeline_boundary_and_duplicate_skip_are_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Diagnostic boundaries (ids only) are visible for every outcome: the
    pipeline start/end stamps and — critically for the incident — the
    SILENT duplicate-skip path now announces itself, so a stale claimed
    update id can be identified from the logs alone."""
    import logging as _logging

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()
    inbound = make_inbound(update_id=7960, text="do you sell mugs")

    with caplog.at_level(_logging.INFO, logger="app.telegram"):
        asyncio.run(process_inbound(inbound, telegram=telegram))
        asyncio.run(process_inbound(inbound, telegram=telegram))  # redelivery

    text = caplog.text
    assert "pipeline start update_id=7960" in text
    assert "outcome=reply_sent" in text
    assert text.count("stage=duplicate_skip") == 1  # only the redelivery
    # Identifiers only — message contents never reach the pipeline log.
    assert "do you sell mugs" not in text


def test_agent_turn_timeout_sends_friendly_error_and_never_retries(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """LIVE-HANG REGRESSION (service level): a hung agent turn (e.g. a Neon
    connection stalled on a dead socket) used to block this customer's
    session lock FOREVER with no log and no reply. The turn is now bounded:
    on timeout the customer gets exactly one friendly message, the update
    stays durably claimed (no second turn possible), and the stage is
    logged at WARNING with the exception type only."""
    import logging as _logging
    import time as _time

    def slow_run_chat(message: str, session_id: str) -> Any:
        _time.sleep(1.0)  # hangs past the (patched) timeout
        raise AssertionError("turn must not complete after a timeout")

    monkeypatch.setattr("app.telegram.service.run_chat", slow_run_chat)
    monkeypatch.setattr("app.telegram.service.TURN_TIMEOUT_SECONDS", 0.05)
    telegram = FakeTelegram()
    inbound = make_inbound(update_id=7970, text="where are the hats")

    with caplog.at_level(_logging.WARNING, logger="app.telegram"):
        asyncio.run(process_inbound(inbound, telegram=telegram))

    assert telegram.sent == [(555_001, FRIENDLY_ERROR_MESSAGE)]
    text = caplog.text
    assert "stage=agent_turn" in text
    assert "exception_type=TimeoutError" in text
    # Never logged: message content, credentials, payloads.
    assert "where are the hats" not in text


def test_agent_turn_timeout_timing_logs_present(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """The bounded stages announce start/end with durations, so the precise
    hung operation is identifiable from the log alone (live-hang diagnosis)."""
    import logging as _logging

    def behavior(message: str, session_id: str) -> MagicMock:
        return MagicMock(reply="We have hats!", session_id=session_id)

    patch_run_chat(monkeypatch, behavior)
    telegram = FakeTelegram()

    with caplog.at_level(_logging.INFO, logger="app.telegram"):
        asyncio.run(process_inbound(make_inbound(update_id=7971), telegram=telegram))

    text = caplog.text
    assert "stage=claim_read" in text and "event=end" in text
    assert "stage=claim_write" in text
    assert "stage=agent_turn" in text and "duration_ms=" in text