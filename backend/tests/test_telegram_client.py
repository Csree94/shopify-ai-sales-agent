"""Tests for the Telegram Bot API client (Stage 1).

All HTTP I/O goes through ``httpx.MockTransport`` — no real Telegram calls,
no network. Covers the sendMessage payload/URL shape, the ``{ok, result}``
envelope, typed error mapping, bounded transient retries, token
redaction in error messages, and the sanitized timing logs around each
outbound HTTP attempt (start / status / duration / retry backoff),
asserted to never contain the token, URL, or message text.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from app.telegram.client import TelegramClient
from app.telegram.errors import (
    TelegramAuthError,
    TelegramConfigError,
    TelegramConnectionError,
    TelegramError,
    TelegramRateLimitedError,
    TelegramResponseError,
    TelegramServerError,
    TelegramTimeoutError,
)

FAKE_TOKEN = "123456:FAKE-TEST-TOKEN"


def make_client(handler: Any, **kwargs: Any) -> TelegramClient:
    return TelegramClient(FAKE_TOKEN, transport=httpx.MockTransport(handler), **kwargs)


def ok_result(result: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result if result is not None else {}})


def test_empty_token_raises_config_error():
    with pytest.raises(TelegramConfigError):
        TelegramClient("")


def test_send_message_payload_and_url_shape():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return ok_result({"message_id": 1})

    asyncio.run(make_client(handler).send_message(777, "hello there"))

    # Token sits in the URL path (Telegram's documented convention); only
    # the bot-id prefix is asserted so the fake token is not echoed further.
    assert captured["url"].startswith("https://api.telegram.org/bot123456:")

    assert captured["url"].endswith("/sendMessage")
    body = captured["body"].decode()
    assert '"chat_id":777' in body.replace(" ", "")
    assert "hello there" in body
    assert '"link_preview_options":{"is_disabled":true}' in body.replace(" ", "")


def test_send_message_success_unwraps_result():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return ok_result({"message_id": 42})

    asyncio.run(make_client(handler).send_message(1, "hi"))
    assert calls == [1]


def test_ok_false_envelope_maps_to_response_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "chat not found"})

    with pytest.raises(TelegramResponseError) as excinfo:
        asyncio.run(make_client(handler).send_message(1, "hi"))
    assert "chat not found" in str(excinfo.value)


def test_malformed_json_body_maps_to_response_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json{{", headers={
            "content-type": "application/json"
        })

    with pytest.raises(TelegramResponseError):
        asyncio.run(make_client(handler).send_message(1, "hi"))


def test_401_maps_to_auth_error_and_hides_token():
    def handler(request: httpx.Request) -> httpx.Response:
        # Malicious-looking body echoing the token must NOT propagate.
        return httpx.Response(401, json={
            "ok": False, "description": f"Unauthorized for bot{FAKE_TOKEN}"
        })

    with pytest.raises(TelegramAuthError) as excinfo:
        asyncio.run(make_client(handler).send_message(1, "hi"))
    assert FAKE_TOKEN not in str(excinfo.value)


def test_403_maps_to_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"ok": False, "description": "forbidden"})

    with pytest.raises(TelegramAuthError):
        asyncio.run(make_client(handler).send_message(1, "hi"))


def test_429_maps_to_rate_limited_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"ok": False, "description": "too many"})

    with pytest.raises(TelegramRateLimitedError):
        asyncio.run(
            make_client(handler, max_retries=0).send_message(1, "hi")
        )


def test_500_retries_then_maps_to_server_error():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(500, json={"ok": False, "description": "boom"})

    with pytest.raises(TelegramServerError):
        asyncio.run(make_client(handler, max_retries=2).send_message(1, "hi"))
    # first attempt + 2 bounded retries
    assert len(attempts) == 3


def test_429_then_success_recovers_within_retry_budget():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, json={"ok": False, "description": "flood"})
        return ok_result({"message_id": 7})

    asyncio.run(make_client(handler, max_retries=1).send_message(1, "hi"))
    assert len(attempts) == 2


def test_4xx_is_never_retried():
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"ok": False, "description": "bad chat id"})

    with pytest.raises(TelegramResponseError):
        asyncio.run(make_client(handler, max_retries=3).send_message(1, "hi"))
    assert len(attempts) == 1


def test_timeout_maps_to_timeout_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(TelegramTimeoutError):
        # max_retries=0 keeps the test quick (no backoff sleeps).
        asyncio.run(make_client(handler, max_retries=0).send_message(1, "hi"))


def test_transport_error_maps_to_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(TelegramConnectionError):
        asyncio.run(make_client(handler, max_retries=0).send_message(1, "hi"))


def test_error_text_never_contains_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"description": "server error"})

    with pytest.raises(TelegramError) as excinfo:
        asyncio.run(make_client(handler, max_retries=0).send_message(1, "hi"))
    assert FAKE_TOKEN not in str(excinfo.value)


def test_aclose_is_idempotent_enough_for_shutdown():
    async def scenario() -> None:
        client = make_client(lambda request: ok_result())
        await client.send_message(1, "hi")
        await client.aclose()
        await client.aclose()  # closing twice must not raise

    asyncio.run(scenario())


# ----------------------------------------------------------------------
# Sanitized timing logs around the outbound HTTP attempt
# ----------------------------------------------------------------------


def _request_log_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records
            if rec.name == "app.telegram" and "telegram request " in rec.getMessage()]


def test_send_message_success_logs_start_and_end(caplog: pytest.LogCaptureFixture):
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_result({"message_id": 3})

    with caplog.at_level("INFO", logger="app.telegram"):
        asyncio.run(make_client(handler).send_message(1, "secret reply text"))

    lines = _request_log_records(caplog)
    assert any("stage=sendMessage event=start attempt=1" in line for line in lines)
    assert any("stage=sendMessage event=end status=200 duration_ms=" in line
               and "attempt=1" in line for line in lines)


def test_timing_logs_never_contain_token_url_or_text(caplog: pytest.LogCaptureFixture):
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_result({"message_id": 3})

    with caplog.at_level("INFO", logger="app.telegram"):
        asyncio.run(make_client(handler).send_message(1, "secret reply text"))

    lines = _request_log_records(caplog)
    assert lines  # timing logs exist
    for line in lines:
        assert FAKE_TOKEN not in line
        assert "https://" not in line
        assert "api.telegram.org" not in line
        assert "secret reply text" not in line


def test_retryable_status_logs_error_line_and_backoff(caplog: pytest.LogCaptureFixture):
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) <= 2:
            return httpx.Response(500, json={"ok": False, "description": "boom"})
        return ok_result({"message_id": 9})

    with caplog.at_level("INFO", logger="app.telegram"):
        asyncio.run(make_client(handler, max_retries=2).send_message(1, "hi"))

    lines = _request_log_records(caplog)
    assert any("event=start attempt=2" in line for line in lines)
    assert any("stage=sendMessage event=http_error status=500 " in line
               for line in lines)
    assert any("event=retry_backoff reason=http_status_500 " in line
               for line in lines)
    assert any("event=end status=200" in line for line in lines)  # final attempt


def test_transport_error_logs_failed_attempt_without_exception_text(
        caplog: pytest.LogCaptureFixture):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused (may embed the url)")

    with caplog.at_level("INFO", logger="app.telegram"):
        with pytest.raises(TelegramConnectionError):
            # max_retries=0 keeps the test quick (no backoff sleeps).
            asyncio.run(make_client(handler, max_retries=0).send_message(1, "hi"))

    lines = _request_log_records(caplog)
    assert any("event=request_failed outcome=transport_error "
               "exception_type=ConnectError" in line for line in lines)
    for line in lines:
        # Exception MESSAGES are never logged — only the type name.
        assert "connection refused" not in line
        assert FAKE_TOKEN not in line