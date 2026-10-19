"""Telegram Bot API client (httpx, async) — Stage 1 transport only.

Implements the minimal outbound surface:

- ``POST sendMessage`` — deliver text replies to a chat (payload shape pinned
  by tests: ``chat_id``, ``text``, link previews disabled).

Design rules (matching this project's client conventions):
- The bot token lives ONLY in the httpx ``base_url`` path
  (``https://api.telegram.org/bot{token}``, Telegram's documented convention)
  and is therefore never added to logs, error messages, state, or payloads.
  Error messages reference the API path name (e.g. ``sendMessage``), never
  the URL.
- Explicit per-client timeout; bounded retries with exponential backoff for
  TRANSIENT failures only (timeouts, connection errors, 429/5xx). Never
  retries 4xx. Retries stop after ``max_retries`` attempts and are surfaced
  as typed errors — never silently converted into success.
- Responses that do not match Telegram's ``{ok: true, result: ...}``
  envelope raise ``TelegramResponseError`` (HTTP-level mismatches raise the
  corresponding typed error first).
- ``transport`` is a test-injection seam (``httpx.MockTransport``); production
  code never passes it. Failures must be checked against ``TelegramError``
  subclasses; every error string is safe to log.
- Sending timing logs: each HTTP attempt logs (attempt number, duration in
  ms, HTTP status); retry backoff waits are logged with the triggering
  condition (exception TYPE or HTTP status only). NEVER logged: the request
  URL (the bot token lives in the base_url path), the bot token or webhook
  secret, the JSON payload, or any message text. The log line carries the
  Bot API method name (``stage=sendMessage``) — an API-path literal (no
  token), never a URL.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from app.telegram.errors import (
    TelegramAuthError,
    TelegramConfigError,
    TelegramConnectionError,
    TelegramRateLimitedError,
    TelegramResponseError,
    TelegramServerError,
    TelegramTimeoutError,
)

_TELEGRAM_API_BASE = "https://api.telegram.org"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 2  # bounded: at most 2 retries after the first attempt

# Retried HTTP statuses: transient server-side/ratelimit conditions only.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_RETRY_BACKOFF_SECONDS = 0.25  # first backoff; doubles per retry (0.25, 0.5, ...)

# Same sanitized "app.telegram" tree as the pipeline/dedup timing logs (see
# app.telegram.__init__: one stderr handler at INFO+). Identifiers only.
_logger = logging.getLogger("app.telegram")


class TelegramClient:
    """Typed async Bot API access for sending text replies to chats."""

    def __init__(
        self,
        bot_token: str,
        *,
        base_url: str = _TELEGRAM_API_BASE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Build the client.

        Raises ``TelegramConfigError`` when the bot token is empty (a
        placeholder token is filtered out earlier, in config validation).
        """
        if not bot_token:
            raise TelegramConfigError(
                "Telegram integration is not configured (set TELEGRAM_BOT_TOKEN)"
            )
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=f"{base_url}/bot{bot_token}",
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Accept": "application/json"},
            transport=transport,
        )

    async def aclose(self) -> None:
        """Release the underlying httpx pool (application shutdown)."""
        await self._client.aclose()

    async def send_message(self, chat_id: int, text: str) -> None:
        """``POST sendMessage`` — deliver the reply (raises typed errors)."""
        payload = {
            "chat_id": chat_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        await self._request_json("POST", "sendMessage", json_payload=payload)

    # ------------------------------------------------------------------
    # Request core
    # ------------------------------------------------------------------

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> Any:
        """Perform one Bot API call with bounded retries; return ``result``.

        Retry policy: transient only (``httpx.TimeoutException``,
        ``httpx.TransportError``, HTTP 429/500/502/503/504), at most
        ``max_retries`` retries with exponential backoff. All permanent
        outcomes raise typed errors whose messages contain no credential
        material.
        """
        retries_left = self._max_retries
        backoff = _RETRY_BACKOFF_SECONDS
        attempt = 0

        while True:
            attempt += 1
            attempt_started = time.monotonic()
            _logger.info(
                "telegram request stage=%s event=start attempt=%d",
                path,
                attempt,
            )
            try:
                response = await self._client.request(method, path, json=json_payload)
            except httpx.TimeoutException as exc:
                self._log_slow_or_failed_attempt(
                    path, attempt, attempt_started, outcome="timeout"
                )
                if retries_left > 0:
                    retries_left -= 1
                    self._log_retry_wait(path, attempt, attempt + 1, backoff, "TimeoutError")
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise TelegramTimeoutError("Telegram request timed out") from exc
            except httpx.TransportError as exc:
                self._log_slow_or_failed_attempt(
                    path, attempt, attempt_started, outcome="transport_error",
                    exception_type=type(exc).__name__,
                )
                if retries_left > 0:
                    retries_left -= 1
                    self._log_retry_wait(
                        path, attempt, attempt + 1, backoff, type(exc).__name__
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                raise TelegramConnectionError("Could not reach Telegram") from exc

            duration_ms = int((time.monotonic() - attempt_started) * 1000)
            _logger.info(
                "telegram request stage=%s event=%s status=%d "
                "duration_ms=%d attempt=%d",
                path,
                "end" if response.is_success else "http_error",
                response.status_code,
                duration_ms,
                attempt,
            )

            if response.status_code in _RETRYABLE_STATUS_CODES and retries_left > 0:
                retries_left -= 1
                self._log_retry_wait(
                    path, attempt, attempt + 1, backoff,
                    f"http_status_{response.status_code}",
                )
                await asyncio.sleep(backoff)
                backoff *= 2
                continue

            self._raise_for_status(response, path)
            try:
                body = response.json()
            except ValueError as exc:
                raise TelegramResponseError(
                    "Telegram returned a non-JSON body"
                ) from exc
            return self._unwrap(body, path)

    # ------------------------------------------------------------------
    # Sanitized timing logs (identifiers only — never URL/token/payload)
    # ------------------------------------------------------------------

    @staticmethod
    def _log_slow_or_failed_attempt(
        path: str,
        attempt: int,
        attempt_started: float,
        *,
        outcome: str,
        exception_type: str | None = None,
    ) -> None:
        """End-of-attempt line for exception outcomes (exception TYPE only)."""
        parts = [
            f"telegram request stage={path} event=request_failed "
            f"outcome={outcome} exception_type={exception_type} "
            f"duration_ms={int((time.monotonic() - attempt_started) * 1000)} "
            f"attempt={attempt}",
        ]
        _logger.info(" ".join(parts))

    @staticmethod
    def _log_retry_wait(
        path: str,
        attempt: int,
        next_attempt: int,
        backoff_seconds: float,
        reason: str,
    ) -> None:
        """One line per bounded backoff wait (exception TYPE/status only)."""
        _logger.info(
            "telegram request stage=%s event=retry_backoff "
            "reason=%s stop_ms=%d next_attempt=%d",
            path,
            reason,
            int(backoff_seconds * 1000),
            next_attempt,
        )

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        """Map an HTTP error response onto a typed integration error."""
        if response.is_success:
            return
        status_code = response.status_code
        if status_code in (401, 403):
            raise TelegramAuthError(
                f"Telegram rejected the bot token ({status_code}) for {path}"
            )
        if status_code == 429:
            raise TelegramRateLimitedError(
                f"Telegram rate limit hit ({status_code}) for {path}"
            )
        if status_code >= 500:
            raise TelegramServerError(
                f"Telegram server error ({status_code}) for {path}"
            )
        raise TelegramResponseError(
            f"Unexpected Telegram response ({status_code}) for {path}: "
            f"{self._safe_detail(response)}"
        )

    @staticmethod
    def _unwrap(body: Any, path: str) -> Any:
        """Validate Telegram's ``{ok: bool, result: ...}`` envelope.

        A 200 response with ``ok: false`` is Bot API failure semantics, not
        success — the sanitized ``description`` is included (truncated).
        """
        if not isinstance(body, dict) or body.get("ok") is not True:
            description = body.get("description") if isinstance(body, dict) else None
            raise TelegramResponseError(
                f"Telegram Bot API call failed for {path}: {str(description)[:200]}"
            )
        return body.get("result")

    @staticmethod
    def _safe_detail(response: httpx.Response) -> str:
        """A short sanitized detail — never echoes the token or full body."""
        try:
            body: Any = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(body, dict):
            return str(body.get("description"))[:200]
        return str(body)[:200]
