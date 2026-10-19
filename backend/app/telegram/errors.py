"""Typed integration errors for the Telegram Bot API client (Stage 1).

Follows the project's ``shopify_errors`` / ``gemini_client`` style: typed,
safe to log, and never carrying credential material. The bot token lives only
inside the client's URL path; no error message, log line, or repr ever
contains it. Transport callers catch ``TelegramError`` and translate to
customer-safe behavior — details are for server-side diagnosability only.
"""

from __future__ import annotations


class TelegramError(Exception):
    """Base error for every Telegram integration failure."""


class TelegramConfigError(TelegramError):
    """The Telegram integration is misconfigured (e.g. missing bot token)."""


class TelegramAuthError(TelegramError):
    """Telegram rejected the bot token (HTTP 401/403). Never includes the token."""


class TelegramRateLimitedError(TelegramError):
    """Telegram rate-limited the Bot API call (HTTP 429)."""


class TelegramServerError(TelegramError):
    """Telegram answered with a server error (HTTP 5xx)."""


class TelegramTimeoutError(TelegramError):
    """A request to the Bot API timed out (connect or read)."""


class TelegramConnectionError(TelegramError):
    """The Bot API could not be reached (DNS, refused, reset, TLS)."""


class TelegramResponseError(TelegramError):
    """Telegram answered with a malformed/unexpected response body."""
