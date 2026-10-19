"""Telegram transport package (Stage 1: config + client + schemas only).

Deliberately NO webhook route and NO agent wiring yet — that is Stage 2.
Structure follows the project's client/error conventions:

- ``errors``  — typed integration errors, safe to log, never containing the
  bot token or webhook secret
- ``client``  — ``TelegramClient`` (httpx Bot API; the only outbound path;
  bounded transient retries)
- ``schemas`` — defensive webhook update contracts + extraction helpers
"""

import logging

from app.telegram.client import TelegramClient


def _configure_terminal_logging() -> logging.Logger:
    """One stderr handler for the ``app.telegram`` logger tree (INFO+).

    Guarantees the webhook→pipeline diagnostic lines appear in the uvicorn
    terminal without any logging configuration (the project default is an
    unconfigured root logger, where INFO is dropped silently and only
    WARNING+ would surface). Idempotent: one handler for the process
    lifetime. Records carry identifiers only — never tokens, secrets,
    payloads, or message contents (pipeline rules in
    ``app.telegram.service``). Propagation stays enabled so pytest's
    ``caplog`` keeps seeing records. The asctime field is included so
    wall-clock GAPS between stages (webhook accepted → pipeline start →
    claim → turn → sendMessage) are measurable in the terminal — that is
    where the unexplained ~45 s of the live minute delay will localize.
    """
    logger = logging.getLogger("app.telegram")
    if not logger.handlers:
        handler = logging.StreamHandler()  # stderr — uvicorn terminal
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
        )
        logger.addHandler(handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


_configure_terminal_logging()

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
from app.telegram.schemas import (
    InboundMessage,
    TelegramChat,
    TelegramMessage,
    TelegramUpdate,
    TelegramUser,
    extract_inbound,
    parse_update,
)

__all__ = [
    "InboundMessage",
    "TelegramAuthError",
    "TelegramChat",
    "TelegramClient",
    "TelegramConfigError",
    "TelegramConnectionError",
    "TelegramError",
    "TelegramMessage",
    "TelegramRateLimitedError",
    "TelegramResponseError",
    "TelegramServerError",
    "TelegramTimeoutError",
    "TelegramUpdate",
    "TelegramUser",
    "extract_inbound",
    "parse_update",
]
