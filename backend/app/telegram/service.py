"""Telegram → agent turn pipeline (Stage 2).

Connects accepted private-chat Telegram messages to the EXISTING agent entry
point ``app.agent.run_chat`` — no second agent, no bypassed LangGraph:

- Session identity: ``session_id = f"tg:{user_id}"`` — the ``tg:`` prefix
  keeps Telegram threads namespaced apart from the ordinary web surface
  (whose ids arrive via ``POST /api/chat``), inside one shared LangGraph
  checkpoint namespace. Gemini primary / NVIDIA Nemotron fallback behavior
  is whatever ``run_chat`` already does — untouched.
- Duplicate protection: one durable ``try_claim`` per update (see
  ``app.telegram.dedup``) before the turn. The claim runs INSIDE the same
  per-session asyncio lock as the turn, so two updates from one user can
  never race the dedup ledger's checkpoint write (a concurrent write on the
  same ``tg-dedup:{user_id}`` thread could otherwise clobber the other's
  durable record) and cannot interleave conversation checkpoint writes.
- Agent failures (config, agent, Shopify tool failures, checkpoint store)
  all become ONE customer-safe friendly message — never internal error
  details, never credentials. The turn contract keeps Shopify read-only.
- Reply delivery failures are deliberately non-retried background failures
  (the webhook already acked); nothing raises out of this module.

Note on sync I/O: ``run_chat`` is a synchronous LLM turn; it runs through
``asyncio.to_thread`` so the event loop (and other customers' turns) stay
responsive while one turn executes.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import traceback
from collections import OrderedDict

from app.agent import ChatAgentError, run_chat
from app.telegram.client import TelegramClient
from app.telegram.dedup import DuplicateGuard, duplicate_guard
from app.telegram.schemas import InboundMessage


def reset_session_locks() -> None:
    """Test helper mirroring ``dedup.reset_duplicate_guard`` (per-process
    map, one uvicorn loop in production)."""
    session_locks.reset()

FRIENDLY_ERROR_MESSAGE = (
    "Sorry, I couldn't process that request right now. Please try again shortly."
)

logger = logging.getLogger("app.telegram")

# Redaction patterns for any descriptive text that may reach the log: a
# Telegram bot-token-shaped credential and a Neon-style postgres DSN. Only
# ever applied to traceback text — exception MESSAGES are never logged
# themselves (stage + exception TYPE only), and customer message contents,
# request payloads, and raw chat text are never logged at all.
_TOKENISH = re.compile(r"bot\d+:[A-Za-z0-9_\-]{10,}")
_DSNISH = re.compile(r"postgres(?:ql)?[\w\+]*://\S+", re.IGNORECASE)


def _redact(text: str) -> str:
    text = _DSNISH.sub("<redacted-dsn>", text)
    text = _TOKENISH.sub("bot<redacted-token>", text)
    return text


def _log_failure(
    stage: str,
    exc: BaseException,
    *,
    update_id: int | None = None,
    with_traceback: bool = False,
    level: int = logging.ERROR,
) -> None:
    """One sanitized log line for a Telegram-pipeline failure.

    Logs the processing stage and the exception TYPE only — never exception
    messages (they may embed provider/store details), never credentials, DSNs,
    or message contents. A redacted traceback is attached only for unexpected
    exceptions (per request of the follow-up review-diagnosis convention).
    """
    parts = [f"telegram pipeline failure stage={stage}",
             f"exception_type={type(exc).__name__}"]
    if update_id is not None:
        parts.append(f"update_id={int(update_id)}")
    if with_traceback:
        parts.append(
            "traceback=" + _redact(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
        )
    logger.log(level, " | ".join(parts))

SESSION_PREFIX = "tg:"
SESSION_LOCK_CAP = 1024  # bound the per-session lock map

# Generous bound on ONE agent turn (LLM + Shopify tool loops legitimately
# take tens of seconds): without it, a hung Neon/LLM call inside the sync
# turn blocks this session's asyncio.Lock forever — every later message from
# this customer waits silently behind a dead connection (the live hang).
# A timeout resolves to the existing friendly-error handling, and because
# claiming precedes the turn, a turn can never run twice.
TURN_TIMEOUT_SECONDS = 180.0


def telegram_session_id(user_id: int) -> str:
    """The LangGraph session/thread id for one Telegram customer."""
    return f"{SESSION_PREFIX}{user_id}"


class SessionLockMap:
    """Per-customer turn serialization with a bounded lock map."""

    def __init__(self) -> None:
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def reset(self) -> None:
        """Test helper: drops every lock (asyncio.Lock objects bind to the
        event loop that first waits on them; tests run per-scenario loops,
        so the map must not leak locks across tests — production has one
        loop for the process and never needs this)."""
        self._locks.clear()

    def lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        self._locks.move_to_end(session_id)
        while len(self._locks) > SESSION_LOCK_CAP:
            self._locks.popitem(last=False)  # evict the least-recently used
        return lock


session_locks = SessionLockMap()


async def process_inbound(
    inbound: InboundMessage,
    *,
    telegram: TelegramClient,
    guard: DuplicateGuard | None = None,
) -> None:
    """Full pipeline for one inbound message; never raises.

    Under ONE per-session lock (serialized per customer):
    1. Claim the update (durable + per-process dedup).
    2. Run ONE existing agent turn.
    Then deliver the reply to the originating chat (outside the lock).

    Every failure path resolves to a customer-safe friendly message (or a
    swallowed delivery failure) — internal errors never reach the chat.
    ``run_chat`` executes in a worker thread via ``asyncio.to_thread`` so
    the (synchronous) LLM turn never blocks the event loop. Failures are
    logged with one sanitized line EACH (stage + exception type; redacted
    traceback only for unexpected exceptions) — see ``_log_failure``; the
    customer still sees only the friendly message.
    """
    guard = guard or duplicate_guard
    session_id = telegram_session_id(inbound.user.id)
    # Diagnostic boundary (ids only — never message content or payloads).
    logger.info("telegram pipeline start update_id=%s", inbound.update_id)

    async with session_locks.lock_for(session_id):
        # The claim shares the turn's per-session lock: concurrent different
        # updates from one user therefore serialize their ledger writes too
        # (the dedup ledger has no compare-and-swap; a concurrent checkpoint
        # write on the same thread could drop the other update's record).
        try:
            claimed = await guard.try_claim(inbound.user.id, inbound.update_id)
        except Exception as exc:
            # Ledger unavailable: fail safe — answer friendly, run nothing.
            _log_failure("dedup_claim", exc, update_id=inbound.update_id)
            await _safe_send(telegram, inbound.chat_id, FRIENDLY_ERROR_MESSAGE)
            return
        if not claimed:
            # A previously durably-claimed update reaching us again: silent by
            # design for the customer, LOGGED for diagnosability (ids only).
            logger.info(
                "telegram pipeline skipped stage=duplicate_skip update_id=%s",
                inbound.update_id,
            )
            return  # duplicate delivery: no second turn, no second reply

        try:
            turn_started = time.monotonic()
            logger.info(
                "telegram turn stage=agent_turn event=start update_id=%s",
                inbound.update_id,
            )
            outcome = await asyncio.wait_for(
                asyncio.to_thread(run_chat, inbound.text, session_id),
                timeout=TURN_TIMEOUT_SECONDS,
            )
            logger.info(
                "telegram turn stage=agent_turn event=end "
                "duration_ms=%d update_id=%s",
                int((time.monotonic() - turn_started) * 1000),
                inbound.update_id,
            )
        except asyncio.TimeoutError as exc:
            # Expected control-flow bound (NOT a bug): log type only, then the
            # friendly message — the durably claimed update cannot re-run.
            _log_failure(
                "agent_turn", exc, update_id=inbound.update_id,
                level=logging.WARNING,
            )
            await _safe_send(telegram, inbound.chat_id, FRIENDLY_ERROR_MESSAGE)
            return
        except Exception as exc:
            # Agent/config/Shopify/checkpoint failure — controlled, friendly,
            # never internal details (matches the 503 conventions of the web
            # surface's sanitized gate reasons). The agent's typed control
            # flow errors are expected outcomes; anything else is a bug and
            # gets a redacted traceback in the log.
            _log_failure(
                "agent_turn",
                exc,
                update_id=inbound.update_id,
                with_traceback=not isinstance(exc, (ChatAgentError, ValueError)),
            )
            await _safe_send(telegram, inbound.chat_id, FRIENDLY_ERROR_MESSAGE)
            return

    if outcome.reply:
        try:
            await telegram.send_message(inbound.chat_id, outcome.reply)
            logger.info(
                "telegram pipeline end update_id=%s outcome=reply_sent",
                inbound.update_id,
            )
        except Exception as exc:
            # The turn succeeded but delivery failed; the webhook already
            # acked Telegram, so raising would change nothing — the message
            # is lost-of-record in this stage (no retry queue exists yet).
            _log_failure(
                "deliver_reply", exc, update_id=inbound.update_id,
                level=logging.WARNING,
            )
            return
    else:
        # The agent produced NO reply text: explicitly visible (ids only),
        # so an empty-reply turn can never look like a dropped message.
        logger.info(
            "telegram pipeline end update_id=%s outcome=empty_reply",
            inbound.update_id,
        )


async def _safe_send(
    telegram: TelegramClient,
    chat_id: int,
    text: str,
    *,
    stage: str = "deliver_friendly_error",
) -> None:
    """Delivery inside a failure path: log the failure stage/type, never raise."""
    try:
        await telegram.send_message(chat_id, text)
    except Exception as exc:
        _log_failure(stage, exc, level=logging.WARNING)
        return  # delivery failure inside a failure path: nothing more to do
