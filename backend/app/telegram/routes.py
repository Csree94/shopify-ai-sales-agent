"""Telegram webhook endpoint — Stage 2.

Receives Telegram updates at the external path ``POST /api/telegram/webhook``
(router ``prefix="/api"`` + path literal ``/telegram/webhook``). Contract
with Telegram:

- Telegram POSTs one JSON update per call and retries on non-2xx. The route
  validates what it can, schedules what it must, and answers ``200
  {"ok": true}`` for anything that parses as an update (even non-text ones,
  which are skipped) — a 4xx/5xx would make Telegram re-deliver forever.
- Verification: ``X-Telegram-Bot-Api-Secret-Token`` is required and compared
  in constant time against ``TELEGRAM_WEBHOOK_SECRET``. Verification is
  NEVER silently skipped: a missing/empty provided header fails the
  comparison. A 401 goes back to Telegram, not to any customer.
- Disabled integration (no configured client) → controlled 503.
- Accepted updates: acknowledged immediately; the turn runs in a managed
  background task (tracked on ``app.state.telegram_background_tasks``, see
  ``app.main.lifespan``) bounded by an asyncio semaphore so a slow LLM turn
  never causes Telegram webhook timeouts + unbounded re-deliveries.
- Secrets: the bot token never appears in any response or error (it lives
  only inside the client's URL path); the webhook secret is only ever
  compared, never stored on the request path or echoed.
"""

from __future__ import annotations

import asyncio
import hmac
import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from app.config import load_telegram_settings
from app.telegram.schemas import extract_inbound, parse_update

router = APIRouter(prefix="/api", tags=["telegram"])

# Same logger tree as the background pipeline; the package __init__ installs
# one stderr handler at INFO so the full webhook→reply path is visible in the
# uvicorn terminal without any logging configuration (ids only, never
# payloads or message contents).
logger = logging.getLogger("app.telegram")

WEBHOOK_PATH = "/telegram/webhook"
SECRET_HEADER = "X-Telegram-Bot-Api-Secret-Token"

# Bounded worker pool: at most this many agent turns run concurrently per
# process; additional deliveries are acknowledged and queue as tasks.
TURN_SEMAPHORE_LIMIT = 5
turn_semaphore = asyncio.Semaphore(TURN_SEMAPHORE_LIMIT)


class WebhookAck(BaseModel):
    """Minimal 200 response body for Telegram webhook deliveries."""

    ok: bool = True


def _make_done_callback(tracked: set | None):
    """Done-callback factory: consumes the task exception (prevents
    unhandled-task warnings) and discards the task from the tracker set.

    Exceptions are consumed deliberately, but NOT silently anymore: any
    exception escaping the pipeline is logged via the service's sanitized
    ``_log_failure`` (stage + type + redacted traceback). ``process_inbound``
    resolves every failure path internally, so a raised exception here
    indicates a bug rather than a customer outcome.
    """

    def _done(task: "asyncio.Task[None]") -> None:
        if not task.cancelled():
            exc = task.exception()  # consumed by design; see note above
            if exc is not None:
                # A failing task here means the pipeline's own broad handling
                # was escaped — close the diagnostic black hole: stage +
                # exception type, redacted traceback, never message contents.
                from app.telegram.service import _log_failure

                _log_failure(
                    "webhook_background_task", exc, with_traceback=True
                )
        if tracked is not None:
            tracked.discard(task)

    return _done


@router.post(WEBHOOK_PATH, response_model=WebhookAck)
async def telegram_webhook(request: Request) -> WebhookAck:
    """Receive one Telegram update; fast-ack and process in the background."""
    telegram = getattr(request.app.state, "telegram_client", None)
    if telegram is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram integration is not configured",
        )

    settings = load_telegram_settings()
    if settings is None:
        # Config vanished between startup and this request: fail controlled,
        # never with an unauthenticated webhook.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram integration is not configured",
        )

    provided = request.headers.get(SECRET_HEADER)
    if not provided or not hmac.compare_digest(
        provided.encode("utf-8"), settings.webhook_secret.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook secret token",
        )

    try:
        payload = await request.json()
    except Exception:
        logger.info("webhook skipped reason=malformed_body")
        return WebhookAck()  # acknowledge garbage; do not cause retry loops

    update = parse_update(payload)
    if update is None:
        logger.info("webhook skipped reason=unparsable_update")
        return WebhookAck()  # unparsable: ack + skip

    inbound = extract_inbound(update)
    if inbound is None:
        # Ids only — the payload itself is never logged.
        logger.info(
            "webhook skipped reason=unacceptable_message_kind update_id=%s",
            update.update_id,
        )
        return WebhookAck()  # edits / channel posts / non-text / groups / blank

    from app.telegram.service import process_inbound  # local: cycle-free import

    async def _process() -> None:
        async with turn_semaphore:
            await process_inbound(inbound, telegram=telegram)

    task = asyncio.create_task(_process())
    # Strong reference + consumed-exception callback: asyncio garbage-collects
    # unreferenced tasks and warns on unconsumed exceptions — neither happens.
    tracked = getattr(request.app.state, "telegram_background_tasks", None)
    task.add_done_callback(_make_done_callback(tracked))
    if tracked is not None:
        tracked.add(task)
    # Diagnosability boundary: proves THIS build scheduled the turn (ids only).
    logger.info(
        "webhook accepted update_id=%s background_turn_scheduled=true",
        inbound.update_id,
    )
    return WebhookAck()
