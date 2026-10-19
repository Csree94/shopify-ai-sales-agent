"""FastAPI entrypoint for the Shopify AI Sales Agent backend.

Phase 3: health endpoint. Phase 4: read-only Shopify product retrieval.
Phase 6/7: inventory probe (quantities + strict levels). Phase 8: the
read-only Gemini/LangGraph sales-agent chat surface (with the gated
Nemotron fallback). Stage 2: the optional Telegram webhook surface, with a
lifespan that initializes only Telegram infrastructure (the Neon
checkpointer self-initializes lazily at first use inside the agent and is
NOT touched here).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.chat import router as chat_router
from app.config import load_telegram_settings
from app.products import router as products_router
from app.telegram.client import TelegramClient
from app.telegram.routes import router as telegram_router

APP_NAME = "Shopify AI Sales Agent"
APP_VERSION = "0.1.0"

_telegram_logger = logging.getLogger("app.telegram")


async def _drain_background_tasks(tasks: set) -> None:
    """Await outstanding Telegram turn tasks before the client closes.

    ``return_exceptions=True``: every failure path inside a task is already
    resolved to a customer-safe outcome; draining here only guarantees that
    shutdown does not abandon a turn mid-flight (an abandoned turn could
    otherwise lose the reply AND have already claimed the update id).
    """
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown for the OPTIONAL Telegram integration only.

    - Telegram enabled (real token): one ``TelegramClient`` per process on
      ``app.state.telegram_client`` plus a background-task tracker set.
    - Telegram disabled (unset/placeholder token; unchanged behavior):
      ``telegram_client`` stays ``None``; the webhook route answers 503 and
      everything else (``/health``, ``/api/*``) behaves exactly as before.
    - Shutdown: drain outstanding turn tasks FIRST (they may still be
      delivering a reply), then close the Telegram HTTP pool.
    """
    settings = load_telegram_settings()
    app.state.telegram_client = (
        TelegramClient(settings.bot_token) if settings is not None else None
    )
    app.state.telegram_background_tasks = set()
    # Diagnostic boundary: proves WHICH build and WHICH process is serving
    # (the pid lets a stale pre-update process be identified immediately;
    # ids/configuration booleans only — never token or secret material).
    _telegram_logger.info(
        "telegram lifespan startup pid=%s enabled=%s",
        os.getpid(),
        settings is not None,
    )

    yield

    tasks = getattr(app.state, "telegram_background_tasks", None)
    if tasks:
        await _drain_background_tasks(tasks)
    client = getattr(app.state, "telegram_client", None)
    if client is not None:
        await client.aclose()
    _telegram_logger.info(
        "telegram lifespan shutdown pid=%s drained_tasks=%s client_closed=%s",
        os.getpid(),
        bool(tasks),
        client is not None,
    )


def create_app() -> FastAPI:
    """Build the FastAPI application (factory; tests may build fresh apps)."""
    app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
    app.include_router(products_router)
    app.include_router(chat_router)
    app.include_router(telegram_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        """Predictable health payload for liveness checks."""
        return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}

    return app


app = create_app()
