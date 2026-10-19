"""Pydantic schemas for the verified subset of Telegram webhook updates.

Transport contracts only — they describe what Telegram sends and carry no
business logic (and no webhooks yet in Stage 1; this layer is consumed by a
later stage). Parsing is defensive: a non-text update (photo, sticker, voice,
callback query, …), an edited message, a channel post, a non-private chat, or
blank text parses/extracts to ``None`` rather than raising, so a future
webhook can acknowledge-and-skip safely.

Secrets: these schemas hold only public profile fields and chat identifiers.
No bot token or webhook secret exists in any schema (the token lives only in
the client's URL path; the secret is verified at the request boundary in a
later stage and never stored).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TelegramUser(BaseModel):
    """Public Telegram user profile (subset used for identity mapping)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: int = Field(..., description="Telegram user id (external identity)")
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    language_code: str | None = None


class TelegramChat(BaseModel):
    """Telegram chat identifier (private chat: user id == chat id)."""

    model_config = ConfigDict(extra="ignore")

    id: int
    type: str | None = None


class TelegramMessage(BaseModel):
    """The verified subset of an incoming Telegram message."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    message_id: int
    from_user: TelegramUser | None = Field(default=None, alias="from")
    chat: TelegramChat
    date: int | None = None
    text: str | None = None


class TelegramUpdate(BaseModel):
    """One webhook update delivery from Telegram."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    update_id: int
    message: TelegramMessage | None = None
    edited_message: TelegramMessage | None = None


class InboundMessage(BaseModel):
    """Normalized inbound customer message extracted from an update."""

    model_config = ConfigDict(frozen=True)

    chat_id: int
    user: TelegramUser
    text: str
    message_id: int
    update_id: int
    sent_at: datetime | None = None


def parse_update(payload: Any) -> TelegramUpdate | None:
    """Parse a raw webhook JSON payload; ``None`` when it is not an update.

    Defensive against malformed bodies: a non-dict payload or a dict that
    does not validate returns ``None`` instead of raising, so a later webhook
    can acknowledge (200) rather than trigger Telegram retry loops.
    """
    if not isinstance(payload, dict):
        return None
    try:
        return TelegramUpdate.model_validate(payload)
    except Exception:
        return None


def extract_inbound(update: TelegramUpdate) -> InboundMessage | None:
    """Normalize an update into an inbound customer message (or ``None``).

    ``None`` (skip) for: edits, channel posts, non-text messages, messages
    without a user, messages from non-private chats, and blank text. Private
    1:1 chats only in this stage — the sales agent talks to one customer at
    a time.
    """
    message = update.message  # edited_message / channel posts: deliberately ignored
    if message is None:
        return None
    if message.from_user is None:
        return None
    chat = message.chat
    if chat.type != "private":
        return None
    text = message.text
    if text is None or not text.strip():
        return None
    return InboundMessage(
        chat_id=chat.id,
        user=message.from_user,
        text=text.strip(),
        message_id=message.message_id,
        update_id=update.update_id,
        sent_at=(
            datetime.fromtimestamp(message.date, tz=UTC) if message.date is not None else None
        ),
    )
