"""Tests for Telegram update schemas and defensive parsing (Stage 1).

Pure mocks — no HTTP at all. Covers defensive parse of garbage payloads,
extraction of valid private text messages, and skipping of every unsupported
delivery kind a webhook may receive.
"""

from __future__ import annotations

from typing import Any

from app.telegram.schemas import extract_inbound, parse_update


def make_update(
    *,
    update_id: int = 1001,
    text: str = "what products do you have",
    chat_id: int = 555_001,
    user_id: int = 42,
    chat_type: str = "private",
) -> dict[str, Any]:
    """A verified Telegram update shape (private chat, text message)."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 10,
            "from": {
                "id": user_id,
                "first_name": "Ada",
                "username": "ada_shop",
                "language_code": "en",
            },
            "chat": {"id": chat_id, "type": chat_type},
            "date": 1_790_000_000,
            "text": text,
        },
    }


# --- parse_update -----------------------------------------------------------


def test_parse_update_rejects_non_dict_payloads():
    assert parse_update(None) is None
    assert parse_update([1, 2, 3]) is None
    assert parse_update("update") is None
    assert parse_update(123) is None


def test_parse_update_rejects_update_without_id():
    assert parse_update({"foo": "bar"}) is None
    assert parse_update({"message": {}}) is None


def test_parse_update_accepts_valid_update():
    update = parse_update(make_update())
    assert update is not None
    assert update.update_id == 1001
    assert update.message is not None
    assert update.message.chat.id == 555_001


def test_parse_update_ignores_unknown_fields():
    payload = make_update()
    payload["channel_post"] = {"message_id": 3}
    payload["callback_query"] = {"id": "x"}
    update = parse_update(payload)
    assert update is not None  # known subset parsed, extras ignored


# --- extract_inbound ---------------------------------------------------------


def test_extract_inbound_valid_private_text():
    update = parse_update(make_update(text="  hello bot  "))
    assert update is not None
    inbound = extract_inbound(update)
    assert inbound is not None
    assert inbound.chat_id == 555_001
    assert inbound.user.id == 42
    assert inbound.text == "hello bot"  # stripped
    assert inbound.update_id == 1001
    assert inbound.message_id == 1011
    assert inbound.sent_at is not None


def test_extract_inbound_skips_edited_message():
    payload = make_update(update_id=3003)
    payload["edited_message"] = payload.pop("message")
    update = parse_update(payload)
    assert update is not None
    assert extract_inbound(update) is None


def test_extract_inbound_skips_group_messages():
    update = parse_update(make_update(update_id=3004, chat_type="group"))
    assert update is not None
    assert extract_inbound(update) is None


def test_extract_inbound_skips_channel_messages():
    update = parse_update(make_update(update_id=3005, chat_type="channel"))
    assert update is not None
    assert extract_inbound(update) is None


def test_extract_inbound_skips_message_without_user():
    payload = make_update(update_id=6006)
    del payload["message"]["from"]
    update = parse_update(payload)
    assert update is not None
    assert extract_inbound(update) is None


def test_extract_inbound_skips_blank_text():
    update = parse_update(make_update(update_id=7007, text="   "))
    assert update is not None
    assert extract_inbound(update) is None


def test_extract_inbound_skips_non_text_message():
    payload = make_update(update_id=2002)
    del payload["message"]["text"]
    payload["message"]["photo"] = [{"file_id": "abc"}]
    update = parse_update(payload)
    assert update is not None
    assert extract_inbound(update) is None