"""Reserved session-id validation for the web chat surface (review fix 2).

The Telegram surface owns the `tg:{user_id}` conversation threads and the
`tg-dedup:{user_id}` dedup-ledger threads (one shared LangGraph checkpoint
namespace). A client-supplied web session id colliding with either prefix
would let the unauthenticated web surface read or corrupt a Telegram
customer's checkpoint history — so both prefixes must be rejected with 422
while every previously valid session id keeps working unchanged.

Offline only: run_chat is patched at the ``app.chat.run_chat`` seam; no
provider, Shopify, or Telegram I/O ever happens.
"""

from __future__ import annotations

import types

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.chat import _session_id
from app.main import create_app

CHAT_URL = "/api/chat"


def make_inbound_call(patched_calls: list[str]):
    def behavior(message: str, session_id: str) -> types.SimpleNamespace:
        patched_calls.append(session_id)
        return types.SimpleNamespace(
            reply="ok",
            session_id=session_id,
            tools_used=[],
            model_used="test",
            fallback_used=False,
        )

    return behavior


# --- direct _session_id validation ------------------------------------------


def test_reserved_tg_prefix_rejected():
    with pytest.raises(HTTPException) as excinfo:
        _session_id("tg:42")
    assert excinfo.value.status_code == 422


def test_reserved_tg_dedup_prefix_rejected():
    with pytest.raises(HTTPException) as excinfo:
        _session_id("tg-dedup:42")
    assert excinfo.value.status_code == 422


def test_reserved_prefix_rejected_after_surrounding_whitespace():
    """Validation runs on the stripped value: padding cannot smuggle a
    reserved id through (it would strip to a reserved thread id)."""
    with pytest.raises(HTTPException) as excinfo:
        _session_id("  tg:42  ")
    assert excinfo.value.status_code == 422
    with pytest.raises(HTTPException):
        _session_id("\t tg-dedup:42 \n")


def test_prefix_lookalikes_still_allowed():
    """Only the exact reserved prefixes are rejected; strings that merely
    CONTAIN the reserved tokens, or that start with 'tg' (no colon), stay
    valid so no pre-existing naming style regresses."""
    for okay in ("tg", "tg42", "tgt:1", "x-tg:1", "tg-lane:7", "session-1"):
        assert _session_id(okay) == okay


def test_ordinary_valid_session_ids_unchanged():
    assert _session_id("s-gem-ok") == "s-gem-ok"
    assert _session_id(" s-multi ") == " s-multi "  # passthrough, unstripped
    assert _session_id(None) == "__single_turn__"
    assert _session_id("") == "__single_turn__"
    assert _session_id("   ") == "__single_turn__"


# --- endpoint-level contract -------------------------------------------------


@pytest.fixture()
def chat_harness(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    monkeypatch.setattr("app.chat.run_chat", make_inbound_call(calls))
    with TestClient(create_app()) as test_client:
        yield test_client, calls


def test_endpoint_rejects_tg_prefix_with_422(chat_harness):
    test_client, calls = chat_harness
    response = test_client.post(
        CHAT_URL, json={"message": "hi", "session_id": "tg:42"}
    )
    assert response.status_code == 422
    assert calls == []  # the agent must not run for a rejected id
    assert "reserved" in response.text


def test_endpoint_rejects_tg_dedup_prefix_with_422(chat_harness):
    test_client, calls = chat_harness
    response = test_client.post(
        CHAT_URL, json={"message": "hi", "session_id": "tg-dedup:42"}
    )
    assert response.status_code == 422
    assert calls == []


def test_endpoint_error_body_never_leaks_reserved_thread_ids(chat_harness):
    """The 422 body must state the reservation rule without echoing a raw
    signed ``tg-dedup:{id}`` thread name as usable advice."""
    test_client, _ = chat_harness
    response = test_client.post(
        CHAT_URL, json={"message": "hi", "session_id": "tg:7"}
    )
    assert response.status_code == 422
    assert "tg:7" not in response.text


def test_endpoint_accepts_ordinary_session_ids(chat_harness):
    test_client, calls = chat_harness
    response = test_client.post(
        CHAT_URL, json={"message": "What hats do you sell?", "session_id": "s-ok"}
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "s-ok"
    assert calls == ["s-ok"]


def test_endpoint_accepts_missing_session_id(chat_harness):
    test_client, calls = chat_harness
    response = test_client.post(CHAT_URL, json={"message": "hi"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "__single_turn__"
    assert calls == ["__single_turn__"]
