"""Phase 5 tests: Shopify authentication (client-credentials grant).

All HTTP is intercepted with `httpx.MockTransport` — no test touches the
network or uses real credentials. Synthetic credentials are clearly marked
(`fake-...`) and are not real secrets.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import httpx
import pytest

from app.config import ShopifySettings
from app.shopify_auth import (
    REFRESH_MARGIN_SECONDS,
    AccessToken,
    ShopifyTokenManager,
)
from app.shopify_errors import ShopifyAuthError

CC_SETTINGS = ShopifySettings(
    store_domain="test-store.myshopify.com",
    api_version="2026-10",
    client_id="fake-client-id",       # synthetic, not a credential
    client_secret="fake-client-secret",  # synthetic, not a credential
)

TOKEN_URL = "https://test-store.myshopify.com/admin/oauth/access_token"


def _token_response(expires_in: int | None = 86399) -> dict:
    body = {"access_token": "shppa_fake_access_token", "scope": "read_products"}
    if expires_in is not None:
        body["expires_in"] = expires_in
    return body


def _make_manager(
    handler: Callable[[httpx.Request], httpx.Response],
) -> ShopifyTokenManager:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return ShopifyTokenManager(CC_SETTINGS, http_client)


def test_acquires_token_via_client_credentials_grant():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["content_type"] = request.headers["Content-Type"]
        captured["form"] = dict(part.split("=", 1) for part in request.content.decode().split("&"))
        return httpx.Response(200, json=_token_response())

    manager = _make_manager(handler)
    token = manager.get_access_token()

    assert token == "shppa_fake_access_token"
    assert captured["url"] == TOKEN_URL
    assert captured["content_type"].startswith("application/x-www-form-urlencoded")
    assert captured["form"]["grant_type"] == "client_credentials"
    assert captured["form"]["client_id"] == "fake-client-id"
    assert captured["form"]["client_secret"] == "fake-client-secret"


def test_token_is_cached_until_expiry():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_token_response())

    manager = _make_manager(handler)
    first = manager.get_access_token()
    second = manager.get_access_token()

    assert first == second == "shppa_fake_access_token"
    assert calls["count"] == 1  # no second HTTP call while cached


def test_token_refreshes_near_expiry():
    """Below the refresh margin the manager must proactively re-acquire."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_token_response())

    manager = _make_manager(handler)
    manager.get_access_token()  # initial acquire

    # Age the cached token so its remaining lifetime falls below the
    # proactive-refresh margin: the next get must re-acquire over HTTP.
    manager._cached = AccessToken(  # type: ignore[union-attr]
        value="shppa_fake_access_token",
        expires_at=time.monotonic() + REFRESH_MARGIN_SECONDS - 10,
    )

    token = manager.get_access_token()

    assert token == "shppa_fake_access_token"
    assert calls["count"] == 2


def test_invalid_credentials_raise_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": "invalid_client", "error_description": "nope"}
        )

    manager = _make_manager(handler)
    with pytest.raises(ShopifyAuthError) as excinfo:
        manager.get_access_token()
    assert "SHOPIFY_CLIENT_ID" in str(excinfo.value)
    # Secrets never appear in error messages.
    assert "fake-client-secret" not in str(excinfo.value)


def test_missing_access_token_in_response_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"expires_in": 86399})

    manager = _make_manager(handler)
    with pytest.raises(ShopifyAuthError) as excinfo:
        manager.get_access_token()
    assert "access_token" in str(excinfo.value)


def test_static_token_mode_skips_http():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_token_response())

    static_settings = ShopifySettings(
        store_domain="test-store.myshopify.com",
        api_version="2026-10",
        admin_api_token="static-shppa-token",
    )
    manager = ShopifyTokenManager(
        static_settings, httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert manager.mode == "static_token"
    assert manager.get_access_token() == "static-shppa-token"
    assert calls["count"] == 0  # no token endpoint hit in static mode


def test_invalidated_token_reacquires():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json=_token_response())

    manager = _make_manager(handler)
    manager.get_access_token()
    manager.invalidate_cached_token()
    manager.get_access_token()

    assert calls["count"] == 2


def test_access_token_repr_redacts_value():
    token = AccessToken(value="shppa_fake_access_token", expires_at=time.monotonic() + 10)
    assert "shppa_fake_access_token" not in repr(token)
