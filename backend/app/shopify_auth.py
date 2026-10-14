"""Shopify Admin API token handling (Phase 5).

Two supported authentication modes:
- **client_credentials**: acquires Admin API access tokens from Shopify using
  `SHOPIFY_CLIENT_ID`/`SHOPIFY_CLIENT_SECRET` via the officially documented
  client-credentials grant:

      POST https://{store}.myshopify.com/admin/oauth/access_token
      grant_type=client_credentials&client_id=...&client_secret=...

  The token is valid for ~24 hours (`expires_in` is 86399 seconds per
  shopify.dev). Tokens are cached in memory and refreshed proactively before
  expiry (plus immediately after an upstream 401).
- **static_token**: uses the custom-app Admin API access token directly.

Secrets (client secret, tokens) live only in request bodies/headers internal
to this module and are never logged, printed, or included in error messages.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import httpx

from app.config import ShopifySettings
from app.shopify_errors import (
    ShopifyAuthError,
    ShopifyTimeoutError,
    ShopifyUpstreamError,
)

# Refresh a cached token this many seconds before its reported expiry.
REFRESH_MARGIN_SECONDS = 300.0


@dataclass(frozen=True)
class AccessToken:
    """A short-lived Admin API access token with a monotonic expiry."""

    value: str
    expires_at: float  # time.monotonic() based

    @property
    def seconds_remaining(self) -> float:
        return self.expires_at - time.monotonic()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"AccessToken(value=<redacted>, "
            f"seconds_remaining={self.seconds_remaining:.0f})"
        )


class ShopifyTokenManager:
    """Acquires, caches, and refreshes Admin API access tokens."""

    def __init__(
        self,
        settings: ShopifySettings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._http = http_client or httpx.Client(
            timeout=settings.timeout_seconds
        )
        self._cached: AccessToken | None = None
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._settings.auth_mode

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing beforehand if near expiry."""
        if self._settings.auth_mode == "static_token":
            return self._settings.admin_api_token
        with self._lock:
            cached = self._cached
            # Proactive refresh: re-acquire once the remaining lifetime falls
            # within the margin, so in-flight calls never ride an expiring
            # token down to zero.
            if cached is not None and cached.seconds_remaining > REFRESH_MARGIN_SECONDS:
                return cached.value
            fresh = self._acquire_client_credentials_token()
            self._cached = fresh
            return fresh.value

    def invalidate_cached_token(self) -> None:
        """Drop the cached token (e.g. after an upstream 401) so the next call
        re-authenticates. Safe to call in any mode."""
        with self._lock:
            self._cached = None

    # -- internals ----------------------------------------------------------

    def _acquire_client_credentials_token(self) -> AccessToken:
        """Perform the client-credentials grant against Shopify."""
        body = {
            "grant_type": "client_credentials",
            "client_id": self._settings.client_id,
            "client_secret": self._settings.client_secret,
        }
        try:
            response = self._http.post(
                self._token_url,
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data=body,
            )
        except httpx.TimeoutException as exc:
            raise ShopifyTimeoutError(
                f"Token request timed out after {self._settings.timeout_seconds:g}s."
            ) from exc
        except httpx.RequestError as exc:
            raise ShopifyAuthError(
                "Failed to reach Shopify for token acquisition (network error)."
            ) from exc

        status = response.status_code
        if status in (400, 401, 403):
            # Shopify returns 400/401 for invalid client credentials; 403 when
            # the app lacks direct API access. Includes reference to the
            # variables to check, never their values.
            raise ShopifyAuthError(
                f"Shopify rejected the client-credentials token request "
                f"(HTTP {status}). Check SHOPIFY_CLIENT_ID and "
                f"SHOPIFY_CLIENT_SECRET, and that the app has Admin API "
                f"access enabled in the Shopify dashboard."
            )
        if status >= 500:
            raise ShopifyUpstreamError(
                f"Shopify server error during token acquisition (HTTP {status})."
            )
        if status != 200:
            raise ShopifyAuthError(
                f"Unexpected Shopify status during token acquisition (HTTP {status})."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ShopifyAuthError(
                "Shopify token response was not valid JSON."
            ) from exc

        token = (payload or {}).get("access_token")
        if not token:
            raise ShopifyAuthError(
                "Shopify token response is missing 'access_token'."
            )
        try:
            expires_in = float((payload or {}).get("expires_in") or 86400)
        except (TypeError, ValueError):
            expires_in = 86400.0

        # Store the token's true expiry; the refresh margin policy lives in
        # get_access_token. Ensure at least a small usable lifetime.
        lifetime = max(expires_in, 1.0)
        return AccessToken(value=token, expires_at=time.monotonic() + lifetime)

    @property
    def _token_url(self) -> str:
        return f"https://{self._settings.store_domain}/admin/oauth/access_token"
