"""Shared Shopify error hierarchy (imported by both auth and client modules).

Split out so `shopify_auth` and `shopify_client` can share exceptions without
a circular import. Error messages never contain secret values: tokens are
only ever referenced as header/credentials names, never their contents.
"""
from __future__ import annotations


class ShopifyClientError(Exception):
    """Base class for Shopify client errors (message never contains secrets)."""


class ShopifyAuthError(ShopifyClientError):
    """Authentication/authorization failed (401/403): token, credentials, or scopes invalid."""


class ShopifyThrottledError(ShopifyClientError):
    """Shopify leaky-bucket throttle; optionally carries a retry hint (seconds)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ShopifyTimeoutError(ShopifyClientError):
    """Request timed out before Shopify responded."""


class ShopifyUpstreamError(ShopifyClientError):
    """Shopify returned an unexpected status or an unusable payload."""
