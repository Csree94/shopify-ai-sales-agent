"""Environment configuration for Shopify Admin API access.

Values are read from environment variables at call time (not at import
time) so tests can inject per-case environments. Secrets are never logged
or printed: the settings repr redacts the Admin API access token.

No settings are hardcoded beyond safe defaults (timeout, supported API
version list). Real credentials only ever live in the git-ignored `.env`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit

# Shopify releases a new version every 3 months; the 4 stable versions
# currently supported as of 2026-10 (per shopify.dev/docs/api/usage/versioning).
DEFAULT_API_VERSION = "2026-10"
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_GQL_PAGE_SIZE = 250


class ConfigurationError(RuntimeError):
    """Raised when required Shopify configuration is missing or invalid."""


@dataclass(frozen=True)
class ShopifySettings:
    """Validated Shopify Admin API connection settings."""

    store_domain: str
    api_version: str
    admin_api_token: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def graphql_url(self) -> str:
        return f"https://{self.store_domain}/admin/api/{self.api_version}/graphql.json"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ShopifySettings(store_domain={self.store_domain!r}, "
            f"api_version={self.api_version!r}, "
            f"admin_api_token=<redacted>, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )


def _extract_domain(raw: str) -> str:
    """Normalise a store domain, tolerating a pasted full admin URL."""
    candidate = raw.strip()
    if "://" in candidate:
        parts = urlsplit(candidate)
        candidate = (parts.hostname or "").strip(".")
    if candidate.startswith("www."):
        candidate = candidate[4:]
    if not candidate.lower().endswith(".myshopify.com"):
        raise ConfigurationError(
            "SHOPIFY_STORE_DOMAIN must be a *.myshopify.com domain "
            "(Admin API calls must target the myshopify domain)."
        )
    return candidate.lower()


def load_shopify_settings(env: dict[str, str] | None = None) -> ShopifySettings:
    """Build settings from the given mapping (default: os.environ).

    Raises ConfigurationError naming the missing/invalid variable but never
    including any secret value.
    """
    source = os.environ if env is None else env

    domain_raw = (source.get("SHOPIFY_STORE_DOMAIN") or "").strip()
    if not domain_raw:
        raise ConfigurationError(
            "SHOPIFY_STORE_DOMAIN is not set (expected e.g. your-store.myshopify.com)."
        )
    domain = _extract_domain(domain_raw)

    version = (source.get("SHOPIFY_API_VERSION") or DEFAULT_API_VERSION).strip()
    supported = _supported_versions()
    if version not in supported:
        raise ConfigurationError(
            f"SHOPIFY_API_VERSION {version!r} is not a currently supported "
            f"version. Supported: {', '.join(supported)} (latest stable: "
            f"{supported[-1]})."
        )

    token = (source.get("SHOPIFY_ADMIN_ACCESS_TOKEN") or "").strip()
    if not token:
        raise ConfigurationError(
            "SHOPIFY_ADMIN_ACCESS_TOKEN is not set. Create a Shopify custom app "
            "with the read_products scope and paste its Admin API access token "
            "into your local .env (never commit it)."
        )

    timeout_raw = (source.get("SHOPIFY_HTTP_TIMEOUT_SECONDS") or "").strip()
    try:
        timeout = float(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT_SECONDS
    except ValueError as exc:
        raise ConfigurationError(
            "SHOPIFY_HTTP_TIMEOUT_SECONDS must be a number of seconds."
        ) from exc
    if timeout <= 0 or timeout > 60:
        raise ConfigurationError(
            "SHOPIFY_HTTP_TIMEOUT_SECONDS must be between 0 and 60 seconds."
        )

    return ShopifySettings(
        store_domain=domain,
        api_version=version,
        admin_api_token=token,
        timeout_seconds=timeout,
    )


def _supported_versions() -> tuple[str, ...]:
    """Stable versions currently served by Shopify (checked 2026-10-14)."""
    return ("2026-01", "2026-04", "2026-07", "2026-10")
