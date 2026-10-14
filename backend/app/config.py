"""Environment configuration for the Shopify AI Sales Agent backend.

Values are read from environment variables at call time (not at import
time) so tests can inject per-case environments. Local `backend/.env` (and
root `.env`) files are loaded silently via python-dotenv without printing
any values. Secrets are never logged or printed: the settings repr redacts
all credential fields.

Authentication modes (Phase 5):
- ``client_credentials`` — default when SHOPIFY_CLIENT_ID and
  SHOPIFY_CLIENT_SECRET are both set; tokens are acquired dynamically.
- ``static_token`` — legacy/custom-app Admin API access token
  (SHOPIFY_ADMIN_ACCESS_TOKEN), used when client credentials are absent.
  Placeholder-looking values are treated as "not configured".
No settings are hardcoded beyond safe defaults (timeout, supported API
version list). Real credentials only ever live in the git-ignored `.env`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

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
    admin_api_token: str = ""
    client_id: str = ""
    client_secret: str = ""
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @property
    def auth_mode(self) -> str:
        """Which authentication flow to use (decided in load_shopify_settings)."""
        return "client_credentials" if self.client_id else "static_token"

    @property
    def graphql_url(self) -> str:
        return f"https://{self.store_domain}/admin/api/{self.api_version}/graphql.json"

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"ShopifySettings(store_domain={self.store_domain!r}, "
            f"api_version={self.api_version!r}, "
            f"auth_mode={self.auth_mode!r}, "
            f"admin_api_token=<redacted>, client_id=<redacted>, "
            f"client_secret=<redacted>, "
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


def _looks_like_placeholder(value: str) -> bool:
    """Heuristic for template values ('placeholder_get_from_dev_dashboard'
    and similar) so placeholder-only environments fail with a helpful
    message instead of sending a placeholder token to Shopify."""
    lowered = value.lower()
    return (
        "placeholder" in lowered
        or lowered.startswith("your_")
        or lowered == "generate_a_random_string_for_hmac_check"
    )


def load_local_env_files(paths: list[Path] | None = None) -> list[str]:
    """Load local .env files via python-dotenv, silently.

    Existing process environment variables always win (override=False), so
    runtime invocations can still override file contents. Never prints any
    values. Returns the list of files that were found and loaded.
    """
    if paths is None:
        base = Path(__file__).resolve().parent.parent  # backend/
        paths = [base / ".env", base.parent / ".env"]  # backend/.env, root .env
    loaded: list[str] = []
    for path in paths:
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)
            loaded.append(str(path))
    return loaded


def load_shopify_settings(env: dict[str, str] | None = None) -> ShopifySettings:
    """Build settings from the given mapping (default: os.environ).

    When reading the real environment, backend/.env and root .env are loaded
    first (silently, without clobbering existing variables). Raises
    ConfigurationError naming the missing/invalid variable but never
    including any secret value.
    """
    if env is None:
        load_local_env_files()
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

    client_id = (source.get("SHOPIFY_CLIENT_ID") or "").strip()
    client_secret = (source.get("SHOPIFY_CLIENT_SECRET") or "").strip()
    admin_token_raw = (source.get("SHOPIFY_ADMIN_ACCESS_TOKEN") or "").strip()
    admin_token = "" if _looks_like_placeholder(admin_token_raw) else admin_token_raw

    if client_id and not client_secret:
        raise ConfigurationError(
            "SHOPIFY_CLIENT_SECRET is missing or empty; client-credentials "
            "authentication needs both SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET."
        )
    if client_secret and not client_id:
        raise ConfigurationError(
            "SHOPIFY_CLIENT_ID is missing or empty; client-credentials "
            "authentication needs both SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET."
        )

    settings = ShopifySettings(
        store_domain=domain,
        api_version=version,
        admin_api_token=admin_token,
        client_id=client_id,
        client_secret=client_secret,
        timeout_seconds=_parse_timeout(source),
    )

    if not (client_id or admin_token):
        raise ConfigurationError(
            "No Shopify credentials configured. Provide either both "
            "SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET (client-credentials "
            "grant, preferred) or a real SHOPIFY_ADMIN_ACCESS_TOKEN (custom "
            "app Admin API token) in your local .env (never commit it)."
        )

    return settings


def _parse_timeout(source: dict[str, str]) -> float:
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
    return timeout


def _supported_versions() -> tuple[str, ...]:
    """Stable versions currently served by Shopify (checked 2026-10-14)."""
    return ("2026-01", "2026-04", "2026-07", "2026-10")
