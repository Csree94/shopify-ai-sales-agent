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

# Model default (verified against ai.google.dev model docs, 2026-10-15):
# gemini-3.6-flash is a Stable endpoint today, but per Google's lifecycle page
# it is a short-term availability model with an announced retirement window of
# 2026-11-19 (successor: gemini-3.8-flash, GA 2026-09-02 → ≥2027-09-02). The
# switch from gemini-3.8-flash was made for live-availability reasons pending
# a review of the 503s; revisit GEMINI_MODEL before 2026-11-19.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
# NVIDIA's OpenAI-compatible endpoint (integrate.api.nvidia.com/v1).
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class ConfigurationError(RuntimeError):
    """Raised when required Shopify configuration is missing or invalid."""


@dataclass(frozen=True)
class GeminiSettings:
    """Validated Gemini (Google AI) settings — Phase 8.

    The API key is treated exactly like the Shopify credentials: read from
    the environment at call time, redacted in repr, never logged. The model
    names are plain identifiers, safe to display.
    """

    api_key: str
    model: str = DEFAULT_GEMINI_MODEL

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"GeminiSettings(model={self.model!r}, api_key=<redacted>)"


@dataclass(frozen=True)
class NvidiaSettings:
    """Validated NVIDIA Nemotron settings — Phase 8 generation-only fallback.

    ``api_key`` may be empty when no fallback is configured; ``agent`` code
    must treat an empty key as 'fallback unavailable', never as an empty
    credential. The key is redacted in repr and never logged.
    """

    api_key: str = ""
    model: str = DEFAULT_NVIDIA_MODEL
    base_url: str = DEFAULT_NVIDIA_BASE_URL

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"NvidiaSettings(model={self.model!r}, base_url={self.base_url!r}, "
            "api_key=<redacted>)"
        )


@dataclass(frozen=True)
class LLMSettings:
    """Phase 8 LLM wiring used by /api/chat.

    ``fallback_enabled`` comes from ``LLM_FALLBACK_ENABLED`` (default true).
    Even when enabled, the fallback only runs if ``nvidia.api_key`` is set —
    availability of both is checked by the agent, not assumed here.
    """

    gemini: GeminiSettings
    nvidia: NvidiaSettings
    fallback_enabled: bool = True

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"LLMSettings(gemini={self.gemini!r}, nvidia={self.nvidia!r}, "
            f"fallback_enabled={self.fallback_enabled!r})"
        )


def load_llm_settings(
    env: dict[str, str] | None = None,
) -> LLMSettings:
    """Validate Gemini + optional NVIDIA fallback settings (never prints values)."""
    if env is None:
        load_local_env_files()
    source = os.environ if env is None else env

    gemini_key_raw = (source.get("GEMINI_API_KEY") or "").strip()
    gemini_key = "" if _looks_like_placeholder(gemini_key_raw) else gemini_key_raw
    if not gemini_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is not set (expected a Google AI Studio API key "
            "in your local .env — never commit it)."
        )

    gemini_model = (source.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()

    nvidia_api_key_raw = (source.get("NVIDIA_API_KEY") or "").strip()
    nvidia_api_key = (
        "" if _looks_like_placeholder(nvidia_api_key_raw) else nvidia_api_key_raw
    )
    nvidia = NvidiaSettings(
        api_key=nvidia_api_key,
        model=(source.get("NVIDIA_LLM_MODEL") or DEFAULT_NVIDIA_MODEL).strip(),
        base_url=(source.get("NVIDIA_LLM_BASE_URL") or DEFAULT_NVIDIA_BASE_URL).strip(),
    )

    fallback_raw = (source.get("LLM_FALLBACK_ENABLED") or "").strip().lower()
    if fallback_raw in ("", "1", "true", "yes", "on"):
        fallback_enabled = True
    elif fallback_raw in ("0", "false", "no", "off"):
        fallback_enabled = False
    else:
        raise ConfigurationError(
            "LLM_FALLBACK_ENABLED must be one of: 1/true/yes/on or 0/false/no/off."
        )

    return LLMSettings(
        gemini=GeminiSettings(api_key=gemini_key, model=gemini_model),
        nvidia=nvidia,
        fallback_enabled=fallback_enabled,
    )


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


# -- Phase M1: Neon PostgreSQL (persistent conversation checkpointing) -------


@dataclass(frozen=True)
class NeonSettings:
    """Validated Neon/PostgreSQL connection settings (Phase M1).

    ``dsn`` comes from ``NEON_DATABASE_URL`` and is a secret — it is never
    printed, logged, or echoed anywhere (repr/str redact it on purpose).
    ``agent.memory_db`` builds the checkpoint store from it; without it the
    app keeps the exact in-memory default.
    """

    dsn: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "NeonSettings(dsn=<redacted>)"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return "NeonSettings(dsn=<redacted>)"


def load_neon_settings(env: dict[str, str] | None = None) -> NeonSettings:
    """Validate the Neon checkpoint-store DSN (never prints values).

    Returns None when ``NEON_DATABASE_URL`` is unset or a placeholder, which
    callers turn into in-memory mode (local development / existing tests).
    """
    if env is None:
        load_local_env_files()
    source = os.environ if env is None else env

    dsn_raw = (source.get("NEON_DATABASE_URL") or "").strip()
    if not dsn_raw or _looks_like_placeholder(dsn_raw):
        return None
    if not dsn_raw.lower().startswith(("postgres://", "postgresql://")):
        raise ConfigurationError(
            "NEON_DATABASE_URL must be a postgres:// or postgresql:// "
            "connection string (value itself is never echoed)."
        )
    return NeonSettings(dsn=dsn_raw)
