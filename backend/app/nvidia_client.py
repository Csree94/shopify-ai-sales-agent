"""Thin NVIDIA Nemotron service (OpenAI-compatible) — generation-only fallback.

Design notes:
- The `OpenAI` client is built from validated `NvidiaSettings`; the API key
  lives only in memory and is never logged, echoed, or stored anywhere.
- `generate_completion()` performs exactly one chat completion with NO SDK
  retries (`max_retries=0`, matching the project's no-auto-retry rule) and a
  bounded timeout.
- This module has NO tool support by design: the fallback receives the same
  user question and the verified Shopify context the primary attempt
  collected, and produces text only — it can never query Shopify itself or
  invent a second data source.
- Tests inject a MockTransport-backed `httpx.Client` via `http_client`,
  which the OpenAI SDK explicitly accepts.
"""

from __future__ import annotations

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    OpenAIError,
)

from app.config import NvidiaSettings

DEFAULT_NVIDIA_TIMEOUT_SECONDS = 30.0
MAX_NVIDIA_TIMEOUT_SECONDS = 60.0


class NvidiaError(Exception):
    """Base class for NVIDIA fallback errors (messages never contain secrets)."""


class NvidiaAuthError(NvidiaError):
    """Invalid/missing NVIDIA API key (HTTP 401/403)."""


class NvidiaUnavailableError(NvidiaError):
    """Throttled/server failure (HTTP 429/5xx) or network error."""


class NvidiaTimeoutError(NvidiaError):
    """Request timed out."""


class NvidiaRequestError(NvidiaError):
    """Bad request or unusable response."""


def validate_timeout(seconds: float) -> float:
    if seconds <= 0 or seconds > MAX_NVIDIA_TIMEOUT_SECONDS:
        raise ValueError(
            "NVIDIA timeout must be between 0 and "
            f"{MAX_NVIDIA_TIMEOUT_SECONDS:g} seconds."
        )
    return seconds


def make_nvidia_client(
    settings: NvidiaSettings,
    *,
    httpx_client: httpx.Client | None = None,
    timeout_seconds: float = DEFAULT_NVIDIA_TIMEOUT_SECONDS,
) -> OpenAI:
    """Build the NVIDIA OpenAI-compatible client from validated settings."""
    validate_timeout(timeout_seconds)
    kwargs: dict = {
        "base_url": settings.base_url,
        "api_key": settings.api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,  # project rule: no automatic retries
    }
    if httpx_client is not None:
        kwargs["http_client"] = httpx_client
    return OpenAI(**kwargs)


def _translate(exc: Exception) -> NvidiaError:
    """Translate EXPECTED provider-side failures (openai/httpx errors).

    Unexpected programming errors are not caught by this module's wrapper —
    they propagate so bugs stay visible instead of being reported as
    provider outages.
    """
    if isinstance(exc, APITimeoutError):
        return NvidiaTimeoutError("NVIDIA fallback request timed out.")
    if isinstance(exc, APIConnectionError):
        return NvidiaUnavailableError("Failed to reach the NVIDIA fallback endpoint.")
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None) or 0
        if status in (401, 403):
            return NvidiaAuthError("NVIDIA rejected the API key (auth failure).")
        if status == 429:
            return NvidiaUnavailableError(
                "NVIDIA rate-limited the fallback request (429)."
            )
        if status >= 500:
            return NvidiaUnavailableError(
                f"NVIDIA server error (HTTP {status})."
            )
        return NvidiaRequestError(
            f"NVIDIA rejected the fallback request (HTTP {status})."
        )
    return NvidiaRequestError("NVIDIA fallback request failed (parse error).")


def generate_completion(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_question: str,
    verified_context_json: str,
) -> str:
    """One generation-only completion; returns the reply text or raises.

    `verified_context_json` is the exact Shopify data snapshot the primary
    attempt collected (already validated by the project's own normalisers).
    There is deliberately no `tools` argument here: the fallback cannot query
    Shopify, so grounding depends entirely on the passed snapshot — an empty
    snapshot must never reach this function (agent-level gate).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Customer question: {user_question}\n\n"
                "Verified Shopify context (the ONLY product/stock data you may "
                "use; values marked null are unknown — say so, do not invent "
                "products, prices, variants, SKUs or stock):\n"
                f"{verified_context_json}"
            ),
        },
    ]
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except (OpenAIError, httpx.HTTPError) as exc:
        # Only expected provider-side failures are translated; anything
        # else (a programming bug) propagates unchanged.
        raise _translate(exc) from exc

    choice = completion.choices[0] if getattr(completion, "choices", None) else None
    if choice is None:
        raise NvidiaRequestError("NVIDIA fallback returned no choices.")
    text = getattr(choice.message, "content", None)
    if not (text or "").strip():
        raise NvidiaRequestError("NVIDIA fallback returned an empty answer.")
    return text
