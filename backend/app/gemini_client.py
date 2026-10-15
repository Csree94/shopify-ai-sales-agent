"""Thin Gemini (Google AI) service for the Phase 8 sales agent.

Design notes:
- The `genai.Client` is built from validated `GeminiSettings` (see
  `app.config.load_llm_settings`); the API key lives only in memory and is
  never logged, echoed, or stored anywhere.
- `generate_once()` performs exactly ONE API round-trip and reports raw
  outcomes. Tool selection is surfaced as `function_calls` with automatic
  function calling disabled, so the agent (not the SDK) executes the
  read-only Shopify tools and controls the loop.
- Errors are translated into the small local exception set below; the
  upstream HTTP status is preserved on `status` so the agent can decide
  fallback eligibility (429/5xx/timeout → eligible; auth/4xx → not).
- Tests inject a plain `httpx.Client` (mock transport) via `http_options`
  — the installed SDK accepts it through `HttpOptions.httpxClient`.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as gtypes

from app.config import GeminiSettings

DEFAULT_GEMINI_TIMEOUT_SECONDS = 30.0
MAX_GEMINI_TIMEOUT_SECONDS = 60.0


class GeminiError(Exception):
    """Base class for Gemini service errors (messages never contain secrets)."""


class GeminiAuthError(GeminiError):
    """Invalid/missing API key (HTTP 401/403): NOT fallback-eligible."""


class GeminiUnavailableError(GeminiError):
    """Throttled or upstream/server failure (HTTP 429/5xx): fallback-eligible."""


class GeminiTimeoutError(GeminiError):
    """Request timed out: fallback-eligible."""


class GeminiRequestError(GeminiError):
    """Bad request or unusable response (other 4xx/parse): NOT fallback-eligible."""


class GeminiSafetyRefusal(GeminiRequestError):
    """Model declined to answer (finishReason SAFETY/BLOCK): NOT fallback-eligible."""


@dataclass
class GeminiTurn:
    """One Gemini round-trip outcome.

    Exactly one of `text` / `function_calls` is populated for STOP finishes.
    """

    text: str | None = None
    # Signature carried on the replayed model-text turn (Gemini 3.x baseline
    # requirement; bytes as returned by the SDK, or None).
    thought_signature: bytes | None = None
    function_calls: list[dict] | None = None
    finish_reason: str | None = None


def validate_timeout(seconds: float) -> float:
    if seconds <= 0 or seconds > MAX_GEMINI_TIMEOUT_SECONDS:
        raise ValueError(
            "Gemini timeout must be between 0 and "
            f"{MAX_GEMINI_TIMEOUT_SECONDS:g} seconds."
        )
    return seconds


def make_gemini_client(
    settings: GeminiSettings,
    *,
    httpx_client: httpx.Client | None = None,
    timeout_seconds: float = DEFAULT_GEMINI_TIMEOUT_SECONDS,
) -> genai.Client:
    """Build a Gemini Developer API client from validated settings.

    `httpx_client` is the documented test seam: a MockTransport-backed
    httpx.Client keeps unit tests fully offline.

    UNIT NOTE (verified against the installed SDK source): `HttpOptions.timeout`
    is in MILLISECONDS — the SDK divides it by 1000 before handing it to
    httpx. Our public surface is seconds (consistent with the Shopify
    timeout settings), so the conversion happens here, exactly once.

    NO-RETRY NOTE (verified against the installed SDK source): without an
    explicit `HttpRetryOptions`, the SDK's `HttpOptions.retry_options`
    default is None, which today also means "never retry" — but a future
    SDK update could silently flip that default to auto-retry repeats,
    which the project's no-retry rule forbids. We pin `attempts=1`
    explicitly so the behavior stays deterministic.
    """
    validate_timeout(timeout_seconds)
    # The SDK expects milliseconds for HttpOptions.timeout (see docstring).
    http_options: dict = {
        "timeout": timeout_seconds * 1000.0,
        "retryOptions": gtypes.HttpRetryOptions(attempts=1),
    }
    if httpx_client is not None:
        http_options["httpxClient"] = httpx_client
    return genai.Client(
        api_key=settings.api_key,
        http_options=gtypes.HttpOptions(**http_options),
    )


def _api_error_detail(exc: Exception) -> str:
    """Sanitized single-line detail from the SDK's structured APIError.

    The installed SDK raises ``google.genai.errors.APIError`` (or
    ClientError/ServerError) with ``.status`` (the payload's status string,
    e.g. 'INVALID_ARGUMENT') and ``.message`` (the one-line human-readable
    error description). Both are safe, bounded fields — unlike ``.details``,
    which holds the FULL response body and is never read here. Used so a
    request-rejection (HTTP 400) names the rejected field/cause project-side
    instead of surfacing as an unexplained status code.
    """
    status_str = str(getattr(exc, "status", None) or "").strip()
    message = str(getattr(exc, "message", None) or "").strip()
    # Defense in depth: never echo multi-line provider text; keep it short.
    message = " ".join(message.split())
    if len(message) > 200:
        message = message[:200] + "…"
    parts = [p for p in (status_str, message) if p]
    return f" [{': '.join(parts)}]" if parts else ""


def _is_native_google_payload(details) -> bool:
    """True when the ≥500 response body has the shape of a NATIVE Google
    error payload (JSON with an 'error' mapping), as opposed to an opaque
    page an intermediary (proxy/firewall) might inject. Only the shape is
    inspected — the body content is never printed, logged, or stored."""
    return (
        isinstance(details, dict)
        and isinstance(details.get("error"), dict)
    )


def _translate(exc: Exception) -> GeminiError:
    """Map google.genai.errors onto the local hierarchy via the HTTP status.

    In the installed SDK, the HTTP status code lives on ``.code`` (int);
    ``.status`` is the response *payload's* status string (e.g.
    'RESOURCE_EXHAUSTED') and is often None — do not read it for the code.

    For ≥500 failures the returned error carries a
    ``native_google_payload`` flag (shape-only classification of the
    response body: True = the 503 came from Google's own error format,
    False = likely an intermediary). The body content itself is discarded.

    For other 4xx rejections the message includes the SDK's structured
    ``.status`` string and sanitized one-line ``.message`` so the actual
    bad-request cause is diagnosable without exposing any body or secret.
    """
    translated: GeminiError
    native = False
    if isinstance(exc, genai_errors.APIError):
        try:
            status = int(getattr(exc, "code", None))
        except (TypeError, ValueError):
            status = 0
        native = _is_native_google_payload(getattr(exc, "details", None))
        if status in (401, 403):
            translated = GeminiAuthError(
                "Gemini rejected the API key (auth failure)."
            )
        elif status == 429:
            translated = GeminiUnavailableError(
                "Gemini rate-limited the request (429)."
            )
        elif status >= 500:
            translated = GeminiUnavailableError(
                f"Gemini server error (HTTP {status})"
                + (" (native Google error payload)." if native else
                   " (response body did not have Google's error shape — "
                   "possibly an intermediary).")
            )
        else:
            translated = GeminiRequestError(
                f"Gemini rejected the request (HTTP {status})."
                f"{_api_error_detail(exc)}"
            )
    else:
        if isinstance(exc, httpx.TimeoutException):
            translated = GeminiTimeoutError("Gemini request timed out.")
        else:
            translated = GeminiRequestError(
                "Gemini request failed (network/parse error)."
            )
    translated.native_google_payload = native
    return translated


def _finish_reason(response) -> str | None:
    try:
        candidate = response.candidates[0]
        return str(candidate.finish_reason).partition(".")[-1]
    except (IndexError, AttributeError):
        return None


def generate_once(
    client: genai.Client,
    *,
    model: str,
    system_prompt: str,
    contents: list,
    tool_declarations: list[gtypes.FunctionDeclaration],
) -> GeminiTurn:
    """One live round-trip; returns text, function calls, or raises.

    `contents` is a list of `gtypes.Content`-shaped dicts (role + parts) the
    agent maintains, so multi-turn state lives in the graph, not the SDK.

    NO-SAMPLING NOTE (per Google's authoritative migration guide for
    gemini-3.6-flash and all later models, ai.google.dev/gemini-api/docs/
    whats-new-gemini-3.6): temperature/top_p/top_k are deprecated — the API
    ignores them and may reject them (HTTP 400) in current and future
    generations — and candidate_count is unsupported on Gemini 3+. They are
    therefore never sent; determinism comes from the strict system prompt
    (answers must come only from verified Shopify tool data).
    """
    config = gtypes.GenerateContentConfig(
        system_instruction=system_prompt,
        automatic_function_calling=gtypes.AutomaticFunctionCallingConfig(
            disable=True
        ),
        tools=[gtypes.Tool(function_declarations=list(tool_declarations))],
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _translate(exc) from exc

    finish_reason = _finish_reason(response)
    if finish_reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
        raise GeminiSafetyRefusal(
            f"Gemini declined to answer (finishReason: {finish_reason})."
        )

    try:
        parts = response.candidates[0].content.parts
    except (IndexError, AttributeError, TypeError) as exc:
        raise GeminiRequestError("Gemini returned no usable parts.") from exc

    text: str | None = None
    text_signature = None
    calls: list[dict] = []
    for part in parts or []:
        signature = getattr(part, "thought_signature", None)
        fc = getattr(part, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            call = {"name": fc.name, "args": dict(getattr(fc, "args", {}) or {})}
            if signature:
                # Gemini 3.x baseline: signatures on function-call parts
                # MUST be preserved and replayed with the turn they came
                # from (ai.google.dev/gemini-api/docs thinking guide),
                # otherwise later model calls can be rejected.
                call["thought_signature"] = signature
            calls.append(call)
        elif getattr(part, "text", None):
            text = (text or "") + part.text
            if signature:
                text_signature = signature

    if calls:
        return GeminiTurn(function_calls=calls, finish_reason=finish_reason)
    if finish_reason == "STOP" and text:
        return GeminiTurn(
            text=text, thought_signature=text_signature, finish_reason=finish_reason
        )
    if text:
        # Non-STOP text (e.g. MAX_TOKENS): still safe to show, but flagged.
        return GeminiTurn(
            text=text, thought_signature=text_signature, finish_reason=finish_reason
        )
    raise GeminiRequestError(
        f"Gemini returned neither text nor function calls (finishReason: {finish_reason})."
    )
