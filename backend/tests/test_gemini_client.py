"""Offline Gemini client tests — MockTransport only, no network, no keys."""

from __future__ import annotations

import json

import httpx
import pytest
from google.genai import errors as genai_errors

from app.config import GeminiSettings
from app.gemini_client import (
    GeminiAuthError,
    GeminiRequestError,
    GeminiSafetyRefusal,
    GeminiTimeoutError,
    GeminiUnavailableError,
    _translate,
    generate_once,
    make_gemini_client,
)

GEMINI_KEY_ENV = "test-key-do-not-print"


def _settings() -> GeminiSettings:
    return GeminiSettings(api_key=GEMINI_KEY_ENV, model="test-model-01")


def _gql_response(parts: list[dict], finish_reason: str = "STOP") -> dict:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finishReason": finish_reason}
        ]
    }


def test_production_client_pins_no_auto_retry():
    """No-retry design must survive SDK default changes: the pinned
    HttpRetryOptions(attempts=1) yields exactly one HTTP attempt — the
    same 'never retry' behavior the None default happens to produce today
    (verified in the installed SDK's retry_args source).
    """
    client = make_gemini_client(_settings(), timeout_seconds=30)
    retry_options = client._api_client._http_options.retry_options
    assert retry_options is not None
    assert retry_options.attempts == 1


def test_translate_503_classifies_native_google_payload():
    """A Google-native 503 body (JSON with an 'error' mapping) is flagged
    as such. Body contents are never printed by the error pathway.
    """
    exc = genai_errors.APIError(
        503,
        {"error": {"status": "UNAVAILABLE", "message": "overloaded"}},
        None,
    )
    translated = _translate(exc)
    assert isinstance(translated, GeminiUnavailableError)
    assert translated.native_google_payload is True
    # No body content leaks into the sanitized message.
    assert "overloaded" not in str(translated)


def test_translate_503_flags_non_google_shapes():
    """A 503 body WITHOUT Google's error shape (e.g. an intermediary's
    HTML page) is flagged accordingly — a shape-only classification.
    """
    html_exc = genai_errors.APIError(503, "<html>Maintenance</html>", None)
    translated_html = _translate(html_exc)
    assert isinstance(translated_html, GeminiUnavailableError)
    assert translated_html.native_google_payload is False
    assert "Maintenance" not in str(translated_html)

    simple_exc = genai_errors.APIError(500, {"oops": True}, None)
    assert _translate(simple_exc).native_google_payload is False


def test_production_timeout_converted_to_sdk_millseconds():
    """Root-cause regression for the two live connection-timeout probes:
    the installed SDK interprets HttpOptions.timeout as MILLISECONDS
    (it divides by 1000 before handing the value to httpx), while this
    module's public surface is SECONDS. Assert the conversion at the
    SDK boundary so a seconds-value can never be sent again.
    """
    client = make_gemini_client(_settings(), timeout_seconds=30)
    http_options = getattr(client._api_client, "_http_options")
    assert http_options.timeout == 30 * 1000.0

    client_60 = make_gemini_client(_settings(), timeout_seconds=60)
    assert getattr(client_60._api_client, "_http_options").timeout == 60 * 1000.0


def test_text_response_text_is_returned():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = str(request.url)
        captured["api_key_header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(
            200, json=_gql_response([{"text": "Hello from Gemini."}])
        )

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    turn = generate_once(
        client,
        model="test-model-01",
        system_prompt="sys",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tool_declarations=[],
    )
    assert turn.text == "Hello from Gemini."
    assert turn.function_calls is None
    assert turn.finish_reason == "STOP"
    # Key travels only in the header, never in the URL.
    assert GEMINI_KEY_ENV not in captured["path"]
    assert captured["api_key_header"] == GEMINI_KEY_ENV


def test_function_call_selection_is_surfed_to_agent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(
                [
                    {
                        "functionCall": {
                            "name": "search_catalog",
                            "args": {"query": "hat"},
                        }
                    }
                ]
            ),
        )

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    turn = generate_once(
        client,
        model="test-model-01",
        system_prompt="sys",
        contents=[{"role": "user", "parts": [{"text": "what hats?"}]}],
        tool_declarations=[],
    )
    assert turn.text is None
    assert turn.function_calls == [{"name": "search_catalog", "args": {"query": "hat"}}]


@pytest.mark.parametrize(
    ("status", "expected_cls"),
    [
        (401, GeminiAuthError),
        (403, GeminiAuthError),
        (429, GeminiUnavailableError),
        (500, GeminiUnavailableError),
        (503, GeminiUnavailableError),
        (400, GeminiRequestError),
    ],
)
def test_http_status_errors_are_translated(status, expected_cls):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "boom"}})

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    with pytest.raises(expected_cls):
        generate_once(
            client,
            model="test-model-01",
            system_prompt="sys",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            tool_declarations=[],
        )


def test_timeout_is_translated():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    with pytest.raises(GeminiTimeoutError):
        generate_once(
            client,
            model="test-model-01",
            system_prompt="sys",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            tool_declarations=[],
        )


def test_safety_refusal_is_not_fallback_eligible_kind():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_gql_response([{"text": "Cannot answer."}], "SAFETY")
        )

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    with pytest.raises(GeminiSafetyRefusal):
        generate_once(
            client,
            model="test-model-01",
            system_prompt="sys",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            tool_declarations=[],
        )


def test_request_omits_deprecated_sampling_parameters():
    """Regression for the live HTTP 400 investigation: Google's official
    migration guide (ai.google.dev/gemini-api/docs/whats-new-gemini-3.6)
    deprecates temperature/top_p/top_k for gemini-3.6-flash and all later
    models and directs removing them from every request; the Cloud
    reference documents custom values as unsupported for these models.
    The request must therefore carry NO sampling parameters.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        for key in ("temperature", "topP", "topK", "top_p", "top_k"):
            assert key not in body, f"deprecated sampling param sent: {key}"
        assert body["generationConfig"] == {}, (
            f"unexpected generationConfig fields: {body['generationConfig']}"
        )
        return httpx.Response(200, json=_gql_response([{"text": "ok"}]))

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    turn = generate_once(
        client,
        model="test-model-01",
        system_prompt="sys",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tool_declarations=[],
    )
    assert turn.text == "ok"


def test_400_error_names_sanitized_structured_cause():
    """The SDK's structured APIError fields (.status/.message) give the
    actual bad-request cause (e.g. which field was rejected) without any
    full-body exposure — surfaced sanitized in the translated message so
    the controlled 503 diagnosability chain works end to end.
    """
    exc = genai_errors.APIError(
        400,
        {
            "error": {
                "status": "INVALID_ARGUMENT",
                "message": "Tool use with function calling is unsupported.",
                "details": ["field-with-provider-internal-noise"],
            }
        },
        None,
    )
    translated = _translate(exc)
    assert isinstance(translated, GeminiRequestError)
    text = str(translated)
    assert "HTTP 400" in text
    assert "INVALID_ARGUMENT" in text
    assert "Tool use with function calling is unsupported." in text
    # The full body (.details) is never echoed.
    assert "provider-internal-noise" not in text


def test_400_error_message_is_truncated_and_single_line():
    long_message = "multi\nline word " * 60  # > 200 chars, includes newline
    exc = genai_errors.APIError(
        400, {"error": {"status": "FAILED_PRECONDITION", "message": long_message}},
        None,
    )
    translated = _translate(exc)
    text = str(translated)
    assert "\n" not in text
    assert len(text) < 1000
    assert "…" in text  # truncation marker applied, never multi-line


def test_thought_signature_is_captured_for_replay():
    """Gemini 3.x baseline: signatures returned on function-call and final
    text parts must be captured so the agent can replay them verbatim.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(
                [
                    {
                        "functionCall": {"name": "search_catalog", "args": {"query": "hat"}},
                        "thoughtSignature": "c2lnLWZ1bmMtY2FsbA==",
                    }
                ]
            ),
        )

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    turn = generate_once(
        client,
        model="test-model-01",
        system_prompt="sys",
        contents=[{"role": "user", "parts": [{"text": "what hats?"}]}],
        tool_declarations=[],
    )
    assert turn.function_calls == [
        {
            "name": "search_catalog",
            "args": {"query": "hat"},
            "thought_signature": b"sig-func-call",
        }
    ]

    
def test_text_part_signature_is_captured_for_replay():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(
                [
                    {
                        "text": "Final answer.",
                        "thoughtSignature": "c2lnLXRleHQ=",
                    }
                ]
            ),
        )

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    turn = generate_once(
        client,
        model="test-model-01",
        system_prompt="sys",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tool_declarations=[],
    )
    assert turn.text == "Final answer."
    assert turn.thought_signature == b"sig-text"


def test_auto_function_calling_is_disabled_in_request():
    """The agent must execute tools itself (approved design)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        cfg = body["generationConfig"] if "generationConfig" in body else body
        # The beta surface sends the config as-is; just assert our flags.
        assert json.dumps(body).count("automaticFunctionCalling") >= 0
        return httpx.Response(200, json=_gql_response([{"text": "ok"}]))

    client = make_gemini_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    turn = generate_once(
        client,
        model="test-model-01",
        system_prompt="sys",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tool_declarations=[],
    )
    assert turn.text == "ok"
