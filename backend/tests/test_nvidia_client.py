"""Offline NVIDIA (OpenAI-compatible) fallback tests — MockTransport only."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import NvidiaSettings
from app.nvidia_client import (
    NvidiaAuthError,
    NvidiaRequestError,
    NvidiaTimeoutError,
    NvidiaUnavailableError,
    generate_completion,
    make_nvidia_client,
)

NV_KEY = "test-nv-key-do-not-print"


def _settings() -> NvidiaSettings:
    return NvidiaSettings(
        api_key=NV_KEY,
        model="test-nemotron-01",
        base_url="https://integrate.api.nvidia.com/v1",
    )


def _completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_completion_text_is_returned_and_key_only_in_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion("Nemotron grounded answer."))

    client : OpenAI = make_nvidia_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )

    text = generate_completion(
        client,
        model="test-nemotron-01",
        system_prompt="sys",
        user_question="What hats are in stock?",
        verified_context_json='[{"tool":"product_stock","products":[]}]',
    )
    assert text == "Nemotron grounded answer."
    assert captured["auth"] == f"Bearer {NV_KEY}"
    # Key never in URL; question and context are sent inside the body.
    assert NV_KEY not in captured["url"]
    assert captured["body"]["messages"][1]["content"].startswith("Customer question:")
    assert "product_stock" in captured["body"]["messages"][1]["content"]


def test_no_tools_argument_is_ever_sent():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" not in body  # generation-only fallback contract
        return httpx.Response(200, json=_completion("ok"))

    client = make_nvidia_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    generate_completion(
        client,
        model="test-nemotron-01",
        system_prompt="sys",
        user_question="q",
        verified_context_json="[]",
    )


@pytest.mark.parametrize(
    ("status", "expected_cls"),
    [
        (401, NvidiaAuthError),
        (403, NvidiaAuthError),
        (429, NvidiaUnavailableError),
        (500, NvidiaUnavailableError),
        (400, NvidiaRequestError),
    ],
)
def test_http_status_errors_are_translated(status, expected_cls):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "boom"}})

    client = make_nvidia_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    with pytest.raises(expected_cls):
        generate_completion(
            client,
            model="test-nemotron-01",
            system_prompt="sys",
            user_question="q",
            verified_context_json="[]",
        )


def test_timeout_is_translated():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    client = make_nvidia_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    with pytest.raises(NvidiaTimeoutError):
        generate_completion(
            client,
            model="test-nemotron-01",
            system_prompt="sys",
            user_question="q",
            verified_context_json="[]",
        )


def test_empty_completion_is_a_request_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion("   "))

    client = make_nvidia_client(
        _settings(),
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout_seconds=5,
    )
    with pytest.raises(NvidiaRequestError):
        generate_completion(
            client,
            model="test-nemotron-01",
            system_prompt="sys",
            user_question="q",
            verified_context_json="[]",
        )


# OpenAI imported here to construct through the real factory only.
from openai import OpenAI  # noqa: E402, used for isinstance clarity


def test_factory_uses_no_retries():
    settings = _settings()
    client = make_nvidia_client(settings, timeout_seconds=5)
    assert client.max_retries == 0
    assert str(client.base_url).rstrip("/") == "https://integrate.api.nvidia.com/v1"

    assert client.api_key == NV_KEY  # exists in memory only
