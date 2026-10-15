"""Phase 8 chat-agent tests — all offline (MockTransport for every provider).

No test needs a real GEMINI/NVIDIA/Shopify credential: provider clients are
built through the production factories but backed by `httpx.MockTransport`,
and LLM environment values are synthetic `test-*` strings that must never
appear in any response.
"""

from __future__ import annotations

import json

import httpx
import psycopg
from fastapi.testclient import TestClient

import app.agent as agent
import app.memory_db as memory_db
import app.products as products
from app.config import ConfigurationError, load_llm_settings
from app.gemini_client import make_gemini_client
from langgraph.checkpoint.memory import InMemorySaver
from app.nvidia_client import make_nvidia_client

# Reuse the Shopify test fixtures of the existing suite.
from test_shopify_products import (
    _gql_response,
    _install_factory,
    _inventory_products_payload,
    _make_client,
)

GEM_KEY = "test-gem-key-e2e-do-not-print"
NV_KEY = "test-nv-key-e2e-do-not-print"

CHAT_URL = "/api/chat"


def _llm_env(
    monkeypatch, *, fallback_enabled: bool = True, nvidia_key: str = NV_KEY
) -> None:
    """Deterministic LLM env: setenv wins over .env (load_dotenv override=False)."""
    monkeypatch.setenv("GEMINI_API_KEY", GEM_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setenv("NVIDIA_API_KEY", nvidia_key)
    monkeypatch.setenv("NVIDIA_LLM_MODEL", "test-nemotron-01")
    monkeypatch.setenv("NVIDIA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "true" if fallback_enabled else "false")


def _hat_product(title: str = "Hat") -> dict:
    return {
        "id": f"gid://shopify/Product/{title}",
        "title": title,
        "totalInventory": 7,
        "variants": {
            "nodes": [
                {
                    "id": f"gid://shopify/ProductVariant/{title}",
                    "title": "Default",
                    "sku": f"SKU-{title}",
                    "price": "10.00",
                    "availableForSale": True,
                }
            ]
        },
    }


def _shopify_factory(monkeypatch, responses: list[dict], calls: dict):
    shopify_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        shopify_calls.append(json.loads(request.content)["variables"])
        return httpx.Response(
            200,
            json=_gql_response(responses[len(shopify_calls) - 1]),
        )

    _install_factory(monkeypatch, _make_client(handler))
    calls["shopify"] = shopify_calls
    return shopify_calls


def _install_gemini(
    monkeypatch, gemini_responses: list, calls: dict
) -> None:
    """gemini_responses: list of callables -> httpx.Response factory."""
    gemini_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        gemini_requests.append(json.loads(request.content))
        index = len(gemini_requests) - 1
        factory = gemini_responses[min(index, len(gemini_responses) - 1)]
        return factory()

    def factory():
        settings = load_llm_settings()
        client = make_gemini_client(
            settings.gemini,
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
            timeout_seconds=5,
        )
        return client, settings.gemini.model, agent.CHAT_SYSTEM_PROMPT

    monkeypatch.setattr(agent, "gemini_factory", factory)
    calls["gemini"] = gemini_requests


def _install_nvidia(monkeypatch, nvidia_responses: list, calls: dict) -> None:
    nvidia_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nvidia_requests.append(json.loads(request.content))
        index = min(len(nvidia_requests) - 1, len(nvidia_responses) - 1)
        payload = nvidia_responses[index]
        if isinstance(payload, httpx.Response):
            return payload
        return httpx.Response(
            200, json=payload if isinstance(payload, dict) else (
                {"choices": []} if payload == "choices-empty" else _nv_text(payload)
            )
        )

    def factory():
        settings = load_llm_settings()
        client = make_nvidia_client(
            settings.nvidia,
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
            timeout_seconds=5,
        )
        return client, settings.nvidia.model, agent.CHAT_SYSTEM_PROMPT

    monkeypatch.setattr(agent, "nvidia_factory", factory)
    calls["nvidia"] = nvidia_requests


def _g_text(text: str):
    def factory() -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    return factory


def _g_tool_call(name: str, args: dict, signature: str | None = None):
    def factory() -> httpx.Response:
        part: dict = {"functionCall": {"name": name, "args": args}}
        if signature:
            part["thoughtSignature"] = signature
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [part]},
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    return factory


def _g_status(status: int):
    def factory() -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "upstream"}})

    return factory


def _nv_text(text: str):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


# -- required cases -----------------------------------------------------------


def test_gemini_success_without_nemotron(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [_g_tool_call("search_catalog", {"query": "hat", "limit": 5}), _g_text(
            "We have the Classic Hat in stock at $10.00."
        )],
        calls,
    )
    _install_nvidia(monkeypatch, [], calls)  # must remain empty
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "What hats do you sell?", "session_id": "s-gem-ok"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["tools_used"] == ["search_catalog"]
    assert body["model_used"] == "gemini-3.6-flash"
    assert body["fallback_used"] is False
    assert "Classic Hat" in body["reply"] or "$10.00" in body["reply"]
    # Gemini round-trips: tool selection + final generation; NVIDIA cold.
    assert len(calls["gemini"]) == 2
    assert len(calls["nvidia"]) == 0
    # The verified tool result reached the LLM as function output.
    second_body = calls["gemini"][1]
    serialized = json.dumps(second_body)
    assert "SKU-Hat" in serialized  # real store data, invented nowhere


def test_thought_signature_is_replayed_on_next_model_call(monkeypatch):
    """Regression for the Gemini 3.x 400 investigation: signatures returned
    on function-call parts must be replayed verbatim on the persisted model
    turn (official Gemini baseline requirement; Google's own migration
    checklist lists signature preservation for 3.x function calling).
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call(
                "search_catalog", {"query": "hat"}, signature="c2lnLWhhdA=="
            ),
            _g_text("Answer grounded in the store data."),
        ],
        calls,
    )
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(monkeypatch, [_inventory_products_payload([_hat_product()])], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "What hats exist?", "session_id": "s-sig-replay"},
        )

    assert response.status_code == 200
    # The SECOND model call must carry the persisted model turn with the
    # signature Gemini attached to the function-call part.
    second_contents = calls["gemini"][1]["contents"]
    model_turns = [
        c for c in second_contents
        if c.get("role") == "model" and any(
            "functionCall" in p for p in c.get("parts", [])
        )
    ]
    assert model_turns, "function-call model turn missing from replay"
    parts = model_turns[0]["parts"]
    assert any(
        p.get("thoughtSignature") == "c2lnLWhhdA=="
        for p in parts
    ), f"signature not replayed: {parts}"


def test_eligible_gemini_failure_falls_back_to_nemotron(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("product_stock", {"product_query": "hat"}),
            _g_status(429),
        ],
        calls,
    )
    _install_nvidia(
        monkeypatch,
        [
            {"choices": [{"message": {"role": "assistant", "content": "Fallback answer."}}]}
        ],
        calls,
    )
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "Is the Hat in stock?", "session_id": "s-fb-ok"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Fallback answer."
    assert body["fallback_used"] is True
    assert body["model_used"] == "test-nemotron-01"
    assert body["tools_used"] == ["product_stock"]
    # Only ONE Nemotron call, and it received the SAME question + context.
    assert len(calls["nvidia"]) == 1
    user_content = calls["nvidia"][0]["messages"][1]["content"]
    assert user_content.startswith("Customer question: Is the Hat in stock?")
    assert "SKU-Hat" in user_content
    # Exactly one Shopify query despite the provider switch.
    assert len(calls["shopify"]) == 1


def test_gemini_http_503_triggers_nemotron_fallback(monkeypatch):
    """Regression for the LIVE Phase 8 evidence: an observed Gemini HTTP 503
    (with verified Shopify context present) must route the grounded answer to
    the Nemotron fallback through the real POST /api/chat flow. 503 maps to
    the same eligible GeminiUnavailableError class as 429 in the client.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("product_stock", {"product_query": "hat"}),
            _g_status(503),
        ],
        calls,
    )
    _install_nvidia(
        monkeypatch,
        [
            {"choices": [{"message": {"role": "assistant", "content": "Fallback answer for 503."}}]}
        ],
        calls,
    )
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "Is the Hat in stock?", "session_id": "s-503-fb"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Fallback answer for 503."
    assert body["fallback_used"] is True
    assert body["model_used"] == "test-nemotron-01"
    assert body["tools_used"] == ["product_stock"]
    # One Shopify query total despite the provider switch; one Nemotron call.
    assert len(calls["nvidia"]) == 1
    assert len(calls["shopify"]) == 1
    user_content = calls["nvidia"][0]["messages"][1]["content"]
    assert user_content.startswith("Customer question: Is the Hat in stock?")
    assert "SKU-Hat" in user_content


def test_gemini_503_before_store_access_reports_no_context_reason(monkeypatch):
    """The OBSERVED live shape (Gemini 503 on the first model call): with a
    message WITHOUT a SKU reference there is nothing to ground on, so the
    gate must NOT attempt Nemotron (never-guess rule) or Shopify, and the
    controlled 503 must say exactly why — including the native-payload
    classification.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_status(503)], calls)
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],  # must never be reached
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hello", "session_id": "s-503-nocxt"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    # Behavior change (approved Option B), explained: the no-context gate now
    # first attempts a deterministic SKU lookup for explicit SKU references,
    # so a message WITHOUT one reports exactly why the fallback stayed shut.
    assert "contains no SKU product reference" in detail
    assert "Gemini server error (HTTP 503)" in detail
    assert "native Google error payload" in detail
    # The fallback was correctly not tempted: zero NVIDIA calls, and the
    # deterministic lookup never ran for a message without a SKU reference.
    assert len(calls["nvidia"]) == 0
    assert len(calls["shopify"]) == 0


def test_gemini_503_fallback_grounds_sku_question_with_deterministic_lookup(monkeypatch):
    """Approved Option B regression: for '...SKU-Hat?' with Gemini down on the
    FIRST call, ONE deterministic read-only inventory lookup verifies the
    product and Nemotron answers from that real store data.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_status(503)], calls)
    _install_nvidia(
        monkeypatch,
        [{"choices": [{"message": {"role": "assistant", "content": "Grounded fallback answer."}}]}],
        calls,
    )
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={
                "message": "Hi! Can you tell me the price and available stock of SKU-Hat?",
                "session_id": "s-fb-sku",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["model_used"] == "test-nemotron-01"
    # Truthful surface: the deterministic lookup IS the store query that ran.
    assert body["tools_used"] == ["product_stock"]
    assert len(calls["shopify"]) == 1  # exactly one read-only lookup, no duplicates
    assert len(calls["nvidia"]) == 1
    shopify_query = calls["shopify"][0]["query"]
    assert shopify_query.lower().startswith("sku:")
    nvidia_context = calls["nvidia"][0]["messages"][1]["content"]
    assert "SKU-Hat" in nvidia_context


def test_unverified_sku_closes_gate_without_guessing(monkeypatch):
    """SKU-style question whose store lookups (inventory probe, then one
    distinct catalog search) return NO matching SKU: the gate stays closed,
    zero NVIDIA calls, controlled 503 names the verification failure.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_status(503)], calls)
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [
            _inventory_products_payload([_hat_product(title="Cap")]),
            _inventory_products_payload([_hat_product(title="Cap")]),
        ],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={
                "message": "Price and stock of SKU-Hat?",
                "session_id": "s-fb-nomatch",
            },
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "could not be verified in the Shopify store" in detail
    # Two DISTINCT read-only queries (inventory probe + catalog search), no
    # duplicate of the same lookup, and Nemotron never invoked.
    assert len(calls["shopify"]) == 2
    assert len(calls["nvidia"]) == 0


def test_shopify_failure_during_deterministic_retrieval_maps_upstream(monkeypatch):
    """A Shopify failure during the deterministic lookup is NOT a fallback
    matter: it propagates and keeps the exact pre-existing HTTP mapping
    (upstream failure → 502), with Nemotron never invoked.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_status(503)], calls)
    _install_nvidia(monkeypatch, [], calls)
    shopify_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        shopify_calls.append(json.loads(request.content)["variables"])
        return httpx.Response(503, json={"errors": [{"message": "shopify down"}]})

    _install_factory(monkeypatch, _make_client(handler))
    calls["shopify"] = shopify_calls

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={
                "message": "Price and stock of SKU-Hat?",
                "session_id": "s-fb-shopify-fail",
            },
        )

    assert response.status_code == 502
    assert len(shopify_calls) == 1  # failed once; no silent retry, no second call
    assert len(calls["nvidia"]) == 0


def test_untracked_inventory_still_permitted_when_price_verified(monkeypatch):
    """Verified product with REAL price but stock quantity reported as null:
    that is a verified data condition, never an invention — the fallback
    stays permitted and the context carries the null through unchanged.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_status(503)], calls)
    _install_nvidia(
        monkeypatch,
        [{"choices": [{"message": {"role": "assistant", "content": "Price known, stock unknown."}}]}],
        calls,
    )
    untracked = _hat_product(title="Hat")
    untracked["totalInventory"] = None
    untracked["variants"]["nodes"][0]["inventoryQuantity"] = None
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([untracked])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={
                "message": "Price and available stock of SKU-Hat?",
                "session_id": "s-fb-nullqty",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["tools_used"] == ["product_stock"]
    assert len(calls["shopify"]) == 1
    assert len(calls["nvidia"]) == 1


def test_gate_reason_reports_nemotron_failure_after_context(monkeypatch):
    """Gemini 503 AFTER a tool call (context exists) with Nemotron ALSO
    failing: the controlled 503 must report that the fallback request
    itself failed (with its sanitized category), not a generic reason.
    """
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("search_catalog", {"query": "hat"}),
            _g_status(503),
        ],
        calls,
    )
    _install_nvidia(
        monkeypatch,
        [httpx.Response(503, json={"error": {"message": "nvidia down"}})],
        calls,
    )
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hello", "session_id": "s-both-503"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "the Nemotron fallback request failed" in detail
    assert len(calls["nvidia"]) == 1
    assert len(calls["shopify"]) == 1


def test_ineligible_gemini_failure_returns_controlled_error(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_status(401)], calls)
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([])],  # must never be reached
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hi", "session_id": "s-auth"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "non-retryable" in detail
    # No fallback for auth failures; no Shopify query either.
    assert len(calls["nvidia"]) == 0
    assert len(calls["shopify"]) == 0


def test_gemini_499_cancelled_is_non_retryable_and_never_triggers_fallback(monkeypatch):
    """Regression for the OBSERVED live shape: Gemini answering HTTP 499
    with Google's canonical CANCELLED error payload. Pinned under the
    CURRENT policy (no production/policy change): the SDK's structured
    status/message appear sanitized in the controlled detail, the failure
    classifies as GeminiRequestError → INELIGIBLE, and the NVIDIA fallback
    is never attempted (zero NVIDIA calls) — 4xx stays non-retryable.
    """
    calls: dict = {}
    _llm_env(monkeypatch)

    def _g_499() -> httpx.Response:
        return httpx.Response(
            499,
            json={
                "error": {
                    "status": "CANCELLED",
                    "message": "The operation was cancelled.",
                }
            },
        )

    _install_gemini(monkeypatch, [_g_499], calls)
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([])],  # must never be reached
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hi", "session_id": "s-499-cancelled"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "non-retryable" in detail
    assert "HTTP 499" in detail
    assert "CANCELLED" in detail
    assert "The operation was cancelled." in detail
    # Never-guess policy keeps 4xx ineligible: no NVIDIA call, no Shopify
    # query (the gate is never reached for an INELIGIBLE classification).
    assert len(calls["nvidia"]) == 0
    assert len(calls["shopify"]) == 0


def test_both_providers_fail_returns_controlled_503(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("search_catalog", {"query": "hat"}),
            _g_status(503),
        ],
        calls,
    )
    _install_nvidia(monkeypatch, [{"choices": []}], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hello", "session_id": "s-both-fail"},
        )

    assert response.status_code == 503
    # One Nemotron attempt only: no retries, no repeated generations.
    assert len(calls["nvidia"]) == 1
    assert len(calls["shopify"]) == 1


def test_fallback_disabled_skips_nemotron(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch, fallback_enabled=False)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("search_catalog", {"query": "hat"}),
            _g_status(429),
        ],
        calls,
    )
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hello", "session_id": "s-disabled"},
        )

    assert response.status_code == 503
    assert len(calls["nvidia"]) == 0
    assert len(calls["shopify"]) == 1


def test_missing_nvidia_key_skips_fallback(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch, nvidia_key="placeholder_nvidia_key")
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("search_catalog", {"query": "hat"}),
            _g_status(500),
        ],
        calls,
    )
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([_hat_product()])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hello", "session_id": "s-no-key"},
        )

    assert response.status_code == 503
    assert len(calls["nvidia"]) == 0


def test_shopify_tool_failure_keeps_existing_http_mapping(monkeypatch):
    calls: dict = {}

    def shopify_handler(request: httpx.Request) -> httpx.Response:
        calls["shopify"] = calls.get("shopify", 0) + 1
        return httpx.Response(401, json={"errors": "unauthorized"})

    _install_factory(monkeypatch, _make_client(shopify_handler))
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [_g_tool_call("search_catalog", {"query": "hat"})],
        calls,
    )
    _install_nvidia(monkeypatch, [], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "hi", "session_id": "s-shopify-401"},
        )

    # Existing production mapping (not a fallback case).
    assert response.status_code == 401
    assert len(calls["nvidia"]) == 0
    assert calls["shopify"] == 1


def test_missing_store_data_is_never_fabricated(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [
            _g_tool_call("search_catalog", {"query": "rocket ship"}),
            _g_text(
                "I don't have any product matching that — I can't quote a price."
            ),
        ],
        calls,
    )
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(
        monkeypatch,
        [_inventory_products_payload([])],
        calls,
    )

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "Do you sell rocket ships?", "session_id": "s-empty"},
        )

    assert response.status_code == 200
    reply = response.json()["reply"]
    assert "don't have" in reply.lower()
    # The empty result really was delivered to the model as verified context.
    serialized = json.dumps(calls["gemini"][1])
    assert '"products": []' in serialized


def test_multi_turn_session_keeps_conversation(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_text("First reply."), _g_text("Second reply.")], calls)
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(monkeypatch, [], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        first = client.post(
            CHAT_URL, json={"message": "hi", "session_id": "s-multi"}
        )
        second = client.post(
            CHAT_URL, json={"message": "what about mugs?", "session_id": "s-multi"}
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["reply"] == "First reply."
    assert second.json()["reply"] == "Second reply."
    # The second Gemini request carries the full prior conversation.
    second_contents = calls["gemini"][1]["contents"]
    texts = json.dumps(second_contents)
    assert "First reply." in texts
    assert "what about mugs?" in texts


def test_no_provider_keys_leak_into_responses(monkeypatch):
    calls: dict = {}
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [_g_text("plain answer")], calls)
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(monkeypatch, [_inventory_products_payload([_hat_product()])], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products?limit=1")
        response = client.post(
            CHAT_URL, json={"message": "hi", "session_id": "s-secrets"}
        )

    assert GEM_KEY not in response.text
    assert NV_KEY not in response.text


def test_empty_message_maps_to_400(monkeypatch):
    _llm_env(monkeypatch)
    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(CHAT_URL, json={"message": "   "})
    assert response.status_code == 400


def test_missing_gemini_config_surfaces_as_500(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")

    def broken_factory():
        raise ConfigurationError("GEMINI_API_KEY is not set.")

    monkeypatch.setattr(agent, "gemini_factory", broken_factory)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(CHAT_URL, json={"message": "hi", "session_id": "s-cfg"})
    assert response.status_code == 500


def test_checkpointer_seam_used_and_session_state_persists(monkeypatch):
    """Phase M1 compatibility: run_chat gets its store through the
    memory_db seam (agent.get_checkpointer), and the SAME session_id
    accumulates thread state in that one checkpointer instance across
    turns — with no invented identity merging between sessions.
    """
    provided = InMemorySaver()  # a distinct instance the test controls
    monkeypatch.setattr(agent, "get_checkpointer", lambda: provided)
    _llm_env(monkeypatch)
    _install_gemini(
        monkeypatch,
        [_g_text("First reply."), _g_text("Second reply remembers the first.")],
        calls := {},
    )
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(monkeypatch, [], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        r1 = client.post(
            CHAT_URL,
            json={"message": "Hello there!", "session_id": "s-mem-1"},
        )
        r2 = client.post(
            CHAT_URL,
            json={"message": "What did I just ask you?", "session_id": "s-mem-1"},
        )
        r3 = client.post(
            CHAT_URL,
            json={"message": "What did I just ask you?", "session_id": "s-mem-2"},
        )

    assert r1.status_code == 200 and r2.status_code == 200 and r3.status_code == 200
    # InMemorySaver.storage nests thread_id → checkpoint_ns → checkpoint ids.
    thread_1 = provided.storage["s-mem-1"]
    thread_2 = provided.storage["s-mem-2"]
    assert thread_1 and thread_2  # separate keys, no merging
    namespace = thread_1[""]
    assert len(namespace) >= 2  # two turns checkpointed under the same thread


def test_midflight_neon_failure_maps_to_controlled_503(monkeypatch):
    """Approved Phase M1 follow-up: a Neon failure WHILE the graph runs
    (not just at connection setup) maps to the same controlled, sanitized
    503 — never an unhandled 500, never a silent in-memory downgrade, and
    the DSN/credentials are never echoed. Conversation history itself
    stays server-side, so nothing is silently lost.
    """

    def closing_store():
        raise psycopg.OperationalError("connection reset by server (fake text)")

    monkeypatch.setattr(agent, "get_checkpointer", closing_store)
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [], calls := {})
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(monkeypatch, [], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "Hello there!", "session_id": "s-mem-midfail"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "checkpoint store" in detail and "mid-operation" in detail
    # The raw provider text is never echoed.
    assert "connection reset by server" not in detail
    assert len(calls["gemini"]) == 0
    assert len(calls["nvidia"]) == 0
    assert len(calls["shopify"]) == 0


def test_unreachable_checkpoint_store_maps_to_controlled_503(monkeypatch):
    """Configured-but-unreachable Neon must surface as the controlled 503
    mapping (not 500, not a silent in-memory downgrade), with a sanitized
    detail and zero provider/store side traffic.
    """

    def broken_store():
        raise memory_db.MemoryDependencyError(
            "The configured Postgres checkpoint store is unreachable "
            "(connection or schema setup failed; details withheld)."
        )

    monkeypatch.setattr(agent, "get_checkpointer", broken_store)
    _llm_env(monkeypatch)
    _install_gemini(monkeypatch, [], calls := {})
    _install_nvidia(monkeypatch, [], calls)
    _shopify_factory(monkeypatch, [], calls)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.post(
            CHAT_URL,
            json={"message": "Hello there!", "session_id": "s-mem-db-down"},
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "checkpoint store unavailable" in detail
    # Nothing else ran: no providers, no Shopify, no fallback temptation.
    assert len(calls["gemini"]) == 0
    assert len(calls["nvidia"]) == 0
    assert len(calls["shopify"]) == 0
