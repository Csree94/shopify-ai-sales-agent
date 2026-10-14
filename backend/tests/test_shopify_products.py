"""Phase 4 tests: mocked Shopify HTTP integration tests.

All HTTP is intercepted with `httpx.MockTransport` — no test here touches the
network or needs real Shopify credentials. The `/health` contract test lives
in `test_health.py` and must keep passing.
"""
from __future__ import annotations

import json
from typing import Callable

import httpx
from fastapi.testclient import TestClient

import app.products as products
from app.config import ShopifySettings
from app.shopify_client import ShopifyAdminClient

TEST_SETTINGS = ShopifySettings(
    store_domain="test-store.myshopify.com",
    api_version="2026-10",
    admin_api_token="fake-token-for-tests-only",  # synthetic; not a credential
    timeout_seconds=5.0,
)

GRAPHQL_URL = "https://test-store.myshopify.com/admin/api/2026-10/graphql.json"


def _gql_response(data: dict | None, errors: list | None = None) -> dict:
    body: dict = {"data": data}
    if errors is not None:
        body["errors"] = errors
    return body


def _product_cursor_page(
    titles: list[str], end_cursor: str | None, has_next: bool
) -> dict:
    return {
        "products": {
            "pageInfo": {"endCursor": end_cursor, "hasNextPage": has_next},
            "nodes": [_product_node(t) for t in titles],
        }
    }


def _product_node(title: str) -> dict:
    return {
        "id": f"gid://shopify/Product/1-{title}",
        "title": title,
        "description": f"The {title} product",
        "totalInventory": 7,
        "priceRange": {
            "minVariantPrice": {"amount": "10.00", "currencyCode": "USD"},
            "maxVariantPrice": {"amount": "25.00", "currencyCode": "USD"},
        },
        "variants": {
            "nodes": [
                {
                    "id": f"gid://shopify/ProductVariant/1-{title}",
                    "title": "Default",
                    "sku": f"SKU-{title}",
                    "price": {"amount": "10.00", "currencyCode": "USD"},
                    "availableForSale": True,
                }
            ]
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> ShopifyAdminClient:
    return ShopifyAdminClient(
        settings=TEST_SETTINGS,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _install_factory(monkeypatch, client: ShopifyAdminClient) -> None:
    monkeypatch.setattr(products, "client_factory", lambda: client)


def test_products_endpoint_happy_path(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth_header"] = request.headers.get(
            "X-Shopify-Access-Token"
        )
        captured["variables"] = json.loads(request.content)["variables"]
        return httpx.Response(
            200, json=_gql_response(_product_cursor_page(["Hat", "Tee"], "c1", True))
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert [p["title"] for p in payload["products"]] == ["Hat", "Tee"]
    hat = payload["products"][0]
    assert hat["description"] == "The Hat product"
    assert hat["price_min"] == "10.00"
    assert hat["price_max"] == "25.00"
    assert hat["currency"] == "USD"
    assert hat["availability"] is True
    assert hat["total_inventory"] == 7
    assert hat["variants"][0]["sku"] == "SKU-Hat"
    assert hat["variants"][0]["available_for_sale"] is True
    assert payload["page_info"] == {"has_next_page": True, "end_cursor": "c1"}

    # Outgoing request is correct and the (fake) token goes only in the header.
    assert captured["url"] == GRAPHQL_URL
    assert captured["auth_header"] == "fake-token-for-tests-only"
    assert "fake-token" not in response.text
    assert captured["variables"] == {"first": 2, "after": None, "query": None}


def test_products_search_and_pagination(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["variables"] = json.loads(request.content)["variables"]
        page = _product_cursor_page(["Screwdriver"], "c9", False)
        return httpx.Response(200, json=_gql_response(page))

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/search",
            params={"q": "title:screwdriver", "limit": 1, "cursor": "c0"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert [p["title"] for p in payload["products"]] == ["Screwdriver"]
    assert payload["page_info"]["has_next_page"] is False
    assert captured["variables"] == {
        "first": 1,
        "after": "c0",
        "query": "title:screwdriver",
    }


def test_empty_store_returns_empty_page(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(
                {"products": {"pageInfo": {"endCursor": None, "hasNextPage": False}, "nodes": []}}
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 200
    assert response.json() == {
        "products": [],
        "page_info": {"has_next_page": False, "end_cursor": None},
    }


def test_auth_failure_maps_to_401(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errors": "unauthorized"})

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()
    # No credential material leaks into the response.
    assert "fake-token" not in response.text


def test_throttle_maps_to_429_with_retry_after(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Retry-After": "2.5"},
            json=_gql_response(
                None,
                errors=[{"message": "Too many requests", "extensions": {"code": "THROTTLED"}}],
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 429
    assert response.headers["retry-after"].startswith("2")


def test_upstream_http_429_without_retry_after_uses_fallback(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"errors": "rate limited"})

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 429
    # Documented fallback when Shopify sends no usable Retry-After.
    assert response.headers["retry-after"] == "5"


def test_upstream_http_429_preserves_retry_after(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"Retry-After": "9"}, json={"errors": "slow down"}
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "9"


def test_mixed_data_and_errors_rejects_partial_result(monkeypatch):
    """GraphQL errors + usable-looking data must NOT be served as complete."""

    def handler(request: httpx.Request) -> httpx.Response:
        partial_page = _product_cursor_page(["Hat"], "c1", False)
        # Data present alongside a field-level error: serving `data` alone
        # could silently drop fields, so the documented behaviour is 502.
        return httpx.Response(
            200,
            json=_gql_response(
                partial_page,
                errors=[{"message": "Field 'description' is missing"}],
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 502
    assert "missing" in response.json()["detail"]


def test_throttled_code_takes_precedence_over_partial_data(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        partial_page = _product_cursor_page(["Hat"], "c1", False)
        return httpx.Response(
            200,
            headers={"Retry-After": "2"},
            json=_gql_response(
                partial_page,
                errors=[
                    {"message": "Too many requests", "extensions": {"code": "THROTTLED"}}
                ],
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"


def test_timeout_maps_to_504(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("simulated timeout")

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 504


def test_shopify_api_errors_map_to_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(None, errors=[{"message": "invalid query field"}]),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 502
    assert "invalid query field" in response.json()["detail"]


def test_server_error_maps_to_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="maintenance")

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products")

    assert response.status_code == 502


def test_search_requires_q_param():
    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products/search")

    assert response.status_code == 422
