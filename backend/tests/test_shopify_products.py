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
                    "price": "10.00",
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


def _cc_settings() -> "ShopifySettings":
    return ShopifySettings(
        store_domain="test-store.myshopify.com",
        api_version="2026-10",
        client_id="fake-client-id",
        client_secret="fake-client-secret",
        timeout_seconds=5.0,
    )


def test_scalar_variant_price_regression(monkeypatch):
    """Regression for the live failure on 2026-10:

    `Selections can't be made on scalars (field 'price' returns Money but
    has selections ["amount", "currencyCode"])`.

    The outgoing query must select `price` as a bare scalar, and the parsed
    variant price must be the monetary string with currency from the
    product's price range.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        query = body["query"]
        # Reproduce the exact live error if the invalid selection is sent.
        if "price {" in query:
            return httpx.Response(
                200,
                json=_gql_response(
                    None,
                    errors=[
                        {
                            "message": "Selections can't be made on scalars "
                            "(field 'price' returns Money but has selections "
                            "[\"amount\", \"currencyCode\"])"
                        }
                    ],
                ),
            )
        # Corrected query: variant price is selected as a bare scalar.
        assert "price\n" in query
        return httpx.Response(
            200, json=_gql_response(_product_cursor_page(["Hat"], "c1", False))
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products?limit=1")

    assert response.status_code == 200
    product = response.json()["products"][0]
    variant = product["variants"][0]
    # Scalar Money stays a monetary string exactly as Shopify returned it.
    assert variant["price"] == "10.00"
    # Currency is carried by the product's MoneyV2 price range.
    assert variant["currency"] == "USD"
    assert product["price_min"] == "10.00"
    assert product["price_max"] == "25.00"


def test_inventory_endpoint_success(monkeypatch):
    """Phase 6 inventory probe: per-variant quantities returned as integers."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["variables"] = body["variables"]
        captured["query"] = body["query"]
        product = {
            "id": "gid://shopify/Product/10",
            "title": "Hat",
            "totalInventory": 12,
            "variants": {
                "nodes": [
                    {
                        "id": "gid://shopify/ProductVariant/10",
                        "title": "Default",
                        "sku": "SKU-Hat",
                        "price": "10.00",
                        "inventoryQuantity": 12,
                        "availableForSale": True,
                    }
                ]
            },
        }
        return httpx.Response(
            200,
            json=_gql_response(
                {"products": {"pageInfo": {"endCursor": "x1", "hasNextPage": True}, "nodes": [product]}}
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products/inventory?limit=1")

    assert response.status_code == 200
    # The probe query selects the single step-up scalar.
    assert "inventoryQuantity" in captured["query"]
    assert captured["variables"] == {"first": 1, "after": None, "query": None}

    product = response.json()["products"][0]
    variant = product["variants"][0]
    assert variant["sku"] == "SKU-Hat"
    assert variant["inventory_quantity"] == 12
    # price intentionally absent from the minimal probe shape
    assert "price" not in variant
    assert variant["available_for_sale"] is True
    assert product["total_inventory"] == 12
    assert response.json()["page_info"] == {"has_next_page": True, "end_cursor": "x1"}


def test_inventory_null_and_missing_quantity_stay_null(monkeypatch):
    """A null/missing quantity is a data condition, not an error or crash."""

    def handler(request: httpx.Request) -> httpx.Response:
        product = {
            "id": "gid://shopify/Product/11",
            "title": "Nullish",
            "totalInventory": None,
            "variants": {
                "nodes": [
                    {
                        "id": "gid://shopify/ProductVariant/11",
                        "title": "Default",
                        "sku": None,
                        "inventoryQuantity": None,
                        "availableForSale": None,
                    },
                    {"id": "gid://shopify/ProductVariant/12", "title": "NoField", "sku": "SKU-12", "availableForSale": False},
                ]
            },
        }
        return httpx.Response(
            200,
            json=_gql_response(
                {"products": {"pageInfo": {"endCursor": None, "hasNextPage": False}, "nodes": [product]}}
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products/inventory")

    assert response.status_code == 200
    products = response.json()["products"]
    assert products[0]["total_inventory"] is None
    variants = products[0]["variants"]
    assert variants[0]["inventory_quantity"] is None
    assert variants[0]["sku"] is None
    assert variants[0]["available_for_sale"] is None
    # Missing field on the second variant also normalises to None, not TypeError.
    assert variants[1]["inventory_quantity"] is None
    assert variants[1]["available_for_sale"] is False


def test_inventory_pagination_forwards_cursor(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["variables"] = json.loads(request.content)["variables"]
        return httpx.Response(
            200,
            json=_gql_response(
                {"products": {"pageInfo": {"endCursor": "inv9", "hasNextPage": False}, "nodes": []}}
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/inventory", params={"limit": 2, "cursor": "inv0"}
        )

    assert response.status_code == 200
    assert captured["variables"] == {"first": 2, "after": "inv0", "query": None}
    assert response.json()["page_info"] == {"has_next_page": False, "end_cursor": "inv9"}


def test_inventory_missing_scope_maps_to_502(monkeypatch):
    """Access denied on the probe field = read_inventory not granted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(
                None,
                errors=[
                    {"message": "Access denied for inventoryQuantity field. "
                     "Required access: `read_inventory` access scope."}
                ],
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products/inventory")

    assert response.status_code == 502
    assert "read_inventory" in response.json()["detail"]


def test_inventory_auth_failure_maps_to_401(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errors": "unauthorized"})

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products/inventory")

    assert response.status_code == 401
    assert "fake-token" not in response.text


def _levels_node(title: str, quantities: list[dict] | None, with_item: bool = True) -> dict:
    variant = {
        "id": f"gid://shopify/ProductVariant/lv-{title}",
        "title": "Default",
        "sku": f"SKU-{title}",
        "price": "10.00",
        "inventoryQuantity": 7,
        "availableForSale": True,
    }
    if with_item:
        variant["inventoryItem"] = {
            "inventoryLevels":{"nodes": ([{"quantities": quantities}] if quantities is not None else [])}
        }
    return {
        "id": f"gid://shopify/Product/lv-{title}",
        "title": title,
        "totalInventory": 7,
        "variants": {"nodes": [variant]},
    }


def _inventory_products_payload(products: list[dict]) -> dict:
    return {"products": {"pageInfo": {"endCursor": None, "hasNextPage": False}, "nodes": products}}


def test_inventory_levels_success(monkeypatch):
    """Strict read_inventory probe: location-level quantities per variant."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["query"] = body["query"]
        captured["variables"] = body["variables"]
        return httpx.Response(
            200,
            json=_gql_response(
                _inventory_products_payload(
                    [_levels_node("Hat", [{"name": "available", "quantity": 5}])]
                )
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/inventory", params={"limit": 1, "detail": "levels"}
        )

    assert response.status_code == 200
    assert "inventoryLevels" in captured["query"]
    # Shopify 2026-10: InventoryLevel.quantities requires the names argument
    # (regression for the live validation error "Field 'quantities' is missing
    # required arguments: names").
    assert "quantities(names: [\"available\"])" in captured["query"]
    assert "mutation" not in captured["query"].lower()
    assert captured["variables"] == {"first": 1, "after": None, "query": None}

    variant = response.json()["products"][0]["variants"][0]
    assert variant["inventory_levels"] == [
        {"quantities": [{"name": "available", "quantity": 5}]}
    ]
    assert variant["inventory_quantity"] == 7


def test_inventory_levels_null_and_empty_are_data_conditions(monkeypatch):
    """None (no item) and [] (chain present, empty) stay data conditions."""

    def handler(request: httpx.Request) -> httpx.Response:
        nodes = [
            _levels_node("Untracked", None, with_item=False),
            _levels_node("NoLevels", None),
            _levels_node("Stocked", [{"name": "available", "quantity": 3}]),
        ]
        # The no-item variant must have no inventoryItem key at all.
        assert "inventoryItem" not in nodes[0]["variants"]["nodes"][0]
        return httpx.Response(200, json=_gql_response(_inventory_products_payload(nodes)))

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/inventory", params={"detail": "levels"}
        )

    assert response.status_code == 200
    # Each helper node was a separate product with one variant.
    products = response.json()["products"]
    v_untracked = products[0]["variants"][0]
    v_no_levels = products[1]["variants"][0]
    v_stocked = products[2]["variants"][0]
    assert v_untracked["inventory_levels"] is None  # no inventoryItem present
    assert v_no_levels["inventory_levels"] == []    # chain present, empty levels
    assert v_stocked["inventory_levels"] == [
        {"quantities": [{"name": "available", "quantity": 3}]}
    ]


def test_inventory_levels_missing_scope_maps_to_502(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_gql_response(
                None,
                errors=[
                    {"message": "Access denied for inventoryLevels field. "
                     "Required access: `read_inventory` access scope."}
                ],
            ),
        )

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/inventory", params={"detail": "levels"}
        )

    assert response.status_code == 502
    assert "read_inventory" in response.json()["detail"]


def test_inventory_levels_auth_failure_maps_to_401(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"errors": "unauthorized"})

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/inventory", params={"detail": "levels"}
        )

    assert response.status_code == 401
    assert "fake-token" not in response.text


def test_default_inventory_behaviour_unchanged_in_levels_mode_off(monkeypatch):
    """Without detail=levels the endpoint uses the Phase 6 query and shape."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["query"] = body["query"]
        product = {
            "id": "gid://shopify/Product/20",
            "title": "Default",
            "totalInventory": 4,
            "variants": {
                "nodes": [
                    {"id": "gid://shopify/ProductVariant/20", "title": "Default", "sku": "SKU-20", "price": "5.00", "inventoryQuantity": 4, "availableForSale": True}
                ]
            },
        }
        return httpx.Response(200, json=_gql_response(_inventory_products_payload([product])))

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get("/api/products/inventory")

    assert response.status_code == 200
    assert "inventoryQuantity" in captured["query"]
    assert "inventoryLevels" not in captured["query"]
    assert "inventoryItem" not in captured["query"]
    variant = response.json()["products"][0]["variants"][0]
    assert variant["inventory_quantity"] == 4
    assert "inventory_levels" not in variant


def test_inventory_invalid_detail_rejected(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:  # must never be hit
        raise AssertionError("no Shopify request expected for invalid detail")

    _install_factory(monkeypatch, _make_client(handler))

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as client:
        response = client.get(
            "/api/products/inventory", params={"detail": "write"}
        )

    assert response.status_code == 422


def test_products_use_dynamically_acquired_token(monkeypatch):
    """Client-credentials mode: token endpoint hit first, token header
    injected into the GraphQL call, secret never in any response."""
    graphql_requests: list[httpx.Request] = []
    token_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/admin/oauth/access_token"):
            token_calls["count"] += 1
            assert request.headers["Content-Type"].startswith(
                "application/x-www-form-urlencoded"
            )
            form = dict(
                part.split("=", 1) for part in request.content.decode().split("&")
            )
            assert form["grant_type"] == "client_credentials"
            assert form["client_id"] == "fake-client-id"
            assert form["client_secret"] == "fake-client-secret"
            return httpx.Response(
                200, json={"access_token": "shppa_fake_token", "expires_in": 86399}
            )
        graphql_requests.append(request)
        return httpx.Response(
            200, json=_gql_response(_product_cursor_page(["Shirt"], None, False))
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = ShopifyAdminClient(settings=_cc_settings(), http_client=http_client)
    _install_factory(monkeypatch, client)

    from app.main import app as fastapi_app

    with TestClient(fastapi_app) as test_client:
        response = test_client.get("/api/products?limit=1")

    assert response.status_code == 200
    assert [p["title"] for p in response.json()["products"]] == ["Shirt"]
    assert token_calls["count"] == 1
    assert len(graphql_requests) == 1
    assert (
        graphql_requests[0].headers["X-Shopify-Access-Token"] == "shppa_fake_token"
    )
    # Neither the token nor the client secret leaks into the API response.
    assert "shppa_fake_token" not in response.text
    assert "fake-client-secret" not in response.text
