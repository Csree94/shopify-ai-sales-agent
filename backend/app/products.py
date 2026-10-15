"""FastAPI routes for read-only Shopify product retrieval (Phase 4).

Exposed endpoints:
  GET /api/products           -> list products (paginated)
  GET /api/products/search?q= -> search products (paginated)

Authentication/throttle/timeout failures from the Shopify client are mapped
to explicit HTTP status codes. Credentials are read from the environment and
never appear in responses, logs, or error messages.
"""
from __future__ import annotations

from typing import Callable

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.shopify_client import (
    ShopifyAdminClient,
    ShopifyAuthError,
    ShopifyClientError,
    ShopifyThrottledError,
    ShopifyTimeoutError,
    ShopifyUpstreamError,
    normalise_inventory_levels_product,
    normalise_inventory_product,
    normalise_product,
)

router = APIRouter(prefix="/api", tags=["products"])

# Documented fallback when Shopify throttles but provides no usable
# Retry-After hint (see backend/docs/shopify-setup.md, error table).
DEFAULT_THROTTLE_BACKOFF_SECONDS = 5.0

# Indirection point for tests: the router builds the client through this
# factory, so tests can monkeypatch it with a mock-transport client.
client_factory: Callable[[], ShopifyAdminClient] = ShopifyAdminClient


def get_client() -> ShopifyAdminClient:
    if client_factory is not ShopifyAdminClient:
        return client_factory()
    return ShopifyAdminClient()


# -- response models -------------------------------------------------------


class VariantOut(BaseModel):
    id: str | None = None
    title: str | None = None
    sku: str | None = None
    price: str | None = None
    currency: str | None = None
    available_for_sale: bool | None = None


class ProductOut(BaseModel):
    id: str | None = None
    title: str | None = None
    description: str | None = None
    price_min: str | None = None
    price_max: str | None = None
    currency: str | None = None
    availability: bool | None = None
    total_inventory: int | None = None
    variants: list[VariantOut] = Field(default_factory=list)


class PageInfoOut(BaseModel):
    has_next_page: bool = False
    end_cursor: str | None = None


class ProductListResponse(BaseModel):
    products: list[ProductOut] = Field(default_factory=list)
    page_info: PageInfoOut = Field(default_factory=PageInfoOut)


# -- inventory probe models (Phase 6) ---------------------------------------


class InventoryVariantOut(BaseModel):
    id: str | None = None
    title: str | None = None
    sku: str | None = None
    inventory_quantity: int | None = None
    available_for_sale: bool | None = None


class InventoryProductOut(BaseModel):
    id: str | None = None
    title: str | None = None
    total_inventory: int | None = None
    variants: list[InventoryVariantOut] = Field(default_factory=list)


class InventoryListResponse(BaseModel):
    products: list[InventoryProductOut] = Field(default_factory=list)
    page_info: PageInfoOut = Field(default_factory=PageInfoOut)


# -- strict read_inventory probe models (Phase 7, detail=levels) -------------


class InventoryQuantityOut(BaseModel):
    name: str | None = None
    quantity: int | None = None


class InventoryLevelEntryOut(BaseModel):
    quantities: list[InventoryQuantityOut] = Field(default_factory=list)


class InventoryLevelsVariantOut(BaseModel):
    id: str | None = None
    title: str | None = None
    sku: str | None = None
    inventory_quantity: int | None = None
    inventory_levels: list[InventoryLevelEntryOut] | None = None
    available_for_sale: bool | None = None


class InventoryLevelsProductOut(BaseModel):
    id: str | None = None
    title: str | None = None
    total_inventory: int | None = None
    variants: list[InventoryLevelsVariantOut] = Field(default_factory=list)


class InventoryLevelsListResponse(BaseModel):
    products: list[InventoryLevelsProductOut] = Field(default_factory=list)
    page_info: PageInfoOut = Field(default_factory=PageInfoOut)


# -- error mapping ---------------------------------------------------------


def _to_http_exception(exc: ShopifyClientError) -> HTTPException:
    if isinstance(exc, ShopifyAuthError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, ShopifyThrottledError):
        # Always advertise a retry hint: the upstream value when Shopify sent
        # a usable one, otherwise the documented 5s fallback.
        retry_after = exc.retry_after or DEFAULT_THROTTLE_BACKOFF_SECONDS
        return HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": f"{retry_after:g}"},
        )
    if isinstance(exc, ShopifyTimeoutError):
        return HTTPException(status_code=504, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


def _run(call) -> ProductListResponse:
    """Execute a client call, normalise nodes, and translate errors."""
    try:
        connection = call()
    except ShopifyClientError as exc:
        raise _to_http_exception(exc) from exc
    nodes = connection.get("nodes") or []
    page = connection.get("pageInfo") or {}
    return ProductListResponse(
        products=[normalise_product(node) for node in nodes],
        page_info=PageInfoOut(
            has_next_page=bool(page.get("hasNextPage")),
            end_cursor=page.get("endCursor"),
        ),
    )


# -- endpoints ---------------------------------------------------------------


@router.get("/products", response_model=ProductListResponse)
def list_products(
    limit: int = Query(10, ge=1, le=250, description="Page size (1-250)."),
    cursor: str | None = Query(None, description="Opaque page cursor."),
) -> ProductListResponse:
    """List the store's products, newest first, one page at a time."""
    return _run(lambda: get_client().list_products(limit=limit, cursor=cursor))


@router.get("/products/search", response_model=ProductListResponse)
def search_products(
    q: str = Query(
        ..., min_length=1, description="Shopify search query, e.g. 'title:hat'."
    ),
    limit: int = Query(10, ge=1, le=250),
    cursor: str | None = Query(None),
) -> ProductListResponse:
    """Search products with Shopify's `products(query:)` syntax."""
    return _run(
        lambda: get_client().search_products(query=q, limit=limit, cursor=cursor)
    )


@router.get(
    "/products/inventory",
    response_model=InventoryListResponse | InventoryLevelsListResponse,
)
def list_inventory(
    limit: int = Query(5, ge=1, le=250, description="Page size (1-250)."),
    cursor: str | None = Query(None, description="Opaque page cursor."),
    detail: Literal["quantities", "levels"] = Query(
        "quantities",
        description=
            "'quantities' (default): per-variant stock quantity only; "
            "'levels' (Phase 7): strict `read_inventory` probe including the "
            "InventoryLevel chain (location-level quantities).",
    ),
) -> InventoryListResponse | InventoryLevelsListResponse:
    """Read-only inventory probe (permission test for `read_inventory`).

    Reuses product pagination, authentication, and error handling. Null
    quantities/levels are data conditions and never permission failures; a
    missing `read_inventory` scope surfaces as HTTP 502 with Shopify's own
    error message (strict when `detail=levels`).
    """
    try:
        connection = get_client().list_inventory(
            limit=limit, cursor=cursor, detail=detail
        )
    except ShopifyClientError as exc:
        raise _to_http_exception(exc) from exc
    nodes = connection.get("nodes") or []
    page = connection.get("pageInfo") or {}
    page_info = PageInfoOut(
        has_next_page=bool(page.get("hasNextPage")),
        end_cursor=page.get("endCursor"),
    )
    if detail == "levels":
        return InventoryLevelsListResponse(
            products=[normalise_inventory_levels_product(node) for node in nodes],
            page_info=page_info,
        )
    return InventoryListResponse(
        products=[normalise_inventory_product(node) for node in nodes],
        page_info=page_info,
    )
