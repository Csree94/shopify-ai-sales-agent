"""Minimal Shopify Admin (GraphQL) client for read-only product retrieval.

- Talks to `https://{store_domain}/admin/api/{version}/graphql.json`.
- Sends the Admin API access token via the `X-Shopify-Access-Token` header;
  the token is never written to logs, errors, or request URLs.
- Only read operations (products/products search). Never mutates products or
  inventory.

Errors are translated into typed exceptions so the HTTP layer can map them
to clear status codes:
  - ShopifyAuthError        -> 401/403 (bad/missing token, scopes)
  - ShopifyThrottledError   -> 429 (leaky-bucket throttle), carries retry hint
  - ShopifyTimeoutError     -> 504
  - ShopifyUpstreamError    -> 502 (unexpected status / API errors / parsing)

The httpx client is injectable, so tests can supply `httpx.MockTransport`
and never touch the network.
"""
from __future__ import annotations

import httpx

from app.config import MAX_GQL_PAGE_SIZE, ShopifySettings, load_shopify_settings

PRODUCT_FIELDS = """
  id
  title
  description
  totalInventory
  priceRange {
    minVariantPrice { amount currencyCode }
    maxVariantPrice { amount currencyCode }
  }
  variants(first: 50) {
    nodes {
      id
      title
      sku
      price { amount currencyCode }
      availableForSale
    }
  }
"""

PRODUCTS_QUERY = """
query Products($first: Int!, $after: String, $query: String) {
  products(first: $first, after: $after, query: $query) {
    pageInfo { endCursor hasNextPage }
    nodes {
""" + PRODUCT_FIELDS + """
    }
  }
}
"""


class ShopifyClientError(Exception):
    """Base class for Shopify client errors (message never contains secrets)."""


class ShopifyAuthError(ShopifyClientError):
    """Authentication/authorization failed (401/403): token or scopes invalid."""


class ShopifyThrottledError(ShopifyClientError):
    """Shopify leaky-bucket throttle; optionally carries a retry hint (seconds)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ShopifyTimeoutError(ShopifyClientError):
    """Request timed out before Shopify responded."""


class ShopifyUpstreamError(ShopifyClientError):
    """Shopify returned an unexpected status or an unusable payload."""


class ShopifyAdminClient:
    """Read-only GraphQL Admin API client bound to validated settings."""

    def __init__(
        self,
        settings: ShopifySettings | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or load_shopify_settings()
        self._http = http_client or httpx.Client(
            timeout=self._settings.timeout_seconds
        )

    @property
    def settings(self) -> ShopifySettings:
        return self._settings

    # -- public queries ----------------------------------------------------

    def list_products(self, limit: int, cursor: str | None = None) -> dict:
        """List products (newest first by default). `limit` is 1..250."""
        limit = _clamp_limit(limit)
        return self._fetch_products(first=limit, after=cursor, query=None)

    def search_products(
        self, query: str, limit: int, cursor: str | None = None
    ) -> dict:
        """Search products with Shopify's `query` syntax (e.g. `title:hat`)."""
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        limit = _clamp_limit(limit)
        return self._fetch_products(first=limit, after=cursor, query=query.strip())

    # -- internals ---------------------------------------------------------

    def _fetch_products(
        self, first: int, after: str | None, query: str | None
    ) -> dict:
        """Run the products query and return raw, validated connection data."""
        variables: dict = {"first": first, "after": after, "query": query}
        payload = self._request(PRODUCTS_QUERY, variables)
        connection = payload.get("products")
        if not isinstance(connection, dict):
            raise ShopifyUpstreamError("Shopify response missing 'products' node.")
        return connection

    def _request(self, query: str, variables: dict) -> dict:
        """POST one GraphQL request; translate failures into typed errors."""
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": self._settings.admin_api_token,
        }
        try:
            response = self._http.post(
                self._settings.graphql_url,
                headers=headers,
                json={"query": query, "variables": variables},
            )
        except httpx.TimeoutException as exc:
            raise ShopifyTimeoutError(
                f"Shopify request timed out after {self._settings.timeout_seconds:g}s."
            ) from exc
        except httpx.RequestError as exc:
            raise ShopifyUpstreamError(
                "Failed to reach Shopify (network error)."
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise ShopifyAuthError(
                f"Shopify rejected the Admin API request (HTTP {status}). "
                "Check SHOPIFY_ADMIN_ACCESS_TOKEN and the app's read_products scope."
            )
        if status == 402:
            raise ShopifyUpstreamError(
                "Shopify reports this app's API access is inactive or unpaid (HTTP 402)."
            )
        if status == 429:
            # Upstream throttle signalled at the HTTP layer. Preserve a valid
            # Retry-After when Shopify sends one; no automatic retries here —
            # the HTTP layer surfaces the hint to the caller.
            raise ShopifyThrottledError(
                "Shopify throttled the request (HTTP 429).",
                retry_after=_parse_retry_after(response),
            )
        if status >= 500:
            raise ShopifyUpstreamError(f"Shopify server error (HTTP {status}).")
        if status != 200:
            raise ShopifyUpstreamError(
                f"Unexpected Shopify status code (HTTP {status})."
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ShopifyUpstreamError(
                "Shopify returned a non-JSON response body."
            ) from exc

        # Shopify error responses never echo credentials, so bodies are safe
        # to surface; the token itself is only ever placed in a request header.

        errors = body.get("errors")

        if errors:
            # Policy: any GraphQL error rejects the whole response, except
            # throttling (retried at the caller). The query shape is fixed;
            # an error means the payload may be silently missing fields (SKUs,
            # prices, ...), which we cannot verify against the contract — so
            # partial `data` is never served as a complete result.
            entries = errors if isinstance(errors, list) else [errors]
            messages = []
            throttled = False
            for entry in entries:
                code = (entry or {}).get(  # type: ignore[union-attr]
                    "extensions", {}
                ).get("code") if isinstance(entry, dict) else None
                text = (
                    (entry or {}).get("message", "")  # type: ignore[union-attr]
                    if isinstance(entry, dict)
                    else str(entry)
                )
                if code == "THROTTLED" or "throttl" in text.lower():
                    throttled = True
                if text:
                    messages.append(text)
            if throttled:
                raise ShopifyThrottledError(
                    "Shopify throttled the request (leaky bucket).",
                    retry_after=_parse_retry_after(response),
                )
            raise ShopifyUpstreamError(
                f"Shopify API errors: {'; '.join(m for m in messages if m) or 'unknown'}"
            )

        data = body.get("data")
        if not isinstance(data, dict):
            raise ShopifyUpstreamError("Shopify response has no 'data' object.")
        return data


# -- normalisation ---------------------------------------------------------


def normalise_product(node: dict) -> dict:
    """Map a raw GraphQL product node to the public product shape."""
    price_range = node.get("priceRange") or {}
    minimum = price_range.get("minVariantPrice") or {}
    maximum = price_range.get("maxVariantPrice") or {}
    variants = [
        {
            "id": v.get("id"),
            "title": v.get("title"),
            "sku": v.get("sku"),
            "price": _amount(v.get("price")),
            "currency": ((v.get("price") or {}).get("currencyCode")),
            "available_for_sale": v.get("availableForSale"),
        }
        for v in ((node.get("variants") or {}).get("nodes") or [])
    ]
    total_inventory = node.get("totalInventory")
    available: bool | None
    variant_availability = [v["available_for_sale"] for v in variants]
    if total_inventory is not None:
        available = total_inventory > 0
    elif variants and all(a is not None for a in variant_availability):
        available = all(bool(a) for a in variant_availability)
    else:
        available = None  # availability not reported for this product
    return {
        "id": node.get("id"),
        "title": node.get("title"),
        "description": node.get("description"),
        "price_min": _amount(minimum),
        "price_max": _amount(maximum),
        "currency": minimum.get("currencyCode"),
        "availability": available,
        "total_inventory": total_inventory,
        "variants": variants,
    }


def _amount(money_node: dict) -> str | None:
    """Return the money amount as a numeric string (semantic-preserving)."""
    value = (money_node or {}).get("amount")
    return None if value is None else str(value)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Extract a valid numeric Retry-After hint, or None when absent/unparseable."""
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        value = float(header)
    except ValueError:
        return None
    return value if value >= 0 else None


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_GQL_PAGE_SIZE))
