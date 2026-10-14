# Shopify Admin API setup (Phase 4 — product retrieval)

## What this phase uses

- **GraphQL Admin API only.** The REST Admin API is legacy and Shopify
  announced it stops serving traffic on **2026-11-02**, so product retrieval
  is implemented on GraphQL (`/admin/api/{version}/graphql.json`).
- Default API version: **`2026-10`** (stable 2026-10-01 → 2026-10-16, per
  shopify.dev current versioning). Also accepted: `2026-07`, `2026-04`,
  `2026-01`. Configurable via `SHOPIFY_API_VERSION`; the client validates it.
- **Read-only.** The client only runs `products(...)` queries. It never
  creates, updates, or deletes products or inventory.

## Required scopes

Create a **custom app** in the Shopify admin:

1. Store admin → *Settings → Apps and sales channels → Develop apps*.
2. *Create an app*, then under *Configuration → Admin API integration* grant:
   - `read_products` — required (product titles, descriptions, prices,
     variants, SKU, availability).
3. *Install the app* and reveal the **Admin API access token**.

### Optional (only if you need exact stock counts later)

- `read_inventory` — would let us report precise per-location quantities.
  Phase 4 reports availability from `totalInventory` / `availableForSale`,
  which `read_products` already provides, so this scope is not required now.

## Environment configuration

Set these in your **local `.env`** (copy from `.env.example`). The real
`.env` is git-ignored; placeholders live only in `.env.example`.

| Variable                      | Meaning                                        |
|-------------------------------|------------------------------------------------|
| `SHOPIFY_STORE_DOMAIN`        | `your-store.myshopify.com` (the Admin API must target the myshopify domain) |
| `SHOPIFY_ADMIN_ACCESS_TOKEN`  | Admin API access token from the custom app (secret — never commit). |
| `SHOPIFY_API_VERSION`         | Optional, defaults to `2026-10`.               |
| `SHOPIFY_HTTP_TIMEOUT_SECONDS`| Optional, defaults to `15`, capped at `60`.    |

## HTTP surface (Phase 4)

- `GET /api/products?limit=10&cursor=<opaque>` — list products, newest first.
- `GET /api/products/search?q=<shopify-query>&limit=10&cursor=<opaque>` —
  search using Shopify `products(query:)` syntax, e.g. `title:hat`,
  `product_type:screws`, `tag:sale`.
- Responses: `{ "products": [...], "page_info": { "has_next_page", "end_cursor" } }`.
  Pagination: pass `page_info.end_cursor` as the next request's `cursor`.
  Each product: `id`, `title`, `description`, `price_min`, `price_max`,
  `currency`, `availability`, `total_inventory`, `variants[]`
  (`id`, `title`, `sku`, `price`, `currency`, `available_for_sale`).

## Error handling

| Situation                                | HTTP      | Notes                          |
|------------------------------------------|-----------|--------------------------------|
| Bad token / missing `read_products` scope | `401`     | Detail explains what to check, no secret values. |
| Shopify leaky-bucket throttle (HTTP 200 + `THROTTLED` code) | `429` | `Retry-After` echoed from Shopify when provided. |
| Shopify throttle signalled as HTTP 429 | `429` | Same handling; `Retry-After` preserved when usable. |
| Throttle with no usable `Retry-After` | `429` | Documented fallback: `Retry-After: 5`. No automatic retries in this phase. |
| Timeout                                   | `504`     | Configurable via `SHOPIFY_HTTP_TIMEOUT_SECONDS`. |
| Other Shopify/network failures            | `502`     | Unexpected status, API errors, or unusable payload. |
| Configuration missing/invalid             | —         | Raised as `ConfigurationError` when a client is built; surfaces as a 500 with a named-variable message. |

## Partial GraphQL responses

The products query uses a fixed field shape, so a response containing any
GraphQL `errors` array cannot be verified field-by-field (Silently dropped
fields such as SKUs or prices cannot be distinguished from empty values).
Policy:

- `THROTTLED` errors still surface as `429` regardless of any accompanying
  `data` (a retry is meaningful, and no result is claimed).
- Any other GraphQL error makes the response **unsafe to serve as a complete
  result** and is surfaced as an upstream failure (`502`). Partial `data` is
  never returned as though it were complete — the client never silently
  claims a full page.

## Limits

- Page size is clamped to Shopify's maximum `first: 250`.
- Variant list capped at `first: 50` per product.
- Tests are fully mocked (`httpx.MockTransport`); no live calls. **Live
  connectivity has not been verified yet** — configure a real custom app and
  run the service against your dev store to confirm end-to-end.
