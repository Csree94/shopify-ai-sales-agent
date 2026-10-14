# Shopify Admin API setup (Phase 4 — product retrieval; Phase 5 — authentication)

## What this phase uses

- **GraphQL Admin API only.** The REST Admin API is legacy and Shopify
  announced it stops serving traffic on **2026-11-02**, so product retrieval
  is implemented on GraphQL (`/admin/api/{version}/graphql.json`).
- Default API version: **`2026-10`** (stable 2026-10-01 → 2026-10-16, per
  shopify.dev current versioning). Also accepted: `2026-07`, `2026-04`,
  `2026-01`. Configurable via `SHOPIFY_API_VERSION`; the client validates it.
- **Read-only.** The client only runs `products(...)` queries. It never
  creates, updates, or deletes products or inventory.

## Authentication (Phase 5)

Two supported modes, decided automatically from your `.env`:

1. **Client-credentials grant (preferred).** Set `SHOPIFY_CLIENT_ID` and
   `SHOPIFY_CLIENT_SECRET`. The backend requests tokens per Shopify's
   official client-credentials flow for apps with direct API access:

   `POST https://{store}.myshopify.com/admin/oauth/access_token` with
   form fields `grant_type=client_credentials`, `client_id`, `client_secret`.

   The response contains `access_token` and `expires_in` ≈ **86399**
   seconds (token validity ≈ **24 hours**, per shopify.dev's access token
   documentation). The backend caches the token in memory and refreshes
   it 5 minutes before expiry (and after any upstream 401). Tokens are
   never logged.
2. **Static custom-app token.** If no client credentials are configured,
   a real (non-placeholder) `SHOPIFY_ADMIN_ACCESS_TOKEN` from a custom
   app's *Admin API access token* is used directly. Note the granted
   scopes in this mode come from the app's Admin API configuration.

Client credentials take precedence when both are configured; a
placeholder-only static token never silently overrides them.

## Required scopes

Create the app in one of these places, matching your authentication mode:

- *Dev Dashboard app* (for the client-credentials grant):
  shopify.dev dashboard → *Store access → Direct API access*, enable Admin
  API access and select scopes.
- *Custom app* (for the static token mode):
  Store admin → *Settings → Apps and sales channels → Develop apps* →
  create an app → Admin API integration — grant:

  - `read_products` — required (product titles, descriptions, prices,
    variants, SKU, availability).

### Permission (scope) testing matrix

Every scope the planned project needs, why, and how each is tested
individually. Never assume a permission works just because it is listed in
the app settings — each one is proven with its own request.

| Scope | Feature that needs it | How we test it individually |
|---|---|---|
| `read_products` | Product listing & search (`GET /api/products*`) — Phases 4–5 | Send `GET /api/products?limit=3` against the dev store: expect HTTP 200 with real product titles. If the scope is missing, Shopify returns GraphQL errors we surface as 502 with the message echoed — that failure is the scope test failing. |
| `read_inventory` | Per-variant stock quantities for stock-aware sales answers (probe implemented in Phase 6) | Minimal probe implemented: `GET /api/products/inventory?limit=5` (scalar `inventoryQuantity` per variant). Success: HTTP 200 with per-variant integer `inventory_quantity` values. Missing scope: HTTP **502** putting Shopify's own "Access denied… `read_inventory`" message in `detail`. A `null` quantity is a data condition, **not** a permission error. Escalation probe if inconclusive: `inventoryLevel` (the `InventoryLevel` object strictly requires `read_inventory`). |
| Storefront token scopes (`unauthenticated_read_product_listings`) | Future storefront-facing chat/checkout display | Storefront API probe: `shop { products(first:1) { nodes { title } } }`. Not required in Phase 5; requires `SHOPIFY_STOREFRONT_ACCESS_TOKEN` only when that feature is built. |
| `write_*` scopes | **None.** The agent is read-only; we will never grant write scopes in this project. | Not applicable — deliberately not requested. |

Each planned feature ships with its own minimal probe before relying on
any scope; failures against the dev store are reported as setup issues,
not code issues.

## Environment configuration

Set these in your **local `.env`** (copy from `.env.example`). The real
`.env` is git-ignored; placeholders live only in `.env.example`.

| Variable                      | Meaning                                        |
|-------------------------------|------------------------------------------------|
| `SHOPIFY_STORE_DOMAIN`        | `your-store.myshopify.com` (the Admin API must target the myshopify domain) |
| `SHOPIFY_CLIENT_ID`           | Client id for the client-credentials grant (Dev Dashboard app). |
| `SHOPIFY_CLIENT_SECRET`       | Client secret for the client-credentials grant (secret — never commit). |
| `SHOPIFY_ADMIN_ACCESS_TOKEN`  | Admin API access token from the custom app (secret — never commit). Optional when client credentials are set. |
| `SHOPIFY_API_VERSION`         | Optional, defaults to `2026-10`.               |
| `SHOPIFY_HTTP_TIMEOUT_SECONDS`| Optional, defaults to `15`, capped at `60`.    |

`SHOPIFY_STOREFRONT_ACCESS_TOKEN` is **not** used in this phase — it is
reserved for future storefront-facing features.

Files `backend/.env` and the repo-root `.env` are loaded automatically at
startup (python-dotenv, silently, no values printed). Existing process
environment variables always take precedence over file values.

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

## Inventory probe test procedure (Phase 6)

`GET /api/products/inventory?limit=5` is a **read-only** permission test for
`read_inventory`. It reuses product pagination, authentication, and error
handling, with one added scalar per variant (`inventoryQuantity`).

**Swagger steps:**

1. From `backend/`, run `..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000`.
2. Open `http://127.0.0.1:8000/docs`.
3. Expand **GET /api/products/inventory** → **Try it out** → set `limit` (e.g. `5`) → **Execute**.
4. Interpret:
   - `200` with integer `inventory_quantity` values → access to variant stock quantities works.
   - `200` with `"inventory_quantity": null` → the store returned null/missing values (data condition, not a permission issue).
   - `502` with an "Access denied…" message naming `read_inventory` → scope not granted; record it in the permission matrix.
   - `401` → credentials/Direct API access issue (see error table). `429` → throttled; honor `Retry-After`.

The endpoint never writes or adjusts inventory or any other store data: the
GraphQL operation is a pure `products(...)` read query.

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
