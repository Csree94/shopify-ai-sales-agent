# Project Handover — Shopify AI Sales Agent (Project 2)

> Purpose of this file: hand this document to a new AI assistant (or developer) so
> development can continue without losing context. Read it top-to-bottom before
> touching the repository. Every section distinguishes **verified facts** from
> **user-reported facts** and **planned/NOT TESTED items** — do not inflate them.

Created: 2026-10 (during the Phase 7 review round).
Updated: Phase 7 live `detail=levels` result recorded (successful strict probe,
with the documented scope-attribution caveat).
Repository: `C:\Users\harik\Desktop\Sree_new\shopify-ai-sales-agent`
Remote: `https://github.com/Csree94/shopify-ai-sales-agent.git`

---

## 1. Project overview

**What it is.** Project 2 is a Shopify "AI Sales Agent": a Python backend that
talks to a Shopify store on the user's behalf (product listing, search, and —
being phased in — inventory awareness), built so that later phases can connect
an LLM, a chat surface (Telegram), and a checkout flow. The frontend is a
React + TypeScript + Vite scaffold. The immediate technical goal through
Phase 7 has been a solid, tested, permission-aware foundation: authentication,
read-only product retrieval, and individually proven Shopify API permissions.

**User-facing goals (per the plan).** An AI assistant that can answer customer
questions about products (name, description, price, variants, SKU,
availability, eventually location-level stock), suggest products, and support
checkout-related flows through a Telegram chat surface and an admin dashboard.
None of the chat/LLM/checkout/dashboard parts are implemented yet.

**Technologies.**
- Backend: Python (venv at repo root `.venv\`), FastAPI, pydantic,
  [httpx](https://www.python-httpx.org) (with `MockTransport` for all tests),
  `python-dotenv` (silent `.env` loading), pytest.
- Shopify access: **GraphQL Admin API only, version `2026-10`** (latest stable).
  The REST Admin API is legacy per Shopify and was announced to stop serving
  traffic on **2026-11-02** — it is deliberately not used.
- Frontend: React + TypeScript + Vite + oxlint scaffold (built unchanged since
  Phase 1; only React+TS code present from the initial scaffold).
- Direct pins live in `backend/requirements.txt`; a full transitive lock is in
  `requirements-proof.txt` at the repo root.

**Project boundaries (must not be crossed).**
- This is **Project 2 — the Shopify AI Sales Agent only**.
- Project 1 and the **Inventra** project are separate repositories. Never open,
  modify, or integrate them into Project 2, and never copy code between them.
- All Shopify operations are strictly **read-only**. No `write_*` scopes are
  granted and none will be requested.

---

## 2. Environment and setup

- **Repository path:** `C:\Users\harik\Desktop\Sree_new\shopify-ai-sales-agent`
- **GitHub remote:** `origin` = `https://github.com/Csree94/shopify-ai-sales-agent.git`
- **Default branch:** `main` (push-only fast-forwards; no force-push policy).

**Backend run/verify (from the `backend/` directory):**
```bash
# Git Bash (recommended; the project has been developed with bash on Windows)
../.venv/Scripts/python.exe -m pytest -q       # full backend suite
../.venv/Scripts/python.exe -m pip check       # dependency sanity
../.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```
```powershell
# PowerShell equivalents (note: no space after the two dots)
cd C:\Users\harik\Desktop\Sree_new\shopify-ai-sales-agent\backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

**Frontend (rarely needed so far):** `cd frontend && npm install && npm run build`
(tsc + vite build; verified clean in earlier phases).

**Environment-variable NAMES** (all live in git-ignored `.env` files —
`backend/.env` is loaded first, then the repo-root `.env`; existing process
environment variables always win; values are never printed):
- Shopify: `SHOPIFY_STORE_DOMAIN` (`*.myshopify.com`),
  `SHOPIFY_CLIENT_ID`, `SHOPIFY_CLIENT_SECRET` (client-credentials grant —
  preferred auth mode), `SHOPIFY_ADMIN_ACCESS_TOKEN` (static custom-app token —
  fallback only; placeholder values are treated as "not set"),
  `SHOPIFY_API_VERSION` (default `2026-10`),
  `SHOPIFY_HTTP_TIMEOUT_SECONDS` (default 15, max 60),
  `SHOPIFY_STOREFRONT_ACCESS_TOKEN` (present as a placeholder only; unused).
- Future features (placeholders exist, nothing read them yet):
  `GEMINI_API_KEY` (LLM), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`,
  `DATABASE_URL` (do not provision without approval), `REDIS_URL`.

**Editor instructions are** `backend/docs/shopify-setup.md` (**auth modes,
required scopes, permission-test procedures**). That file is the source of
truth for Shopify config; this handover supplements it with project history.

---

## 3. Architecture and implementation

**Backend layout.**
- `backend/app/main.py` — FastAPI entrypoint; registers the products router;
  `GET /health` (contract frozen since Phase 1).
- `backend/app/config.py` — `ShopifySettings` loaded **per call** from
  environment (not import time), so tests inject env dicts. Validations:
  myshopify-domain normalisation, supported API-version list
  (`2026-01 … 2026-10`, latest `2026-10`), timeout bounds, auth credentials.
  `auth_mode` = `client_credentials` (when client id+secret are set) or
  `static_token`; client credentials take precedence, and placeholder-looking
  static tokens never silently override them. The dataclass `repr` redacts all
  credential fields.
- `backend/app/shopify_auth.py` — `ShopifyTokenManager`:
  - client-credentials grant → `POST https://{store}/admin/oauth/access_token`
    with form fields `grant_type=client_credentials`, `client_id`,
    `client_secret` (officially documented flow; verified against shopify.dev);
  - tokens valid ≈ 24 h (`expires_in` ≈ 86399), cached in memory (thread-safe
    lock), refreshed **proactively 300 s before expiry**, and invalidated after
    any upstream 401 (next call re-authenticates; no in-request auto-retry);
  - secrets only ever travel in request headers/bodies; nothing is logged.
- `backend/app/shopify_errors.py` — shared exception hierarchy
  (`ShopifyAuthError`, `ShopifyThrottledError` (+ `retry_after`),
  `ShopifyTimeoutError`, `ShopifyUpstreamError`).
- `backend/app/shopify_client.py` — read-only GraphQL client over
  `https://{store}/admin/api/{version}/graphql.json`:
  - `list_products`, `search_products` (Shopify `products(query:)` syntax),
    `list_inventory(limit, cursor, detail="quantities"|"levels")`;
  - typed error translation: HTTP 401/403 → auth, GraphQL `THROTTLED` or
    HTTP 429 → throttle (+ `Retry-After` parsed, unused values dropped),
    timeouts → 504, everything else/parse issues/GraphQL errors → 502;
  - **any GraphQL `errors` array rejects the whole response** (no partial
    results are served as complete — fixed query shape means silently missing
    fields cannot be distinguished from empty ones);
  - product normalisation; **`ProductVariant.price` is the scalar `Money`**
    type (a monetary string); currency comes from `priceRange` (`MoneyV2`).
- `backend/app/products.py` — routes (see §3 routes) + pydantic response
  models + HTTP error mapping.

**Frontend.** Initial React+TS+Vite scaffold only (`frontend/src/App.tsx`,
hero image, oxlint config). No API integration on the frontend yet. Tests:
backend-only (pytest); frontend verified via `npm run build`.

**Routes (all read-only):**
| Route | Status | Since |
|---|---|---|
| `GET /health` | implemented (contract frozen) | Phase 1 commit |
| `GET /api/products` (`limit`, `cursor`) | implemented + live-tested | Phase 4 |
| `GET /api/products/search?q=` | implemented (same pagination) | Phase 4 |
| `GET /api/products/inventory?limit&cursor&detail` | implemented (`detail=quantities` default; `detail=levels` = strict `read_inventory` probe, now sending `quantities(names: ["available"])`) | Phase 6/7 (**live strictly-probed result: PASS — see §5/§6**) |
| Checkout / Telegram / RAG / LLM / dashboard / DB | not implemented | planned |

**API boundary rules (important).**
- Admin API = `https://{store}.myshopify.com/admin/api/{v}/graphql.json` with
  Admin access tokens and `read_*`/`write_*` scopes.
- Storefront API = `https://{store}.myshopify.com/api/{v}/graphql.json` with a
  **different** token header (`X-Shopify-Storefront-Access-Token`) and
  `unauthenticated_*` scopes. They are **not interchangeable**; the Storefront
  API has not been used at all yet.

**Shopify permissions noted in the app setup doc:** `read_products` (used by
everything), `read_inventory` (probe phase), storefront scopes (future). No
`write_*` scopes — never request them.

---

## 4. Completed phases and commits

All four commits below are **verified pushed to `origin/main`** (local HEAD ==
origin/main checked after each push; fast-forward only).

| Commit | Message | Phase | Suite at that commit |
|---|---|---|---|
| `b6d6280` | `chore: initialize Shopify AI sales agent project` | Phases 1–3: frontend scaffold, FastAPI `/health` + contract test, env/dependency foundation, `.gitignore` | 1 test |
| `5cd3ae6` | `feat: add Shopify product retrieval API` | Phase 4: GraphQL product listing/search, pagination, typed errors | 22 tests |
| `6b3688e` | `feat: add Shopify authentication and fix product query` | Phase 5: client-credentials auth + token refresh; silent `.env` loading; scalar-`Money` price fix | 39 tests |
| `ffa908f` | `feat: add read-only Shopify inventory endpoint` | Phase 6: `GET /api/products/inventory` (`inventoryQuantity` scalar) | 44 tests |

**Uncommitted work (Phase 7, in progress — do NOT invent a commit for it):**
- Modified, uncommitted files: `backend/app/products.py`,
  `backend/app/shopify_client.py`, `backend/docs/shopify-setup.md`,
  `backend/tests/test_shopify_products.py`.
- Content: strict `read_inventory` probe via `detail=levels` on the inventory
  endpoint
  (`inventoryItem → inventoryLevels(first:5) → quantities(names:["available"]){name,quantity}`),
  response-model split, docs update, 6 new mocked tests + the `names` fix
  regression assertion.
- Suite at this state: **50 passed** (`pytest -q` re-run at the fixed state, no
  cache side-effects), `pip check` clean, `git diff --check` clean.
- Live progress during the Phase 7 round [user-reported]: the first live run
  failed with Shopify's validation error
  `Field 'quantities' is missing required arguments: names`; the query was
  corrected to send `names: ["available"]` and the **second live run
  succeeded: five products with location-level quantities
  (e.g. `{"name": "available", "quantity": 50}`) and
  `page_info.has_next_page: true`** (the exact HTTP status code was not
  displayed and is not recorded). See §5/§6.
- Work stopped at: **awaiting review + explicit approval to commit the Phase 7
  changes**. Do not commit/push without the verification checklist in §9(8).
  See §7/§8.

---

## 5. Live testing evidence

Legend: **[verified by agent]** = reproduced by me through outputs I obtained;
**[user-reported]** = performed by the user on their real store; **[mocked]** =
`httpx.MockTransport`, no network.

- **Phase 4/5 — product retrieval: **GET /api/products?limit=3`` returned real
  Shopify product data. [user-reported] First live attempt (during Phase 5)
  exposed an invalid nested selection on `ProductVariant.price`
  (`Selections can't be made on scalars (field 'price' returns Money ……)`);
  this was fixed to the bare scalar `price` and the user confirmed a successful
  retry. That exact error is preserved in a regression test. [verified by agent, mocked]
- **Phase 6 — inventory quantity probe: **GET /api/products/inventory?limit=5``
  returned HTTP 200 with real product and variant inventory quantities.
  [user-reported] Example structure (from the response contract):
  `{"products":[{"title":…,"total_inventory":…,"variants":[{"sku":…,
  "inventory_quantity":<int>,…}]}],"page_info":{…}}`.
  ⚠ Scope caveat: success proves the query/schema work, but `inventoryQuantity`
  may be `read_products`-gated — see §6 for why `detail=levels` is required to
  prove `read_inventory` itself.
- **Phase 7 — `detail=levels` strict probe: TESTED-PASS (live).** [user-reported]
  - First live attempt failed with Shopify's own validation error (verbatim,
    do not lose it):
    `Shopify API errors: Field 'quantities' is missing required arguments: names`.
    This was a schema-argument problem, not a permission verdict — the query
    was confirmed against shopify.dev (`InventoryLevel.quantities` requires
    `names: [String!]!`) and fixed to `quantities(names: ["available"])`,
    matching the Phase 5 query-fix pattern. The sent-argument shape is now
    asserted by the `test_inventory_levels_success` regression test.
  - Second live attempt: `GET /api/products/inventory?limit=5&detail=levels`
    returned a successful JSON response: five products, location-level
    quantities on the displayed variants including `name: "available"` with
    `quantity: 50`, and `page_info.has_next_page: true`. The surfaced output
    did not explicitly display the numeric HTTP status code; it is therefore
    not recorded rather than guessed.
- **Auth/token flow:** client-credentials token acquisition has passed all
  mocked tests (including expiry-margin refresh and 401-invalidation) and is
  exercised by every live request the user made above; no separate live
  token-lifecycle observation has been made.
- All **50** current backend tests are fully mocked; no test touches the live
  store. Run `pytest` anytime — it never sends traffic.

---

## 6. Permissions matrix (record only verified results)

| Scope | Purpose / feature | Implemented? | Test method | Result | Remaining uncertainty |
|---|---|---|---|---|---|
| `read_products` | Product listing + search (+ price/variant/SKU data) | Yes (Phases 4–5) | Live `GET /api/products?limit=3` through FastAPI/Swagger | **TESTED-PASS** [user-reported] | None known; all fields returned |
| `inventoryQuantity` fields (scalar quantities per variant) | Inventory-aware availability in answers | Yes (Phase 6) | Live `GET /api/products/inventory?limit=5` | **TESTED-PASS** [user-reported] | ⚠ Which scope gates this field is not publicly unambiguous (`read_products` vs `read_inventory`); success does **not** independently prove `read_inventory` |
| `read_inventory` (strict, `InventoryLevel`) | Location-level stock (planned stock-aware feature) | Probe implemented (Phase 7), feature not built | Live Swagger `GET /api/products/inventory?limit=5&detail=levels` (after the `names` query fix) | **TESTED-PASS — strict retrieval** [user-reported]: 5 products with location-level quantities (e.g. `{"name":"available","quantity":50}`), `has_next_page: true`; HTTP status not explicitly displayed, not recorded | **Scope-attribution caveat:** shopify.dev documents `InventoryLevel` as requiring `read_inventory`, so the successful strict chain strongly indicates the scope is granted — but this is not framed as definitive, independently falsified/confirmed scope verification (docs were read at snippet level per §9(5), and the Shopify dashboard's scope list was never used to declare a result). Null/empty `inventory_levels` would have been data conditions, not scope verdicts; that case never arose. |
| `read_inventory` (legacy fields e.g. `quantityAvailable`) | Fallback knowledge | n/a | documented only | NOT TESTED | Legacy fields are reportedly tied to `read_inventory` per shopify.dev; we did not rely on them |
| Storefront `unauthenticated_read_product_listings` + `SHOPIFY_STOREFRONT_ACCESS_TOKEN` | Future storefront-facing display | No | Storefront GraphQL probe (`/api/{v}/graphql.json`) — **not planned until the feature exists** | NOT TESTED | Requires creating a storefront token (an app-config change → user approval first) |
| `write_*` scopes | Product/order mutation | **Never** — agent is read-only | not applicable (never requested) | Deliberately excluded | — |

**Key distinction (teacher's requirement):** scopes shown in the Shopify
dashboard are *claims*; a scope is only *proven* when the specific read
operation succeeds through the API of record. The dashboard was never used to
declare a result in any row above.

---

## 7. Current project state (verified at handover time)

- **Branch:** `main`. **HEAD:** `ffa908f`
  (`ffa909bd…` full: `ffa908fce099e43466b8f05e3fb924d339936095` at last check;
  = `origin/main`).
- **Working tree:** NOT clean — Phase 7 is uncommitted (4 modified files
  listed in §4, including the `names` query fix + regression test; plus the
  doc updates recorded this round). **Nothing staged.** Nothing unexpected
  tracked; `docs/PROJECT_HANDOVER.md` itself is still untracked (`?? docs/`).
- **Tests:** 50 passed, 0 failed; `pip check` clean (both re-verified from the
  `backend/` directory at handover time).
- **`backend/.env`:** exists with real-looking domain + client credentials
  (presence verified heuristically only — values never displayed) and
  git-ignored (`.gitignore` pattern `.env`), never staged or tracked.
- **Exact stop point:** Phase 7 code + docs + tests are complete and passing,
  and the **live `detail=levels` Swagger test has now succeeded** (§5/§6,
  including the corrected `quantities(names: ["available"])` query). The docs
  now record that result with the explicit retrieval-vs-scope-attribution
  distinction. The very next action is **review + commit of the Phase 7
  changes — only on the user's explicit instruction** (§9(8)); no live store
  request is pending.

---

## 8. Pending work and recommended order

**Immediate next actions (in order).**
1. ~~User runs the live strict probe~~ **DONE.** Result recorded in §5/§6:
   first run failed on the (then-required) missing `names` argument; after the
   one-line query fix and a regression test, the live run passed. Remaining
   homework from it: record the outcome in `backend/docs/shopify-setup.md`
   (done in this documentation round).
2. **Review + commit the Phase 7 changes** (4 code/test files + 2 doc files,
   plus `docs/PROJECT_HANDOVER.md` if desired) with a message such as
   `feat: add strict read_inventory levels probe` — **only when the user
   explicitly asks**, after full pytest + `pip check` + staged-diff secret
   scan; verify HEAD == origin/main afterwards.
3. Phase 8 (planned): Telegram integration or LLM integration (per the
   teacher's plan) — still requiring the sales-agent chat design.

**Remaining permission/verification work.**
- ~~Record `detail=levels` outcome in the matrix (§6).~~ **Done** — recorded
  as TESTED-PASS (strict retrieval) with the scope-attribution caveat kept
  explicit.
- Later, when features exist: per-location inventory feature (using proven
  `read_inventory` results), storefront token + probe (needs user-approved
  token creation), LLM scope check (`GEMINI_API_KEY` usage), Telegram webhook
  secret handling.

**Feature backlog (not started).** Telegram integration, RAG product-search
grounding, LLM answer generation (`GEMINI_API_KEY` button already pinned in
`requirements.txt` as `google-genai`), checkout flow (likely Storefront API;
read-only or the teacher's decision), admin dashboard, database persistence
(`psycopg` pinned but unused), security hardening, rate limiting.

**Optional improvements (nice-to-have).**
- Request-lifetime Shopify client/token-manager singleton (currently a fresh
  `httpx` client per request; fine at current scale).
- Root README; move/copy of `requirements-proof.txt` documentation.
- Considering GraphQL query cost/complexity headers for throttling telemetry.
- Visitor-facing caching of product pages (only if the agent needs it).

---

## 9. Known issues and decisions (do not reverse without discussion)

1. **Scalar `Money` price: `ProductVariant.price` is a monetary string;
   currency comes from `priceRange` (`MoneyV2`).** This was forced by a live
   schema error and is regression-tested. Reverting it will break live calls.
2. **GraphQL error policy: any `errors` array rejects the whole response**
   (502). Never "upgrade" this to partial-result merging without a structural
   reason and new tests.
3. **Auth precedence: client credentials > static token**; placeholder-looking
   values (contain `placeholder`, start `your_`, or the webhook placeholder)
   are treated as unset (helpful `ConfigurationError` instead of a doomed call).
4. **`SHOPIFY_API_VERSION` validation** against a frozen supported list
   (`2026-01`, `2026-04`, `2026-07`, `2026-10`); bump the list only after
   re-checking shopify.dev's currently supported versions.
5. **Docs read snippet-level.** shopify.dev could not be fetched directly from
   this environment (DNS blocked); all schema claims relied on official-page
   snippets via search plus live-server behavior. When a live error contradicts
   a doc snippet, trust the live error and re-verify the schema.
6. **No auto-retries.** 429 answers carry `Retry-After` (upstream value or a
   documented 5 s fallback), but the backend never retries automatically.
7. **One `httpx.Client` per request** (no pooling) — known inefficiency,
   acceptable so far; do not "fix" casually during another feature.
8. **Commit/push only after verification** (full pytest + `pip check` +
   staged-diff secret scan) and only when the user asks. Never
   reset/rebase/force/discard.
9. **Test counts per commit are historical facts** (22→39→44→50). Do not
   re-align old commits' numbers when the suite grows.

---

## 10. Instructions for the next AI assistant

1. **Read this document first**, but then **verify the real repository state**
   (`git status`/`git log`/tests) before changing anything — never assume a
   feature is complete because it is listed here. Uncommitted work may exist
   or may have been moved/reverted by the user since this file was written.
2. Continue from the **last verified checkpoint** (§7). If the working tree
   differs from §7, ask the user what changed before touching anything.
3. Work in **small, beginner-friendly steps**; explain technical concepts
   simply; do not bundle unrelated changes into one step.
4. Respect the boundaries: **Project 2 only**; never touch Project 1 or
   Inventra; never grant/revoke Shopify scopes; strictly read-only Shopify
   operations; never fetch/print tokens, secrets or `.env` contents.
5. **Permission tests are guided, one at a time.** Give the user exact steps
   (PowerShell + Swagger), state the expected pass/fail signal in advance, then
   record the actual result in §6 and `backend/docs/shopify-setup.md`.
6. **Never commit or push without** running the backend suite + `pip check` +
   a staged-diff secret scan, showing the user what will be committed, and
   waiting for explicit approval. No resets, rebases, force-pushes, or
   discards unless the user explicitly instructs it.
7. When a live test requires the user's action, **stop and wait** for their
   result; do not claim connectivity that has not been demonstrated.
8. Maintain this handover file as the project grows (append phase outcomes,
   keep the matrix current, mark unverified claims clearly).
