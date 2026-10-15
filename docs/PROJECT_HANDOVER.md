# Project Handover — Shopify AI Sales Agent (Project 2)

> Purpose of this file: hand this document to a new AI assistant (or developer) so
> development can continue without losing context. Read it top-to-bottom before
> touching the repository. Every section distinguishes **verified facts** from
> **user-reported facts** and **planned/NOT TESTED items** — do not inflate them.

Created: 2026-10 (during the Phase 7 review round).
Updated: Phase 8 (Gemini/LangGraph sales agent + NVIDIA fallback) and Phase M1
(Neon-backed persistent conversation memory) are IMPLEMENTED, TESTED, and
PUSHED (`e8ca2ae` on `origin/main`). Telegram integration is the next planned
phase (see §8/§11 for the exact continuation prompt).
Repository: `C:\Users\harik\Desktop\Sree_new\shopify-ai-sales-agent`
Remote: `https://github.com/Csree94/shopify-ai-sales-agent.git`

---

## 1. Project overview

**What it is.** Project 2 is a Shopify "AI Sales Agent": a Python backend that
talks to a Shopify store on the user's behalf (read-only product listing,
search, inventory), an LLM chat service (LangGraph agent with a Gemini
primary model and an NVIDIA Nemotron generation-only fallback), a FastAPI chat
surface (`POST /api/chat`), and persistent per-session conversation memory in
Neon PostgreSQL. The frontend is a React + TypeScript + Vite scaffold (still
bare). Still **not** implemented: Telegram chat surface, FAQ/policy retrieval
(RAG), proactive product recommendations, a checkout flow, admin dashboard,
caching/limiting, deployment.

**User-facing goals (per the plan).** An AI assistant that can answer customer
questions about products (name, description, price, variants, SKU,
availability), suggest products, and support checkout-related flows through a
Telegram chat surface and an admin dashboard. Chat answering from store data
is **implemented live-verified**; the remaining surfaces are not.

**Technologies.**
- Backend: Python (venv at repo root `.venv\`), FastAPI, pydantic, LangGraph
  (state graph + checkpointer), `google-genai` (Gemini), `openai` client
  (NVIDIA), [httpx](https://www.python-httpx.org) (with `MockTransport` for
  all tests), `psycopg` 3 + `langgraph-checkpoint-postgres` (Neon memory),
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
- Project 1 and the **Inventra** project are separate repositories. **Boundary
  update (user-approved):** inspecting Inventra **read-only** is now the
  sanctioned next task (to find reusable Telegram integration code — see §8),
  but Inventra must still never be modified, and copying code into Project 2
  requires the user's explicit approval and clean adaptation.
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
- LLM (Phase 8 — now actively read): `GEMINI_API_KEY`,
  `GEMINI_MODEL` (default `gemini-3.6-flash`; see §9(10) — sampling params
  must never be sent), `NVIDIA_API_KEY`, `NVIDIA_LLM_MODEL`
  (`nvidia/nemotron-3-ultra-550b-a55b`), `NVIDIA_LLM_BASE_URL`
  (`https://integrate.api.nvidia.com/v1`), `LLM_FALLBACK_ENABLED`.
- Memory (Phase M1 — now actively read): `NEON_DATABASE_URL` (postgres:// DSN;
  unset/placeholder → in-memory mode; setup steps in `.env.example`).
- Still future (placeholders exist, nothing reads them yet):
  `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, `DATABASE_URL`, `REDIS_URL`.

**Editor instructions are** `backend/docs/shopify-setup.md` (**auth modes,
required scopes, permission-test procedures**). That file is the source of
truth for Shopify config; this handover supplements it with project history.

---

## 3. Architecture and implementation

**Phase 1–7 core (stable, fully preserved):**
- `backend/app/main.py` — FastAPI entrypoint; registers the products and chat
  routers; `GET /health` (contract frozen since Phase 1).
- `backend/app/config.py` — per-call settings loading (`ShopifySettings`,
  `LLMSettings`, `NeonSettings` + loaders inject env dicts for tests).
  Validations: myshopify-domain normalisation, supported API-version list,
  timeout bounds, auth credentials, DSN scheme; `auth_mode` =
  `client_credentials` over `static_token`; placeholder-looking values
  (contain `placeholder`, start `your_`, or the known webhook placeholder)
  are treated as unset. **All credential-bearing reprs are redacted**
  (`NeonSettings(dsn=<redacted>)`, LLM keys redacted) — never fetch/print
  secrets; nothing is logged.
- `backend/app/shopify_auth.py` — `ShopifyTokenManager`: client-credentials
  grant, ≈24 h tokens cached in memory (thread-safe lock), proactively
  refreshed 300 s before expiry, invalidated after any 401 (no in-request
  auto-retry); secrets only in headers/bodies, never logged.
- `backend/app/shopify_errors.py` — shared exception hierarchy
  (`ShopifyAuthError`, `ShopifyThrottledError` (+ `retry_after`),
  `ShopifyTimeoutError`, `ShopifyUpstreamError`).
- `backend/app/shopify_client.py` — read-only GraphQL client:
  `list_products`, `search_products`, `list_inventory(detail="quantities"|"levels")`;
  typed error translation (401/403 auth, 429/THROTTLED throttle + Retry-After,
  timeouts → 504, everything else → 502); **any GraphQL `errors` array rejects
  the whole response** (no partial results served as complete);
  `ProductVariant.price` is the scalar `Money` string, currency from
  `priceRange` (`MoneyV2`).
- `backend/app/products.py` — read-only routes + pydantic models + HTTP error
  mapping.

**Phase 8 (LLM sales agent):**
- `backend/app/gemini_client.py` — one `genai.Client` per turn from validated
  settings; `generate_once()` does exactly ONE API round-trip; automatic
  function calling is disabled (the agent executes tools itself). Request
  compliance fixes (verified against Google's official docs and the installed
  SDK source): **no sampling parameters ever sent** (deprecated for 3.6-flash
  and all later models), `thought_signature` values returned by Gemini are
  captured and **replayed verbatim** on subsequent model calls (Gemini 3.x
  baseline), `HttpOptions.timeout` is milliseconds (seconds converted once),
  and `HttpRetryOptions(attempts=1)` pins the no-auto-retry rule. Errors are
  translated to typed classes via HTTP status: 429/5xx/timeout →
  fallback-eligible; 401/403, other 4xx (incl. **HTTP 499 `CANCELLED`**),
  safety refusals, network/parse errors → not eligible. 4xx messages include
  the SDK's structured `status`/one-line `message` (sanitized; the full body
  `.details` is never read).
- `backend/app/agent.py` — LangGraph state graph (`llm` ↔ `tools`,
  `recursion_limit`, `InMemorySaver`-compatible checkpointing via
  `get_checkpointer()`): each turn passes ONLY the new user message;
  `verified_context` accumulates validated Shopify tool results
  (`search_catalog`, `product_stock` — exactly two read-only tools; each
  Shopify query runs at most once per turn). Fallback gates: eligible Gemini
  failure + fallback enabled + NVIDIA key + **verified Shopify context**.
  Approved addition: when Gemini fails BEFORE tool selection on an explicit
  `SKU-…`/`sku:` reference, ONE deterministic read-only lookup grounds the
  fallback (never the same lookup twice; never any invented product data).
  Fallback outcome is generation-only Nemotron over the verified context.
- `backend/app/chat.py` — `POST /api/chat` contract: 200
  `{reply, session_id, tools_used, model_used, fallback_used}`; 400 empty
  message; 500 config; 503 controlled provider/store failures (with sanitized
  gate reason categories); Shopify tool failures keep their exact
  401/429/502/504 mapping.
- `backend/app/nvidia_client.py` — OpenAI-compatible NVIDIA client
  (`max_retries=0`), generation-only (no tools), same never-invent rules.

**Phase M1 (persistent conversation memory):**
- `backend/app/memory_db.py` — `get_checkpointer()`: when `NEON_DATABASE_URL`
  is configured, the official `PostgresSaver` (`langgraph-checkpoint-postgres`)
  over ONE source-verified thread-safe long-lived psycopg3 connection
  (options mirror `from_conn_string`: `autocommit=True`,
  `prepare_threshold=0`, `row_factory=dict_row`) with **once-only idempotent
  `setup()`** under a start lock; unconfigured → the cached `InMemorySaver`
  (exact pre-M1 behavior). A configured-but-unreachable store raises
  `MemoryDependencyError` → controlled 503 — **never a silent downgrade**;
  mid-operation store failures (`psycopg.Error`) map to the same 503.
- Test isolation: `backend/tests/conftest.py` — autouse fixture pins every
  test to in-memory mode (never reads the real DSN, never connects to Neon)
  with explicit fake-Neon stubs only where documented.

**Frontend.** Initial React+TS+Vite scaffold only (`frontend/src/App.tsx`,
hero image, oxlint config). No API integration on the frontend yet. Tests:
backend-only (pytest); frontend verified via `npm run build`.

**Routes:**
| Route | Status | Since |
|---|---|---|
| `GET /health` | implemented (contract frozen) | Phase 1 commit |
| `GET /api/products` (`limit`, `cursor`) | implemented + live-tested | Phase 4 |
| `GET /api/products/search?q=` | implemented (same pagination) | Phase 4 |
| `GET /api/products/inventory?limit&cursor&detail` | implemented (`detail=quantities` default; `detail=levels` = strict `read_inventory` probe, sending `quantities(names: ["available"])`) | Phase 6/7 (live strict probe: PASS, see §5/§6) |
| `POST /api/chat` | implemented; live-verified end-to-end (see §5) | Phase 8 (memory M1) |
| Checkout / Telegram / RAG / docs-retrieval / recommendations / dashboard | not implemented | planned (§8) |

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

All commits below are **verified pushed to `origin/main`** (local HEAD ==
origin/main checked after each push; fast-forward only; no force-push ever).

| Commit | Message | Phase | Suite at that commit |
|---|---|---|---|
| `b6d6280` | `chore: initialize Shopify AI sales agent project` | Phases 1–3: frontend scaffold, FastAPI `/health` + contract test, env/dependency foundation, `.gitignore` | 1 test |
| `5cd3ae6` | `feat: add Shopify product retrieval API` | Phase 4: GraphQL product listing/search, pagination, typed errors | 22 tests |
| `6b3688e` | `feat: add Shopify authentication and fix product query` | Phase 5: client-credentials auth + token refresh; silent `.env` loading; scalar-`Money` price fix | 39 tests |
| `ffa908f` | `feat: add read-only Shopify inventory endpoint` | Phase 6: `GET /api/products/inventory` (`inventoryQuantity` scalar) | 44 tests |
| `1b1420d` | `feat: add strict read_inventory levels probe` | Phase 7: `detail=levels` strict inventory chain, response split, docs, 6 new mocked tests + `names` regression | 50 tests |
| `88bfa4c` | `Complete Phase 8 LangGraph sales agent` | Phase 8: Gemini 3.6-flash primary (no sampling params, thought-signature replay, sanitized structured 4xx diagnostics), read-only Shopify tool grounding, gated generation-only NVIDIA Nemotron fallback incl. approved deterministic SKU retrieval, HTTP 499 CANCELLED non-retryable policy + test | 108 tests |
| `e8ca2ae` | `Add Neon persistent conversation memory` | Phase M1: Neon-backed PostgresSaver checkpointer (pinned `langgraph-checkpoint-postgres==3.2.0`), in-memory fallback, no-silent-downgrade error mapping, mid-operation failure mapping, `.env.example` Neon docs with corrected table grants, test-isolation fixture | 120 tests |

---

## 5. Live testing evidence

Legend: **[verified by agent]** = reproduced through outputs the agent obtained;
**[user-reported]** = performed by the user on their real store/services;
**[mocked]** = `httpx.MockTransport`, no network.

- **Phase 4/5 — product retrieval:** `GET /api/products?limit=3` returned real
  Shopify product data. [user-reported] First live attempt (during Phase 5)
  exposed an invalid nested selection on `ProductVariant.price` (`Selections
  can't be made on scalars (field 'price' returns Money …)`); fixed to the
  bare scalar `price`; user confirmed a successful retry; preserved in a
  regression test. [verified by agent, mocked]
- **Phase 6 — inventory quantity probe:** `GET /api/products/inventory?limit=5`
  returned HTTP 200 with real per-variant quantities. [user-reported]
  ⚠ Scope caveat: `inventoryQuantity` may be `read_products`-gated — see §6.
- **Phase 7 — `detail=levels` strict probe: TESTED-PASS (live).** [user-reported]
  First attempt failed with `Field 'quantities' is missing required arguments:
  names`; fixed to `quantities(names: ["available"])` + regression test;
  second attempt returned five products with location-level quantities
  (e.g. `{"name": "available", "quantity": 50}`), `has_next_page: true`
  (HTTP status not displayed, not recorded). Scope-attribution caveat in §6.
- **Phase 8 — chat agent (one approved live request + user-side run):**
  - The approved live `POST /api/chat` (SKU-Hat question) returned **HTTP 200**
    with `fallback_used: true`, `model_used: nvidia/nemotron-3-ultra-550b-a55b`,
    `tools_used: ["search_catalog", "product_stock"]` — Gemini executed both
    Shopify tools, a later Gemini step failed (eligible class), and the fallback
    answered **grounded** in verified store data. [verified by agent]
  - The answer honestly declined SKU-Hat details that were NOT in the verified
    store data (never-invent rule working as approved). Whether the real store
    carries `SKU-Hat` remains unconfirmed store-datum — not a code question.
  - Earlier live rounds observed Gemini provider-side failures: HTTP 503
    (native Google error payload) and HTTP 499 `CANCELLED`; both are classified
    correctly (503 → fallback-eligible; 499 → non-retryable, now pinned by a
    test). [verified by agent via sanitized detail strings]
- **Phase M1 — Neon persistence: verified live across a backend restart.**
  [user-reported] With a real `NEON_DATABASE_URL` configured, the checkpoint
  tables were created on first use and conversation state survived a FastAPI
  restart (same session). The offline suite exercises the same code paths with
  stubs only.
- **Auth/token flow:** client-credentials token acquisition passed all mocked
  tests and is exercised by every live store request above; no separate
  live token-lifecycle observation has been made.
- **All 120 current backend tests are fully mocked/offline.** No automated
  test touches the live store, Neon, Gemini, or NVIDIA (enforced by
  `backend/tests/conftest.py`). Run `pytest` anytime — it never sends traffic.

---

## 6. Permissions matrix (record only verified results)

| Scope | Purpose / feature | Implemented? | Test method | Result | Remaining uncertainty |
|---|---|---|---|---|---|
| `read_products` | Product listing + search (+ price/variant/SKU data) | Yes (Phases 4–5) | Live `GET /api/products?limit=3` through FastAPI/Swagger | **TESTED-PASS** [user-reported] | None known; all fields returned |
| `inventoryQuantity` fields (scalar quantities per variant) | Inventory-aware availability in answers | Yes (Phase 6) | Live `GET /api/products/inventory?limit=5` | **TESTED-PASS** [user-reported] | ⚠ Which scope gates this field is not publicly unambiguous (`read_products` vs `read_inventory`); success does **not** independently prove `read_inventory` |
| `read_inventory` (strict, `InventoryLevel`) | Location-level stock (planned stock-aware feature) | Probe implemented (Phase 7), feature not built | Live Swagger `GET /api/products/inventory?limit=5&detail=levels` (after the `names` query fix) | **TESTED-PASS — strict retrieval** [user-reported]: 5 products with location-level quantities (e.g. `{"name":"available","quantity":50}`), `has_next_page: true`; HTTP status not displayed, not recorded | **Scope-attribution caveat:** shopify.dev documents `InventoryLevel` as requiring `read_inventory`, so the strict chain strongly indicates the grant — but this is not framed as definitive, independently falsified/confirmed scope verification (per §9(11), dashboard scope lists were never used to declare a result). Null/empty `inventory_levels` are data conditions, not scope verdicts; that case never arose. |
| `read_inventory` (legacy fields e.g. `quantityAvailable`) | Fallback knowledge | n/a | documented only | NOT TESTED | We never relied on them |
| Storefront `unauthenticated_read_product_listings` + `SHOPIFY_STOREFRONT_ACCESS_TOKEN` | Future storefront-facing display | No | Storefront GraphQL probe (`/api/{v}/graphql.json`) — **not planned until the feature exists** | NOT TESTED | Requires creating a storefront token (app-config change → user approval first) |
| `write_*` scopes | Product/order mutation | **Never** — agent is read-only | not applicable | Deliberately excluded | — |

**Key distinction (teacher's requirement):** scopes shown in the Shopify
dashboard are *claims*; a scope is only *proven* when the specific read
operation succeeds through the API of record. The dashboard was never used to
declare a result in any row above.

---

## 7. Current project state (verified at handover time)

- **Branch:** `main`. **HEAD:** `e8ca2ae` (`Add Neon persistent conversation
  memory`) = `origin/main`; full log verified this round:
  `b6d6280` → `5cd3ae6` → `6b3688e` → `ffa908f` → `1b1420d` → `88bfa4c` →
  `e8ca2ae`.
- **Working tree:** clean ("nothing to commit"); nothing staged; Phase 8 and
  Phase M1 are fully committed AND pushed.
- **Tests:** 120 passed, 0 failed; `pip check` clean; `git diff --check`
  clean (all re-verified immediately before the M1 push and again this round
  via `pip check`).
- **`backend/.env`:** exists with real-looking domain/client credentials and
  (since M1) a real `NEON_DATABASE_URL` (real live connection proven; values
  never inspected or displayed; git-ignored, never staged/tracked).
- **Exact stop point:** all committed work is pushed. The next task (§8) is
  the Telegram phase: read-only inspection of the Inventra project for
  reusable integration code, then connecting Telegram to the existing agent.
  Nothing of that is started.

---

## 8. Pending work and recommended order

**Immediate next task (user-approved plan).**
1. **Inventra inspection (read-only).** Locate the earlier Inventra project on
   the user's machine (ask for its path if not obvious), read its Telegram
   integration code read-only, and report what is actually implemented vs
   assumed — do **not** assume Inventra's Telegram code is complete or working
   until inspected. No code copying without the user's explicit approval and
   clean adaptation (token/webhook-secret env names must map to
   `TELEGRAM_BOT_TOKEN` / `TELEGRAM_WEBHOOK_SECRET`).
2. **Connect Telegram to the existing agent.** Chat surface over the existing
   `run_chat()` result contract (`reply`, `session_id`, `fallback_used`…);
   map provider errors to safe Telegram messages; webhook verification per
   Telegram rules; long polling vs webhook to be decided with the user.
   Secrets handled exactly like every other env var (never printed/staged).

**Then (planned order after checking the teacher's actual specification):**
FAQs/policy retrieval (semantic memory / RAG — Neon + embeddings), product
recommendations, remaining frontend work, security hardening, caching/limiting
(`REDIS_URL` placeholder), admin dashboard/WebSocket, deployment, and testing
requirements. Confirm each against the teacher's actual specification before
building; do not self-invent requirements.

**Feature backlog (verified-not-started).** Telegram integration (next),
FAQ/policy retrieval, product recommendations, checkout flow (likely
Storefront API; read-only or the teacher's decision), admin dashboard,
security hardening, rate limiting, deployment.

**Optional improvements (nice-to-have).**
- Request-lifetime Shopify client/token-manager singleton (currently fresh
  per request; fine at current scale).
- psycopg pooling (Neon latency): current design is one cached long-lived
  connection guarded by the saver's internal lock; revisit only if live
  performance shows a need.
- Root README; `requirements-proof.txt` documentation notes.
- GraphQL cost/complexity headers for throttling telemetry.

---

## 9. Known issues and decisions (do not reverse without discussion)

**Phase 1–7 decisions (still binding).**
1. **Scalar `Money` price: `ProductVariant.price` is a monetary string;
   currency comes from `priceRange` (`MoneyV2`).** Forced by a live schema
   error; regression-tested. Reverting breaks live calls.
2. **GraphQL error policy: any `errors` array rejects the whole response**
   (502). Never upgrade to partial-result merging without a structural reason
   and new tests.
3. **Auth precedence: client credentials > static token**; placeholder-looking
   values are treated as unset (helpful `ConfigurationError` instead of a
   doomed call).
4. **`SHOPIFY_API_VERSION` validation** against a frozen supported list
   (`2026-01`, `2026-04`, `2026-07`, `2026-10`); bump only after re-checking
   shopify.dev.
5. **No auto-retries.** 429 answers carry `Retry-After` (upstream or a
   documented 5 s fallback), but the backend never retries automatically.
6. **One `httpx.Client` per request** (no pooling) — known inefficiency,
   acceptable; do not "fix" casually.
7. **Commit/push only after verification** (full pytest + `pip check` +
   staged-diff secret scan) and only when the user asks. Never
   reset/rebase/force/discard.
8. **Test counts per commit are historical facts** (22→39→44→50→108→120). Do
   not re-align old commits' numbers when the suite grows.
9. **Docs read snippet-level.** shopify.dev could not be fetched directly in
   some rounds (DNS blocks); schema claims relied on official-page snippets
   plus live-error behavior. When a live error contradicts a doc snippet,
   trust the live error and re-verify the schema.

**Phase 8 decisions (LLM agent — do not regress these).**
10. **No sampling parameters are ever sent to Gemini.**
    `temperature`/`top_p`/`top_k`/`candidate_count` are deprecated for
    `gemini-3.6-flash` and ALL later Google models (official migration guide:
    "remove from all requests"; may return HTTP 400). Determinism comes from
    the strict system prompt (answers only from verified Shopify tool data).
    Do not reintroduce them "to tune" behavior — remove-and-verify instead.
11. **Thought signatures are captured and replayed verbatim** on persisted
    model turns (Gemini 3.x baseline for multi-turn function calling;
    verified SDK behavior: inbound base64-decode → bytes in state →
    base64 re-encode on the wire).
12. **Gemini error classification stays:** 429/5xx/timeout → fallback-eligible;
    401/403, other 4xx (incl. HTTP 499 `CANCELLED` — pinned by a dedicated
    test), safety refusals, network/parse errors → NOT eligible. 4xx messages
    embed the SDK's structured `status`/sanitized one-line `message`; the full
    body `.details` is never read and never echoed.
13. **Never-invent rule:** agent answers use ONLY verified Shopify data.
    Missing/null data means "unknown", never invented. The NVIDIA fallback is
    generation-only (no tools, no second data source) and runs only when ALL
    gates open: eligible Gemini failure + fallback enabled + NVIDIA key +
    verified context. Approved grounding extension: for explicit `SKU-…`
    references with no context (Gemini failed before tool selection), ONE
    deterministic read-only lookup grounds the fallback (`product_stock`
    first, one distinct `search_catalog` attempt only on no-match — never the
    same lookup twice).
14. **Controlled-503 diagnosability:** gate-closure reasons are static,
    sanitized categories surfaced in the 503 detail (never bodies/DSNs).

**Phase M1 decisions (memory).**
15. **Storage split:** persistent conversation memory uses Neon PostgreSQL
    via `langgraph-checkpoint-postgres` when `NEON_DATABASE_URL` is valid;
    without it the app keeps the exact in-memory behavior. An unreachable
    configured store raises a sanitized `MemoryDependencyError` → controlled
    503 — **NEVER silently downgrade to memory** (that would invisibly lose
    persistence). Mid-operation `psycopg.Error` maps the same way (history
    itself stays server-side; nothing is silently lost).
16. **Connection lifecycle (source-verified):** ONE long-lived psycopg
    connection mirroring `from_conn_string` options (`autocommit=True`,
    `prepare_threshold=0`, `row_factory=dict_row`); `PostgresSaver`'s own
    `threading.Lock` serializes all DB ops; `setup()` runs once (idempotent).
    Revisit pooling (psycopg-pool) only with live performance evidence.
17. **Test isolation:** `backend/tests/conftest.py` pins every test to
    in-memory mode and clears the singleton between tests; the fake-Neon
    factory test opts into its stub explicitly. The suite NEVER reads the
    real DSN or touches real Neon in any environment or ordering.
18. **Memory-type plan (for later phases):** short-term = the checkpointer
    (done); episodic/long-term = LangGraph Store namespaces sharing the same
    Neon DB (not started); semantic = policies/FAQs (pgvector later), procedural
    = system prompt/tools kept in code. Do not build Store-based memory before
    the corresponding feature (FAQ retrieval) is approved.
19. **Checkpointer table grants (docs):** dedicated role over the actual
    setup-created tables `checkpoints`, `checkpoint_blobs`,
    `checkpoint_writes` (read/write) + `checkpoint_migrations` (SELECT) —
    never the admin superuser.

---

## 10. Instructions for the next AI assistant

1. **Read this document first**, then **verify the real repository state**
   (`git status`/`git log`/tests) before changing anything — never assume a
   feature is complete because it is listed here.
2. Continue from §7 (`e8ca2ae`, clean tree). If the tree differs, ask the user
   what changed before touching anything.
3. Work in **small, beginner-friendly steps**; explain concepts simply; do not
   bundle unrelated changes.
4. Respect boundaries: **Project 2 only**; Inventra inspection is READ-ONLY
   (never modify it, no wholesale copying without approval); never
   grant/revoke Shopify scopes; strictly read-only Shopify operations; never
   fetch/print tokens, secrets, DSNs, or `.env` contents.
5. **Permission/live tests are guided, one at a time.** Give exact steps,
   state the expected signal in advance, then record the actual result.
6. **Never commit or push without** the backend suite + `pip check` +
   staged-diff secret scan, showing the user what will be committed, and
   waiting for explicit approval. No resets, rebases, force-pushes, discards.
7. When a live test needs the user's action, **stop and wait** for their
   result; do not claim connectivity that has not been demonstrated.
8. Maintain this handover as the project grows (append phase outcomes, keep
   §6 and the decisions current, mark unverified claims clearly).

---

## 11. New Chat Continuation Prompt

> Copy everything inside the block below into a new conversation to resume
> exactly where this session left off. It states the verified state, the
> binding decisions, the exact next task, and the working style — so nothing
> already completed gets redone.

```text
Continue the Shopify AI Sales Agent project (Project 2). Read the repo and
docs/PROJECT_HANDOVER.md first, then verify the real state with git
status/log and a test run before touching anything.

VERIFIED STATE (do not redo any of this):
- Repo: shopify-ai-sales-agent; remote origin = github Csree94/shopify-ai-sales-agent;
  branch main, clean tree; HEAD = e8ca2ae "Add Neon persistent conversation
  memory" (= origin/main). Prior commits chain: b6d6280, 5cd3ae6, 6b3688e,
  ffa908f, 1b1420d, 88bfa4c "Complete Phase 8 LangGraph sales agent",
  e8ca2ae. Never amend or force-push.
- Backend: FastAPI + LangGraph sales agent. Primary LLM = Gemini
  (gemini-3.6-flash via GEMINI_API_KEY/GEMINI_MODEL), fallback = NVIDIA
  Nemotron (generation-only; NVIDIA_API_KEY), read-only Shopify GraphQL Admin
  tools (search_catalog, product_stock), chat endpoint POST /api/chat.
  Persistent per-session conversation memory in Neon PostgreSQL
  (langgraph-checkpoint-postgres, NEON_DATABASE_URL); in-memory fallback when
  unconfigured; failing configured store => controlled 503, never a silent
  downgrade. Test isolation: backend/tests/conftest.py pins every automated
  test to in-memory mode (never real Neon).
- Suite: 120 backend tests, fully offline; pip check clean; git diff --check
  clean. Real Neon persistence across a backend restart was verified live.
  Real store data verified live via /api/products and
  /api/products/inventory?detail=levels URLs; client-credentials Shopify auth.

BINDING CONSTRAINTS AND DECISIONS (from PROJECT_HANDOVER.md §9 — do not regress):
- Shopify is the ONLY source of truth for prices/stock; missing data = say
  unknown; never invent product facts; agent answers need verified Shopify
  context (fallback runs only when all gates open; explicit SKU questions can
  ground ONE deterministic read-only lookup when Gemini fails before tool
  use).
- Never send sampling parameters (temperature/top_p/top_k) to Gemini 3.6+;
  replay Gemini 3.x thought signatures verbatim on model turns.
- HTTP 499 CANCELLED is non-retryable (pinned by test); 429/5xx/timeout are
  fallback-eligible; 4xx/auth/safety are not.
- Read-only Shopify only; no write_* scopes ever; Admin GraphQL 2026-10 only.
- Never print/echo .env values, API keys, tokens, or the Neon DSN; all
  credential reprs are redacted; nothing logged.
- No auto-retries anywhere; no code changes or commits without my approval;
  full pytest + pip check + secret scan before any commit; no
  resets/rebases/force-pushes.

EXACT NEXT TASK (Phase: Telegram, per the approved plan):
1. Inspect the earlier "Inventra" project READ-ONLY to identify reusable
   Telegram integration code. First find the path (ask me if not obvious),
   read-only only: report what it actually implements (bot client creation,
   polling vs webhook, webhook verification, message routing, error mapping)
   and what is missing — do not assume it is complete or working until
   inspected. Do not copy anything without my explicit approval.
2. Then connect Telegram to the existing agent: a Telegram surface that
   reuses run_chat()'s existing result fields (reply/session_id/
   fallback_used), maps provider/store errors to safe Telegram messages,
   validates Telegram webhook secrets, and treats TELEGRAM_BOT_TOKEN /
   TELEGRAM_WEBHOOK_SECRET exactly like every other secret in this project.
   Design polling-vs-webhook with me before building.

AFTER THAT (planned order — confirm each against my teacher's actual
specification before building; do not self-invent requirements):
FAQs/store-policy retrieval (semantic memory / RAG on Neon), product
recommendations, remaining frontend, security hardening, caching/limiting,
admin dashboard/WebSocket, deployment, teacher-spec testing.

WORKING STYLE: small beginner-friendly steps; explain concepts simply;
distinguish verified facts from user-reported facts and from NOT-TESTED
items; keep docs/PROJECT_HANDOVER.md updated as the source of truth; ask
before any live API call, secret access, or git operation; stop for review
at each approved step boundary.
```
