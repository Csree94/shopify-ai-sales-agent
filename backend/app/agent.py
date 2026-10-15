"""Gemini + LangGraph sales agent (Phase 8) with a gated NVIDIA fallback.

Architecture (approved plan):
- Primary: Gemini (`gemini-3.8-flash`, configurable) inside a LangGraph
  state graph ("llm" node ↔ "tools" node). Automatic function calling is
  disabled, so THIS module executes tools — each Shopify query runs at
  most once per turn and its validated result is stored in the graph
  state as *verified context*.
- Fallback: NVIDIA Nemotron is strictly GENERATION-ONLY. It receives the
  same customer question and the verified Shopify context snapshot; it
  has no tools and can never query Shopify or invent a second data
  source. Fallback runs only when ALL of these hold:
    * the Gemini failure is fallback-eligible (429/5xx/timeout),
    * verified Shopify context exists (never guess without it),
    * fallback is enabled (LLM_FALLBACK_ENABLED),
    * the NVIDIA API key is configured.
  If Gemini fails BEFORE any Shopify context was retrieved (planning
  phase, auth failure, safety refusal, bad request, or an application
  bug), a controlled error is raised instead — never an ungrounded guess.
- Shopify failures during tool execution are NOT fallback-eligible either
  (the context would be incomplete); they are re-raised for the route's
  existing 401/429/502/504 mapping.
- If both providers fail, one controlled error is raised; the HTTP layer
  maps it to a safe 503. Shopify data stays the only source of truth;
  missing/null data means "unknown", never invented facts.
- Conversation continuity uses LangGraph's InMemorySaver (in-process,
  keyed by session_id). No database, no Telegram, no writes.
- LangSmith tracing stays OFF (nothing here reads or sets LangSmith env).
"""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass, field
from typing import Annotated, Callable, TypedDict

from google import genai
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.config import (
    ConfigurationError,
    LLMSettings,
    load_llm_settings,
)
from app.gemini_client import (
    GeminiError,
    GeminiTimeoutError,
    GeminiUnavailableError,
    generate_once,
    make_gemini_client,
)
from app.nvidia_client import (
    NvidiaError,
    generate_completion,
    make_nvidia_client,
)
from app.shopify_client import (
    normalise_inventory_product,
    normalise_product,
)

MAX_TOOL_LOOPS = 3  # at most 3 tool round-trips per chat turn

CHAT_SYSTEM_PROMPT = (
    "You are the assistant of a real Shopify store. Answer customer "
    "questions using ONLY what the search_catalog and product_stock tools "
    "return. Product names, prices, variants, SKUs and stock quantities "
    "must come exclusively from those tool results — never invent them. "
    "If the tools return nothing relevant (or null values), say clearly "
    "that you don't have that information. Be brief, friendly, and "
    "read-only: you cannot place, change or cancel orders."
)

# The read-only Shopify tools (exactly two, per the approved plan).
TOOL_INPUT_SCHEMAS = {
    "search_catalog": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms, e.g. 'hat'."},
            "limit": {"type": "integer", "description": "Max products (default 5)."},
        },
        "required": ["query"],
    },
    "product_stock": {
        "type": "object",
        "properties": {
            "product_query": {
                "type": "string",
                "description": "Product title terms to look up stock for.",
            },
            "limit": {"type": "integer", "description": "Max products (default 3)."},
        },
        "required": ["product_query"],
    },
}

ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"


# -- errors surfaced to the HTTP layer ---------------------------------------


class AgentConfigError(RuntimeError):
    """LLM providers not configured (missing/invalid API key)."""


class ChatAgentError(RuntimeError):
    """Controlled failure after the fallback path was exhausted/unavailable."""

    def __init__(self, message: str, *, fallback_attempted: bool) -> None:
        super().__init__(message)
        self.fallback_attempted = fallback_attempted


# -- module-level provider factories (test seam, like products.client_factory) --

gemini_factory: Callable[[], tuple[genai.Client, str, str]] | None = None
nvidia_factory: Callable[[], tuple[OpenAI, str, str]] | None = None
shopify_tool_executor: dict[str, Callable[..., dict]] | None = None


def _default_gemini_unit() -> tuple[genai.Client, str, str]:
    settings: LLMSettings = load_llm_settings()
    return (
        make_gemini_client(settings.gemini),
        settings.gemini.model,
        CHAT_SYSTEM_PROMPT,
    )


def _default_nvidia_unit() -> tuple[OpenAI, str, str]:
    settings = load_llm_settings()
    if not settings.nvidia.api_key:
        raise AgentConfigError(
            "NVIDIA fallback is not configured (NVIDIA_API_KEY missing)."
        )
    return (
        make_nvidia_client(settings.nvidia),
        settings.nvidia.model,
        CHAT_SYSTEM_PROMPT,
    )


def _tool_functions(client):
    """Bind the two read-only tools to ONE production client instance."""

    def search_catalog(query: str, limit: int = 5) -> dict:
        connection = client.search_products(query=query, limit=limit)
        nodes = connection.get("nodes") or []
        return {
            "tool": "search_catalog",
            "products": [normalise_product(node) for node in nodes],
        }

    def product_stock(product_query: str, limit: int = 3) -> dict:
        connection = client.list_inventory(limit=limit, query=product_query)
        nodes = connection.get("nodes") or []
        return {
            "tool": "product_stock",
            "products": [normalise_inventory_product(node) for node in nodes],
        }

    return {"search_catalog": search_catalog, "product_stock": product_stock}


# -- LangGraph state and nodes ------------------------------------------------


class AgentState(TypedDict, total=False):
    # Append-reducers: the checkpointer accumulates across chat turns, so
    # every invoke passes only the NEW user message and each node returns
    # only its delta.
    contents: Annotated[list, operator.add]  # Gemini conversation (Content dicts)
    verified_context: Annotated[list, operator.add]  # validated Shopify results
    tools_used: Annotated[list[str], operator.add]
    tool_calls: list[dict]  # pending Gemini selections (replaced, not appended)
    answer: str
    finished: bool
    error_kind: str  # '' | ELIGIBLE | INELIGIBLE | SHOPIFY
    error_message: str


def _replayable_part(call_or_text, signature=None) -> dict:
    """One persisted model part that replays EXACTLY what Gemini returned.

    Function-call turns: {functionCall: {...}, thoughtSignature?...}; text
    turns: {text: ..., thoughtSignature?...}. The signature value is opaque
    (SDK bytes/str) and is never interpreted — only carried back.
    """
    if isinstance(call_or_text, str):
        part: dict = {"text": call_or_text}
    else:
        part = {
            "functionCall": {
                "name": call_or_text["name"],
                "args": call_or_text["args"],
            }
        }
        signature = call_or_text.get("thought_signature") or signature
    if signature:
        part["thoughtSignature"] = signature
    return part


def _new_input(message: str) -> dict:
    # Delta input: merged onto the checkpointed state via the add-reducers.
    return {"contents": [{"role": "user", "parts": [{"text": message}]}]}


def record_error(kind: str, message: str) -> dict:
    return {
        "error_kind": kind,
        "error_message": message,
        "finished": True,
    }


def llm_node(state: AgentState, *, client, model: str) -> dict:
    try:
        turn = generate_once(
            client,
            model=model,
            system_prompt=CHAT_SYSTEM_PROMPT,
            contents=list(state["contents"]),
            tool_declarations=_tool_declarations(),
        )
    except GeminiError as exc:
        kind = (
            ELIGIBLE
            if isinstance(exc, (GeminiUnavailableError, GeminiTimeoutError))
            else INELIGIBLE
        )
        return record_error(kind, str(exc))

    if turn.function_calls:
        return {
            "tool_calls": turn.function_calls,
            "contents": [
                {
                    "role": "model",
                    # Replay each part VERBATIM, carrying any Gemini 3.x
                    # thought signature it was returned with (best-effort:
                    # the SDK hands signatures back as an opaque value; the
                    # official migration checklist lists their preservation
                    # as a baseline Gemini 3.x requirement). Otherwise Gemini
                    # 3.x can reject the next model call as malformed.
                    "parts": [
                        _replayable_part(c)
                        for c in turn.function_calls
                    ],
                }
            ],
            "answer": "",
            "finished": False,
            "error_kind": "",
            "error_message": "",
        }
    if turn.text:
        return {
            # The model's own reply becomes part of the persisted
            # conversation so multi-turn sessions retain context (with any
            # thought signature it carried, per the Gemini 3.x baseline).
            "contents": [
                {"role": "model", "parts": [_replayable_part(turn.text, turn.thought_signature)]}
            ],
            "answer": turn.text,
            "finished": True,
            "error_kind": "",
            "error_message": "",
        }
    return record_error(
        INELIGIBLE, "Gemini returned neither text nor tool calls."
    )


def tools_node(state: AgentState, *, executor: Callable) -> dict:
    # `executor` maps tool name -> callable(**args) bound to ONE production
    # Shopify client; the state deltas (contents/verified_context/tools_used)
    # accumulate through the append-reducers.
    try:
        contents: list = []
        verified_context: list = []
        tools_used: list = []
        for call in state.get("tool_calls") or []:
            func = executor[call["name"]]
            result = func(**(call.get("args") or {}))
            tools_used.append(call["name"])
            verified_context.append(result)
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call["name"],
                                "response": {"name": call["name"], "content": result},
                            }
                        }
                    ],
                }
            )
    # Shopify failures are deliberately NOT caught here: they propagate out
    # of graph.invoke so the route's production mapping keeps the exact
    # subclass (401 auth / 429 throttle / 502 upstream / 504 timeout).
    except ValueError as exc:
        return record_error(INELIGIBLE, f"Tool argument problem: {exc}")
    return {
        "contents": contents,
        "verified_context": verified_context,
        "tools_used": tools_used,
        "tool_calls": [],
        "answer": "",
        "finished": False,
        "error_kind": "",
        "error_message": "",
    }


# Module-persistent in-memory memory store: multi-turn conversations survive
# across run_chat() calls for the same session_id (in-process only, no DB).
checkpointer = InMemorySaver()


def build_graph(client, model: str, executor: Callable):
    """Compile the checked two-node loop; each Shopify query runs at most
    once per turn because tool_calls are cleared after execution and the
    loop is capped by the recursion limit."""
    graph = StateGraph(AgentState)

    def _llm(state: AgentState) -> AgentState:
        return llm_node(state, client=client, model=model)

    def _tools(state: AgentState) -> AgentState:
        return tools_node(state, executor=executor)

    graph.add_node("llm", _llm)
    graph.add_node("tools", _tools)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges(
        "llm",
        lambda s: (
            "__end__"
            if s.get("finished")
            else ("tools" if s.get("tool_calls") else "__end__")
        ),
    )
    graph.add_conditional_edges(
        "tools",
        # A failed tool execution (or an error already recorded) must not
        # trigger another generation attempt — no ungrounded repetition.
        lambda s: "__end__" if s.get("error_kind") else "llm",
    )
    return graph.compile(checkpointer=checkpointer)


# -- orchestration ------------------------------------------------------------


@dataclass
class ChatOutcome:
    reply: str
    session_id: str
    tools_used: list[str] = field(default_factory=list)
    model_used: str = ""
    fallback_used: bool = False


def run_chat(message: str, session_id: str) -> ChatOutcome:
    """One chat turn; see module docstring for the full contract.

    Raises:
    - AgentConfigError: providers not configured (no valid keys).
    - ShopifyClientError: the Shopify tools failed (mapped by the route).
    - ChatAgentError: eligible provider failure without a successful fallback.
    - ValueError: empty/invalid request arguments (HTTP 422 family).
    """
    if not message.strip():
        raise ValueError("Chat message must not be empty.")

    try:
        gemini_client, gemini_model, _prompt = _gemini_unit()
    except ConfigurationError as exc:
        raise AgentConfigError(str(exc)) from exc

    executor = shopify_tool_executor or _tool_functions(
        _shopify_client_for_tools()
    )
    graph = build_graph(gemini_client, gemini_model, executor)

    final_state: AgentState = graph.invoke(
        _new_input(message),
        config={
            "configurable": {"thread_id": session_id},
            "recursion_limit": 2 * MAX_TOOL_LOOPS + 4,
        },
    )

    kind = final_state.get("error_kind", "")
    error_message = final_state.get("error_message", "")
    answer = final_state.get("answer", "")

    if not answer and kind == INELIGIBLE:
        raise ChatAgentError(
            f"Gemini failed with a non-retryable error: {error_message}",
            fallback_attempted=False,
        )
    if not answer:
        # kind == ELIGIBLE here (finished-no-answer); try the fallback gate.
        fallback, gate_reason = _try_fallback_gate(
            message=message,
            verified_context=final_state.get("verified_context", []),
            gemini_error=error_message,
            executor=executor,
        )
        if fallback is not None:
            fallback.session_id = session_id
            if not fallback.tools_used:
                # Gemini-path fallback: report the tools Gemini itself used.
                fallback.tools_used = final_state.get("tools_used", [])
            return fallback

    if not answer:
        # The controlled failure now states WHY the fallback gate did not
        # produce an answer (diagnosability), using only sanitized,
        # static categories — never response bodies or credentials.
        raise ChatAgentError(
            f"Gemini failed and no grounded answer was possible: "
            f"{error_message or 'unknown provider error'} ({gate_reason})",
            fallback_attempted=True,
        )
    return ChatOutcome(
        reply=answer,
        session_id=session_id,
        tools_used=final_state.get("tools_used", []),
        model_used=gemini_model,
    )


def _shopify_client_for_tools():
    from app.products import get_client  # local import: reuse the exact
    # production factory (same auth, clamps, and error handling).

    return get_client()


def _gemini_unit() -> tuple[genai.Client, str, str]:
    factory = gemini_factory if gemini_factory is not None else _default_gemini_unit
    try:
        return factory()
    except ConfigurationError as exc:
        raise AgentConfigError(str(exc)) from exc


def _tool_declarations():
    from google.genai import types as gtypes

    return [
        gtypes.FunctionDeclaration(
            name=name,
            description=(
                "Search the store catalog for products (names, prices, "
                "variants, SKUs)."
                if name == "search_catalog"
                else "Look up current stock quantities for matching products."
            ),
            parameters_json_schema=TOOL_INPUT_SCHEMAS[name],
        )
        for name in TOOL_INPUT_SCHEMAS
    ]


def _try_fallback_gate(
    *, message: str, verified_context: list, gemini_error: str, executor
) -> tuple[ChatOutcome | None, str]:
    """Run the generation-only fallback if every gate opens.

    Returns ``(outcome, "")`` on success, or ``(None, reason)`` when a gate
    closed. Reasons are sanitized, static categories so the controlled 503
    can state exactly why no fallback answer was produced (diagnosability).

    Gates: eligible Gemini failure (checked by the caller passing only ELIGIBLE),
    verified Shopify context present, fallback enabled, NVIDIA key configured,
    and a successful Nemotron completion. Expected NVIDIA failures
    (`NvidiaError`) close the gate and are reported with their static category;
    unexpected programming errors are NOT swallowed — they propagate so bugs
    stay visible instead of masquerading as provider outages.

    Approved Option B: when Gemini failed BEFORE tool selection (no verified
    context), a SKU-style product reference in the message can still be
    grounded through ONE deterministic read-only Shopify lookup (same executor
    functions and normalisers as the LangGraph tools) — the fallback may then
    answer from that real data. Without a verified product the gate stays
    closed (never-guess); Shopify lookup failures propagate and keep their
    exact non-fallback-eligible HTTP mapping (they are context failures,
    not provider outages).
    """
    settings = load_llm_settings_or_none()
    if settings is None:
        return None, "LLM configuration was unavailable"
    if not settings.fallback_enabled or not settings.nvidia.api_key:
        return None, "the fallback is disabled or the NVIDIA key is unconfigured"
    retrieval_tools: list[str] = []
    if not verified_context:
        # Deterministic grounding (approved Option B): only for explicit SKU
        # references. Nothing is ever served from anything the store did not
        # actually return — this replaces the previous unconditional close.
        context, retrieval_tools = _retrieve_verified_context(executor, message)
        if context:
            verified_context = context
        elif _extract_sku(message) is None:
            return (
                None,
                "no verified Shopify context existed and the message contains "
                "no SKU product reference — the fallback was not attempted "
                "(ungrounded answers are never served)",
            )
        else:
            return (
                None,
                "the SKU-style product reference could not be verified in the "
                "Shopify store, so the fallback was not attempted (ungrounded "
                "answers are never served)",
            )
    unit = _nvidia_unit(settings)
    if unit is None:
        return None, "the Nemotron fallback client could not be initialized"
    client, model, system_prompt = unit
    try:
        reply = generate_completion(
            client,
            model=model,
            system_prompt=system_prompt,
            user_question=message,
            verified_context_json=json.dumps(
                verified_context, ensure_ascii=False, default=str
            ),
        )
    except NvidiaError as exc:
        return None, f"the Nemotron fallback request failed: {exc}"
    return (
        ChatOutcome(
            reply=reply,
            session_id="",  # filled by the caller
            tools_used=retrieval_tools,  # truthful: what actually queried the store
            model_used=model,
            fallback_used=True,
        ),
        "",
    )


# -- deterministic SKU grounding helpers (approved Option B) ------------------

# "SKU-Hat", "sku: Hat", "SKU_Hat" ... — conservative: only messages with an
# explicit SKU reference get the deterministic lookup; every other question
# keeps the graph's exact current behavior.
_SKU_PATTERN = re.compile(r"\bSKU[-_:\s]*([A-Za-z0-9][\w-]{0,63})\b", re.IGNORECASE)
SKU_LOOKUP_LIMIT = 5


def _extract_sku(message: str) -> str | None:
    """The SKU capture from the message (e.g. 'Hat' from 'SKU-Hat'), or None."""
    match = _SKU_PATTERN.search(message)
    return match.group(1) if match else None


def _sku_matches_product(sku: str, result: dict) -> bool:
    """True when any returned variant carries the requested SKU (case-insensitive
    containment, so a captured 'Hat' matches the store SKU 'SKU-Hat').
    Validation happens ONLY on what the store returned — nothing is invented."""
    for product in result.get("products") or []:
        for variant in product.get("variants") or []:
            variant_sku = variant.get("sku")
            if variant_sku and sku.lower() in str(variant_sku).lower():
                return True
    return False


def _retrieve_verified_context(executor, message: str) -> tuple[list, list[str]]:
    """Deterministic read-only retrieval for SKU-style questions.

    Uses the SAME executor functions and normalisers as the LangGraph tools:
    one `product_stock` attempt (price + stock in one query), plus a single
    `search_catalog` attempt only when the inventory probe returned no SKU
    match — two DISTINCT stores-danger queries, never the same lookup twice.
    Shopify failures propagate (not fallback-eligible, exact HTTP mapping).
    """
    sku = _extract_sku(message)
    if sku is None:
        return [], []
    inventory = executor["product_stock"](
        product_query=f"sku:{sku}", limit=SKU_LOOKUP_LIMIT
    )
    if _sku_matches_product(sku, inventory):
        return [inventory], ["product_stock"]
    catalog = executor["search_catalog"](query=sku, limit=SKU_LOOKUP_LIMIT)
    if _sku_matches_product(sku, catalog):
        return [catalog], ["search_catalog"]
    return [], []


def load_llm_settings_or_none() -> LLMSettings | None:
    try:
        return load_llm_settings()
    except ConfigurationError:
        return None


def _nvidia_unit(settings: LLMSettings) -> tuple[OpenAI, str, str] | None:
    try:
        factory = (
            nvidia_factory if nvidia_factory is not None else _default_nvidia_unit
        )
        return factory()
    except AgentConfigError:
        return None
