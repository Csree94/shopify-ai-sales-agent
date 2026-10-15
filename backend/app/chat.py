"""HTTP surface for the Phase 8 AI sales agent.

`POST /api/chat` maps agent outcomes to a stable, non-streaming contract:

- 200 `{reply, session_id, tools_used, model_used, fallback_used}` on
  success (primary Gemini or fallback Nemotron; `fallback_used` is True
  only when the generation-only NVIDIA fallback produced the answer).
- 422 (pydantic) for malformed request bodies; 400 for an empty message.
- 500 when LLM configuration is missing/invalid (same convention as the
  Shopify configuration surface: named-variable message, no secrets).
- Shopify tool failures keep their exact existing mapping (401/429 with
  Retry-After/502/504) via the production translator.
- 503 when an eligible provider failure occurs and the fallback is
  disabled/unconfigured/also-failed, or an ineligible provider failure
  reaches the user (auth, safety refusal, bad request, app bug) — no
  ungrounded guesses are ever served instead.

No request or response field ever contains an API key; the reply values the
providers return are plain text answers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent import (
    AgentConfigError,
    ChatAgentError,
    ChatOutcome,
    run_chat,
)
from app.products import to_http_exception
from app.shopify_client import ShopifyClientError

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="Customer question.")
    session_id: str | None = Field(
        default=None, description="Client-chosen conversation key."
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tools_used: list[str] = Field(default_factory=list)
    model_used: str
    fallback_used: bool = False


def _session_id(session_id: str | None) -> str:
    # A missing client id gets a request-local session: the conversation is
    # still checkpointed in memory, just never continued from another call.
    return session_id if session_id and session_id.strip() else "__single_turn__"


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """One conversational turn with the read-only Shopify sales agent."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Chat message must not be empty.")
    session_id = _session_id(request.session_id)
    try:
        outcome: ChatOutcome = run_chat(request.message, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AgentConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ShopifyClientError as exc:
        raise to_http_exception(exc) from exc
    except ChatAgentError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(
        reply=outcome.reply,
        session_id=outcome.session_id,
        tools_used=outcome.tools_used,
        model_used=outcome.model_used,
        fallback_used=outcome.fallback_used,
    )
