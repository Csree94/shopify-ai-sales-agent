"""FastAPI entrypoint for the Shopify AI Sales Agent backend.

Phase 3: health endpoint. Phase 4: read-only Shopify product retrieval.
No Telegram, RAG, agents, checkout or database code yet.
"""
from fastapi import FastAPI

from app.products import router as products_router

APP_NAME = "Shopify AI Sales Agent"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.include_router(products_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Predictable health payload for liveness checks."""
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}
