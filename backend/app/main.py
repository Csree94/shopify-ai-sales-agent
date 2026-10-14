"""Minimal FastAPI entrypoint for the Shopify AI Sales Agent backend.

Phase 3 skeleton: a single, predictable health endpoint.
No Shopify, Telegram, agents, RAG or database code yet.
"""
from fastapi import FastAPI

APP_NAME = "Shopify AI Sales Agent"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/health")
def health() -> dict[str, str]:
    """Predictable health payload for liveness checks."""
    return {"status": "ok", "app": APP_NAME, "version": APP_VERSION}
