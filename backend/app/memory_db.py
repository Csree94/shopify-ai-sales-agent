"""Neon PostgreSQL checkpoint store (Phase M1).

Factory for the LangGraph checkpointer:

- When ``NEON_DATABASE_URL`` is validly configured, the conversation
  checkpointer is the official ``langgraph.checkpoint.postgres.PostgresSaver``
  over ONE long-lived psycopg3 connection. Lifecycle safety is
  source-verified: ``PostgresSaver`` serializes every database operation
  through its own ``threading.Lock`` (all ops go through ``_cursor()``,
  which wraps ``self.lock``), so exactly one thread touches the psycopg
  connection at a time — psycopg's basic thread-safety rule is thus met.
  The connection mirrors the library's own ``from_conn_string`` options
  (``autocommit=True``, ``prepare_threshold=0``, ``row_factory=dict_row``)
  minus the context-manager scope (the saver is process-lifetime).
  ``PostgresSaver.setup()`` (idempotent DDL) runs once at first use.
- When Neon is NOT configured, the factory returns ``InMemorySaver()`` —
  the unchanged local-development/test behavior.
- An unreachable misconfigured database NEVER silently degrades to memory
  (that would invisibly disable persistence); it raises
  ``MemoryDependencyError`` which the chat surface maps to a controlled 503.

The DSN is a secret: it lives only inside psycopg and is never printed,
logged, or echoed (``NeonSettings`` reprs are redacted).
"""

from __future__ import annotations

import threading
from typing import Callable

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import load_neon_settings

# Module-level test seam (same convention as agent.gemini_factory /
# agent.nvidia_factory): a callable returning the app's checkpointer.
checkpointer_factory: Callable[[], BaseCheckpointSaver] | None = None


class MemoryDependencyError(RuntimeError):
    """The configured Postgres checkpoint store could not be reached.

    Mapped by the chat surface to a controlled 503 — with a sanitized
    message that never contains the DSN or connection details.
    """


_postgres_checkpointer: PostgresSaver | None = None
_postgres_start_lock = threading.Lock()
_memory_checkpointer: InMemorySaver | None = None


def _build_postgres_saver(dsn: str) -> PostgresSaver:
    """Open the process-lifetime connection and prepare the checkpoint tables.

    ``setup()`` runs the library's idempotent migrations once. Any psycopg
    failure is re-raised as MemoryDependencyError without echoing the DSN.
    """
    from psycopg.rows import dict_row as _dict_row

    try:
        # Source-verified options (PostgresSaver.from_conn_string):
        # autocommit + prepare-threshold 0 + dict row factory.
        conn = psycopg.connect(
            dsn,
            autocommit=True,
            prepare_threshold=0,
            row_factory=_dict_row,  # type: ignore[arg-type]
        )
        saver = PostgresSaver(conn)  # type: ignore[arg-type]
        saver.setup()
    except Exception as exc:
        # Keep an already-prepared MemoryDependencyError intact; everything
        # else (psycopg failures, schema errors) becomes ONE sanitized kind —
        # never a DSN or connection detail, never a silent fallback.
        if isinstance(exc, MemoryDependencyError):
            raise
        raise MemoryDependencyError(
            "The configured Postgres checkpoint store is unreachable "
            "(connection or schema setup failed; details withheld)."
        ) from exc
    return saver  # type: ignore[return-value]


def _default_checkpointer() -> BaseCheckpointSaver:
    """Checkpointer when no test seam overrides it (cached per process).

    Both caches are decided under the same lock so concurrent first requests
    can never produce divergent saver instances.
    """
    global _postgres_checkpointer, _memory_checkpointer
    with _postgres_start_lock:
        if _postgres_checkpointer is not None:
            return _postgres_checkpointer
        if _memory_checkpointer is not None:
            # CACHED single instance: multi-turn continuity requires the same
            # saver across run_chat() calls (existing behavior, unchanged).
            return _memory_checkpointer
        settings = load_neon_settings()
        if settings is None:
            # Exact existing behavior for local development and tests.
            _memory_checkpointer = InMemorySaver()
            return _memory_checkpointer
        if _postgres_checkpointer is None:
            _postgres_checkpointer = _build_postgres_saver(settings.dsn)
        return _postgres_checkpointer  # type: ignore[return-value]


def get_checkpointer() -> BaseCheckpointSaver:
    """The app's conversation checkpointer (seam-aware, lifecycle-managed)."""
    factory = checkpointer_factory if checkpointer_factory is not None else _default_checkpointer
    return factory()


def reset_postgres_singleton() -> None:
    """Test helper: clears the cached Postgres saver (never used in app flow)."""
    global _postgres_checkpointer, _memory_checkpointer
    with _postgres_start_lock:
        _postgres_checkpointer = None
        _memory_checkpointer = None
