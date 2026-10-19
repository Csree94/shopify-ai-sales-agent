"""Duplicate-update protection for the Telegram webhook (Stage 2).

Telegram may redeliver webhook updates; a duplicate run is NOT harmless (it
would cost a full LLM turn, post a duplicate reply, and pollute the
conversation checkpoint with a duplicate user turn). This module implements
the smallest robust design that uses ONLY the existing LangGraph checkpointer
(in-memory for local dev/tests, Neon ``PostgresSaver`` when configured):

Design (the three audited-in layers, scoped per Telegram user):

1. **Durable ledger** — one tiny dedicated StateGraph per user, checkpointed
   under thread ``"tg-dedup:{user_id}"`` by the SAME ``get_checkpointer()``
   the app already uses (no new database, no new models). Its single state
   channel ``seen_update_ids`` accumulates processed update ids through a
   custom reducer that deduplicates and caps the list at ``DEDUP_SEEN_CAP``
   most-recent ids — bounded state; checkpoint snapshots grow one per update
   exactly like ordinary conversation turns already do.

   **Checkpoint ops are the SYNC API run in a worker thread.** The
   production saver (Neon) is the SYNC ``langgraph.checkpoint.postgres
   .PostgresSaver`` whose ``aget_tuple``/``aput`` (and friends) raise
   ``NotImplementedError`` — only ``AsyncPostgresSaver`` implements the
   async surface. ``/api/chat`` never noticed because the conversation
   graph uses the sync ``invoke`` path; the claim graph used to await
   ``aget_state``/``ainvoke`` and therefore failed EVERY claim in
   production — while the in-memory saver used in tests implements the
   async surface natively, so the test suite stayed green. Claim
   reads/writes now go through ``asyncio.to_thread`` around the graph's
   sync ``get_state``/``invoke`` (both savers implement sync; the Neon
   connection stays serialized by the saver's own internal lock, and the
   per-session lock in ``app.telegram.service`` ensures one worker thread
   at a time per user).
2. **Per-process fast registry** — a bounded ``OrderedDict`` guarded by a
   ``threading.Lock``. The lock is held ONLY for non-awaiting dict
   operations; the ledger I/O happens outside it, and a final re-check under
   the lock decides the claim. Two concurrent duplicate deliveries in one
   process therefore cannot both claim (exactly one wins the guarded
   registration step).
3. **Per-session locking** — kept in ``app.telegram.service`` so two
   different updates from the same user cannot interleave checkpoint writes
   on the same conversation thread.

Claim order: registry check → ledger read → guarded re-check + registration
→ ledger write → agent turn. Claiming BEFORE the turn guarantees no
duplicate reply for a redelivered update (claim-first semantics).

**Precise guarantees and limitations — NOT exactly-once:**
- Single process: concurrent duplicates are prevented (2); restarts are
  covered by the durable ledger (1). Multi-worker deployments can still race
  (LangGraph checkpoints have no compare-and-swap): run ONE uvicorn worker.
- A process dying between the ledger write and the completed turn leaves the
  update claimed but unanswered (fail-safe: never duplicates; a rare crash
  may drop one message — Telegram does not redeliver after a 200 ack anyway).
- If the checkpointer is unreachable, the claim raises ``LedgerError``; the
  service answers with the friendly error and runs no agent, and the failed
  claim is un-registered so a later working attempt can claim it.
- LIVE INCIDENT (regression-tested): the process-lifetime Neon connection
  dies when the server suspends the idle compute between messages — every
  later checkpoint op then fails INSTANTLY on the broken socket
  (psycopg ``conn.broken``), wrapping as ``LedgerError`` for the rest of the
  process's life ("one update succeeded, the rest got the friendly
  error"). A transport-broken Postgres connection (detected by psycopg's
  own connection state — never by message text) now triggers ONE bounded
  reconnect (``memory_db.reopen_postgres_checkpointer``) and ONE bounded
  retry of the same claim op. Safe because the read is pure and the ledger
  write is idempotent (the deduplicating reducer swallows a repeated id;
  the in-process registry still lets exactly ONE turn run). Timeouts are
  deliberately NEVER retried (commit state unknown) and remain fail-safe.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

import app.memory_db as memory_db
from app.memory_db import get_checkpointer

DEDUP_THREAD_PREFIX = "tg-dedup:"
DEDUP_SEEN_CAP = 1000  # bounded ledger state (ids retained per user)
REGISTRY_CAP = 2048  # bounded per-process fast registry

# Bounded claim I/O: the production saver is a process-lifetime SYNC psycopg
# connection with no connect/keepalive/statement timeout (memory_db), so a
# dead or idle-dropped Neon TCP connection would otherwise block the claim
# thread indefinitely — silently holding the customer's per-session lock and
# stalling every later message (the live "pipeline start then silence" hang).
# Timeouts convert that stall into the existing LedgerError handling.
CLAIM_READ_TIMEOUT_SECONDS = 30.0
CLAIM_WRITE_TIMEOUT_SECONDS = 30.0

# Bounded ONE-TIME reconnect after a transport-dead Neon connection:
# a fresh psycopg connect + the saver's idempotent setup() DDL. Deliberately
# bounded like the claim ops; converts an unreachable rebuild into the
# existing fail-safe LedgerError handling (never an unbounded stall).
CLAIM_REOPEN_TIMEOUT_SECONDS = 45.0

_logger = logging.getLogger("app.telegram")


def _cap_seen(existing: list[int] | None, new: list[int]) -> list[int]:
    """Deduplicate and cap the ledger channel (most-recent ids win)."""
    merged: OrderedDict[int, None] = OrderedDict()
    for update_id in existing or []:
        merged[update_id] = None
    for update_id in new:
        merged[update_id] = None
        if len(merged) > DEDUP_SEEN_CAP:
            merged.popitem(last=False)  # evict the oldest tracked id
    return list(merged.keys())


class DedupState(TypedDict, total=False):
    """State of the tiny dedicated dedup graph.

    ``seen_update_ids`` accumulates through the bounded ``_cap_seen``
    reducer: append-dedup-cap, always keeping the most recent ids.
    """

    seen_update_ids: Annotated[list[int], _cap_seen]


class LedgerError(RuntimeError):
    """The dedup ledger (checkpointer) could not be read or written."""


def dedup_thread_id(user_id: int) -> str:
    """Thread key for one Telegram user's durable dedup ledger."""
    return f"{DEDUP_THREAD_PREFIX}{user_id}"


def _claim_cause_chain(exc: BaseException) -> str:
    """The exception TYPE chain only — never messages, never DSNs/credentials.

    The classification of the wrapped checkpoint failure (connection reset
    vs. schema/auth error) is exactly what the sanitized ``LedgerError``
    hides, so the failure line carries the cause chain's TYPE names.
    """
    names = [type(exc).__name__]
    node = exc.__cause__ or exc.__context__
    hops = 0
    while node is not None and hops < 5:
        names.append(type(node).__name__)
        node = node.__cause__ or node.__context__
        hops += 1
    return "<-".join(names)


async def _reconnect_and_retry(
    stage: str,
    update_id: int,
    failed_exc: Exception,
    checkpointer: BaseCheckpointSaver,
    timeout_seconds: float,
    make_call: Callable[[BaseCheckpointSaver], Callable[[], object]],
) -> object | None:
    """ONE bounded reconnect + retry of a claim op after a DEAD connection.

    Triggered ONLY when the failing graph's own checkpointer reports
    psycopg's ``conn.broken`` — psycopg marks a socket the server dropped
    (Neon idle suspension), NOT a clean shutdown, NOT a schema or
    permission error. The claim then runs ONE bounded reconnect (``memory_db.reopen_postgres_checkpointer``), recompiles the
    claim graph against the fresh saver, and retries the op once.

    Safety case analysis:
    - Read retry: pure read; nothing to undo.
    - Write retry: the ledger reducer dedupes a repeated update id and the
      per-process registry already holds the claim, so exactly ONE turn can
      run regardless of whether the first write committed.
    - ``None`` return (not broken / reconnect failed / retry failed):
      the caller's existing fail-safe handling runs unchanged.
    - Timeouts are never retried (commit state unknowable in budget).
    """
    dead = memory_db.checkpointer_connection_broken(checkpointer)
    if not dead:
        return None
    _logger.warning(
        "telegram claim stage=%s failed exception_type=%s cause_chain=%s "
        "dead_connection=true action=reconnect_and_retry update_id=%s",
        stage,
        type(failed_exc).__name__,
        _claim_cause_chain(failed_exc),
        update_id,
    )
    try:
        await asyncio.wait_for(
            asyncio.to_thread(memory_db.reopen_postgres_checkpointer),
            timeout=CLAIM_REOPEN_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        _logger.warning(
            "telegram claim stage=%s reconnect timed out "
            "after_seconds=%.0f update_id=%s exception_type=TimeoutError",
            stage,
            CLAIM_REOPEN_TIMEOUT_SECONDS,
            update_id,
        )
        return None
    except Exception as exc:
        _logger.warning(
            "telegram claim stage=%s reconnect failed exception_type=%s "
            "cause_chain=%s update_id=%s",
            stage,
            type(exc).__name__,
            _claim_cause_chain(exc),
            update_id,
        )
        return None
    # Build the retry call against the FRESH saver (graph recompiled via the
    # checked-out factory so no stale connection is reused).
    retry_graph = build_claim_graph()
    retry_started = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(make_call(retry_graph)),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        _logger.warning(
            "telegram claim stage=%s retry timed out "
            "after_seconds=%.0f update_id=%s exception_type=TimeoutError",
            stage,
            timeout_seconds,
            update_id,
        )
        return None
    except Exception as exc:
        _logger.warning(
            "telegram claim stage=%s retry failed exception_type=%s "
            "cause_chain=%s update_id=%s",
            stage,
            type(exc).__name__,
            _claim_cause_chain(exc),
            update_id,
        )
        return None
    _logger.info(
        "telegram claim stage=%s event=end retry=true "
        "duration_ms=%d update_id=%s",
        stage,
        int((time.monotonic() - retry_started) * 1000),
        update_id,
    )
    return result


def build_claim_graph() -> CompiledStateGraph:
    """A fresh compiled claim graph bound to the app checkpointer.

    Compiled per call (cheap) so tests that reset the memory singleton see
    the correct saver and no stale graph is ever reused across resets.
    """
    graph = StateGraph(DedupState)

    def _record(state: DedupState) -> dict:
        # The claim was validated by the caller; the node simply completes so
        # the reduced (deduplicated, capped) state is checkpointed.
        return {}

    graph.add_node("record", _record)
    graph.add_edge(START, "record")
    graph.add_edge("record", END)
    return graph.compile(checkpointer=get_checkpointer())


def _claim_config(user_id: int) -> dict:
    return {"configurable": {"thread_id": dedup_thread_id(user_id)}}


class DuplicateGuard:
    """Per-process registry + durable ledger claim (one process)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent: OrderedDict[int, None] = OrderedDict()

    async def try_claim(self, user_id: int, update_id: int) -> bool:
        """Full claim operation; ``True`` when the caller should process.

        Raises ``LedgerError`` when the checkpointer read/write fails (the
        registry entry, if any, is then undone for this id).
        """
        graph = build_claim_graph()
        config = _claim_config(user_id)

        registry_key = (user_id, update_id)
        with self._lock:  # fast path only — no awaits inside
            if registry_key in self._recent:
                return False

        try:
            # SYNC checkpoint API in a worker thread, BOUNDED by a timeout (see
            # constants): the underlying sync psycopg ops cannot be cancelled
            # once started, so wait_for releases the event loop and converts a
            # dead-connection stall into the LedgerError handling below while
            # the abandoned worker thread dies off with the socket.
            started = time.monotonic()
            _logger.info(
                "telegram claim stage=claim_read event=start update_id=%s",
                update_id,
            )
            snapshot = await asyncio.wait_for(
                asyncio.to_thread(graph.get_state, config),
                timeout=CLAIM_READ_TIMEOUT_SECONDS,
            )
            _logger.info(
                "telegram claim stage=claim_read event=end "
                "duration_ms=%d update_id=%s",
                int((time.monotonic() - started) * 1000),
                update_id,
            )
            values = snapshot.values or {}
            seen = values.get("seen_update_ids") or []
        except asyncio.TimeoutError as exc:
            _logger.warning(
                "telegram claim stage=claim_read timed out "
                "after_seconds=%.0f update_id=%s exception_type=TimeoutError",
                CLAIM_READ_TIMEOUT_SECONDS,
                update_id,
            )
            raise LedgerError(
                "The Telegram dedup ledger is currently unavailable "
                "(details withheld)."
            ) from exc
        except Exception as exc:  # psycopg / checkpoint-store failures
            # Transport-dead connection (Neon idle suspension kills the idle
            # socket): ONE bounded reconnect against the failing graph's OWN
            # checkpointer + one bounded retry of the (pure) read. Every other
            # failure keeps the exact fail-safe behavior below.
            retried = await _reconnect_and_retry(
                "claim_read",
                update_id,
                exc,
                getattr(graph, "checkpointer", None),
                CLAIM_READ_TIMEOUT_SECONDS,
                lambda retry_graph: lambda: retry_graph.get_state(config),
            )
            if retried is None:
                _logger.warning(
                    "telegram claim stage=claim_read failed "
                    "exception_type=%s cause_chain=%s dead_connection=false "
                    "update_id=%s",
                    type(exc).__name__,
                    _claim_cause_chain(exc),
                    update_id,
                )
                raise LedgerError(
                    "The Telegram dedup ledger is currently unavailable "
                    "(details withheld)."
                ) from exc
            snapshot = retried
            values = snapshot.values or {}
            seen = values.get("seen_update_ids") or []
        if update_id in seen:
            with self._lock:
                self._register_recent(registry_key)
            return False

        # Claim (registry first — concurrent duplicates lose this guarded
        # re-check), then persist durably; undo the registration on failure.
        with self._lock:
            if registry_key in self._recent:
                return False
            self._register_recent(registry_key)
        try:
            started = time.monotonic()
            _logger.info(
                "telegram claim stage=claim_write event=start update_id=%s",
                update_id,
            )
            await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: graph.invoke(
                        {"seen_update_ids": [update_id]}, config
                    )
                ),
                timeout=CLAIM_WRITE_TIMEOUT_SECONDS,
            )
            _logger.info(
                "telegram claim stage=claim_write event=end "
                "duration_ms=%d update_id=%s",
                int((time.monotonic() - started) * 1000),
                update_id,
            )
        except asyncio.TimeoutError as exc:
            _logger.warning(
                "telegram claim stage=claim_write timed out "
                "after_seconds=%.0f update_id=%s exception_type=TimeoutError",
                CLAIM_WRITE_TIMEOUT_SECONDS,
                update_id,
            )
            # The write may or may not have been committed after the timeout —
            # the SAME fail-safe as an outright write failure: undo the
            # registration and raise LedgerError so NO agent turn runs on an
            # uncertain claim (claim-first semantics keep duplicates impossible).
            with self._lock:
                self._recent.pop(registry_key, None)
            raise LedgerError(
                "The Telegram dedup ledger could not record the claim "
                "(details withheld)."
            ) from exc
        except Exception as exc:
            # Same one-shot reconnect as the read: the retry is SAFE because
            # the ledger reducer DEDUPES a repeated update id and the
            # registry (already held) guarantees exactly one turn runs.
            retried = await _reconnect_and_retry(
                "claim_write",
                update_id,
                exc,
                getattr(graph, "checkpointer", None),
                CLAIM_WRITE_TIMEOUT_SECONDS,
                lambda retry_graph: lambda: retry_graph.invoke(
                    {"seen_update_ids": [update_id]}, config
                ),
            )
            if retried is None:
                # Fail-safe: unclaimed-with-certainty — un-register so a later
                # WORKING attempt can claim this update (never run the agent).
                with self._lock:
                    self._recent.pop(registry_key, None)
                _logger.warning(
                    "telegram claim stage=claim_write failed "
                    "exception_type=%s cause_chain=%s dead_connection=false "
                    "update_id=%s",
                    type(exc).__name__,
                    _claim_cause_chain(exc),
                    update_id,
                )
                raise LedgerError(
                    "The Telegram dedup ledger could not record the claim "
                    "(details withheld)."
                ) from exc
            # The retried write persisted durably; the registry entry stays:
            # the claim is now certain, and the turn runs exactly once below.
        return True

    def _register_recent(self, registry_key: tuple[int, int]) -> None:
        """Mark an (user_id, update_id) pair as handled in-process (bound
        least-recent cache — keys are per-user so distinct users never
        collide)."""
        self._recent[registry_key] = None
        while len(self._recent) > REGISTRY_CAP:
            self._recent.popitem(last=False)
        return None

    def reset(self) -> None:
        """Test helper: clears the per-process cache (never used app-side)."""
        with self._lock:
            self._recent.clear()


duplicate_guard = DuplicateGuard()


def reset_duplicate_guard() -> None:
    """Test helper mirroring ``memory_db.reset_postgres_singleton``."""
    duplicate_guard.reset()
