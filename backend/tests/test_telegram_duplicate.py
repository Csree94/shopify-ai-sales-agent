"""Tests for the Telegram duplicate-update guard (Stage 2).

The guard runs against the in-memory checkpointer pinned by the conftest
(no network, no real Neon). Covers: durable claims across repeated calls,
per-user thread isolation, concurrent same-process claims, bounded ledger
retention, registry resets, and ledger-failure semantics (no silent claims,
no swallowed errors).
"""

from __future__ import annotations

import asyncio

import pytest

from app.telegram.dedup import (
    DEDUP_SEEN_CAP,
    DuplicateGuard,
    LedgerError,
    build_claim_graph,
    dedup_thread_id,
)


async def _read_seen(user_id: int) -> list[int]:
    graph = build_claim_graph()
    snapshot = await graph.aget_state(
        {"configurable": {"thread_id": dedup_thread_id(user_id)}}
    )
    return (snapshot.values or {}).get("seen_update_ids") or []


def test_first_claim_accepted_second_rejected():
    guard = DuplicateGuard()
    assert asyncio.run(guard.try_claim(42, 1001)) is True
    assert asyncio.run(guard.try_claim(42, 1001)) is False  # in-process registry
    # Durable: a fresh guard (simulated restart) still sees the ledger entry.
    fresh = DuplicateGuard()
    assert asyncio.run(fresh.try_claim(42, 1001)) is False


def test_claims_are_per_user():
    guard = DuplicateGuard()
    assert asyncio.run(guard.try_claim(42, 1001)) is True
    assert asyncio.run(guard.try_claim(7, 1001)) is True


def test_concurrent_duplicate_claims_accept_exactly_one():
    guard = DuplicateGuard()

    async def scenario() -> list[bool]:
        results = await asyncio.gather(
            guard.try_claim(42, 1002),
            guard.try_claim(42, 1002),
            guard.try_claim(42, 1002),
        )
        return list(results)

    assert asyncio.run(scenario()).count(True) == 1


def test_ledger_state_is_bounded():
    """Claiming more ids than the cap keeps only the most recent ones."""
    guard = DuplicateGuard()

    async def scenario() -> None:
        update_id = 0
        for _ in range(DEDUP_SEEN_CAP + 25):
            update_id += 1
            claimed = await guard.try_claim(42, update_id)
            assert claimed is True
        # Premature assertions must not mask the final ledger check.

    asyncio.run(scenario())
    seen = asyncio.run(_read_seen(42))
    assert len(seen) == DEDUP_SEEN_CAP
    # The most recent update id and the boundary id are retained; the oldest
    # evicted entries are gone.
    assert DEDUP_SEEN_CAP + 25 in seen
    assert 1 not in seen


def test_different_updates_from_one_user_all_claim():
    guard = DuplicateGuard()
    assert asyncio.run(guard.try_claim(42, 1001)) is True
    assert asyncio.run(guard.try_claim(42, 1002)) is True
    assert asyncio.run(guard.try_claim(42, 1003)) is True


def test_read_failure_raises_ledger_error(monkeypatch: pytest.MonkeyPatch):
    class ExplodingGraph:
        # The claim seam is the graph's SYNC state API (get_state/invoke) since
        # the production-only-API fix; see test_claim_works_with_sync_only_….
        def get_state(self, config: dict) -> object:
            raise RuntimeError("connection failed")

    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: ExplodingGraph())
    guard = DuplicateGuard()
    with pytest.raises(LedgerError):
        asyncio.run(guard.try_claim(42, 1004))


def test_write_failure_raises_and_allows_later_claim(
    monkeypatch: pytest.MonkeyPatch,
):
    class WriteExplodingGraph:
        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            raise RuntimeError("write failed")

    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: WriteExplodingGraph())
    guard = DuplicateGuard()
    with pytest.raises(LedgerError):
        asyncio.run(guard.try_claim(42, 1005))

    # The failed claim must be UN-registered so a later working attempt can
    # still claim the update (regression guard for the undo path).
    class WorkingGraph:
        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            return None

    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: WorkingGraph())
    assert asyncio.run(guard.try_claim(42, 1005)) is True


def test_read_timeout_raises_ledger_error_and_allows_later_claim(
    monkeypatch: pytest.MonkeyPatch,
):
    """LIVE-HANG REGRESSION: a Neon connection stalled on a dead TCP socket
    used to block the claim thread indefinitely (sync socket ops with no
    timeout), silently holding the customer's session lock — the observed
    'pipeline start then silence' hang. The claim read is now bounded: a
    timeout raises LedgerError, the registration is undone, and a later
    WORKING attempt can claim the same update."""
    import app.telegram.dedup as dedup
    import time as _time

    class HangingGraph:
        def get_state(self, config: dict) -> object:
            _time.sleep(1.0)  # stalls past the (patched) timeout
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            return None

    monkeypatch.setattr(dedup, "CLAIM_READ_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: HangingGraph())
    guard = DuplicateGuard()
    with pytest.raises(LedgerError):
        asyncio.run(guard.try_claim(42, 1007))

    # The registration was undone: a fast graph can still claim this update.
    class WorkingGraph:
        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            return None

    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: WorkingGraph())
    assert asyncio.run(guard.try_claim(42, 1007)) is True


def test_write_timeout_raises_ledger_error_and_allows_later_claim(
    monkeypatch: pytest.MonkeyPatch,
):
    """Claim-first fail-safe: if the durable WRITE times out, no turn may run
    (the write's commit state is unknown). The registration is undone so a
    later working attempt claims it — a duplicate turn stays impossible."""
    import app.telegram.dedup as dedup
    import time as _time

    class SlowWriteGraph:
        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            _time.sleep(1.0)

    monkeypatch.setattr(dedup, "CLAIM_WRITE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: SlowWriteGraph())
    guard = DuplicateGuard()
    with pytest.raises(LedgerError):
        asyncio.run(guard.try_claim(42, 1008))

    class WorkingGraph:
        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            return None

    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: WorkingGraph())
    assert asyncio.run(guard.try_claim(42, 1008)) is True


def test_same_inflight_write_failure_not_double_claimed(monkeypatch: pytest.MonkeyPatch):
    """Concurrent claims with a failing ledger: neither path may double-claim
    a *different* update; the failing update raises for at most one caller
    and the registry undo must not resurrect a concurrent claim."""
    calls: list[str] = []

    class RacyGraph:
        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            calls.append(str(config["configurable"]["thread_id"]))
            raise RuntimeError("write failed")

    monkeypatch.setattr("app.telegram.dedup.build_claim_graph", lambda: RacyGraph())
    guard = DuplicateGuard()

    async def scenario() -> list[object]:
        return await asyncio.gather(
            guard.try_claim(42, 1006),
            guard.try_claim(42, 1006),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())
    # Exactly one caller received True OR both hit the failure path — but the
    # bug shape we guard against is "True AND True"; assert that cannot occur.
    successes = [r for r in results if r is True]
    assert len(successes) <= 1


def test_claim_works_with_sync_only_postgres_style_saver(
    monkeypatch: pytest.MonkeyPatch,
):
    """PRODUCTION-INCIDENT REGRESSION: the app's Neon production saver is the
    SYNC ``PostgresSaver`` (langgraph-checkpoint-postgres 3.2.0) whose async
    checkpoint methods raise ``NotImplementedError``. The claim graph used to
    await ``aget_state``/``ainvoke`` and therefore failed EVERY claim in
    production (every bot message answered with the friendly error) while
    all in-memory tests stayed green. The claim must succeed through the
    SYNC checkpoint API against a saver shaped exactly like production's."""
    import app.memory_db as memory_db
    from test_telegram_synconly_checkpoint import SyncOnlyPostgresStyleSaver

    saver = SyncOnlyPostgresStyleSaver()
    monkeypatch.setattr(memory_db, "checkpointer_factory", lambda: saver)
    guard = DuplicateGuard()

    assert asyncio.run(guard.try_claim(9, 3001)) is True
    # The record must be DURABLE (written through the sync put API): a fresh
    # guard — a restart simulator — must see update 3001 as already claimed.
    assert asyncio.run(DuplicateGuard().try_claim(9, 3001)) is False

    stored = saver.get_tuple({"configurable": {"thread_id": dedup_thread_id(9)}})
    assert stored is not None
    channel_values = stored.checkpoint.get("channel_values") or {}
    assert 3001 in (channel_values.get("seen_update_ids") or [])
    # And the per-user isolation still holds with the sync saver:
    assert asyncio.run(guard.try_claim(9, 3002)) is True


def test_dead_neon_connection_reconnects_once_and_claims(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """LIVE-INCIDENT REGRESSION (updates 54395662/…63): Neon suspends the
    idle compute between messages; the process-lifetime psycopg connection
    is transport-broken (psycopg ``conn.broken``), so every later checkpoint
    op failed INSTANTLY and the old code wrapped it as ``LedgerError``
    forever ("one update succeeded, the rest got the friendly error") until
    a process restart. Now: one bounded reconnect against the failing
    graph's OWN checkpointer + one bounded retry, then the claim succeeds.
    The register/claim/durability guarantees stay exactly the same."""
    import app.memory_db as memory_db
    import app.telegram.dedup as dedup
    from types import SimpleNamespace

    from test_telegram_synconly_checkpoint import SyncOnlyPostgresStyleSaver

    class NeonStyleSaver(SyncOnlyPostgresStyleSaver):
        """Sync-only saver PLUS the production ``PostgresSaver.conn`` shape."""

        def __init__(self) -> None:
            super().__init__()
            # SimpleNamespace mirrors psycopg3's Blocked/... attribute only:
            # ``broken=True`` means the server dropped the socket.
            self.conn = SimpleNamespace(broken=True)

    saver_dead = NeonStyleSaver()
    saver_fresh = NeonStyleSaver()
    saver_fresh.conn.broken = False  # a REBUILT connection: fresh socket
    current = {"saver": saver_dead}
    monkeypatch.setattr(
        memory_db, "checkpointer_factory", lambda: current["saver"]
    )

    real_build = dedup.build_claim_graph

    class DeadSocketGraph:
        """A graph compiled against the DEAD saver: every checkpoint op fails
        instantly, exactly like the live incident (nothing is ever repaired
        in place — reopening builds a NEW connection/saver)."""

        def __init__(self) -> None:
            self.checkpointer = current["saver"]

        def get_state(self, config: dict) -> object:
            raise RuntimeError("server closed the connection unexpectedly")

        def invoke(self, state: dict, config: dict) -> None:
            raise RuntimeError("server closed the connection unexpectedly")

    graph_calls: list[int] = []
    reconnect_calls: list[int] = []

    def fake_build_claim_graph():
        graph_calls.append(1)
        if len(graph_calls) == 1:
            return DeadSocketGraph()
        # Every attempt after the reconnect recompiles the REAL claim graph
        # against the now-current (fresh) saver — durability actually works.
        return real_build()

    def fake_reconnect() -> object:
        reconnect_calls.append(1)
        current["saver"] = saver_fresh
        return saver_fresh

    monkeypatch.setattr(dedup, "build_claim_graph", fake_build_claim_graph)
    monkeypatch.setattr(memory_db, "reopen_postgres_checkpointer", fake_reconnect)

    guard = DuplicateGuard()
    with caplog.at_level("INFO", logger="app.telegram"):
        assert asyncio.run(guard.try_claim(42, 2001)) is True

    # ONE retry per failing op: the stale graph fails the read AND the write,
    # each reconnects once and retries once against the fresh connection.
    assert reconnect_calls == [1, 1]
    assert len(graph_calls) == 3

    # The claim is durable in the FRESH saver's ledger AND per-user isolation
    # and duplicate protection are unchanged.
    assert asyncio.run(DuplicateGuard().try_claim(42, 2001)) is False
    assert asyncio.run(guard.try_claim(42, 2002)) is True
    stored = saver_fresh.get_tuple({"configurable": {"thread_id": dedup_thread_id(42)}})
    channel_values = stored.checkpoint.get("channel_values") or {}
    assert 2001 in (channel_values.get("seen_update_ids") or [])

    # Sanitized diagnostics: failure lines carry TYPE names + the
    # dead_connection flag, but NEVER exception messages or payloads.
    texts = [rec.getMessage() for rec in caplog.records]
    assert any("stage=claim_read failed" in t and "exception_type=RuntimeError" in t
               and "dead_connection=true" in t for t in texts)
    assert any("stage=claim_write failed" in t and "dead_connection=true" in t
               for t in texts)
    retries = [t for t in texts if "event=end retry=true" in t]
    assert len(retries) == 2  # one read retry + one write retry
    assert not any("server closed the connection unexpectedly" in t
                   for t in texts)


def test_dead_connection_and_failed_reconnect_stays_fail_safe(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Broken connector + reconnect ALSO fails → LedgerError, the registry
    entry is undone, no agent turn and no unsafe fallback (the store stays
    Postgres-or-failed — never silently in-memory)."""
    import app.memory_db as memory_db
    import app.telegram.dedup as dedup
    from types import SimpleNamespace

    from test_telegram_synconly_checkpoint import SyncOnlyPostgresStyleSaver

    class NeonStyleSaver(SyncOnlyPostgresStyleSaver):
        def __init__(self) -> None:
            super().__init__()
            self.conn = SimpleNamespace(broken=True)

    saver = NeonStyleSaver()
    monkeypatch.setattr(memory_db, "checkpointer_factory", lambda: saver)

    class DeadSocketGraph:
        checkpointer = saver

        def get_state(self, config: dict) -> object:
            raise RuntimeError("connection failed")

        def invoke(self, state: dict, config: dict) -> None:
            return None

    monkeypatch.setattr(dedup, "build_claim_graph", lambda: DeadSocketGraph())

    def failing_reconnect() -> object:
        raise RuntimeError("reconnect failed")

    monkeypatch.setattr(
        memory_db, "reopen_postgres_checkpointer", failing_reconnect
    )

    guard = DuplicateGuard()
    with caplog.at_level("INFO", logger="app.telegram"):
        with pytest.raises(LedgerError):
            asyncio.run(guard.try_claim(42, 2003))

    assert any("reconnect failed" in rec.getMessage()
               for rec in caplog.records)

    # The registration was undone: a later fully-working attempt can claim.
    class WorkingGraph:
        checkpointer = saver

        def get_state(self, config: dict) -> object:
            class Snapshot:
                values: dict = {}

            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            return None

    monkeypatch.setattr(dedup, "build_claim_graph", lambda: WorkingGraph())
    assert asyncio.run(guard.try_claim(42, 2003)) is True


def test_non_connection_failure_never_attempts_reconnect(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Plain checkpoint failures (schema/permissions/unknown bugs) keep the
    exact fail-fast fail-safe behavior — no reconnect, no retry, no agent."""
    import app.memory_db as memory_db
    import app.telegram.dedup as dedup

    reconnect_calls: list[int] = []

    class ExplodingGraph:
        # Plain object (no psycopg connection): never reconnect-eligible.
        def get_state(self, config: dict) -> object:
            raise RuntimeError("schema mismatch")

    monkeypatch.setattr(dedup, "build_claim_graph", lambda: ExplodingGraph())

    def no_reconnect() -> object:
        reconnect_calls.append(1)
        raise AssertionError("reconnect must not be attempted")

    monkeypatch.setattr(memory_db, "reopen_postgres_checkpointer", no_reconnect)

    guard = DuplicateGuard()
    with caplog.at_level("INFO", logger="app.telegram"):
        with pytest.raises(LedgerError):
            asyncio.run(guard.try_claim(42, 2004))

    assert reconnect_calls == []
    texts = [rec.getMessage() for rec in caplog.records]
    assert any("dead_connection=false" in t for t in texts)
    assert not any("action=reconnect_and_retry" in t for t in texts)


def test_write_retry_after_reconnect_succeeds_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
):
    """Broken connector during the WRITE: one reconnect + one retry; the
    ledger reducer dedupes a repeated id so the durability write stays
    idempotent and the registry still allows exactly ONE turn."""
    import app.memory_db as memory_db
    import app.telegram.dedup as dedup
    from types import SimpleNamespace

    from test_telegram_synconly_checkpoint import SyncOnlyPostgresStyleSaver

    class NeonStyleSaver(SyncOnlyPostgresStyleSaver):
        def __init__(self) -> None:
            super().__init__()
            self.conn = SimpleNamespace(broken=True)

    saver_dead = NeonStyleSaver()
    saver_fresh = NeonStyleSaver()
    saver_fresh.conn.broken = False
    current = {"saver": saver_dead}
    monkeypatch.setattr(
        memory_db, "checkpointer_factory", lambda: current["saver"]
    )
    real_build = dedup.build_claim_graph

    class Snapshot:
        values: dict = {}

    class DeadWriteGraph:
        """The socket drops exactly between the read and the write."""

        def __init__(self) -> None:
            self.checkpointer = current["saver"]

        def get_state(self, config: dict) -> object:
            return Snapshot()

        def invoke(self, state: dict, config: dict) -> None:
            raise RuntimeError("connection lost during write")

    graph_calls: list[int] = []
    reconnect_calls: list[int] = []

    def fake_build_claim_graph():
        graph_calls.append(1)
        if len(graph_calls) == 1:
            return DeadWriteGraph()
        return real_build()  # retry against the fresh (repaired) saver

    def fake_reconnect() -> object:
        reconnect_calls.append(1)
        current["saver"] = saver_fresh
        return saver_fresh

    monkeypatch.setattr(dedup, "build_claim_graph", fake_build_claim_graph)
    monkeypatch.setattr(memory_db, "reopen_postgres_checkpointer", fake_reconnect)

    guard = DuplicateGuard()
    assert asyncio.run(guard.try_claim(42, 2005)) is True
    assert reconnect_calls == [1]
    assert len(graph_calls) == 2
    # The write succeeded through the retry: durable + one-claim-only.
    assert asyncio.run(DuplicateGuard().try_claim(42, 2005)) is False
    stored = saver_fresh.get_tuple(
        {"configurable": {"thread_id": dedup_thread_id(42)}}
    )
    channel_values = stored.checkpoint.get("channel_values") or {}
    assert 2005 in (channel_values.get("seen_update_ids") or [])


def test_sync_only_saver_claim_never_hits_async_api(
    monkeypatch: pytest.MonkeyPatch,
):
    """The claim graph must not call aget_state/ainvoke at all — a saver
    whose async methods raise NotImplementedError (PostgresSaver shape)
    must yield a successful claim, never LedgerError/friendly-error."""
    import app.memory_db as memory_db
    from test_telegram_synconly_checkpoint import SyncOnlyPostgresStyleSaver

    saver = SyncOnlyPostgresStyleSaver()
    monkeypatch.setattr(memory_db, "checkpointer_factory", lambda: saver)
    # Reuse one guard; both claims succeed despite the poison async surface.
    guard = DuplicateGuard()
    assert asyncio.run(guard.try_claim(11, 4001)) is True
    assert asyncio.run(guard.try_claim(11, 4002)) is True
