"""Sync-only checkpointer stub mirroring production ``PostgresSaver``.

PRODUCTION INCIDENT (regression harness): the app's Neon production saver is
the SYNC ``langgraph.checkpoint.postgres.PostgresSaver`` (langgraph-
checkpoint-postgres 3.2.0), whose ``aget_tuple`` / ``aput`` (and all other
async checkpoint ops) raise ``NotImplementedError`` — only
``AsyncPostgresSaver`` implements them. The dedup claim graph originally
AWAITED ``aget_state``/``ainvoke`` and therefore failed every production
claim with ``LedgerError`` → the observed "friendly error" for every real
bot message — while the test suite stayed green because the pinned
``InMemorySaver`` implements the async surface natively.

This stub reproduces that exact failure shape offline: sync
``get_tuple``/``put`` backed by an in-memory dict, async methods explicitly
raising ``NotImplementedError`` like the real PostgresSaver. It is wired
through the ``memory_db.checkpointer_factory`` test seam.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    get_checkpoint_id,
)


class SyncOnlyPostgresStyleSaver(BaseCheckpointSaver):
    """In-memory saver with ONLY the sync checkpoint API (PostgresSaver shape)."""

    def __init__(self) -> None:
        super().__init__()
        self._threads: dict[str, tuple[dict, Any]] = {}

    def list(
        self,
        config: dict | None,
        *,
        filter: dict | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if thread_id is not None and thread_id in self._threads:
            yield self.get_tuple({"configurable": {"thread_id": thread_id}})  # type: ignore[misc]

    def get_tuple(self, config: dict) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        stored = self._threads.get(thread_id)
        if stored is None:
            return None
        checkpoint, metadata = stored
        return CheckpointTuple(
            config={
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": checkpoint["id"],
                }
            },
            checkpoint=copy.deepcopy(checkpoint),
            metadata=copy.deepcopy(metadata),
            parent_config=None,
        )

    def put(
        self,
        config: dict,
        checkpoint: dict,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        self._threads[thread_id] = (copy.deepcopy(checkpoint), copy.deepcopy(metadata))
        return {
            "configurable": {
                **config["configurable"],
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: dict,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
    ) -> dict:
        # Not exercised by the single-node claim graph or the conversation
        # graph in these tests; mirror the sync-side completeness only.
        self._threads[config["configurable"]["thread_id"]][0].setdefault(
            "pending_writes", []
        ).extend(writes)
        return {
            "configurable": {
                **config["configurable"],
                "checkpoint_id": get_checkpoint_id(config),
            }
        }

    # -- Explicitly PostgresSaver-shaped async surface: unsupported. -------
    async def aget_tuple(self, config: dict) -> Any:
        raise NotImplementedError

    async def aput(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def aput_writes(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def alist(self, **kwargs: Any) -> Any:
        raise NotImplementedError
