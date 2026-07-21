"""Deferred, loss-resistant queue for best-effort salience reinforcement appends.

``checkout_memory`` used to await the 'surfaced' reinforcement append before
returning its packet, putting a synchronous JSONL append plus a full graph
projection on the critical path of the front-door read. Measured on this
machine at 500 events that write cost ~27 ms against a ~19 ms warm checkout.

The reinforcement itself is non-authoritative observability state, so the
packet never depends on it having landed — only on it landing *eventually*.
This queue buys that "eventually" back with three layers, in order of
preference:

1. An event-loop drain task scheduled right after the packet is returned.
2. :meth:`DeferredReinforcementQueue.flush`, awaited by ``MemoryFabric.close``.
3. A process-exit hook that appends whatever is still pending straight to the
   append-only log, with no event loop and no graph projection involved.

Layer 3 is safe precisely because ``.eventloom/*.jsonl`` is the source of truth
and the graph is a replayable projection — and because ``memory.reinforcement``
registers an extractor that yields no entities and no edges
(``zaxy/extract/rules_memory.py``), so skipping projection for it drops nothing.

A queue holding pending specs registers itself in a module-level strong set, so
neither it nor the fabric it drains through can be collected with unflushed
writes outstanding; it deregisters as soon as it drains empty.
"""

from __future__ import annotations

import asyncio
import atexit
from contextlib import suppress
from typing import Any, Protocol, cast

from zaxy.security import validate_payload

__all__ = ["DeferredReinforcementQueue", "ReinforcementSink"]


class ReinforcementSink(Protocol):
    """The exact fabric surface the deferred queue drains through."""

    session_manager: Any

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any: ...

    def _record_degraded_operation(self, operation: str, reason: str) -> None: ...


_PENDING_QUEUES: set[DeferredReinforcementQueue] = set()
_ATEXIT_REGISTERED = False


def _drain_pending_queues_at_exit() -> None:
    """Land every queue's outstanding specs on the log during interpreter exit."""
    for queue in list(_PENDING_QUEUES):
        with suppress(Exception):
            queue.drain_to_log_sync()


def _ensure_atexit_registered() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_drain_pending_queues_at_exit)
        _ATEXIT_REGISTERED = True


class DeferredReinforcementQueue:
    """Hold reinforcement event specs until they can be appended off the read path."""

    def __init__(self, *, sink: ReinforcementSink) -> None:
        self._sink = sink
        self._pending: list[tuple[str, dict[str, Any]]] = []
        self._task: asyncio.Task[None] | None = None

    @property
    def pending_count(self) -> int:
        """Number of specs enqueued but not yet appended to the log."""
        return len(self._pending)

    def enqueue(self, spec: dict[str, Any], *, session_id: str) -> None:
        """Queue one reinforcement spec and schedule a drain, without awaiting it."""
        _ensure_atexit_registered()
        self._pending.append((session_id, spec))
        _PENDING_QUEUES.add(self)
        self._schedule_drain()

    def _schedule_drain(self) -> None:
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop to defer onto (sync caller). flush()/atexit still land it.
            return
        self._task = loop.create_task(self._drain())

    async def _drain(self) -> None:
        """Append pending specs one at a time, keeping failures queued for retry."""
        while self._pending:
            session_id, spec = self._pending[0]
            try:
                await self._sink._append_event_spec(spec, session_id=session_id)
            except Exception:
                # Best-effort: leave the spec queued so flush()/atexit retries it,
                # and surface the degrade exactly as the inline path used to.
                with suppress(Exception):
                    self._sink._record_degraded_operation(
                        "append", "salience_reinforcement_unavailable"
                    )
                return
            # Pop only after the append landed; entries are only ever appended at
            # the tail, so index 0 is still the spec just written.
            self._pending.pop(0)
        _PENDING_QUEUES.discard(self)

    async def flush(self) -> None:
        """Await any in-flight drain, then append everything still pending."""
        task = self._task
        if task is not None and not task.done():
            with suppress(Exception):
                await task
        await self._drain()

    def drain_to_log_sync(self) -> None:
        """Append outstanding specs directly to the JSONL log with no loop or graph.

        The last-resort exit path. Writes only the append-only source of truth;
        the graph projection is rebuildable and reinforcement events project
        nothing anyway.
        """
        while self._pending:
            session_id, spec = self._pending[0]
            try:
                eventlog = self._sink.session_manager.get(session_id).eventlog
                eventlog.append(
                    str(spec["event_type"]),
                    actor=str(spec["actor"]),
                    payload=validate_payload(cast(dict[str, Any], spec["payload"])),
                    thread=session_id,
                )
            except Exception:
                return
            self._pending.pop(0)
        _PENDING_QUEUES.discard(self)
