"""Unit tests for the deferred salience-reinforcement queue."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from zaxy.core.reinforcement_queue import (
    _PENDING_QUEUES,
    DeferredReinforcementQueue,
    _drain_pending_queues_at_exit,
    _ensure_atexit_registered,
)
from zaxy.event import EventLog


def _spec(checkout_id: str = "checkout:0001") -> dict[str, Any]:
    return {
        "event_type": "memory.reinforcement",
        "actor": "zaxy-memory",
        "thread": "agent-1",
        "payload": {
            "kind": "surfaced",
            "targets": [{"seq": 1, "hash": "a" * 64}],
            "source": {"checkout_id": checkout_id},
            "authority_status": "non_authoritative",
        },
    }


class _Sink:
    """Minimal fabric stand-in recording appends and degrade reports."""

    def __init__(self, *, log: EventLog | None = None, fail: bool = False) -> None:
        self.appended: list[tuple[str, dict[str, Any]]] = []
        self.degrades: list[tuple[str, str]] = []
        self.fail = fail
        self._log = log

    @property
    def session_manager(self) -> Any:
        log = self._log

        class _Session:
            eventlog = log

        class _Manager:
            def get(self, session_id: str) -> Any:
                return _Session()

        return _Manager()

    async def _append_event_spec(self, event: dict[str, Any], *, session_id: str) -> Any:
        if self.fail:
            raise RuntimeError("append unavailable")
        self.appended.append((session_id, event))
        return None

    def _record_degraded_operation(self, operation: str, reason: str) -> None:
        self.degrades.append((operation, reason))


@pytest.fixture(autouse=True)
def _clear_registry() -> Any:
    """Keep the module-level pending-queue registry isolated per test."""
    _PENDING_QUEUES.clear()
    yield
    _PENDING_QUEUES.clear()


async def test_enqueue_defers_the_append_until_the_loop_runs() -> None:
    """Enqueue must return without appending, leaving the spec pending."""
    sink = _Sink()
    queue = DeferredReinforcementQueue(sink=sink)

    queue.enqueue(_spec(), session_id="agent-1")

    assert sink.appended == []
    assert queue.pending_count == 1


async def test_scheduled_drain_lands_the_append_on_the_next_loop_pass() -> None:
    """The drain task must append without any explicit flush call."""
    sink = _Sink()
    queue = DeferredReinforcementQueue(sink=sink)

    queue.enqueue(_spec(), session_id="agent-1")
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(sink.appended) == 1
    assert queue.pending_count == 0


async def test_flush_appends_everything_pending() -> None:
    """Flush must drain the whole backlog, preserving enqueue order."""
    sink = _Sink()
    queue = DeferredReinforcementQueue(sink=sink)
    queue.enqueue(_spec("checkout:1"), session_id="agent-1")
    queue.enqueue(_spec("checkout:2"), session_id="agent-2")

    await queue.flush()

    assert [session for session, _ in sink.appended] == ["agent-1", "agent-2"]
    assert queue.pending_count == 0


async def test_flush_is_idempotent_when_nothing_is_pending() -> None:
    """Flushing an empty queue must be a no-op, not an error."""
    sink = _Sink()
    queue = DeferredReinforcementQueue(sink=sink)

    await queue.flush()
    await queue.flush()

    assert sink.appended == []


async def test_failed_append_keeps_the_spec_queued_and_records_a_degrade() -> None:
    """A failing append must not drop the reinforcement; it stays queued for retry."""
    sink = _Sink(fail=True)
    queue = DeferredReinforcementQueue(sink=sink)
    queue.enqueue(_spec(), session_id="agent-1")

    await queue.flush()

    assert queue.pending_count == 1
    assert ("append", "salience_reinforcement_unavailable") in sink.degrades


async def test_retry_after_a_transient_failure_lands_the_original_spec() -> None:
    """A spec kept queued through a failure must append once the sink recovers."""
    sink = _Sink(fail=True)
    queue = DeferredReinforcementQueue(sink=sink)
    queue.enqueue(_spec("checkout:retry"), session_id="agent-1")
    await queue.flush()

    sink.fail = False
    await queue.flush()

    assert len(sink.appended) == 1
    assert sink.appended[0][1]["payload"]["source"]["checkout_id"] == "checkout:retry"
    assert queue.pending_count == 0


def test_enqueue_without_a_running_loop_still_holds_the_spec() -> None:
    """A sync caller has no loop to defer onto; the spec must stay pending, not vanish."""
    sink = _Sink()
    queue = DeferredReinforcementQueue(sink=sink)

    queue.enqueue(_spec(), session_id="agent-1")

    assert queue.pending_count == 1
    assert queue in _PENDING_QUEUES


def test_drain_to_log_sync_writes_the_spec_to_the_append_only_log(tmp_path: Path) -> None:
    """The exit path must seal the reinforcement into JSONL with no loop and no graph."""
    log = EventLog(str(tmp_path / "events.jsonl"))
    sink = _Sink(log=log)
    queue = DeferredReinforcementQueue(sink=sink)
    queue.enqueue(_spec(), session_id="agent-1")

    queue.drain_to_log_sync()

    events = log.read_all()
    assert [event.type for event in events] == ["memory.reinforcement"]
    assert events[0].payload["kind"] == "surfaced"
    assert queue.pending_count == 0
    assert queue not in _PENDING_QUEUES


def test_drain_to_log_sync_keeps_the_spec_when_the_log_is_unwritable() -> None:
    """A failed exit-path write must leave the spec queued rather than silently drop it."""
    sink = _Sink(log=None)
    queue = DeferredReinforcementQueue(sink=sink)
    queue.enqueue(_spec(), session_id="agent-1")

    queue.drain_to_log_sync()

    assert queue.pending_count == 1


def test_exit_hook_drains_every_registered_queue(tmp_path: Path) -> None:
    """The atexit hook must land pending specs across all live queues."""
    first_log = EventLog(str(tmp_path / "first.jsonl"))
    second_log = EventLog(str(tmp_path / "second.jsonl"))
    first = DeferredReinforcementQueue(sink=_Sink(log=first_log))
    second = DeferredReinforcementQueue(sink=_Sink(log=second_log))
    first.enqueue(_spec("checkout:first"), session_id="agent-1")
    second.enqueue(_spec("checkout:second"), session_id="agent-2")

    _drain_pending_queues_at_exit()

    assert len(first_log.read_all()) == 1
    assert len(second_log.read_all()) == 1
    assert not _PENDING_QUEUES


def test_atexit_registration_happens_once() -> None:
    """The exit hook must be registered exactly once per process, not per enqueue."""
    import zaxy.core.reinforcement_queue as module

    registered: list[Any] = []
    original_register = module.atexit.register
    module.atexit.register = lambda fn: registered.append(fn) or fn  # type: ignore[assignment]
    original_flag = module._ATEXIT_REGISTERED
    module._ATEXIT_REGISTERED = False
    try:
        _ensure_atexit_registered()
        _ensure_atexit_registered()
    finally:
        module.atexit.register = original_register  # type: ignore[assignment]
        module._ATEXIT_REGISTERED = original_flag

    assert len(registered) == 1
