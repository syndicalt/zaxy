"""Phase 0 integrity fixes on MemoryFabric.

Covers two review findings:

* A2 (C5) -- ``close()`` must be a no-op for an injected/non-owning fabric so a
  shared server fabric is not dropped to disconnected and forced to reopen an
  embedded store it already holds a lock on.
* C3 (H5) -- the primary write path must offload the blocking ``eventlog.append``
  (open + exclusive flock + fsync) to a worker thread instead of stalling the
  event loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import zaxy.core.fabric as fabric_mod
from zaxy.core.fabric import MemoryFabric


def _owning_fabric(tmp_path: Path) -> MemoryFabric:
    return MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)


@pytest.mark.asyncio
async def test_close_is_noop_when_not_owning_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fabric = MemoryFabric(
        eventloom_path=str(tmp_path / ".eventloom"),
        tracer_disabled=True,
        owns_connections=False,
    )
    # A non-owning fabric is born connected: the host owns the lifecycle.
    assert fabric._connected is True

    closed = {"graph": False, "tracer": False}

    async def _mark_graph() -> None:
        closed["graph"] = True

    async def _mark_tracer() -> None:
        closed["tracer"] = True

    monkeypatch.setattr(fabric.graph, "close", _mark_graph)
    monkeypatch.setattr(fabric.tracer, "close", _mark_tracer)

    await fabric.close()

    # The fix: a shared fabric stays connected and never tears down host-owned
    # components, so the next call does not reopen the embedded store.
    assert fabric._connected is True
    assert closed == {"graph": False, "tracer": False}


@pytest.mark.asyncio
async def test_close_tears_down_when_owning_connections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fabric = _owning_fabric(tmp_path)
    fabric._connected = True

    closed = {"graph": False, "tracer": False}

    async def _mark_graph() -> None:
        closed["graph"] = True

    async def _mark_tracer() -> None:
        closed["tracer"] = True

    monkeypatch.setattr(fabric.graph, "close", _mark_graph)
    monkeypatch.setattr(fabric.tracer, "close", _mark_tracer)

    await fabric.close()

    # An owning fabric performs a real teardown.
    assert fabric._connected is False
    assert closed == {"graph": True, "tracer": True}


@pytest.mark.asyncio
async def test_append_offloads_blocking_write_to_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fabric = _owning_fabric(tmp_path)
    await fabric.connect()
    try:
        offloaded: list[str] = []
        real_to_thread = asyncio.to_thread

        async def _spy_to_thread(func, /, *args, **kwargs):  # type: ignore[no-untyped-def]
            offloaded.append(getattr(func, "__name__", ""))
            return await real_to_thread(func, *args, **kwargs)

        monkeypatch.setattr(fabric_mod.asyncio, "to_thread", _spy_to_thread)

        event = await fabric.append(
            "memory.note",
            actor="user",
            payload={"content": "hello"},
            session_id="agent-1",
        )

        # The blocking eventlog.append ran on a worker thread, not inline.
        assert "append" in offloaded
        # ...and the write is still correct and hash-chained.
        eventlog = fabric.session_manager.get("agent-1").eventlog
        stored = eventlog.read_all()
        assert stored[-1].seq == event.seq
        assert stored[-1].type == "memory.note"
        assert eventlog.verify().ok is True
    finally:
        await fabric.close()
