"""Hardening tests: bounded embedded lock acquisition + orphan self-termination.

Covers the two fixes that stop the Claude Code incident class:
1. ``EmbeddedGraphStore`` open / probe / execute fail fast with
   :class:`EmbeddedProjectionLockedError` (then degrade) instead of hanging
   forever when a stale process holds the single-writer lock.
2. The MCP owner self-terminates (PR_SET_PDEATHSIG + getppid watchdog +
   atexit) so a reconnect can never strand a lock-holding zombie.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zaxy.embedded_graph_internals import (
    EmbeddedProjectionLockedError,
    await_blocking_with_timeout,
    is_embedded_projection_lock_error,
)
from zaxy.embedded_graph_store import EmbeddedGraphStore

# --------------------------------------------------------------------------- #
# Bounded daemon-thread runner                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_await_blocking_returns_value_quickly() -> None:
    result = await await_blocking_with_timeout(lambda: 42, timeout=5.0, operation="unit")
    assert result == 42


@pytest.mark.asyncio
async def test_await_blocking_times_out_with_legible_error() -> None:
    started = time.monotonic()
    with pytest.raises(EmbeddedProjectionLockedError) as info:
        await await_blocking_with_timeout(
            lambda: time.sleep(2.0), timeout=0.1, operation="open"
        )
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert info.value.reason == "acquisition-timeout"
    assert info.value.operation == "open"
    assert info.value.timeout_seconds == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_await_blocking_reraises_non_lock_errors() -> None:
    def _boom() -> None:
        raise RuntimeError("schema syntax error")

    with pytest.raises(RuntimeError, match="schema syntax error"):
        await await_blocking_with_timeout(_boom, timeout=5.0, operation="open")


@pytest.mark.asyncio
async def test_await_blocking_translates_engine_lock_string() -> None:
    def _engine_locked() -> None:
        raise RuntimeError("Could not set lock on file: /x/embedded.kuzu")

    with pytest.raises(EmbeddedProjectionLockedError) as info:
        await await_blocking_with_timeout(_engine_locked, timeout=5.0, operation="open")
    assert info.value.reason == "engine-reported-held"


def test_is_embedded_projection_lock_error_matches_typed_and_string() -> None:
    assert is_embedded_projection_lock_error(
        EmbeddedProjectionLockedError(reason="acquisition-timeout", operation="open")
    )
    assert is_embedded_projection_lock_error(
        RuntimeError("Could not set lock on file /tmp/x/embedded.kuzu")
    )
    assert not is_embedded_projection_lock_error(RuntimeError("disk full"))
    assert not is_embedded_projection_lock_error(
        RuntimeError("Could not set lock on file /tmp/x/embedded.sqlite")
    )


# --------------------------------------------------------------------------- #
# EmbeddedGraphStore: execute / probe / close / connect                       #
# --------------------------------------------------------------------------- #


def _connected_store() -> EmbeddedGraphStore:
    store = EmbeddedGraphStore(Path("/tmp/embedded.kuzu"))
    store._database = object()
    store._connection = MagicMock()
    return store


def test_execute_translates_engine_lock_raise_to_typed_error() -> None:
    store = _connected_store()
    store._connection.execute.side_effect = RuntimeError(
        "Could not set lock on file /x/embedded.kuzu"
    )
    with pytest.raises(EmbeddedProjectionLockedError) as info:
        store._execute("CREATE NODE TABLE IF NOT EXISTS Foo() PRIMARY KEY()")
    assert info.value.reason == "engine-reported-held"


def test_execute_propagates_unrelated_runtime_errors() -> None:
    store = _connected_store()
    store._connection.execute.side_effect = RuntimeError("syntax error near '('")
    with pytest.raises(RuntimeError, match="syntax error"):
        store._execute("CREATE NODE TABLE IF NOT EXISTS Foo() PRIMARY KEY()")


@pytest.mark.asyncio
async def test_acquire_write_lock_probe_succeeds_when_store_writable() -> None:
    store = _connected_store()
    store._execute = MagicMock()
    await store.acquire_write_lock_probe()
    args, _ = store._execute.call_args
    assert "MERGE (p:BenchmarkProjection" in args[0]
    assert args[1] == {"key": "__lockprobe__"}


@pytest.mark.asyncio
async def test_acquire_write_lock_probe_times_out_under_contention() -> None:
    store = _connected_store()
    store._lock_timeout_override = 0.1
    store._execute = MagicMock(side_effect=lambda *a, **k: time.sleep(2.0))
    with pytest.raises(EmbeddedProjectionLockedError) as info:
        await store.acquire_write_lock_probe()
    assert info.value.reason == "acquisition-timeout"
    assert info.value.operation == "write-lock-probe"


@pytest.mark.asyncio
async def test_close_checkpoints_wal_and_is_resilient() -> None:
    store = _connected_store()
    store._execute = MagicMock(side_effect=RuntimeError("checkpoint busy"))
    await store.close()  # must not raise even if CHECKPOINT fails
    store._execute.assert_called_once_with("CHECKPOINT")
    assert store._connection is None
    assert store._database is None


@pytest.mark.asyncio
async def test_close_skips_checkpoint_when_not_connected() -> None:
    store = EmbeddedGraphStore(Path("/tmp/embedded.kuzu"))
    store._execute = MagicMock()
    await store.close()
    store._execute.assert_not_called()


@pytest.mark.asyncio
async def test_connect_propagates_lock_error_without_quarantining(tmp_path: Path) -> None:
    """A bounded-open lock timeout surfaces directly; it must never be mistaken
    for a corrupt/incompatible store (which would be moved aside)."""
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    with (
        patch(
            "zaxy.embedded_graph_store.await_blocking_with_timeout",
            AsyncMock(
                side_effect=EmbeddedProjectionLockedError(
                    reason="acquisition-timeout", operation="open"
                )
            ),
        ),
        patch("importlib.util.find_spec", return_value=MagicMock()),
        pytest.raises(EmbeddedProjectionLockedError),
    ):
        await store.connect()
    # No quarantine backup was created for a lock error.
    assert not list((tmp_path).glob("*.pre-ladybug.bak*"))


# --------------------------------------------------------------------------- #
# MCP server: degrade-to-null on lock contention                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_setup_degrades_to_null_when_lock_unrecoverable(tmp_path: Path) -> None:
    from zaxy.mcp_server import ZaxyMCPServer
    from zaxy.null_projection_store import NullProjectionStore

    server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
    locked_graph = AsyncMock()
    locked_graph.connect.side_effect = EmbeddedProjectionLockedError(
        reason="acquisition-timeout", operation="open"
    )
    server.graph = locked_graph
    # Reap does not recover an owner in this scenario -> degrade immediately.
    with patch.object(server, "_reap_embedded_owner", return_value=False):
        await server._connect_projection_with_lock_recovery()

    assert server._projection_degraded is not None
    assert server._projection_degraded["reason"] == "embedded_projection_locked"
    assert server._projection_backend == "null"
    assert isinstance(server.graph, NullProjectionStore)
    # The persistent fabric must read through the null store, not the locked one.
    assert server._fabric.graph is server.graph


@pytest.mark.asyncio
async def test_setup_recovers_when_reap_clears_lock(tmp_path: Path) -> None:
    from zaxy.mcp_server import ZaxyMCPServer

    server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
    locked_then_free = AsyncMock()
    # First attempt (locked) raises; the retry after a successful reap succeeds.
    locked_then_free.connect.side_effect = [
        EmbeddedProjectionLockedError(reason="acquisition-timeout", operation="open"),
        None,
        None,
    ]
    server.graph = locked_then_free
    with patch.object(server, "_reap_embedded_owner", return_value=True):
        await server._connect_projection_with_lock_recovery()

    assert server._projection_degraded is None
    assert server.graph is locked_then_free
    assert locked_then_free.connect.await_count == 2


# --------------------------------------------------------------------------- #
# Orphan self-termination watchdog                                            #
# --------------------------------------------------------------------------- #


def test_orphan_watchdog_signals_shutdown_when_parent_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    import zaxy.mcp_server as mcp_server

    shutdown = asyncio.Event()
    pids = iter([1000, 1000, 1000, 1])  # parent reparents to init(1) on the 4th poll

    monkeypatch.setattr(mcp_server, "_current_parent_pid", lambda: next(pids))
    monkeypatch.setattr(mcp_server, "_ORPHAN_WATCHDOG_POLL_SECONDS", 0.0)
    # PR_SET_PDEATHSIG is best-effort; skip the ctypes path on this test.
    monkeypatch.setattr(mcp_server, "_install_parent_death_signal", lambda: None)

    mcp_server._install_orphan_watchdog(shutdown)

    deadline = time.monotonic() + 2.0
    while not shutdown.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    # The watchdog set shutdown once the parent pid diverged from the initial
    # seed (one _current_parent_pid() seeds initial_ppid, then the poll loop
    # detects the change).
    assert shutdown.is_set()


def test_orphan_watchdog_signals_shutdown_when_pid_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing parent-pid probe (OSError) must also trigger shutdown."""
    import zaxy.mcp_server as mcp_server

    shutdown = asyncio.Event()

    calls = {"n": 0}

    def _raising_pid() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return 1000  # seed initial_ppid successfully
        raise OSError("proc gone")  # subsequent loop probes fail

    monkeypatch.setattr(mcp_server, "_current_parent_pid", _raising_pid)
    monkeypatch.setattr(mcp_server, "_ORPHAN_WATCHDOG_POLL_SECONDS", 0.0)
    monkeypatch.setattr(mcp_server, "_install_parent_death_signal", lambda: None)

    mcp_server._install_orphan_watchdog(shutdown)

    deadline = time.monotonic() + 2.0
    while not shutdown.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert shutdown.is_set()


def test_reap_embedded_owner_returns_false_for_non_embedded(tmp_path: Path) -> None:
    from zaxy.mcp_server import ZaxyMCPServer

    server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
    server._projection_backend = "neo4j"
    assert server._reap_embedded_owner() is False


def test_reap_embedded_owner_returns_false_when_coordinator_raises(
    tmp_path: Path,
) -> None:
    from zaxy.mcp_server import ZaxyMCPServer

    server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
    server._projection_backend = "embedded"
    with patch(
        "zaxy.mcp_runtime.EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path",
        side_effect=OSError("boom"),
    ):
        assert server._reap_embedded_owner() is False


def test_reap_embedded_owner_returns_true_when_coordinator_repairs(
    tmp_path: Path,
) -> None:
    from zaxy.mcp_server import ZaxyMCPServer

    server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
    server._projection_backend = "embedded"
    coordinator = MagicMock()
    coordinator.repair_stale_runtime.return_value = {
        "repaired": True,
        "reaped_pid": 12345,
    }
    with patch(
        "zaxy.mcp_runtime.EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path",
        return_value=coordinator,
    ):
        assert server._reap_embedded_owner() is True
    coordinator.repair_stale_runtime.assert_called_once_with(
        reap=True, expected_graph_path=server._embedded_graph_path
    )


@pytest.mark.asyncio
async def test_setup_degrades_when_reap_succeeds_but_lock_persists(
    tmp_path: Path,
) -> None:
    """Reap runs (repaired=True) but the retry still contends -> degrade anyway."""
    from zaxy.mcp_server import ZaxyMCPServer
    from zaxy.null_projection_store import NullProjectionStore

    server = ZaxyMCPServer(eventloom_path=str(tmp_path / ".eventloom"))
    locked_graph = AsyncMock()
    # Both the first attempt and the post-reap retry contend.
    locked_graph.connect.side_effect = EmbeddedProjectionLockedError(
        reason="acquisition-timeout", operation="open"
    )
    server.graph = locked_graph
    with patch.object(server, "_reap_embedded_owner", return_value=True):
        await server._connect_projection_with_lock_recovery()

    assert server._projection_degraded is not None
    assert server._projection_backend == "null"
    assert isinstance(server.graph, NullProjectionStore)
    assert server._fabric.graph is server.graph


# --------------------------------------------------------------------------- #
# Orphan self-termination: parent-death signal + atexit closer                #
# --------------------------------------------------------------------------- #


def test_close_owner_claim_atexit_invokes_close_and_is_resilient() -> None:
    import zaxy.mcp_server as mcp_server

    claim = MagicMock()
    mcp_server._close_owner_claim_atexit(claim)
    claim.close.assert_called_once_with()

    # A claim whose close() raises must not propagate (atexit must not abort).
    raising = MagicMock()
    raising.close.side_effect = OSError("already closed")
    mcp_server._close_owner_claim_atexit(raising)  # no raise


def test_parent_death_signal_skipped_on_non_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    import zaxy.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server.sys, "platform", "darwin")
    with patch.object(ctypes, "CDLL", side_effect=AssertionError("must not load libc")):
        mcp_server._install_parent_death_signal()  # returns without touching ctypes


def test_parent_death_signal_tolerates_libc_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import zaxy.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server.sys, "platform", "linux")
    with patch("ctypes.CDLL", side_effect=OSError("no libc")):
        # Best-effort: a libc load failure must not raise.
        mcp_server._install_parent_death_signal()


def test_parent_death_signal_warns_on_nonzero_prctl(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    import zaxy.mcp_server as mcp_server

    monkeypatch.setattr(mcp_server.sys, "platform", "linux")
    libc = MagicMock()
    libc.prctl.return_value = 1  # nonzero => kernel rejected the request
    with (
        patch("ctypes.CDLL", return_value=libc),
        caplog.at_level(logging.WARNING, logger="zaxy.mcp_server"),
    ):
        mcp_server._install_parent_death_signal()
    libc.prctl.assert_called_once()
    assert any("pdeathsig_install_failed" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Abandoned-daemon accounting + race-free backup claims                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_await_blocking_success_joins_worker_without_abandonment() -> None:
    """The common path reclaims its thread and records no abandonment."""
    import threading

    from zaxy.embedded_graph_internals import abandoned_lock_op_stats

    before = abandoned_lock_op_stats()["total"]
    # earlier tests may have legitimately abandoned workers; assert no NEW ones
    lingering_before = {
        id(t) for t in threading.enumerate() if t.name == "zaxy-embedded-lock-op" and t.is_alive()
    }
    result = await await_blocking_with_timeout(lambda: 41 + 1, timeout=5.0, operation="test-op")
    assert result == 42
    assert abandoned_lock_op_stats()["total"] == before
    lingering_after = {
        id(t) for t in threading.enumerate() if t.name == "zaxy-embedded-lock-op" and t.is_alive()
    }
    assert lingering_after <= lingering_before  # this call's worker was joined


@pytest.mark.asyncio
async def test_await_blocking_timeout_records_abandoned_worker(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A timed-out lock op is abandoned by design but MUST be observable."""
    import logging
    import threading

    from zaxy.embedded_graph_internals import abandoned_lock_op_stats

    release = threading.Event()
    before = abandoned_lock_op_stats()["total"]
    with (
        caplog.at_level(logging.WARNING, logger="zaxy.embedded_graph_internals"),
        pytest.raises(EmbeddedProjectionLockedError),
    ):
        await await_blocking_with_timeout(release.wait, timeout=0.05, operation="test-block")
    stats = abandoned_lock_op_stats()
    assert stats["total"] == before + 1
    assert stats["live"] >= 1
    assert any("abandoned after timeout" in record.message for record in caplog.records)
    release.set()  # unblock the daemon so it exits with the test process cleanly


def test_claim_backup_path_is_exclusive_per_claimant(tmp_path: Path) -> None:
    """Two claims for the same store's backup name can never collide.

    The old exists()-then-replace scan was a TOCTOU: two processes self-healing
    the same projection could pick the same .bak name and silently clobber each
    other's backup. Claiming via O_CREAT|O_EXCL makes each name single-winner.
    """
    from zaxy.embedded_graph_internals import _claim_backup_path

    store = tmp_path / "embedded.kuzu"
    first = _claim_backup_path(store)
    second = _claim_backup_path(store)  # simulates the concurrent claimant
    assert first != second
    assert first.name == "embedded.kuzu.pre-ladybug.bak"
    assert second.name == "embedded.kuzu.pre-ladybug.bak.1"
    assert first.exists() and second.exists()  # placeholders hold the claims


def test_claim_backup_path_never_shadows_an_existing_wal_backup(tmp_path: Path) -> None:
    """A leftover .bak.wal from an older migration blocks that name entirely."""
    from zaxy.embedded_graph_internals import _claim_backup_path

    store = tmp_path / "embedded.kuzu"
    (tmp_path / "embedded.kuzu.pre-ladybug.bak.wal").write_bytes(b"old wal backup")
    claimed = _claim_backup_path(store)
    assert claimed.name == "embedded.kuzu.pre-ladybug.bak.1"
