"""Tests for embedded MCP runtime ownership and proxy coordination."""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zaxy.mcp_runtime import EmbeddedMcpRuntimeCoordinator, EmbeddedMcpRuntimePaths


def _short_socket_coordinator(tmp_path: Path) -> EmbeddedMcpRuntimeCoordinator:
    """Create runtime paths with a short Unix socket path for Linux limits."""
    runtime_dir = tmp_path / "r"
    socket_path = Path(f"/tmp/zaxy-test-{os.getpid()}-{id(tmp_path)}.sock")
    return EmbeddedMcpRuntimeCoordinator(
        EmbeddedMcpRuntimePaths(
            runtime_dir=runtime_dir,
            lock_path=runtime_dir / "owner.lock",
            owner_path=runtime_dir / "owner.json",
            socket_path=socket_path,
        )
    )


def test_store_keyed_coordinator_coordinates_same_store_across_eventloom_paths(tmp_path):
    """The owner lock must key on the STORE, not the eventloom path, so two
    processes that open the same embedded store coordinate even if they resolved
    their eventloom path differently (the divergence that allowed multiple owners).
    """
    eventloom = tmp_path / ".eventloom"
    store = eventloom / "projections" / "embedded.kuzu"
    store.parent.mkdir(parents=True, exist_ok=True)

    # Standard layout: the store-keyed runtime resolves to <eventloom>/runtime,
    # matching the eventloom-keyed location (no transition orphaning).
    by_store = EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path(store)
    by_eventloom = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(eventloom)
    assert by_store.paths.lock_path.resolve() == by_eventloom.paths.lock_path.resolve()

    # Same store reached via a different (un-normalized) path string still keys
    # to the same lock — so a divergent eventloom arg can't mint a second owner.
    divergent = EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path(
        eventloom / "x" / ".." / "projections" / "embedded.kuzu"
    )
    assert divergent.paths.lock_path.resolve() == by_store.paths.lock_path.resolve()

    owner = by_store.try_claim_owner()
    second = divergent.try_claim_owner()
    assert owner is not None
    assert second is None  # same store -> same lock -> refused
    owner.close()


def test_owner_pid_reapable_only_when_verified(tmp_path, monkeypatch):
    """The reaper must refuse to kill anything it cannot verify is a broken
    zaxy-serve owner for THIS store — the safety guard on a dangerous operation.
    """
    coord = _short_socket_coordinator(tmp_path)

    assert coord._owner_pid_is_reapable(None, None) is False  # no record
    assert coord._owner_pid_is_reapable({"pid": "x"}, None) is False  # bad pid
    assert coord._owner_pid_is_reapable({"pid": os.getpid()}, None) is False  # never self
    # A dead pid (real os.kill -> ProcessLookupError) is not reapable.
    assert coord._owner_pid_is_reapable({"pid": 2_147_483_646}, None) is False

    # Pretend the pid is alive for the remaining checks.
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    # A zaxy serve whose owner record lacks a usable graph_path -> not reapable
    # when a store is expected.
    monkeypatch.setattr(coord, "_process_cmdline", lambda pid: "python zaxy serve")
    assert coord._owner_pid_is_reapable({"pid": 4242, "graph_path": None}, "/b/embedded.kuzu") is False

    # Alive but not a zaxy serve -> never reaped.
    monkeypatch.setattr(coord, "_process_cmdline", lambda pid: "/usr/bin/python other_app.py")
    assert coord._owner_pid_is_reapable({"pid": 4242}, None) is False

    # A zaxy serve, but for a DIFFERENT store -> never reaped.
    monkeypatch.setattr(coord, "_process_cmdline", lambda pid: "python /x/zaxy serve")
    assert (
        coord._owner_pid_is_reapable(
            {"pid": 4242, "graph_path": "/a/.eventloom/projections/embedded.kuzu"},
            "/b/.eventloom/projections/embedded.kuzu",
        )
        is False
    )

    # A zaxy serve for THIS store -> reapable.
    assert (
        coord._owner_pid_is_reapable(
            {"pid": 4242, "graph_path": "/b/.eventloom/projections/embedded.kuzu"},
            "/b/.eventloom/projections/embedded.kuzu",
        )
        is True
    )


def test_repair_reaps_verified_broken_owner_and_reclaims(tmp_path, monkeypatch):
    """A live-but-broken owner (lock held, dead socket) is reaped and the runtime
    reclaimed — but only with reap=True and only when verified.
    """
    coord = _short_socket_coordinator(tmp_path)
    holder = coord.try_claim_owner()  # stand in for the broken owner holding the lock
    assert holder is not None
    coord.paths.owner_path.parent.mkdir(parents=True, exist_ok=True)
    coord.paths.owner_path.write_text(
        json.dumps(
            {
                "pid": 4242,
                "graph_path": "/b/.eventloom/projections/embedded.kuzu",
                "socket_path": str(coord.paths.socket_path),  # never created -> dead socket
            }
        ),
        encoding="utf-8",
    )

    # reap=False must NOT touch a held lock (read-only diagnostic contract).
    passive = coord.repair_stale_runtime()
    assert passive["repaired"] is False and passive["owner_active"] is True

    # reap=True: verified broken owner -> terminate (simulated by releasing the
    # held lock) -> reclaim.
    monkeypatch.setattr(coord, "_owner_pid_is_reapable", lambda record, expected: True)

    def _fake_terminate(pid, **kwargs):
        holder.close()  # the broken owner "dies", freeing the flock
        return True

    monkeypatch.setattr(coord, "_terminate_pid", _fake_terminate)
    report = coord.repair_stale_runtime(
        reap=True, expected_graph_path="/b/.eventloom/projections/embedded.kuzu"
    )
    assert report["repaired"] is True
    assert report["reaped_pid"] == 4242


def test_process_cmdline_reads_self_and_handles_missing(tmp_path):
    """_process_cmdline returns a real command line on Linux and None when absent."""
    coord = _short_socket_coordinator(tmp_path)
    if Path("/proc").exists():
        mine = coord._process_cmdline(os.getpid())
        assert mine is not None and len(mine) > 0
    # A pid with no /proc entry (or no /proc at all) -> None, never raises.
    assert coord._process_cmdline(2_147_483_646) is None


def test_terminate_pid_sends_sigterm_then_reports_gone(tmp_path, monkeypatch):
    """_terminate_pid sends SIGTERM and reports success once the pid is gone.

    Uses a fake os.kill (a real direct child becomes a zombie that still answers
    kill(pid, 0); a real reaped owner is not our child, so production behavior is
    the gone-after-SIGTERM path modeled here).
    """
    import signal

    coord = _short_socket_coordinator(tmp_path)
    sent: list[int] = []
    state = {"alive": True}

    def _fake_kill(pid: int, sig: int) -> None:
        sent.append(sig)
        if sig == signal.SIGTERM:
            state["alive"] = False
        if sig == 0 and not state["alive"]:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", _fake_kill)
    assert coord._terminate_pid(123456, timeout_seconds=2.0) is True
    assert signal.SIGTERM in sent


def test_terminate_pid_reports_true_for_a_nonexistent_pid(tmp_path):
    """A pid that does not exist is treated as already-terminated (real os.kill)."""
    coord = _short_socket_coordinator(tmp_path)
    assert coord._terminate_pid(2_147_483_646, timeout_seconds=1.0) is True


def test_terminate_pid_escalates_to_sigkill_and_reports_failure(tmp_path, monkeypatch):
    """If SIGTERM doesn't end the process, escalate to SIGKILL; if nothing works,
    report False rather than claim success.
    """
    import signal

    coord = _short_socket_coordinator(tmp_path)

    # Survives SIGTERM, dies on SIGKILL.
    state = {"alive": True}

    def _kill_dies_on_sigkill(pid: int, sig: int) -> None:
        if sig == signal.SIGKILL:
            state["alive"] = False
        if sig == 0 and not state["alive"]:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", _kill_dies_on_sigkill)
    assert coord._terminate_pid(123, timeout_seconds=0.1) is True

    # Unkillable -> False (never falsely claims success).
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert coord._terminate_pid(123, timeout_seconds=0.1) is False


def test_embedded_runtime_allows_single_owner_claim(tmp_path):
    """Only one embedded MCP process should own a workspace runtime."""
    coordinator = _short_socket_coordinator(tmp_path)

    owner = coordinator.try_claim_owner()
    duplicate = coordinator.try_claim_owner()

    assert owner is not None
    assert duplicate is None
    owner.close()


def test_embedded_runtime_removes_stale_owner_record_when_lock_is_free(tmp_path):
    """A stale owner record should not block a new clean runtime start."""
    coordinator = _short_socket_coordinator(tmp_path)
    coordinator.paths.owner_path.parent.mkdir(parents=True, exist_ok=True)
    coordinator.paths.owner_path.write_text(
        '{"pid": 999999999, "socket_path": "/tmp/missing-zaxy.sock"}',
        encoding="utf-8",
    )

    owner = coordinator.try_claim_owner()

    assert owner is not None
    assert not coordinator.paths.owner_path.exists()
    owner.close()


def test_owner_claim_writes_runtime_record_with_socket_path(tmp_path):
    """The owner record should give later workers enough information to proxy."""
    coordinator = _short_socket_coordinator(tmp_path)
    owner = coordinator.try_claim_owner()
    assert owner is not None

    owner.write_ready_record(
        workspace_root=tmp_path,
        projection_backend="embedded",
        graph_path=tmp_path / ".eventloom" / "projections" / "embedded.kuzu",
    )
    record = coordinator.read_owner_record()

    assert record is not None
    assert record["pid"] == os.getpid()
    assert record["projection_backend"] == "embedded"
    assert record["socket_path"] == str(coordinator.paths.socket_path)
    owner.close()


def test_repair_stale_runtime_cleans_unlocked_owner_files(tmp_path):
    """Init and doctor should clean stale embedded owner records before startup."""
    coordinator = _short_socket_coordinator(tmp_path)
    coordinator.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    coordinator.paths.owner_path.write_text(
        '{"pid": 999999999, "socket_path": "/tmp/missing-zaxy.sock"}',
        encoding="utf-8",
    )
    coordinator.paths.socket_path.write_text("", encoding="utf-8")

    report = coordinator.repair_stale_runtime()

    assert report["status"] == "ok"
    assert report["repaired"] is True
    assert not coordinator.paths.owner_path.exists()
    assert not coordinator.paths.socket_path.exists()


def test_owner_claim_close_is_idempotent(tmp_path):
    """Repeated close calls should not fail during shutdown cleanup."""
    coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(tmp_path / ".eventloom")
    owner = coordinator.try_claim_owner()
    assert owner is not None

    owner.write_ready_record(
        workspace_root=tmp_path,
        projection_backend="embedded",
        graph_path=tmp_path / ".eventloom" / "projections" / "embedded.kuzu",
    )

    owner.close()
    owner.close()

    assert not coordinator.paths.owner_path.exists()


def test_wait_for_owner_record_requires_published_record(tmp_path):
    """A held lock without a ready record should produce an actionable failure."""
    coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(tmp_path / ".eventloom")

    with pytest.raises(RuntimeError, match="has not published"):
        coordinator.wait_for_owner_record(timeout_seconds=0.01)


def test_wait_for_owner_record_requires_healthy_socket(tmp_path):
    """A stale record with a missing socket should not be treated as ready."""
    coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(tmp_path / ".eventloom")
    coordinator.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    coordinator.paths.owner_path.write_text(
        '{"pid": 999999999, "socket_path": "/tmp/missing-zaxy.sock"}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="did not open its proxy socket"):
        coordinator.wait_for_owner_record(timeout_seconds=0.01)


def test_read_owner_record_ignores_corrupt_json(tmp_path):
    """Corrupt owner metadata should be treated as missing runtime state."""
    coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(tmp_path / ".eventloom")
    coordinator.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    coordinator.paths.owner_path.write_text("{not-json", encoding="utf-8")

    assert coordinator.read_owner_record() is None


def test_try_claim_owner_closes_lock_file_on_unexpected_lock_error(tmp_path):
    """Unexpected lock failures should not leak file handles."""
    coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(tmp_path / ".eventloom")
    lock_file = MagicMock()
    lock_file.fileno.return_value = 123

    with (
        patch("pathlib.Path.open", return_value=lock_file),
        patch("fcntl.flock", side_effect=RuntimeError("lock subsystem failed")),
        pytest.raises(RuntimeError, match="lock subsystem failed"),
    ):
        coordinator.try_claim_owner()

    lock_file.close.assert_called_once_with()


def test_socket_accepts_connections_returns_false_for_missing_socket(tmp_path):
    """Socket health checks should fail closed for missing owner sockets."""
    assert (
        EmbeddedMcpRuntimeCoordinator._socket_accepts_connections(tmp_path / "missing.sock")
        is False
    )


def test_repair_runtime_reports_active_owner_with_healthy_socket(tmp_path):
    """Doctor should report a healthy owner rather than disrupting it."""
    coordinator = _short_socket_coordinator(tmp_path)
    owner = coordinator.try_claim_owner()
    assert owner is not None

    stop = threading.Event()
    ready = threading.Event()

    def socket_server() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(coordinator.paths.socket_path))
            server.listen(1)
            server.settimeout(0.05)
            ready.set()
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                with conn:
                    pass

    thread = threading.Thread(target=socket_server)
    thread.start()
    try:
        assert ready.wait(timeout=1)
        owner.write_ready_record(
            workspace_root=tmp_path,
            projection_backend="embedded",
            graph_path=tmp_path / ".eventloom" / "projections" / "embedded.kuzu",
        )

        report = coordinator.repair_stale_runtime()

        assert report["status"] == "ok"
        assert report["owner_active"] is True
        assert report["repaired"] is False
        assert report["pid"] == os.getpid()
    finally:
        stop.set()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(coordinator.paths.socket_path))
        except OSError:
            pass
        thread.join(timeout=1)
        owner.close()


def test_repair_runtime_warns_when_lock_is_held_without_healthy_socket(tmp_path):
    """Doctor should surface an explicit recovery action for broken active owners."""
    coordinator = EmbeddedMcpRuntimeCoordinator.from_eventloom_path(tmp_path / ".eventloom")
    owner = coordinator.try_claim_owner()
    assert owner is not None
    coordinator.paths.owner_path.write_text(
        '{"pid": 999999999, "socket_path": "/tmp/missing-zaxy.sock"}',
        encoding="utf-8",
    )

    try:
        report = coordinator.repair_stale_runtime()

        assert report["status"] == "warning"
        assert report["owner_active"] is True
        assert "run zaxy doctor again" in report["action"]
    finally:
        owner.close()


def test_wait_for_owner_record_accepts_healthy_socket(tmp_path):
    """Duplicate workers should proceed when the owner socket is reachable."""
    coordinator = _short_socket_coordinator(tmp_path)
    coordinator.paths.runtime_dir.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    ready = threading.Event()

    def socket_server() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(coordinator.paths.socket_path))
            server.listen(1)
            server.settimeout(0.05)
            ready.set()
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                with conn:
                    pass

    thread = threading.Thread(target=socket_server)
    thread.start()
    try:
        assert ready.wait(timeout=1)
        coordinator.paths.owner_path.write_text(
            (
                '{"pid": 123, "socket_path": '
                f'"{coordinator.paths.socket_path}", "projection_backend": "embedded"}}'
            ),
            encoding="utf-8",
        )

        record = coordinator.wait_for_owner_record(timeout_seconds=1)

        assert record["pid"] == 123
    finally:
        stop.set()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(coordinator.paths.socket_path))
        except OSError:
            pass
        thread.join(timeout=1)
