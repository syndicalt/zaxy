"""Tests for embedded MCP runtime ownership and proxy coordination."""

from __future__ import annotations

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
