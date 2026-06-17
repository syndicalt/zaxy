"""Workspace runtime coordination for local MCP servers.

Embedded graph backends can only be opened by one read-write process at a time.
This module coordinates that single owner so duplicate stdio MCP processes can
proxy to it instead of opening the embedded graph themselves.
"""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, cast


@dataclass(frozen=True)
class EmbeddedMcpRuntimePaths:
    """Filesystem paths used to coordinate one embedded MCP owner."""

    runtime_dir: Path
    lock_path: Path
    owner_path: Path
    socket_path: Path


class EmbeddedMcpOwnerClaim:
    """Held owner lock for the lifetime of the embedded MCP owner process."""

    def __init__(self, paths: EmbeddedMcpRuntimePaths, lock_file: IO[str]) -> None:
        self.paths = paths
        self._lock_file = lock_file
        self._closed = False

    def write_ready_record(
        self,
        *,
        workspace_root: str | Path,
        projection_backend: str,
        graph_path: str | Path,
    ) -> None:
        """Publish owner connection metadata after the socket is listening."""
        record = {
            "pid": os.getpid(),
            "workspace_root": str(Path(workspace_root).resolve()),
            "projection_backend": projection_backend,
            "graph_path": str(Path(graph_path)),
            "socket_path": str(self.paths.socket_path),
            "started_at": time.time(),
        }
        self.paths.owner_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.paths.owner_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.paths.owner_path)

    def close(self) -> None:
        """Release the owner lock and remove runtime metadata."""
        if self._closed:
            return
        self._closed = True
        for path in (self.paths.owner_path, self.paths.socket_path):
            with suppress(FileNotFoundError):
                path.unlink()
        try:
            import fcntl

            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()


class EmbeddedMcpRuntimeCoordinator:
    """Coordinates a single embedded MCP owner per Eventloom directory."""

    def __init__(self, paths: EmbeddedMcpRuntimePaths) -> None:
        self.paths = paths

    @classmethod
    def _from_runtime_dir(cls, runtime_dir: Path) -> EmbeddedMcpRuntimeCoordinator:
        return cls(
            EmbeddedMcpRuntimePaths(
                runtime_dir=runtime_dir,
                lock_path=runtime_dir / "zaxy-embedded-owner.lock",
                owner_path=runtime_dir / "zaxy-embedded-owner.json",
                socket_path=runtime_dir / "zaxy-embedded-owner.sock",
            )
        )

    @classmethod
    def from_eventloom_path(cls, eventloom_path: str | Path) -> EmbeddedMcpRuntimeCoordinator:
        return cls._from_runtime_dir(Path(eventloom_path) / "runtime")

    @classmethod
    def from_embedded_graph_path(
        cls, embedded_graph_path: str | Path
    ) -> EmbeddedMcpRuntimeCoordinator:
        """Coordinate on the embedded STORE path, not the eventloom path.

        The owner lock must protect the actual store: two processes that open the
        same embedded store must coordinate even if they resolved their eventloom
        path differently (the divergence that let multiple owners corrupt one
        store). The runtime dir is derived from the *resolved* store path, so it
        is identical for any process pointing at the same store. For the standard
        layout (``<eventloom>/projections/embedded.kuzu``) this resolves to the
        same ``<eventloom>/runtime`` location as :meth:`from_eventloom_path`.
        """
        store = Path(embedded_graph_path).resolve()
        eventloom = store.parent.parent if store.parent.name == "projections" else store.parent
        return cls._from_runtime_dir(eventloom / "runtime")

    def try_claim_owner(self) -> EmbeddedMcpOwnerClaim | None:
        """Return an owner claim, or None when another owner is active."""
        self.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        lock_file = self.paths.lock_path.open("a+", encoding="utf-8")
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return None
        except Exception:
            lock_file.close()
            raise

        self._clear_stale_runtime_files()
        return EmbeddedMcpOwnerClaim(self.paths, lock_file)

    def read_owner_record(self) -> dict[str, Any] | None:
        """Read current owner metadata if present and valid."""
        try:
            record = json.loads(self.paths.owner_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return cast(dict[str, Any], record)

    def wait_for_owner_record(self, timeout_seconds: float = 10.0) -> dict[str, Any]:
        """Wait for the active owner to publish a usable socket path."""
        deadline = time.monotonic() + timeout_seconds
        last_record: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            record = self.read_owner_record()
            if record is not None:
                last_record = record
                socket_path = Path(str(record.get("socket_path", "")))
                if socket_path.exists() and self._socket_accepts_connections(socket_path):
                    return record
            time.sleep(0.05)
        if last_record is None:
            raise RuntimeError("embedded MCP owner is active but has not published a runtime record")
        raise RuntimeError(
            "embedded MCP owner did not open its proxy socket at "
            f"{last_record.get('socket_path')}"
        )

    def repair_stale_runtime(
        self, *, reap: bool = False, expected_graph_path: str | Path | None = None
    ) -> dict[str, Any]:
        """Clean stale owner metadata, optionally reaping a live-broken owner.

        With ``reap=False`` (default) this only cleans metadata when the owner
        lock is free (a dead owner) and otherwise reports — preserving the
        read-only diagnostic contract. With ``reap=True`` a *live-but-broken*
        owner (lock held, owner socket not accepting) is recovered: the owner
        process is terminated and the lock reclaimed, but only after verifying it
        is genuinely a ``zaxy serve`` process for *this* store (never a healthy
        owner, never another workspace's server, never a non-Zaxy process).
        """
        had_runtime_files = self.paths.owner_path.exists() or self.paths.socket_path.exists()
        owner = self.try_claim_owner()
        if owner is not None:
            owner.close()
            return {
                "status": "ok",
                "message": (
                    "embedded MCP runtime is clean"
                    if not had_runtime_files
                    else "stale embedded MCP runtime metadata was removed"
                ),
                "repaired": had_runtime_files,
                "owner_active": False,
                "owner_path": str(self.paths.owner_path),
                "socket_path": str(self.paths.socket_path),
            }

        record = self.read_owner_record()
        socket_path = Path(str(record.get("socket_path", ""))) if record else self.paths.socket_path
        if record is not None and socket_path.exists() and self._socket_accepts_connections(socket_path):
            return {
                "status": "ok",
                "message": f"embedded MCP owner is active at pid {record.get('pid')}",
                "repaired": False,
                "owner_active": True,
                "owner_path": str(self.paths.owner_path),
                "socket_path": str(socket_path),
                "pid": record.get("pid"),
            }

        # Live-but-broken owner: lock held, but the owner socket is dead. Reap
        # only when asked AND the holder is verifiably a zaxy serve for this store.
        if reap and self._owner_pid_is_reapable(record, expected_graph_path):
            pid = int(record["pid"])  # type: ignore[index]
            reaped = self._terminate_pid(pid)
            if reaped:
                self._clear_stale_runtime_files()
                reclaimed = self.try_claim_owner()
                if reclaimed is not None:
                    reclaimed.close()
                    return {
                        "status": "ok",
                        "message": f"reaped broken embedded MCP owner pid {pid} and reclaimed runtime",
                        "repaired": True,
                        "reaped_pid": pid,
                        "owner_active": False,
                        "owner_path": str(self.paths.owner_path),
                        "socket_path": str(socket_path),
                    }

        return {
            "status": "warning",
            "message": "embedded MCP owner lock is held but no healthy owner socket is available",
            "repaired": False,
            "owner_active": True,
            "owner_path": str(self.paths.owner_path),
            "socket_path": str(socket_path),
            "action": "Fully exit stale Codex/Zaxy processes for this workspace, then run zaxy doctor again.",
        }

    def _owner_pid_is_reapable(
        self, record: dict[str, Any] | None, expected_graph_path: str | Path | None
    ) -> bool:
        """Return whether the recorded owner is a safe-to-reap broken zaxy serve.

        Conservative by construction: a process is reapable only when it is alive,
        is not this process, its command line is a ``zaxy serve``, and — when an
        expected store path is given — the owner record's ``graph_path`` resolves
        to that same store. Any unverifiable condition (including non-Linux where
        ``/proc`` is unavailable) returns False, so nothing is killed on doubt.
        """
        if not isinstance(record, dict):
            return False
        pid = record.get("pid")
        if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
            return False
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError, OSError):
            return False
        cmdline = self._process_cmdline(pid)
        if cmdline is None or not ("zaxy" in cmdline and "serve" in cmdline):
            return False
        if expected_graph_path is not None:
            recorded = record.get("graph_path")
            if not isinstance(recorded, str):
                return False
            with suppress(OSError):
                if Path(recorded).resolve() != Path(expected_graph_path).resolve():
                    return False
        return True

    @staticmethod
    def _process_cmdline(pid: int) -> str | None:
        """Return a process's command line via /proc, or None if unavailable."""
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except (FileNotFoundError, OSError):
            return None
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace")

    @staticmethod
    def _terminate_pid(pid: int, *, timeout_seconds: float = 3.0) -> bool:
        """Terminate a pid (SIGTERM then SIGKILL); return whether it is gone."""
        import signal

        with suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                return True
            time.sleep(0.05)
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.05)
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        return False

    def _clear_stale_runtime_files(self) -> None:
        for path in (self.paths.owner_path, self.paths.socket_path):
            with suppress(FileNotFoundError):
                path.unlink()

    @staticmethod
    def _socket_accepts_connections(socket_path: Path) -> bool:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(0.2)
            client.connect(str(socket_path))
            return True
        except OSError:
            return False
        finally:
            client.close()
