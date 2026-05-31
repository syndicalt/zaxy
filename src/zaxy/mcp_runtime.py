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
    def from_eventloom_path(cls, eventloom_path: str | Path) -> EmbeddedMcpRuntimeCoordinator:
        eventloom = Path(eventloom_path)
        runtime_dir = eventloom / "runtime"
        return cls(
            EmbeddedMcpRuntimePaths(
                runtime_dir=runtime_dir,
                lock_path=runtime_dir / "zaxy-embedded-owner.lock",
                owner_path=runtime_dir / "zaxy-embedded-owner.json",
                socket_path=runtime_dir / "zaxy-embedded-owner.sock",
            )
        )

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

    def repair_stale_runtime(self) -> dict[str, Any]:
        """Clean stale owner metadata when no live owner lock is held."""
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

        return {
            "status": "warning",
            "message": "embedded MCP owner lock is held but no healthy owner socket is available",
            "repaired": False,
            "owner_active": True,
            "owner_path": str(self.paths.owner_path),
            "socket_path": str(socket_path),
            "action": "Fully exit stale Codex/Zaxy processes for this workspace, then run zaxy doctor again.",
        }

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
