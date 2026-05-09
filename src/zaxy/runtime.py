"""Local runtime dependency orchestration for Zaxy."""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

RunFn = Callable[..., subprocess.CompletedProcess[str]]
PortProbe = Callable[[str, int], bool]
SleepFn = Callable[[float], None]


@dataclass(frozen=True)
class RuntimeCheck:
    """Local Neo4j runtime posture without mutating container state."""

    status: str
    message: str


@dataclass(frozen=True)
class LocalNeo4jRuntime:
    """Best-effort local Neo4j bootstrapper for development MCP startup."""

    uri: str
    user: str
    password: str
    enabled: bool = True
    image: str = "neo4j:5.26-community"
    container_name: str = "zaxy-neo4j"
    startup_timeout_seconds: float = 45.0
    runner: RunFn = subprocess.run
    port_probe: PortProbe | None = None
    sleeper: SleepFn = time.sleep

    def check(self) -> RuntimeCheck:
        """Report local Neo4j/Docker posture without starting containers."""
        endpoint = self._local_endpoint()
        if not self.enabled:
            return RuntimeCheck("warning", "Local Neo4j auto-start is disabled")
        if endpoint is None:
            return RuntimeCheck("ok", f"Neo4j URI {self.uri} is not a local auto-start target")
        host, port = endpoint
        if self._port_open(host, port):
            return RuntimeCheck("ok", f"Neo4j is reachable at {host}:{port}")
        if self._docker_available():
            return RuntimeCheck("warning", "Neo4j is not reachable; Docker is available")
        return RuntimeCheck("warning", "Neo4j is not reachable; Docker is unavailable")

    def ensure_available(self) -> None:
        """Start a local Neo4j container when localhost Bolt is unavailable."""
        endpoint = self._local_endpoint()
        if not self.enabled or endpoint is None:
            return

        host, port = endpoint
        if self._port_open(host, port):
            return

        if not self._docker_available():
            raise RuntimeError(
                "Local Neo4j is not reachable and Docker is required for automatic "
                "startup. Start Neo4j yourself, install/start Docker, or run "
                "zaxy serve --neo4j-uri <bolt-uri>."
            )

        if self._container_running():
            return
        if self._container_exists():
            self._docker(["docker", "start", self.container_name])
        else:
            self._docker([
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "-p",
                "127.0.0.1:7474:7474",
                "-p",
                "127.0.0.1:7687:7687",
                "-e",
                f"NEO4J_AUTH={self.user}/{self.password}",
                self.image,
            ])

        self._wait_for_port(host, port)

    def _local_endpoint(self) -> tuple[str, int] | None:
        parsed = urlparse(self.uri)
        if parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}:
            return None
        host = parsed.hostname
        port = parsed.port or 7687
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return None
        if port != 7687:
            return None
        return host, port

    def _port_open(self, host: str, port: int) -> bool:
        if self.port_probe is not None:
            return self.port_probe(host, port)
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    def _docker_available(self) -> bool:
        completed = self._run([
            "docker",
            "version",
            "--format",
            "{{.Server.Version}}",
        ])
        return completed.returncode == 0

    def _container_running(self) -> bool:
        completed = self._run([
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}}",
            self.container_name,
        ])
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def _container_exists(self) -> bool:
        completed = self._run([
            "docker",
            "inspect",
            "-f",
            "{{.Name}}",
            self.container_name,
        ])
        return completed.returncode == 0

    def _wait_for_port(self, host: str, port: int) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() <= deadline:
            if self._port_open(host, port):
                return
            self.sleeper(0.5)
        raise RuntimeError(
            f"Started Neo4j container '{self.container_name}', but Bolt did not "
            f"become reachable at {host}:{port} within "
            f"{self.startup_timeout_seconds:g}s."
        )

    def _docker(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        completed = self._run(cmd)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise RuntimeError(f"Docker command failed: {' '.join(cmd)}: {detail}")
        return completed

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr or ""
            return subprocess.CompletedProcess(
                cmd,
                124,
                stdout=stdout,
                stderr=stderr or f"Timed out after {exc.timeout:g}s",
            )
