"""Local runtime dependency orchestration for Zaxy."""

from __future__ import annotations

import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    display_name: str = "Neo4j"

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


@dataclass(frozen=True)
class LocalEmbeddedGraphRuntime:
    """Local embedded graph projection posture.

    Embedded graph backends are projection files, not services. The runtime
    check therefore reports filesystem readiness and `ensure_available` only
    creates the parent directory needed for lazy projection creation.
    """

    path: str | Path
    display_name: str = "embedded graph"

    def check(self) -> RuntimeCheck:
        """Report local embedded projection posture without mutating files."""
        graph_path = Path(self.path)
        if graph_path.exists():
            return RuntimeCheck("ok", f"Embedded graph projection exists at {graph_path}")
        if graph_path.parent.exists():
            return RuntimeCheck("ok", f"Embedded graph projection will be created lazily at {graph_path}")
        return RuntimeCheck(
            "ok",
            f"Embedded graph projection parent will be created lazily at {graph_path.parent}",
        )

    def ensure_available(self) -> None:
        """Create the local projection parent directory."""
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LocalPgGraphRuntime:
    """Best-effort local pgGraph/PostgreSQL bootstrapper for development startup."""

    dsn: str
    enabled: bool = True
    image: str = "pgvector/pgvector:pg17"
    container_name: str = "zaxy-pggraph"
    pggraph_repo: str | Path | None = None
    startup_timeout_seconds: float = 45.0
    runner: RunFn = subprocess.run
    port_probe: PortProbe | None = None
    sleeper: SleepFn = time.sleep
    display_name: str = "pgGraph"

    def check(self) -> RuntimeCheck:
        """Report local pgGraph/PostgreSQL bootstrap posture without mutation."""
        endpoint = self._local_endpoint()
        if not self.enabled:
            return RuntimeCheck("warning", "Local pgGraph auto-start is disabled")
        if endpoint is None:
            return RuntimeCheck("ok", f"pgGraph DSN {self._redacted_dsn()} is not a local auto-start target")
        host, port = endpoint
        if self._port_open(host, port):
            return RuntimeCheck("ok", f"pgGraph/PostgreSQL is reachable at {host}:{port}")
        if not self._docker_available():
            return RuntimeCheck("warning", "pgGraph is not reachable; Docker is unavailable")
        if self._quickstart_script() is None:
            return RuntimeCheck(
                "warning",
                "pgGraph is not reachable; Docker is available; set PGGRAPH_REPO to a local "
                "pgGraph checkout containing scripts/quickstart.sh for automatic extension bootstrap",
            )
        return RuntimeCheck("warning", "pgGraph is not reachable; Docker and pgGraph installer are available")

    def ensure_available(self) -> None:
        """Start local Postgres and run the pgGraph installer for local development."""
        endpoint = self._local_endpoint()
        if not self.enabled or endpoint is None:
            return

        host, port = endpoint
        if self._port_open(host, port):
            return

        if not self._docker_available():
            raise RuntimeError(
                "Local pgGraph is not reachable and Docker is required for automatic startup. "
                "Start pgGraph/PostgreSQL yourself, install/start Docker, or configure PGGRAPH_DSN."
            )

        script = self._quickstart_script()
        if script is None:
            raise RuntimeError(
                "Local pgGraph is not reachable, and automatic startup requires the pgGraph installer. "
                "Set PGGRAPH_REPO to a local pgGraph checkout containing scripts/quickstart.sh, "
                "or start a pgGraph-enabled PostgreSQL endpoint yourself."
            )

        if self._container_running():
            pass
        elif self._container_exists():
            self._run_checked(["docker", "start", self.container_name])
        else:
            database = self._database()
            user = self._user()
            password = self._password()
            self._run_checked(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self.container_name,
                    "-p",
                    f"127.0.0.1:{port}:5432",
                    "-e",
                    f"POSTGRES_USER={user}",
                    "-e",
                    f"POSTGRES_PASSWORD={password}",
                    "-e",
                    f"POSTGRES_DB={database}",
                    self.image,
                ]
            )

        self._wait_for_port(host, port)
        self._install_pggraph(script)

    def _local_endpoint(self) -> tuple[str, int] | None:
        parsed = urlparse(self.dsn)
        if parsed.scheme not in {"postgresql", "postgres"}:
            return None
        host = parsed.hostname
        port = parsed.port or 5432
        if host not in {"localhost", "127.0.0.1", "::1"}:
            return None
        return host, port

    def _database(self) -> str:
        parsed = urlparse(self.dsn)
        return (parsed.path or "/postgres").lstrip("/") or "postgres"

    def _user(self) -> str:
        return urlparse(self.dsn).username or "postgres"

    def _password(self) -> str:
        return urlparse(self.dsn).password or "postgres"

    def _postgres_major(self) -> str:
        image_tag = self.image.rsplit(":", maxsplit=1)[-1]
        digits = "".join(char for char in image_tag if char.isdigit())
        return digits or "17"

    def _quickstart_script(self) -> Path | None:
        if self.pggraph_repo is None:
            return None
        script = Path(self.pggraph_repo).expanduser() / "scripts" / "quickstart.sh"
        if script.is_file():
            return script
        return None

    def _install_pggraph(self, script: Path) -> None:
        self._run_checked(
            [
                "bash",
                str(script),
                "docker",
                self.container_name,
                self._postgres_major(),
                self._database(),
                self._user(),
            ]
        )

    def _redacted_dsn(self) -> str:
        parsed = urlparse(self.dsn)
        if parsed.password is None:
            return self.dsn
        netloc = parsed.netloc.replace(f":{parsed.password}@", ":****@")
        return parsed._replace(netloc=netloc).geturl()

    def _port_open(self, host: str, port: int) -> bool:
        if self.port_probe is not None:
            return self.port_probe(host, port)
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    def _docker_available(self) -> bool:
        completed = self._run(["docker", "version", "--format", "{{.Server.Version}}"])
        return completed.returncode == 0

    def _container_running(self) -> bool:
        completed = self._run(["docker", "inspect", "-f", "{{.State.Running}}", self.container_name])
        return completed.returncode == 0 and completed.stdout.strip() == "true"

    def _container_exists(self) -> bool:
        completed = self._run(["docker", "inspect", "-f", "{{.Name}}", self.container_name])
        return completed.returncode == 0

    def _wait_for_port(self, host: str, port: int) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() <= deadline:
            if self._port_open(host, port):
                return
            self.sleeper(0.5)
        raise RuntimeError(
            f"Started pgGraph/PostgreSQL container '{self.container_name}', but PostgreSQL did not "
            f"become reachable at {host}:{port} within {self.startup_timeout_seconds:g}s."
        )

    def _run_checked(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        completed = self._run(cmd)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
            raise RuntimeError(f"Runtime command failed: {' '.join(cmd)}: {detail}")
        return completed

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self.runner(cmd, capture_output=True, text=True, timeout=600)
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
