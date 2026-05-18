"""Tests for local runtime dependency orchestration."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any

from zaxy.runtime import LocalNeo4jRuntime, LocalPgGraphRuntime


class FakeRunner:
    """Record docker commands and return configured results."""

    def __init__(self, results: list[subprocess.CompletedProcess[str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.results = results or []

    def __call__(self, cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if self.results:
            return self.results.pop(0)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def result(cmd: list[str] | None = None, returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(cmd or [], returncode, stdout=stdout, stderr=stderr)


def make_pggraph_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "pggraph"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "quickstart.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return repo


def test_local_neo4j_runtime_starts_named_container_when_port_closed() -> None:
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
        result(returncode=1),
        result(returncode=1),
        result(),
    ])
    port_states = iter([False, True])
    runtime = LocalNeo4jRuntime(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpassword",
        runner=runner,
        port_probe=lambda _host, _port: next(port_states),
        sleeper=lambda _seconds: None,
    )

    runtime.ensure_available()

    assert runner.calls == [
        ["docker", "version", "--format", "{{.Server.Version}}"],
        ["docker", "inspect", "-f", "{{.State.Running}}", "zaxy-neo4j"],
        ["docker", "inspect", "-f", "{{.Name}}", "zaxy-neo4j"],
        [
            "docker",
            "run",
            "-d",
            "--name",
            "zaxy-neo4j",
            "-p",
            "127.0.0.1:7474:7474",
            "-p",
            "127.0.0.1:7687:7687",
            "-e",
            "NEO4J_AUTH=neo4j/testpassword",
            "neo4j:5.26-community",
        ],
    ]


def test_local_neo4j_runtime_skips_docker_when_port_is_open() -> None:
    runner = FakeRunner()
    runtime = LocalNeo4jRuntime(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpassword",
        runner=runner,
        port_probe=lambda _host, _port: True,
    )

    runtime.ensure_available()

    assert runner.calls == []


def test_local_neo4j_runtime_check_does_not_start_container_when_port_closed() -> None:
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
    ])
    runtime = LocalNeo4jRuntime(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpassword",
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    check = runtime.check()

    assert check.status == "warning"
    assert check.message == "Neo4j is not reachable; Docker is available"
    assert runner.calls == [["docker", "version", "--format", "{{.Server.Version}}"]]


def test_local_neo4j_runtime_ignores_non_local_uris() -> None:
    runner = FakeRunner()
    runtime = LocalNeo4jRuntime(
        uri="bolt://neo4j:7687",
        user="neo4j",
        password="testpassword",
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    runtime.ensure_available()

    assert runner.calls == []


def test_local_neo4j_runtime_reports_missing_docker_actionably() -> None:
    runner = FakeRunner([
        result(returncode=1, stderr="docker unavailable"),
    ])
    runtime = LocalNeo4jRuntime(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpassword",
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "Docker is required" in str(exc)
        assert "zaxy serve --neo4j-uri" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_local_neo4j_runtime_reports_missing_docker_binary_actionably() -> None:
    def missing_runner(_cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    runtime = LocalNeo4jRuntime(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="testpassword",
        runner=missing_runner,
        port_probe=lambda _host, _port: False,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "Docker is required" in str(exc)
        assert "zaxy serve --neo4j-uri" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_local_pggraph_runtime_check_reports_actionable_installer_gap(tmp_path: Path) -> None:
    """pgGraph checks should distinguish Docker readiness from missing extension installer."""
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
    ])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=tmp_path / "missing-pggraph",
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    check = runtime.check()

    assert check.status == "warning"
    assert "pgGraph is not reachable" in check.message
    assert "PGGRAPH_REPO" in check.message
    assert runner.calls == [["docker", "version", "--format", "{{.Server.Version}}"]]


def test_local_pggraph_runtime_check_handles_disabled_remote_reachable_and_installer_ready(tmp_path: Path) -> None:
    """pgGraph checks should cover non-mutating readiness states."""
    disabled = LocalPgGraphRuntime(
        dsn="postgresql://postgres:secret@db.internal:5432/zaxy",
        enabled=False,
    )
    remote = LocalPgGraphRuntime(
        dsn="postgresql://postgres:secret@db.internal:5432/zaxy",
        runner=FakeRunner(),
    )
    reachable = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        runner=FakeRunner(),
        port_probe=lambda _host, _port: True,
    )
    installer_ready_runner = FakeRunner([result(stdout="24.0.0\n")])
    installer_ready = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=make_pggraph_repo(tmp_path),
        runner=installer_ready_runner,
        port_probe=lambda _host, _port: False,
    )

    assert disabled.check().message == "Local pgGraph auto-start is disabled"
    assert remote.check().status == "ok"
    assert "secret" not in remote.check().message
    assert reachable.check().message == "pgGraph/PostgreSQL is reachable at localhost:5432"
    assert installer_ready.check().message == "pgGraph is not reachable; Docker and pgGraph installer are available"


def test_local_pggraph_runtime_starts_postgres_and_runs_quickstart(tmp_path: Path) -> None:
    """pgGraph bootstrap should start a local Postgres container then install pgGraph."""
    repo = make_pggraph_repo(tmp_path)
    quickstart = repo / "scripts" / "quickstart.sh"
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
        result(returncode=1),
        result(returncode=1),
        result(),
        result(),
    ])
    port_states = iter([False, True])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://zaxy:secret@localhost:5432/zaxy",
        pggraph_repo=repo,
        runner=runner,
        port_probe=lambda _host, _port: next(port_states),
        sleeper=lambda _seconds: None,
    )

    runtime.ensure_available()

    assert runner.calls == [
        ["docker", "version", "--format", "{{.Server.Version}}"],
        ["docker", "inspect", "-f", "{{.State.Running}}", "zaxy-pggraph"],
        ["docker", "inspect", "-f", "{{.Name}}", "zaxy-pggraph"],
        [
            "docker",
            "run",
            "-d",
            "--name",
            "zaxy-pggraph",
            "-p",
            "127.0.0.1:5432:5432",
            "-e",
            "POSTGRES_USER=zaxy",
            "-e",
            "POSTGRES_PASSWORD=secret",
            "-e",
            "POSTGRES_DB=zaxy",
            "pgvector/pgvector:pg17",
        ],
        ["bash", str(quickstart), "docker", "zaxy-pggraph", "17", "zaxy", "zaxy"],
    ]


def test_local_pggraph_runtime_starts_existing_container_and_runs_quickstart(tmp_path: Path) -> None:
    """Existing stopped pgGraph containers should be started before extension bootstrap."""
    repo = make_pggraph_repo(tmp_path)
    quickstart = repo / "scripts" / "quickstart.sh"
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
        result(stdout="false\n"),
        result(stdout="/zaxy-pggraph\n"),
        result(),
        result(),
    ])
    port_states = iter([False, True])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:15432/zaxy",
        pggraph_repo=repo,
        image="pgvector/pgvector:pg18",
        runner=runner,
        port_probe=lambda _host, _port: next(port_states),
        sleeper=lambda _seconds: None,
    )

    runtime.ensure_available()

    assert runner.calls == [
        ["docker", "version", "--format", "{{.Server.Version}}"],
        ["docker", "inspect", "-f", "{{.State.Running}}", "zaxy-pggraph"],
        ["docker", "inspect", "-f", "{{.Name}}", "zaxy-pggraph"],
        ["docker", "start", "zaxy-pggraph"],
        ["bash", str(quickstart), "docker", "zaxy-pggraph", "18", "zaxy", "postgres"],
    ]


def test_local_pggraph_runtime_skips_when_disabled_remote_or_open() -> None:
    """Non-local or already reachable pgGraph endpoints should not invoke Docker."""
    disabled_runner = FakeRunner()
    remote_runner = FakeRunner()
    open_runner = FakeRunner()

    LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        enabled=False,
        runner=disabled_runner,
        port_probe=lambda _host, _port: False,
    ).ensure_available()
    LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@db.internal:5432/zaxy",
        runner=remote_runner,
        port_probe=lambda _host, _port: False,
    ).ensure_available()
    LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        runner=open_runner,
        port_probe=lambda _host, _port: True,
    ).ensure_available()

    assert disabled_runner.calls == []
    assert remote_runner.calls == []
    assert open_runner.calls == []


def test_local_pggraph_runtime_skips_unsupported_dsn_scheme() -> None:
    runner = FakeRunner()
    runtime = LocalPgGraphRuntime(
        dsn="sqlite:///tmp/zaxy.db",
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    assert runtime.check().status == "ok"
    runtime.ensure_available()

    assert runner.calls == []


def test_local_pggraph_runtime_socket_probe_reports_closed_port(monkeypatch) -> None:
    def closed_connection(_address: tuple[str, int], timeout: float) -> object:
        raise OSError("closed")

    monkeypatch.setattr(socket, "create_connection", closed_connection)
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        runner=FakeRunner([result(returncode=1)]),
    )

    assert runtime.check().message == "pgGraph is not reachable; Docker is unavailable"


def test_local_pggraph_runtime_reports_missing_docker_actionably(tmp_path: Path) -> None:
    runner = FakeRunner([
        result(returncode=1, stderr="docker unavailable"),
    ])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=make_pggraph_repo(tmp_path),
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "Docker is required" in str(exc)
        assert "PGGRAPH_DSN" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_local_pggraph_runtime_check_reports_missing_docker() -> None:
    runner = FakeRunner([
        result(returncode=1, stderr="docker unavailable"),
    ])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    check = runtime.check()

    assert check.status == "warning"
    assert check.message == "pgGraph is not reachable; Docker is unavailable"


def test_local_pggraph_runtime_reports_missing_docker_binary_actionably(tmp_path: Path) -> None:
    def missing_runner(_cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=make_pggraph_repo(tmp_path),
        runner=missing_runner,
        port_probe=lambda _host, _port: False,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "Docker is required" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_local_pggraph_runtime_handles_command_timeout_as_runtime_failure(tmp_path: Path) -> None:
    def timeout_runner(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd, timeout=1, output=b"partial")

    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=make_pggraph_repo(tmp_path),
        runner=timeout_runner,
        port_probe=lambda _host, _port: False,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "Docker is required" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_local_pggraph_runtime_raises_when_pggraph_repo_is_missing() -> None:
    """Automatic pgGraph bootstrap should not silently run plain Postgres without pgGraph."""
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
    ])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=None,
        runner=runner,
        port_probe=lambda _host, _port: False,
        sleeper=lambda _seconds: None,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "PGGRAPH_REPO" in str(exc)
        assert "pgGraph installer" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert runner.calls == [["docker", "version", "--format", "{{.Server.Version}}"]]


def test_local_pggraph_runtime_times_out_when_started_container_never_opens_port(tmp_path: Path) -> None:
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
        result(returncode=1),
        result(returncode=1),
        result(),
    ])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=make_pggraph_repo(tmp_path),
        runner=runner,
        port_probe=lambda _host, _port: False,
        sleeper=lambda _seconds: None,
        startup_timeout_seconds=0.01,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "did not become reachable" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_local_pggraph_runtime_reports_command_failures(tmp_path: Path) -> None:
    repo = make_pggraph_repo(tmp_path)
    runner = FakeRunner([
        result(stdout="24.0.0\n"),
        result(stdout="false\n"),
        result(stdout="/zaxy-pggraph\n"),
        result(returncode=1, stderr="start failed"),
    ])
    runtime = LocalPgGraphRuntime(
        dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        pggraph_repo=repo,
        runner=runner,
        port_probe=lambda _host, _port: False,
    )

    try:
        runtime.ensure_available()
    except RuntimeError as exc:
        assert "docker start zaxy-pggraph" in str(exc)
        assert "start failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
