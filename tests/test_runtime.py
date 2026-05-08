"""Tests for local runtime dependency orchestration."""

from __future__ import annotations

import subprocess
from typing import Any

from zaxy.runtime import LocalNeo4jRuntime


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
