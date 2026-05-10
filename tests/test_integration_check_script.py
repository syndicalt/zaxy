"""Tests for the Neo4j integration check helper."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT_PATH = Path("scripts/integration-check.sh")


def test_integration_check_script_has_valid_bash_syntax() -> None:
    """The integration helper should remain shell-parseable."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_integration_check_script_can_start_neo4j_services_explicitly() -> None:
    """Start mode should generate TLS certs and boot both Neo4j test services."""
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--start" in script
    assert "generate-certs.sh" in script
    assert "neo4j-test neo4j-tls" in script
    assert "7688" in script
    assert "7689" in script


def test_integration_check_script_can_skip_graph_tests_when_services_are_absent() -> None:
    """Skip mode should avoid graph integration tests only after an explicit check."""
    script = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--skip-if-unavailable" in script
    assert "--ignore=tests/test_graph.py" in script
    assert "docker compose up -d neo4j-test neo4j-tls" in script


def test_testing_docs_describe_integration_check_helper() -> None:
    """The documented test workflow should point at the helper."""
    docs = Path("docs/testing.md").read_text(encoding="utf-8")

    assert "scripts/integration-check.sh --start" in docs
    assert "scripts/integration-check.sh --skip-if-unavailable" in docs
    assert "scripts/integration-check.sh --require" in docs
