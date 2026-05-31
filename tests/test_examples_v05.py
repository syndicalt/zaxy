"""Smoke tests for v0.5 public examples."""

from __future__ import annotations

import json
import subprocess
import sys


def test_single_agent_memory_example_runs() -> None:
    """Single-agent example should run without sidecars and print JSON evidence."""
    result = subprocess.run(
        [sys.executable, "examples/single_agent_memory.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "single-agent-demo"
    assert payload["bootstrap"]["session_id"] == "single-agent-demo"
    assert payload["checkout"]["session_id"] == "single-agent-demo"
    assert payload["event_count"] >= 2


def test_coordinate_three_worker_example_runs() -> None:
    """Coordinate example should produce a complete mission summary."""
    result = subprocess.run(
        [sys.executable, "examples/coordinate_three_worker_project.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mission_id"] == "auth-main"
    assert payload["worker_count"] == 3
    assert payload["accepted_count"] >= 1
    assert payload["handoff_id"]
    assert payload["approval_packet_id"].startswith("auth-main:approval:")
    assert payload["approval_findings_count"] >= 3
    assert payload["approval_reviewed_count"] == 3
    assert payload["approval_promoted_count"] == 1
    assert "resolve_conflict" in payload["approval_next_actions"]
    assert "refresh_stale_evidence" in payload["approval_next_actions"]
    assert payload["inspection_sections"] == [
        "brief",
        "worker_ledgers",
        "findings",
        "evidence",
        "decisions",
        "promoted_state",
        "handoffs",
        "conflicts",
        "approval_packet",
    ]
    assert payload["audit_event_count"] >= 10
    assert payload["audit_has_event_hashes"] is True


def test_langgraph_example_runs_without_langgraph_dependency() -> None:
    """LangGraph example should smoke-test the dependency-light adapter."""
    result = subprocess.run(
        [sys.executable, "examples/langgraph_memory.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "langgraph-demo"
    assert payload["has_zaxy_context"] is True
    assert payload["kind"] in {"memory_checkout", "context_assembly"}


def test_openai_compatible_example_runs_without_provider_dependency() -> None:
    """OpenAI-compatible example should smoke-test outside-MCP model activation."""
    result = subprocess.run(
        [sys.executable, "examples/openai_compatible_memory.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "openai-compatible-demo"
    assert payload["has_zaxy_context"] is True
    assert payload["kind"] == "memory_checkout"
    assert payload["assistant_content"] == "Memory was injected: True"


def test_claude_compatible_example_runs_without_provider_dependency() -> None:
    """Claude-compatible example should smoke-test outside-MCP model activation."""
    result = subprocess.run(
        [sys.executable, "examples/claude_compatible_memory.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "claude-compatible-demo"
    assert payload["has_zaxy_context"] is True
    assert payload["kind"] == "memory_checkout"
    assert payload["assistant_content"] == "Memory was injected: True"
