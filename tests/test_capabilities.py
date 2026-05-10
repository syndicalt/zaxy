"""Tests for model-facing Zaxy memory capability manifests."""

from __future__ import annotations

from pathlib import Path

from zaxy.capabilities import build_memory_capabilities, format_memory_capabilities
from zaxy.event import EventLog


def test_capabilities_manifest_guides_periodic_memory_refresh(tmp_path: Path) -> None:
    """The manifest should tell a model how to use Zaxy throughout a session."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Built memory capabilities."},
        thread="agent",
    )

    manifest = build_memory_capabilities(
        eventloom_path=eventloom,
        session_id="agent",
        workspace_root=tmp_path,
    )

    assert manifest["session_id"] == "agent"
    assert manifest["purpose"].startswith("Zaxy is the active persistent memory substrate")
    assert manifest["status"]["eventloom"]["latest_seq"] == 1
    assert manifest["status"]["eventloom"]["integrity_ok"] is True
    assert manifest["recommended_next_call"]["tool"] == "memory_checkout"
    assert "after_compaction_or_resume" in manifest["ambient_loop"]
    assert manifest["ambient_loop"]["before_major_work"]["tool"] == "memory_checkout"
    assert manifest["ambient_loop"]["after_meaningful_work"]["tool"] == "context_after_turn"
    assert manifest["ambient_loop"]["when_context_is_used"]["tool"] == "memory_feedback"
    assert "memory_checkout" in {tool["name"] for tool in manifest["tools"]}
    assert "Do not treat session-start memory as sufficient" in manifest["prompt"]


def test_format_memory_capabilities_is_prompt_ready_and_concise(tmp_path: Path) -> None:
    """Human/model text should be compact and emphasize the ambient loop."""
    manifest = build_memory_capabilities(
        eventloom_path=tmp_path / ".eventloom",
        session_id="agent",
        workspace_root=tmp_path,
    )

    text = format_memory_capabilities(manifest)

    assert "# Zaxy Memory Contract" in text
    assert "Session: agent" in text
    assert "memory_checkout" in text
    assert "after compaction/resume" in text
    assert len(text.splitlines()) <= 32
