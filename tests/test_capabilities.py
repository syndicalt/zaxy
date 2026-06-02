"""Tests for model-facing Zaxy memory capability manifests."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.capabilities import (
    build_memory_bootstrap,
    build_memory_capabilities,
    format_memory_bootstrap,
    format_memory_capabilities,
)
from zaxy.codex_capture import write_codex_capture_config
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
    assert manifest["reminder_policy"]["triggers"] == [
        "session_start",
        "resume",
        "compaction",
        "long_session",
        "long_tool_run",
        "where_are_we_question",
    ]
    assert manifest["reminder_policy"]["event_type"] == "memory.reminder.suggested"
    assert "memory_checkout" in {tool["name"] for tool in manifest["tools"]}
    assert "Do not treat session-start memory as sufficient" in manifest["prompt"]


def test_capabilities_manifest_ignores_native_eventloom_log(tmp_path: Path) -> None:
    """Capabilities should tolerate native Eventloom event logs next to Zaxy logs."""
    eventloom = tmp_path / ".eventloom"
    eventloom.mkdir()
    (eventloom / "events.jsonl").write_text(
        json.dumps(
            {
                "id": "evt_demo_goal",
                "type": "goal.created",
                "actorId": "user",
                "threadId": "thread_main",
                "payload": {"title": "Native Eventloom event"},
                "integrity": {
                    "hash": "sha256:abc123",
                    "previousHash": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    EventLog(eventloom / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Built mixed Eventloom status."},
        thread="agent",
    )

    manifest = build_memory_capabilities(
        eventloom_path=eventloom,
        session_id="agent",
        workspace_root=tmp_path,
    )

    assert manifest["status"]["eventloom"]["latest_seq"] == 1
    assert manifest["status"]["eventloom"]["integrity_ok"] is True
    assert manifest["status"]["eventloom"]["skipped_log_count"] == 1


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
    assert "memory.reminder.suggested" in text
    assert len(text.splitlines()) <= 32


def test_memory_bootstrap_packages_session_start_handoff(tmp_path: Path) -> None:
    """Bootstrap should give a model one compact startup packet for Zaxy-aware work."""
    eventloom = tmp_path / ".eventloom"
    EventLog(eventloom / "agent.jsonl").append(
        "decision.made",
        actor="codex",
        payload={"summary": "Memory Checkout is the context contract."},
        thread="agent",
    )
    write_codex_capture_config(
        workspace=tmp_path,
        eventloom_path=eventloom,
        session_id="agent",
        codex_home=tmp_path / "codex-home",
    )

    bootstrap = build_memory_bootstrap(
        eventloom_path=eventloom,
        session_id="agent",
        workspace_root=tmp_path,
        current_task="continue the roadmap",
    )

    assert bootstrap["session_id"] == "agent"
    assert bootstrap["mode"] == "session_start"
    assert bootstrap["startup_sequence"][0]["tool"] == "memory_capabilities"
    assert bootstrap["startup_sequence"][1]["tool"] == "memory_checkout"
    assert bootstrap["startup_sequence"][1]["arguments"]["query"] == "continue the roadmap"
    assert bootstrap["capture"]["configured"] is True
    assert bootstrap["capture"]["running"] is False
    assert bootstrap["trust_policy"] == {
        "prefer": "cited current facts from memory_checkout",
        "ignore": "uncited, superseded, or warning-bearing context until refreshed",
        "record": "meaningful decisions, completed work, corrections, and retrieval feedback",
    }
    assert "Call memory_checkout before answering roadmap or implementation questions." in bootstrap["prompt"]


def test_format_memory_bootstrap_is_short_and_actionable(tmp_path: Path) -> None:
    """Bootstrap text should be small enough to inject into session-start context."""
    bootstrap = build_memory_bootstrap(
        eventloom_path=tmp_path / ".eventloom",
        session_id="agent",
        workspace_root=tmp_path,
    )

    text = format_memory_bootstrap(bootstrap)

    assert "# Zaxy Session Bootstrap" in text
    assert "Session: agent" in text
    assert "1. memory_capabilities" in text
    assert "2. memory_checkout" in text
    assert "Capture: not configured, not running" in text
    assert len(text.splitlines()) <= 24
