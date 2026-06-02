"""Tests for replay-only purpose control-plane diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import zaxy.purpose_control as purpose_control
from zaxy.coordination import CoordinationManager
from zaxy.event import EventLog
from zaxy.purpose_control import (
    build_purpose_feedback,
    build_purpose_lanes,
    build_purpose_status,
    format_purpose_feedback,
    format_purpose_lanes,
    format_purpose_status,
)
from zaxy.resources.coordinationbench.unsupported_runner import main as unsupported_runner_main


def _write_purpose_fixture(eventloom: Path) -> None:
    log = EventLog(eventloom / "default.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        thread="default",
        payload={
            "query": "support escalation",
            "purpose": {
                "profile": "support",
                "role": "support-agent",
                "evidence_policy": "customer_thread_and_current_status_required",
            },
            "retention": {
                "purpose_policy": {
                    "suppressed_count": 2,
                    "suppressed_reasons": {"stale_status": 1},
                    "suppressed_examples": [{"id": "old-case", "reason": "stale_status"}],
                }
            },
            "diagnostics": {
                "evidence_policy": {
                    "status": "missing",
                    "missing": ["current_status"],
                    "suggested_queries": ["refresh customer escalation status"],
                }
            },
            "quality": {
                "required_action": {
                    "type": "memory_checkout",
                    "query": "refresh customer escalation status",
                    "missing_slots": ["current_status"],
                }
            },
            "warnings": ["Checkout has stale support evidence."],
        },
    )
    log.append(
        "memory.reinforced",
        actor="assistant",
        thread="default",
        payload={
            "purpose": {"profile": "support"},
            "citation": "event:default:1",
            "outcome": "used",
        },
    )
    log.append(
        "memory.feedback",
        actor="assistant",
        thread="default",
        payload={
            "purpose": {"profile": "support"},
            "citation": "event:default:1",
            "feedback": "rejected",
        },
    )
    log.append(
        "memory.evidence.excluded",
        actor="assistant",
        thread="default",
        payload={
            "purpose": {"profile": "support"},
            "citation": "event:default:1",
        },
    )


def test_purpose_status_summarizes_checkout_feedback_and_coordinate_without_graph(tmp_path: Path) -> None:
    eventloom = tmp_path / ".eventloom"
    _write_purpose_fixture(eventloom)
    manager = CoordinationManager(eventloom_path=eventloom)
    manager.start_mission("ship-1", objective="Ship coordinated release")
    manager.create_worker("ship-1", "worker-a")
    manager.assign("ship-1", "worker-a", "Audit docs")
    accepted = manager.report_finding(
        "ship-1",
        "worker-a",
        summary="Docs gate passed.",
        actor="worker-a",
        evidence=[{"kind": "test", "reference": "pytest tests/test_docs.py -q"}],
        finding_id="finding-accepted",
    )
    manager.review_finding("ship-1", accepted.finding_id, status="accepted")
    manager.promote_finding("ship-1", accepted.finding_id)
    manager.report_finding(
        "ship-1",
        "worker-a",
        summary="Pending worker diagnostic.",
        actor="worker-a",
        evidence=[],
        finding_id="finding-pending",
    )

    status = build_purpose_status(eventloom)

    assert status["active_profile"] == "support"
    assert status["evidence_policy_status"]["status"] == "needs_refresh"
    assert status["suppression"]["count"] == 2
    assert status["suppression"]["reasons"] == {"stale_status": 2}
    assert status["refresh_suggestions"][0]["query"] == "refresh customer escalation status"
    assert status["consequence_history"]["positive_count"] == 1
    assert status["consequence_history"]["negative_count"] == 2
    assert status["consequence_history"]["targets"][0]["suppression_candidate"] is True
    mission = status["coordinate"]["missions"][0]
    assert mission["mission_id"] == "ship-1"
    assert mission["accepted_count"] == 1
    assert mission["pending_count"] == 1
    assert mission["approval_packet_count"] == 1
    assert "active profile: support" in format_purpose_status(status)
    assert "accepted=1 pending=1" in format_purpose_status(status)


def test_purpose_lanes_and_feedback_filters_are_replay_only(tmp_path: Path) -> None:
    eventloom = tmp_path / ".eventloom"
    _write_purpose_fixture(eventloom)

    lanes = build_purpose_lanes(eventloom)
    feedback = build_purpose_feedback(eventloom, profile="support", outcome="negative")

    assert lanes["lanes"][0]["profile"] == "support"
    assert lanes["lanes"][0]["checkout_count"] == 1
    assert lanes["lanes"][0]["evidence_policy_fail_count"] == 1
    assert lanes["lanes"][0]["positive_feedback_count"] == 1
    assert lanes["lanes"][0]["negative_feedback_count"] == 2
    assert feedback["targets"][0]["target"] == "citation:event:default:1"
    assert feedback["targets"][0]["negative_count"] == 2
    assert "support: checkouts=1" in format_purpose_lanes(lanes)
    assert "suppression-candidate" in format_purpose_feedback(feedback)


def test_purpose_control_handles_empty_and_nested_payloads(tmp_path: Path) -> None:
    eventloom = tmp_path / ".eventloom"
    first = EventLog(eventloom / "first.jsonl")
    first.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        thread="agent-a",
        payload={
            "payload": {
                "purpose": "review",
                "diagnostics": {
                    "evidence_plan": {"suggested_queries": ["find verification evidence"]},
                    "purpose_policy": {
                        "suppressed_count": 1,
                        "suppressed_reasons": {"low_trust_inference": 1},
                    },
                },
                "quality": {"required_action": None},
            }
        },
    )
    first.append(
        "memory.feedback",
        actor="assistant",
        thread="agent-a",
        payload={"purpose": "review", "source_event_hash": "abc123", "feedback": "Accepted"},
    )
    second = EventLog(eventloom / "second.jsonl")
    second.append(
        "memory.feedback",
        actor="assistant",
        thread="agent-b",
        payload={"purpose": "coding", "entity_name": "coverage", "entity_type": "test", "feedback": "failed"},
    )
    second.append(
        "memory.feedback",
        actor="assistant",
        thread="agent-b",
        payload={"purpose": "coding", "target_id": "lane-9", "feedback": "irrelevant"},
    )
    second.append(
        "memory.feedback",
        actor="assistant",
        thread="agent-b",
        payload={"purpose": "coding", "feedback": "unknown"},
    )

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert format_purpose_lanes(build_purpose_lanes(empty_dir)) == "Purpose lanes:\n  none"
    assert format_purpose_feedback(build_purpose_feedback(empty_dir)) == "Purpose feedback:\n  none"

    lanes = build_purpose_lanes(eventloom, session_id="agent-a")
    assert lanes["lanes"][0]["profile"] == "review"
    assert lanes["lanes"][0]["evidence_policy_pass_count"] == 1
    assert lanes["lanes"][0]["suppressed_reasons"] == {"low_trust_inference": 1}
    assert lanes["lanes"][0]["positive_feedback_count"] == 1
    assert lanes["lanes"][0]["negative_feedback_count"] == 0
    assert lanes["lanes"][0]["refresh_suggestions"][0]["query"] == "find verification evidence"

    feedback = build_purpose_feedback(eventloom, session_id="agent-b", outcome="negative")
    assert {target["target"] for target in feedback["targets"]} == {
        "entity:test:coverage",
        "target:lane-9",
    }
    assert all(target["profile"] == "coding" for target in feedback["targets"])


def test_unsupported_coordination_runner_writes_failure_stub(tmp_path: Path, capsys) -> None:
    output = tmp_path / "quarq" / "result.json"

    code = unsupported_runner_main([
        "--adapter",
        "quarq",
        "--workload",
        str(tmp_path / "workload.json"),
        "--output",
        str(output),
    ])

    assert code == 78
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["name"] == "quarq"
    assert payload["status"] == "unsupported"
    assert payload["adapter_contract"] == "coordinationbench-v1"
    assert "same-harness runner adapter" in payload["reason"]
    assert "same-harness runner adapter" in capsys.readouterr().err


def test_purpose_control_edge_cases_are_replay_only(tmp_path: Path) -> None:
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "edge.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        thread="agent-a",
        payload={
            "query": "fallback purpose",
            "diagnostics": {"purpose": "coding"},
            "retention": {
                "purpose_policy": {
                    "suppressed_count": True,
                    "suppressed_reasons": {"bool-does-not-count": True, "string-does-not-count": "1"},
                }
            },
            "guidance": {"recommended_next_call": {"tool": "memory_checkout", "query": "refresh code"}},
            "quality": {"required_action": None},
        },
    )
    log.append(
        "memory.reinforced",
        actor="assistant",
        thread="agent-a",
        payload={"purpose": "coding", "source_group": "checkout-sources"},
    )
    log.append(
        "memory.evidence.excluded",
        actor="assistant",
        thread="agent-a",
        payload={"purpose": "coding", "target": "bad-source"},
    )
    log.append(
        "memory.feedback",
        actor="assistant",
        thread="agent-a",
        payload={"purpose": "", "feedback": "used", "target": "ignored-empty-purpose"},
    )
    log.append(
        "memory.feedback",
        actor="assistant",
        thread="agent-a",
        payload={"purpose": "coding", "feedback": "used"},
    )

    lanes = build_purpose_lanes(eventloom)
    feedback = build_purpose_feedback(eventloom)

    assert lanes["lanes"][0]["profile"] == "coding"
    assert lanes["lanes"][0]["suppressed_count"] == 0
    assert lanes["lanes"][0]["suppressed_reasons"] == {
        "bool-does-not-count": 0,
        "string-does-not-count": 0,
    }
    assert lanes["lanes"][0]["refresh_suggestions"][0]["query"] == "refresh code"
    assert {target["target"] for target in feedback["targets"]} == {
        "source_group:checkout-sources",
        "target:bad-source",
    }
    assert format_purpose_lanes({"lanes": ["bad-row"]}) == "Purpose lanes:"
    assert format_purpose_feedback({"targets": ["bad-row"]}) == "Purpose feedback:"


def test_purpose_control_bounds_unique_suggestions() -> None:
    lanes = [
        purpose_control.PurposeLane(
            profile=f"profile-{index}",
            role="role",
            evidence_policy="policy",
            refresh_suggestions=[{"query": f"q-{index}"}],
        )
        for index in range(12)
    ]

    suggestions = purpose_control._bounded_suggestions(lanes)

    assert len(suggestions) == 10
    assert suggestions[-1] == {"query": "q-9"}


def test_purpose_control_handles_unavailable_or_stale_coordinate_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    eventloom = tmp_path / ".eventloom"
    log = EventLog(eventloom / "coordinate.jsonl")
    log.append(
        "coordination.mission.started",
        actor="coordinator",
        thread="missing-mission",
        payload={"mission_id": "missing-mission"},
    )

    original = sys.modules.get("zaxy.coordination")
    monkeypatch.setitem(sys.modules, "zaxy.coordination", None)
    unavailable = build_purpose_status(eventloom)
    assert unavailable["coordinate"] == {"available": False, "missions": []}
    if original is not None:
        monkeypatch.setitem(sys.modules, "zaxy.coordination", original)
    else:
        monkeypatch.delitem(sys.modules, "zaxy.coordination", raising=False)

    class EmptyManager:
        def __init__(self, *, eventloom_path: Path) -> None:
            self.eventloom_path = eventloom_path

        def brief(self, mission_id: str):
            raise ValueError(mission_id)

        def approval_packet(self, mission_id: str):
            raise AssertionError("approval packet should not be requested")

    monkeypatch.setattr("zaxy.coordination.CoordinationManager", EmptyManager)
    stale = build_purpose_status(eventloom)
    assert stale["coordinate"] == {"available": True, "missions": []}
    assert purpose_control._feedback_outcome("memory.feedback", {}) is None
