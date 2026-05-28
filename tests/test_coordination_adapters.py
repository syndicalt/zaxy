"""Tests for dependency-light coordination adapters."""

from __future__ import annotations

import pytest

from zaxy.adapters.coordination import CoordinationAdapter


def test_coordination_adapter_reports_and_promotes_worker_finding(tmp_path) -> None:
    """The shared adapter should expose the Coordinate lifecycle as JSON payloads."""
    adapter = CoordinationAdapter(eventloom_path=tmp_path / ".eventloom", actor="lead")

    mission = adapter.start_mission("auth-main", objective="Ship auth refactor")
    worker = adapter.create_worker("auth-main", "auth-api")
    assignment = adapter.assign("auth-main", "auth-api", "Trace API auth failures")
    finding = adapter.report_finding(
        "auth-main",
        "auth-api",
        summary="API failures trace to expired JWKS cache handling",
        evidence=[{"kind": "source", "reference": "src/auth.py:12"}],
        confidence=0.91,
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    decision = adapter.apply_approval(
        "auth-main",
        [{"finding_id": finding["finding_id"], "status": "accepted", "promote": True}],
    )
    checkout = adapter.checkout("auth-main")
    handoff = adapter.handoff("auth-main", summary="Auth refactor is ready")

    assert mission["event_type"] == "coordination.mission.created"
    assert worker["worker_id"] == "auth-api"
    assert assignment["summary"] == "Trace API auth failures"
    assert finding["finding_id"]
    assert finding["event_hash"]
    assert decision["promoted_count"] == 1
    assert checkout["accepted_findings"][0]["claim_value"] == "expired-jwks-cache"
    assert checkout["excluded_pending_count"] == 0
    assert handoff["event_type"] == "coordination.handoff.created"


def test_coordination_adapter_rejects_empty_evidence_items(tmp_path) -> None:
    """Adapter inputs should stay structured and not silently coerce bad evidence."""
    adapter = CoordinationAdapter(eventloom_path=tmp_path / ".eventloom")
    adapter.start_mission("auth-main", objective="Ship auth refactor")
    adapter.create_worker("auth-main", "auth-api")

    with pytest.raises(ValueError, match="evidence"):
        adapter.report_finding(
            "auth-main",
            "auth-api",
            summary="Finding",
            evidence=[{}],
        )
