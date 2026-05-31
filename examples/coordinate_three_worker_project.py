"""Three-worker Zaxy Coordinate example.

Run with:

    python examples/coordinate_three_worker_project.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from zaxy.coordination import CoordinationManager


def run_demo(eventloom_path: str | Path) -> dict[str, Any]:
    """Run a three-worker mission and return a compact verification summary."""
    manager = CoordinationManager(eventloom_path=Path(eventloom_path))
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    for worker_id, assignment in [
        ("auth-api", "Trace API auth failures"),
        ("auth-ui", "Check browser refresh behavior"),
        ("auth-tests", "Verify test evidence"),
    ]:
        manager.create_worker("auth-main", worker_id, actor="lead")
        manager.assign("auth-main", worker_id, assignment, actor="lead")

    api_finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API failures trace to expired JWKS cache handling.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    ui_finding = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="API failures trace to missing browser refresh state.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    stale_finding = manager.report_finding(
        "auth-main",
        "auth-tests",
        summary="Legacy feature flag explanation is stale.",
        actor="auth-test-agent",
        evidence=[
            {
                "kind": "source",
                "reference": "docs/auth-flags.md",
                "stale": True,
                "superseded_by": "decision:jwks-cache",
            }
        ],
        claim_key="auth.failure.cause",
        claim_value="legacy-feature-flag",
    )
    assert api_finding.finding_id is not None
    assert ui_finding.finding_id is not None
    assert stale_finding.finding_id is not None
    approval_packet = manager.approval_packet("auth-main")
    approval_result = manager.apply_approval_decisions(
        "auth-main",
        [
            {
                "finding_id": api_finding.finding_id,
                "status": "accepted",
                "rationale": "Command-backed and matches observed API behavior.",
                "promote": True,
            },
            {
                "finding_id": ui_finding.finding_id,
                "status": "conflicted",
                "rationale": "Conflicts with the accepted API evidence; needs browser trace follow-up.",
            },
            {
                "finding_id": stale_finding.finding_id,
                "status": "deferred",
                "rationale": "Superseded source should be refreshed before any promotion.",
            },
        ],
        actor="lead",
    )

    brief = manager.brief("auth-main")
    checkout = manager.checkout("auth-main")
    handoff = manager.create_handoff(
        "auth-main",
        summary="Accepted API auth cause is ready for implementation planning.",
        next_steps=["Patch JWKS cache refresh handling"],
        risks=["UI refresh path still needs review"],
        actor="lead",
    )
    inspection = manager.inspect_mission("auth-main").to_dict()
    audit_report = manager.audit_report("auth-main")
    approval_next_actions = sorted(
        {
            action["code"]
            for finding in approval_packet.findings
            for action in finding.next_actions
        }
    )
    return {
        "mission_id": brief.mission_id,
        "worker_count": len(brief.workers),
        "accepted_count": len(brief.accepted_findings),
        "pending_count": len(brief.pending_findings),
        "conflict_count": len(brief.conflicts),
        "excluded_pending_count": checkout.excluded_pending_count,
        "handoff_id": handoff.handoff_id,
        "checkout_prompt": checkout.prompt,
        "approval_packet_id": approval_packet.packet_id,
        "approval_findings_count": len(approval_packet.findings),
        "approval_next_actions": approval_next_actions,
        "approval_reviewed_count": approval_result.reviewed_count,
        "approval_promoted_count": approval_result.promoted_count,
        "inspection_sections": [
            key
            for key in [
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
            if key in inspection
        ],
        "audit_event_count": audit_report.summary["event_count"],
        "audit_has_event_hashes": all(len(event["event_hash"]) == 64 for event in audit_report.events),
    }


def main() -> None:
    """Run the example in a temporary Eventloom directory and print JSON."""
    with tempfile.TemporaryDirectory(prefix="zaxy-coordinate-example-") as tmp:
        print(json.dumps(run_demo(Path(tmp) / ".eventloom"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
