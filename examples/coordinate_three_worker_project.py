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
    manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="API failures trace to missing browser refresh state.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    manager.report_finding(
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
    manager.review_finding(
        "auth-main",
        api_finding.finding_id,
        status="accepted",
        actor="lead",
        rationale="Command-backed and matches observed API behavior.",
    )
    manager.promote_finding("auth-main", api_finding.finding_id, actor="lead")

    brief = manager.brief("auth-main")
    checkout = manager.checkout("auth-main")
    handoff = manager.create_handoff(
        "auth-main",
        summary="Accepted API auth cause is ready for implementation planning.",
        next_steps=["Patch JWKS cache refresh handling"],
        risks=["UI refresh path still needs review"],
        actor="lead",
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
    }


def main() -> None:
    """Run the example in a temporary Eventloom directory and print JSON."""
    with tempfile.TemporaryDirectory(prefix="zaxy-coordinate-example-") as tmp:
        print(json.dumps(run_demo(Path(tmp) / ".eventloom"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
