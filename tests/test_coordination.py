"""Tests for high-level multi-agent coordination state."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from zaxy.coordination import ConflictState, CoordinationManager, LocalSemanticConflictDetector
from zaxy.coordination_git import build_test_result_evidence, capture_git_metadata

_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "coordinate_three_worker_project.py"
_EXAMPLE_SPEC = importlib.util.spec_from_file_location("coordinate_three_worker_project", _EXAMPLE_PATH)
assert _EXAMPLE_SPEC is not None and _EXAMPLE_SPEC.loader is not None
_EXAMPLE_MODULE = importlib.util.module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_EXAMPLE_MODULE)
run_demo = _EXAMPLE_MODULE.run_demo



def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_coordination_manager_promotes_accepted_findings_to_parent_session(tmp_path: Path) -> None:
    """Accepted worker findings should become parent mission state without erasing worker history."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")

    mission = manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    worker = manager.create_worker("auth-main", "auth-api", actor="lead")
    assignment = manager.assign(
        "auth-main",
        "auth-api",
        "Trace API auth failures",
        actor="lead",
    )
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API failures trace to expired JWKS cache handling.",
        actor="auth-api-agent",
        evidence=[
            {
                "kind": "command",
                "reference": "pytest tests/test_auth.py -q",
                "summary": "reproduced failing token refresh test",
            }
        ],
        confidence=0.91,
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )

    review = manager.review_finding(
        "auth-main",
        finding.finding_id,
        status="accepted",
        actor="lead",
        rationale="Evidence is command-backed and scoped to API auth.",
    )
    promotion = manager.promote_finding("auth-main", finding.finding_id, actor="lead")

    parent_events = manager.session_manager.replay("auth-main").events
    worker_events = manager.session_manager.replay("auth-api").events

    assert mission.event.type == "coordination.mission.created"
    assert worker.event.thread == "auth-main"
    assert assignment.event.thread == "auth-main"
    assert finding.event.thread == "auth-api"
    assert review.event.thread == "auth-main"
    assert promotion.event.thread == "auth-main"
    assert [event.type for event in parent_events] == [
        "coordination.mission.created",
        "coordination.worker.created",
        "coordination.assignment.created",
        "coordination.finding.reviewed",
        "coordination.finding.promoted",
    ]
    assert [event.type for event in worker_events] == ["coordination.finding.reported"]
    assert promotion.summary == finding.summary
    assert promotion.evidence[0]["reference"] == "pytest tests/test_auth.py -q"


def test_coordination_proof_packet_scopes_and_labels_authority(tmp_path: Path) -> None:
    """Proof packets should not over-claim accepted support or hide diagnostic rows."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("release-rc1", objective="Ship release", actor="lead")
    for worker_id in ("auth-api", "auth-docs", "auth-ui", "auth-stale", "auth-reject", "auth-defer"):
        manager.create_worker("release-rc1", worker_id, actor="lead")
    accepted_used = manager.report_finding(
        "release-rc1",
        "auth-api",
        summary="Expired JWKS cache is the accepted auth failure cause.",
        actor="auth-api-agent",
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    accepted_other = manager.report_finding(
        "release-rc1",
        "auth-docs",
        summary="Docs need release note updates.",
        actor="auth-docs-agent",
        claim_key="release.docs",
        claim_value="needs-update",
    )
    pending = manager.report_finding(
        "release-rc1",
        "auth-ui",
        summary="UI retry claim is still pending.",
        actor="auth-ui-agent",
    )
    stale = manager.report_finding(
        "release-rc1",
        "auth-stale",
        summary="Old flag state is superseded.",
        actor="auth-stale-agent",
        evidence=[{"kind": "file", "reference": "flags.json", "stale": True}],
    )
    rejected = manager.report_finding(
        "release-rc1",
        "auth-reject",
        summary="Rejected browser refresh cause.",
        actor="auth-reject-agent",
    )
    deferred = manager.report_finding(
        "release-rc1",
        "auth-defer",
        summary="Deferred rollout metric claim.",
        actor="auth-defer-agent",
    )
    for finding in (accepted_used, accepted_other):
        manager.review_finding("release-rc1", finding.finding_id, status="accepted", actor="lead")
        manager.promote_finding("release-rc1", finding.finding_id, actor="lead")
    manager.review_finding("release-rc1", rejected.finding_id, status="rejected", actor="lead")
    manager.review_finding("release-rc1", deferred.finding_id, status="deferred", actor="lead")
    artifact = {
        "artifact_id": "sha256:proof",
        "query": "Compose accepted release findings.",
        "operations": [
            {
                "name": "coordinate_parent_state_synthesis",
                "answer_key": "coordinate_handoff_answer",
                "support_source_ids": [accepted_used.finding_id],
            }
        ],
        "result": {
            "answer_key": "coordinate_handoff_answer",
            "answer": "Accepted cause: expired JWKS cache.",
            "confidence": 0.9,
        },
        "answer_candidates": [
            {
                "answer": "Accepted cause: expired JWKS cache.",
                "support_source_ids": [accepted_used.finding_id],
            }
        ],
        "ledger_rows": [
            {"source_group": accepted_used.finding_id, "include_reason": "accepted_parent_state"},
            {"source_group": accepted_other.finding_id, "include_reason": "unused_accepted_state"},
            {"source_group": pending.finding_id, "include_reason": "diagnostic_pending"},
            {"source_group": stale.finding_id, "include_reason": "diagnostic_stale"},
            {"source_group": rejected.finding_id, "include_reason": "diagnostic_rejected"},
            {"source_group": deferred.finding_id, "include_reason": "diagnostic_deferred"},
        ],
    }

    packet = manager.proof_packet("release-rc1", artifact).to_dict()

    assert packet["accepted_finding_ids"] == [accepted_used.finding_id]
    assert {row["source_group"]: row["status"] for row in packet["non_authoritative_rows"]} == {
        accepted_other.finding_id: "accepted_not_used",
        pending.finding_id: "pending",
        stale.finding_id: "stale",
        rejected.finding_id: "rejected",
        deferred.finding_id: "deferred",
    }


def test_coordination_proof_packet_links_handoff_event_ref(tmp_path: Path) -> None:
    """Handoff-scoped proof packets should bind to a concrete handoff event."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("release-rc1", objective="Ship release", actor="lead")
    manager.create_worker("release-rc1", "auth-api", actor="lead")
    finding = manager.report_finding(
        "release-rc1",
        "auth-api",
        summary="Expired JWKS cache is accepted.",
        actor="auth-api-agent",
    )
    manager.review_finding("release-rc1", finding.finding_id, status="accepted", actor="lead")
    manager.promote_finding("release-rc1", finding.finding_id, actor="lead")
    handoff = manager.create_handoff(
        "release-rc1",
        summary="Release handoff ready.",
        actor="lead",
    )
    artifact = {
        "artifact_id": "sha256:proof",
        "query": "Compose accepted handoff.",
        "answer_candidates": [{"answer": "Accepted cause", "support_source_ids": [finding.finding_id]}],
        "ledger_rows": [{"source_group": finding.finding_id, "include_reason": "accepted_parent_state"}],
    }

    packet = manager.proof_packet(
        "release-rc1",
        artifact,
        decision_scope="handoff",
        handoff_id=handoff.handoff_id,
    ).to_dict()

    assert packet["handoff_event_ref"] == {
        "handoff_id": handoff.handoff_id,
        "seq": handoff.event.seq,
        "hash": handoff.event.hash,
    }


def test_coordination_proof_packet_requires_known_handoff_for_handoff_scope(tmp_path: Path) -> None:
    """Handoff-scoped proof packets should not silently float without provenance."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("release-rc1", objective="Ship release", actor="lead")
    artifact = {"artifact_id": "sha256:proof", "query": "Compose accepted handoff.", "ledger_rows": []}

    with pytest.raises(ValueError, match="handoff_id is required"):
        manager.proof_packet("release-rc1", artifact, decision_scope="handoff")

    with pytest.raises(ValueError, match="Unknown handoff_id"):
        manager.proof_packet(
            "release-rc1",
            artifact,
            decision_scope="handoff",
            handoff_id="release-rc1:handoff:missing",
        )


def test_coordination_proof_packet_surfaces_mixed_identity_non_authoritative_rows(tmp_path: Path) -> None:
    """A pending fact under an accepted source group must not be hidden as authoritative."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("release-rc1", objective="Ship release", actor="lead")
    manager.create_worker("release-rc1", "auth-api", actor="lead")
    manager.create_worker("release-rc1", "auth-ui", actor="lead")
    accepted = manager.report_finding(
        "release-rc1",
        "auth-api",
        summary="Expired JWKS cache is accepted.",
        actor="auth-api-agent",
    )
    pending = manager.report_finding(
        "release-rc1",
        "auth-ui",
        summary="Browser refresh cause is still pending.",
        actor="auth-ui-agent",
    )
    manager.review_finding("release-rc1", accepted.finding_id, status="accepted", actor="lead")
    manager.promote_finding("release-rc1", accepted.finding_id, actor="lead")
    artifact = {
        "artifact_id": "sha256:proof",
        "query": "Compose accepted release findings.",
        "answer_candidates": [{"answer": "Accepted cause", "support_source_ids": [accepted.finding_id]}],
        "ledger_rows": [
            {
                "source_group": accepted.finding_id,
                "fact_id": pending.finding_id,
                "include_reason": "diagnostic_pending_under_accepted_group",
            }
        ],
    }

    packet = manager.proof_packet("release-rc1", artifact).to_dict()

    assert packet["non_authoritative_rows"] == [
        {
            "source_group": accepted.finding_id,
            "status": "pending",
            "include_reason": "diagnostic_pending_under_accepted_group",
            "exclude_reason": None,
            "fact_id": pending.finding_id,
        }
    ]


def test_capture_git_metadata_reports_branch_worktree_changed_files_and_diff(tmp_path: Path) -> None:
    """Git metadata capture should be read-only and structured for finding evidence."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Zaxy Test")
    (repo / "auth.py").write_text("TOKEN_TTL = 10\n", encoding="utf-8")
    _git(repo, "add", "auth.py")
    _git(repo, "commit", "-m", "initial")
    (repo / "auth.py").write_text("TOKEN_TTL = 20\n", encoding="utf-8")
    (repo / "new_auth.py").write_text("ENABLED = True\n", encoding="utf-8")

    metadata = capture_git_metadata(repo)

    assert metadata["kind"] == "git"
    assert metadata["repo_root"] == str(repo.resolve())
    assert metadata["worktree"] == str(repo.resolve())
    assert metadata["branch"] in {"master", "main"}
    assert metadata["detached"] is False
    assert metadata["dirty"] is True
    assert metadata["head"]
    assert {"path": "auth.py", "status": "M", "operation": "modified"} in metadata["changed_files"]
    assert {"path": "new_auth.py", "status": "??", "operation": "untracked"} in metadata["changed_files"]
    assert metadata["worktrees"][0]["path"] == str(repo.resolve())
    assert "auth.py" in metadata["diff_summary"]


def test_capture_git_metadata_uses_read_only_git_invocations(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(cmd, *, cwd, check, capture_output, text, env):  # type: ignore[no-untyped-def]
        calls.append((list(cmd), dict(env)))
        args = tuple(cmd[2:])
        stdout = {
            ("rev-parse", "--show-toplevel"): str(tmp_path),
            ("rev-parse", "--abbrev-ref", "HEAD"): "feature/auth",
            ("rev-parse", "HEAD"): "a" * 40,
            ("status", "--porcelain=v1", "-z", "--untracked-files=all"): "",
            ("diff", "--stat", "--summary"): "",
            ("worktree", "list", "--porcelain"): f"worktree {tmp_path}\nHEAD {'a' * 40}\nbranch refs/heads/feature/auth\n",
        }.get(args)
        if stdout is None:
            raise AssertionError(f"unexpected git command: {cmd}")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    metadata = capture_git_metadata(tmp_path)

    assert metadata["branch"] == "feature/auth"
    assert calls
    assert all(call[0][0:2] == ["git", "--no-optional-locks"] for call in calls)
    assert all(call[1]["GIT_OPTIONAL_LOCKS"] == "0" for call in calls)


def test_coordination_ledger_counts_test_result_evidence(tmp_path: Path) -> None:
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Auth tests passed with retry enabled.",
        actor="auth-api-agent",
        evidence=[build_test_result_evidence("pytest tests/test_auth.py -q", status="passed")],
    )

    ledger = manager.performance_ledger("auth-main")

    assert ledger.workers[0].test_backed_findings == 1


def test_three_worker_coordinate_example_proves_accepted_checkout_is_clean(tmp_path: Path) -> None:
    result = run_demo(tmp_path / ".eventloom")

    assert result["mission_id"] == "auth-main"
    assert result["worker_count"] == 3
    assert result["accepted_count"] == 1
    assert result["pending_count"] == 0
    assert result["conflict_count"] >= 1
    assert result["excluded_pending_count"] == 0
    assert "expired JWKS cache" in result["checkout_prompt"]
    assert "missing browser refresh" not in result["checkout_prompt"]


def test_coordination_brief_separates_accepted_pending_and_conflicted_findings(tmp_path: Path) -> None:
    """The coordinator brief should expose governed state, not a shared scratchpad."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    api_finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="JWT failures come from expired JWKS cache.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    ui_finding = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="JWT failures come from missing browser refresh.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    manager.review_finding("auth-main", api_finding.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", api_finding.finding_id, actor="lead")

    brief = manager.brief("auth-main")

    assert brief.mission_id == "auth-main"
    assert brief.objective == "Ship auth refactor"
    assert [worker.worker_id for worker in brief.workers] == ["auth-api", "auth-ui"]
    assert [finding.finding_id for finding in brief.accepted_findings] == [api_finding.finding_id]
    assert [finding.finding_id for finding in brief.pending_findings] == [ui_finding.finding_id]
    assert brief.conflicts
    assert {finding.finding_id for finding in brief.conflicts[0].findings} == {
        api_finding.finding_id,
        ui_finding.finding_id,
    }
    payload = brief.to_dict()
    assert payload["accepted_findings"][0]["status"] == "accepted"
    assert payload["pending_findings"][0]["status"] == "pending"
    assert payload["conflicts"][0]["claim_key"] == "auth.failure.cause"


def test_coordination_checkout_defaults_to_accepted_parent_state_only(tmp_path: Path) -> None:
    """Accepted checkout should exclude worker scratch findings unless diagnostics are requested."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
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
        summary="UI refresh handling is missing retry state.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    manager.review_finding("auth-main", api_finding.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", api_finding.finding_id, actor="lead")

    checkout = manager.checkout("auth-main")

    assert [finding.finding_id for finding in checkout.accepted_findings] == [api_finding.finding_id]
    assert checkout.pending_findings == []
    assert checkout.conflicts == []
    assert checkout.excluded_pending_count == 1
    assert checkout.excluded_conflict_count == 1
    payload = checkout.to_dict()
    assert payload["accepted_findings"][0]["summary"] == "API failures trace to expired JWKS cache handling."
    assert payload["pending_findings"] == []
    assert payload["conflicts"] == []
    assert ui_finding.finding_id not in payload["prompt"]
    assert "API failures trace to expired JWKS cache handling." in payload["prompt"]
    assert payload["purpose"]["profile"] == "coordinate"
    assert payload["purpose"]["evidence_policy"] == "accepted_parent_state_with_citations_required"
    assert "Purpose profile: coordinate" in payload["prompt"]
    assert "Authority: accepted parent state only" in payload["prompt"]


def test_coordination_checkout_can_include_pending_and_conflict_diagnostics(tmp_path: Path) -> None:
    """Operators should be able to request non-authoritative diagnostics explicitly."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    accepted = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    pending = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="Missing browser refresh may cause API failures.",
        actor="auth-ui-agent",
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    manager.review_finding("auth-main", accepted.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", accepted.finding_id, actor="lead")

    checkout = manager.checkout("auth-main", include_diagnostics=True)

    assert [finding.finding_id for finding in checkout.accepted_findings] == [accepted.finding_id]
    assert [finding.finding_id for finding in checkout.pending_findings] == [pending.finding_id]
    assert checkout.conflicts
    assert checkout.excluded_pending_count == 0
    assert checkout.excluded_conflict_count == 0
    assert "Diagnostics" in checkout.prompt
    assert pending.finding_id in checkout.prompt


def test_coordination_performance_ledger_scores_worker_outcomes(tmp_path: Path) -> None:
    """The agent performance ledger should summarize useful worker output from replay."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    accepted = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API failures trace to expired JWKS cache handling.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    rejected = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API failures trace to an unrelated CSS regression.",
        actor="auth-api-agent",
        evidence=[],
        claim_key="auth.failure.cause",
        claim_value="css-regression",
    )
    duplicate = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI also saw expired JWKS cache handling.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    manager.review_finding("auth-main", accepted.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", accepted.finding_id, actor="lead")
    manager.review_finding("auth-main", rejected.finding_id, status="rejected", actor="lead")

    ledger = manager.performance_ledger("auth-main")

    assert ledger.mission_id == "auth-main"
    assert ledger.worker_count == 2
    assert ledger.total_findings == 3
    api = ledger.worker("auth-api")
    ui = ledger.worker("auth-ui")
    assert api.total_findings == 2
    assert api.accepted_findings == 1
    assert api.promoted_findings == 1
    assert api.rejected_findings == 1
    assert api.missing_evidence_count == 1
    assert api.test_backed_findings == 1
    assert api.acceptance_rate == 0.5
    assert api.missing_evidence_rate == 0.5
    assert api.test_backed_rate == 0.5
    assert ui.total_findings == 1
    assert ui.duplicate_finding_count == 1
    assert ui.duplicate_finding_rate == 1.0
    payload = ledger.to_dict()
    assert payload["workers"][0]["worker_id"] == "auth-api"
    assert payload["workers"][1]["duplicate_finding_count"] == 1
    assert duplicate.finding_id in payload["workers"][1]["duplicate_finding_ids"]


def test_coordination_handoff_records_final_parent_session_event(tmp_path: Path) -> None:
    """Final handoff should be replayable from the parent mission Eventloom session."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")

    result = manager.create_handoff(
        "auth-main",
        summary="Auth mission complete.",
        next_steps=["Release branch", "Monitor JWKS refresh"],
        risks=["Token cache metrics are sparse"],
        actor="lead",
    )

    assert result.event.type == "coordination.handoff.created"
    assert result.event.thread == "auth-main"
    assert result.event.payload["handoff_id"] == result.handoff_id
    assert result.summary == "Auth mission complete."
    assert result.evidence == [
        {"kind": "next_step", "reference": "Release branch"},
        {"kind": "next_step", "reference": "Monitor JWKS refresh"},
        {"kind": "risk", "reference": "Token cache metrics are sparse"},
    ]
    events = manager.session_manager.replay("auth-main").events
    assert events[-1].payload == {
        "mission_id": "auth-main",
        "handoff_id": result.handoff_id,
        "summary": "Auth mission complete.",
        "next_steps": ["Release branch", "Monitor JWKS refresh"],
        "risks": ["Token cache metrics are sparse"],
        "status": "created",
    }


def test_coordination_approval_packet_exports_pending_and_conflicted_findings(tmp_path: Path) -> None:
    """Remote reviewers should get a portable packet of findings needing decisions."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    accepted = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    pending = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="Missing browser refresh may cause API failures.",
        actor="auth-ui-agent",
        evidence=[],
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    manager.review_finding("auth-main", accepted.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", accepted.finding_id, actor="lead")

    packet = manager.approval_packet("auth-main")

    assert packet.mission_id == "auth-main"
    assert packet.objective == "Ship auth refactor"
    assert packet.pending_count == 1
    assert packet.conflict_count == 1
    assert [finding.finding_id for finding in packet.findings] == [pending.finding_id]
    assert packet.findings[0].requires_evidence is True
    assert packet.findings[0].conflict_keys == ["auth.failure.cause"]
    payload = packet.to_dict()
    assert payload["packet_id"].startswith("auth-main:approval:")
    assert payload["findings"][0]["allowed_statuses"] == ["accepted", "conflicted", "deferred", "rejected"]
    assert payload["decisions_template"] == [
        {
            "finding_id": pending.finding_id,
            "status": "deferred",
            "rationale": "",
            "promote": False,
        }
    ]


def test_coordination_approval_packet_recommends_next_actions(tmp_path: Path) -> None:
    """Approval packets should make pending, conflicted, stale, and evidence-poor actions explicit."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    pending = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Command-backed API finding is ready for review.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    needs_evidence = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="Browser refresh claim still needs a trace.",
        actor="auth-ui-agent",
        evidence=[],
    )
    stale = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="Stale config snapshot should not be promoted.",
        actor="auth-ui-agent",
        evidence=[
            {
                "kind": "file",
                "reference": "src/auth/config.py",
                "status": "superseded",
                "superseded_by": "decision:jwks-cache",
            }
        ],
    )
    accepted_counterclaim = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Accepted API finding reports the cache failure cause.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    conflict = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI worker reports a different failure cause.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
        claim_key="auth.failure.cause",
        claim_value="missing-browser-refresh",
    )
    manager.review_finding("auth-main", accepted_counterclaim.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", accepted_counterclaim.finding_id, actor="lead")

    packet = manager.approval_packet("auth-main").to_dict()
    findings = {finding["finding_id"]: finding for finding in packet["findings"]}

    assert [action["code"] for action in findings[pending.finding_id]["next_actions"]] == [
        "review_finding"
    ]
    assert findings[needs_evidence.finding_id]["next_actions"][0] == {
        "code": "add_evidence",
        "label": "Attach evidence before accepting or promoting this finding.",
        "recommended_status": "deferred",
    }
    assert findings[stale.finding_id]["next_actions"][0] == {
        "code": "refresh_stale_evidence",
        "label": "Refresh superseded or stale evidence before promotion.",
        "recommended_status": "deferred",
        "superseded_by": "decision:jwks-cache",
    }
    assert findings[conflict.finding_id]["next_actions"][0] == {
        "code": "resolve_conflict",
        "label": "Resolve conflicting claim keys before accepting or promoting this finding.",
        "recommended_status": "conflicted",
        "conflict_keys": ["auth.failure.cause"],
    }


def test_coordination_review_export_renders_static_markdown_without_writes(tmp_path: Path) -> None:
    """Review export should be portable, readable, and replay-only."""
    eventloom_path = tmp_path / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    pending = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
        claim_key="auth.failure.cause",
        claim_value="expired-jwks-cache",
    )
    before_events = {
        path.name: len(manager.session_manager.replay(path.stem).events)
        for path in sorted(eventloom_path.glob("*.jsonl"))
    }

    review_export = manager.review_export("auth-main")

    assert review_export.mission_id == "auth-main"
    assert review_export.packet.packet_id.startswith("auth-main:approval:")
    assert review_export.markdown.startswith("# Zaxy Coordinate Review: auth-main\n")
    assert "Objective: Ship auth refactor" in review_export.markdown
    assert "Findings needing review: 1" in review_export.markdown
    assert "- Next action: review_finding - Review and decide whether to accept, reject, defer, or mark conflicted." in review_export.markdown
    assert f"## {pending.finding_id}" in review_export.markdown
    assert "Status options: accepted, conflicted, deferred, rejected" in review_export.markdown
    assert "- Evidence: command `pytest tests/test_auth.py -q`" in review_export.markdown
    assert "```json" in review_export.markdown
    assert '"promote": false' in review_export.markdown
    assert review_export.to_dict()["read_only"] is True
    after_events = {
        path.name: len(manager.session_manager.replay(path.stem).events)
        for path in sorted(eventloom_path.glob("*.jsonl"))
    }
    assert after_events == before_events


def test_coordination_audit_report_cites_eventloom_sequence_and_hash(tmp_path: Path) -> None:
    """Mission audit reports should be replay-only and cite Eventloom provenance for every step."""
    eventloom_path = tmp_path / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    mission = manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    worker = manager.create_worker("auth-main", "auth-api", actor="lead")
    assignment = manager.assign("auth-main", "auth-api", "trace API auth failures", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    review = manager.review_finding(
        "auth-main",
        finding.finding_id,
        status="accepted",
        rationale="Command-backed.",
        actor="lead",
    )
    promotion = manager.promote_finding("auth-main", finding.finding_id, actor="lead")
    handoff = manager.create_handoff(
        "auth-main",
        summary="Auth mission ready for release.",
        next_steps=["Release branch"],
        actor="lead",
    )
    before_events = {
        path.name: len(manager.session_manager.replay(path.stem).events)
        for path in sorted(eventloom_path.glob("*.jsonl"))
    }

    report = manager.audit_report("auth-main")

    assert report.mission_id == "auth-main"
    assert report.read_only is True
    assert report.summary["event_count"] == 7
    assert report.summary["worker_count"] == 1
    assert report.summary["accepted_findings"] == 1
    assert [event["event_type"] for event in report.events] == [
        "coordination.mission.created",
        "coordination.worker.created",
        "coordination.assignment.created",
        "coordination.finding.reported",
        "coordination.finding.reviewed",
        "coordination.finding.promoted",
        "coordination.handoff.created",
    ]
    for result in [mission, worker, assignment, finding, review, promotion, handoff]:
        assert {
            "session_id": result.event.thread,
            "event_seq": result.event.seq,
            "event_hash": result.event.hash,
        }.items() <= report.events[[event["event_hash"] for event in report.events].index(result.event.hash)].items()
        assert f"seq={result.event.seq}" in report.markdown
        assert result.event.hash in report.markdown
    assert "## Eventloom Audit Trail" in report.markdown
    assert "coordination.finding.promoted" in report.markdown
    after_events = {
        path.name: len(manager.session_manager.replay(path.stem).events)
        for path in sorted(eventloom_path.glob("*.jsonl"))
    }
    assert after_events == before_events


def test_coordination_audit_report_reconstructs_proof_packet_fields(tmp_path: Path) -> None:
    """Proof packet audit rows should expose enough payload to rebuild authority decisions."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
    )
    review = manager.review_finding("auth-main", finding.finding_id, status="accepted", actor="lead")
    promotion = manager.promote_finding("auth-main", finding.finding_id, actor="lead")
    handoff = manager.create_handoff("auth-main", summary="Auth mission ready.", actor="lead")
    artifact = {
        "artifact_id": "sha256:proof",
        "query": "Compose accepted handoff.",
        "answer_candidates": [{"answer": "Accepted cause", "support_source_ids": [finding.finding_id]}],
        "ledger_rows": [
            {"source_group": finding.finding_id, "include_reason": "accepted_parent_state"},
            {"source_group": "pending-finding", "include_reason": "diagnostic_pending"},
        ],
    }
    proof = manager.proof_packet(
        "auth-main",
        artifact,
        decision_scope="handoff",
        handoff_id=handoff.handoff_id,
    ).to_dict()
    manager.session_manager.get("auth-main").eventlog.append(
        "coordination.proof_packet.created",
        actor="coordinator",
        payload=proof,
        thread="auth-main",
    )

    report = manager.audit_report("auth-main")

    proof_event = next(event for event in report.events if event["event_type"] == "coordination.proof_packet.created")
    assert proof_event["artifact_id"] == "sha256:proof"
    assert proof_event["decision_scope"] == "handoff"
    assert proof_event["authority_scope"] == "parent_accepted_state"
    assert proof_event["accepted_finding_ids"] == [finding.finding_id]
    assert proof_event["review_event_refs"] == [
        {"seq": review.event.seq, "hash": review.event.hash, "finding_id": finding.finding_id}
    ]
    assert proof_event["promotion_event_refs"] == [
        {"seq": promotion.event.seq, "hash": promotion.event.hash, "finding_id": finding.finding_id}
    ]
    assert proof_event["worker_source_event_refs"][0]["finding_id"] == finding.finding_id
    assert proof_event["non_authoritative_rows"] == [
        {
            "source_group": "pending-finding",
            "status": "unknown",
            "include_reason": "diagnostic_pending",
            "exclude_reason": None,
        }
    ]
    assert proof_event["handoff_event_ref"] == {
        "handoff_id": handoff.handoff_id,
        "seq": handoff.event.seq,
        "hash": handoff.event.hash,
    }


def test_coordination_inspection_links_handoff_to_proof_packet_refs(tmp_path: Path) -> None:
    """Replay-backed inspection should show which proof packet supports a handoff."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
    )
    manager.review_finding("auth-main", finding.finding_id, status="accepted", actor="lead")
    manager.promote_finding("auth-main", finding.finding_id, actor="lead")
    handoff = manager.create_handoff("auth-main", summary="Auth mission ready.", actor="lead")
    proof = manager.proof_packet(
        "auth-main",
        {
            "artifact_id": "sha256:proof",
            "query": "Compose accepted handoff.",
            "answer_candidates": [{"answer": "Accepted cause", "support_source_ids": [finding.finding_id]}],
            "ledger_rows": [{"source_group": finding.finding_id, "include_reason": "accepted_parent_state"}],
        },
        decision_scope="handoff",
        handoff_id=handoff.handoff_id,
    ).to_dict()
    proof_event = manager.session_manager.get("auth-main").eventlog.append(
        "coordination.proof_packet.created",
        actor="coordinator",
        payload=proof,
        thread="auth-main",
    )

    inspection = manager.inspect_mission("auth-main").to_dict()

    handoff_record = inspection["handoffs"][0]
    assert handoff_record["handoff_id"] == handoff.handoff_id
    assert handoff_record["proof_event_refs"] == [
        {
            "seq": proof_event.seq,
            "hash": proof_event.hash,
            "artifact_id": "sha256:proof",
            "decision_scope": "handoff",
            "authority_scope": "parent_accepted_state",
            "accepted_finding_ids": [finding.finding_id],
        }
    ]


def test_coordination_proof_trace_replays_artifact_handoff_and_ledger(tmp_path: Path) -> None:
    """Proof trace should reconstruct the full Eventloom chain behind a handoff answer."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    finding = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
    )
    review = manager.review_finding("auth-main", finding.finding_id, status="accepted", actor="lead")
    promotion = manager.promote_finding("auth-main", finding.finding_id, actor="lead")
    handoff = manager.create_handoff("auth-main", summary="Auth mission ready.", actor="lead")
    artifact = {
        "schema_version": "synthesis_artifact_v1",
        "artifact_id": "sha256:proof",
        "query": "Compose accepted handoff.",
        "answer_candidates": [
            {
                "rank": 1,
                "answer": "Accepted cause",
                "support_source_ids": [finding.finding_id],
            }
        ],
        "ledger_rows": [
            {
                "fact_id": finding.finding_id,
                "source_group": finding.finding_id,
                "include_reason": "accepted_parent_state",
            }
        ],
    }
    artifact_event = manager.session_manager.get("auth-main").eventlog.append(
        "memory.synthesis.artifact.created",
        actor="coordinator",
        payload=artifact,
        thread="auth-main",
    )
    proof = manager.proof_packet(
        "auth-main",
        artifact,
        decision_scope="handoff",
        handoff_id=handoff.handoff_id,
    ).to_dict()
    proof_event = manager.session_manager.get("auth-main").eventlog.append(
        "coordination.proof_packet.created",
        actor="coordinator",
        payload=proof,
        thread="auth-main",
    )

    trace = manager.proof_trace("auth-main", handoff_id=handoff.handoff_id).to_dict()

    assert trace["proof_event"]["seq"] == proof_event.seq
    assert trace["artifact_event"]["seq"] == artifact_event.seq
    assert trace["handoff_event"]["seq"] == handoff.event.seq
    assert trace["accepted_finding_ids"] == [finding.finding_id]
    assert trace["review_event_refs"] == [
        {"seq": review.event.seq, "hash": review.event.hash, "finding_id": finding.finding_id}
    ]
    assert trace["promotion_event_refs"] == [
        {"seq": promotion.event.seq, "hash": promotion.event.hash, "finding_id": finding.finding_id}
    ]
    assert trace["answer_candidates"] == artifact["answer_candidates"]
    assert trace["ledger_rows"] == artifact["ledger_rows"]


def test_coordination_brief_marks_explicitly_stale_findings(tmp_path: Path) -> None:
    """Explicit stale metadata should be surfaced without semantic guessing."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    stale = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Old flag-missing theory from superseded branch.",
        actor="auth-api-agent",
        evidence=[
            {
                "kind": "transcript",
                "reference": "eventloom://old/events/3#abc",
                "stale": True,
                "superseded_by": "decision:jwks-cache",
            }
        ],
        claim_key="auth.failure.cause",
        claim_value="flag-missing",
    )

    brief = manager.brief("auth-main")
    checkout = manager.checkout("auth-main", include_diagnostics=True)
    ledger = manager.performance_ledger("auth-main")
    packet = manager.approval_packet("auth-main")

    assert [finding.finding_id for finding in brief.stale_findings] == [stale.finding_id]
    assert brief.to_dict()["stale_findings"][0]["stale"] is True
    assert brief.to_dict()["stale_findings"][0]["superseded_by"] == "decision:jwks-cache"
    assert [finding.finding_id for finding in checkout.stale_findings] == [stale.finding_id]
    assert ledger.worker("auth-api").stale_claim_count == 1
    assert ledger.worker("auth-api").stale_claim_rate == 1.0
    assert packet.findings[0].stale is True


def test_coordination_brief_preserves_status_based_stale_evidence(tmp_path: Path) -> None:
    """Status-based stale markers should survive evidence normalization."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    stale = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Old theory was superseded.",
        actor="auth-api-agent",
        evidence=[
            {
                "kind": "transcript",
                "reference": "eventloom://old/events/4#def",
                "status": "superseded",
            }
        ],
    )

    brief = manager.brief("auth-main")

    assert [finding.finding_id for finding in brief.stale_findings] == [stale.finding_id]
    assert brief.stale_findings[0].evidence[0]["status"] == "superseded"


def test_coordination_brief_detects_source_state_conflicts_from_evidence(tmp_path: Path) -> None:
    """Findings citing incompatible source snapshots should be flagged without LLM guessing."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    api = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API worker saw JWKS timeout handling in auth config.",
        actor="auth-api-agent",
        evidence=[
            {
                "kind": "file",
                "reference": "src/auth/config.py",
                "source_sha256": "a" * 64,
            }
        ],
        claim_key="auth.config.timeout",
        claim_value="30s",
    )
    ui = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI worker saw a different auth config snapshot.",
        actor="auth-ui-agent",
        evidence=[
            {
                "kind": "file",
                "reference": "src/auth/config.py",
                "source_sha256": "b" * 64,
            }
        ],
        claim_key="auth.config.retry",
        claim_value="enabled",
    )

    brief = manager.brief("auth-main")
    checkout = manager.checkout("auth-main", include_diagnostics=True)
    packet = manager.approval_packet("auth-main")

    source_conflict = next(conflict for conflict in brief.conflicts if conflict.conflict_type == "source_state")
    assert source_conflict.claim_key == "source:src/auth/config.py"
    assert source_conflict.source_reference == "src/auth/config.py"
    assert {finding.finding_id for finding in source_conflict.findings} == {
        api.finding_id,
        ui.finding_id,
    }
    assert source_conflict.reason == "conflicting_source_snapshots"
    assert {finding.finding_id for finding in brief.conflicted_findings} == {api.finding_id, ui.finding_id}
    assert checkout.conflicts[0].to_dict()["conflict_type"] == "source_state"
    assert "source_state source:src/auth/config.py" in checkout.prompt
    assert packet.findings[0].conflict_keys == ["source:src/auth/config.py"]
    assert brief.pending_findings[0].evidence[0]["source_sha256"] == "a" * 64


def test_coordination_brief_can_use_explicit_semantic_conflict_adapter(tmp_path: Path) -> None:
    """Semantic conflict detection should be opt-in and clearly labeled."""

    def detector(findings):
        left = next(finding for finding in findings if finding.worker_id == "auth-api")
        right = next(finding for finding in findings if finding.worker_id == "auth-ui")
        return [
            ConflictState(
                claim_key="semantic:auth.failure.cause",
                findings=[left, right],
                conflict_type="semantic",
                reason="local_reranker_contradiction",
            )
        ]

    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom", semantic_conflict_detector=detector)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    api = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API worker says expired JWKS cache is the likely cause.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    ui = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI worker says browser refresh handling is the likely cause.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:42"}],
    )

    brief = manager.brief("auth-main")

    semantic = next(conflict for conflict in brief.conflicts if conflict.conflict_type == "semantic")
    assert semantic.claim_key == "semantic:auth.failure.cause"
    assert semantic.reason == "local_reranker_contradiction"
    assert {finding.finding_id for finding in semantic.findings} == {api.finding_id, ui.finding_id}
    assert "semantic semantic:auth.failure.cause" in manager.checkout(
        "auth-main",
        include_diagnostics=True,
    ).prompt


def test_coordination_semantic_conflict_detection_is_off_by_default(tmp_path: Path) -> None:
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API worker says expired JWKS cache is the likely cause.",
        actor="auth-api-agent",
    )
    manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI worker says browser refresh handling is the likely cause.",
        actor="auth-ui-agent",
    )

    assert [conflict.conflict_type for conflict in manager.brief("auth-main").conflicts] == []


def test_local_semantic_conflict_detector_finds_shared_subject_contradictions(tmp_path: Path) -> None:
    manager = CoordinationManager(
        eventloom_path=tmp_path / ".eventloom",
        semantic_conflict_detector=LocalSemanticConflictDetector(),
    )
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    api = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Token refresh retry is enabled in auth middleware.",
        actor="auth-api-agent",
        evidence=[{"kind": "file", "reference": "src/auth/session.py:42"}],
    )
    ui = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="Token refresh retry is disabled in browser session handling.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/ui/session.ts:88"}],
    )

    brief = manager.brief("auth-main")

    semantic = next(conflict for conflict in brief.conflicts if conflict.conflict_type == "semantic")
    assert semantic.claim_key == "semantic:refresh-retry-token"
    assert semantic.reason == "local_lexical_contradiction:disabled/enabled"
    assert {finding.finding_id for finding in semantic.findings} == {api.finding_id, ui.finding_id}


def test_local_semantic_conflict_detector_ignores_unrelated_antonyms(tmp_path: Path) -> None:
    manager = CoordinationManager(
        eventloom_path=tmp_path / ".eventloom",
        semantic_conflict_detector=LocalSemanticConflictDetector(),
    )
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Token refresh retry is enabled in auth middleware.",
        actor="auth-api-agent",
    )
    manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="Billing invoice export is disabled in the admin panel.",
        actor="auth-ui-agent",
    )

    assert [conflict.conflict_type for conflict in manager.brief("auth-main").conflicts] == []


def test_coordination_rejects_semantic_conflicts_with_unknown_findings(tmp_path: Path) -> None:
    def detector(findings):
        unknown = ConflictState(
            claim_key="semantic:bad",
            findings=[findings[0], findings[0].__class__(**{**findings[0].to_dict(), "finding_id": "invented"})],
            conflict_type="semantic",
        )
        return [unknown]

    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom", semantic_conflict_detector=detector)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API worker says expired JWKS cache is the likely cause.",
        actor="auth-api-agent",
    )

    try:
        manager.brief("auth-main")
    except ValueError as exc:
        assert "unknown semantic conflict finding_id" in str(exc)
    else:
        raise AssertionError("expected unknown semantic conflict finding_id rejection")


def test_coordination_records_detected_source_conflicts_for_graph_projection(tmp_path: Path) -> None:
    """Detected conflicts should be materializable as idempotent Eventloom graph facts."""
    eventloom_path = tmp_path / ".eventloom"
    manager = CoordinationManager(eventloom_path=eventloom_path)
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    manager.create_worker("auth-main", "auth-ui", actor="lead")
    api = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="API worker saw one auth config snapshot.",
        actor="auth-api-agent",
        evidence=[{"kind": "file", "reference": "src/auth/config.py", "source_sha256": "a" * 64}],
    )
    ui = manager.report_finding(
        "auth-main",
        "auth-ui",
        summary="UI worker saw another auth config snapshot.",
        actor="auth-ui-agent",
        evidence=[{"kind": "file", "reference": "src/auth/config.py", "source_sha256": "b" * 64}],
    )

    first = manager.record_detected_conflicts("auth-main", actor="zaxy")
    second = manager.record_detected_conflicts("auth-main", actor="zaxy")

    assert len(first) == 1
    assert second == []
    event = first[0].event
    assert event.type == "coordination.conflict.detected"
    assert event.thread == "auth-main"
    assert event.payload["conflict_type"] == "source_state"
    assert event.payload["source_reference"] == "src/auth/config.py"
    assert event.payload["finding_ids"] == [api.finding_id, ui.finding_id]
    parent_events = CoordinationManager(eventloom_path=eventloom_path).session_manager.replay("auth-main").events
    assert [item.type for item in parent_events].count("coordination.conflict.detected") == 1


def test_coordination_apply_approval_decisions_reviews_and_promotes(tmp_path: Path) -> None:
    """Remote approval decisions should append normal review and promotion events."""
    manager = CoordinationManager(eventloom_path=tmp_path / ".eventloom")
    manager.start_mission("auth-main", objective="Ship auth refactor", actor="lead")
    manager.create_worker("auth-main", "auth-api", actor="lead")
    accepted = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Expired JWKS cache causes API failures.",
        actor="auth-api-agent",
        evidence=[{"kind": "command", "reference": "pytest tests/test_auth.py -q"}],
    )
    deferred = manager.report_finding(
        "auth-main",
        "auth-api",
        summary="Browser refresh needs more evidence.",
        actor="auth-api-agent",
    )

    result = manager.apply_approval_decisions(
        "auth-main",
        [
            {
                "finding_id": accepted.finding_id,
                "status": "accepted",
                "rationale": "Command-backed.",
                "promote": True,
            },
            {
                "finding_id": deferred.finding_id,
                "status": "deferred",
                "rationale": "Needs browser trace.",
            },
        ],
        actor="reviewer",
    )

    assert result.mission_id == "auth-main"
    assert result.reviewed_count == 2
    assert result.promoted_count == 1
    assert [event.type for event in result.events] == [
        "coordination.finding.reviewed",
        "coordination.finding.promoted",
        "coordination.finding.reviewed",
    ]
    brief = manager.brief("auth-main")
    assert [finding.finding_id for finding in brief.accepted_findings] == [accepted.finding_id]
    assert [finding.finding_id for finding in brief.deferred_findings] == [deferred.finding_id]
