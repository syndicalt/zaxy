"""Zaxy-owned executable adapter for the public CoordinationBench contract."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zaxy.coordination import CoordinationManager
from zaxy.release import package_version


@dataclass(frozen=True)
class FindingDecision:
    """Public-signal decision for one CoordinationBench finding."""

    finding_id: str
    status: str
    stale: bool = False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--runtime")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    workload = json.loads(Path(args.workload).read_text(encoding="utf-8"))
    runtime = json.loads(Path(args.runtime).read_text(encoding="utf-8")) if args.runtime else None
    with tempfile.TemporaryDirectory(prefix="zaxy-coordinationbench-") as tempdir:
        result = build_coordinationbench_result(
            workload,
            runtime=runtime,
            eventloom_path=Path(tempdir) / ".eventloom",
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def build_coordinationbench_result(
    workload: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
    eventloom_path: str | Path,
) -> dict[str, Any]:
    """Build a CoordinationBench result from public workload/runtime payloads."""
    manager = CoordinationManager(eventloom_path=eventloom_path)
    cases = [
        _case_result(manager, case, runtime_case=_runtime_case(runtime, str(case["case_id"])))
        for case in workload.get("cases", [])
    ]
    return {
        "adapter": {"name": "zaxy-coordinate", "version": package_version()},
        "workload_fingerprint": str(workload["fingerprint"]),
        "cases": cases,
    }


def _case_result(
    manager: CoordinationManager,
    case: dict[str, Any],
    *,
    runtime_case: dict[str, Any] | None,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    mission_id = f"coordbench-{case_id}"
    manager.start_mission(mission_id, objective=str(case.get("objective") or ""), actor="coordinationbench")
    worker_ids = _worker_ids(case)
    for worker_id in worker_ids:
        manager.create_worker(mission_id, worker_id, actor="coordinationbench")
        manager.assign(mission_id, worker_id, f"CoordinationBench case {case_id}", actor="coordinationbench")
    public_findings = [dict(finding) for finding in case.get("findings", [])]
    decisions = _decide_findings(public_findings)
    decisions_by_id = {decision.finding_id: decision for decision in decisions}
    for finding in public_findings:
        finding_id = str(finding["finding_id"])
        decision = decisions_by_id[finding_id]
        manager.report_finding(
            mission_id,
            str(finding["worker_id"]),
            summary=str(finding.get("summary") or ""),
            actor=str(finding["worker_id"]),
            evidence=_zaxy_evidence(finding, stale=decision.stale),
            claim_key=_optional_str(finding.get("claim_key")),
            claim_value=_optional_str(finding.get("claim_value")),
            finding_id=finding_id,
        )
    for decision in decisions:
        manager.review_finding(
            mission_id,
            decision.finding_id,
            status=decision.status,
            actor="coordinationbench-policy",
            rationale=_rationale(decision),
        )
        if decision.status == "accepted":
            manager.promote_finding(mission_id, decision.finding_id, actor="coordinationbench-policy")
    manager.record_detected_conflicts(mission_id, actor="coordinationbench-policy")
    brief = manager.brief(mission_id)
    accepted = sorted(finding.finding_id for finding in brief.accepted_findings)
    stale = sorted(finding.finding_id for finding in brief.stale_findings if finding.finding_id not in accepted)
    rejected = sorted(
        {
            finding.finding_id
            for finding in [*brief.rejected_findings, *brief.conflicted_findings, *brief.deferred_findings]
            if finding.finding_id not in accepted
        }
    )
    accepted_group = _accepted_value_group(public_findings, accepted)
    return {
        "case_id": case_id,
        "accepted_findings": accepted,
        "rejected_findings": rejected,
        "stale_findings": stale,
        "conflicts": _conflicts(brief.conflicts),
        "answers": [
            _answer(question, accepted_group)
            for question in case.get("questions", [])
        ],
        "audit": {
            "replayable": True,
            "notes": _audit_notes(runtime_case),
        },
    }


def _decide_findings(findings: list[dict[str, Any]]) -> list[FindingDecision]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        by_key[str(finding.get("claim_key") or "")].append(finding)
    decisions: list[FindingDecision] = []
    for scoped in by_key.values():
        if not scoped:
            continue
        if all(not finding.get("evidence") for finding in scoped):
            decisions.extend(
                FindingDecision(finding_id=str(finding["finding_id"]), status="rejected")
                for finding in scoped
            )
            continue
        accepted_value = _winning_value(scoped)
        accepted = _best_finding([finding for finding in scoped if str(finding.get("claim_value")) == accepted_value])
        accepted_id = str(accepted["finding_id"])
        for finding in scoped:
            finding_id = str(finding["finding_id"])
            if finding_id == accepted_id:
                decisions.append(FindingDecision(finding_id=finding_id, status="accepted"))
                continue
            decisions.append(
                FindingDecision(
                    finding_id=finding_id,
                    status="rejected",
                    stale=_looks_stale(finding, accepted),
                )
            )
    return decisions


def _winning_value(findings: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for finding in findings:
        grouped[str(finding.get("claim_value") or "")].append(finding)
    return max(
        grouped,
        key=lambda value: (
            _value_score(grouped[value]),
            len(grouped[value]),
            max(_timestamp(finding) for finding in grouped[value]),
        ),
    )


def _value_score(findings: list[dict[str, Any]]) -> float:
    evidence_score = sum(_finding_source_score(finding) for finding in findings)
    corroboration_bonus = 2.0 * max(0, len(findings) - 1)
    current_bonus = 0.0
    for finding in findings:
        text = f"{finding.get('summary', '')} {finding.get('claim_value', '')}".casefold()
        if any(marker in text for marker in ("current", "supersedes", "after", "fixed", "ready")):
            current_bonus += 2.0
        if any(marker in text for marker in ("old", "older", "earlier", "legacy", "before")):
            current_bonus -= 2.0
    return evidence_score + corroboration_bonus + current_bonus


def _best_finding(findings: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        findings,
        key=lambda finding: (
            bool(finding.get("evidence")),
            -_timestamp(finding).timestamp(),
        ),
    )


def _finding_source_score(finding: dict[str, Any]) -> float:
    evidence = finding.get("evidence", [])
    if not evidence:
        return -4.0
    return sum(_reference_score(str(reference)) for reference in evidence)


def _reference_score(reference: str) -> float:
    if reference.startswith("audit-signed://"):
        return 9.0
    if reference.startswith(("erp://", "docusign://", "contractrepo://")):
        return 7.0
    if reference.startswith(("ci://", "eventloom://", "manifest://", "roadmap://")):
        return 6.0
    if reference.startswith(("crm://", "prd://", "call://")):
        return 5.0
    if reference.startswith(("ticket://", "docs/", "logs/")):
        return 4.0
    if reference.startswith("chat://"):
        return 2.0
    if reference.startswith(("worker-note://", "noise://")):
        return -3.0
    return 1.0


def _looks_stale(finding: dict[str, Any], accepted: dict[str, Any]) -> bool:
    if str(finding.get("claim_value")) == str(accepted.get("claim_value")):
        return False
    text = str(finding.get("summary") or "").casefold()
    if any(marker in text for marker in ("stale", "old", "older", "earlier", "legacy", "before")):
        return True
    return _timestamp(finding) < _timestamp(accepted) and any(
        marker in str(accepted.get("summary") or "").casefold()
        for marker in ("current", "supersedes", "after", "fixed", "ready")
    )


def _worker_ids(case: dict[str, Any]) -> list[str]:
    worker_ids = {str(worker.get("worker_id")) for worker in case.get("workers", []) if worker.get("worker_id")}
    worker_ids.update(str(finding["worker_id"]) for finding in case.get("findings", []))
    return sorted(worker_ids)


def _zaxy_evidence(finding: dict[str, Any], *, stale: bool) -> list[dict[str, Any]]:
    evidence = []
    for reference in finding.get("evidence", []):
        item = {"kind": _evidence_kind(str(reference)), "reference": str(reference)}
        if stale:
            item["stale"] = True
        evidence.append(item)
    return evidence


def _evidence_kind(reference: str) -> str:
    if "://" in reference:
        return reference.split("://", 1)[0]
    if reference.startswith("docs/"):
        return "document"
    if reference.startswith("logs/"):
        return "log"
    return "source"


def _accepted_value_group(
    findings: list[dict[str, Any]],
    accepted_ids: list[str],
) -> list[dict[str, Any]]:
    if not accepted_ids:
        return []
    accepted_id = accepted_ids[0]
    accepted = next(finding for finding in findings if str(finding["finding_id"]) == accepted_id)
    return [
        finding
        for finding in findings
        if finding.get("claim_key") == accepted.get("claim_key")
        and finding.get("claim_value") == accepted.get("claim_value")
    ]


def _answer(question: dict[str, Any], accepted_group: list[dict[str, Any]]) -> dict[str, Any]:
    if not accepted_group:
        answer = "No supported answer found."
        evidence: list[str] = []
    else:
        accepted = _best_finding(accepted_group)
        answer = f"{accepted.get('claim_key')} is {accepted.get('claim_value')}."
        evidence = sorted(
            {
                str(reference)
                for finding in accepted_group
                for reference in finding.get("evidence", [])
            }
        )
    return {
        "question_id": str(question["question_id"]),
        "answer": answer,
        "evidence": evidence,
    }


def _conflicts(conflicts: list[Any]) -> list[dict[str, Any]]:
    payload = []
    for conflict in conflicts:
        finding_ids = sorted(finding.finding_id for finding in conflict.findings)
        if len(finding_ids) < 2:
            continue
        payload.append(
            {
                "finding_ids": finding_ids,
                "reason": conflict.reason or conflict.conflict_type,
            }
        )
    return sorted(payload, key=lambda item: item["finding_ids"])


def _runtime_case(runtime: dict[str, Any] | None, case_id: str) -> dict[str, Any] | None:
    if runtime is None:
        return None
    for case in runtime.get("cases", []):
        if str(case.get("case_id")) == case_id:
            return dict(case)
    return None


def _audit_notes(runtime_case: dict[str, Any] | None) -> str:
    if runtime_case is None:
        return "zaxy coordination replay from public workload"
    return (
        "zaxy coordination replay from public workload and runtime manifest; "
        f"runtime_events={runtime_case.get('event_count', 0)}"
    )


def _rationale(decision: FindingDecision) -> str:
    if decision.status == "accepted":
        return "public-signal policy selected this finding for parent mission state"
    if decision.stale:
        return "public-signal policy rejected this finding as stale"
    return "public-signal policy rejected this finding"


def _timestamp(finding: dict[str, Any]) -> datetime:
    value = str(finding.get("timestamp") or "1970-01-01T00:00:00Z")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


if __name__ == "__main__":
    raise SystemExit(main())
