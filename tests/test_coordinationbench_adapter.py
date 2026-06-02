"""Tests for the external CoordinationBench adapter contract."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.coordinationbench_adapter import (
    _evidence_kind,
    _reference_score,
    build_coordinationbench_result,
    main,
)


def test_coordinationbench_adapter_consolidates_public_findings(tmp_path: Path) -> None:
    workload = {
        "version": "coordinationbench-test",
        "fingerprint": "abc123",
        "cases": [
            {
                "case_id": "pricing-policy-temporal",
                "domain": "policy",
                "objective": "Resolve current pricing policy.",
                "workers": [
                    {"worker_id": "worker-history"},
                    {"worker_id": "worker-policy"},
                    {"worker_id": "worker-crm"},
                ],
                "findings": [
                    {
                        "finding_id": "pricing-old-discount",
                        "worker_id": "worker-history",
                        "claim_key": "policy.discount",
                        "claim_value": "legacy-15-percent",
                        "summary": "Older transcript says enterprise renewal discount is 15 percent.",
                        "timestamp": "2026-05-20T10:00:00Z",
                        "evidence": ["eventloom://pricing/events/4#legacy-discount"],
                    },
                    {
                        "finding_id": "pricing-current-approval",
                        "worker_id": "worker-policy",
                        "claim_key": "policy.discount",
                        "claim_value": "approval-required",
                        "summary": "Current policy supersedes automatic discounts.",
                        "timestamp": "2026-05-26T16:00:00Z",
                        "evidence": ["eventloom://pricing/events/19#approval-required"],
                    },
                    {
                        "finding_id": "pricing-crm-approval",
                        "worker_id": "worker-crm",
                        "claim_key": "policy.discount",
                        "claim_value": "approval-required",
                        "summary": "CRM confirms the approval-required policy.",
                        "timestamp": "2026-05-27T09:00:00Z",
                        "evidence": ["crm://deal/8841#renewal-policy"],
                    },
                    {
                        "finding_id": "pricing-unsupported-exception",
                        "worker_id": "worker-crm",
                        "claim_key": "policy.exception",
                        "claim_value": "automatic-exception",
                        "summary": "Unsupported side note says an automatic exception exists.",
                        "timestamp": "2026-05-27T09:10:00Z",
                        "evidence": [],
                    },
                ],
                "questions": [
                    {
                        "question_id": "pricing-current-policy",
                        "prompt": "What discount policy is current for the renewal?",
                    }
                ],
            }
        ],
    }

    result = build_coordinationbench_result(workload, eventloom_path=tmp_path / ".eventloom")

    assert result["adapter"]["name"] == "zaxy-coordinate"
    assert result["workload_fingerprint"] == "abc123"
    case = result["cases"][0]
    assert case["accepted_findings"] == ["pricing-current-approval"]
    assert case["rejected_findings"] == [
        "pricing-crm-approval",
        "pricing-old-discount",
        "pricing-unsupported-exception",
    ]
    assert case["stale_findings"] == ["pricing-old-discount"]
    assert case["conflicts"] == [
        {
            "finding_ids": [
                "pricing-crm-approval",
                "pricing-current-approval",
                "pricing-old-discount",
            ],
            "reason": "conflicting_claim_values",
        }
    ]
    assert case["answers"] == [
        {
            "question_id": "pricing-current-policy",
            "answer": "policy.discount is approval-required.",
            "evidence": [
                "eventloom://pricing/events/19#approval-required",
            ],
        }
    ]
    assert case["audit"]["replayable"] is True
    assert "gold" not in json.dumps(result)


def test_coordinationbench_adapter_builds_source_aware_answer_and_excludes_stale_terms(tmp_path: Path) -> None:
    """Public-derived answer cases should use current accepted sources, not stale high-authority history."""
    workload = {
        "version": "coordinationbench-test",
        "fingerprint": "source-aware",
        "cases": [
            {
                "case_id": "release-state-current",
                "domain": "release",
                "objective": "Resolve current release gate state.",
                "workers": [
                    {"worker_id": "worker-audit"},
                    {"worker_id": "worker-release"},
                    {"worker_id": "worker-docs"},
                ],
                "findings": [
                    {
                        "finding_id": "release-old-blocked",
                        "worker_id": "worker-audit",
                        "claim_key": "release.state",
                        "claim_value": "blocked-by-kuzu-lock",
                        "summary": "Old audit-signed record says the release was blocked by Kuzu lock contention.",
                        "timestamp": "2026-05-20T10:00:00Z",
                        "evidence": ["audit-signed://release/old-blocked"],
                    },
                    {
                        "finding_id": "release-current-ready",
                        "worker_id": "worker-release",
                        "claim_key": "release.state",
                        "claim_value": "ready-for-1.0.2",
                        "summary": "Current release checklist supersedes the Kuzu lock blocker and marks 1.0.2 ready.",
                        "timestamp": "2026-05-28T16:00:00Z",
                        "evidence": ["manifest://release/1.0.2#ready"],
                    },
                    {
                        "finding_id": "release-docs-ready",
                        "worker_id": "worker-docs",
                        "claim_key": "release.state",
                        "claim_value": "ready-for-1.0.2",
                        "summary": "Docs confirm current 1.0.2 release readiness.",
                        "timestamp": "2026-05-28T17:00:00Z",
                        "evidence": ["docs/release.md#ready-1.0.2"],
                    },
                ],
                "questions": [
                    {
                        "question_id": "release-current-state",
                        "prompt": "What is the current release state?",
                        "expected_terms": ["ready-for-1.0.2"],
                        "forbidden_terms": ["blocked-by-kuzu-lock"],
                    }
                ],
            }
        ],
    }

    result = build_coordinationbench_result(workload, eventloom_path=tmp_path / ".eventloom")

    case = result["cases"][0]
    assert case["accepted_findings"] == ["release-current-ready"]
    assert case["stale_findings"] == ["release-old-blocked"]
    answer = case["answers"][0]
    assert "ready-for-1.0.2" in answer["answer"]
    assert "blocked-by-kuzu-lock" not in json.dumps(answer).casefold()
    assert answer["evidence"] == [
        "manifest://release/1.0.2#ready",
    ]
    assert case["returned_text"] == answer["answer"]
    assert case["answer_candidate"]["answer"] == answer["answer"]
    assert case["answer_candidate"]["support_source_ids"] == [
        "release-current-ready",
    ]
    assert case["support_source_ids"] == ["release-current-ready"]
    assert set(case["excluded_source_ids"]) == {"release-docs-ready", "release-old-blocked"}
    assert case["non_authoritative_rows_injected"] == 0
    assert case["synthesis_artifact"]["answer_candidates"] == [case["answer_candidate"]]
    assert {
        (row["source_group"], row["include_reason"], row.get("exclude_reason", ""))
        for row in case["synthesis_artifact"]["ledger_rows"]
    } == {
        ("release-current-ready", "accepted_parent_state", ""),
        ("release-docs-ready", "stale_or_rejected_state", "rejected"),
        ("release-old-blocked", "stale_or_rejected_state", "stale"),
    }


def test_coordinationbench_adapter_excludes_unsupported_same_value_from_answer_support(tmp_path: Path) -> None:
    """Same-value findings without evidence should not become proof support."""
    workload = {
        "version": "coordinationbench-test",
        "fingerprint": "same-value-unsupported",
        "cases": [
            {
                "case_id": "release-same-value",
                "objective": "Resolve release state.",
                "workers": [{"worker_id": "worker-release"}, {"worker_id": "worker-notes"}],
                "findings": [
                    {
                        "finding_id": "release-supported",
                        "worker_id": "worker-release",
                        "claim_key": "release.state",
                        "claim_value": "ready",
                        "summary": "Current manifest says release is ready.",
                        "timestamp": "2026-05-28T16:00:00Z",
                        "evidence": ["manifest://release#ready"],
                    },
                    {
                        "finding_id": "release-unsupported-same",
                        "worker_id": "worker-notes",
                        "claim_key": "release.state",
                        "claim_value": "ready",
                        "summary": "Scratch note repeats ready without evidence.",
                        "timestamp": "2026-05-28T17:00:00Z",
                        "evidence": [],
                    },
                ],
                "questions": [{"question_id": "release-state", "prompt": "What is the release state?"}],
            }
        ],
    }

    result = build_coordinationbench_result(workload, eventloom_path=tmp_path / ".eventloom")

    case = result["cases"][0]
    assert case["accepted_findings"] == ["release-supported"]
    assert case["support_source_ids"] == ["release-supported"]
    assert case["excluded_source_ids"] == ["release-unsupported-same"]
    assert case["answer_candidate"]["support_source_ids"] == ["release-supported"]
    assert case["answer_candidate"]["excluded_source_ids"] == ["release-unsupported-same"]
    assert {
        (row["source_group"], row["include_reason"], row.get("exclude_reason", ""))
        for row in case["synthesis_artifact"]["ledger_rows"]
    } == {
        ("release-supported", "accepted_parent_state", ""),
        ("release-unsupported-same", "stale_or_rejected_state", "rejected"),
    }


def test_coordinationbench_adapter_handles_empty_and_runtime_cases(tmp_path: Path) -> None:
    workload = {
        "version": "coordinationbench-test",
        "fingerprint": "empty-case",
        "cases": [
            {
                "case_id": "unsupported-only",
                "objective": "Find supported answer.",
                "workers": [],
                "findings": [
                    {
                        "finding_id": "unsupported",
                        "worker_id": "worker-notes",
                        "claim_key": "",
                        "claim_value": "",
                        "summary": "",
                        "timestamp": "",
                        "evidence": [],
                    }
                ],
                "questions": [{"question_id": "q-empty", "prompt": "What is supported?"}],
            }
        ],
    }
    runtime = {"cases": [{"case_id": "unsupported-only", "event_count": 12}]}

    result = build_coordinationbench_result(workload, runtime=runtime, eventloom_path=tmp_path / ".eventloom")

    case = result["cases"][0]
    assert case["accepted_findings"] == []
    assert case["rejected_findings"] == ["unsupported"]
    assert case["answers"] == [
        {"question_id": "q-empty", "answer": "No supported answer found.", "evidence": []}
    ]
    assert case["audit"]["notes"].endswith("runtime_events=12")


def test_coordinationbench_adapter_cli_writes_result(tmp_path: Path) -> None:
    workload_path = tmp_path / "workload.json"
    runtime_path = tmp_path / "runtime.json"
    output_path = tmp_path / "result.json"
    workload_path.write_text(
        json.dumps(
            {
                "version": "coordinationbench-test",
                "fingerprint": "cli-case",
                "cases": [
                    {
                        "case_id": "cli",
                        "objective": "Resolve cli case.",
                        "workers": [{"worker_id": "worker-docs"}],
                        "findings": [
                            {
                                "finding_id": "doc-current",
                                "worker_id": "worker-docs",
                                "claim_key": "release.state",
                                "claim_value": "ready",
                                "summary": "Current release is ready.",
                                "timestamp": "2026-05-27T09:00:00Z",
                                "evidence": ["docs/release.md#ready", "logs/release.log#ready"],
                            }
                        ],
                        "questions": [{"question_id": "q-cli", "prompt": "What is the release state?"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runtime_path.write_text(json.dumps({"cases": [{"case_id": "missing", "event_count": 99}]}), encoding="utf-8")

    assert main(["--workload", str(workload_path), "--runtime", str(runtime_path), "--output", str(output_path)]) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["workload_fingerprint"] == "cli-case"
    assert result["cases"][0]["accepted_findings"] == ["doc-current"]
    assert result["cases"][0]["audit"]["notes"] == "zaxy coordination replay from public workload"


def test_coordinationbench_adapter_scores_source_types() -> None:
    assert _reference_score("audit-signed://approval/1") == 9.0
    assert _reference_score("erp://invoice/1") == 7.0
    assert _reference_score("docusign://envelope/1") == 7.0
    assert _reference_score("contractrepo://agreement/1") == 7.0
    assert _reference_score("ci://run/1") == 6.0
    assert _reference_score("manifest://release/1") == 6.0
    assert _reference_score("roadmap://item/1") == 6.0
    assert _reference_score("prd://feature/1") == 5.0
    assert _reference_score("call://transcript/1") == 5.0
    assert _reference_score("ticket://bug/1") == 4.0
    assert _reference_score("docs/runbook.md#step") == 4.0
    assert _reference_score("logs/build.log#step") == 4.0
    assert _reference_score("chat://thread/1") == 2.0
    assert _reference_score("worker-note://scratch/1") == -3.0
    assert _reference_score("noise://scratch/1") == -3.0
    assert _reference_score("unknown://source/1") == 1.0
    assert _evidence_kind("docs/runbook.md#step") == "document"
    assert _evidence_kind("logs/build.log#step") == "log"
    assert _evidence_kind("plain-reference") == "source"
