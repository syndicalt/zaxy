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
                "crm://deal/8841#renewal-policy",
                "eventloom://pricing/events/19#approval-required",
            ],
        }
    ]
    assert case["audit"]["replayable"] is True
    assert "gold" not in json.dumps(result)


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
