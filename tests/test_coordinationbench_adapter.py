"""Tests for the external CoordinationBench adapter contract."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.coordinationbench_adapter import build_coordinationbench_result


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
