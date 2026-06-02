"""Tests for deterministic synthesis artifact payloads."""

from __future__ import annotations

import json

import pytest

from zaxy.core import MemoryCheckout
from zaxy.synthesis_artifact import (
    build_synthesis_artifact,
    build_synthesis_candidate_event_payload,
    build_synthesis_evidence_event_payload,
    normalize_synthesis_outcome,
    synthesis_outcome_event_type,
)


def _checkout() -> MemoryCheckout:
    return MemoryCheckout(
        session_id="agent-1",
        query="How much did I spend on bike expenses in total?",
        prompt="# Memory Checkout\nThis prompt should not affect artifact identity.",
        working_set={},
        ref={"name": "HEAD", "target_seq": 12, "target_hash": "a" * 64},
        current_facts=[],
        evidence=[
            {
                "content": "session_id=answer-1 I spent $120 on a bike helmet.",
                "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                "source_lane": "verbatim",
            },
            {
                "content": "session_id=answer-2 I spent $25 on a chain.",
                "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
                "source_lane": "verbatim",
            },
        ],
        provenance=[],
        retention={},
        warnings=["one low-salience source excluded"],
        guidance={},
        quality={
            "answerability": "answer_from_memory",
            "confidence": 0.86,
            "required_action": {
                "type": "memory_checkout",
                "missing_slots": ["source"],
                "suggested_queries": [
                    {"slot": "source", "query": "bike expenses supporting source"}
                ],
            },
        },
        diagnostics={
            "evidence_plan_status": {
                "satisfied": False,
                "observed_source_groups": 1,
                "required_source_groups": 2,
                "refresh_query": "bike expenses supporting source",
            },
            "slot_plan": {
                "version": "slot_plan_v1",
                "answer_type": "sum",
                "operation": "sum_values",
                "required_slots": ["source", "numeric"],
            },
            "synthesis": {
                "answer_candidates": [
                    {
                        "rank": 1,
                        "type": "currency",
                        "confidence": 0.83,
                        "answer_key": "currency_total_answer",
                        "answer": "$145",
                        "support_source_ids": ["answer-1", "answer-2"],
                        "excluded_source_ids": ["answer-4"],
                    }
                ],
                "ledger_rows": [
                    {
                        "fact_id": "currency:0:0",
                        "source_group": "answer-1",
                        "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                        "kind": "currency",
                        "value": "120",
                        "unit": "USD",
                        "label": "bike helmet",
                        "raw_span": "$120 on a bike helmet",
                        "normalized_identity": "currency:bike helmet:120",
                        "include_reason": "currency_amount",
                        "exclude_reason": "",
                        "confidence": 0.86,
                    },
                    {
                        "fact_id": "currency:2:0",
                        "source_group": "answer-4",
                        "citation": "eventloom://agent-1/events/4#dddddddddddd",
                        "kind": "currency",
                        "value": "40",
                        "unit": "USD",
                        "label": "bike lights",
                        "raw_span": "$40 on bike lights",
                        "normalized_identity": "currency:bike lights:40",
                        "include_reason": "currency_amount",
                        "exclude_reason": "duplicate_identity",
                        "confidence": 0.61,
                    },
                ],
            },
            "skill_analytics": {
                "rollback_candidates": [
                    {
                        "skill_id": "deploy-cache-check",
                        "reason": "contradicted",
                        "rollback_to": "v1",
                    }
                ]
            },
        },
        context_counts={"verbatim": 2},
        replay_event_count=9,
        compacted=False,
        assembly_policy={"source_recall": "reserved"},
    )


def test_synthesis_artifact_is_deterministic_and_json_safe() -> None:
    """Artifact identity should depend on checkout/candidate evidence, not prompt text."""
    first = build_synthesis_artifact(_checkout())
    second = build_synthesis_artifact(_checkout())

    assert first == second
    assert first["schema_version"] == "synthesis_artifact_v1"
    assert first["artifact_id"].startswith("sha256:")
    json.dumps(first, sort_keys=True)


def test_synthesis_artifact_preserves_candidates_support_and_verification() -> None:
    """Artifacts should preserve the model-facing answer and its proof packet."""
    artifact = build_synthesis_artifact(_checkout())

    assert artifact["answer_candidates"] == [
        {
            "rank": 1,
            "type": "currency",
            "confidence": 0.83,
            "answer_key": "currency_total_answer",
            "answer": "$145",
            "support_source_ids": ["answer-1", "answer-2"],
            "excluded_source_ids": ["answer-4"],
        }
    ]
    assert artifact["support_packet"] == {
        "citations": [
            "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "eventloom://agent-1/events/2#bbbbbbbbbbbb",
        ],
        "source_groups": ["answer-1", "answer-2"],
        "snippets": [
            "session_id=answer-1 I spent $120 on a bike helmet.",
            "session_id=answer-2 I spent $25 on a chain.",
        ],
    }
    assert artifact["verification"] == {
        "warnings": ["one low-salience source excluded"],
        "missing_evidence": [
            {
                "observed_source_groups": 1,
                "required_source_groups": 2,
                "refresh_query": "bike expenses supporting source",
                "missing_slots": ["source"],
                "suggested_queries": [
                    {"slot": "source", "query": "bike expenses supporting source"}
                ],
            }
        ],
        "contradictions": [
            {
                "type": "skill_memory",
                "skill_id": "deploy-cache-check",
                "reason": "contradicted",
                "rollback_to": "v1",
            }
        ],
        "dedupe_decisions": [
            {
                "fact_id": "currency:2:0",
                "source_group": "answer-4",
                "citation": "eventloom://agent-1/events/4#dddddddddddd",
                "normalized_identity": "currency:bike lights:40",
                "include_reason": "currency_amount",
                "exclude_reason": "duplicate_identity",
                "confidence": 0.61,
            }
        ],
    }


def test_synthesis_artifact_preserves_auditable_ledger_rows() -> None:
    """Artifacts should retain ledger include/exclude decisions for audit."""
    artifact = build_synthesis_artifact(_checkout())

    assert artifact["ledger_rows"] == [
        {
            "fact_id": "currency:0:0",
            "source_group": "answer-1",
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "kind": "currency",
            "value": "120",
            "unit": "USD",
            "label": "bike helmet",
            "raw_span": "$120 on a bike helmet",
            "normalized_identity": "currency:bike helmet:120",
            "include_reason": "currency_amount",
            "exclude_reason": "",
            "confidence": 0.86,
        },
        {
            "fact_id": "currency:2:0",
            "source_group": "answer-4",
            "citation": "eventloom://agent-1/events/4#dddddddddddd",
            "kind": "currency",
            "value": "40",
            "unit": "USD",
            "label": "bike lights",
            "raw_span": "$40 on bike lights",
            "normalized_identity": "currency:bike lights:40",
            "include_reason": "currency_amount",
            "exclude_reason": "duplicate_identity",
            "confidence": 0.61,
        },
    ]


def test_synthesis_artifact_uses_typed_packet_diagnostics() -> None:
    """Artifact payloads should consume the same typed packet shape checkout emits."""
    checkout = _checkout()
    checkout.diagnostics["synthesis"] = {
        "mode": "multi_source_aggregation",
        "answer_candidates": [
            {
                "rank": 1,
                "type": "currency",
                "confidence": 0.91,
                "answer_key": "currency_total_answer",
                "answer": "$145",
                "support_source_ids": ["answer-1", "answer-2"],
                "excluded_source_ids": [],
            }
        ],
        "ledger_rows": [
            {
                "fact_id": "currency:0:0",
                "source_group": "answer-1",
                "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                "kind": "currency",
                "value": "120",
                "unit": "USD",
                "label": "bike helmet",
                "raw_span": "$120 on a bike helmet",
                "normalized_identity": "currency:bike helmet:120",
                "include_reason": "currency_amount",
                "exclude_reason": "",
                "confidence": 0.86,
            }
        ],
    }

    artifact = build_synthesis_artifact(checkout)

    assert artifact["answer_candidates"][0]["confidence"] == 0.91
    assert artifact["ledger_rows"][0]["source_group"] == "answer-1"


def test_synthesis_artifact_preserves_operation_result_metadata() -> None:
    """Artifacts should keep additive operation/result packet metadata for audit."""
    checkout = _checkout()
    checkout.diagnostics["synthesis"] = {
        "typed_packet": {
            "schema_version": "synthesis_packet_v1",
            "operations": [
                {
                    "name": "sum_values",
                    "kind": "currency",
                    "answer_key": "currency_total_answer",
                    "support_source_ids": ["answer-1", "answer-2"],
                }
            ],
            "result": {
                "answer_key": "currency_total_answer",
                "answer": "$145",
                "confidence": 0.91,
            },
            "answer_candidates": [
                {
                    "rank": 1,
                    "type": "currency",
                    "confidence": 0.91,
                    "answer_key": "currency_total_answer",
                    "answer": "$145",
                    "support_source_ids": ["answer-1", "answer-2"],
                    "excluded_source_ids": [],
                }
            ],
            "ledger_rows": [
                {
                    "fact_id": "currency:0:0",
                    "source_group": "answer-1",
                    "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                    "kind": "currency",
                    "value": "120",
                    "include_reason": "currency_amount",
                }
            ],
        }
    }

    artifact = build_synthesis_artifact(checkout)

    assert artifact["operations"] == [
        {
            "name": "sum_values",
            "kind": "currency",
            "answer_key": "currency_total_answer",
            "support_source_ids": ["answer-1", "answer-2"],
        }
    ]
    assert artifact["result"] == {
        "answer_key": "currency_total_answer",
        "answer": "$145",
        "confidence": 0.91,
    }
    assert artifact["answer_candidates"][0]["answer"] == "$145"
    assert artifact["ledger_rows"][0]["source_group"] == "answer-1"


def test_synthesis_artifact_keeps_top_level_operations_result_from_flat_synthesis() -> None:
    """Flat diagnostics.synthesis operation/result fields should persist in artifacts."""
    checkout = _checkout()
    checkout.diagnostics["synthesis"]["operations"] = [
        {
            "name": "sum_values",
            "kind": "currency",
            "answer_key": "currency_total_answer",
            "support_source_ids": ["answer-1", "answer-2"],
        }
    ]
    checkout.diagnostics["synthesis"]["result"] = {
        "answer_key": "currency_total_answer",
        "answer": "$145",
        "confidence": 0.83,
    }

    artifact = build_synthesis_artifact(checkout)

    assert artifact["operations"] == checkout.diagnostics["synthesis"]["operations"]
    assert artifact["result"] == checkout.diagnostics["synthesis"]["result"]
    assert artifact["answer_candidates"] == checkout.diagnostics["synthesis"]["answer_candidates"]
    assert artifact["ledger_rows"] == checkout.diagnostics["synthesis"]["ledger_rows"]


def test_synthesis_artifact_requires_answer_candidates() -> None:
    """Checkout without answer candidates should not produce an empty artifact."""
    checkout = _checkout()
    checkout.diagnostics["synthesis"]["answer_candidates"] = []

    try:
        build_synthesis_artifact(checkout)
    except ValueError as exc:
        assert "answer_candidates" in str(exc)
    else:
        raise AssertionError("expected missing answer_candidates to fail")


def test_synthesis_candidate_event_payload_preserves_support_citations() -> None:
    """Candidate feedback should keep the selected answer, citations, and quality."""
    checkout = _checkout()
    candidate = checkout.diagnostics["synthesis"]["answer_candidates"][0]

    payload = build_synthesis_candidate_event_payload(
        checkout=checkout,
        candidate=candidate,
        outcome="used",
        reason="answer used in final response",
    )

    assert payload == {
        "query": "How much did I spend on bike expenses in total?",
        "outcome": "used",
        "answer_candidate": candidate,
        "quality": {"answerability": "answer_from_memory", "confidence": 0.86},
        "slot_plan": {
            "version": "slot_plan_v1",
            "answer_type": "sum",
            "operation": "sum_values",
            "required_slots": ["source", "numeric"],
        },
        "support_source_ids": ["answer-1", "answer-2"],
        "excluded_source_ids": ["answer-4"],
        "citations": [
            "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "eventloom://agent-1/events/2#bbbbbbbbbbbb",
        ],
        "reason": "answer used in final response",
        "ref": {"name": "HEAD", "target_seq": 12, "target_hash": "a" * 64},
    }


def test_synthesis_candidate_event_payload_requires_checkout_candidate_match() -> None:
    """Candidate feedback should not write outcomes for foreign answer candidates."""
    with pytest.raises(ValueError, match="diagnostics.synthesis.answer_candidates"):
        build_synthesis_candidate_event_payload(
            checkout=_checkout(),
            candidate={
                "rank": 1,
                "type": "currency",
                "answer": "$145",
                "support_source_ids": ["answer-99"],
            },
            outcome="used",
        )


def test_synthesis_evidence_event_payload_preserves_row_and_candidate_context() -> None:
    """Evidence-row feedback should keep row provenance plus synthesis context."""
    checkout = _checkout()
    candidate = checkout.diagnostics["synthesis"]["answer_candidates"][0]
    row = checkout.diagnostics["synthesis"]["ledger_rows"][0]

    payload = build_synthesis_evidence_event_payload(
        checkout=checkout,
        row=row,
        outcome="used",
        candidate=candidate,
        reason="row supported arithmetic",
    )

    assert payload == {
        "query": "How much did I spend on bike expenses in total?",
        "outcome": "used",
        "evidence_row": row,
        "answer_candidate": candidate,
        "quality": {"answerability": "answer_from_memory", "confidence": 0.86},
        "slot_plan": {
            "version": "slot_plan_v1",
            "answer_type": "sum",
            "operation": "sum_values",
            "required_slots": ["source", "numeric"],
        },
        "source_group": "answer-1",
        "fact_id": "currency:0:0",
        "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
        "support_source_ids": ["answer-1", "answer-2"],
        "reason": "row supported arithmetic",
        "ref": {"name": "HEAD", "target_seq": 12, "target_hash": "a" * 64},
    }


def test_synthesis_evidence_event_payload_rejects_anonymous_row() -> None:
    """Evidence-row feedback should name a cited synthesis row."""
    with pytest.raises(ValueError, match="fact_id, source_group, or citation"):
        build_synthesis_evidence_event_payload(
            checkout=_checkout(),
            row={"kind": "currency", "value": "40"},
            outcome="used",
        )


def test_synthesis_outcome_normalization_and_event_types() -> None:
    """Helpful feedback should reinforce usage, while exclusions use evidence audit events."""
    assert normalize_synthesis_outcome("helpful") == "used"
    assert synthesis_outcome_event_type("used") == "memory.synthesis.used"
    assert synthesis_outcome_event_type("excluded") == "memory.evidence.excluded"
