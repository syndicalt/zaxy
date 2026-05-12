"""Tests for shared Memory Checkout policy helpers."""

from __future__ import annotations

from zaxy.checkout import (
    build_checkout_diagnostics,
    build_checkout_guidance,
    build_checkout_quality,
    format_memory_checkout_prompt,
)


def test_checkout_policy_handles_uncited_current_fact_once_for_core_and_mcp() -> None:
    """Shared policy should drive degraded-state answerability for every interface."""
    current_facts = [
        {
            "content": "Memory Checkout is current.",
            "source": "keyword",
            "score": 0.74,
            "citation": None,
            "valid_from": "2026-05-10T12:00:00Z",
            "valid_to": None,
            "source_lane": "graph",
        }
    ]
    evidence: list[dict[str, object]] = []
    retention = {"policy": "current_only", "superseded_contexts_excluded": 0}
    warnings = ["Checkout contains current facts without Eventloom citations."]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 1},
        current_facts=current_facts,
        evidence=evidence,
        retention=retention,
        warnings=warnings,
    )
    guidance = build_checkout_guidance(
        query="What is current?",
        current_facts=current_facts,
        retention=retention,
        evidence=evidence,
    )
    quality = build_checkout_quality(
        diagnostics=diagnostics,
        guidance=guidance,
    )
    prompt = format_memory_checkout_prompt(
        query="What is current?",
        assembly_prompt="# Active Memory Working Set\n- Memory Checkout is current.",
        current_facts=current_facts,
        evidence=evidence,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics == {
        "source_lanes": {"graph": 1},
        "citation_count": 0,
        "current_citation_count": 0,
        "current_fact_count": 1,
        "superseded_contexts_excluded": 0,
        "warning_count": 1,
        "feedback_recommended": False,
        "feedback_tool": "memory_feedback",
        "feedback_reason": "Reinforce cited context if it materially informed the next response.",
    }
    assert quality == {
        "answerability": "refresh_recommended",
        "confidence": 0.29,
        "reasons": [
            "Retrieved current facts, but they lack Eventloom citations.",
            "Checkout contains warnings that reduce confidence.",
        ],
        "required_action": guidance["recommended_next_call"],
    }
    assert "## Checkout Quality" in prompt
    assert "refresh_recommended" in prompt
    assert "Current citations: 0" in prompt


def test_checkout_diagnostics_summarize_inferred_context_dependency() -> None:
    """Checkout diagnostics should summarize inferred graph-path reliance."""
    current_facts = [
        {
            "content": "Task 7 likely implemented the Memory Checkout decision.",
            "source": "traversal",
            "score": 0.94,
            "citation": "eventloom://agent-1/events/12#aaaaaaaaaaaa",
            "valid_from": "2026-05-10T12:00:00Z",
            "valid_to": None,
            "source_lane": "graph",
            "score_explanation": {
                "inferred_edge_count": 1,
                "inferred_edge_trust": 0.86,
                "inferred_edge_trust_multiplier": 1.08,
                "inferred_edge_method_coverage": 1.0,
                "inferred_edge_source_coverage": 1.0,
                "inferred_edge_evidence_coverage": 1.0,
                "inferred_relation_types": ["likely_implemented_decision"],
                "inference_methods": ["task_completed_decision_citation_v1"],
            },
        },
        {
            "content": "Task 8 has a weak inferred relation.",
            "source": "traversal",
            "score": 0.42,
            "citation": None,
            "valid_from": "2026-05-10T12:05:00Z",
            "valid_to": None,
            "source_lane": "graph",
            "score_explanation": {
                "inferred_edge_count": 1,
                "inferred_edge_trust": 0.0,
                "inferred_edge_trust_multiplier": 0.65,
                "inferred_edge_method_coverage": 0.0,
                "inferred_edge_source_coverage": 0.0,
                "inferred_edge_evidence_coverage": 0.0,
                "inferred_relation_types": ["weak_inferred_relation"],
                "inference_methods": ["unknown"],
            },
        },
    ]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 2},
        current_facts=current_facts,
        evidence=[current_facts[0]],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="What decision did task 7 implement?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=[current_facts[0]],
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="What decision did task 7 implement?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=[current_facts[0]],
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["inferred_context"] == {
        "context_count": 2,
        "current_fact_count": 2,
        "citation_count": 1,
        "edge_count": 2,
        "average_trust": 0.43,
        "average_multiplier": 0.865,
        "method_coverage": 0.5,
        "source_coverage": 0.5,
        "evidence_coverage": 0.5,
        "low_trust_count": 1,
        "relation_types": ["likely_implemented_decision", "weak_inferred_relation"],
        "inference_methods": ["task_completed_decision_citation_v1", "unknown"],
    }
    assert "Checkout depends on inferred graph paths; inspect inferred_context diagnostics." in guidance["trust"]
    assert "Low-trust inferred graph paths were included; treat them as leads, not facts." in guidance["ignore"]
    assert "Checkout includes inferred graph paths." in quality["reasons"]
    assert "Inferred graph context: contexts=2, edges=2, average_trust=0.43" in prompt
    assert "relations=likely_implemented_decision, weak_inferred_relation" in prompt
    assert "methods=task_completed_decision_citation_v1, unknown" in prompt
