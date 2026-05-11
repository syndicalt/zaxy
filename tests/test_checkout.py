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
