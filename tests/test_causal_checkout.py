"""Focused alpha.1 Memory Checkout diagnostics tests."""

from __future__ import annotations

from zaxy.checkout import (
    build_checkout_diagnostics,
    build_checkout_guidance,
    build_checkout_quality,
    format_memory_checkout_prompt,
)


def test_checkout_diagnostics_summarize_causal_and_consolidation_context() -> None:
    """Causal edges and consolidation candidates should be non-authoritative diagnostics."""
    current_facts = [
        {
            "content": "test failure was caused by the missing checkout diagnostic.",
            "entity_name": "test failure",
            "entity_type": "outcome",
            "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
            "score_explanation": {
                "inferred_relation_types": ["causal_caused"],
                "inference_methods": ["explicit_outcome_citation_v1"],
                "inferred_edge_count": 1,
                "inferred_edge_trust": 0.91,
                "inferred_edge_trust_multiplier": 1.09,
            },
        },
        {
            "content": "consolidation candidate summarizes an episode.",
            "entity_name": "consolidation:episode:" + "a" * 24,
            "entity_type": "consolidation_candidate",
            "citation": "eventloom://agent-1/events/55#bbbbbbbbbbbb",
            "metadata": {
                "candidate_type": "episode",
                "review_status": "pending",
                "authority_status": "non_authoritative",
            },
        },
    ]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 2},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )

    assert diagnostics["causal_context"] == {
        "context_count": 1,
        "edge_count": 1,
        "relation_types": ["causal_caused"],
        "methods": ["explicit_outcome_citation_v1"],
        "average_trust": 0.91,
        "authority_status": "non_authoritative",
    }
    assert diagnostics["consolidation_candidates"] == {
        "candidate_count": 1,
        "candidate_types": ["episode"],
        "pending_count": 1,
        "accepted_count": 0,
        "authority_status": "non_authoritative",
    }
    assert diagnostics["inferred_context"]["context_count"] == 1


def test_checkout_guidance_marks_causal_and_pending_consolidation_non_authoritative() -> None:
    """Guidance should keep causal and pending consolidation context explanatory."""
    current_facts = [
        {
            "content": "test failure was caused by the missing checkout diagnostic.",
            "entity_name": "test failure",
            "entity_type": "outcome",
            "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
            "score_explanation": {
                "inferred_relation_types": ["causal_caused"],
                "inference_methods": ["explicit_outcome_citation_v1"],
                "inferred_edge_count": 1,
                "inferred_edge_trust": 0.91,
                "inferred_edge_trust_multiplier": 1.09,
            },
        },
        {
            "content": "consolidation candidate summarizes an episode.",
            "entity_name": "consolidation:episode:" + "a" * 24,
            "entity_type": "consolidation_candidate",
            "citation": "eventloom://agent-1/events/55#bbbbbbbbbbbb",
            "metadata": {
                "candidate_type": "episode",
                "review_status": "pending",
                "authority_status": "non_authoritative",
            },
        },
    ]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 2},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="Why did the checkout test fail?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="Why did the checkout test fail?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert "Use causal_context as explanatory memory, not as authoritative state." in guidance["trust"]
    assert (
        "Do not treat proposed causal edges as accepted facts without review status."
        in guidance["ignore"]
    )
    assert (
        "Use consolidation candidates as cited summaries that still require review."
        in guidance["trust"]
    )
    assert (
        "Do not treat review-pending consolidation candidates as authoritative memory."
        in guidance["ignore"]
    )
    assert "Causal context: contexts=1, edges=1, average_trust=0.91" in prompt
    assert "authority=non_authoritative" in prompt
    assert "relations=causal_caused" in prompt
    assert "methods=explicit_outcome_citation_v1" in prompt
    assert "Consolidation candidates: candidates=1, pending=1, accepted=0" in prompt
    assert "types=episode" in prompt
