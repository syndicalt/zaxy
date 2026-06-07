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
        "rejected_count": 0,
        "conflicted_count": 0,
        "stale_count": 0,
        "superseded_count": 0,
        "valid_to_count": 0,
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
        "Do not treat consolidation candidates as authoritative memory without a separate promotion event."
        in guidance["ignore"]
    )
    assert "Review-pending consolidation candidates still require disposition." in guidance["ignore"]
    assert "Causal context: contexts=1, edges=1, average_trust=0.91" in prompt
    assert "authority=non_authoritative" in prompt
    assert "relations=causal_caused" in prompt
    assert "methods=explicit_outcome_citation_v1" in prompt
    assert "Consolidation candidates: candidates=1, pending=1, accepted=0" in prompt
    assert "types=episode" in prompt


def test_checkout_guidance_marks_accepted_consolidation_candidate_non_authoritative() -> None:
    """Accepted review status is still not authority promotion in alpha.1."""
    current_facts = [
        {
            "content": "accepted consolidation candidate summarizes an episode.",
            "entity_name": "consolidation:episode:" + "a" * 24,
            "entity_type": "consolidation_candidate",
            "citation": "eventloom://agent-1/events/55#bbbbbbbbbbbb",
            "metadata": {
                "candidate_type": "episode",
                "review_status": "accepted",
                "authority_status": "non_authoritative",
            },
        },
    ]

    guidance = build_checkout_guidance(
        query="What should I remember?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )

    assert (
        "Use consolidation candidates as cited summaries that still require review."
        in guidance["trust"]
    )
    assert (
        "Do not treat consolidation candidates as authoritative memory without a separate promotion event."
        in guidance["ignore"]
    )
    assert "Review-pending consolidation candidates still require disposition." not in guidance["ignore"]


def test_checkout_ignores_unsupported_causal_relation_types() -> None:
    """Only registered causal graph relation labels should produce causal diagnostics."""
    current_facts = [
        {
            "content": "unsupported causal label should remain ordinary inferred context.",
            "entity_name": "test failure",
            "entity_type": "outcome",
            "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
            "score_explanation": {
                "inferred_relation_types": ["causal_reward_hack", "causal_unknown"],
                "inference_methods": ["explicit_outcome_citation_v1"],
                "inferred_edge_count": 2,
                "inferred_edge_trust": 0.91,
                "inferred_edge_trust_multiplier": 1.09,
            },
        }
    ]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 1},
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

    assert "causal_context" not in diagnostics
    assert diagnostics["inferred_context"]["context_count"] == 1
    assert "Use causal_context as explanatory memory, not as authoritative state." not in guidance["trust"]
    assert (
        "Do not treat proposed causal edges as accepted facts without review status."
        not in guidance["ignore"]
    )


def test_checkout_reads_flattened_consolidation_candidate_metadata() -> None:
    """Real checkout facts flatten selected consolidation metadata at the top level."""
    current_facts = [
        {
            "content": "consolidation candidate summarizes a checkout episode.",
            "entity_name": "consolidation:episode:" + "a" * 24,
            "entity_type": "consolidation_candidate",
            "candidate_type": "episode",
            "review_status": "pending",
            "authority_status": "non_authoritative",
            "citation": "eventloom://agent-1/events/55#bbbbbbbbbbbb",
        }
    ]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 1},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="What consolidation candidates are pending?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )

    assert diagnostics["consolidation_candidates"] == {
        "candidate_count": 1,
        "candidate_types": ["episode"],
        "pending_count": 1,
        "accepted_count": 0,
        "rejected_count": 0,
        "conflicted_count": 0,
        "stale_count": 0,
        "superseded_count": 0,
        "valid_to_count": 0,
        "authority_status": "non_authoritative",
    }
    assert (
        "Do not treat consolidation candidates as authoritative memory without a separate promotion event."
        in guidance["ignore"]
    )
    assert "Review-pending consolidation candidates still require disposition." in guidance["ignore"]


def test_checkout_diagnostics_count_consolidation_review_and_stale_states() -> None:
    """Alpha.2 checkout should expose review-gated candidate disposition counts."""
    current_facts = [
        {
            "content": "pending episode candidate.",
            "entity_name": "consolidation:episode:" + "a" * 24,
            "entity_type": "consolidation_candidate",
            "candidate_type": "episode",
            "review_status": "pending",
            "authority_status": "non_authoritative",
            "citation": "eventloom://agent-1/events/55#aaaaaaaaaaaa",
        },
        {
            "content": "accepted claim candidate.",
            "entity_name": "consolidation:claim:" + "b" * 24,
            "entity_type": "consolidation_candidate",
            "candidate_type": "claim",
            "review_status": "accepted",
            "authority_status": "non_authoritative",
            "citation": "eventloom://agent-1/events/56#bbbbbbbbbbbb",
        },
        {
            "content": "conflicted procedure candidate.",
            "entity_name": "consolidation:procedure:" + "c" * 24,
            "entity_type": "consolidation_candidate",
            "candidate_type": "procedure",
            "metadata": {
                "review_status": "conflicted",
                "authority_status": "non_authoritative",
                "stale": True,
            },
            "citation": "eventloom://agent-1/events/57#cccccccccccc",
        },
        {
            "content": "rejected episode candidate.",
            "entity_name": "consolidation:episode:" + "d" * 24,
            "entity_type": "consolidation_candidate",
            "candidate_type": "episode",
            "review_status": "rejected",
            "authority_status": "non_authoritative",
            "valid_to": "2026-06-07T12:00:00Z",
            "citation": "eventloom://agent-1/events/58#dddddddddddd",
        },
        {
            "content": "superseded claim candidate.",
            "entity_name": "consolidation:claim:" + "e" * 24,
            "entity_type": "consolidation_candidate",
            "candidate_type": "claim",
            "review_status": "accepted",
            "authority_status": "non_authoritative",
            "superseded_by": "consolidation:claim:" + "f" * 24,
            "citation": "eventloom://agent-1/events/59#eeeeeeeeeeee",
        },
    ]

    diagnostics = build_checkout_diagnostics(
        source_lanes={"graph": 5},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="Which consolidation candidates need review?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="Which consolidation candidates need review?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["consolidation_candidates"] == {
        "candidate_count": 5,
        "candidate_types": ["episode", "claim", "procedure"],
        "pending_count": 1,
        "accepted_count": 2,
        "rejected_count": 1,
        "conflicted_count": 1,
        "stale_count": 1,
        "superseded_count": 1,
        "valid_to_count": 1,
        "authority_status": "non_authoritative",
    }
    assert (
        "Accepted consolidation reviews are dispositions only; they are not authority promotion."
        in guidance["ignore"]
    )
    assert (
        "Stale, conflicted, rejected, or superseded consolidation candidates are not current authoritative memory."
        in guidance["ignore"]
    )
    assert "Consolidation candidates: candidates=5, pending=1, accepted=2" in prompt
    assert "rejected=1" in prompt
    assert "conflicted=1" in prompt
    assert "stale=1" in prompt
    assert "superseded=1" in prompt
    assert "valid_to=1" in prompt
