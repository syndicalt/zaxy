"""Tests for shared Memory Checkout policy helpers."""

from __future__ import annotations

from zaxy.checkout import (
    build_checkout_diagnostics,
    build_checkout_guidance,
    build_checkout_quality,
    build_compact_answer_contexts,
    format_memory_checkout_prompt,
)
from zaxy.context import Context
from zaxy.core import ContextAssembly, build_memory_checkout


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
        "evidence_set": {"groups": []},
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
    assert "No cited evidence was retrieved." in prompt


def test_security_checkout_policy_marks_unsupported_answer_non_actionable() -> None:
    """High-risk purpose evidence failures should change checkout quality and prompt."""
    current_facts = [
        {
            "content": "Credential exposure found in auth config.",
            "source": "keyword",
            "score": 0.82,
            "citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa",
            "valid_from": "2026-06-02T12:00:00Z",
            "valid_to": None,
            "source_lane": "graph",
            "entity_name": "auth credential exposure",
            "entity_type": "security_finding",
        }
    ]
    evidence = [dict(current_facts[0])]
    retention = {"policy": "current_only", "superseded_contexts_excluded": 0}

    diagnostics = build_checkout_diagnostics(
        query="review auth credential exposure",
        purpose="security",
        source_lanes={"graph": 1},
        current_facts=current_facts,
        evidence=evidence,
        retention=retention,
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="review auth credential exposure",
        purpose="security",
        current_facts=current_facts,
        retention=retention,
        evidence=evidence,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="review auth credential exposure",
        assembly_prompt="# Active Memory Working Set\n- Credential exposure found.",
        current_facts=current_facts,
        evidence=evidence,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["purpose_ontology_lens"]["profile"] == "security"
    assert diagnostics["purpose_ontology_lens"]["current_fact_roles"][0]["roles"] == [
        "credential",
        "auth",
    ]
    assert diagnostics["evidence_policy"]["satisfied"] is False
    assert diagnostics["evidence_policy"]["missing_requirements"] == ["mitigation_or_risk_owner"]
    assert quality["answerability"] == "refresh_recommended"
    assert quality["required_action"]["mode"] == "require_refresh"
    assert "Purpose evidence policy is not satisfied" in quality["reasons"][-1]
    assert "Evidence policy failure: Security memory requires mitigation" in prompt
    assert "## Checkout Quality" in prompt
    assert "refresh_recommended" in prompt
    assert "Current citations: 1" in prompt


def test_checkout_purpose_profile_conditions_guidance_and_prompt() -> None:
    """Purpose profiles should make retrieval-time ontology explicit."""
    current_facts = [
        {
            "content": "Release blocker: JWKS cache expires before refresh.",
            "source": "traversal",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/12#aaaaaaaaaaaa",
            "valid_from": "2026-05-10T12:00:00Z",
            "valid_to": None,
            "source_lane": "graph",
        }
    ]
    purpose = {
        "profile": "review",
        "task": "release-review",
        "expected_action": "approve_or_block",
    }

    diagnostics = build_checkout_diagnostics(
        query="current release risk",
        purpose=purpose,
        source_lanes={"graph": 1},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="current release risk",
        purpose=purpose,
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="current release risk",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["purpose"]["profile"] == "review"
    assert diagnostics["purpose"]["task"] == "release-review"
    assert diagnostics["purpose"]["expected_action"] == "approve_or_block"
    assert diagnostics["purpose"]["ontology_lens"] == [
        "risk",
        "regression",
        "missing_test",
        "accepted_decision",
        "blocker",
    ]
    assert guidance["purpose"]["evidence_policy"] == "cited_current_facts_required"
    assert "Use the purpose evidence policy: cited_current_facts_required." in guidance["trust"]
    assert (
        "Applied purpose profile review with evidence policy cited_current_facts_required."
        in quality["reasons"]
    )
    assert "## Purpose Profile" in prompt
    assert "Expected action: approve_or_block" in prompt


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


def test_checkout_guides_multi_source_aggregation() -> None:
    """Aggregation checkout should tell the model to synthesize across sources."""
    current_facts = [
        {
            "content": "answer-1: I attended Rachel and Mike's wedding.",
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "valid_from": "2026-05-10T12:00:00Z",
            "valid_to": None,
            "source_lane": "verbatim",
        },
        {
            "content": "answer-2: I attended Emily and Sarah's wedding.",
            "source": "verbatim",
            "score": 0.9,
            "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            "valid_from": "2026-05-10T12:05:00Z",
            "valid_to": None,
            "source_lane": "verbatim",
        },
    ]
    evidence = current_facts
    diagnostics = build_checkout_diagnostics(
        query="How many weddings did I attend?",
        source_lanes={"verbatim": 2},
        current_facts=current_facts,
        evidence=evidence,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="How many weddings did I attend?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=evidence,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="How many weddings did I attend?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=evidence,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["synthesis"]["mode"] == "multi_source_aggregation"
    assert diagnostics["synthesis"]["citation_count"] == 2
    assert diagnostics["evidence_set"]["status"]["satisfied"] is True
    assert "Query requires multi-source synthesis from cited memory." in quality["reasons"]
    assert guidance["synthesis"]["mode"] == "multi_source_aggregation"
    assert "Group evidence by distinct cited source" in prompt
    assert "Do not answer aggregation questions from a single top memory" in prompt


def test_checkout_blocks_aggregation_when_required_source_groups_are_missing() -> None:
    """Aggregation checkout should not be answerable from one cited source group."""
    current_facts = [
        {
            "content": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        }
    ]
    diagnostics = build_checkout_diagnostics(
        query="How many weddings did I attend?",
        source_lanes={"verbatim": 1},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="How many weddings did I attend?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="How many weddings did I attend?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["evidence_plan_status"] == {
        "required_source_groups": 2,
        "observed_source_groups": 1,
        "satisfied": False,
        "refresh_query": "broader cited evidence for: How many weddings did I attend?",
    }
    assert quality["answerability"] == "refresh_recommended"
    assert quality["required_action"] == {
        "type": "memory_checkout",
        "reason": "Evidence plan requires 2 cited source groups, but checkout has 1.",
        "query": "broader cited evidence for: How many weddings did I attend?",
        "missing_slots": ["source"],
        "suggested_queries": [
            {
                "slot": "source",
                "query": "broader cited evidence for: How many weddings did I attend?",
            }
        ],
    }
    assert "Evidence plan requires 2 cited source groups, but checkout has 1." in quality["reasons"]
    assert "Evidence plan status: observed_source_groups=1, required_source_groups=2, satisfied=False" in prompt


def test_memory_checkout_exposes_evidence_plan_for_aggregation() -> None:
    """Checkout should expose the evidence shape required by the query."""
    assembly = ContextAssembly(
        session_id="agent-1",
        prompt="# Retrieved Context",
        contexts=[
            Context(
                content="session_id=answer-1 I attended Rachel and Mike's wedding.",
                source="verbatim",
                score=0.91,
                metadata={"citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa"},
            ),
            Context(
                content="session_id=answer-2 I attended Emily and Sarah's wedding.",
                source="verbatim",
                score=0.9,
                metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
            ),
            Context(
                content="A low-value uncited graph summary.",
                source="keyword",
                score=0.99,
            ),
        ],
        replay_event_count=0,
    )

    checkout = build_memory_checkout(
        query="How many weddings did I attend?",
        assembly=assembly,
    )

    assert checkout.diagnostics["evidence_plan"] == {
        "mode": "multi_source_aggregation",
        "needs_source_lane": True,
        "source_lane_slots": 8,
        "required_source_groups": 2,
        "promote_cited_sources": True,
        "reasons": ["personal_memory", "aggregation", "aggregation_question"],
    }
    assert checkout.current_facts[0]["citation"] == "eventloom://agent-1/events/1#aaaaaaaaaaaa"
    assert checkout.current_facts[1]["citation"] == "eventloom://agent-1/events/2#bbbbbbbbbbbb"
    assert "Evidence plan: mode=multi_source_aggregation" in checkout.prompt


def test_memory_checkout_exposes_slot_plan_for_numeric_aggregation() -> None:
    """Checkout should expose per-slot retrieval requirements for composed answers."""
    assembly = ContextAssembly(
        session_id="agent-1",
        prompt="# Retrieved Context",
        contexts=[
            Context(
                content="session_id=answer-1 I spent $120 on a bike helmet.",
                source="verbatim",
                score=0.91,
                metadata={"citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa"},
            )
        ],
        replay_event_count=0,
    )

    checkout = build_memory_checkout(
        query="How much did I spend on bike expenses in total?",
        assembly=assembly,
    )

    assert checkout.diagnostics["slot_plan"] == {
        "version": "slot_plan_v1",
        "query": "How much did I spend on bike expenses in total?",
        "answer_type": "sum",
        "operation": "sum_values",
        "required_slots": ["source", "numeric"],
        "optional_slots": ["exact", "semantic"],
        "slots": [
            {
                "name": "source",
                "strategy": "source_citation",
                "required": True,
                "budget": 8,
                "query": "How much did I spend on bike expenses in total?",
            },
            {
                "name": "numeric",
                "strategy": "numeric_value",
                "required": True,
                "kinds": ["currency"],
                "operation": "sum_values",
            },
            {
                "name": "exact",
                "strategy": "exact_terms",
                "required": False,
                "terms": ["much", "spend", "bike", "expenses", "total"],
            },
            {
                "name": "semantic",
                "strategy": "semantic_similarity",
                "required": False,
                "query": "How much did I spend on bike expenses in total?",
            },
        ],
    }
    assert checkout.quality["required_action"] == {
        "type": "memory_checkout",
        "reason": "Evidence plan requires 2 cited source groups, but checkout has 1.",
        "query": "broader cited evidence for: How much did I spend on bike expenses in total?",
        "missing_slots": ["source"],
        "suggested_queries": [
            {
                "slot": "source",
                "query": "broader cited evidence for: How much did I spend on bike expenses in total?",
            }
        ],
    }
    assert "Slot plan: required=source, numeric; optional=exact, semantic" in checkout.prompt
    assert "Missing slots: source" in checkout.prompt


def test_memory_checkout_uses_evidence_selection_for_aggregation() -> None:
    """Production checkout should promote cited source groups before uncited summaries."""
    assembly = ContextAssembly(
        session_id="agent-1",
        prompt="# Retrieved Context",
        contexts=[
            Context(
                content="A high-score uncited summary says there were weddings.",
                source="keyword",
                score=0.99,
            ),
            Context(
                content="session_id=answer-1 I attended Rachel and Mike's wedding.",
                source="verbatim",
                score=0.9,
                metadata={"citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa"},
            ),
            Context(
                content="session_id=answer-2 I attended Emily and Sarah's wedding.",
                source="verbatim",
                score=0.89,
                metadata={"citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb"},
            ),
        ],
        replay_event_count=0,
    )

    checkout = build_memory_checkout(
        query="How many weddings did I attend?",
        assembly=assembly,
    )

    assert [fact["citation"] for fact in checkout.current_facts[:2]] == [
        "eventloom://agent-1/events/1#aaaaaaaaaaaa",
        "eventloom://agent-1/events/2#bbbbbbbbbbbb",
    ]
    assert checkout.current_facts[2]["citation"] is None
    assert checkout.quality["answerability"] == "answer_from_memory"


def test_memory_checkout_exposes_evidence_plan_for_absence() -> None:
    """Checkout should tell the model when cited contrast evidence is required."""
    assembly = ContextAssembly(
        session_id="agent-1",
        prompt="# Retrieved Context",
        contexts=[
            Context(
                content="session_id=answer-1 I mentioned my cat Luna.",
                source="verbatim",
                score=0.88,
                metadata={"citation": "eventloom://agent-1/events/4#cccccccccccc"},
            )
        ],
        replay_event_count=0,
    )

    checkout = build_memory_checkout(
        query="Did I mention my hamster?",
        assembly=assembly,
    )

    assert checkout.diagnostics["evidence_plan"] == {
        "mode": "absence_check",
        "needs_source_lane": True,
        "source_lane_slots": 4,
        "required_source_groups": 1,
        "promote_cited_sources": True,
        "reasons": ["personal_memory", "absence_check"],
    }
    assert checkout.guidance["synthesis"]["mode"] == "absence_check"


def test_checkout_groups_synthesis_evidence_by_source_identity() -> None:
    """Aggregation checkout should expose a source-grouped evidence table."""
    evidence = [
        {
            "content": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        },
        {
            "content": "longmemeval_session_id=answer-1 Reception details mentioned dancing.",
            "source": "verbatim",
            "score": 0.83,
            "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            "source_lane": "verbatim",
        },
        {
            "content": "longmemeval_session_id=answer-2 I attended Emily and Sarah's wedding.",
            "source": "graph",
            "score": 0.89,
            "citation": "eventloom://agent-1/events/3#cccccccccccc",
            "source_lane": "graph",
        },
    ]

    diagnostics = build_checkout_diagnostics(
        query="How many weddings did I attend?",
        source_lanes={"graph": 1, "verbatim": 2},
        current_facts=evidence,
        evidence=evidence,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="How many weddings did I attend?",
        current_facts=evidence,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=evidence,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="How many weddings did I attend?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=evidence,
        evidence=evidence,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    groups = diagnostics["synthesis"]["evidence_groups"]
    assert groups == [
        {
            "source_id": "answer-1",
            "evidence_count": 2,
            "citation_count": 2,
            "citations": [
                "eventloom://agent-1/events/1#aaaaaaaaaaaa",
                "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            ],
            "source_lanes": ["verbatim"],
            "top_score": 0.91,
            "snippet": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
        },
        {
            "source_id": "answer-2",
            "evidence_count": 1,
            "citation_count": 1,
            "citations": ["eventloom://agent-1/events/3#cccccccccccc"],
            "source_lanes": ["graph"],
            "top_score": 0.89,
            "snippet": "longmemeval_session_id=answer-2 I attended Emily and Sarah's wedding.",
        },
    ]
    assert "## Synthesis Evidence" in prompt
    assert "source_id=answer-1" in prompt
    assert "evidence_count=2" in prompt
    assert "eventloom://agent-1/events/1#aaaaaaaaaaaa" in prompt


def test_checkout_exposes_structured_numeric_answer_candidate() -> None:
    """Synthesis bundles should become structured answer candidates in checkout diagnostics."""
    synthesis_fact = {
        "content": "\n".join(
            [
                "zaxy_synthesis_bundle=true",
                "synthesis_mode=multi_source_aggregation",
                "query=How much did I spend on bike expenses in total?",
                "source_count=3",
                "candidate_rank=1 candidate_type=currency candidate_confidence=0.83",
                "candidate_support=answer-1,answer-2,answer-3",
                "currency_values=$120,$40,$25",
                "currency_total_answer=$185",
                "currency_excluded_source_ids=answer-4",
                (
                    'ledger_row={"fact_id":"currency:0:0","source_group":"answer-1",'
                    '"citation":"eventloom://agent-1/events/1#aaaaaaaaaaaa",'
                    '"kind":"currency","value":"120","unit":"USD","label":"helmet",'
                    '"raw_span":"$120 helmet","normalized_identity":"currency:helmet:120",'
                    '"include_reason":"currency_amount","exclude_reason":"","confidence":0.83}'
                ),
                (
                    'ledger_row={"fact_id":"currency:3:0","source_group":"answer-4",'
                    '"citation":"eventloom://agent-1/events/4#dddddddddddd",'
                    '"kind":"currency","value":"40","unit":"USD","label":"helmet",'
                    '"raw_span":"$40 helmet","normalized_identity":"currency:helmet:40",'
                    '"include_reason":"currency_amount","exclude_reason":"duplicate_identity","confidence":0.58}'
                ),
                "- source_id=answer-1 citation=eventloom://agent-1/events/1#aaaaaaaaaaaa snippet=helmet",
            ]
        ),
        "source": "verbatim",
        "score": 0.99,
        "citation": "eventloom://agent-1/events/99#999999999999",
        "source_lane": "source_synthesis",
    }

    diagnostics = build_checkout_diagnostics(
        query="How much did I spend on bike expenses in total?",
        source_lanes={"source_synthesis": 1, "verbatim": 3},
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="How much did I spend on bike expenses in total?",
        current_facts=[synthesis_fact],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=[synthesis_fact],
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="How much did I spend on bike expenses in total?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["synthesis"]["answer_candidates"] == [
        {
            "rank": 1,
            "type": "currency",
            "confidence": 0.83,
            "answer_key": "currency_total_answer",
            "answer": "$185",
            "support_source_ids": ["answer-1", "answer-2", "answer-3"],
            "excluded_source_ids": ["answer-4"],
        }
    ]
    assert diagnostics["synthesis"]["ledger_rows"] == [
        {
            "fact_id": "currency:0:0",
            "source_group": "answer-1",
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "kind": "currency",
            "value": "120",
            "unit": "USD",
            "label": "helmet",
            "raw_span": "$120 helmet",
            "normalized_identity": "currency:helmet:120",
            "include_reason": "currency_amount",
            "exclude_reason": "",
            "confidence": 0.83,
        },
        {
            "fact_id": "currency:3:0",
            "source_group": "answer-4",
            "citation": "eventloom://agent-1/events/4#dddddddddddd",
            "kind": "currency",
            "value": "40",
            "unit": "USD",
            "label": "helmet",
            "raw_span": "$40 helmet",
            "normalized_identity": "currency:helmet:40",
            "include_reason": "currency_amount",
            "exclude_reason": "duplicate_identity",
            "confidence": 0.58,
        },
    ]
    assert "Answer candidate: rank=1, type=currency, answer=$185, confidence=0.83" in prompt
    assert "support=answer-1, answer-2, answer-3" in prompt
    assert prompt.index("## Answer Candidates") < prompt.index("## Current Facts")
    assert prompt.index("## Answer Candidates") < prompt.index("## Evidence")


def test_checkout_prefers_typed_synthesis_packet_over_rendered_text() -> None:
    """Typed packets should survive malformed or partial rendered bundle text."""
    synthesis_fact = {
        "content": "\n".join(
            [
                "zaxy_synthesis_bundle=true",
                "candidate_rank=1 candidate_type=currency candidate_confidence=0.10",
                "currency_total_answer=$999",
            ]
        ),
        "source": "verbatim",
        "score": 0.99,
        "citation": "eventloom://agent-1/events/99#999999999999",
        "source_lane": "source_synthesis",
        "synthesis_packet": {
            "schema_version": "synthesis_packet_v1",
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
        },
    }

    diagnostics = build_checkout_diagnostics(
        query="How much did I spend on bike expenses in total?",
        source_lanes={"source_synthesis": 1},
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )

    assert diagnostics["synthesis"]["answer_candidates"][0]["answer"] == "$145"
    assert diagnostics["synthesis"]["answer_candidates"][0]["confidence"] == 0.91
    assert diagnostics["synthesis"]["ledger_rows"][0]["source_group"] == "answer-1"


def test_checkout_exposes_typed_operation_result_metadata() -> None:
    """Checkout diagnostics should keep additive operation/result packet metadata."""
    synthesis_fact = {
        "content": "zaxy_synthesis_bundle=true",
        "source": "verbatim",
        "score": 0.99,
        "citation": "eventloom://agent-1/events/99#999999999999",
        "source_lane": "source_synthesis",
        "synthesis_packet": {
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
            "ledger_rows": [],
        },
    }

    diagnostics = build_checkout_diagnostics(
        query="How much did I spend on bike expenses in total?",
        source_lanes={"source_synthesis": 1},
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )

    assert diagnostics["synthesis"]["operations"] == [
        {
            "name": "sum_values",
            "kind": "currency",
            "answer_key": "currency_total_answer",
            "support_source_ids": ["answer-1", "answer-2"],
        }
    ]
    assert diagnostics["synthesis"]["result"] == {
        "answer_key": "currency_total_answer",
        "answer": "$145",
        "confidence": 0.91,
    }


def test_checkout_builds_compact_answer_contexts_for_synthesis() -> None:
    """Checkout should expose a small model-facing synthesis surface."""
    current_facts = [
        {
            "content": (
                "zaxy_synthesis_bundle=true\n"
                "synthesis_mode=multi_source_aggregation\n"
                "date_interval_answer=14 days. 15 days (including the last day) is also acceptable.\n"
                "- source_id=answer-1 snippet=Since I started with Rachel on 2/15.\n"
                "- source_id=answer-2 snippet=The house I loved was on March 1st."
            ),
            "source": "verbatim",
            "score": 1.2,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        }
    ]
    evidence = [
        *current_facts,
        {
            "content": (
                "longmemeval_session_id=answer-2 "
                "The house I loved was on March 1st."
            ),
            "source": "verbatim",
            "score": 1.0,
            "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            "source_lane": "verbatim",
        },
    ]
    diagnostics = build_checkout_diagnostics(
        query="How many days did it take to find a house after starting with Rachel?",
        source_lanes={"verbatim": 2},
        current_facts=current_facts,
        evidence=evidence,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    quality = {"answerability": "answer_from_memory", "confidence": 0.95}

    compact = build_compact_answer_contexts(
        query="How many days did it take to find a house after starting with Rachel?",
        current_facts=current_facts,
        evidence=evidence,
        diagnostics=diagnostics,
        quality=quality,
    )

    joined = "\n".join(compact)
    assert compact[0].startswith("memory_checkout_compact=true")
    assert "checkout_synthesis=true" in compact[0]
    assert "date_interval_answer=14 days. 15 days" in joined
    assert "source_id=answer-1" in joined
    assert "source_id=answer-2" in joined
    assert len(joined) < 1800


def test_checkout_compact_contexts_put_evidence_before_control_only_metadata() -> None:
    """Compact checkout should spend top context slots on cited evidence."""
    evidence = [
        {
            "content": "longmemeval_session_id=answer-1 I attended Rachel and Mike's wedding.",
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        },
        {
            "content": "longmemeval_session_id=answer-2 I attended Emily and Sarah's wedding.",
            "source": "verbatim",
            "score": 0.89,
            "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            "source_lane": "verbatim",
        },
    ]
    diagnostics = build_checkout_diagnostics(
        query="How many weddings did I attend?",
        source_lanes={"verbatim": 2},
        current_facts=evidence,
        evidence=evidence,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )

    compact = build_compact_answer_contexts(
        query="How many weddings did I attend?",
        current_facts=evidence,
        evidence=evidence,
        diagnostics=diagnostics,
        quality={"answerability": "answer_from_memory", "confidence": 0.88},
    )

    assert compact[0].startswith("memory_checkout_compact=true")
    assert "checkout_fact=true" in compact[0]
    assert "longmemeval_session_id=answer-1" in compact[0]
    assert "memory_checkout=true" in compact[0]
    assert not any(
        context.startswith("memory_checkout_compact=true\nmemory_checkout=true\nquery=")
        and "checkout_evidence_group=true" not in context
        and "checkout_fact=true" not in context
        and "checkout_synthesis=true" not in context
        for context in compact[:5]
    )


def test_checkout_compact_contexts_keep_fact_snippets_in_top_five() -> None:
    """Evidence group diagnostics should not push answer-bearing facts below top five."""
    current_facts = [
        {
            "content": f"longmemeval_session_id=answer-{index} I attended wedding {index}.",
            "source": "verbatim",
            "score": 1.0 - (index * 0.01),
            "citation": f"eventloom://agent-1/events/{index}#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        }
        for index in range(1, 7)
    ]
    diagnostics = build_checkout_diagnostics(
        query="How many weddings did I attend?",
        source_lanes={"verbatim": len(current_facts)},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )

    compact = build_compact_answer_contexts(
        query="How many weddings did I attend?",
        current_facts=current_facts,
        evidence=current_facts,
        diagnostics=diagnostics,
        quality={"answerability": "answer_from_memory", "confidence": 0.88},
    )

    top_five = "\n".join(compact[:5])
    assert "checkout_fact=true" in top_five
    assert "I attended wedding 1." in top_five


def test_checkout_guides_absence_checks_without_overclaiming() -> None:
    """Absence checkout should distinguish missing evidence from proved absence."""
    current_facts = [
        {
            "content": "The user mentioned cat Luna.",
            "source": "verbatim",
            "score": 0.88,
            "citation": "eventloom://agent-1/events/4#cccccccccccc",
            "valid_from": "2026-05-10T12:00:00Z",
            "valid_to": None,
            "source_lane": "verbatim",
        }
    ]
    diagnostics = build_checkout_diagnostics(
        query="Did I mention my hamster?",
        source_lanes={"verbatim": 1},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="Did I mention my hamster?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    prompt = format_memory_checkout_prompt(
        query="Did I mention my hamster?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["synthesis"]["mode"] == "absence_check"
    assert "Query requires absence checking against cited memory." in quality["reasons"]
    assert guidance["synthesis"]["mode"] == "absence_check"
    assert "Do not treat a missing search hit as proof" in prompt
