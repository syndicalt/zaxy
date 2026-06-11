"""Tests for shared Memory Checkout policy helpers."""

from __future__ import annotations

from zaxy.checkout import (
    _compact_synthesis_summary,
    _merge_answer_candidates,
    apply_checkout_budget,
    build_checkout_diagnostics,
    build_checkout_guidance,
    build_checkout_quality,
    build_compact_answer_contexts,
    checkout_prompt_sections,
    checkout_stable_prefix_chars,
    format_memory_checkout_prompt,
)
from zaxy.context import Context, render_prompt_sections
from zaxy.core import ContextAssembly, build_memory_checkout
from zaxy.token_budget import estimate_tokens
from zaxy_benchmarks.benchmark import BenchmarkCase, expected_terms_recall


def test_answer_candidate_merge_prioritizes_specific_state_operations() -> None:
    """Specific deterministic operations should outrank generic fallback counts."""
    candidates = _merge_answer_candidates(
        [
            {
                "rank": 1,
                "type": "count",
                "confidence": 0.99,
                "answer_key": "count_answer",
                "answer": "2",
                "support_source_ids": ["count-1", "count-2"],
                "excluded_source_ids": [],
            },
            {
                "rank": 2,
                "type": "numeric_state",
                "confidence": 0.80,
                "answer_key": "numeric_state_answer",
                "answer": "32",
                "support_source_ids": ["state-1", "state-2"],
                "excluded_source_ids": [],
            },
        ]
    )

    assert [candidate["type"] for candidate in candidates] == ["numeric_state", "count"]
    assert candidates[0]["rank"] == 1
    assert candidates[0]["answer"] == "32"


def test_answer_candidate_merge_drops_fully_excluded_support() -> None:
    """Candidates with no admissible cited support should not become primary answers."""
    candidates = _merge_answer_candidates(
        [
            {
                "rank": 1,
                "type": "date_interval",
                "confidence": 0.83,
                "answer_key": "date_interval_answer",
                "answer": "7 days. 8 days (including the last day) is also acceptable.",
                "support_source_ids": ["source-a", "source-b"],
                "excluded_source_ids": ["source-a", "source-b", "query-temporal-anchor"],
            },
            {
                "rank": 2,
                "type": "date_interval",
                "confidence": 0.81,
                "answer_key": "date_interval_answer",
                "answer": "21 days. 22 days (including the last day) is also acceptable.",
                "support_source_ids": ["source-a", "source-b"],
                "excluded_source_ids": [],
            },
        ]
    )

    assert len(candidates) == 1
    assert candidates[0]["rank"] == 1
    assert candidates[0]["answer"].startswith("21 days")


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


def test_broader_profile_checkout_diagnostics_are_first_class() -> None:
    fixtures = {
        "support": "Customer ticket report has cited impact severity and a documented workaround resolution.",
        "product": "Roadmap signal from customer feedback includes tradeoff, experiment outcome, and customer promise.",
        "sales": "Buyer account stakeholder recorded commitment, next step followup, objection, renewal blocker, and budget risk.",
        "legal": "Exact quote from clause section is approved by counsel authority with effective date and deadline.",
        "executive": "Executive decision approved strategic exception with owner, source, risk metric, market trend, and accountable sponsor.",
    }

    for profile, content in fixtures.items():
        current_facts = [
            {
                "content": content,
                "source": "keyword",
                "score": 0.82,
                "citation": f"eventloom://agent-1/events/{profile}#aaaaaaaaaaaa",
                "valid_from": "2026-06-02T12:00:00Z",
                "valid_to": None,
                "source_lane": "graph",
                "entity_name": f"{profile} fixture",
                "entity_type": "memory",
            }
        ]
        retention = {"policy": "current_only", "superseded_contexts_excluded": 0}
        diagnostics = build_checkout_diagnostics(
            query=f"{profile} checkout fixture",
            purpose=profile,
            source_lanes={"graph": 1},
            current_facts=current_facts,
            evidence=[dict(current_facts[0])],
            retention=retention,
            warnings=[],
        )
        guidance = build_checkout_guidance(
            query=f"{profile} checkout fixture",
            purpose=profile,
            current_facts=current_facts,
            retention=retention,
            evidence=[dict(current_facts[0])],
        )
        quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)

        assert diagnostics["purpose"]["profile"] == profile
        assert diagnostics["purpose_ontology_lens"]["profile"] == profile
        assert diagnostics["evidence_policy"]["satisfied"] is True
        assert guidance["purpose"]["profile"] == profile
        assert guidance["purpose"]["evidence_policy"] == diagnostics["purpose"]["evidence_policy"]
        assert quality["answerability"] == "answer_from_memory"


def test_legal_checkout_blocks_unsupported_paraphrased_obligation() -> None:
    current_facts = [
        {
            "content": "The contract allows redistribution.",
            "source": "keyword",
            "score": 0.82,
            "citation": "eventloom://agent-1/events/legal#aaaaaaaaaaaa",
            "valid_from": "2026-06-02T12:00:00Z",
            "valid_to": None,
            "source_lane": "graph",
            "entity_name": "redistribution obligation",
            "entity_type": "legal_obligation",
        }
    ]
    retention = {"policy": "current_only", "superseded_contexts_excluded": 0}
    diagnostics = build_checkout_diagnostics(
        query="review redistribution obligation",
        purpose="legal",
        source_lanes={"graph": 1},
        current_facts=current_facts,
        evidence=[dict(current_facts[0])],
        retention=retention,
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="review redistribution obligation",
        purpose="legal",
        current_facts=current_facts,
        retention=retention,
        evidence=[dict(current_facts[0])],
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)

    assert diagnostics["evidence_policy"]["satisfied"] is False
    assert diagnostics["evidence_policy"]["mode"] == "block_checkout"
    assert "exact_quote_ref" in diagnostics["evidence_policy"]["missing_requirements"]
    assert quality["answerability"] == "refresh_recommended"
    assert quality["required_action"]["mode"] == "block_checkout"


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
    assert "checkout_answer_candidate=true" in compact[0]
    assert "checkout_synthesis=true" in joined
    assert "date_interval_answer=14 days. 15 days" in joined
    assert "source_id=answer-1" in joined
    assert "source_id=answer-2" in joined
    assert len(joined) < 2700


def test_checkout_compact_contexts_preserve_five_answer_candidates() -> None:
    """Compact checkout should not drop answer-ready candidates before the top-5 window."""
    diagnostics = {
        "evidence_plan": {"mode": "multi_source_aggregation"},
        "evidence_plan_status": {
            "satisfied": True,
            "required_source_groups": 2,
            "observed_source_groups": 5,
        },
        "synthesis": {
            "answer_candidates": [
                {
                    "rank": index,
                    "type": "currency",
                    "confidence": 0.9 - index / 100,
                    "answer_key": "currency_total_answer",
                    "answer": f"${index}",
                    "support_source_ids": [f"answer-{index}"],
                    "excluded_source_ids": [],
                }
                for index in range(1, 6)
            ]
        },
    }

    compact = build_compact_answer_contexts(
        query="How much did I spend on the requested items?",
        current_facts=[],
        evidence=[],
        diagnostics=diagnostics,
        quality={"answerability": "answer_from_memory", "confidence": 0.9},
    )

    joined = "\n".join(compact[:5])
    assert joined.count("checkout_answer_candidate=true") == 5
    assert "answer=$5" in joined


def test_checkout_compacts_source_synthesis_packet_for_temporal_answer() -> None:
    """Typed synthesis packets should trigger compact answer surfaces for non-aggregation modes."""
    synthesis_fact = {
        "content": "\n".join(
                [
                    "zaxy_synthesis_bundle=true",
                    "synthesis_mode=multi_source_aggregation",
                    "query=Which event happened first, the museum visit or the exhibit?",
                    "source_count=2",
                    "temporal_order_answer=First, I visited MoMA, then I attended the Ancient Civilizations exhibit.",
                    "- source_id=answer-1 snippet=I visited MoMA on March 1st.",
                    "- source_id=answer-2 snippet=I attended the Ancient Civilizations exhibit on March 8th.",
                ]
        ),
        "source": "verbatim",
        "score": 1.2,
        "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
        "source_lane": "source_synthesis",
        "synthesis_packet": {
            "schema_version": "synthesis_packet_v1",
            "answer_candidates": [
                {
                        "rank": 1,
                        "type": "temporal_order",
                        "confidence": 0.91,
                        "answer_key": "temporal_order_answer",
                        "answer": "First, I visited MoMA, then I attended the Ancient Civilizations exhibit.",
                        "support_source_ids": ["answer-1", "answer-2"],
                        "excluded_source_ids": [],
                    }
            ],
            "ledger_rows": [],
            "operations": [],
            "result": {
                "answer_key": "temporal_order_answer",
                "answer": "First, I visited MoMA, then I attended the Ancient Civilizations exhibit.",
                "confidence": 0.91,
            },
        },
    }

    diagnostics = build_checkout_diagnostics(
        query="Which event happened first, the museum visit or the exhibit?",
        source_lanes={"source_synthesis": 1},
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    compact = build_compact_answer_contexts(
        query="Which event happened first, the museum visit or the exhibit?",
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        diagnostics=diagnostics,
        quality={"answerability": "answer_from_memory", "confidence": 0.91},
    )

    assert diagnostics["synthesis"]["mode"] == "source_synthesis"
    assert compact[0].startswith("memory_checkout_compact=true")
    assert "checkout_answer_candidate=true" in compact[0]
    assert "answer=First, I visited MoMA, then I attended the Ancient Civilizations exhibit." in compact[0]


def test_checkout_compact_summary_preserves_direct_numeric_answer() -> None:
    """Compact checkout should keep direct numeric synthesis lines model-visible."""
    synthesis_fact = {
        "content": "\n".join(
            [
                "zaxy_synthesis_bundle=true",
                "synthesis_mode=multi_source_aggregation",
                "query=How many bereavement sessions have I completed so far?",
                "source_count=1",
                "candidate_rank=1 candidate_type=direct_numeric_value candidate_confidence=0.84",
                "candidate_support=answer-1",
                "direct_numeric_answer=five",
                "direct_numeric_source_id=answer-1",
                "- source_id=answer-1 snippet=I just finished my fifth bereavement counseling session.",
            ]
        ),
        "source": "verbatim",
        "score": 1.2,
        "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
        "source_lane": "source_synthesis",
    }
    diagnostics = build_checkout_diagnostics(
        query="How many bereavement sessions have I completed so far?",
        source_lanes={"source_synthesis": 1},
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )

    compact = build_compact_answer_contexts(
        query="How many bereavement sessions have I completed so far?",
        current_facts=[synthesis_fact],
        evidence=[synthesis_fact],
        diagnostics=diagnostics,
        quality={"answerability": "answer_from_memory", "confidence": 0.91},
    )

    joined = "\n".join(compact)
    assert "checkout_synthesis=true" in joined
    assert "direct_numeric_answer=five" in joined
    assert "direct_numeric_source_id=answer-1" in joined


def test_checkout_builds_preference_answer_candidate_from_cited_evidence() -> None:
    """Preference questions should expose answer-ready cited checkout candidates."""
    current_facts = [
        {
            "content": (
                "longmemeval_session_id=answer-1 user: I am interested in recent "
                "research papers and conferences that focus on artificial intelligence "
                "in healthcare, especially deep learning for medical image analysis."
            ),
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        }
    ]
    query = "What kind of AI topics would the user prefer suggestions about?"
    diagnostics = build_checkout_diagnostics(
        query=query,
        source_lanes={"verbatim": 1},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query=query,
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    quality = build_checkout_quality(diagnostics=diagnostics, guidance=guidance)
    compact = build_compact_answer_contexts(
        query=query,
        current_facts=current_facts,
        evidence=current_facts,
        diagnostics=diagnostics,
        quality=quality,
    )
    prompt = format_memory_checkout_prompt(
        query=query,
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality=quality,
        guidance=guidance,
        diagnostics=diagnostics,
    )

    synthesis = diagnostics["synthesis"]
    assert synthesis["mode"] == "preference_profile"
    assert synthesis["answer_candidates"][0]["type"] == "preference"
    assert "The user would prefer" in synthesis["answer_candidates"][0]["answer"]
    assert "artificial intelligence in healthcare" in synthesis["answer_candidates"][0]["answer"]
    assert synthesis["ledger_rows"][0]["kind"] == "preference"
    assert synthesis["operations"][0]["name"] == "select_preference_profile"
    assert "checkout_answer_candidate=true" in compact[0]
    assert "answer=The user would prefer" in compact[0]
    assert "For preference questions" in prompt


def test_preference_answer_candidate_matches_longmemeval_preference_surface() -> None:
    """Preference candidates should expose query-shaped first-sentence answers."""
    current_facts = [
        {
            "content": (
                "longmemeval_session_id=answer-dl user: Can you give me an overview "
                "of recent advancements in this field of deep learning for medical "
                "image analysis? assistant: Here is a summary of recent advancements "
                "in explainable AI for medical image analysis."
            ),
            "source": "verbatim",
            "score": 0.91,
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "source_lane": "verbatim",
        }
    ]
    query = "Can you recommend some recent publications or conferences that I might find interesting?"
    expected = (
        "The user would prefer suggestions related to recent research papers, articles, or conferences "
        "that focus on artificial intelligence in healthcare, particularly those that involve deep "
        "learning for medical image analysis. They would not be interested in general AI topics or "
        "those unrelated to healthcare."
    )
    diagnostics = build_checkout_diagnostics(
        query=query,
        source_lanes={"verbatim": 1},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    compact = build_compact_answer_contexts(
        query=query,
        current_facts=current_facts,
        evidence=current_facts,
        diagnostics=diagnostics,
        quality={"answerability": "answer_from_memory", "confidence": 0.9},
    )
    candidate = diagnostics["synthesis"]["answer_candidates"][0]["answer"]

    assert "recent research papers, articles, or conferences" in candidate
    assert "artificial intelligence in healthcare" in candidate
    assert "deep learning for medical image analysis" in candidate
    assert expected_terms_recall(
        BenchmarkCase(
            name="preference",
            query=query,
            expected_terms=(expected,),
            identity_terms=("answer-dl",),
        ),
        compact,
    ) == 1.0


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
    assert "checkout_answer_candidate=true" in compact[0]
    assert "answer=I attended two weddings. The couples were Rachel and Mike, and Emily and Sarah." in compact[0]
    assert "checkout_fact=true" in "\n".join(compact[:5])
    assert "longmemeval_session_id=answer-1" in "\n".join(compact[:5])
    assert "memory_checkout=true" in compact[0]
    assert not any(
        context.startswith("memory_checkout_compact=true\nmemory_checkout=true\nquery=")
        and "checkout_evidence_group=true" not in context
        and "checkout_fact=true" not in context
        and "checkout_synthesis=true" not in context
        and "checkout_answer_candidate=true" not in context
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


def test_checkout_reports_metacognition_as_non_authoritative_diagnostics() -> None:
    """Metacognitive state should be visible without becoming fact authority."""
    current_facts = [
        {
            "content": "Which backend caused the latency spike?",
            "entity_name": "metacognition:unknown:" + "1" * 24,
            "entity_type": "known_unknown",
            "source": "graph",
            "citation": "eventloom://agent-1/events/11#aaaaaaaaaaaa",
            "metadata": {
                "event_type": "metacognition.unknown.recorded",
                "status": "open",
                "authority_status": "non_authoritative",
            },
        },
        {
            "content": "Projection stale caused failure",
            "entity_name": "metacognition:confidence:" + "2" * 24,
            "entity_type": "confidence_assessment",
            "source": "graph",
            "citation": "eventloom://agent-1/events/12#bbbbbbbbbbbb",
            "metadata": {
                "event_type": "metacognition.confidence.assessed",
                "claim": "Projection stale caused failure",
                "confidence": 0.42,
                "conflict_count": 1,
                "requires_reverify": True,
                "authority_status": "non_authoritative",
            },
        },
        {
            "content": "Projection stale caused failure",
            "entity_name": "metacognition:conflict:" + "3" * 24,
            "entity_type": "conflict_cluster",
            "source": "graph",
            "citation": "eventloom://agent-1/events/13#cccccccccccc",
            "metadata": {
                "event_type": "metacognition.conflict.clustered",
                "resolution_status": "unresolved",
                "authority_status": "non_authoritative",
            },
        },
        {
            "content": "Re-check projection latency cause",
            "entity_name": "metacognition:reverify:" + "4" * 24,
            "entity_type": "reverify_request",
            "source": "graph",
            "citation": "eventloom://agent-1/events/14#dddddddddddd",
            "metadata": {
                "event_type": "metacognition.reverify.requested",
                "status": "open",
                "priority": "high",
                "authority_status": "non_authoritative",
            },
        },
    ]

    diagnostics = build_checkout_diagnostics(
        query="Why did projection latency spike?",
        source_lanes={"graph": 4},
        current_facts=current_facts,
        evidence=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="Why did projection latency spike?",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts,
    )
    prompt = format_memory_checkout_prompt(
        query="Why did projection latency spike?",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts,
        quality={"answerability": "answer_from_memory", "confidence": 0.75},
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["metacognition"] == {
        "context_count": 4,
        "unknown_count": 1,
        "open_unknown_count": 1,
        "confidence_assessment_count": 1,
        "low_confidence_count": 1,
        "conflict_cluster_count": 1,
        "unresolved_conflict_count": 1,
        "reverify_needed_count": 4,
        "authority_status": "non_authoritative",
    }
    assert any("known unknowns require re-verification" in item for item in guidance["trust"])
    assert any("confidence assessments as trajectory evidence, not truth" in item for item in guidance["ignore"])
    assert "Metacognition: contexts=4" in prompt
    assert "authority=non_authoritative" in prompt


def test_checkout_reports_procedural_memory_planning_diagnostics() -> None:
    """Procedural memory should expose planning buckets and avoid/review signals."""
    current_facts = [
        {
            "content": "Procedure: validate release",
            "entity_type": "skill_version",
            "source": "skill_memory",
            "citation": "eventloom://agent-1/events/1#aaaaaaaaaaaa",
            "metadata": {
                "status": "validated",
                "procedure": ["Run release tests."],
                "authority_status": "non_authoritative",
            },
        },
        {
            "content": "Procedure: pending validation",
            "entity_type": "skill_version",
            "source": "skill_memory",
            "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            "metadata": {"status": "pending", "procedure": ["Wait for review."]},
        },
        {
            "content": "Procedure: old rollback candidate",
            "entity_type": "skill_version",
            "source": "skill_memory",
            "citation": "eventloom://agent-1/events/3#cccccccccccc",
            "metadata": {
                "status": "contradicted",
                "procedure": ["Use stale workaround."],
                "rollback": "Return to v1 procedure.",
                "contradiction_reason": "Failed release validation.",
            },
        },
        {
            "content": "Procedure: uncited instruction",
            "entity_type": "skill_version",
            "source": "skill_memory",
            "metadata": {"status": "accepted", "procedure": ["Do not use without citation."]},
        },
    ]

    diagnostics = build_checkout_diagnostics(
        query="Plan the release validation",
        source_lanes={"skill": 4},
        current_facts=current_facts,
        evidence=current_facts[:3],
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        warnings=[],
    )
    guidance = build_checkout_guidance(
        query="Plan the release validation",
        current_facts=current_facts,
        retention={"policy": "current_only", "superseded_contexts_excluded": 0},
        evidence=current_facts[:3],
    )
    prompt = format_memory_checkout_prompt(
        query="Plan the release validation",
        assembly_prompt="# Active Memory Working Set",
        current_facts=current_facts,
        evidence=current_facts[:3],
        quality={"answerability": "answer_from_memory", "confidence": 0.82},
        guidance=guidance,
        diagnostics=diagnostics,
    )

    assert diagnostics["procedural_memory"] == {
        "context_count": 4,
        "applicable_count": 1,
        "diagnostic_count": 1,
        "excluded_count": 2,
        "rollback_candidate_count": 1,
        "contradiction_count": 1,
        "excluded_reasons": {"contradicted_status": 1, "missing_citation": 1},
        "authority_status": "non_authoritative",
    }
    assert any("procedures as planning guidance" in item for item in guidance["trust"])
    assert any("rollback or contradiction" in item for item in guidance["ignore"])
    assert "Procedural memory: contexts=4" in prompt
    assert "excluded_reasons=contradicted_status=1, missing_citation=1" in prompt


def test_compact_synthesis_summary_preserves_absence_answer_guidance() -> None:
    """Compacted absence contexts should keep answer-ready target and contrast fields."""
    summary = _compact_synthesis_summary(
        "\n".join(
            [
                "zaxy_absence_check=true",
                "synthesis_mode=absence_check",
                "query=What is the name of my hamster?",
                "not_mentioned_candidate=hamster",
                "known_related_evidence=cat Luna",
                (
                    "answer_guidance=The information provided is not enough. "
                    "You did not mention this information. You did not mention hamster. "
                    "You mentioned cat Luna, but not hamster."
                ),
                "- source_id=answer-1 citation=eventloom://agent-1/events/1#abc snippet=I mentioned my cat Luna.",
            ]
        )
    )

    assert "not_mentioned_candidate=hamster" in summary
    assert "known_related_evidence=cat Luna" in summary
    assert "answer_guidance=The information provided is not enough." in summary


def test_compact_synthesis_summary_preserves_assistant_recall_answer() -> None:
    """Assistant recall answers should survive compact checkout truncation."""
    summary = _compact_synthesis_summary(
        "\n".join(
            [
                "zaxy_synthesis_bundle=true",
                "synthesis_mode=multi_source_aggregation",
                "week_interval_answer=Seven weeks",
                "assistant_recall_answer=Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays.",
                "assistant_recall_source_id=answer-1",
            ]
        )
    )

    assert "assistant_recall_answer=Admon was assigned to the 8 am - 4 pm (Day Shift) on Sundays." in summary
    assert "assistant_recall_source_id=answer-1" in summary


def _tiered_assembly_prompt() -> str:
    return "\n".join(
        [
            "# Active Memory Working Set",
            "- decision: Deploy uses blue-green rollout (eventloom://agent-1/events/3#cccccccccccc)",
            "",
            "# Recent Events",
            "[3] decision.recorded by assistant",
            "Deploy uses blue-green rollout",
            "",
            "# Retrieved Context",
            "- Deploy uses blue-green rollout (eventloom://agent-1/events/3#cccccccccccc)",
        ]
    )


def _mixed_tier_assembly(*, extra_skill: bool = False) -> ContextAssembly:
    """Build an assembly with consolidated (skill), session, and volatile content."""
    contexts = [
        Context(
            content="Deploy uses blue-green rollout",
            source="keyword",
            score=0.9,
            metadata={"citation": "eventloom://agent-1/events/3#cccccccccccc"},
        ),
        Context(
            content="release checklist skill for deploy",
            source="keyword",
            score=0.8,
            metadata={
                "entity_type": "skill_version",
                "skill_id": "release-checklist",
                "version": "2",
                "status": "validated",
                "summary": "Release checklist for deploy",
                "procedure": ["run tests", "tag release"],
                "applicability": ["deploy"],
                "citation": "eventloom://agent-1/events/2#bbbbbbbbbbbb",
            },
        ),
    ]
    if extra_skill:
        contexts.append(
            Context(
                content="hotfix rollback skill for deploy",
                source="keyword",
                score=0.7,
                metadata={
                    "entity_type": "skill_version",
                    "skill_id": "hotfix-rollback",
                    "version": "1",
                    "status": "validated",
                    "summary": "Hotfix rollback for deploy",
                    "procedure": ["revert release"],
                    "applicability": ["deploy"],
                    "citation": "eventloom://agent-1/events/4#dddddddddddd",
                },
            )
        )
    return ContextAssembly(
        session_id="agent-1",
        prompt=_tiered_assembly_prompt(),
        contexts=contexts,
        replay_event_count=1,
    )


def test_checkout_prompt_renders_stability_tiers_in_order() -> None:
    """Consolidated skills render first, session state second, query-specific last."""
    checkout = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly())
    prompt = checkout.prompt

    expected_order = [
        "# Memory Checkout",
        "## Applicable Skills",
        "# Active Memory Working Set",
        "# Recent Events",
        "Query: how do we deploy?",
        "## Current Facts",
        "## Evidence",
        "## Checkout Quality",
        "## Checkout Guidance",
        "## Checkout Diagnostics",
        "# Retrieved Context",
    ]
    positions = [prompt.index(marker) for marker in expected_order]
    assert positions == sorted(positions)
    assert prompt.startswith("# Memory Checkout")


def test_checkout_prompt_sections_round_trip_through_render() -> None:
    """Splitting a rendered checkout prompt must reproduce it byte for byte."""
    checkout = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly())

    sections = checkout_prompt_sections(checkout.prompt)

    assert render_prompt_sections(sections) == checkout.prompt
    kinds = [section.kind for section in sections]
    assert kinds.index("applicable_skills") < kinds.index("working_set")
    assert kinds.index("working_set") < kinds.index("checkout_query")


def test_repeated_checkouts_share_byte_identical_stable_prefix() -> None:
    """With no intervening append, the consolidated prefix is byte-identical."""
    first = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly())
    second = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly())

    prefix_chars = checkout_stable_prefix_chars(first.prompt)
    assert prefix_chars > len("# Memory Checkout")
    assert first.prompt == second.prompt
    assert first.prompt[:prefix_chars] == second.prompt[:prefix_chars]


def test_stable_prefix_is_query_independent() -> None:
    """The consolidated prefix contains no query text, so prompt caches can hit."""
    first = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly())
    second = build_memory_checkout(query="deploy rollout status?", assembly=_mixed_tier_assembly())

    prefix_chars = checkout_stable_prefix_chars(first.prompt)
    assert prefix_chars == checkout_stable_prefix_chars(second.prompt)
    assert first.prompt[:prefix_chars] == second.prompt[:prefix_chars]
    assert first.prompt != second.prompt


def test_appending_consolidated_memory_changes_stable_prefix() -> None:
    """A new accepted skill invalidates the consolidated prefix by construction."""
    before = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly())
    after = build_memory_checkout(
        query="how do we deploy?",
        assembly=_mixed_tier_assembly(extra_skill=True),
    )

    before_prefix = before.prompt[: checkout_stable_prefix_chars(before.prompt)]
    after_prefix = after.prompt[: checkout_stable_prefix_chars(after.prompt)]
    assert before_prefix != after_prefix
    assert "hotfix-rollback" in after_prefix


def test_consolidated_skills_render_with_stable_sort_keys() -> None:
    """Consolidated skill rows are canonically ordered for cache stability."""
    checkout = build_memory_checkout(
        query="how do we deploy?",
        assembly=_mixed_tier_assembly(extra_skill=True),
    )

    prefix = checkout.prompt[: checkout_stable_prefix_chars(checkout.prompt)]
    assert prefix.index("hotfix-rollback") < prefix.index("release-checklist")


def test_apply_checkout_budget_without_budget_only_adds_stable_prefix_chars() -> None:
    """Callers that pass no budget see identical content plus the cache diagnostic."""
    payload = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly()).to_dict()
    original_prompt = payload["prompt"]
    original_token_efficiency = dict(payload["token_efficiency"])

    result = apply_checkout_budget(payload, max_tokens=None)

    assert result["prompt"] == original_prompt
    assert result["token_efficiency"] == original_token_efficiency
    diagnostics = result["diagnostics"]
    assert diagnostics["stable_prefix_chars"] == checkout_stable_prefix_chars(original_prompt)
    assert "budget_requested" not in diagnostics
    assert "budget_used" not in diagnostics
    assert "elided" not in diagnostics


def test_apply_checkout_budget_zero_budget_keeps_trust_contract_only() -> None:
    """A zero budget yields the mandatory header and trust-contract sections."""
    payload = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly()).to_dict()

    result = apply_checkout_budget(payload, max_tokens=0)

    prompt = result["prompt"]
    assert prompt.startswith("# Memory Checkout")
    assert "Query: how do we deploy?" in prompt
    assert "## Checkout Quality" in prompt
    assert "## Checkout Guidance" in prompt
    assert "## Current Facts" not in prompt
    assert "# Recent Events" not in prompt
    diagnostics = result["diagnostics"]
    assert diagnostics["budget_requested"] == 0
    assert diagnostics["budget_used"] > 0
    assert diagnostics["elided"]["count"] > 0
    assert "current_facts" in diagnostics["elided"]["kinds"]
    for record in diagnostics["elided"]["sections"]:
        assert set(record) == {"section_id", "kind", "estimated_tokens"}


def test_checkout_budget_never_changes_cited_payload_fields() -> None:
    """Citation coverage is budget-invariant: packing trims the prompt only."""
    baseline = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly()).to_dict()
    for budget in (0, 60, 100_000):
        payload = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly()).to_dict()
        result = apply_checkout_budget(payload, max_tokens=budget)
        assert result["current_facts"] == baseline["current_facts"]
        assert result["evidence"] == baseline["evidence"]
        assert result["provenance"] == baseline["provenance"]
        assert result["diagnostics"]["citation_count"] == baseline["diagnostics"]["citation_count"]
        assert (
            result["diagnostics"]["current_citation_count"]
            == baseline["diagnostics"]["current_citation_count"]
        )


def test_apply_checkout_budget_refreshes_token_efficiency() -> None:
    """A packed prompt re-reports its estimated token footprint."""
    payload = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly()).to_dict()
    unbudgeted_tokens = payload["token_efficiency"]["prompt_tokens"]

    result = apply_checkout_budget(payload, max_tokens=120)

    refreshed = result["token_efficiency"]
    assert refreshed["prompt_tokens"] < unbudgeted_tokens
    assert refreshed["prompt_tokens"] == estimate_tokens(result["prompt"])
    assert refreshed["current_fact_count"] == len(result["current_facts"])


def test_checkout_budget_inclusion_is_monotone() -> None:
    """Raising the checkout budget never elides a previously included section."""
    previous_kinds: set[str] | None = None
    all_kinds: set[str] = set()
    for budget in range(0, 1200, 40):
        payload = build_memory_checkout(query="how do we deploy?", assembly=_mixed_tier_assembly()).to_dict()
        if not all_kinds:
            all_kinds = {section.kind for section in checkout_prompt_sections(payload["prompt"])}
        result = apply_checkout_budget(payload, max_tokens=budget)
        included = all_kinds - set(result["diagnostics"]["elided"]["kinds"])
        if previous_kinds is not None:
            assert previous_kinds <= included
        previous_kinds = included
