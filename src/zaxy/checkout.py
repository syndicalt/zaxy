"""Shared Memory Checkout policy helpers.

This module owns model-facing checkout diagnostics, guidance, quality scoring,
and prompt formatting so every interface exposes the same trust contract.
"""

from __future__ import annotations

import re
from typing import Any

from zaxy.causal import CAUSAL_RELATION_TYPES, causal_relation_to_graph_relation
from zaxy.evidence import build_evidence_set, evaluate_evidence_policy
from zaxy.evidence_candidates import candidate_type_priority, checkout_candidate_projection
from zaxy.purpose import PurposeProfile, purpose_ontology_lens, purpose_profile
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.synthesis_packet import synthesis_packet_from_items

_COMPACT_CONTEXT_LIMIT = 8
_COMPACT_ANSWER_CANDIDATE_LIMIT = 5
_COMPACT_SNIPPET_LIMIT = 500
_CAUSAL_GRAPH_RELATION_TYPES = frozenset(
    causal_relation_to_graph_relation(relation_type) for relation_type in CAUSAL_RELATION_TYPES
)


def build_checkout_diagnostics(
    *,
    query: str | None = None,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
    source_lanes: dict[str, int],
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    retention: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build observable checkout metrics used by quality scoring and prompts."""
    diagnostics = {
        "source_lanes": source_lanes,
        "citation_count": len(evidence),
        "current_citation_count": sum(1 for fact in current_facts if fact.get("citation")),
        "current_fact_count": len(current_facts),
        "superseded_contexts_excluded": retention.get("superseded_contexts_excluded", 0),
        "warning_count": len(warnings),
        "feedback_recommended": bool(evidence),
        "feedback_tool": "memory_feedback",
        "feedback_reason": "Reinforce cited context if it materially informed the next response.",
    }
    purpose_policy = retention.get("purpose_policy")
    if isinstance(purpose_policy, dict):
        diagnostics["purpose_policy"] = purpose_policy
    profile = purpose_profile(purpose)
    if profile.profile != "general":
        diagnostics["purpose"] = profile.to_dict()
    evidence_plan = _checkout_evidence_plan(query)
    if evidence_plan:
        diagnostics["evidence_plan"] = evidence_plan
    slot_plan = _checkout_slot_plan(query)
    if slot_plan:
        diagnostics["slot_plan"] = slot_plan
    inferred_context = _inferred_context_diagnostics(current_facts)
    if inferred_context["context_count"]:
        diagnostics["inferred_context"] = inferred_context
    causal_context = _causal_context_diagnostics(current_facts)
    if causal_context["context_count"]:
        diagnostics["causal_context"] = causal_context
    consolidation_candidates = _consolidation_candidate_diagnostics(current_facts)
    if consolidation_candidates["candidate_count"]:
        diagnostics["consolidation_candidates"] = consolidation_candidates
    evidence_set = build_evidence_set(
        query=query,
        evidence_plan=evidence_plan,
        current_facts=current_facts,
        evidence=evidence,
    )
    diagnostics["evidence_set"] = evidence_set.to_diagnostics()
    ontology_lens = purpose_ontology_lens(profile)
    if ontology_lens.applied:
        diagnostics["purpose_ontology_lens"] = {
            **ontology_lens.to_diagnostics(),
            "current_fact_roles": _purpose_role_matches(ontology_lens, current_facts),
            "evidence_roles": _purpose_role_matches(ontology_lens, evidence),
        }
    evidence_policy = evaluate_evidence_policy(
        profile=profile,
        query=query,
        current_facts=current_facts,
        evidence=evidence,
        evidence_set=evidence_set,
    )
    if evidence_policy is not None:
        diagnostics["evidence_policy"] = evidence_policy.to_diagnostics()
    synthesis = _checkout_synthesis_diagnostics(
        query=query,
        current_facts=current_facts,
        evidence=evidence,
        source_lanes=source_lanes,
        evidence_groups=evidence_set.groups,
    )
    if synthesis:
        diagnostics["synthesis"] = synthesis
    if evidence_set.status:
        diagnostics["evidence_plan_status"] = evidence_set.status
    return diagnostics


def build_checkout_guidance(
    *,
    query: str,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
    current_facts: list[dict[str, Any]],
    retention: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build model-facing trust, ignore, refresh, and feedback guidance."""
    feedback_payloads = [
        payload
        for fact in current_facts
        if (payload := build_checkout_feedback_payload(fact, query, purpose=purpose)) is not None
    ][:3]
    trust = [
        "Use current_facts as the primary working memory for this turn.",
        "Use cited evidence and provenance when making claims about remembered context.",
    ]
    ignore = [
        "Do not treat superseded contexts as current facts.",
        "Do not rely on uncited facts without checking memory again or asking the user.",
    ]
    if not evidence:
        trust.append("Treat this checkout as low-confidence because it has no cited evidence.")
    if retention.get("superseded_contexts_excluded", 0):
        ignore.append("Superseded contexts were excluded from current_facts but remain auditable.")
    purpose_policy = retention.get("purpose_policy")
    if isinstance(purpose_policy, dict) and purpose_policy.get("suppressed_count"):
        ignore.append(
            "Purpose policy suppressed "
            f"{purpose_policy.get('suppressed_count')} retrieved rows before checkout projection."
        )
    profile = purpose_profile(purpose)
    purpose_guidance = _purpose_guidance(profile)
    if purpose_guidance:
        trust.extend(purpose_guidance["trust"])
        ignore.extend(purpose_guidance["ignore"])
    inferred_context = _inferred_context_diagnostics(current_facts)
    if inferred_context["context_count"]:
        trust.append("Checkout depends on inferred graph paths; inspect inferred_context diagnostics.")
    if inferred_context["low_trust_count"]:
        ignore.append("Low-trust inferred graph paths were included; treat them as leads, not facts.")
    causal_context = _causal_context_diagnostics(current_facts)
    if causal_context["context_count"]:
        trust.append("Use causal_context as explanatory memory, not as authoritative state.")
        ignore.append("Do not treat proposed causal edges as accepted facts without review status.")
    consolidation_candidates = _consolidation_candidate_diagnostics(current_facts)
    if consolidation_candidates["candidate_count"]:
        trust.append("Use consolidation candidates as cited summaries that still require review.")
        ignore.append(
            "Do not treat consolidation candidates as authoritative memory without a separate promotion event."
        )
    if consolidation_candidates["pending_count"]:
        ignore.append("Review-pending consolidation candidates still require disposition.")
    synthesis = _checkout_synthesis_guidance(
        query=query,
        current_facts=current_facts,
        evidence=evidence,
    )
    if synthesis:
        trust.extend(synthesis["trust"])
        ignore.extend(synthesis["ignore"])
    return {
        "trust": trust,
        "ignore": ignore,
        "purpose": purpose_guidance,
        "synthesis": synthesis,
        "recommended_next_call": {
            "tool": "memory_checkout",
            "query": f"current decisions, blockers, and next actions for: {query}",
            "reason": (
                "Refresh memory before major follow-up work, after compaction/resume, "
                "or when task scope changes."
            ),
        },
        "feedback": {
            "tool": "memory_feedback",
            "when": "After cited context materially informs a response.",
            "payloads": feedback_payloads,
        },
    }


def build_checkout_quality(
    *,
    diagnostics: dict[str, Any],
    guidance: dict[str, Any],
) -> dict[str, Any]:
    """Build the Memory Checkout answerability signal."""
    current_fact_count = _int_metric(diagnostics.get("current_fact_count"))
    current_citation_count = _int_metric(diagnostics.get("current_citation_count"))
    superseded_excluded = _int_metric(diagnostics.get("superseded_contexts_excluded"))
    warning_count = _int_metric(diagnostics.get("warning_count"))
    reasons: list[str] = []
    if current_fact_count and current_citation_count:
        reasons.append("Retrieved current facts with Eventloom citations.")
    elif current_fact_count:
        reasons.append("Retrieved current facts, but they lack Eventloom citations.")
    else:
        reasons.append("No current facts were retrieved.")
    if superseded_excluded:
        reasons.append("Superseded contexts were excluded from current facts.")
    if warning_count:
        reasons.append("Checkout contains warnings that reduce confidence.")
    purpose = diagnostics.get("purpose")
    if isinstance(purpose, dict):
        reasons.append(
            "Applied purpose profile "
            f"{purpose.get('profile')} with evidence policy {purpose.get('evidence_policy')}."
        )
    purpose_policy = diagnostics.get("purpose_policy")
    if isinstance(purpose_policy, dict) and _int_metric(purpose_policy.get("suppressed_count")):
        reasons.append("Purpose policy suppressed non-matching retrieved rows before projection.")
    inferred_context = diagnostics.get("inferred_context")
    if isinstance(inferred_context, dict) and _int_metric(inferred_context.get("context_count")):
        reasons.append("Checkout includes inferred graph paths.")
    synthesis = diagnostics.get("synthesis")
    if isinstance(synthesis, dict):
        mode = synthesis.get("mode")
        if mode == "multi_source_aggregation":
            reasons.append("Query requires multi-source synthesis from cited memory.")
        elif mode == "absence_check":
            reasons.append("Query requires absence checking against cited memory.")
    evidence_plan_status = diagnostics.get("evidence_plan_status")
    evidence_plan_block: dict[str, Any] | None = None
    if isinstance(evidence_plan_status, dict) and not evidence_plan_status.get("satisfied"):
        required_groups = _int_metric(evidence_plan_status.get("required_source_groups"))
        observed_groups = _int_metric(evidence_plan_status.get("observed_source_groups"))
        reason = (
            f"Evidence plan requires {required_groups} cited source groups, "
            f"but checkout has {observed_groups}."
        )
        reasons.append(reason)
        evidence_plan_block = {
            "type": "memory_checkout",
            "reason": reason,
            "query": str(evidence_plan_status.get("refresh_query", "broader cited evidence")),
            "missing_slots": _missing_slots_for_evidence_status(diagnostics),
            "suggested_queries": _suggested_slot_queries(diagnostics, evidence_plan_status),
        }
    confidence = 0.25
    confidence += min(current_fact_count, 2) * 0.22
    confidence += min(current_citation_count, 2) * 0.28
    if superseded_excluded and current_fact_count:
        confidence += 0.07
    confidence = min(0.95, confidence)
    confidence -= min(0.35, warning_count * 0.18)
    if evidence_plan_block is not None:
        confidence -= 0.25
    evidence_policy = diagnostics.get("evidence_policy")
    evidence_policy_block: dict[str, Any] | None = None
    if isinstance(evidence_policy, dict) and not evidence_policy.get("satisfied"):
        failure_reasons = _text_list(evidence_policy.get("failure_reasons"))
        missing_requirements = _text_list(evidence_policy.get("missing_requirements"))
        suggested_queries = _text_list(evidence_policy.get("suggested_queries"))
        reason = (
            "Purpose evidence policy is not satisfied"
            + (f": {'; '.join(failure_reasons)}" if failure_reasons else ".")
        )
        reasons.append(reason)
        evidence_policy_block = {
            "type": "memory_checkout",
            "reason": reason,
            "mode": evidence_policy.get("mode"),
            "missing_requirements": missing_requirements,
            "suggested_queries": suggested_queries,
            "query": suggested_queries[0] if suggested_queries else "refresh purpose evidence",
        }
        confidence -= 0.3
    confidence = round(max(0.0, confidence), 2)
    recommended_next_call = guidance.get("recommended_next_call")
    required_action = recommended_next_call if isinstance(recommended_next_call, dict) else None
    if not current_fact_count:
        answerability = "ask_user"
        required_action = {
            "type": "ask_user",
            "reason": (
                "No current facts were retrieved; ask the user for the missing context "
                "before answering from memory."
            ),
        }
    elif evidence_policy_block is not None:
        answerability = "refresh_recommended"
        required_action = evidence_policy_block
    elif evidence_plan_block is not None:
        answerability = "refresh_recommended"
        required_action = evidence_plan_block
    elif current_citation_count and not warning_count and confidence >= 0.75:
        answerability = "answer_from_memory"
        required_action = None
    else:
        answerability = "refresh_recommended"
    return {
        "answerability": answerability,
        "confidence": confidence,
        "reasons": reasons,
        "required_action": required_action,
    }


def format_memory_checkout_prompt(
    *,
    query: str,
    assembly_prompt: str,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    quality: dict[str, Any],
    guidance: dict[str, Any],
    diagnostics: dict[str, Any],
) -> str:
    """Format the prompt-ready Memory Checkout contract."""
    lines = ["# Memory Checkout", f"Query: {query}"]
    _append_purpose(lines, diagnostics.get("purpose"))
    _append_answer_candidates(lines, diagnostics.get("synthesis"))
    lines.extend(["", "## Current Facts"])
    compact_contexts = diagnostics.get("compact_contexts")
    if isinstance(compact_contexts, list) and compact_contexts:
        lines = ["# Memory Checkout", f"Query: {query}"]
        _append_purpose(lines, diagnostics.get("purpose"))
        _append_answer_candidates(lines, diagnostics.get("synthesis"))
        lines.extend(["", "## Compact Answer Context"])
        for context in compact_contexts:
            if isinstance(context, str):
                lines.append(f"- {_trim_text(context, 700)}")
        lines.extend(["", "## Current Facts"])
    if current_facts:
        for fact in current_facts:
            citation = f" ({fact['citation']})" if fact.get("citation") else ""
            lines.append(f"- {fact['content']}{citation}")
    else:
        lines.append("- No current facts were retrieved.")
    lines.extend(["", "## Evidence"])
    if evidence:
        for item in evidence:
            lines.append(f"- {item['citation']}: {item['content']}")
    else:
        lines.append("- No cited evidence was retrieved.")
    _append_applicable_skills(lines, diagnostics.get("skills"))
    _append_skill_analytics(lines, diagnostics.get("skill_analytics"))
    lines.extend(["", "## Checkout Quality"])
    lines.append(f"- Answerability: {quality.get('answerability')}")
    lines.append(f"- Confidence: {quality.get('confidence')}")
    for reason in quality.get("reasons", []):
        lines.append(f"- Reason: {reason}")
    _append_required_action(lines, quality.get("required_action"))
    lines.extend(["", "## Checkout Guidance"])
    for item in guidance.get("trust", []):
        lines.append(f"- Trust: {item}")
    for item in guidance.get("ignore", []):
        lines.append(f"- Ignore: {item}")
    purpose_guidance = guidance.get("purpose")
    if isinstance(purpose_guidance, dict):
        lines.extend(["", "## Purpose Guidance"])
        lines.append(f"- Profile: {purpose_guidance.get('profile')}")
        lines.append(f"- Evidence policy: {purpose_guidance.get('evidence_policy')}")
        lines.append(f"- Expected action: {purpose_guidance.get('expected_action')}")
        for lens in _text_list(purpose_guidance.get("ontology_lens")):
            lines.append(f"- Lens: {lens}")
    synthesis = guidance.get("synthesis")
    if isinstance(synthesis, dict):
        lines.extend(["", "## Synthesis Guidance"])
        lines.append(f"- Mode: {synthesis.get('mode')}")
        lines.append(f"- Evidence needed: {synthesis.get('evidence_needed')}")
        for step in _text_list(synthesis.get("steps")):
            lines.append(f"- Step: {step}")
    _append_synthesis_evidence(lines, diagnostics.get("synthesis"))
    recommended_next_call = guidance.get("recommended_next_call")
    if isinstance(recommended_next_call, dict):
        lines.append(
            "- Suggested next call: "
            f"{recommended_next_call.get('tool')}({recommended_next_call.get('query')!r})"
        )
    feedback = guidance.get("feedback")
    if isinstance(feedback, dict) and feedback.get("payloads"):
        lines.append(f"- Feedback: call {feedback.get('tool')} with a listed payload after use.")
    source_lanes = diagnostics.get("source_lanes")
    lines.extend(["", "## Checkout Diagnostics"])
    evidence_plan = diagnostics.get("evidence_plan")
    if isinstance(evidence_plan, dict):
        reasons = _text_list(evidence_plan.get("reasons"))
        reason_text = f", reasons={', '.join(reasons)}" if reasons else ""
        lines.append(
            "- Evidence plan: "
            f"mode={evidence_plan.get('mode')}, "
            f"required_source_groups={evidence_plan.get('required_source_groups')}, "
            f"source_lane_slots={evidence_plan.get('source_lane_slots')}"
            f"{reason_text}"
        )
    evidence_plan_status = diagnostics.get("evidence_plan_status")
    if isinstance(evidence_plan_status, dict):
        lines.append(
            "- Evidence plan status: "
            f"observed_source_groups={evidence_plan_status.get('observed_source_groups')}, "
            f"required_source_groups={evidence_plan_status.get('required_source_groups')}, "
            f"satisfied={evidence_plan_status.get('satisfied')}"
        )
    purpose_lens = diagnostics.get("purpose_ontology_lens")
    if isinstance(purpose_lens, dict):
        lines.append(
            "- Purpose ontology lens: "
            f"profile={purpose_lens.get('profile')}, "
            f"relationship_roles={', '.join(_text_list(purpose_lens.get('relationship_roles')))}"
        )
    evidence_policy = diagnostics.get("evidence_policy")
    if isinstance(evidence_policy, dict):
        lines.append(
            "- Evidence policy: "
            f"profile={evidence_policy.get('profile')}, "
            f"mode={evidence_policy.get('mode')}, "
            f"satisfied={evidence_policy.get('satisfied')}"
        )
        for reason in _text_list(evidence_policy.get("failure_reasons")):
            lines.append(f"- Evidence policy failure: {reason}")
    slot_plan = diagnostics.get("slot_plan")
    if isinstance(slot_plan, dict):
        required_slots = _text_list(slot_plan.get("required_slots"))
        optional_slots = _text_list(slot_plan.get("optional_slots"))
        lines.append(
            "- Slot plan: "
            f"required={', '.join(required_slots) if required_slots else 'none'}; "
            f"optional={', '.join(optional_slots) if optional_slots else 'none'}"
        )
    missing_slots = _text_list(_dict_value(quality.get("required_action"), "missing_slots"))
    if missing_slots:
        lines.append(f"- Missing slots: {', '.join(missing_slots)}")
    lines.append(f"- Source lanes: {_format_source_lanes(source_lanes)}")
    lines.append(f"- Citations: {diagnostics.get('citation_count', 0)}")
    lines.append(f"- Current citations: {diagnostics.get('current_citation_count', 0)}")
    lines.append(f"- Current facts: {diagnostics.get('current_fact_count', 0)}")
    lines.append(
        f"- Superseded contexts excluded: {diagnostics.get('superseded_contexts_excluded', 0)}"
    )
    inferred_context = diagnostics.get("inferred_context")
    if isinstance(inferred_context, dict):
        inferred_line = (
            "- Inferred graph context: "
            f"contexts={inferred_context.get('context_count', 0)}, "
            f"edges={inferred_context.get('edge_count', 0)}, "
            f"average_trust={inferred_context.get('average_trust', 0)}"
        )
        relation_types = _text_list(inferred_context.get("relation_types"))
        if relation_types:
            inferred_line += f", relations={', '.join(relation_types)}"
        inference_methods = _text_list(inferred_context.get("inference_methods"))
        if inference_methods:
            inferred_line += f", methods={', '.join(inference_methods)}"
        lines.append(inferred_line)
    causal_context = diagnostics.get("causal_context")
    if isinstance(causal_context, dict):
        causal_line = (
            "- Causal context: "
            f"contexts={causal_context.get('context_count', 0)}, "
            f"edges={causal_context.get('edge_count', 0)}, "
            f"average_trust={causal_context.get('average_trust', 0)}, "
            f"authority={causal_context.get('authority_status', 'non_authoritative')}"
        )
        relation_types = _text_list(causal_context.get("relation_types"))
        if relation_types:
            causal_line += f", relations={', '.join(relation_types)}"
        methods = _text_list(causal_context.get("methods"))
        if methods:
            causal_line += f", methods={', '.join(methods)}"
        lines.append(causal_line)
    consolidation_candidates = diagnostics.get("consolidation_candidates")
    if isinstance(consolidation_candidates, dict):
        consolidation_line = (
            "- Consolidation candidates: "
            f"candidates={consolidation_candidates.get('candidate_count', 0)}, "
            f"pending={consolidation_candidates.get('pending_count', 0)}, "
            f"accepted={consolidation_candidates.get('accepted_count', 0)}, "
            f"authority={consolidation_candidates.get('authority_status', 'non_authoritative')}"
        )
        candidate_types = _text_list(consolidation_candidates.get("candidate_types"))
        if candidate_types:
            consolidation_line += f", types={', '.join(candidate_types)}"
        lines.append(consolidation_line)
    if diagnostics.get("feedback_recommended"):
        lines.append(
            "- Feedback: call "
            f"{diagnostics.get('feedback_tool', 'memory_feedback')} after using cited context."
        )
    lines.extend(["", assembly_prompt])
    return "\n".join(lines).strip()


def build_compact_answer_contexts(
    *,
    query: str,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    quality: dict[str, Any],
) -> list[str]:
    """Build a compact model-facing checkout surface for answer synthesis."""
    contract = _compact_contract(query=query, diagnostics=diagnostics, quality=quality)
    support_items = [
        *_compact_answer_candidate_items(diagnostics),
        *_compact_synthesis_items(current_facts, evidence),
    ]
    support_items.extend(_compact_fact_items(current_facts, used=len(support_items) + 1))
    support_items.extend(_compact_evidence_group_items(diagnostics))
    if not support_items:
        return [contract]
    return [
        _prepend_compact_contract(contract, support_items[0]),
        *support_items[1:_COMPACT_CONTEXT_LIMIT],
    ]


def _checkout_synthesis_diagnostics(
    *,
    query: str | None,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    source_lanes: dict[str, int],
    evidence_groups: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not query:
        return None
    intent = classify_retrieval_intent(query, limit=max(1, len(current_facts)))
    reasons = set(intent.reasons)
    mode: str | None = None
    if {"aggregation", "aggregation_question"} & reasons:
        mode = "multi_source_aggregation"
    elif "absence_check" in reasons:
        mode = "absence_check"
    elif "preference_profile" in reasons:
        mode = "preference_profile"
    synthesis_packet = synthesis_packet_from_items([*current_facts, *evidence])
    has_synthesis_packet = bool(
        synthesis_packet.answer_candidates
        or synthesis_packet.ledger_rows
        or synthesis_packet.operations
        or synthesis_packet.result
    )
    if mode is None and has_synthesis_packet:
        mode = "source_synthesis"
    if mode is None:
        return None
    citations = {
        citation
        for item in [*current_facts, *evidence]
        if isinstance((citation := item.get("citation")), str) and citation
    }
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "reasons": sorted(reasons),
        "source_lane_slots": intent.source_lane_slots,
        "current_fact_count": len(current_facts),
        "citation_count": len(citations),
        "source_lanes": source_lanes,
        "evidence_groups": evidence_groups,
    }
    candidate_projection = checkout_candidate_projection(
        query,
        [
            str(item.get("content", ""))
            for item in [*current_facts, *evidence]
            if item.get("content") and item.get("citation")
        ],
        limit=max(1, len(current_facts)),
    )
    answer_candidates = _merge_answer_candidates(
        [
            *synthesis_packet.answer_candidates,
            *candidate_projection.answer_candidates,
        ]
    )
    if answer_candidates:
        diagnostics["answer_candidates"] = answer_candidates
    ledger_rows = _merge_dict_rows(
        [
            *synthesis_packet.ledger_rows,
            *candidate_projection.ledger_rows,
        ],
        identity_keys=("fact_id", "source_group", "kind", "value", "label"),
    )
    if ledger_rows:
        diagnostics["ledger_rows"] = ledger_rows
    operations = _merge_dict_rows(
        [
            *synthesis_packet.operations,
            *candidate_projection.operations,
        ],
        identity_keys=("name", "answer_key", "kind"),
    )
    if operations:
        diagnostics["operations"] = operations
    result = synthesis_packet.result or candidate_projection.result
    if result:
        diagnostics["result"] = result
    return diagnostics


def _checkout_synthesis_guidance(
    *,
    query: str,
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    intent = classify_retrieval_intent(query, limit=max(1, len(current_facts)))
    reasons = set(intent.reasons)
    if {"aggregation", "aggregation_question"} & reasons:
        return {
            "mode": "multi_source_aggregation",
            "evidence_needed": "Use every relevant cited memory in the checkout before deriving a count, sum, duration, or list.",
            "steps": [
                "Group evidence by distinct cited source or session before deriving the answer.",
                "Compute the requested count, sum, duration, or list from the grouped evidence.",
                "If the cited source set looks incomplete, call memory_checkout again with a broader aggregation query.",
            ],
            "trust": [
                "For aggregation questions, treat cited memories as inputs to derive an answer rather than expecting one fact to contain the final answer."
            ],
            "ignore": [
                "Do not answer aggregation questions from a single top memory when the checkout contains multiple relevant sources."
            ],
        }
    if "absence_check" in reasons:
        return {
            "mode": "absence_check",
            "evidence_needed": "Use cited positive mentions to decide whether the requested memory is absent or only unsupported by the current checkout.",
            "steps": [
                "Look for cited memories that mention nearby alternatives or the same topic.",
                "Only say the user did not mention something when cited evidence supports the contrast.",
                "If the checkout lacks nearby cited evidence, ask the user or refresh memory instead of asserting absence.",
            ],
            "trust": [
                "For absence checks, rely on cited nearby memories and explicitly distinguish not found from contradicted."
            ],
            "ignore": [
                "Do not treat a missing search hit as proof that the user never mentioned something."
            ],
        }
    if "preference_profile" in reasons:
        return {
            "mode": "preference_profile",
            "evidence_needed": "Use cited user-preference evidence and preserve the distinction between positive preferences and generic non-preferences.",
            "steps": [
                "Extract the user's concrete preference signals from cited memory.",
                "Answer in preference-profile form: what the user would prefer and what they may not prefer.",
                "Do not invent new preferences beyond cited evidence; refresh memory if the profile is underspecified.",
            ],
            "trust": [
                "For preference questions, use cited remembered behavior and stated interests as the support for the preference profile."
            ],
            "ignore": [
                "Do not convert broad assistant suggestions into user preferences unless cited user context supports them."
            ],
        }
    return None


def _merge_answer_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge packet/projection candidates and prefer answer-ready surfaces."""
    ranked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(_drop_dominated_aggregate_candidates(candidates), key=_answer_candidate_sort_key):
        answer = candidate.get("answer")
        if not isinstance(answer, str) or not answer:
            continue
        identity = (str(candidate.get("type", "")), " ".join(answer.casefold().split()))
        if identity in seen:
            continue
        seen.add(identity)
        payload = dict(candidate)
        payload["rank"] = len(ranked) + 1
        ranked.append(payload)
    return ranked


def _drop_dominated_aggregate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        if _aggregate_candidate_key(candidate) is None:
            kept.append(candidate)
            continue
        support_count = len(_text_list(candidate.get("support_source_ids")))
        dominated = any(
            other is not candidate
            and _aggregate_candidate_key(other) == _aggregate_candidate_key(candidate)
            and len(_text_list(other.get("support_source_ids"))) > support_count
            for other in candidates
        )
        if not dominated:
            kept.append(candidate)
    return kept


def _aggregate_candidate_key(candidate: dict[str, Any]) -> tuple[str, str] | None:
    answer_key = str(candidate.get("answer_key") or "")
    if answer_key.endswith("_total_answer") or answer_key in {"count_answer", "count_answer_text"}:
        return str(candidate.get("type", "")), answer_key
    return None


def _answer_candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, int, int, str]:
    confidence = candidate.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, int | float) and not isinstance(confidence, bool) else 0.0
    answer = candidate.get("answer")
    answer_text = answer if isinstance(answer, str) else ""
    answer_key = str(candidate.get("answer_key") or "")
    support_count = len(_text_list(candidate.get("support_source_ids")))
    return (
        candidate_type_priority(candidate),
        -confidence_value,
        -_answer_surface_score(answer_text, answer_key),
        -support_count,
        len(answer_text),
        answer_text,
    )


def _answer_surface_score(answer: str, answer_key: str) -> int:
    """Rank candidates by usefulness as a first-pass model answer."""
    if not answer:
        return 0
    score = 0
    if answer_key.endswith("_text") or answer_key.endswith("_answer"):
        score += 2
    if re.search(r"[A-Za-z]", answer):
        score += 2
    if re.search(r"\b(?:I|The user|You|First|Then|Finally|Yes|No)\b", answer):
        score += 2
    if len(answer.split()) >= 4:
        score += 1
    return score


def _merge_dict_rows(rows: list[dict[str, Any]], *, identity_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        identity = tuple(str(row.get(key, "")) for key in identity_keys)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped


def _compact_synthesis_items(
    current_facts: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for item in [*current_facts, *evidence]:
        content = item.get("content")
        if not isinstance(content, str) or not content:
            continue
        lowered = content.casefold()
        if "zaxy_synthesis_bundle=true" not in lowered and "zaxy_absence_check=true" not in lowered:
            continue
        summary = _compact_synthesis_summary(content)
        if summary in seen:
            continue
        seen.add(summary)
        items.append(
            "\n".join(
                [
                    "checkout_synthesis=true",
                    f"citation={item.get('citation')}",
                    f"source_lane={item.get('source_lane')}",
                    summary,
                ]
            )
        )
    return items


def _compact_answer_candidate_items(diagnostics: dict[str, Any]) -> list[str]:
    synthesis = diagnostics.get("synthesis")
    if not isinstance(synthesis, dict):
        return []
    candidates = synthesis.get("answer_candidates")
    if not isinstance(candidates, list):
        return []
    items: list[str] = []
    for candidate in candidates[:_COMPACT_ANSWER_CANDIDATE_LIMIT]:
        if not isinstance(candidate, dict):
            continue
        answer = candidate.get("answer")
        if not isinstance(answer, str) or not answer:
            continue
        items.append(
            "\n".join(
                [
                    "checkout_answer_candidate=true",
                    f"candidate_type={candidate.get('type')}",
                    f"candidate_rank={candidate.get('rank')}",
                    f"candidate_confidence={candidate.get('confidence')}",
                    f"answer_key={candidate.get('answer_key')}",
                    f"answer={_trim_text(answer, _COMPACT_SNIPPET_LIMIT)}",
                    "support_source_ids=" + ",".join(_text_list(candidate.get("support_source_ids"))),
                    "excluded_source_ids=" + ",".join(_text_list(candidate.get("excluded_source_ids"))),
                ]
            )
        )
    return items


def _compact_contract(
    *,
    query: str,
    diagnostics: dict[str, Any],
    quality: dict[str, Any],
) -> str:
    evidence_plan = diagnostics.get("evidence_plan")
    evidence_status = diagnostics.get("evidence_plan_status")
    return "\n".join(
        [
            "memory_checkout_compact=true",
            "memory_checkout=true",
            f"query={query}",
            f"purpose_profile={_dict_value(diagnostics.get('purpose'), 'profile')}",
            f"answerability={quality.get('answerability')}",
            f"confidence={quality.get('confidence')}",
            f"evidence_plan_mode={_dict_value(evidence_plan, 'mode')}",
            f"evidence_plan_satisfied={_dict_value(evidence_status, 'satisfied')}",
            f"required_source_groups={_dict_value(evidence_status, 'required_source_groups')}",
            f"observed_source_groups={_dict_value(evidence_status, 'observed_source_groups')}",
        ]
    )


def _prepend_compact_contract(contract: str, context: str) -> str:
    return "\n".join([contract, context])


def _compact_synthesis_summary(content: str) -> str:
    lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- source_id="):
            if len(lines) >= 10:
                break
            lines.append(_trim_text(line, _COMPACT_SNIPPET_LIMIT))
            continue
        if any(
            line.startswith(prefix)
            for prefix in (
                "zaxy_synthesis_bundle=",
                "zaxy_absence_check=",
                "synthesis_mode=",
                "query=",
                "source_count=",
                "currency_",
                "minute_",
                "hour_",
                "day_",
                "month_",
                "date_interval_",
                "direct_numeric_",
                "future_age_",
                "age_",
                "page_",
                "issue_",
                "not_mentioned_",
                "known_related_evidence=",
                "answer_guidance=",
                "assistant_recall_",
                "quoted_target_duration_",
            )
        ):
            lines.append(_trim_text(line, _COMPACT_SNIPPET_LIMIT))
    if lines:
        return "\n".join(lines[:12])
    return _trim_text(" ".join(content.split()), _COMPACT_SNIPPET_LIMIT)


def _compact_evidence_group_items(diagnostics: dict[str, Any]) -> list[str]:
    evidence_set = diagnostics.get("evidence_set")
    groups = evidence_set.get("groups") if isinstance(evidence_set, dict) else None
    if not isinstance(groups, list):
        return []
    items: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        citations = _text_list(group.get("citations"))
        citation_text = ",".join(citations)
        items.append(
            "\n".join(
                [
                    "checkout_evidence_group=true",
                    f"source_id={group.get('source_id')}",
                    f"evidence_count={group.get('evidence_count')}",
                    f"citation_count={group.get('citation_count')}",
                    f"citations={citation_text}",
                    f"source_lanes={','.join(_text_list(group.get('source_lanes')))}",
                    f"snippet={_trim_text(str(group.get('snippet', '')), _COMPACT_SNIPPET_LIMIT)}",
                ]
            )
        )
    return items


def _compact_fact_items(
    current_facts: list[dict[str, Any]],
    *,
    used: int,
) -> list[str]:
    remaining = max(0, _COMPACT_CONTEXT_LIMIT - used)
    items: list[str] = []
    seen: set[str] = set()
    for fact in current_facts:
        content = fact.get("content")
        if not isinstance(content, str) or not content:
            continue
        key = " ".join(content.split()).casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(
            "\n".join(
                [
                    "checkout_fact=true",
                    f"citation={fact.get('citation')}",
                    f"source_lane={fact.get('source_lane')}",
                    f"score={fact.get('score')}",
                    f"snippet={_trim_text(content, _COMPACT_SNIPPET_LIMIT)}",
                ]
            )
        )
        if len(items) >= remaining:
            break
    return items


def _trim_text(value: str, limit: int) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _dict_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _checkout_evidence_plan(query: str | None) -> dict[str, object] | None:
    if query is None:
        return None
    from zaxy.retrieval_plan import build_evidence_plan

    return build_evidence_plan(query, limit=10).to_dict()


def _checkout_slot_plan(query: str | None) -> dict[str, object] | None:
    if query is None:
        return None
    from zaxy.retrieval_plan import build_slot_plan

    return build_slot_plan(query, limit=10).to_dict()


def _missing_slots_for_evidence_status(diagnostics: dict[str, Any]) -> list[str]:
    evidence_plan_status = diagnostics.get("evidence_plan_status")
    if not isinstance(evidence_plan_status, dict) or evidence_plan_status.get("satisfied"):
        return []
    missing: list[str] = []
    slot_plan = diagnostics.get("slot_plan")
    required_slots = _text_list(slot_plan.get("required_slots")) if isinstance(slot_plan, dict) else []
    if "source" in required_slots:
        missing.append("source")
    return missing


def _suggested_slot_queries(
    diagnostics: dict[str, Any],
    evidence_plan_status: dict[str, Any],
) -> list[dict[str, str]]:
    refresh_query = str(evidence_plan_status.get("refresh_query", "broader cited evidence"))
    suggestions = [
        {"slot": slot, "query": refresh_query}
        for slot in _missing_slots_for_evidence_status(diagnostics)
    ]
    return suggestions


def build_checkout_feedback_payload(
    fact: dict[str, Any],
    query: str,
    *,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
) -> dict[str, Any] | None:
    """Build a memory_feedback payload for a cited current fact."""
    citation = fact.get("citation")
    if not isinstance(citation, str) or not citation:
        return None
    profile = purpose_profile(purpose)
    entity_name = fact.get("entity_name")
    entity_type = fact.get("entity_type")
    payload: dict[str, Any] = {
        "entity_name": entity_name if isinstance(entity_name, str) and entity_name else fact.get("content"),
        "entity_type": entity_type if isinstance(entity_type, str) and entity_type else "memory",
        "feedback": "used",
        "actor": "assistant",
        "query": query,
        "source": fact.get("source"),
        "score": fact.get("score"),
        "citation": citation,
        "importance": 0.6,
    }
    for key in (
        "authority",
        "authority_scope",
        "coordination_status",
        "finding_id",
        "mission_id",
        "stale",
        "status",
        "worker_id",
    ):
        value = fact.get(key)
        if isinstance(value, str | int | float | bool) and value not in ("", None):
            payload[key] = value
    if profile.profile != "general":
        payload["purpose"] = profile.to_dict()
    return {key: value for key, value in payload.items() if value is not None}


def _purpose_role_matches(
    ontology_lens: Any,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in items:
        text = " ".join(
            str(value)
            for value in (
                item.get("content"),
                item.get("entity_name"),
                item.get("entity_type"),
                item.get("source"),
            )
            if value
        )
        roles = ontology_lens.matched_entity_roles(text)
        if not roles:
            continue
        payload = {
            "id": str(item.get("citation") or item.get("entity_name") or item.get("content") or "")[:160],
            "roles": list(roles),
        }
        if item.get("citation"):
            payload["citation"] = str(item["citation"])
        matches.append(payload)
        if len(matches) >= 8:
            break
    return matches


def _purpose_guidance(profile: PurposeProfile) -> dict[str, Any] | None:
    if profile.profile == "general":
        return None
    payload = profile.to_dict()
    trust = [
        (
            "Apply the purpose profile before treating retrieved material as memory; "
            f"expected action is {profile.expected_action}."
        ),
        f"Use the purpose evidence policy: {profile.evidence_policy}.",
    ]
    if profile.required_evidence:
        trust.append("Required evidence: " + ", ".join(profile.required_evidence) + ".")
    ignore = [f"Suppress for this purpose: {item}." for item in profile.suppress[:3]]
    ignore.extend(profile.warnings)
    payload["trust"] = trust
    payload["ignore"] = ignore
    return payload


def _append_purpose(lines: list[str], purpose: Any) -> None:
    if not isinstance(purpose, dict):
        return
    lines.extend(["", "## Purpose Profile"])
    lines.append(f"- Profile: {purpose.get('profile')}")
    lines.append(f"- Role: {purpose.get('role')}")
    lines.append(f"- Task: {purpose.get('task')}")
    lines.append(f"- Risk: {purpose.get('risk')}")
    lines.append(f"- Time horizon: {purpose.get('time_horizon')}")
    lines.append(f"- Expected action: {purpose.get('expected_action')}")
    lines.append(f"- Evidence policy: {purpose.get('evidence_policy')}")
    lenses = _text_list(purpose.get("ontology_lens"))
    if lenses:
        lines.append(f"- Ontology lens: {', '.join(lenses)}")


def _append_required_action(lines: list[str], required_action: Any) -> None:
    if isinstance(required_action, dict):
        action_type = required_action.get("type")
        if action_type == "ask_user":
            lines.append(f"- Required action: ask_user: {required_action.get('reason')}")
        else:
            lines.append(
                "- Required action: "
                f"{required_action.get('tool')}({required_action.get('query')!r})"
            )
    else:
        lines.append("- Required action: none")


def _append_synthesis_evidence(lines: list[str], synthesis: Any) -> None:
    if not isinstance(synthesis, dict):
        return
    groups = synthesis.get("evidence_groups")
    if not isinstance(groups, list) or not groups:
        return
    lines.extend(["", "## Synthesis Evidence"])
    for group in groups:
        if not isinstance(group, dict):
            continue
        citations = _text_list(group.get("citations"))
        citation_text = ", ".join(citations) if citations else "none"
        source_lanes = _text_list(group.get("source_lanes"))
        lane_text = ", ".join(source_lanes) if source_lanes else "unknown"
        lines.append(
            "- "
            f"source_id={group.get('source_id')}; "
            f"evidence_count={group.get('evidence_count', 0)}; "
            f"citations={citation_text}; "
            f"source_lanes={lane_text}; "
            f"snippet={group.get('snippet', '')}"
        )


def _append_answer_candidates(lines: list[str], synthesis: Any) -> None:
    if not isinstance(synthesis, dict):
        return
    candidates = synthesis.get("answer_candidates")
    if not isinstance(candidates, list) or not candidates:
        return
    lines.extend(["", "## Answer Candidates"])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        support = ", ".join(_text_list(candidate.get("support_source_ids")))
        excluded = ", ".join(_text_list(candidate.get("excluded_source_ids")))
        suffix = f"; support={support}" if support else ""
        if excluded:
            suffix += f"; excluded={excluded}"
        lines.append(
            "- Answer candidate: "
            f"rank={candidate.get('rank')}, "
            f"type={candidate.get('type')}, "
            f"answer={candidate.get('answer')}, "
            f"confidence={candidate.get('confidence')}"
            f"{suffix}"
        )


def _append_applicable_skills(lines: list[str], skills: Any) -> None:
    if not isinstance(skills, dict):
        return
    items = skills.get("items")
    if not isinstance(items, list) or not items:
        return
    lines.extend(["", "## Applicable Skills"])
    for item in items:
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("skill_id") or "unknown").strip()
        version = str(item.get("version") or "1").strip()
        status = str(item.get("status") or "unknown").strip()
        citation = str(item.get("citation") or "").strip()
        suffix = f" ({citation})" if citation else ""
        lines.append(f"- {skill_id} v{version} [{status}]{suffix}")
        for step in _text_list(item.get("procedure"))[:5]:
            lines.append(f"  - {step}")


def _append_skill_analytics(lines: list[str], analytics: Any) -> None:
    if not isinstance(analytics, dict):
        return
    promotions = analytics.get("promotion_candidates")
    rollbacks = analytics.get("rollback_candidates")
    if not isinstance(promotions, list):
        promotions = []
    if not isinstance(rollbacks, list):
        rollbacks = []
    if not promotions and not rollbacks and not analytics.get("contradiction_count"):
        return
    lines.extend(["", "## Skill Analytics"])
    lines.append(
        "- "
        f"outcomes={analytics.get('outcome_count', 0)}; "
        f"contradictions={analytics.get('contradiction_count', 0)}"
    )
    for item in promotions[:3]:
        if not isinstance(item, dict):
            continue
        citation = str(item.get("latest_citation") or "").strip()
        suffix = f"; citation={citation}" if citation else ""
        lines.append(
            "- "
            f"promotion_candidate={item.get('skill_id')} v{item.get('version')}; "
            f"successes={item.get('success_count', 0)}; "
            f"average_success_score={item.get('average_success_score')}"
            f"{suffix}"
        )
    for item in rollbacks[:3]:
        if not isinstance(item, dict):
            continue
        citation = str(item.get("latest_citation") or "").strip()
        suffix = f"; citation={citation}" if citation else ""
        lines.append(
            "- "
            f"rollback_candidate={item.get('skill_id')} v{item.get('version')}; "
            f"reason={item.get('reason')}; "
            f"failures={item.get('failure_count', 0)}"
            f"{suffix}"
        )


def _int_metric(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _inferred_context_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    inferred_items = [
        item
        for item in items
        if _inferred_score_explanation(item).get("inferred_edge_count")
    ]
    context_count = len(inferred_items)
    if context_count == 0:
        return {
            "context_count": 0,
            "current_fact_count": 0,
            "citation_count": 0,
            "edge_count": 0,
            "average_trust": 0.0,
            "average_multiplier": 0.0,
            "method_coverage": 0.0,
            "source_coverage": 0.0,
            "evidence_coverage": 0.0,
            "low_trust_count": 0,
        }
    explanations = [_inferred_score_explanation(item) for item in inferred_items]
    return {
        "context_count": context_count,
        "current_fact_count": context_count,
        "citation_count": sum(1 for item in inferred_items if item.get("citation")),
        "edge_count": sum(_int_metric(exp.get("inferred_edge_count")) for exp in explanations),
        "average_trust": _round_metric(_average_metric(explanations, "inferred_edge_trust")),
        "average_multiplier": _round_metric(
            _average_metric(explanations, "inferred_edge_trust_multiplier"),
            digits=3,
        ),
        "method_coverage": _round_metric(_average_metric(explanations, "inferred_edge_method_coverage")),
        "source_coverage": _round_metric(_average_metric(explanations, "inferred_edge_source_coverage")),
        "evidence_coverage": _round_metric(_average_metric(explanations, "inferred_edge_evidence_coverage")),
        "low_trust_count": sum(
            1
            for exp in explanations
            if _float_metric(exp.get("inferred_edge_trust")) < 0.5
        ),
        "relation_types": _unique_explanation_texts(explanations, "inferred_relation_types"),
        "inference_methods": _unique_explanation_texts(explanations, "inference_methods"),
    }


def _inferred_score_explanation(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("score_explanation")
    return value if isinstance(value, dict) else {}


def _causal_context_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    causal_items: list[tuple[dict[str, Any], dict[str, Any], list[str]]] = []
    for item in items:
        explanation = _inferred_score_explanation(item)
        relation_types = [
            relation_type
            for relation_type in _text_list(explanation.get("inferred_relation_types"))
            if relation_type in _CAUSAL_GRAPH_RELATION_TYPES
        ]
        if relation_types:
            causal_items.append((item, explanation, relation_types))
    context_count = len(causal_items)
    if context_count == 0:
        return {
            "context_count": 0,
            "edge_count": 0,
            "relation_types": [],
            "methods": [],
            "average_trust": 0.0,
            "authority_status": "non_authoritative",
        }
    explanations = [explanation for _, explanation, _ in causal_items]
    return {
        "context_count": context_count,
        "edge_count": sum(len(relation_types) for _, _, relation_types in causal_items),
        "relation_types": _unique_texts(
            relation_type
            for _, _, relation_types in causal_items
            for relation_type in relation_types
        ),
        "methods": _unique_explanation_texts(explanations, "inference_methods"),
        "average_trust": _round_metric(_average_metric(explanations, "inferred_edge_trust")),
        "authority_status": "non_authoritative",
    }


def _consolidation_candidate_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [item for item in items if _is_consolidation_candidate(item)]
    if not candidates:
        return {
            "candidate_count": 0,
            "candidate_types": [],
            "pending_count": 0,
            "accepted_count": 0,
            "authority_status": "non_authoritative",
        }
    metadata_values = [_metadata(item) for item in candidates]
    return {
        "candidate_count": len(candidates),
        "candidate_types": _unique_texts(
            _candidate_type(item, metadata)
            for item, metadata in zip(candidates, metadata_values, strict=True)
        ),
        "pending_count": sum(
            1
            for item, metadata in zip(candidates, metadata_values, strict=True)
            if _review_status(item, metadata) == "pending"
        ),
        "accepted_count": sum(
            1
            for item, metadata in zip(candidates, metadata_values, strict=True)
            if _review_status(item, metadata) == "accepted"
        ),
        "authority_status": "non_authoritative",
    }


def _is_consolidation_candidate(item: dict[str, Any]) -> bool:
    if str(item.get("entity_type") or "").strip() == "consolidation_candidate":
        return True
    entity_name = str(item.get("entity_name") or "").strip()
    if entity_name.startswith("consolidation:"):
        return True
    metadata = _metadata(item)
    return any(
        _item_text_field(item, metadata, key)
        for key in ("candidate_type", "consolidation_candidate_type", "candidate_id")
    )


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _candidate_type(item: dict[str, Any], metadata: dict[str, Any]) -> str:
    for key in ("candidate_type", "consolidation_candidate_type"):
        value = _item_text_field(item, metadata, key)
        if value:
            return value
    entity_name = str(item.get("entity_name") or "").strip()
    match = re.match(r"^consolidation:([^:]+):", entity_name)
    return match.group(1) if match else "unknown"


def _review_status(item: dict[str, Any], metadata: dict[str, Any]) -> str:
    return _item_text_field(item, metadata, "review_status").lower()


def _item_text_field(item: dict[str, Any], metadata: dict[str, Any], key: str) -> str:
    for source in (item, metadata):
        value = source.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _average_metric(items: list[dict[str, Any]], key: str) -> float:
    values = [_float_metric(item.get(key)) for item in items]
    return sum(values) / len(values) if values else 0.0


def _float_metric(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


def _round_metric(value: float, *, digits: int = 2) -> float:
    return round(value, digits)


def _unique_explanation_texts(items: list[dict[str, Any]], key: str) -> list[str]:
    values: list[str] = []
    for item in items:
        values.extend(_text_list(item.get(key)))
    return _unique_texts(values)


def _unique_texts(values: Any) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            texts.append(text)
    return texts


def _format_source_lanes(source_lanes: Any) -> str:
    if not isinstance(source_lanes, dict) or not source_lanes:
        return "none"
    return ", ".join(f"{lane}={count}" for lane, count in sorted(source_lanes.items()))
