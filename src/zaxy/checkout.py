"""Shared Memory Checkout policy helpers.

This module owns model-facing checkout diagnostics, guidance, quality scoring,
and prompt formatting so every interface exposes the same trust contract.
"""

from __future__ import annotations

from typing import Any

from zaxy.evidence import build_evidence_set
from zaxy.retrieval_intent import classify_retrieval_intent

_COMPACT_CONTEXT_LIMIT = 8
_COMPACT_SNIPPET_LIMIT = 500


def build_checkout_diagnostics(
    *,
    query: str | None = None,
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
    evidence_plan = _checkout_evidence_plan(query)
    if evidence_plan:
        diagnostics["evidence_plan"] = evidence_plan
    inferred_context = _inferred_context_diagnostics(current_facts)
    if inferred_context["context_count"]:
        diagnostics["inferred_context"] = inferred_context
    evidence_set = build_evidence_set(
        query=query,
        evidence_plan=evidence_plan,
        current_facts=current_facts,
        evidence=evidence,
    )
    diagnostics["evidence_set"] = evidence_set.to_diagnostics()
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
    current_facts: list[dict[str, Any]],
    retention: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build model-facing trust, ignore, refresh, and feedback guidance."""
    feedback_payloads = [
        payload
        for fact in current_facts
        if (payload := build_checkout_feedback_payload(fact, query)) is not None
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
    inferred_context = _inferred_context_diagnostics(current_facts)
    if inferred_context["context_count"]:
        trust.append("Checkout depends on inferred graph paths; inspect inferred_context diagnostics.")
    if inferred_context["low_trust_count"]:
        ignore.append("Low-trust inferred graph paths were included; treat them as leads, not facts.")
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
    lines = ["# Memory Checkout", f"Query: {query}", "", "## Current Facts"]
    compact_contexts = diagnostics.get("compact_contexts")
    if isinstance(compact_contexts, list) and compact_contexts:
        lines = ["# Memory Checkout", f"Query: {query}", "", "## Compact Answer Context"]
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
    if mode is None:
        return None
    citations = {
        citation
        for item in [*current_facts, *evidence]
        if isinstance((citation := item.get("citation")), str) and citation
    }
    return {
        "mode": mode,
        "reasons": sorted(reasons),
        "source_lane_slots": intent.source_lane_slots,
        "current_fact_count": len(current_facts),
        "citation_count": len(citations),
        "source_lanes": source_lanes,
        "evidence_groups": evidence_groups,
    }


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
    return None


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
                "issue_",
                "not_mentioned_",
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


def build_checkout_feedback_payload(fact: dict[str, Any], query: str) -> dict[str, Any] | None:
    """Build a memory_feedback payload for a cited current fact."""
    citation = fact.get("citation")
    if not isinstance(citation, str) or not citation:
        return None
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
    return {key: value for key, value in payload.items() if value is not None}


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
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
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
