"""Shared Memory Checkout policy helpers.

This module owns model-facing checkout diagnostics, guidance, quality scoring,
and prompt formatting so every interface exposes the same trust contract.
"""

from __future__ import annotations

from typing import Any


def build_checkout_diagnostics(
    *,
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
    inferred_context = _inferred_context_diagnostics(current_facts)
    if inferred_context["context_count"]:
        diagnostics["inferred_context"] = inferred_context
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
    return {
        "trust": trust,
        "ignore": ignore,
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
    confidence = 0.25
    confidence += min(current_fact_count, 2) * 0.22
    confidence += min(current_citation_count, 2) * 0.28
    if superseded_excluded and current_fact_count:
        confidence += 0.07
    confidence = min(0.95, confidence)
    confidence -= min(0.35, warning_count * 0.18)
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
    lines.append(f"- Source lanes: {_format_source_lanes(source_lanes)}")
    lines.append(f"- Citations: {diagnostics.get('citation_count', 0)}")
    lines.append(f"- Current citations: {diagnostics.get('current_citation_count', 0)}")
    lines.append(f"- Current facts: {diagnostics.get('current_fact_count', 0)}")
    lines.append(
        f"- Superseded contexts excluded: {diagnostics.get('superseded_contexts_excluded', 0)}"
    )
    inferred_context = diagnostics.get("inferred_context")
    if isinstance(inferred_context, dict):
        lines.append(
            "- Inferred graph context: "
            f"contexts={inferred_context.get('context_count', 0)}, "
            f"edges={inferred_context.get('edge_count', 0)}, "
            f"average_trust={inferred_context.get('average_trust', 0)}"
        )
    if diagnostics.get("feedback_recommended"):
        lines.append(
            "- Feedback: call "
            f"{diagnostics.get('feedback_tool', 'memory_feedback')} after using cited context."
        )
    lines.extend(["", assembly_prompt])
    return "\n".join(lines).strip()


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


def _format_source_lanes(source_lanes: Any) -> str:
    if not isinstance(source_lanes, dict) or not source_lanes:
        return "none"
    return ", ".join(f"{lane}={count}" for lane, count in sorted(source_lanes.items()))
