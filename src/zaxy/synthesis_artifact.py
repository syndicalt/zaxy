"""Deterministic Eventloom-backed synthesis artifact payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from zaxy.evidence import evidence_content, evidence_source_id
from zaxy.synthesis_packet import synthesis_packet_from_diagnostics


def build_synthesis_artifact(checkout: Any) -> dict[str, Any]:
    """Build a stable synthesis artifact from Memory Checkout diagnostics."""
    candidates = _answer_candidates(checkout)
    support_ids = _candidate_support_ids(candidates)
    payload: dict[str, Any] = {
        "schema_version": "synthesis_artifact_v1",
        "query": str(checkout.query),
        "session_id": str(checkout.session_id),
        "checkout": {
            "ref": checkout.ref,
            "replay_event_count": checkout.replay_event_count,
            "quality": _json_object(checkout.quality),
            "diagnostics_summary": _diagnostics_summary(checkout.diagnostics),
        },
        "plan": _artifact_plan(checkout.diagnostics),
        "purpose": _json_object(getattr(checkout, "purpose", {})),
        "answer_candidates": candidates,
        "ledger_rows": _ledger_rows(checkout.diagnostics),
        **_operation_result_payload(checkout.diagnostics),
        "support_packet": _support_packet(checkout, support_ids),
        "verification": _verification(checkout),
    }
    payload["artifact_id"] = _artifact_id(payload)
    return payload


def normalize_synthesis_outcome(outcome: str) -> str:
    """Normalize synthesis feedback outcomes into persisted event outcomes."""
    normalized = outcome.casefold().strip()
    if normalized not in {"used", "helpful", "rejected", "corrected", "excluded"}:
        raise ValueError("outcome must be one of: used, helpful, rejected, corrected, excluded")
    return "used" if normalized == "helpful" else normalized


def synthesis_outcome_event_type(outcome: str) -> str:
    """Return the Eventloom event type for a normalized synthesis outcome."""
    if outcome == "excluded":
        return "memory.evidence.excluded"
    return f"memory.synthesis.{outcome}"


def build_synthesis_candidate_event_payload(
    *,
    checkout: Any,
    candidate: dict[str, Any],
    outcome: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a stable Eventloom payload for synthesis candidate outcome feedback."""
    answer_candidate = _checkout_candidate(checkout, candidate)
    _assert_candidate_feedback_allowed(checkout, outcome, answer_candidate)
    support_ids = _string_list(answer_candidate.get("support_source_ids"))
    excluded_ids = _string_list(answer_candidate.get("excluded_source_ids"))
    diagnostics = _json_object(getattr(checkout, "diagnostics", {}))
    payload: dict[str, Any] = {
        "query": str(getattr(checkout, "query", "")),
        "outcome": outcome,
        "answer_candidate": answer_candidate,
        "quality": _quality_artifact(_json_object(getattr(checkout, "quality", {}))),
        "purpose": _json_object(getattr(checkout, "purpose", {})),
        "slot_plan": diagnostics.get("slot_plan"),
        "support_source_ids": support_ids,
        "excluded_source_ids": excluded_ids,
        "citations": _support_citations(checkout, support_ids),
    }
    if reason:
        payload["reason"] = reason
    ref = getattr(checkout, "ref", None)
    if ref is not None:
        payload["ref"] = ref
    return {key: value for key, value in payload.items() if value not in (None, [], {})}


def build_synthesis_evidence_event_payload(
    *,
    checkout: Any,
    row: dict[str, Any],
    outcome: str,
    candidate: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a stable Eventloom payload for synthesis evidence-row feedback."""
    evidence_row = _clean_evidence_row(row)
    answer_candidate = _checkout_candidate(checkout, candidate) if candidate is not None else {}
    support_ids = _string_list(answer_candidate.get("support_source_ids"))
    payload: dict[str, Any] = {
        "query": str(getattr(checkout, "query", "")),
        "outcome": outcome,
        "evidence_row": evidence_row,
        "answer_candidate": answer_candidate,
        "quality": _quality_artifact(_json_object(getattr(checkout, "quality", {}))),
        "purpose": _json_object(getattr(checkout, "purpose", {})),
        "slot_plan": _json_object(getattr(checkout, "diagnostics", {})).get("slot_plan"),
        "source_group": evidence_row.get("source_group"),
        "fact_id": evidence_row.get("fact_id"),
        "citation": evidence_row.get("citation"),
        "support_source_ids": support_ids,
    }
    if reason:
        payload["reason"] = reason
    ref = getattr(checkout, "ref", None)
    if ref is not None:
        payload["ref"] = ref
    return {key: value for key, value in payload.items() if value not in (None, [], {})}


def _answer_candidates(checkout: Any) -> list[dict[str, Any]]:
    diagnostics = _json_object(getattr(checkout, "diagnostics", {}))
    synthesis = diagnostics.get("synthesis")
    if not isinstance(synthesis, dict):
        raise ValueError("synthesis artifact requires diagnostics.synthesis.answer_candidates")
    candidates = synthesis_packet_from_diagnostics(diagnostics).answer_candidates
    if not candidates:
        raise ValueError("synthesis artifact requires diagnostics.synthesis.answer_candidates")
    return candidates


def _clean_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "rank",
        "type",
        "confidence",
        "answer_key",
        "answer",
        "support_source_ids",
        "excluded_source_ids",
    )
    return {
        key: _json_value(candidate[key])
        for key in allowed
        if key in candidate and _json_value(candidate[key]) is not None
    }


def _checkout_candidate(checkout: Any, candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical checkout candidate selected by client feedback."""
    requested = _clean_candidate(candidate)
    if not requested:
        raise ValueError("candidate must include synthesis answer candidate fields")
    candidates = synthesis_packet_from_diagnostics(
        _json_object(getattr(checkout, "diagnostics", {}))
    ).answer_candidates
    if not candidates:
        raise ValueError("candidate feedback requires diagnostics.synthesis.answer_candidates")
    for checkout_candidate in candidates:
        if _candidate_matches(requested, checkout_candidate):
            return checkout_candidate
    raise ValueError("candidate must match diagnostics.synthesis.answer_candidates")


def _candidate_matches(requested: dict[str, Any], checkout_candidate: dict[str, Any]) -> bool:
    requested_key = _non_empty_text(requested.get("answer_key"))
    checkout_key = _non_empty_text(checkout_candidate.get("answer_key"))
    if requested_key and checkout_key and requested_key != checkout_key:
        return False
    for key in ("rank", "type", "answer"):
        requested_value = requested.get(key)
        if requested_value is not None and requested_value != checkout_candidate.get(key):
            return False
    for key in ("support_source_ids", "excluded_source_ids"):
        requested_ids = _string_list(requested.get(key))
        checkout_ids = _string_list(checkout_candidate.get(key))
        if requested_ids and requested_ids != checkout_ids:
            return False
    return any(key in requested for key in ("answer_key", "rank", "type", "answer"))


def _candidate_support_ids(candidates: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for candidate in candidates:
        for source_id in _string_list(candidate.get("support_source_ids")):
            if source_id not in ids:
                ids.append(source_id)
    return ids


def _ledger_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    return synthesis_packet_from_diagnostics(diagnostics).ledger_rows


def _clean_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "fact_id",
        "source_group",
        "citation",
        "kind",
        "entity",
        "value",
        "unit",
        "time",
        "label",
        "raw_span",
        "normalized_identity",
        "include_reason",
        "exclude_reason",
        "confidence",
    )
    cleaned = {
        key: _json_value(row[key])
        for key in allowed
        if key in row and _json_value(row[key]) is not None
    }
    if not cleaned:
        raise ValueError("row must include synthesis ledger row fields")
    if not any(_non_empty_text(cleaned.get(key)) for key in ("fact_id", "source_group", "citation")):
        raise ValueError("row must include fact_id, source_group, or citation")
    return cleaned


def _operation_result_payload(diagnostics: dict[str, Any]) -> dict[str, Any]:
    packet = synthesis_packet_from_diagnostics(diagnostics)
    payload: dict[str, Any] = {}
    if packet.operations:
        payload["operations"] = packet.operations
    if packet.result:
        payload["result"] = packet.result
    return payload


def _support_packet(checkout: Any, support_ids: list[str]) -> dict[str, Any]:
    citations: list[str] = []
    source_groups: list[str] = []
    snippets: list[str] = []
    support = set(support_ids)
    for item in [*getattr(checkout, "current_facts", []), *getattr(checkout, "evidence", [])]:
        if not isinstance(item, dict):
            continue
        source_id = evidence_source_id(item)
        if support and source_id not in support:
            continue
        citation = item.get("citation")
        if isinstance(citation, str) and citation and citation not in citations:
            citations.append(citation)
        if source_id and source_id not in source_groups:
            source_groups.append(source_id)
        content = evidence_content(item)
        if content and content not in snippets:
            snippets.append(content)
    return {
        "citations": citations,
        "source_groups": source_groups,
        "snippets": snippets,
    }


def _artifact_plan(diagnostics: dict[str, Any]) -> dict[str, Any]:
    slot_plan = diagnostics.get("slot_plan")
    if not isinstance(slot_plan, dict):
        return {}
    return {
        key: _json_value(slot_plan.get(key))
        for key in ("answer_type", "operation", "required_slots", "optional_slots", "required_source_groups", "required_kinds", "reasons")
        if _json_value(slot_plan.get(key)) is not None
    }


def _diagnostics_summary(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _json_value(diagnostics.get(key))
        for key in ("source_lanes", "citation_count", "current_citation_count", "evidence_plan_status")
        if _json_value(diagnostics.get(key)) is not None
    }


def _verification(checkout: Any) -> dict[str, Any]:
    diagnostics = _json_object(getattr(checkout, "diagnostics", {}))
    quality = _json_object(getattr(checkout, "quality", {}))
    payload: dict[str, Any] = {
        "warnings": list(getattr(checkout, "warnings", []) or []),
        "missing_evidence": _missing_evidence(diagnostics, quality),
        "contradictions": _contradictions(diagnostics),
        "dedupe_decisions": _dedupe_decisions(diagnostics),
    }
    evidence_policy_failures = _evidence_policy_failures(diagnostics)
    if evidence_policy_failures:
        payload["evidence_policy_failures"] = evidence_policy_failures
    promotion_gate = _promotion_gate(checkout)
    if not promotion_gate["allowed"]:
        payload["promotion_gate"] = promotion_gate
    return payload


def _missing_evidence(diagnostics: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_status = diagnostics.get("evidence_plan_status")
    if not isinstance(evidence_status, dict) or evidence_status.get("satisfied"):
        return []
    required_action = quality.get("required_action")
    if not isinstance(required_action, dict):
        required_action = {}
    payload = {
        "observed_source_groups": _json_value(evidence_status.get("observed_source_groups")),
        "required_source_groups": _json_value(evidence_status.get("required_source_groups")),
        "refresh_query": _json_value(evidence_status.get("refresh_query")),
        "missing_slots": _json_value(required_action.get("missing_slots")),
        "suggested_queries": _json_value(required_action.get("suggested_queries")),
    }
    cleaned = {key: value for key, value in payload.items() if value not in (None, [], {})}
    return [cleaned] if cleaned else []


def _evidence_policy_failures(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    policy = diagnostics.get("evidence_policy")
    if not isinstance(policy, dict) or policy.get("satisfied"):
        return []
    payload = {
        "profile": _json_value(policy.get("profile")),
        "mode": _json_value(policy.get("mode")),
        "missing_requirements": _json_value(policy.get("missing_requirements")),
        "failure_reasons": _json_value(policy.get("failure_reasons")),
        "suggested_queries": _json_value(policy.get("suggested_queries")),
    }
    cleaned = {key: value for key, value in payload.items() if value not in (None, "", [], {})}
    return [cleaned] if cleaned else []


def _assert_candidate_feedback_allowed(
    checkout: Any,
    outcome: str,
    answer_candidate: dict[str, Any],
) -> None:
    if normalize_synthesis_outcome(outcome) != "used":
        return
    gate = _promotion_gate(checkout, candidate=answer_candidate)
    if gate["allowed"]:
        return
    profile = str(gate.get("profile") or "purpose")
    missing = ", ".join(_string_list(gate.get("missing_requirements"))) or str(
        gate.get("reason") or "required evidence"
    )
    raise ValueError(
        f"cannot promote synthesis candidate for {profile}; promotion gate is blocked: {missing}"
    )


def _promotion_gate(checkout: Any, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    diagnostics = _json_object(getattr(checkout, "diagnostics", {}))
    quality = _json_object(getattr(checkout, "quality", {}))
    policy = diagnostics.get("evidence_policy")
    profile = "general"
    missing_requirements: list[str] = []
    suggested_queries: list[str] = []
    failure_reasons: list[str] = []
    if isinstance(policy, dict):
        profile = str(policy.get("profile") or profile)
        missing_requirements = _string_list(policy.get("missing_requirements"))
        suggested_queries = _string_list(policy.get("suggested_queries"))
        failure_reasons = _string_list(policy.get("failure_reasons"))
    if isinstance(policy, dict) and policy.get("satisfied") is False:
        return {
            "allowed": False,
            "reason": "evidence_policy_unsatisfied",
            "profile": profile,
            "mode": policy.get("mode"),
            "missing_requirements": missing_requirements,
            "failure_reasons": failure_reasons,
            "suggested_queries": suggested_queries,
        }
    if quality.get("answerability") != "answer_from_memory":
        return {
            "allowed": False,
            "reason": "checkout_not_answerable_from_memory",
            "profile": profile,
            "answerability": quality.get("answerability"),
            "missing_requirements": missing_requirements,
            "failure_reasons": failure_reasons,
            "suggested_queries": suggested_queries,
        }
    missing_evidence = _missing_evidence(diagnostics, quality)
    if missing_evidence:
        return {
            "allowed": False,
            "reason": "missing_query_evidence",
            "profile": profile,
            "missing_evidence": missing_evidence,
        }
    if candidate is not None:
        support_ids = _string_list(candidate.get("support_source_ids"))
        unresolved = _unresolved_support_source_ids(checkout, support_ids)
        if unresolved:
            return {
                "allowed": False,
                "reason": "support_sources_missing_citations",
                "profile": profile,
                "unresolved_support_source_ids": unresolved,
            }
    return {"allowed": True, "profile": profile}


def _unresolved_support_source_ids(checkout: Any, support_ids: list[str]) -> list[str]:
    if not support_ids:
        return []
    cited_source_ids = set()
    for item in [*getattr(checkout, "current_facts", []), *getattr(checkout, "evidence", [])]:
        if not isinstance(item, dict):
            continue
        citation = item.get("citation")
        if not isinstance(citation, str) or not citation:
            continue
        cited_source_ids.add(evidence_source_id(item))
    return [source_id for source_id in support_ids if source_id not in cited_source_ids]


def _dedupe_decisions(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for row in _ledger_rows(diagnostics):
        exclude_reason = row.get("exclude_reason")
        if not isinstance(exclude_reason, str) or not exclude_reason:
            continue
        decision = {
            key: _json_value(row.get(key))
            for key in (
                "fact_id",
                "source_group",
                "citation",
                "normalized_identity",
                "include_reason",
                "exclude_reason",
                "confidence",
            )
            if _json_value(row.get(key)) is not None
        }
        if decision:
            decisions.append(decision)
    return decisions


def _contradictions(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    analytics = diagnostics.get("skill_analytics")
    if not isinstance(analytics, dict):
        return []
    rollback_candidates = analytics.get("rollback_candidates")
    if not isinstance(rollback_candidates, list):
        return []
    contradictions: list[dict[str, Any]] = []
    for candidate in rollback_candidates:
        if not isinstance(candidate, dict) or candidate.get("reason") != "contradicted":
            continue
        payload = {
            "type": "skill_memory",
            "skill_id": _json_value(candidate.get("skill_id")),
            "reason": _json_value(candidate.get("reason")),
            "rollback_to": _json_value(candidate.get("rollback_to")),
        }
        cleaned = {key: value for key, value in payload.items() if value not in (None, "", [], {})}
        if cleaned:
            contradictions.append(cleaned)
    return contradictions


def _artifact_id(payload: dict[str, Any]) -> str:
    identity = {key: value for key, value in payload.items() if key != "artifact_id"}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _quality_artifact(quality: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in quality.items()
        if key in {"answerability", "confidence", "reasons"} and isinstance(value, str | int | float | list)
    }


def _support_citations(checkout: Any, support_ids: list[str]) -> list[str]:
    citations: list[str] = []
    support = set(support_ids)
    for item in [*getattr(checkout, "current_facts", []), *getattr(checkout, "evidence", [])]:
        if not isinstance(item, dict):
            continue
        citation = item.get("citation")
        if not isinstance(citation, str) or not citation:
            continue
        if support and evidence_source_id(item) not in support:
            continue
        if citation not in citations:
            citations.append(citation)
    return citations


def _json_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_value(value: Any) -> Any:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value if _json_value(item) is not None]
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if (cleaned := _json_value(item)) is not None
        }
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _non_empty_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
