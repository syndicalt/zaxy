"""Non-authoritative metacognition event contracts.

These helpers build Eventloom append specs for uncertainty, conflict,
confidence, and re-verification state. Generated metacognition is observable
diagnostic state only; it never promotes claims to authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_AUTHORITY_STATUS = "non_authoritative"
_OPEN_STATUS = "open"
_UNRESOLVED_STATUS = "unresolved"

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_EVENTLOOM_CITATION_RE = re.compile(
    r"^eventloom://[^/\s]+/events/[1-9][0-9]*#(?:[0-9a-f]{12}|[0-9a-f]{64})$"
)
_METACOGNITION_ID_RE = re.compile(r"^metacognition:[a-z_]+:[0-9a-f]{24}$")

METACOGNITION_PRIORITIES = frozenset({"low", "normal", "high", "urgent"})


def build_known_unknown_event(
    *,
    actor: str,
    session_id: str,
    question: str,
    reason: str,
    source_events: Sequence[Mapping[str, Any]],
    claim_key: str | None = None,
    gap_type: str | None = None,
    reverify_query: str | None = None,
    unknown_id: str | None = None,
) -> dict[str, Any]:
    """Build an open, cited known-unknown event append spec."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    question = _validate_text(question, field_name="question")
    reason = _validate_text(reason, field_name="reason")
    cited_source_events = _snapshot_source_events(source_events)
    claim_key = _validate_optional_text(claim_key, field_name="claim_key")
    gap_type = _validate_optional_text(gap_type, field_name="gap_type")
    reverify_query = _validate_optional_text(reverify_query, field_name="reverify_query")
    unknown_id = _validate_or_build_id(
        explicit_id=unknown_id,
        id_type="unknown",
        identity={
            "claim_key": claim_key,
            "gap_type": gap_type,
            "question": question,
            "source_events": cited_source_events,
        },
    )

    payload: dict[str, Any] = {
        "unknown_id": unknown_id,
        "question": question,
        "reason": reason,
        "source_events": cited_source_events,
        "status": _OPEN_STATUS,
        "authority_status": _AUTHORITY_STATUS,
    }
    _add_optional(payload, "claim_key", claim_key)
    _add_optional(payload, "gap_type", gap_type)
    _add_optional(payload, "reverify_query", reverify_query)

    return {
        "event_type": "metacognition.unknown.recorded",
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def build_confidence_assessment_event(
    *,
    actor: str,
    session_id: str,
    claim: str,
    confidence: float,
    support_count: int,
    conflict_count: int,
    evidence: Sequence[Mapping[str, Any]],
    method: str,
    requires_reverify: bool = False,
    claim_key: str | None = None,
    assessment_id: str | None = None,
) -> dict[str, Any]:
    """Build an append-only confidence assessment point."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    claim = _validate_text(claim, field_name="claim")
    confidence = _validate_confidence(confidence)
    support_count = _validate_non_negative_int(support_count, field_name="support_count")
    conflict_count = _validate_non_negative_int(conflict_count, field_name="conflict_count")
    evidence = _snapshot_evidence(evidence)
    method = _validate_text(method, field_name="method")
    if not isinstance(requires_reverify, bool):
        raise ValueError("requires_reverify must be a boolean")
    claim_key = _validate_optional_text(claim_key, field_name="claim_key")
    assessment_id = _validate_or_build_id(
        explicit_id=assessment_id,
        id_type="confidence",
        identity={
            "claim": claim,
            "claim_key": claim_key,
            "confidence": confidence,
            "conflict_count": conflict_count,
            "evidence": evidence,
            "method": method,
            "requires_reverify": requires_reverify,
            "support_count": support_count,
        },
    )

    payload: dict[str, Any] = {
        "assessment_id": assessment_id,
        "claim": claim,
        "confidence": confidence,
        "support_count": support_count,
        "conflict_count": conflict_count,
        "evidence": evidence,
        "method": method,
        "requires_reverify": requires_reverify,
        "authority_status": _AUTHORITY_STATUS,
    }
    _add_optional(payload, "claim_key", claim_key)

    return {
        "event_type": "metacognition.confidence.assessed",
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def build_conflict_cluster_event(
    *,
    actor: str,
    session_id: str,
    claim_key: str,
    claim: str,
    supporting_source_events: Sequence[Mapping[str, Any]],
    conflicting_source_events: Sequence[Mapping[str, Any]],
    confidence: float,
    reason: str,
    cluster_id: str | None = None,
) -> dict[str, Any]:
    """Build an unresolved conflict cluster event append spec."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    claim_key = _validate_text(claim_key, field_name="claim_key")
    claim = _validate_text(claim, field_name="claim")
    supporting_source_events = _snapshot_source_events(
        supporting_source_events,
        field_name="supporting_source_events",
    )
    conflicting_source_events = _snapshot_source_events(
        conflicting_source_events,
        field_name="conflicting_source_events",
    )
    confidence = _validate_confidence(confidence)
    reason = _validate_text(reason, field_name="reason")
    cluster_id = _validate_or_build_id(
        explicit_id=cluster_id,
        id_type="conflict_cluster",
        identity={
            "claim": claim,
            "claim_key": claim_key,
            "conflicting_source_events": conflicting_source_events,
            "supporting_source_events": supporting_source_events,
        },
    )

    return {
        "event_type": "metacognition.conflict.clustered",
        "actor": actor,
        "thread": session_id,
        "payload": {
            "cluster_id": cluster_id,
            "claim_key": claim_key,
            "claim": claim,
            "supporting_source_events": supporting_source_events,
            "conflicting_source_events": conflicting_source_events,
            "confidence": confidence,
            "reason": reason,
            "resolution_status": _UNRESOLVED_STATUS,
            "authority_status": _AUTHORITY_STATUS,
        },
    }


def build_reverify_request_event(
    *,
    actor: str,
    session_id: str,
    query: str,
    reason: str,
    source_events: Sequence[Mapping[str, Any]],
    priority: str = "normal",
    claim_key: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build an open re-verification request event append spec."""
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    query = _validate_text(query, field_name="query")
    reason = _validate_text(reason, field_name="reason")
    source_events = _snapshot_source_events(source_events)
    priority = _validate_priority(priority)
    claim_key = _validate_optional_text(claim_key, field_name="claim_key")
    request_id = _validate_or_build_id(
        explicit_id=request_id,
        id_type="reverify",
        identity={
            "claim_key": claim_key,
            "query": query,
            "source_events": source_events,
        },
    )

    payload: dict[str, Any] = {
        "reverify_id": request_id,
        "query": query,
        "reason": reason,
        "source_events": source_events,
        "priority": priority,
        "status": _OPEN_STATUS,
        "authority_status": _AUTHORITY_STATUS,
    }
    _add_optional(payload, "claim_key", claim_key)

    return {
        "event_type": "metacognition.reverify.requested",
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def summarize_metacognition_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize replayed metacognition events without changing authority."""
    summary: dict[str, Any] = {
        "unknown_count": 0,
        "open_unknown_count": 0,
        "confidence_assessment_count": 0,
        "conflict_cluster_count": 0,
        "unresolved_conflict_cluster_count": 0,
        "reverify_request_count": 0,
        "reverify_needed_count": 0,
        "open_unknowns": [],
        "reverify_requests": [],
        "conflict_clusters": [],
    }

    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("event_type") or event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue

        if event_type == "metacognition.unknown.recorded":
            summary["unknown_count"] += 1
            if payload.get("status") == _OPEN_STATUS:
                summary["open_unknown_count"] += 1
                summary["open_unknowns"].append(dict(payload))
        elif event_type == "metacognition.confidence.assessed":
            summary["confidence_assessment_count"] += 1
        elif event_type == "metacognition.conflict.clustered":
            summary["conflict_cluster_count"] += 1
            if payload.get("resolution_status") == _UNRESOLVED_STATUS:
                summary["unresolved_conflict_cluster_count"] += 1
                summary["conflict_clusters"].append(dict(payload))
        elif event_type == "metacognition.reverify.requested":
            summary["reverify_request_count"] += 1
            if payload.get("status") == _OPEN_STATUS:
                summary["reverify_needed_count"] += 1
                summary["reverify_requests"].append(dict(payload))

    return summary


def _validate_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_text(value, field_name=field_name)


def _validate_confidence(value: object) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("confidence must be a number between 0.0 and 1.0")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return confidence


def _validate_non_negative_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _validate_priority(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("priority must be one of: " + ", ".join(sorted(METACOGNITION_PRIORITIES)))
    priority = value.strip().casefold()
    if priority not in METACOGNITION_PRIORITIES:
        raise ValueError("priority must be one of: " + ", ".join(sorted(METACOGNITION_PRIORITIES)))
    return priority


def _snapshot_source_events(
    source_events: Sequence[Mapping[str, Any]],
    *,
    field_name: str = "source_events",
) -> list[dict[str, Any]]:
    if not isinstance(source_events, Sequence) or isinstance(source_events, str | bytes):
        raise ValueError(f"{field_name} must be a non-empty sequence of citations")
    if not source_events:
        raise ValueError(f"{field_name} must be non-empty")

    return [
        _snapshot_source_event(source_event, index=index, field_name=field_name)
        for index, source_event in enumerate(source_events)
    ]


def _snapshot_source_event(
    source_event: Mapping[str, Any],
    *,
    index: int,
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(source_event, Mapping):
        raise ValueError(f"{field_name}[{index}] must be a citation mapping")

    seq = source_event.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
        raise ValueError(f"{field_name}[{index}].seq must be a positive integer")

    event_hash = source_event.get("hash")
    if not isinstance(event_hash, str) or _EVENT_HASH_RE.fullmatch(event_hash) is None:
        raise ValueError(f"{field_name}[{index}].hash must be exactly 64 lowercase hex characters")

    return {"seq": seq, "hash": event_hash}


def _snapshot_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(evidence, Sequence) or isinstance(evidence, str | bytes):
        raise ValueError("evidence must be a sequence of Eventloom citations")

    return [_snapshot_evidence_item(item, index=index) for index, item in enumerate(evidence)]


def _snapshot_evidence_item(item: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"evidence[{index}] must be a citation mapping")

    citation = item.get("citation")
    if not isinstance(citation, str) or _EVENTLOOM_CITATION_RE.fullmatch(citation) is None:
        raise ValueError(
            f"evidence[{index}].citation must be an Eventloom citation with a 12 or 64 character hash"
        )

    snapshot = dict(item)
    for key, value in snapshot.items():
        if isinstance(value, str):
            snapshot[key] = _validate_text(value, field_name=f"evidence[{index}].{key}")
    return snapshot


def _validate_or_build_id(
    *,
    explicit_id: str | None,
    id_type: str,
    identity: Mapping[str, Any],
) -> str:
    if explicit_id is not None:
        explicit_id = _validate_text(explicit_id, field_name=f"{id_type}_id")
        if _METACOGNITION_ID_RE.fullmatch(explicit_id) is None:
            raise ValueError(
                f"{id_type}_id must match metacognition:{{id_type}}:"
                "{24 lowercase hex characters}"
            )
        return explicit_id
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"metacognition:{id_type}:{digest}"


def _add_optional(payload: dict[str, Any], key: str, value: object | None) -> None:
    if value is not None:
        payload[key] = value
