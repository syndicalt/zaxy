"""Review-pending consolidation candidate event contracts.

Zaxy 2.0 alpha.1 consolidation emits cited candidates only. These helpers
build Eventloom append specs and deliberately keep generated abstractions
non-authoritative, including accepted review outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^consolidation:([a-z_]+):([0-9a-f]{24})$")
_AUTHORITY_STATUS = "non_authoritative"

CONSOLIDATION_CANDIDATE_TYPES = frozenset({"episode", "claim", "procedure"})
CONSOLIDATION_INITIAL_REVIEW_STATUS = "pending"
CONSOLIDATION_REVIEW_STATUSES = frozenset({"accepted", "rejected", "deferred", "conflicted"})


def build_consolidation_candidate_event(
    *,
    actor: str,
    session_id: str,
    candidate_type: str,
    title: str,
    summary: str,
    source_events: Sequence[Mapping[str, Any]],
    confidence: float,
    method: str,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Build a cited, review-pending consolidation candidate event spec."""
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    _validate_candidate_type(candidate_type)
    _validate_non_empty_string(title, field_name="title")
    _validate_non_empty_string(summary, field_name="summary")
    cited_source_events = _snapshot_source_events(source_events)
    _validate_confidence(confidence)
    _validate_non_empty_string(method, field_name="method")
    if purpose is not None:
        _validate_non_empty_string(purpose, field_name="purpose")

    payload: dict[str, Any] = {
        "candidate_id": _build_candidate_id(
            candidate_type=candidate_type,
            title=title,
            source_events=cited_source_events,
        ),
        "candidate_type": candidate_type,
        "title": title,
        "summary": summary,
        "source_events": cited_source_events,
        "confidence": confidence,
        "method": method,
        "review_status": CONSOLIDATION_INITIAL_REVIEW_STATUS,
        "authority_status": _AUTHORITY_STATUS,
    }
    if purpose is not None:
        payload["purpose"] = purpose

    return {
        "event_type": "consolidation.candidate.created",
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }


def build_consolidation_review_event(
    *,
    actor: str,
    session_id: str,
    candidate_id: str,
    status: str,
    rationale: str,
) -> dict[str, Any]:
    """Build a consolidation review event spec without authority promotion."""
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    _validate_candidate_id(candidate_id)
    _validate_review_status(status)
    _validate_non_empty_string(rationale, field_name="rationale")

    return {
        "event_type": "consolidation.candidate.reviewed",
        "actor": actor,
        "payload": {
            "candidate_id": candidate_id,
            "status": status,
            "authority_status": _AUTHORITY_STATUS,
            "rationale": rationale,
        },
        "thread": session_id,
    }


def _build_candidate_id(
    *,
    candidate_type: str,
    title: str,
    source_events: Sequence[Mapping[str, Any]],
) -> str:
    identity = {
        "candidate_type": candidate_type,
        "title": title,
        "source_events": list(source_events),
    }
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"consolidation:{candidate_type}:{digest}"


def _snapshot_source_events(source_events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(source_events, Sequence) or isinstance(source_events, str | bytes):
        raise ValueError("source_events must be a non-empty sequence of citations")
    if not source_events:
        raise ValueError("source_events must be non-empty")

    citations = []
    for index, source_event in enumerate(source_events):
        citations.append(_snapshot_source_event(source_event, index=index))
    return citations


def _snapshot_source_event(source_event: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(source_event, Mapping):
        raise ValueError(f"source_events[{index}] must be a citation mapping")

    seq = source_event.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise ValueError(f"source_events[{index}].seq must be an integer")
    if seq <= 0:
        raise ValueError(f"source_events[{index}].seq must be a positive integer")

    event_hash = source_event.get("hash")
    if not isinstance(event_hash, str) or not _EVENT_HASH_RE.fullmatch(event_hash):
        raise ValueError(
            f"source_events[{index}].hash must be exactly 64 lowercase hex characters"
        )

    return {"seq": seq, "hash": event_hash}


def _validate_candidate_type(candidate_type: str) -> None:
    if candidate_type not in CONSOLIDATION_CANDIDATE_TYPES:
        valid = ", ".join(sorted(CONSOLIDATION_CANDIDATE_TYPES))
        raise ValueError(f"candidate_type must be one of: {valid}")


def _validate_candidate_id(candidate_id: object) -> None:
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("candidate_id must be a non-empty string")
    match = _CANDIDATE_ID_RE.fullmatch(candidate_id)
    if match is None:
        raise ValueError(
            "candidate_id must match consolidation:{candidate_type}:{24 lowercase hex characters}"
        )
    candidate_type = match.group(1)
    if candidate_type not in CONSOLIDATION_CANDIDATE_TYPES:
        valid = ", ".join(sorted(CONSOLIDATION_CANDIDATE_TYPES))
        raise ValueError(f"candidate_id candidate_type must be one of: {valid}")


def _validate_review_status(status: str) -> None:
    if status not in CONSOLIDATION_REVIEW_STATUSES:
        valid = ", ".join(sorted(CONSOLIDATION_REVIEW_STATUSES))
        raise ValueError(f"status must be one of: {valid}")


def _validate_confidence(confidence: float) -> None:
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("confidence must be a number between 0.0 and 1.0")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")


def _validate_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
