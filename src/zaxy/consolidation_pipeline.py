"""Deterministic review-gated consolidation proposal pipeline.

This module turns Eventloom history into cited, review-pending consolidation
candidates. It never promotes generated abstractions to authoritative memory.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zaxy.consolidation import (
    CONSOLIDATION_CANDIDATE_TYPES,
    build_consolidation_candidate_event,
)
from zaxy.security import validate_session_id

_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SEGMENT_ID_RE = re.compile(r"^segment:(?P<session_id>[A-Za-z0-9_.-]{1,128}):(?P<start>\d{6})-(?P<end>\d{6})$")
_SUMMARY_MAX_CHARS = 240

ACTIONABLE_EVENT_TYPES = frozenset(
    {
        "tool.call.completed",
        "command.completed",
        "command.result",
        "file.edit.applied",
        "file.edit.completed",
        "task.completed",
        "task.started",
        "task.proposed",
        "coordination.handoff.created",
        "coordination.finding.reported",
        "coordination.assignment.created",
        "coordination.review.completed",
        "subagent.completed",
    }
)
ACTIONABLE_EVENT_PREFIXES = (
    "tool.call.",
    "command.",
    "file.edit.",
    "task.",
    "coordination.",
)
PASSIVE_EVENT_TYPES = frozenset(
    {
        "memory.checkout.completed",
        "memory.query.completed",
        "memory.feedback.recorded",
        "memory.bootstrap.completed",
        "tool.call.completed",  # not passive; kept out by explicit active set below
    }
) - {"tool.call.completed"}


@dataclass(frozen=True)
class ConsolidationSegment:
    """A session-scoped, citation-preserving Eventloom event window."""

    session_id: str
    segment_id: str
    event_type_counts: Mapping[str, int]
    source_events: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        sid = validate_session_id(self.session_id)
        if not isinstance(self.segment_id, str):
            raise ValueError("segment_id must be a session-scoped string")
        match = _SEGMENT_ID_RE.fullmatch(self.segment_id)
        if match is None or match.group("session_id") != sid:
            raise ValueError("segment_id must be session-scoped")
        if not self.source_events:
            raise ValueError("source_events must be non-empty")

        source_rows = [_snapshot_source_event(event, index=index) for index, event in enumerate(self.source_events)]
        seqs = [row["seq"] for row in source_rows]
        if seqs != sorted(seqs):
            raise ValueError("source_events must be ordered by seq")
        if len(set(seqs)) != len(seqs):
            raise ValueError("source_events seq values must be unique")

        expected_id = build_segment_id(sid, seqs)
        if self.segment_id != expected_id:
            raise ValueError("segment_id must match source_events seq range")

        counts = _validate_event_type_counts(self.event_type_counts)
        actual_counts = event_type_counts(source_rows)
        if counts != actual_counts:
            raise ValueError("event_type_counts must match source_events")

    @property
    def source_event_refs(self) -> list[str]:
        return [f"{event['seq']}:{event['hash']}" for event in self.source_events]

    @property
    def source_event_count(self) -> int:
        return len(self.source_events)

    def candidate_source_events(self) -> list[dict[str, Any]]:
        """Return the minimal source-event citation shape used by alpha.1."""
        return [{"seq": int(event["seq"]), "hash": str(event["hash"])} for event in self.source_events]


@dataclass(frozen=True)
class ProposedConsolidation:
    """A reviewable consolidation proposal bound to one source segment."""

    segment: ConsolidationSegment
    candidate_type: str
    title: str
    summary: str
    confidence: float
    method: str
    purpose: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_type not in CONSOLIDATION_CANDIDATE_TYPES:
            raise ValueError("candidate_type must be a supported consolidation type")
        _validate_text(self.title, field_name="title")
        _validate_text(self.summary, field_name="summary")
        _validate_text(self.method, field_name="method")
        if not isinstance(self.confidence, int | float) or isinstance(self.confidence, bool):
            raise ValueError("confidence must be a number between 0.0 and 1.0")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.purpose is not None:
            _validate_text(self.purpose, field_name="purpose")

    def to_candidate_event(self, *, actor: str) -> dict[str, Any]:
        """Build a non-authoritative, pending Eventloom candidate event."""
        return build_consolidation_candidate_event(
            actor=actor,
            session_id=self.segment.session_id,
            candidate_type=self.candidate_type,
            title=self.title,
            summary=self.summary,
            source_events=self.segment.candidate_source_events(),
            confidence=float(self.confidence),
            method=self.method,
            purpose=self.purpose,
        )


def build_segment_id(session_id: str, event_seqs: Sequence[int]) -> str:
    """Build a stable segment identifier scoped to ``session_id``."""
    sid = validate_session_id(session_id)
    if not event_seqs:
        raise ValueError("event_seqs must be non-empty")
    seqs = sorted(_validate_seq(seq) for seq in event_seqs)
    return f"segment:{sid}:{seqs[0]:06d}-{seqs[-1]:06d}"


def event_type_counts(source_events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return sorted event-type counts after validating source rows."""
    counts: Counter[str] = Counter()
    for index, event in enumerate(source_events):
        row = _snapshot_source_event(event, index=index)
        counts[row["event_type"]] += 1
    return dict(sorted(counts.items()))


def select_consolidation_segments(
    events: Sequence[Any],
    *,
    session_id: str,
    window_size: int = 8,
) -> list[ConsolidationSegment]:
    """Select deterministic actionable Eventloom windows for consolidation."""
    sid = validate_session_id(session_id)
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

    source_events: list[dict[str, Any]] = []
    for event in events:
        if _event_session_id(event) != sid:
            continue
        source_event = _source_event_from_event(event)
        if _is_actionable_event_type(source_event["event_type"]):
            source_events.append(source_event)

    segments: list[ConsolidationSegment] = []
    for start in range(0, len(source_events), window_size):
        window = source_events[start : start + window_size]
        if not window:
            continue
        segments.append(
            ConsolidationSegment(
                session_id=sid,
                segment_id=build_segment_id(sid, [int(event["seq"]) for event in window]),
                event_type_counts=event_type_counts(window),
                source_events=window,
            )
        )
    return segments


def generate_consolidation_proposals(
    segments: Sequence[ConsolidationSegment],
    *,
    purpose: str | None = None,
) -> list[ProposedConsolidation]:
    """Generate conservative deterministic proposals for each segment."""
    if purpose is not None:
        _validate_text(purpose, field_name="purpose")
    proposals: list[ProposedConsolidation] = []
    for segment in segments:
        if not isinstance(segment, ConsolidationSegment):
            raise ValueError("segments must contain ConsolidationSegment instances")
        proposals.append(_episode_proposal(segment, purpose=purpose))
        if _has_claim_signal(segment):
            proposals.append(_claim_proposal(segment, purpose=purpose))
        if _has_procedure_signal(segment):
            proposals.append(_procedure_proposal(segment, purpose=purpose))
    return proposals


def _episode_proposal(segment: ConsolidationSegment, *, purpose: str | None) -> ProposedConsolidation:
    return ProposedConsolidation(
        segment=segment,
        candidate_type="episode",
        title=f"Episode {segment.segment_id.rsplit(':', 1)[-1]}",
        summary=_segment_summary(segment),
        confidence=0.68,
        method="deterministic_episode_segment_v1",
        purpose=purpose,
    )


def _claim_proposal(segment: ConsolidationSegment, *, purpose: str | None) -> ProposedConsolidation:
    return ProposedConsolidation(
        segment=segment,
        candidate_type="claim",
        title=f"Claim from {segment.segment_id}",
        summary=(
            f"Candidate claim supported by {segment.source_event_count} cited source events: "
            f"{_segment_summary(segment)}"
        ),
        confidence=0.62,
        method="deterministic_claim_signal_v1",
        purpose=purpose,
    )


def _procedure_proposal(segment: ConsolidationSegment, *, purpose: str | None) -> ProposedConsolidation:
    return ProposedConsolidation(
        segment=segment,
        candidate_type="procedure",
        title=f"Procedure from {segment.segment_id}",
        summary=f"Candidate procedure inferred from observed workflow steps: {_segment_summary(segment)}",
        confidence=0.58,
        method="deterministic_procedure_trace_v1",
        purpose=purpose,
    )


def _has_claim_signal(segment: ConsolidationSegment) -> bool:
    if segment.source_event_count < 2:
        return False
    return any(
        str(event.get("event_type", "")).startswith(("file.edit.", "task.", "coordination."))
        for event in segment.source_events
    )


def _has_procedure_signal(segment: ConsolidationSegment) -> bool:
    event_types = set(segment.event_type_counts)
    if segment.event_type_counts.get("tool.call.completed", 0) >= 2:
        return True
    return bool(
        any(event_type.startswith("file.edit.") for event_type in event_types)
        and any(event_type.startswith(("tool.call.", "command.", "task.")) for event_type in event_types)
    )


def _segment_summary(segment: ConsolidationSegment) -> str:
    summaries = [str(event.get("summary", "")).strip() for event in segment.source_events]
    compact = [summary for summary in summaries if summary]
    summary = " -> ".join(compact[:4]) or segment.segment_id
    if len(summary) > _SUMMARY_MAX_CHARS:
        return summary[: _SUMMARY_MAX_CHARS - 3].rstrip() + "..."
    return summary


def _source_event_from_event(event: Any) -> dict[str, Any]:
    seq = _validate_seq(getattr(event, "seq", None))
    event_hash = _validate_hash(getattr(event, "hash", None), field_name="event hash")
    event_type = _event_type(event)
    payload = getattr(event, "payload", {})
    return {
        "seq": seq,
        "hash": event_hash,
        "event_type": event_type,
        "summary": _event_summary(event_type, payload),
    }


def _event_session_id(event: Any) -> str:
    raw = getattr(event, "thread", "default")
    if not isinstance(raw, str) or not raw:
        return "default"
    return raw


def _event_type(event: Any) -> str:
    raw = getattr(event, "type", None)
    if raw is None:
        raw = getattr(event, "event_type", None)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("event_type must be a non-empty string")
    return raw.strip()


def _event_summary(event_type: str, payload: object) -> str:
    if not isinstance(payload, Mapping):
        return event_type
    parts = [event_type]
    for key in ("status", "tool_name", "command", "path", "summary", "title", "reason"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(" ".join(value.strip().split()))
    return " | ".join(parts[:4])[:_SUMMARY_MAX_CHARS]


def _is_actionable_event_type(event_type: str) -> bool:
    if event_type in PASSIVE_EVENT_TYPES:
        return False
    return event_type in ACTIONABLE_EVENT_TYPES or event_type.startswith(ACTIONABLE_EVENT_PREFIXES)


def _validate_seq(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("event seq must be a positive integer")
    return value


def _validate_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _EVENT_HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be 64 lowercase hex")
    return value


def _snapshot_source_event(event: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise ValueError(f"source_events[{index}] must be a mapping")
    seq = _validate_seq(event.get("seq"))
    event_hash = _validate_hash(event.get("hash"), field_name=f"source_events[{index}].hash")
    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError(f"source_events[{index}].event_type must be a non-empty string")
    row: dict[str, Any] = {
        "seq": seq,
        "hash": event_hash,
        "event_type": event_type.strip(),
    }
    summary = event.get("summary")
    if summary is not None:
        if not isinstance(summary, str):
            raise ValueError(f"source_events[{index}].summary must be a string")
        row["summary"] = " ".join(summary.split())[:_SUMMARY_MAX_CHARS]
    return row


def _validate_event_type_counts(counts: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise ValueError("event_type_counts must be a mapping")
    validated: dict[str, int] = {}
    for event_type, count in counts.items():
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type_counts keys must be non-empty strings")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("event_type_counts values must be non-negative integers")
        if count:
            validated[event_type] = count
    return dict(sorted(validated.items()))


def _validate_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value
