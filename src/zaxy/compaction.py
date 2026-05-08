"""Safety audits for identity-preserving compaction."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zaxy.benchmark import _event_context
from zaxy.embedding import EmbeddingProvider, HashEmbeddingProvider
from zaxy.event import Event, EventLog

_IDENTITY_RE = re.compile(r"\b(?:identity|doc|decision|task|user|goal)-code-\d{4}\b")


@dataclass(frozen=True)
class CompactionAuditReport:
    """Non-destructive safety report for a compaction candidate."""

    safe: bool
    event_count: int
    integrity_ok: bool
    integrity_reason: str | None
    identity_count: int
    identity_recall: float
    citation_coverage: float
    mean_within_cluster_distance: float
    identities: tuple[str, ...]
    identity_hits: tuple[str, ...]
    missing_identities: tuple[str, ...]
    unsafe_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CompactionProjectionRecord:
    """A real source-backed record stored in a compaction projection."""

    kind: str
    event_seq: int
    event_ref: str
    text: str
    identities: tuple[str, ...]
    citations: tuple[str, ...]


@dataclass(frozen=True)
class CompactionProjection:
    """Stored compaction projection with source backpointers."""

    projection_id: str
    strategy: str
    source_event_count: int
    source_identities: tuple[str, ...]
    records: tuple[CompactionProjectionRecord, ...]
    audit: CompactionAuditReport


def audit_event_log(
    eventlog: EventLog,
    *,
    provider: EmbeddingProvider | None = None,
    identity_recall_threshold: float = 1.0,
    citation_coverage_threshold: float = 1.0,
) -> CompactionAuditReport:
    """Audit whether a log is safe for source-preserving compaction.

    The first audit is deliberately conservative: it tests whether a
    one-representative compaction candidate still carries every durable source
    identity. Future compaction operators can use the same report contract with
    medoid or exemplar candidates.
    """
    if not 0.0 <= identity_recall_threshold <= 1.0:
        raise ValueError("identity_recall_threshold must be between 0 and 1")
    if not 0.0 <= citation_coverage_threshold <= 1.0:
        raise ValueError("citation_coverage_threshold must be between 0 and 1")

    provider = provider or HashEmbeddingProvider()
    events = eventlog.read_all()
    integrity = eventlog.verify()
    identities = tuple(
        dict.fromkeys(
            identity
            for event in events
            for identity in _event_identities(event)
        )
    )
    representative = _representative_text(events)
    haystack = representative.casefold()
    identity_hits = tuple(
        identity for identity in identities if identity.casefold() in haystack
    )
    missing_identities = tuple(
        identity for identity in identities if identity.casefold() not in haystack
    )
    identity_recall = (
        len(identity_hits) / len(identities)
        if identities
        else 1.0
    )
    citation_coverage = _citation_coverage(events)
    spread = _mean_within_cluster_distance(
        [_event_context(event.model_dump()) for event in events],
        provider,
    )
    unsafe_reasons = _unsafe_reasons(
        integrity_ok=integrity.ok,
        identity_recall=identity_recall,
        identity_recall_threshold=identity_recall_threshold,
        citation_coverage=citation_coverage,
        citation_coverage_threshold=citation_coverage_threshold,
    )
    return CompactionAuditReport(
        safe=not unsafe_reasons,
        event_count=len(events),
        integrity_ok=integrity.ok,
        integrity_reason=integrity.broken_reason,
        identity_count=len(identities),
        identity_recall=round(identity_recall, 4),
        citation_coverage=round(citation_coverage, 4),
        mean_within_cluster_distance=round(spread, 4),
        identities=identities,
        identity_hits=identity_hits,
        missing_identities=missing_identities,
        unsafe_reasons=tuple(unsafe_reasons),
    )


def build_compaction_projection(
    eventlog: EventLog,
    *,
    provider: EmbeddingProvider | None = None,
    strategy: str = "medoid",
    max_records: int = 5,
) -> CompactionProjection:
    """Build a source-backed compaction projection without rewriting the log."""
    if strategy not in {"medoid", "exemplar"}:
        raise ValueError("strategy must be 'medoid' or 'exemplar'")
    if max_records <= 0:
        raise ValueError("max_records must be positive")

    provider = provider or HashEmbeddingProvider()
    events = eventlog.read_all()
    audit = audit_event_log(eventlog, provider=provider)
    selected = (
        [_select_medoid(events, provider)]
        if strategy == "medoid" and events
        else _select_exemplars(events, provider, max_records)
    )
    selected = [event for event in selected if event is not None]
    records = tuple(
        _projection_record(event, "medoid" if strategy == "medoid" else "exemplar")
        for event in selected
    )
    payload = {
        "strategy": strategy,
        "source_hashes": [event.hash for event in events],
        "records": [record.event_ref for record in records],
    }
    projection_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompactionProjection(
        projection_id=projection_id,
        strategy=strategy,
        source_event_count=len(events),
        source_identities=audit.identities,
        records=records,
        audit=audit,
    )


def write_compaction_projection(
    projection: CompactionProjection,
    path: str | Path,
) -> Path:
    """Write a compaction projection JSON artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(projection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _event_identities(event: Event) -> tuple[str, ...]:
    payload = event.payload
    identities = [
        f"eventloom://{event.thread}/events/{event.seq}#{event.hash[:12]}",
    ]
    path = _string(payload.get("path"))
    if path:
        start_line = payload.get("start_line")
        end_line = payload.get("end_line")
        if isinstance(start_line, int) and isinstance(end_line, int):
            identities.append(f"{path}:{start_line}-{end_line}")
        else:
            identities.append(path)
    source = _string(payload.get("source"))
    turn_index = payload.get("turn_index")
    if source and isinstance(turn_index, int):
        identities.append(f"{source}:turn-{turn_index}")
    elif source:
        identities.append(source)
    for key in ("taskId", "userId", "goalTitle", "title"):
        value = _string(payload.get(key))
        if value:
            identities.append(value)
    content = " ".join(
        value
        for value in (
            _string(payload.get("content")),
            _string(payload.get("summary")),
            _string(payload.get("description")),
        )
        if value
    )
    identities.extend(match.group(0) for match in _IDENTITY_RE.finditer(content))
    return tuple(dict.fromkeys(identities))


def _projection_record(event: Event, kind: str) -> CompactionProjectionRecord:
    identities = _event_identities(event)
    return CompactionProjectionRecord(
        kind=kind,
        event_seq=event.seq,
        event_ref=_event_ref(event),
        text=_event_context(event.model_dump()),
        identities=identities,
        citations=tuple(identity for identity in identities if _is_source_citation(identity)),
    )


def _select_medoid(events: list[Event], provider: EmbeddingProvider) -> Event | None:
    if not events:
        return None
    if len(events) == 1:
        return events[0]
    texts = [_event_context(event.model_dump()) for event in events]
    embeddings = [provider.embed(text) for text in texts]
    best_index = 0
    best_distance = float("inf")
    for left_index, left in enumerate(embeddings):
        distance = statistics.fmean(
            1.0 - _cosine(left, right)
            for right_index, right in enumerate(embeddings)
            if right_index != left_index
        )
        if distance < best_distance:
            best_distance = distance
            best_index = left_index
    return events[best_index]


def _select_exemplars(
    events: list[Event],
    provider: EmbeddingProvider,
    max_records: int,
) -> list[Event]:
    if len(events) <= max_records:
        return list(events)
    selected: list[Event] = []
    remaining = list(events)
    medoid = _select_medoid(remaining, provider)
    if medoid is not None:
        selected.append(medoid)
        remaining.remove(medoid)
    while remaining and len(selected) < max_records:
        selected_embeddings = [
            provider.embed(_event_context(event.model_dump()))
            for event in selected
        ]
        best_event = max(
            remaining,
            key=lambda event: min(
                1.0 - _cosine(
                    provider.embed(_event_context(event.model_dump())),
                    selected_embedding,
                )
                for selected_embedding in selected_embeddings
            ),
        )
        selected.append(best_event)
        remaining.remove(best_event)
    return selected


def _citation_coverage(events: list[Event]) -> float:
    if not events:
        return 1.0
    cited = 0
    for event in events:
        if event.type == "document.indexed":
            cited += 1 if _string(event.payload.get("path")) else 0
        elif event.type == "transcript.turn":
            has_source = _string(event.payload.get("source"))
            has_turn = isinstance(event.payload.get("turn_index"), int)
            cited += 1 if has_source and has_turn else 0
        else:
            cited += 1
    return cited / len(events)


def _representative_text(events: list[Event]) -> str:
    if not events:
        return ""
    event = events[0]
    return "\n".join([_event_context(event.model_dump()), *_event_identities(event)])


def _event_ref(event: Event) -> str:
    return f"eventloom://{event.thread}/events/{event.seq}#{event.hash[:12]}"


def _is_source_citation(identity: str) -> bool:
    return (
        "/" in identity
        or ":turn-" in identity
        or identity.startswith("eventloom://")
    )


def _mean_within_cluster_distance(
    texts: list[str],
    provider: EmbeddingProvider,
) -> float:
    if len(texts) < 2:
        return 0.0
    embeddings = [provider.embed(text) for text in texts]
    distances: list[float] = []
    for left_index, left in enumerate(embeddings):
        for right in embeddings[left_index + 1:]:
            distances.append(1.0 - _cosine(left, right))
    return statistics.fmean(distances) if distances else 0.0


def _unsafe_reasons(
    *,
    integrity_ok: bool,
    identity_recall: float,
    identity_recall_threshold: float,
    citation_coverage: float,
    citation_coverage_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if not integrity_ok:
        reasons.append("integrity check failed")
    if identity_recall < identity_recall_threshold:
        reasons.append(f"identity recall below {identity_recall_threshold:.3f}")
    if citation_coverage < citation_coverage_threshold:
        reasons.append(f"citation coverage below {citation_coverage_threshold:.3f}")
    return reasons


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
