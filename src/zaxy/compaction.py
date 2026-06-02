"""Safety audits for identity-preserving compaction."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from zaxy.benchmark import _event_context
from zaxy.embedding import EmbeddingProvider, HashEmbeddingProvider
from zaxy.event import Event, EventLog
from zaxy.purpose import PurposeProfile, purpose_profile

_IDENTITY_RE = re.compile(r"\b(?:identity|doc|decision|task|user|goal)-code-\d{4}\b")
_PURPOSE_CONSOLIDATION_RULES: dict[str, dict[str, Any]] = {
    "coding": {
        "strategy": "purpose_exemplar",
        "min_records": 8,
        "preserve_all": False,
        "reason": "preserve_invariants_failed_attempts_and_test_evidence",
    },
    "review": {
        "strategy": "purpose_preserve_all",
        "min_records": 0,
        "preserve_all": True,
        "reason": "preserve_blocking_risks_decisions_and_verification",
    },
    "release": {
        "strategy": "purpose_preserve_all",
        "min_records": 0,
        "preserve_all": True,
        "reason": "preserve_release_gates_regressions_and_external_blockers",
    },
    "security": {
        "strategy": "purpose_preserve_all",
        "min_records": 0,
        "preserve_all": True,
        "reason": "preserve_threats_controls_findings_and_risk_acceptance",
    },
    "research": {
        "strategy": "purpose_exemplar",
        "min_records": 8,
        "preserve_all": False,
        "reason": "preserve_claims_sources_contradictions_and_open_questions",
    },
    "support": {
        "strategy": "purpose_exemplar",
        "min_records": 8,
        "preserve_all": False,
        "reason": "preserve_escalations_workarounds_customer_impact_and_resolutions",
    },
    "product": {
        "strategy": "purpose_exemplar",
        "min_records": 8,
        "preserve_all": False,
        "reason": "preserve_roadmap_signals_promises_tradeoffs_and_experiment_outcomes",
    },
    "sales": {
        "strategy": "purpose_preserve_all",
        "min_records": 0,
        "preserve_all": True,
        "reason": "preserve_commitments_objections_renewal_risks_and_account_context",
    },
    "legal": {
        "strategy": "purpose_preserve_all",
        "min_records": 0,
        "preserve_all": True,
        "reason": "preserve_obligations_approvals_deadlines_and_exceptions",
    },
    "executive": {
        "strategy": "purpose_preserve_all",
        "min_records": 0,
        "preserve_all": True,
        "reason": "preserve_strategic_exceptions_market_patterns_risks_and_decisions",
    },
}


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
    authority_scope: str = "authoritative"
    purpose_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompactionProjection:
    """Stored compaction projection with source backpointers."""

    projection_id: str
    strategy: str
    source_event_count: int
    source_identities: tuple[str, ...]
    records: tuple[CompactionProjectionRecord, ...]
    audit: CompactionAuditReport
    purpose: dict[str, Any] = field(default_factory=dict)
    consolidation_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompactionProjectionSearchResult:
    """Projection routing hit with source citations preserved."""

    projection_id: str
    strategy: str
    record: CompactionProjectionRecord
    score: float
    citations: tuple[str, ...]


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


def compaction_remediation_plan(report: CompactionAuditReport) -> list[dict[str, Any]]:
    """Return concrete remediation steps for a failed compaction audit."""
    if report.safe:
        return []
    steps: list[dict[str, Any]] = []
    if not report.integrity_ok:
        steps.append(
            {
                "code": "repair_eventloom_integrity",
                "action": (
                    "Restore the Eventloom log from backup or remove the tampered "
                    "candidate from compaction."
                ),
                "details": {"reason": report.integrity_reason},
            }
        )
    if report.missing_identities:
        steps.append(
            {
                "code": "preserve_missing_identities",
                "action": "Use exemplar projection or increase max_records before compacting.",
                "details": {
                    "missing_identities": list(report.missing_identities),
                    "identity_recall": report.identity_recall,
                },
            }
        )
    if report.citation_coverage < 1.0:
        steps.append(
            {
                "code": "restore_source_citations",
                "action": (
                    "Re-ingest uncited document/transcript events with source paths, "
                    "line ranges, or Eventloom refs before compacting."
                ),
                "details": {"citation_coverage": report.citation_coverage},
            }
        )
    if not steps:
        steps.append(
            {
                "code": "review_unsafe_reasons",
                "action": "Inspect unsafe_reasons and rerun compaction audit after remediation.",
                "details": {"unsafe_reasons": list(report.unsafe_reasons)},
            }
        )
    return steps


def build_compaction_projection(
    eventlog: EventLog,
    *,
    provider: EmbeddingProvider | None = None,
    strategy: str = "medoid",
    max_records: int = 5,
    purpose: PurposeProfile | dict[str, Any] | str | None = None,
) -> CompactionProjection:
    """Build a source-backed compaction projection without rewriting the log."""
    if strategy not in {"medoid", "exemplar"}:
        raise ValueError("strategy must be 'medoid' or 'exemplar'")
    if max_records <= 0:
        raise ValueError("max_records must be positive")

    provider = provider or HashEmbeddingProvider()
    events = eventlog.read_all()
    audit = audit_event_log(eventlog, provider=provider)
    profile = purpose_profile(purpose)
    authoritative_events, diagnostic_events, consolidation_policy = _purpose_consolidation_plan(
        events,
        profile,
        requested_strategy=strategy,
        max_records=max_records,
    )
    effective_strategy = str(consolidation_policy["strategy"])
    selected = _select_projection_events(
        authoritative_events,
        provider=provider,
        strategy=effective_strategy,
        max_records=int(consolidation_policy.get("effective_max_records") or max_records),
    )
    records = tuple(
        _projection_record(
            event,
            _projection_record_kind(event, profile, effective_strategy),
            authority_scope="authoritative",
            purpose_reasons=_purpose_record_reasons(event, profile),
        )
        for event in selected
    )
    diagnostic_identities = tuple(
        identity
        for event in diagnostic_events
        for identity in _event_identities(event)
    )
    consolidation_policy = {
        **consolidation_policy,
        "authoritative_event_seqs": [event.seq for event in authoritative_events],
        "diagnostic_event_seqs": [event.seq for event in diagnostic_events],
        "diagnostic_identities": list(dict.fromkeys(diagnostic_identities)),
    }
    payload = {
        "strategy": effective_strategy,
        "purpose": profile.to_dict(),
        "consolidation_policy": consolidation_policy,
        "source_hashes": [event.hash for event in events],
        "records": [record.event_ref for record in records],
    }
    projection_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CompactionProjection(
        projection_id=projection_id,
        strategy=effective_strategy,
        source_event_count=len(events),
        source_identities=audit.identities,
        records=records,
        audit=audit,
        purpose=profile.to_dict(),
        consolidation_policy=consolidation_policy,
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


def load_compaction_projection(path: str | Path) -> CompactionProjection:
    """Load a source-backed compaction projection JSON artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _projection_from_payload(payload)


def search_compaction_projections(
    projections: list[CompactionProjection] | tuple[CompactionProjection, ...],
    query: str,
    *,
    limit: int = 10,
) -> list[CompactionProjectionSearchResult]:
    """Search projection records as routing candidates with citations."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    results: list[CompactionProjectionSearchResult] = []
    for projection in projections:
        for record in projection.records:
            searchable = " ".join([record.text, *record.identities])
            record_tokens = _tokens(searchable)
            overlap = query_tokens & record_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            results.append(
                CompactionProjectionSearchResult(
                    projection_id=projection.projection_id,
                    strategy=projection.strategy,
                    record=record,
                    score=round(score, 4),
                    citations=record.citations,
                )
            )
    return sorted(results, key=lambda item: item.score, reverse=True)[:limit]


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
    for key in ("mission_id", "worker_id", "finding_id", "claim_key", "claim_value"):
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


def _projection_record(
    event: Event,
    kind: str,
    *,
    authority_scope: str = "authoritative",
    purpose_reasons: tuple[str, ...] = (),
) -> CompactionProjectionRecord:
    identities = _event_identities(event)
    return CompactionProjectionRecord(
        kind=kind,
        event_seq=event.seq,
        event_ref=_event_ref(event),
        text=_event_context(event.model_dump()),
        identities=identities,
        citations=tuple(identity for identity in identities if _is_source_citation(identity)),
        authority_scope=authority_scope,
        purpose_reasons=purpose_reasons,
    )


def _select_projection_events(
    events: list[Event],
    *,
    provider: EmbeddingProvider,
    strategy: str,
    max_records: int,
) -> list[Event]:
    if strategy == "medoid" and events:
        medoid = _select_medoid(events, provider)
        return [medoid] if medoid is not None else []
    if strategy == "coordinate_authoritative":
        return list(events)
    if strategy == "purpose_preserve_all":
        return list(events)
    return _select_exemplars(events, provider, max_records)


def _purpose_consolidation_plan(
    events: list[Event],
    profile: PurposeProfile,
    *,
    requested_strategy: str,
    max_records: int,
) -> tuple[list[Event], list[Event], dict[str, Any]]:
    if profile.profile == "general":
        return list(events), [], {
            "profile": profile.profile,
            "strategy": requested_strategy,
            "requested_strategy": requested_strategy,
            "effective_max_records": max_records,
            "preserve_all": False,
            "authoritative_count": len(events),
            "diagnostic_count": 0,
            "suppressed_count": 0,
            "retain": list(profile.retain),
            "suppress": list(profile.suppress),
            "warnings": list(profile.warnings),
        }
    if profile.profile != "coordinate":
        rule = _PURPOSE_CONSOLIDATION_RULES.get(profile.profile, {})
        preserve_all = bool(rule.get("preserve_all"))
        effective_max_records = (
            len(events)
            if preserve_all
            else max(max_records, int(rule.get("min_records") or max_records))
        )
        return list(events), [], {
            "profile": profile.profile,
            "strategy": str(rule.get("strategy") or "purpose_exemplar"),
            "requested_strategy": requested_strategy,
            "effective_max_records": effective_max_records,
            "preserve_all": preserve_all,
            "authoritative_count": len(events),
            "diagnostic_count": 0,
            "suppressed_count": 0,
            "retain": list(profile.retain),
            "suppress": list(profile.suppress),
            "warnings": list(profile.warnings),
            "purpose_consolidation_reason": str(rule.get("reason") or "purpose_retained"),
        }
    authoritative: list[Event] = []
    diagnostic: list[Event] = []
    for event in events:
        if _is_coordinate_authoritative_event(event):
            authoritative.append(event)
        elif _is_coordinate_diagnostic_event(event):
            diagnostic.append(event)
        else:
            authoritative.append(event)
    return authoritative, diagnostic, {
        "profile": profile.profile,
        "strategy": "coordinate_authoritative",
        "requested_strategy": requested_strategy,
        "effective_max_records": len(authoritative),
        "preserve_all": True,
        "max_records_ignored": len(authoritative) > max_records,
        "authoritative_count": len(authoritative),
        "diagnostic_count": len(diagnostic),
        "suppressed_count": len(diagnostic),
        "retain": list(profile.retain),
        "suppress": list(profile.suppress),
        "warnings": list(profile.warnings),
    }


def _projection_record_kind(event: Event, profile: PurposeProfile, strategy: str) -> str:
    if profile.profile == "coordinate":
        if event.type in {"coordination.proof_packet.created", "coordination.handoff.created"}:
            return "coordinate_proof"
        return "coordinate_authoritative"
    if strategy == "purpose_preserve_all":
        return f"{profile.profile}_retained"
    if strategy == "purpose_exemplar":
        return f"{profile.profile}_exemplar"
    return "medoid" if strategy == "medoid" else "exemplar"


def _purpose_record_reasons(event: Event, profile: PurposeProfile) -> tuple[str, ...]:
    if profile.profile != "coordinate":
        reasons = [
            str(value)
            for value in (
                *profile.retain,
                *profile.ontology_lens[:2],
            )
            if value
        ]
        return tuple(dict.fromkeys(reasons)) or ("purpose_retained",)
    if event.type in {"coordination.proof_packet.created", "coordination.handoff.created"}:
        return ("proof_or_handoff",)
    status = _event_status(event)
    if status in {"accepted", "promoted", "approved"}:
        return ("accepted_parent_state",)
    if _string(event.payload.get("authority_scope")) == "mission-parent":
        return ("mission_parent_authority",)
    return ("coordinate_authoritative",)


def _is_coordinate_authoritative_event(event: Event) -> bool:
    if event.type in {
        "coordination.proof_packet.created",
        "coordination.handoff.created",
        "coordination.finding.promoted",
        "coordination.finding.accepted",
    }:
        return True
    status = _event_status(event)
    return (
        status in {"accepted", "promoted", "approved"} and _coordinate_has_authority_refs(event)
        or _string(event.payload.get("authority_scope")) == "mission-parent"
    )


def _is_coordinate_diagnostic_event(event: Event) -> bool:
    if event.payload.get("stale") is True:
        return True
    status = _event_status(event)
    if status in {"pending", "rejected", "deferred", "stale", "superseded"}:
        return True
    return event.type.startswith("coordination.")


def _event_status(event: Event) -> str | None:
    for key in ("coordination_status", "finding_status", "status"):
        value = _string(event.payload.get(key))
        if value:
            return value.strip().casefold().replace(" ", "_").replace("-", "_")
    return None


def _coordinate_has_authority_refs(event: Event) -> bool:
    for key in (
        "promotion_event_ref",
        "review_event_ref",
        "source_event_ref",
        "handoff_event_ref",
    ):
        if _string(event.payload.get(key)):
            return True
    return (
        isinstance(event.payload.get("source_event_seq"), int)
        and bool(_string(event.payload.get("source_event_hash")))
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


def _projection_from_payload(payload: dict[str, Any]) -> CompactionProjection:
    return CompactionProjection(
        projection_id=str(payload["projection_id"]),
        strategy=str(payload["strategy"]),
        source_event_count=int(payload["source_event_count"]),
        source_identities=tuple(str(value) for value in payload["source_identities"]),
        records=tuple(
            CompactionProjectionRecord(
                kind=str(record["kind"]),
                event_seq=int(record["event_seq"]),
                event_ref=str(record["event_ref"]),
                text=str(record["text"]),
                identities=tuple(str(value) for value in record["identities"]),
                citations=tuple(str(value) for value in record["citations"]),
                authority_scope=str(record.get("authority_scope") or "authoritative"),
                purpose_reasons=tuple(str(value) for value in record.get("purpose_reasons", ())),
            )
            for record in payload["records"]
        ),
        audit=CompactionAuditReport(
            safe=bool(payload["audit"]["safe"]),
            event_count=int(payload["audit"]["event_count"]),
            integrity_ok=bool(payload["audit"]["integrity_ok"]),
            integrity_reason=payload["audit"].get("integrity_reason"),
            identity_count=int(payload["audit"]["identity_count"]),
            identity_recall=float(payload["audit"]["identity_recall"]),
            citation_coverage=float(payload["audit"]["citation_coverage"]),
            mean_within_cluster_distance=float(
                payload["audit"]["mean_within_cluster_distance"]
            ),
            identities=tuple(str(value) for value in payload["audit"]["identities"]),
            identity_hits=tuple(str(value) for value in payload["audit"]["identity_hits"]),
            missing_identities=tuple(
                str(value) for value in payload["audit"]["missing_identities"]
            ),
            unsafe_reasons=tuple(str(value) for value in payload["audit"]["unsafe_reasons"]),
        ),
        purpose=dict(payload.get("purpose") or {}),
        consolidation_policy=dict(payload.get("consolidation_policy") or {}),
    )


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/#-]*", value.casefold())
        if len(token) > 1
    }
