"""Neutral substrate records for documents and transcripts.

Neutral substrate records preserve source text without committing to a
purpose-specific business label at ingestion time. Purpose projections can be
rebuilt later from Eventloom-backed neutral records with source backpointers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PURPOSE_SPECIFIC_LABELS = {
    "churn_risk",
    "customer_escalation",
    "legal_obligation",
    "roadmap_commitment",
    "security_warning",
}


@dataclass(frozen=True)
class NeutralSubstrateRecord:
    """Role-neutral source record extracted from a document or transcript."""

    substrate_id: str
    actor: str
    artifact: str
    action: str
    time: str
    source: str
    quote: str
    uncertainty: str
    permission_scope: str
    candidate_claim: str
    source_event_ref: str

    def to_properties(self) -> dict[str, Any]:
        """Return compact JSON properties for graph projection."""
        return {
            key: value
            for key, value in asdict(self).items()
            if key != "substrate_id" and value not in {"", None}
        }


@dataclass(frozen=True)
class PurposeProjectionRecord:
    """Purpose-specific view rebuilt from a neutral substrate record."""

    projection_id: str
    neutral_substrate_id: str
    purpose_profile: str
    purpose_label: str
    source_event_ref: str
    source_backpointer: str
    quote: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON payload for audits and benchmark fixtures."""
        return asdict(self)


@dataclass(frozen=True)
class PurposeLabelAudit:
    """Audit result for irreversible purpose labels at ingestion time."""

    safe: bool
    forbidden_labels: tuple[str, ...]
    source_event_ref: str
    action: str

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON payload."""
        return asdict(self)


def neutral_document_record(
    *,
    actor: str,
    timestamp: str,
    path: str,
    start_line: int,
    end_line: int,
    content: str,
    source_event_ref: str,
    permission_scope: str | None = None,
    uncertainty: str | None = None,
    candidate_claim: str | None = None,
) -> NeutralSubstrateRecord:
    """Build a role-neutral document substrate record."""
    artifact = f"{path}:{start_line}-{end_line}"
    return NeutralSubstrateRecord(
        substrate_id=f"neutral:document:{artifact}",
        actor=actor,
        artifact=artifact,
        action="document_indexed",
        time=timestamp,
        source=path,
        quote=_quote(content),
        uncertainty=_text(uncertainty) or "unspecified",
        permission_scope=_text(permission_scope) or "project-local",
        candidate_claim=_text(candidate_claim) or _candidate_claim(content),
        source_event_ref=source_event_ref,
    )


def neutral_transcript_record(
    *,
    actor: str,
    timestamp: str,
    source: str,
    turn_index: int,
    role: str,
    content: str,
    source_event_ref: str,
    permission_scope: str | None = None,
    uncertainty: str | None = None,
    candidate_claim: str | None = None,
) -> NeutralSubstrateRecord:
    """Build a role-neutral transcript substrate record."""
    artifact = f"{source}:turn-{turn_index}"
    return NeutralSubstrateRecord(
        substrate_id=f"neutral:transcript:{artifact}",
        actor=actor,
        artifact=artifact,
        action="transcript_turn",
        time=timestamp,
        source=source,
        quote=_quote(content),
        uncertainty=_text(uncertainty) or "sanitized",
        permission_scope=_text(permission_scope) or "session",
        candidate_claim=_text(candidate_claim) or _candidate_claim(content),
        source_event_ref=source_event_ref,
    )


def audit_ingestion_purpose_labels(
    payload: dict[str, Any],
    *,
    source_event_ref: str,
) -> PurposeLabelAudit:
    """Flag purpose-specific labels that were written during ingestion."""
    labels: set[str] = set()
    for key in ("purpose_label", "semantic_label", "entity_type", "label"):
        value = payload.get(key)
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
            if normalized in PURPOSE_SPECIFIC_LABELS:
                labels.add(normalized)
    raw_labels = payload.get("labels")
    if isinstance(raw_labels, list | tuple):
        for item in raw_labels:
            if isinstance(item, str):
                normalized = item.strip().casefold().replace("-", "_").replace(" ", "_")
                if normalized in PURPOSE_SPECIFIC_LABELS:
                    labels.add(normalized)
    ordered = tuple(sorted(labels))
    return PurposeLabelAudit(
        safe=not ordered,
        forbidden_labels=ordered,
        source_event_ref=source_event_ref,
        action="flag_ingestion_purpose_label" if ordered else "ok",
    )


def build_purpose_projection_record(
    neutral: NeutralSubstrateRecord | dict[str, Any],
    *,
    purpose_profile: str,
    purpose_label: str,
) -> PurposeProjectionRecord:
    """Build a purpose-specific projection with neutral source backpointers."""
    record = neutral if isinstance(neutral, NeutralSubstrateRecord) else _neutral_from_mapping(neutral)
    normalized_profile = purpose_profile.strip().casefold().replace(" ", "-")
    normalized_label = purpose_label.strip().casefold().replace("-", "_").replace(" ", "_")
    return PurposeProjectionRecord(
        projection_id=f"purpose:{normalized_profile}:{normalized_label}:{record.substrate_id}",
        neutral_substrate_id=record.substrate_id,
        purpose_profile=normalized_profile,
        purpose_label=normalized_label,
        source_event_ref=record.source_event_ref,
        source_backpointer=record.artifact,
        quote=record.quote,
    )


def _neutral_from_mapping(payload: dict[str, Any]) -> NeutralSubstrateRecord:
    return NeutralSubstrateRecord(
        substrate_id=str(payload["substrate_id"]),
        actor=str(payload.get("actor") or ""),
        artifact=str(payload.get("artifact") or ""),
        action=str(payload.get("action") or ""),
        time=str(payload.get("time") or ""),
        source=str(payload.get("source") or ""),
        quote=str(payload.get("quote") or ""),
        uncertainty=str(payload.get("uncertainty") or ""),
        permission_scope=str(payload.get("permission_scope") or ""),
        candidate_claim=str(payload.get("candidate_claim") or ""),
        source_event_ref=str(payload.get("source_event_ref") or ""),
    )


def _quote(content: str) -> str:
    compact = " ".join(str(content or "").split())
    return compact[:500]


def _candidate_claim(content: str) -> str:
    compact = _quote(content)
    if "." in compact:
        return compact.split(".", 1)[0].strip()
    return compact[:240]


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
