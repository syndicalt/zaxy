"""Evidence-program coverage traces for cited synthesis operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from typing import Protocol


class EvidenceRow(Protocol):
    """Minimal row contract shared by synthesis ledgers."""

    @property
    def kind(self) -> str: ...

    @property
    def source_group(self) -> str: ...

    @property
    def exclude_reason(self) -> str: ...


@dataclass(frozen=True)
class EvidenceSlotSpec:
    """One required or optional evidence slot for a synthesis program."""

    name: str
    kind: str
    required: bool = True
    min_source_groups: int = 1
    min_rows: int = 0


@dataclass(frozen=True)
class EvidenceSlotTrace:
    """Coverage result for one evidence slot."""

    name: str
    kind: str
    required: bool
    min_source_groups: int
    min_rows: int
    source_groups: tuple[str, ...]
    row_count: int
    missing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "min_source_groups": self.min_source_groups,
            "min_rows": self.min_rows,
            "source_groups": list(self.source_groups),
            "row_count": self.row_count,
            "missing": self.missing,
        }


@dataclass(frozen=True)
class EvidenceProgramTrace:
    """Auditable slot-coverage trace for a deterministic synthesis operation."""

    version: str
    operation: str
    answer_type: str
    slots: tuple[EvidenceSlotTrace, ...]

    @property
    def complete(self) -> bool:
        return not any(slot.missing for slot in self.slots if slot.required)

    @property
    def missing_slots(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.slots if slot.required and slot.missing)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "operation": self.operation,
            "answer_type": self.answer_type,
            "complete": self.complete,
            "missing_slots": list(self.missing_slots),
            "slots": [slot.to_dict() for slot in self.slots],
        }


def trace_evidence_program(
    *,
    operation: str,
    answer_type: str,
    slots: tuple[EvidenceSlotSpec, ...],
    rows: Sequence[EvidenceRow],
) -> EvidenceProgramTrace:
    """Build a deterministic slot-coverage trace from included ledger rows."""
    traces: list[EvidenceSlotTrace] = []
    for slot in slots:
        matching_rows = tuple(
            row
            for row in rows
            if row.kind == slot.kind and not row.exclude_reason
        )
        source_groups = tuple(
            dict.fromkeys(
                row.source_group
                for row in matching_rows
            )
        )
        missing = len(source_groups) < slot.min_source_groups or (
            slot.min_rows > 0 and len(matching_rows) < slot.min_rows
        )
        traces.append(
            EvidenceSlotTrace(
                name=slot.name,
                kind=slot.kind,
                required=slot.required,
                min_source_groups=slot.min_source_groups,
                min_rows=slot.min_rows,
                source_groups=source_groups,
                row_count=len(matching_rows),
                missing=missing,
            )
        )
    return EvidenceProgramTrace(
        version="evidence_program_v1",
        operation=operation,
        answer_type=answer_type,
        slots=tuple(traces),
    )


@dataclass(frozen=True)
class TemporalEvidenceProgramSpec:
    """Declarative temporal evidence operation over normalized events."""

    operator: str
    event_class_terms: tuple[str, ...]
    boundary_terms: tuple[str, ...]
    require_boundary: bool = True
    distinct: bool = True

    @property
    def direction(self) -> str | None:
        if self.operator.endswith("_before"):
            return "before"
        if self.operator.endswith("_after"):
            return "after"
        return None


@dataclass(frozen=True)
class TemporalEvidenceRow:
    """One dated, countable event candidate for temporal program execution."""

    event_id: str
    label: str
    event_date: str | date | None
    source_group: str
    citation: str
    canonical_identity: str
    raw_span: str
    action: str = ""
    object_terms: tuple[str, ...] = ()
    include_reason: str = "temporal_event_candidate"
    exclude_reason: str = ""

    @property
    def normalized_date(self) -> date | None:
        if self.event_date is None:
            return None
        if isinstance(self.event_date, date):
            return self.event_date
        try:
            return date.fromisoformat(self.event_date)
        except ValueError:
            return None


@dataclass(frozen=True)
class TemporalEvidenceDecision:
    """Program decision for one temporal evidence row."""

    row: TemporalEvidenceRow
    event_date: date | None
    include_reason: str
    exclude_reason: str = ""

    @property
    def included(self) -> bool:
        return not self.exclude_reason

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event_id": self.row.event_id,
            "source_group": self.row.source_group,
            "citation": self.row.citation,
            "canonical_identity": self.row.canonical_identity,
            "label": self.row.label,
            "raw_span": self.row.raw_span,
            "include_reason": self.include_reason,
        }
        if self.row.action:
            payload["action"] = self.row.action
        if self.row.object_terms:
            payload["object_terms"] = list(self.row.object_terms)
        if self.event_date is not None:
            payload["event_date"] = self.event_date.isoformat()
        if self.exclude_reason:
            payload["exclude_reason"] = self.exclude_reason
        return payload


@dataclass(frozen=True)
class TemporalEvidenceProgramResult:
    """Executed temporal evidence program with auditable row decisions."""

    spec: TemporalEvidenceProgramSpec
    decisions: tuple[TemporalEvidenceDecision, ...]
    boundary: TemporalEvidenceDecision | None

    @property
    def complete(self) -> bool:
        return self.boundary is not None or not self.spec.require_boundary

    @property
    def included_rows(self) -> tuple[TemporalEvidenceRow, ...]:
        return tuple(decision.row for decision in self.decisions if decision.included)

    @property
    def excluded_rows(self) -> tuple[TemporalEvidenceRow, ...]:
        return tuple(
            replace(decision.row, exclude_reason=decision.exclude_reason)
            for decision in self.decisions
            if not decision.included
        )

    @property
    def answer_value(self) -> int:
        return len(self.included_rows)

    def to_dict(self) -> dict[str, object]:
        excluded_reasons: dict[str, int] = {}
        for decision in self.decisions:
            if decision.exclude_reason:
                excluded_reasons[decision.exclude_reason] = excluded_reasons.get(decision.exclude_reason, 0) + 1
        return {
            "version": "temporal_evidence_program_v1",
            "operator": self.spec.operator,
            "direction": self.spec.direction,
            "event_class_terms": list(self.spec.event_class_terms),
            "boundary_terms": list(self.spec.boundary_terms),
            "complete": self.complete,
            "answer_value": self.answer_value,
            "boundary": self._boundary_payload(),
            "coverage": {
                "candidate_count": len(self.decisions),
                "included_count": len(self.included_rows),
                "excluded_count": len(self.excluded_rows),
                "excluded_reasons": excluded_reasons,
            },
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def _boundary_payload(self) -> dict[str, object] | None:
        if self.boundary is None:
            return None
        payload: dict[str, object] = {
            "event_id": self.boundary.row.event_id,
            "source_group": self.boundary.row.source_group,
        }
        if self.boundary.event_date is not None:
            payload["event_date"] = self.boundary.event_date.isoformat()
        return payload


def execute_temporal_evidence_program(
    spec: TemporalEvidenceProgramSpec,
    *,
    rows: Sequence[TemporalEvidenceRow],
) -> TemporalEvidenceProgramResult:
    """Execute a temporal operator over normalized event rows."""
    boundary = _temporal_boundary_decision(spec, rows)
    boundary_date = boundary.event_date if boundary is not None else None
    seen_identities: set[str] = set()
    decisions: list[TemporalEvidenceDecision] = []
    for row in rows:
        row_date = row.normalized_date
        exclude_reason = row.exclude_reason
        if not exclude_reason and boundary is not None and row.event_id == boundary.row.event_id:
            exclude_reason = "temporal_count_target"
        if not exclude_reason and row_date is None:
            exclude_reason = "missing_temporal_count_date"
        if not exclude_reason and boundary_date is not None:
            if spec.direction == "before" and row_date is not None and row_date >= boundary_date:
                exclude_reason = "temporal_count_outside_window"
            if spec.direction == "after" and row_date is not None and row_date <= boundary_date:
                exclude_reason = "temporal_count_outside_window"
        if not exclude_reason and spec.distinct:
            identity = row.canonical_identity or row.event_id
            if identity in seen_identities:
                exclude_reason = "duplicate_identity"
            else:
                seen_identities.add(identity)
        decisions.append(
            TemporalEvidenceDecision(
                row=row,
                event_date=row_date,
                include_reason=row.include_reason,
                exclude_reason=exclude_reason,
            )
        )
    return TemporalEvidenceProgramResult(spec=spec, decisions=tuple(decisions), boundary=boundary)


def _temporal_boundary_decision(
    spec: TemporalEvidenceProgramSpec,
    rows: Sequence[TemporalEvidenceRow],
) -> TemporalEvidenceDecision | None:
    scored: list[tuple[int, str, TemporalEvidenceRow, date]] = []
    boundary_terms = {term.casefold() for term in spec.boundary_terms if term}
    for row in rows:
        row_date = row.normalized_date
        if row_date is None:
            continue
        haystack = " ".join(
            (
                row.label,
                row.raw_span,
                row.action,
                " ".join(row.object_terms),
            )
        ).casefold()
        score = sum(1 for term in boundary_terms if term in haystack)
        if score <= 0:
            continue
        scored.append((score, row.source_group, row, row_date))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    _, _, row, row_date = scored[0]
    return TemporalEvidenceDecision(
        row=row,
        event_date=row_date,
        include_reason="temporal_boundary_event",
    )
