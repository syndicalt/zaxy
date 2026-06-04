"""Evidence-program coverage traces for cited synthesis operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EvidenceSlotTrace:
    """Coverage result for one evidence slot."""

    name: str
    kind: str
    required: bool
    min_source_groups: int
    source_groups: tuple[str, ...]
    missing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "min_source_groups": self.min_source_groups,
            "source_groups": list(self.source_groups),
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
        source_groups = tuple(
            dict.fromkeys(
                row.source_group
                for row in rows
                if row.kind == slot.kind and not row.exclude_reason
            )
        )
        missing = len(source_groups) < slot.min_source_groups
        traces.append(
            EvidenceSlotTrace(
                name=slot.name,
                kind=slot.kind,
                required=slot.required,
                min_source_groups=slot.min_source_groups,
                source_groups=source_groups,
                missing=missing,
            )
        )
    return EvidenceProgramTrace(
        version="evidence_program_v1",
        operation=operation,
        answer_type=answer_type,
        slots=tuple(traces),
    )
