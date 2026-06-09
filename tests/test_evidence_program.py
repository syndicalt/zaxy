from __future__ import annotations

from dataclasses import dataclass

from zaxy.evidence_program import (
    EvidenceSlotSpec,
    TemporalEvidenceProgramSpec,
    TemporalEvidenceRow,
    execute_temporal_evidence_program,
    trace_evidence_program,
)


@dataclass(frozen=True)
class Row:
    kind: str
    source_group: str
    exclude_reason: str = ""


def test_evidence_program_trace_reports_complete_slot_coverage() -> None:
    trace = trace_evidence_program(
        operation="sum_values",
        answer_type="sum",
        slots=(EvidenceSlotSpec(name="currency", kind="currency", min_source_groups=2),),
        rows=(
            Row(kind="currency", source_group="answer-1"),
            Row(kind="currency", source_group="answer-2"),
            Row(kind="currency", source_group="answer-3", exclude_reason="duplicate_identity"),
        ),
    )

    assert trace.complete is True
    assert trace.missing_slots == ()
    assert trace.to_dict() == {
        "version": "evidence_program_v1",
        "operation": "sum_values",
        "answer_type": "sum",
        "complete": True,
        "missing_slots": [],
        "slots": [
            {
                "name": "currency",
                "kind": "currency",
                "required": True,
                "min_source_groups": 2,
                "min_rows": 0,
                "source_groups": ["answer-1", "answer-2"],
                "row_count": 2,
                "missing": False,
            }
        ],
    }


def test_evidence_program_trace_reports_missing_required_slot() -> None:
    trace = trace_evidence_program(
        operation="temporal_sequence",
        answer_type="ordered_list",
        slots=(
            EvidenceSlotSpec(name="event_date", kind="date", min_source_groups=3),
            EvidenceSlotSpec(name="context", kind="source", required=False),
        ),
        rows=(
            Row(kind="date", source_group="answer-1"),
            Row(kind="date", source_group="answer-2"),
        ),
    )

    assert trace.complete is False
    assert trace.missing_slots == ("event_date",)
    payload = trace.to_dict()
    assert payload["complete"] is False
    assert payload["missing_slots"] == ["event_date"]
    assert payload["slots"][0]["source_groups"] == ["answer-1", "answer-2"]


def test_evidence_program_trace_supports_minimum_row_coverage() -> None:
    trace = trace_evidence_program(
        operation="temporal_sequence",
        answer_type="ordered_list",
        slots=(EvidenceSlotSpec(name="temporal_event", kind="temporal_event", min_rows=3),),
        rows=(
            Row(kind="temporal_event", source_group="answer-1"),
            Row(kind="temporal_event", source_group="answer-1"),
            Row(kind="temporal_event", source_group="answer-2"),
        ),
    )

    assert trace.complete is True
    payload = trace.to_dict()
    assert payload["slots"][0]["source_groups"] == ["answer-1", "answer-2"]
    assert payload["slots"][0]["row_count"] == 3
    assert payload["slots"][0]["min_rows"] == 3


def test_temporal_evidence_program_counts_before_boundary_and_excludes_target() -> None:
    program = execute_temporal_evidence_program(
        TemporalEvidenceProgramSpec(
            operator="count_before",
            event_class_terms=("workshop",),
            boundary_terms=("summit",),
        ),
        rows=(
            TemporalEvidenceRow(
                event_id="workshop-a",
                label="frontend workshop",
                event_date="2026-03-01",
                source_group="workshop-a",
                citation="eventloom://events/1#aa",
                canonical_identity="workshop:a",
                raw_span="frontend workshop on March 1",
            ),
            TemporalEvidenceRow(
                event_id="summit",
                label="engineering summit",
                event_date="2026-03-15",
                source_group="summit",
                citation="eventloom://events/2#bb",
                canonical_identity="summit",
                raw_span="engineering summit on March 15",
            ),
            TemporalEvidenceRow(
                event_id="workshop-b",
                label="backend workshop",
                event_date="2026-03-20",
                source_group="workshop-b",
                citation="eventloom://events/3#cc",
                canonical_identity="workshop:b",
                raw_span="backend workshop on March 20",
            ),
        ),
    )

    assert program.complete is True
    assert program.answer_value == 1
    assert [row.event_id for row in program.included_rows] == ["workshop-a"]
    assert {row.event_id: row.exclude_reason for row in program.excluded_rows} == {
        "summit": "temporal_count_target",
        "workshop-b": "temporal_count_outside_window",
    }
    assert program.to_dict()["boundary"] == {
        "event_id": "summit",
        "source_group": "summit",
        "event_date": "2026-03-15",
    }


def test_temporal_evidence_program_deduplicates_and_reports_missing_dates() -> None:
    program = execute_temporal_evidence_program(
        TemporalEvidenceProgramSpec(
            operator="count_after",
            event_class_terms=("maintenance",),
            boundary_terms=("kickoff",),
        ),
        rows=(
            TemporalEvidenceRow(
                event_id="kickoff",
                label="maintenance kickoff",
                event_date="2026-04-01",
                source_group="kickoff",
                citation="eventloom://events/1#aa",
                canonical_identity="kickoff",
                raw_span="maintenance kickoff on April 1",
            ),
            TemporalEvidenceRow(
                event_id="maint-a-1",
                label="filter maintenance",
                event_date="2026-04-03",
                source_group="maint-a",
                citation="eventloom://events/2#bb",
                canonical_identity="maintenance:filter",
                raw_span="filter maintenance on April 3",
            ),
            TemporalEvidenceRow(
                event_id="maint-a-2",
                label="filter maintenance",
                event_date="2026-04-03",
                source_group="maint-a-repeat",
                citation="eventloom://events/3#cc",
                canonical_identity="maintenance:filter",
                raw_span="repeated note about filter maintenance on April 3",
            ),
            TemporalEvidenceRow(
                event_id="maint-missing",
                label="pump maintenance",
                event_date=None,
                source_group="maint-missing",
                citation="eventloom://events/4#dd",
                canonical_identity="maintenance:pump",
                raw_span="pump maintenance happened",
            ),
        ),
    )

    assert program.complete is True
    assert program.answer_value == 1
    payload = program.to_dict()
    assert payload["coverage"]["included_count"] == 1
    assert payload["coverage"]["excluded_reasons"] == {
        "duplicate_identity": 1,
        "missing_temporal_count_date": 1,
        "temporal_count_target": 1,
    }
