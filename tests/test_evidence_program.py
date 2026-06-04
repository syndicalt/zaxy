from __future__ import annotations

from dataclasses import dataclass

from zaxy.evidence_program import EvidenceSlotSpec, trace_evidence_program


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
                "source_groups": ["answer-1", "answer-2"],
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
