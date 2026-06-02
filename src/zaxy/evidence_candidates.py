"""Typed evidence candidates for retrieval-time answer synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zaxy.synthesis import (
    EvidenceLedger,
    EvidenceLedgerRow,
    build_age_average_ledger,
    build_count_ledger,
    build_currency_ledger,
    build_date_ledger,
    build_duration_ledger,
    build_synthesis_plan,
    synthesis_operation_for_plan,
)


@dataclass(frozen=True)
class EvidenceProjection:
    """Rendered evidence candidates plus the source groups that support them."""

    lines: tuple[str, ...]
    source_groups: tuple[str, ...]
    ledger_rows: tuple[dict[str, Any], ...] = ()
    answer_candidates: tuple[dict[str, object], ...] = ()
    operations: tuple[dict[str, object], ...] = ()
    result: dict[str, object] | None = None


def aggregate_candidate_projection(query: str, contexts: list[str]) -> EvidenceProjection:
    """Build deterministic aggregate answer candidates from cited contexts."""
    lines: list[str] = []
    source_groups: list[str] = []
    ledger_rows: list[dict[str, Any]] = []
    answer_candidates: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    rank = 1
    count_ledger = build_count_ledger(query, contexts)
    count_projection = synthesis_operation_for_plan(count_ledger.plan).execute(
        count_ledger,
        query=query,
        rank=rank,
    )
    if count_projection.lines:
        lines.extend(count_projection.lines)
        source_groups.extend(count_projection.support_source_groups)
        ledger_rows.extend(_ledger_row_payloads(count_ledger))
        if count_projection.answer_candidate:
            answer_candidates.append(count_projection.answer_candidate)
            operations.append(_operation_payload(count_ledger, count_projection.answer_candidate))
        rank += 1
    currency_ledger = build_currency_ledger(query, contexts)
    currency_projection = synthesis_operation_for_plan(currency_ledger.plan).execute(
        currency_ledger,
        query=query,
        rank=rank,
    )
    if currency_projection.lines:
        lines.extend(currency_projection.lines)
        source_groups.extend(currency_projection.support_source_groups)
        ledger_rows.extend(_ledger_row_payloads(currency_ledger))
        if currency_projection.answer_candidate:
            answer_candidates.append(currency_projection.answer_candidate)
            operations.append(_operation_payload(currency_ledger, currency_projection.answer_candidate))
        rank += 1
    age_average_ledger = build_age_average_ledger(query, contexts)
    age_average_projection = synthesis_operation_for_plan(age_average_ledger.plan).execute(
        age_average_ledger,
        query=query,
        rank=rank,
    )
    if age_average_projection.lines:
        lines.extend(age_average_projection.lines)
        source_groups.extend(age_average_projection.support_source_groups)
        ledger_rows.extend(_ledger_row_payloads(age_average_ledger))
        if age_average_projection.answer_candidate:
            answer_candidates.append(age_average_projection.answer_candidate)
            operations.append(_operation_payload(age_average_ledger, age_average_projection.answer_candidate))
        rank += 1
    if not count_projection.lines:
        duration_ledger = build_duration_ledger(query, contexts)
        duration_projection = synthesis_operation_for_plan(duration_ledger.plan).execute(
            duration_ledger,
            query=query,
            rank=rank,
        )
        if duration_projection.lines:
            lines.extend(duration_projection.lines)
            source_groups.extend(duration_projection.support_source_groups)
            ledger_rows.extend(_ledger_row_payloads(duration_ledger))
            if duration_projection.answer_candidate:
                answer_candidates.append(duration_projection.answer_candidate)
                operations.append(_operation_payload(duration_ledger, duration_projection.answer_candidate))
            rank += 1
    date_ledger = build_date_ledger(query, contexts)
    date_projection = synthesis_operation_for_plan(date_ledger.plan).execute(
        date_ledger,
        query=query,
        rank=rank,
    )
    if date_projection.lines:
        lines.extend(date_projection.lines)
        source_groups.extend(date_projection.support_source_groups)
        ledger_rows.extend(_ledger_row_payloads(date_ledger))
        if date_projection.answer_candidate:
            answer_candidates.append(date_projection.answer_candidate)
            operations.append(_operation_payload(date_ledger, date_projection.answer_candidate))
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=tuple(dict.fromkeys(source_groups)),
        ledger_rows=tuple(ledger_rows),
        answer_candidates=tuple(answer_candidates),
        operations=tuple(operations),
        result=_result_payload(answer_candidates[0]) if answer_candidates else None,
    )


def _operation_payload(ledger: EvidenceLedger, candidate: dict[str, object]) -> dict[str, object]:
    return {
        "name": ledger.plan.operation,
        "answer_type": ledger.plan.answer_type,
        "kind": ledger.plan.required_kinds[0] if ledger.plan.required_kinds else str(candidate.get("type", "")),
        "answer_key": str(candidate.get("answer_key", "")),
        "support_source_ids": _object_string_list(candidate.get("support_source_ids")),
        "excluded_source_ids": _object_string_list(candidate.get("excluded_source_ids")),
    }


def _object_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _result_payload(candidate: dict[str, object]) -> dict[str, object]:
    return {
        key: candidate[key]
        for key in ("answer_key", "answer", "confidence", "support_source_ids", "excluded_source_ids")
        if key in candidate
    }


def _ledger_row_payloads(ledger: EvidenceLedger) -> list[dict[str, Any]]:
    return [_ledger_row_payload(row) for row in ledger.rows]


def _ledger_row_payload(row: EvidenceLedgerRow) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fact_id": row.fact_id,
        "source_group": row.source_group,
        "citation": row.citation,
        "kind": row.kind,
        "value": row.value,
        "unit": row.unit,
        "label": row.label,
        "raw_span": row.raw_span,
        "normalized_identity": row.normalized_identity,
        "include_reason": row.include_reason,
        "exclude_reason": row.exclude_reason,
        "confidence": row.confidence,
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def aggregate_evidence_score(query: str, context: str) -> int:
    """Return whether one context contains typed evidence before full synthesis succeeds."""
    score = 0
    for ledger in _ranking_evidence_ledgers(query, context):
        included = ledger.included()
        if not included:
            continue
        score += len(included) * 3
        score += max(row.relevance for row in included)
    return score


def _ranking_evidence_ledgers(query: str, context: str) -> tuple[EvidenceLedger, ...]:
    """Build only the ledger families needed for single-source ranking."""
    plan = build_synthesis_plan(query)
    required = set(plan.required_kinds)
    if "date" in required:
        return (build_date_ledger(query, [context], plan=plan),)
    ledgers = []
    if "event" in required:
        ledgers.append(build_count_ledger(query, [context], plan=plan))
    if "currency" in required:
        ledgers.append(build_currency_ledger(query, [context], plan=plan))
    if "duration" in required or "number" in required:
        ledgers.append(build_duration_ledger(query, [context], plan=plan))
    if not ledgers:
        ledgers.append(build_date_ledger(query, [context], plan=plan))
    return tuple(ledgers)


def aggregate_candidate_lines(query: str, contexts: list[str]) -> list[str]:
    """Render deterministic aggregate answer candidates from cited contexts."""
    return list(aggregate_candidate_projection(query, contexts).lines)
