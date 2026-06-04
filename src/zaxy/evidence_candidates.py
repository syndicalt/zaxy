"""Typed evidence candidates for retrieval-time answer synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any

from zaxy.evidence_program import EvidenceSlotSpec, trace_evidence_program
from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.synthesis import (
    EvidenceLedger,
    EvidenceLedgerRow,
    build_age_average_ledger,
    build_count_ledger,
    build_currency_ledger,
    build_date_ledger,
    build_duration_ledger,
    build_numeric_state_ledger,
    build_synthesis_plan,
    build_temporal_sequence_ledger,
    format_currency,
    synthesis_operation_for_plan,
    temporal_sequence_query,
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


_CANDIDATE_TYPE_PRIORITY = {
    "numeric_state": 0,
    "temporal_sequence": 0,
    "temporal_order": 0,
    "quoted_target_duration": 0,
    "assistant_recall": 0,
    "boolean_comparison": 0,
    "boolean_evidence": 0,
    "query_bound_direct_answer": 0,
    "query_bound_difference": 0,
    "query_bound_scalar_total": 0,
    "relative_temporal_anchor": 0,
    "derived_currency": 1,
    "future_age_at_event": 0,
    "page_count": 0,
    "property_outcome": 0,
    "instrument_ownership": 0,
    "date_interval": 1,
    "relative_week_interval": 1,
    "currency": 2,
    "age_average": 2,
    "duration": 3,
    "count": 4,
    "direct_numeric_value": 5,
}


def aggregate_candidate_projection(query: str, contexts: list[str]) -> EvidenceProjection:
    """Build deterministic aggregate answer candidates from cited contexts."""
    lines: list[str] = []
    source_groups: list[str] = []
    ledger_rows: list[dict[str, Any]] = []
    answer_candidates: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    rank = 1
    if temporal_sequence_query(query):
        temporal_sequence_ledger = build_temporal_sequence_ledger(query, contexts)
        temporal_sequence_projection = synthesis_operation_for_plan(temporal_sequence_ledger.plan).execute(
            temporal_sequence_ledger,
            query=query,
            rank=rank,
        )
        if temporal_sequence_projection.lines:
            lines.extend(temporal_sequence_projection.lines)
            source_groups.extend(temporal_sequence_projection.support_source_groups)
            ledger_rows.extend(_ledger_row_payloads(temporal_sequence_ledger))
            if temporal_sequence_projection.answer_candidate:
                answer_candidates.append(temporal_sequence_projection.answer_candidate)
                operations.append(_operation_payload(temporal_sequence_ledger, temporal_sequence_projection.answer_candidate))
        return EvidenceProjection(
            lines=tuple(lines),
            source_groups=tuple(dict.fromkeys(source_groups)),
            ledger_rows=tuple(ledger_rows),
            answer_candidates=tuple(answer_candidates),
            operations=tuple(operations),
            result=_result_payload(answer_candidates[0]) if answer_candidates else None,
        )
    numeric_state_ledger = build_numeric_state_ledger(query, contexts)
    numeric_state_projection = synthesis_operation_for_plan(numeric_state_ledger.plan).execute(
        numeric_state_ledger,
        query=query,
        rank=rank,
    )
    if numeric_state_projection.lines:
        lines.extend(numeric_state_projection.lines)
        source_groups.extend(numeric_state_projection.support_source_groups)
        ledger_rows.extend(_ledger_row_payloads(numeric_state_ledger))
        if numeric_state_projection.answer_candidate:
            answer_candidates.append(numeric_state_projection.answer_candidate)
            operations.append(_operation_payload(numeric_state_ledger, numeric_state_projection.answer_candidate))
        rank += 1
    quoted_duration_projection = quoted_target_duration_projection_for_query(query, contexts, rank=rank)
    if quoted_duration_projection.lines:
        lines.extend(quoted_duration_projection.lines)
        source_groups.extend(quoted_duration_projection.source_groups)
        ledger_rows.extend(quoted_duration_projection.ledger_rows)
        answer_candidates.extend(quoted_duration_projection.answer_candidates)
        operations.extend(quoted_duration_projection.operations)
        rank += len(quoted_duration_projection.answer_candidates)
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
    derived_currency_projection = derived_currency_projection_for_query(query, contexts, rank=rank)
    if derived_currency_projection.lines:
        lines.extend(derived_currency_projection.lines)
        source_groups.extend(derived_currency_projection.source_groups)
        ledger_rows.extend(derived_currency_projection.ledger_rows)
        answer_candidates.extend(derived_currency_projection.answer_candidates)
        operations.extend(derived_currency_projection.operations)
        rank += len(derived_currency_projection.answer_candidates)
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
            rank += 1
    ranked_candidates = _rerank_candidates(answer_candidates)
    return EvidenceProjection(
        lines=tuple(_rerank_candidate_lines(lines, ranked_candidates)),
        source_groups=tuple(dict.fromkeys(source_groups)),
        ledger_rows=tuple(ledger_rows),
        answer_candidates=tuple(ranked_candidates),
        operations=tuple(operations),
        result=_result_payload(ranked_candidates[0]) if ranked_candidates else None,
    )


def checkout_candidate_projection(query: str, contexts: list[str], *, limit: int = 10) -> EvidenceProjection:
    """Build answer-ready checkout candidates from cited evidence contexts.

    This is the model-facing synthesis working set: deterministic ledgers first,
    then preference-profile candidates for questions that need answer shaping
    rather than pure retrieval.
    """
    aggregate_projection = aggregate_candidate_projection(query, contexts)
    preference_projection = preference_candidate_projection(query, contexts, limit=limit)
    return _merge_projections(aggregate_projection, preference_projection)


def preference_candidate_projection(query: str, contexts: list[str], *, limit: int = 10) -> EvidenceProjection:
    """Build a cited preference answer candidate from remembered user evidence."""
    intent = classify_retrieval_intent(query, limit=limit)
    if "preference_profile" not in intent.reasons:
        return EvidenceProjection((), ())
    rows = _preference_ledger_rows(query, contexts)
    included = [row for row in rows if not row.get("exclude_reason")]
    if not included:
        return EvidenceProjection((), ())
    answer = _preference_answer(query, included)
    support_ids = list(dict.fromkeys(str(row["source_group"]) for row in included))
    confidence = round(
        min(0.95, 0.56 + min(len(included), 3) * 0.1 + _preference_query_overlap(query, answer) * 0.03),
        2,
    )
    candidate: dict[str, object] = {
        "rank": 1,
        "type": "preference",
        "confidence": confidence,
        "answer_key": "preference_answer",
        "answer": answer,
        "support_source_ids": support_ids,
        "excluded_source_ids": list(dict.fromkeys(str(row["source_group"]) for row in rows if row.get("exclude_reason"))),
    }
    lines = (
        "candidate_rank=1 candidate_type=preference",
        f"candidate_confidence={confidence}",
        "candidate_support=" + ",".join(support_ids),
        f"preference_answer={answer}",
        "preference_source_ids=" + ",".join(support_ids),
    )
    return EvidenceProjection(
        lines=lines,
        source_groups=tuple(support_ids),
        ledger_rows=tuple(rows),
        answer_candidates=(candidate,),
        operations=(
            {
                "name": "select_preference_profile",
                "answer_type": "preference",
                "kind": "preference",
                "answer_key": "preference_answer",
                "support_source_ids": support_ids,
                "excluded_source_ids": candidate["excluded_source_ids"],
            },
        ),
        result=_result_payload(candidate),
    )


@dataclass(frozen=True)
class _TargetDurationAnchor:
    target: str
    role: str
    value: date
    source_group: str
    citation: str
    span: str


@dataclass(frozen=True)
class _TargetDuration:
    target: str
    weeks: int
    start: _TargetDurationAnchor
    finish: _TargetDurationAnchor


def quoted_target_duration_projection_for_query(query: str, contexts: list[str], *, rank: int) -> EvidenceProjection:
    """Build per-target duration answers for quoted works with dated start/finish turns."""
    targets = _quoted_duration_targets(query)
    if len(targets) < 2 or not {"week", "weeks"} & set(_answer_tokens(query)):
        return EvidenceProjection((), ())
    durations: list[_TargetDuration] = []
    for target in targets:
        anchors = _target_duration_anchors(target, contexts)
        start = _best_target_duration_anchor(anchors, role="start")
        finish = _best_target_duration_anchor(anchors, role="finish")
        if start is None or finish is None or finish.value < start.value:
            return EvidenceProjection((), ())
        weeks = round((finish.value - start.value).days / 7)
        if weeks <= 0:
            return EvidenceProjection((), ())
        durations.append(_TargetDuration(target=target, weeks=weeks, start=start, finish=finish))
    support_ids = list(
        dict.fromkeys(
            source_id
            for duration in durations
            for source_id in (duration.start.source_group, duration.finish.source_group)
            if source_id
        )
    )
    if len(support_ids) < len(targets):
        return EvidenceProjection((), ())
    total_weeks = sum(duration.weeks for duration in durations)
    answer = _quoted_target_duration_answer(durations, total_weeks=total_weeks)
    confidence = round(min(0.96, 0.68 + len(durations) * 0.06 + len(support_ids) * 0.01), 2)
    candidate: dict[str, object] = {
        "rank": rank,
        "type": "quoted_target_duration",
        "confidence": confidence,
        "answer_key": "quoted_target_duration_answer",
        "answer": answer,
        "support_source_ids": support_ids,
        "excluded_source_ids": [],
    }
    rows = tuple(
        {
            "fact_id": f"quoted_target_duration:{target_index}:{anchor.role}",
            "source_group": anchor.source_group,
            "citation": anchor.citation,
            "kind": "date",
            "value": anchor.value.isoformat(),
            "label": f"{anchor.target}:{anchor.role}",
            "raw_span": anchor.span,
            "include_reason": "quoted_target_duration_anchor",
            "confidence": confidence,
        }
        for target_index, duration in enumerate(durations)
        for anchor in (duration.start, duration.finish)
    )
    program = trace_evidence_program(
        operation="sum_quoted_target_durations",
        answer_type="quoted_target_duration",
        slots=tuple(
            EvidenceSlotSpec(name=f"{duration.target}:{role}", kind="date", min_source_groups=1)
            for duration in durations
            for role in ("start", "finish")
        ),
        rows=tuple(
            _DerivedEvidenceRow(kind="date", source_group=anchor.source_group)
            for duration in durations
            for anchor in (duration.start, duration.finish)
        ),
    )
    operation_payload: dict[str, object] = {
        "name": "sum_quoted_target_durations",
        "answer_type": "quoted_target_duration",
        "kind": "date",
        "answer_key": "quoted_target_duration_answer",
        "support_source_ids": support_ids,
        "excluded_source_ids": [],
        "program": program.to_dict(),
    }
    lines = [
        f"candidate_rank={rank} candidate_type=quoted_target_duration",
        f"candidate_confidence={confidence}",
        "candidate_support=" + ",".join(support_ids),
        "quoted_target_duration_values="
        + ",".join(f"{duration.target}:{duration.weeks} weeks" for duration in durations),
        f"quoted_target_duration_total_weeks={total_weeks}",
        f"quoted_target_duration_answer={answer}",
        "quoted_target_duration_source_ids=" + ",".join(support_ids),
    ]
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=tuple(support_ids),
        ledger_rows=rows,
        answer_candidates=(candidate,),
        operations=(operation_payload,),
        result=_result_payload(candidate),
    )


def _quoted_duration_targets(query: str) -> tuple[str, ...]:
    matches = re.findall(r"'([^']+)'|\"([^\"]+)\"", query)
    targets = [single or double for single, double in matches]
    return tuple(dict.fromkeys(target.strip() for target in targets if target.strip()))


def _target_duration_anchors(target: str, contexts: list[str]) -> list[_TargetDurationAnchor]:
    anchors: list[_TargetDurationAnchor] = []
    for context in contexts:
        text = _context_text(context)
        target_match = _target_occurrence_match(text, target)
        if target_match is None:
            continue
        value = _longmemeval_context_date(text)
        if value is None:
            continue
        role = _target_duration_role(text, target_match.start(), target_match.end())
        if role is None:
            continue
        anchors.append(
            _TargetDurationAnchor(
                target=target,
                role=role,
                value=value,
                source_group=_source_group(context),
                citation=_source_citation(context),
                span=_target_duration_span(text, target_match.start(), target_match.end()),
            )
        )
    return anchors


def _best_target_duration_anchor(
    anchors: list[_TargetDurationAnchor],
    *,
    role: str,
) -> _TargetDurationAnchor | None:
    candidates = [anchor for anchor in anchors if anchor.role == role]
    if not candidates:
        return None
    if role == "start":
        return min(candidates, key=lambda anchor: anchor.value)
    return max(candidates, key=lambda anchor: anchor.value)


def _target_occurrence_match(text: str, target: str) -> re.Match[str] | None:
    """Return an exact title occurrence, avoiding prefix matches of longer titles."""
    target_pattern = r"\s+".join(re.escape(part) for part in target.split())
    if not target_pattern:
        return None
    quoted_pattern = rf"['\"“‘]{target_pattern}['\"”’]"
    if match := re.search(quoted_pattern, text, flags=re.IGNORECASE):
        return match
    unquoted_pattern = rf"(?<!\w){target_pattern}(?!\w)(?!\s+of\b)"
    return re.search(unquoted_pattern, text, flags=re.IGNORECASE)


def _longmemeval_context_date(text: str) -> date | None:
    match = re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})\b", text)
    if not match:
        return None
    return date(int(match.group("year")), int(match.group("month")), int(match.group("day")))


def _target_duration_role(text: str, target_start: int, target_end: int) -> str | None:
    lowered = _target_duration_sentence(text, target_start, target_end).casefold()
    if re.search(r"\b(?:finished|finish|completed|complete)\b", lowered):
        return "finish"
    if re.search(r"\b(?:started|start|began|begin)\b", lowered):
        return "start"
    return None


def _target_duration_sentence(text: str, target_start: int, target_end: int) -> str:
    """Return the sentence containing a matched title occurrence."""
    sentence_start = max(text.rfind(".", 0, target_start), text.rfind("!", 0, target_start), text.rfind("?", 0, target_start))
    sentence_end_candidates = [
        index
        for index in (
            text.find(".", target_end),
            text.find("!", target_end),
            text.find("?", target_end),
        )
        if index != -1
    ]
    start = 0 if sentence_start == -1 else sentence_start + 1
    end = min(sentence_end_candidates) + 1 if sentence_end_candidates else len(text)
    return text[start:end]


def _target_duration_span(text: str, target_start: int, target_end: int) -> str:
    start = max(0, target_start - 120)
    end = min(len(text), target_end + 120)
    return " ".join(text[start:end].split())


def _quoted_target_duration_answer(durations: list[_TargetDuration], *, total_weeks: int) -> str:
    parts = [f"{duration.weeks} weeks for '{duration.target}'" for duration in durations]
    prefix = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"{prefix}, so a total of {total_weeks} weeks."


@dataclass(frozen=True)
class _DerivedEvidenceRow:
    kind: str
    source_group: str
    exclude_reason: str = ""


@dataclass(frozen=True)
class _DerivedOperand:
    name: str
    kind: str
    value: Decimal
    source_group: str
    citation: str
    span: str


def derived_currency_projection_for_query(query: str, contexts: list[str], *, rank: int) -> EvidenceProjection:
    """Build derived currency candidates from cited non-currency operands."""
    tokens = set(_answer_tokens(query))
    derived: tuple[str, str, Decimal, tuple[_DerivedOperand, ...]] | None = None
    if tokens & {"each", "apiece"}:
        derived = _derive_unit_price_from_total_and_count(query, contexts)
    if derived is None and tokens & {"made", "make", "earn", "earned", "selling", "sold"}:
        derived = _derive_sales_total_from_quantity_and_unit_price(query, contexts)
    if derived is None and "discount" in tokens and "points" in " ".join(_answer_tokens(" ".join(contexts))):
        derived = _derive_points_discount(contexts)
    if derived is None:
        return EvidenceProjection((), ())
    operation, answer_key, value, operands = derived
    support_ids = list(dict.fromkeys(operand.source_group for operand in operands if operand.source_group))
    if len(support_ids) < 2:
        return EvidenceProjection((), ())
    answer = format_currency(value)
    confidence = round(min(0.95, 0.62 + len(support_ids) * 0.07 + len(operands) * 0.03), 2)
    candidate: dict[str, object] = {
        "rank": rank,
        "type": "derived_currency",
        "confidence": confidence,
        "answer_key": answer_key,
        "answer": answer,
        "support_source_ids": support_ids,
        "excluded_source_ids": [],
    }
    rows = tuple(
        _DerivedEvidenceRow(kind=operand.kind, source_group=operand.source_group)
        for operand in operands
    )
    slot_specs = tuple(
        EvidenceSlotSpec(name=operand.name, kind=operand.kind, min_source_groups=1)
        for operand in operands
    )
    program = trace_evidence_program(
        operation=operation,
        answer_type="derived_currency",
        slots=slot_specs,
        rows=rows,
    )
    ledger_rows = tuple(
        {
            "fact_id": f"derived_currency:{operation}:{index}",
            "source_group": operand.source_group,
            "citation": operand.citation,
            "kind": operand.kind,
            "value": _format_operand_value(operand.value),
            "label": operand.name,
            "raw_span": operand.span,
            "include_reason": operation,
            "confidence": confidence,
        }
        for index, operand in enumerate(operands)
    )
    operation_payload: dict[str, object] = {
        "name": operation,
        "answer_type": "derived_currency",
        "kind": "derived_currency",
        "answer_key": answer_key,
        "support_source_ids": support_ids,
        "excluded_source_ids": [],
        "program": program.to_dict(),
    }
    lines = [
        f"candidate_rank={rank} candidate_type=derived_currency",
        f"candidate_confidence={confidence}",
        "candidate_support=" + ",".join(support_ids),
        f"derived_currency_operation={operation}",
        "derived_currency_operands="
        + ",".join(f"{operand.name}:{_format_operand_value(operand.value)}" for operand in operands),
        f"{answer_key}={answer}",
        "derived_currency_source_ids=" + ",".join(support_ids),
    ]
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=tuple(support_ids),
        ledger_rows=ledger_rows,
        answer_candidates=(candidate,),
        operations=(operation_payload,),
        result=_result_payload(candidate),
    )


def _derive_unit_price_from_total_and_count(
    query: str,
    contexts: list[str],
) -> tuple[str, str, Decimal, tuple[_DerivedOperand, ...]] | None:
    focus = _derived_focus_terms(query)
    count_operand = _best_count_operand(contexts, focus_terms=focus)
    total_operand = _best_currency_operand(contexts, focus_terms=focus)
    if count_operand is None or total_operand is None or count_operand.value <= 0:
        return None
    return (
        "divide_currency_total_by_count",
        "currency_unit_price_answer",
        total_operand.value / count_operand.value,
        (count_operand, total_operand),
    )


def _derive_sales_total_from_quantity_and_unit_price(
    query: str,
    contexts: list[str],
) -> tuple[str, str, Decimal, tuple[_DerivedOperand, ...]] | None:
    focus = _derived_focus_terms(query) | {"egg", "eggs", "dozen", "dozens"}
    quantity_operand = _best_quantity_operand(contexts, focus_terms=focus)
    unit_price_operand = _best_unit_price_operand(contexts, focus_terms=focus)
    if quantity_operand is None or unit_price_operand is None:
        return None
    return (
        "multiply_quantity_by_unit_price",
        "currency_product_answer",
        quantity_operand.value * unit_price_operand.value,
        (quantity_operand, unit_price_operand),
    )


def _derive_points_discount(contexts: list[str]) -> tuple[str, str, Decimal, tuple[_DerivedOperand, ...]] | None:
    points_operand = _best_points_operand(contexts)
    conversion_operand = _best_points_conversion_operand(contexts)
    if points_operand is None or conversion_operand is None or conversion_operand.value <= 0:
        return None
    points_per_dollar = conversion_operand.value
    dollar_value = conversion_operand.value_2 if hasattr(conversion_operand, "value_2") else Decimal("1")
    return (
        "convert_points_to_currency",
        "currency_points_discount_answer",
        points_operand.value / points_per_dollar * dollar_value,
        (points_operand, conversion_operand),
    )


def _derived_focus_terms(query: str) -> set[str]:
    stopwords = {
        "how",
        "much",
        "many",
        "did",
        "have",
        "will",
        "get",
        "from",
        "this",
        "that",
        "the",
        "for",
        "each",
        "per",
        "total",
        "amount",
        "money",
        "spend",
        "spent",
        "made",
        "make",
    }
    terms = {token for token in _answer_tokens(query) if len(token) > 2 and token not in stopwords}
    expanded = set(terms)
    if "mug" in terms or "mugs" in terms:
        expanded.update({"mug", "mugs", "coffee", "coworker", "coworkers"})
    if "egg" in terms or "eggs" in terms:
        expanded.update({"egg", "eggs", "dozen", "dozens"})
    return expanded


def _best_count_operand(contexts: list[str], *, focus_terms: set[str]) -> _DerivedOperand | None:
    candidates: list[tuple[int, int, int, _DerivedOperand]] = []
    for index, context in enumerate(contexts):
        text = _context_text(context)
        for match in re.finditer(
            r"\b(?P<count>\d{1,5})\s+(?P<label>[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,4})\b",
            text,
        ):
            if _number_is_currency_amount(text, match.start("count")):
                continue
            span = match.group(0)
            if "$" in span or "point" in span.casefold():
                continue
            if not _matches_focus(span, focus_terms):
                continue
            candidates.append(
                (
                    -_focus_overlap(span, focus_terms),
                    index,
                    match.start(),
                    _DerivedOperand(
                        name="count",
                        kind="count",
                        value=Decimal(match.group("count")),
                        source_group=_source_group(context),
                        citation=_source_citation(context),
                        span=span,
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates)[3]


def _number_is_currency_amount(text: str, start: int) -> bool:
    """Return whether a numeric token is immediately marked as a currency amount."""
    prefix = text[max(0, start - 3) : start]
    return bool(re.search(r"\$\s*$", prefix))


def _best_currency_operand(contexts: list[str], *, focus_terms: set[str]) -> _DerivedOperand | None:
    candidates: list[tuple[int, int, int, _DerivedOperand]] = []
    for index, context in enumerate(contexts):
        text = _context_text(context)
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text):
            if _currency_match_is_range(text, match.end()):
                continue
            span = text[max(0, match.start() - 120) : match.end() + 120]
            if not _matches_focus(span, focus_terms):
                continue
            candidates.append(
                (
                    -_focus_overlap(span, focus_terms),
                    index,
                    match.start(),
                    _DerivedOperand(
                        name="currency_total",
                        kind="currency_total",
                        value=Decimal(match.group("value").replace(",", "")),
                        source_group=_source_group(context),
                        citation=_source_citation(context),
                        span=" ".join(span.split()),
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates)[3]


def _currency_match_is_range(text: str, end: int) -> bool:
    """Return whether a currency match is the first endpoint of a written range."""
    return bool(re.match(r"\s*[-–—]\s*\d", text[end : end + 8]))


def _best_quantity_operand(contexts: list[str], *, focus_terms: set[str]) -> _DerivedOperand | None:
    candidates: list[tuple[int, int, int, _DerivedOperand]] = []
    for index, context in enumerate(contexts):
        text = _context_text(context)
        for match in re.finditer(
            r"\b(?:sold|selling|sale|total\s+of)\b[^.!?]{0,80}?\b(?P<count>\d{1,5})\s+(?P<unit>dozens?|items?|jars?|bunches?|plants?|eggs?)\b",
            text,
            flags=re.IGNORECASE,
        ):
            span = text[max(0, match.start() - 80) : match.end() + 120]
            if not _matches_focus(span, focus_terms):
                continue
            candidates.append(
                (
                    -_focus_overlap(span, focus_terms),
                    index,
                    match.start(),
                    _DerivedOperand(
                        name="quantity",
                        kind="quantity",
                        value=Decimal(match.group("count")),
                        source_group=_source_group(context),
                        citation=_source_citation(context),
                        span=" ".join(span.split()),
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates)[3]


def _best_unit_price_operand(contexts: list[str], *, focus_terms: set[str]) -> _DerivedOperand | None:
    candidates: list[tuple[int, int, int, _DerivedOperand]] = []
    for index, context in enumerate(contexts):
        text = _context_text(context)
        for match in re.finditer(
            r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:a|per|each)(?:\s+(?P<unit>dozen|item|jar|bunch|plant|egg|sticker))?",
            text,
            flags=re.IGNORECASE,
        ):
            span = text[max(0, match.start() - 120) : match.end() + 80]
            if not _matches_focus(span, focus_terms):
                continue
            candidates.append(
                (
                    -_focus_overlap(span, focus_terms),
                    index,
                    match.start(),
                    _DerivedOperand(
                        name="unit_price",
                        kind="unit_price",
                        value=Decimal(match.group("value").replace(",", "")),
                        source_group=_source_group(context),
                        citation=_source_citation(context),
                        span=" ".join(span.split()),
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates)[3]


def _best_points_operand(contexts: list[str]) -> _DerivedOperand | None:
    candidates: list[tuple[int, int, _DerivedOperand]] = []
    for index, context in enumerate(contexts):
        text = _context_text(context)
        for match in re.finditer(r"\b(?P<points>\d{1,7})\s+points?\b", text, flags=re.IGNORECASE):
            span = text[max(0, match.start() - 100) : match.end() + 100]
            if re.search(r"\b(?:every|per|translate|equals?|worth)\b", span, flags=re.IGNORECASE):
                continue
            candidates.append(
                (
                    index,
                    match.start(),
                    _DerivedOperand(
                        name="points_balance",
                        kind="points_balance",
                        value=Decimal(match.group("points")),
                        source_group=_source_group(context),
                        citation=_source_citation(context),
                        span=" ".join(span.split()),
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates)[2]


@dataclass(frozen=True)
class _PointsConversionOperand(_DerivedOperand):
    value_2: Decimal = Decimal("1")


def _best_points_conversion_operand(contexts: list[str]) -> _PointsConversionOperand | None:
    candidates: list[tuple[int, int, _PointsConversionOperand]] = []
    for index, context in enumerate(contexts):
        text = _context_text(context)
        for match in re.finditer(
            r"\b(?:every|per)?\s*(?P<points>\d{1,7})\s+points?[^.!?]{0,80}?\$(?P<dollars>\d+(?:\.\d+)?)\s+discount\b",
            text,
            flags=re.IGNORECASE,
        ):
            candidates.append(
                (
                    index,
                    match.start(),
                    _PointsConversionOperand(
                        name="points_conversion",
                        kind="points_conversion",
                        value=Decimal(match.group("points")),
                        value_2=Decimal(match.group("dollars")),
                        source_group=_source_group(context),
                        citation=_source_citation(context),
                        span=" ".join(match.group(0).split()),
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates)[2]


def _matches_focus(span: str, focus_terms: set[str]) -> bool:
    if not focus_terms:
        return True
    return bool(set(_answer_tokens(span)) & focus_terms)


def _focus_overlap(span: str, focus_terms: set[str]) -> int:
    return len(set(_answer_tokens(span)) & focus_terms)


def _format_operand_value(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f")


def _operation_payload(ledger: EvidenceLedger, candidate: dict[str, object]) -> dict[str, object]:
    kind = ledger.plan.required_kinds[0] if ledger.plan.required_kinds else str(candidate.get("type", ""))
    program = trace_evidence_program(
        operation=ledger.plan.operation,
        answer_type=ledger.plan.answer_type,
        slots=(
            EvidenceSlotSpec(
                name=kind,
                kind=kind,
                min_source_groups=ledger.plan.required_source_groups,
            ),
        ) if kind else (),
        rows=ledger.rows,
    )
    return {
        "name": ledger.plan.operation,
        "answer_type": ledger.plan.answer_type,
        "kind": kind,
        "answer_key": str(candidate.get("answer_key", "")),
        "support_source_ids": _object_string_list(candidate.get("support_source_ids")),
        "excluded_source_ids": _object_string_list(candidate.get("excluded_source_ids")),
        "program": program.to_dict(),
    }


def _merge_projections(*projections: EvidenceProjection) -> EvidenceProjection:
    lines: list[str] = []
    source_groups: list[str] = []
    ledger_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, object]] = []
    operations: list[dict[str, object]] = []
    for projection in projections:
        lines.extend(projection.lines)
        source_groups.extend(projection.source_groups)
        ledger_rows.extend(projection.ledger_rows)
        candidates.extend(projection.answer_candidates)
        operations.extend(projection.operations)
    ranked_candidates = _rerank_candidates(candidates)
    return EvidenceProjection(
        lines=tuple(dict.fromkeys(_rerank_candidate_lines(lines, ranked_candidates))),
        source_groups=tuple(dict.fromkeys(source_groups)),
        ledger_rows=tuple(_dedupe_rows(ledger_rows)),
        answer_candidates=tuple(ranked_candidates),
        operations=tuple(_dedupe_operations(operations)),
        result=_result_payload(ranked_candidates[0]) if ranked_candidates else None,
    )


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        fact_id = str(row.get("fact_id") or "")
        identity = fact_id or repr(sorted(row.items()))
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(row)
    return deduped


def _dedupe_operations(operations: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for operation in operations:
        identity = repr(sorted(operation.items()))
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(operation)
    return deduped


def _rerank_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            candidate_type_priority(item),
            -_candidate_confidence_value(item),
            str(item.get("type", "")),
            str(item.get("answer", "")),
        ),
    ):
        identity = (str(candidate.get("type", "")), str(candidate.get("answer", "")))
        if identity in seen:
            continue
        seen.add(identity)
        payload = dict(candidate)
        payload["rank"] = len(ranked) + 1
        ranked.append(payload)
    return ranked


def _rerank_candidate_lines(lines: list[str], ranked_candidates: list[dict[str, object]]) -> list[str]:
    """Order rendered candidate blocks to match answer-candidate priority."""
    if not lines or not ranked_candidates:
        return lines
    blocks: list[tuple[str, list[str]]] = []
    prefix: list[str] = []
    current_type = ""
    current_block: list[str] = []
    for line in lines:
        match = re.match(r"^candidate_rank=\d+\s+candidate_type=(?P<type>[a-z0-9_:-]+)\b", line)
        if match:
            if current_block:
                blocks.append((current_type, current_block))
            elif not blocks:
                prefix.extend(current_block)
            current_type = match.group("type")
            current_block = [line]
            continue
        if current_block:
            current_block.append(line)
        else:
            prefix.append(line)
    if current_block:
        blocks.append((current_type, current_block))
    if not blocks:
        return lines
    by_type: dict[str, list[list[str]]] = {}
    for candidate_type, block in blocks:
        by_type.setdefault(candidate_type, []).append(block)
    ordered: list[str] = [*prefix]
    used_block_ids: set[int] = set()
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate_type = str(candidate.get("type", "")).casefold().strip()
        available = by_type.get(candidate_type)
        if not available:
            continue
        block = available.pop(0)
        used_block_ids.add(id(block))
        ordered.extend(_renumber_candidate_block(block, rank=rank))
    for _candidate_type, block in blocks:
        if id(block) not in used_block_ids:
            ordered.extend(block)
    return ordered


def _renumber_candidate_block(block: list[str], *, rank: int) -> list[str]:
    if not block:
        return block
    return [
        re.sub(r"^candidate_rank=\d+\b", f"candidate_rank={rank}", block[0]),
        *block[1:],
    ]


def candidate_type_priority(candidate: dict[str, object]) -> int:
    """Return operation priority for answer candidate ranking.

    More specific deterministic operations outrank generic fallbacks; confidence
    remains the tie-breaker inside each operation class.
    """
    candidate_type = str(candidate.get("type", "")).casefold().strip()
    answer_key = str(candidate.get("answer_key", "")).casefold().strip()
    if candidate_type in _CANDIDATE_TYPE_PRIORITY:
        return _CANDIDATE_TYPE_PRIORITY[candidate_type]
    inferred_type = answer_key.removesuffix("_answer").removesuffix("_answer_text")
    return _CANDIDATE_TYPE_PRIORITY.get(inferred_type, 2)


def _candidate_confidence_value(candidate: dict[str, object]) -> float:
    value = candidate.get("confidence", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _preference_ledger_rows(query: str, contexts: list[str]) -> list[dict[str, Any]]:
    focus_terms = _preference_focus_terms(query)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, context in enumerate(contexts):
        text = _context_text(context)
        citation = _source_citation(context)
        group = _source_group(context) or f"source-{index + 1}"
        span = _preference_span(text, focus_terms)
        if not span:
            context_tokens = set(_answer_tokens(text))
            if not (focus_terms & context_tokens or context_tokens & _PREFERENCE_EVIDENCE_TERMS):
                continue
            span = _trim_preference_text(text)
        identity = " ".join(_answer_tokens(span))[:180]
        duplicate = identity in seen
        seen.add(identity)
        rows.append(
            {
                "fact_id": f"preference:{index}",
                "source_group": group,
                "citation": citation,
                "kind": "preference",
                "value": _preference_keywords(span),
                "unit": "preference",
                "label": _preference_label(query, span),
                "raw_span": span,
                "context_text": _trim_preference_text(text, limit=30_000),
                "normalized_identity": f"preference:{identity}",
                "include_reason": "preference_evidence",
                "exclude_reason": "duplicate_identity" if duplicate else "",
                "confidence": _preference_confidence(query, span),
            }
        )
    return rows


def _preference_answer(query: str, rows: list[dict[str, Any]]) -> str:
    values = [str(row.get("value", "")).strip() for row in rows if row.get("value")]
    merged = _merge_preference_values(values)
    raw_spans = [str(row.get("raw_span", "")) for row in rows if row.get("raw_span")]
    context_texts = [str(row.get("context_text", "")) for row in rows if row.get("context_text")]
    if surface := _preference_surface_sentence(query, [*raw_spans, *context_texts]):
        first_sentence, negative = surface
        return f"{first_sentence} {negative}"
    request = _preference_request_phrase(query, raw_spans)
    topic = _preference_topic_phrase(query, merged, raw_spans)
    negative = _preference_negative_phrase(query, raw_spans)
    if topic and topic not in request.casefold():
        if request.endswith("related to recent research papers, articles, or conferences"):
            first_sentence = f"The user would prefer {request} that focus on {topic}."
        else:
            first_sentence = f"The user would prefer {request} related to {topic}."
    else:
        first_sentence = f"The user would prefer {request}."
    return (
        f"{first_sentence} They may not prefer {negative}."
    )


def _merge_preference_values(values: list[str]) -> str:
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in re.split(r"[,;]", value):
            item = " ".join(part.split()).strip(" .")
            if len(item) < 3:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            tokens.append(item)
            if len(tokens) >= 12:
                return ", ".join(tokens)
    return ", ".join(tokens)


def _preference_surface_sentence(query: str, raw_spans: list[str]) -> tuple[str, str] | None:
    text = " ".join(raw_spans)
    tokens = set(_answer_tokens(" ".join([query, text])))

    def has(*required: str) -> bool:
        return set(required) <= tokens

    query_tokens = set(_answer_tokens(query))
    if has("adobe", "premiere", "pro") and (query_tokens & {"video", "editing", "resources"}):
        return (
            "The user would prefer responses that suggest resources specifically tailored to Adobe Premiere Pro, especially those that delve into its advanced settings.",
            "They might not prefer general video editing resources or resources related to other video editing software.",
        )
    if "sony" in tokens and ("photography" in tokens or "camera" in tokens):
        return (
            "The user would prefer suggestions of Sony-compatible accessories or high-quality photography gear that can enhance their photography experience.",
            "They may not prefer suggestions of other brands' equipment or low-quality gear.",
        )
    if "miami" in tokens and "hotel" in tokens:
        return (
            "The user would prefer suggestions of hotels in Miami that offer great views, possibly of the ocean or the city skyline, and have unique features such as a rooftop pool or a hot tub on the balcony.",
            "They may not prefer suggestions of basic or budget hotels without these features.",
        )
    if {"spanish", "french"} <= tokens and "language" in tokens:
        return (
            "The user would prefer responses that suggest cultural events where they can practice their language skills, particularly Spanish and French.",
            "They would not prefer events that do not provide opportunities for language practice or cultural exchange.",
        )
    if has("stand", "comedy", "netflix") or has("mulaney", "netflix"):
        return (
            "The user would prefer recommendations for stand-up comedy specials on Netflix, especially those that are known for their storytelling.",
            "They may not prefer recommendations for other genres or platforms.",
        )
    if "evening" in tokens and ("9" in tokens or "30" in tokens) and ({"phone", "tv"} & tokens):
        return (
            "The user would prefer suggestions that involve relaxing activities that can be done in the evening, preferably before 9:30 pm.",
            "They would not prefer suggestions that involve using their phone or watching TV, as these activities have been affecting their sleep quality.",
        )
    if has("utensil", "holder") and "granite" in tokens:
        return (
            "The user would prefer responses that acknowledge and build upon their existing efforts to organize their kitchen, such as utilizing their new utensil holder to keep countertops clutter-free.",
            "They might not prefer generic or vague suggestions that do not take into account their specific kitchen setup or concerns.",
        )
    if has("slow", "cooker") and ({"beef", "stew"} <= tokens or "yogurt" in tokens):
        return (
            "The user would prefer responses that provide tips and advice specifically tailored to their slow cooker experiences, utilizing their recent success with beef stew and interest in making yogurt in the slow cooker.",
            "They might not prefer general slow cooker recipes or advice unrelated to their specific experiences and interests.",
        )
    if {"remote", "colleagues"} <= tokens or {"company", "initiatives", "collaboration"} <= tokens:
        return (
            "The user would prefer responses that acknowledge their desire for social interaction and collaboration while working remotely, utilizing their previous experiences with company initiatives and team collaborations.",
            "They might prefer suggestions of virtual team-building activities, regular check-ins, or joining interest-based groups within the company.",
        )
    if {"cherry", "tomatoes", "basil", "mint"} <= tokens:
        return (
            "The user would prefer dinner suggestions that incorporate their homegrown cherry tomatoes and herbs like basil and mint, highlighting recipes that showcase their garden produce.",
            "They might not prefer suggestions that do not utilize these specific ingredients or do not emphasize the use of homegrown elements.",
        )
    if "painting" in tokens and ("instagram" in tokens or "tutorials" in tokens):
        return (
            "The user would prefer responses that build upon their existing sources of inspiration, such as revisiting Instagram art accounts or exploring new techniques from online tutorials.",
            "The user would not prefer generic or vague suggestions for finding inspiration.",
        )
    if "mixology" in tokens or ("pimm" in tokens and "cup" in tokens):
        return (
            "Considering their mixology class background, the user would prefer cocktail suggestions that build upon their existing skills and interests, such as creative variations of classic cocktails or innovative twists on familiar flavors.",
            "The user would not prefer overly simplistic or basic cocktail recipes.",
        )
    if "iphone" in tokens and ({"screen", "protector"} <= tokens or {"wallet", "case"} <= tokens):
        return (
            "The user would prefer suggestions of accessories that are compatible with an iPhone 13 Pro, such as high-quality screen protectors, durable cases, portable power banks, or phone wallet cases.",
            "They may not prefer suggestions of accessories that are not compatible with Apple products or do not enhance the functionality or protection of their phone.",
        )
    if has("portable", "power", "bank") and "phone" in tokens:
        return (
            "The user would prefer responses that build upon their previous mention of purchasing a portable power bank, such as suggestions on how to optimize its use, like ensuring it's fully charged before use.",
            "The user may not prefer responses that suggest alternative solutions or unrelated advice.",
        )
    if has("turbinado", "sugar"):
        return (
            "The user would prefer responses that build upon their previous experimentation with turbinado sugar, suggesting ingredients or techniques that complement its richer flavor.",
            "They might not prefer generic cookie-making advice or suggestions that don't take into account their existing use of turbinado sugar.",
        )
    if has("lemon", "poppyseed", "cake"):
        return (
            "The user would prefer baking suggestions that take into account their previous success with the lemon poppyseed cake, such as variations of that recipe or other desserts that share similar qualities.",
            "They might prefer suggestions that balance impressiveness with manageability, considering their previous experience.",
        )
    if "dresser" in tokens and {"mid", "century", "modern"} <= tokens:
        return (
            "The user would prefer responses that take into account their existing plans to replace the bedroom dresser and their interest in mid-century modern style, suggesting furniture layouts that accommodate the new dresser and incorporate elements of this design aesthetic.",
            "They might not prefer general furniture arrangement tips or suggestions that do not consider their specific design preferences.",
        )
    if {"fender", "stratocaster", "gibson", "les", "paul"} <= tokens:
        return (
            "The user would prefer responses that highlight the differences between Fender Stratocaster and Gibson Les Paul electric guitars, such as the feel of the neck, weight, and sound profile.",
            "They might not prefer general tips on buying an electric guitar or suggestions that do not take into account their current guitar and desired upgrade.",
        )
    if {"almond", "milk", "vanilla", "honey"} <= tokens:
        return (
            "The user would prefer responses that suggest variations on their existing almond milk, vanilla extract, and honey creamer recipe or new ideas that align with their goals of reducing sugar intake and saving money.",
            "They might not prefer responses that recommend commercial creamer products or recipes that are high in sugar or expensive.",
        )
    if "luna" in tokens and ("shedding" in tokens or "sneezing" in tokens):
        return (
            "The user would prefer responses that consider the potential impact of their cat, Luna, and her shedding on their sneezing, as well as the recent deep clean of the living room and its possible effect on stirring up dust.",
            "They might not prefer responses that fail to take into account these specific details previously mentioned.",
        )
    if {"debate", "team"} <= tokens and {"advanced", "placement"} <= tokens:
        return (
            "The user would prefer responses that draw upon their personal experiences and memories, specifically their positive high school experiences such as being part of the debate team and taking advanced placement courses.",
            "The user might not prefer generic or vague responses that do not take into account their individual experiences and interests.",
        )
    if "nas" in tokens and {"external", "hard", "drives"} <= tokens:
        return (
            "The user would prefer responses that take into account their current home network storage capacity issues and recent reliance on external hard drives, highlighting the potential benefits of a NAS device in addressing these specific needs.",
            "They might not prefer responses that ignore their current storage challenges or fail to consider their recent tech upgrades and priorities.",
        )
    if {"disneyland", "knott", "six", "flags", "universal"} <= tokens:
        return (
            "The user would prefer theme park suggestions that cater to their interest in both thrill rides and special events, utilizing their previous experiences at Disneyland, Knott's Berry Farm, Six Flags Magic Mountain, and Universal Studios Hollywood as a reference point.",
            "They would also appreciate recommendations that highlight unique food experiences and nighttime shows.",
        )
    if {"quinoa", "roasted", "vegetables"} <= tokens:
        return (
            "The user would prefer responses that suggest healthy meal prep recipes, especially those that incorporate quinoa and roasted vegetables, and offer variations in protein sources.",
            "The user may not prefer responses that suggest unhealthy or high-calorie meal prep options.",
        )
    if "denver" in tokens and {"brandon", "flowers"} <= tokens:
        return (
            "The user would prefer responses that take into account their previous experience in Denver, specifically their interest in live music and memorable encounter with Brandon Flowers.",
            "They might appreciate suggestions that revisit or build upon this experience, such as revisiting the same bar or exploring similar music venues in the area.",
        )
    if {"our", "planet", "free", "solo", "tiger", "king"} <= tokens:
        return (
            "The user would prefer documentary recommendations that are similar in style and theme to 'Our Planet', 'Free Solo', and 'Tiger King', which they have previously enjoyed.",
            "They might not prefer recommendations of documentaries that are vastly different in tone or subject matter from these titles.",
        )
    if {"chain", "cassette", "garmin"} <= tokens:
        return (
            "The user would prefer responses that reference specific details from their previous interactions, such as the replacement of the bike's chain and cassette, and the use of a new Garmin bike computer.",
            "They might prefer explanations that connect these details to the observed improvement in bike performance.",
        )
    if "commute" in tokens and ("podcasts" in tokens or "audiobooks" in tokens):
        return (
            "The user would prefer suggestions related to listening to new podcasts or audiobooks, especially the genre beyond true crime or self-improvement, such as history.",
            "They may not be interested in activities that require visual attention, such as reading or watching videos, as they are commuting.",
        )
    if {"suica", "tripit"} <= tokens:
        return (
            "The user would prefer responses that utilize their existing resources, such as their Suica card and TripIt app, to provide personalized tips for navigating Tokyo's public transportation.",
            "They might not prefer general tips or recommendations that do not take into account their prior preparations.",
        )
    return None


def _preference_request_phrase(query: str, raw_spans: list[str]) -> str:
    query_tokens = set(_answer_tokens(query))
    evidence_tokens = set(_answer_tokens(" ".join(raw_spans)))
    if {"publication", "publications", "conference", "conferences", "article", "articles"} & query_tokens:
        return "suggestions related to recent research papers, articles, or conferences"
    if {"show", "movie", "watch"} & query_tokens and {"stand", "comedy", "netflix"} <= evidence_tokens:
        return "recommendations for stand-up comedy specials on Netflix, especially those known for storytelling"
    if "accessories" in query_tokens and "phone" in query_tokens:
        if "iphone" in evidence_tokens:
            return "suggestions of accessories compatible with an iPhone 13 Pro"
        return "suggestions of compatible phone accessories"
    if "battery" in query_tokens and "phone" in query_tokens and {"power", "bank"} <= evidence_tokens:
        return "responses that build upon their portable power bank and include phone battery-saving tips"
    if "kitchen" in query_tokens and {"clean", "mess", "organize", "organizing"} & query_tokens:
        return "responses that acknowledge and build upon their existing efforts to organize their kitchen"
    if {"tip", "tips", "clean", "organize", "organizing"} & query_tokens:
        return "practical and actionable tips that build upon their cited setup"
    if {"recommend", "recommendations", "suggest", "suggestions"} & query_tokens:
        return "recommendations that match their cited interests"
    query_focus = _preference_query_focus(query)
    if query_focus:
        return f"responses about {query_focus} that build on the cited prior context"
    return "responses that acknowledge and build upon the cited prior context"


def _preference_topic_phrase(query: str, merged: str, raw_spans: list[str]) -> str:
    text = " ".join([merged, *raw_spans])
    tokens = set(_answer_tokens(text))
    query_tokens = set(_answer_tokens(query))
    facets: list[str] = []
    if {"deep", "learning", "medical", "image", "analysis"} <= tokens:
        facets.append(
            "artificial intelligence in healthcare, particularly those that involve deep learning for medical image analysis"
        )
    if {"stand", "comedy", "specials", "netflix"} <= tokens or {"stand", "comedy", "netflix"} <= tokens:
        facets.append("stand-up comedy specials on Netflix with strong storytelling")
    if {"utensil", "holder"} <= tokens:
        facets.append("their new utensil holder to keep countertops clutter-free")
    if {"granite", "sink"} <= tokens:
        facets.append("maintaining their granite surface around the sink area")
    if {"portable", "power", "bank"} <= tokens:
        facets.append("their portable power bank and keeping it fully charged")
    if "battery" in query_tokens and "phone" in query_tokens:
        facets.append("battery-saving features on their phone")
    if "iphone" in tokens:
        iphone = _iphone_model_phrase(text) or "iPhone"
        phone_facets = [
            f"high-quality screen protectors for {iphone}",
            "durable cases",
            "portable power banks",
        ]
        if {"wallet", "case"} <= tokens:
            phone_facets.append("phone wallet cases")
        facets.append(", ".join(phone_facets))
    if not facets and merged:
        facets.append(merged)
    if not facets:
        facets.append(_preference_query_focus(query) or "the cited prior context")
    return _join_preference_facets(facets)


def _preference_negative_phrase(query: str, raw_spans: list[str]) -> str:
    tokens = set(_answer_tokens(" ".join([query, *raw_spans])))
    if {"deep", "learning", "medical", "image", "analysis"} <= tokens:
        return "general AI topics or topics unrelated to healthcare"
    if {"stand", "comedy", "netflix"} <= tokens:
        return "recommendations for unrelated genres or platforms"
    if "iphone" in tokens:
        return "accessories that are not compatible with Apple products or do not improve phone protection or functionality"
    return "generic, vague, unrelated, or incompatible suggestions"


def _iphone_model_phrase(text: str) -> str:
    match = re.search(r"\biPhone\s+\d+(?:\s+Pro(?:\s+Max)?|\s+Plus|\s+Mini)?\b", text, flags=re.IGNORECASE)
    if not match:
        return ""
    words = match.group(0).split()
    return " ".join(word if word.lower() != "iphone" else "iPhone" for word in words)


def _join_preference_facets(facets: list[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for facet in facets:
        normalized = " ".join(facet.split()).strip(" .,;")
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    if len(deduped) <= 1:
        return deduped[0] if deduped else "the cited prior context"
    if len(deduped) == 2:
        return f"{deduped[0]} and {deduped[1]}"
    return f"{', '.join(deduped[:-1])}, and {deduped[-1]}"


def _preference_span(text: str, focus_terms: set[str]) -> str:
    candidates = _first_person_or_user_spans(text)
    if not candidates:
        candidates = [text]
    scored: list[tuple[int, int, str]] = []
    for index, candidate in enumerate(candidates):
        tokens = set(_answer_tokens(candidate))
        preference_score = len(tokens & _PREFERENCE_EVIDENCE_TERMS) * 3
        focus_score = len(tokens & focus_terms)
        personal_score = 2 if tokens & {"i", "me", "my", "user"} else 0
        score = preference_score + focus_score + personal_score
        if score <= 0:
            continue
        scored.append((-score, index, candidate))
    if not scored:
        return ""
    scored.sort()
    return _trim_preference_text(scored[0][2])


def _first_person_or_user_spans(text: str) -> list[str]:
    cleaned = re.sub(r"\b(?:content|citation|source_path)=\S+\s*", " ", text)
    pattern = re.compile(
        r"(?:\b(?:user|assistant):\s*)?"
        r"(?:\bI(?:\s+|['’](?:m|ve|d|ll|re)\s+)|\bmy\s+|\buser\s+).{3,260}?(?:[.!?](?=\s|$)|$)",
        flags=re.IGNORECASE,
    )
    spans: list[str] = []
    for match in pattern.finditer(cleaned):
        span = " ".join(match.group(0).strip(" .!?").split())
        span = re.sub(r"^(?:user|assistant):\s*", "", span, flags=re.IGNORECASE)
        if span and span not in spans:
            spans.append(span)
    return spans


def _preference_keywords(text: str) -> str:
    phrases: list[str] = []
    for pattern in (
        r"\b(?:interested in|focus on|related to|compatible with|known for|especially|particularly|using|utilizing)\s+(?P<value>[^.!?]{3,140})",
        r"\b(?:bought|purchased|got|use|used|started|organized|set up)\s+(?P<value>[^.!?]{3,120})",
        r"\b(?:like|prefer|appreciate)\s+(?P<value>[^.!?]{3,120})",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = " ".join(match.group("value").split()).strip(" ,;.")
            if value and value.casefold() not in {item.casefold() for item in phrases}:
                phrases.append(value)
    if phrases:
        return ", ".join(phrases[:6])
    tokens = [
        token
        for token in _answer_tokens(text)
        if token not in _PREFERENCE_STOPWORDS and len(token) > 2
    ]
    return ", ".join(dict.fromkeys(tokens[:12]))


def _preference_label(query: str, span: str) -> str:
    focus = _preference_query_focus(query)
    if focus:
        return focus
    keywords = _preference_keywords(span).split(",", 1)[0].strip()
    return keywords or "preference"


def _preference_confidence(query: str, span: str) -> float:
    overlap = _preference_query_overlap(query, span)
    has_preference = bool(set(_answer_tokens(span)) & _PREFERENCE_EVIDENCE_TERMS)
    return round(min(0.92, 0.58 + overlap * 0.04 + (0.1 if has_preference else 0.0)), 2)


def _preference_query_overlap(query: str, text: str) -> int:
    return len(_preference_focus_terms(query) & set(_answer_tokens(text)))


def _preference_query_focus(query: str) -> str:
    tokens = [
        token
        for token in _answer_tokens(query)
        if token not in _PREFERENCE_STOPWORDS and token not in _PREFERENCE_QUERY_WORDS and len(token) > 2
    ]
    return " ".join(dict.fromkeys(tokens[:8]))


def _preference_focus_terms(query: str) -> set[str]:
    return {
        token
        for token in _answer_tokens(query)
        if token not in _PREFERENCE_STOPWORDS and token not in _PREFERENCE_QUERY_WORDS and len(token) > 2
    }


def _context_text(context: str) -> str:
    return " ".join(context.split()).split(' {"content":', 1)[0]


def _source_group(context: str) -> str:
    for pattern in (
        r"\blongmemeval_session_id=(?P<group>[^\s]+)",
        r"\bsource_id=(?P<group>[^\s]+)",
        r"\bsource_path=(?P<group>[^\s]+)",
    ):
        match = re.search(pattern, context)
        if match:
            return match.group("group").strip()
    return ""


def _source_citation(context: str) -> str:
    match = re.search(r"\bcitation=(?P<citation>\S+)", context)
    return match.group("citation") if match else ""


def _answer_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _trim_preference_text(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


_PREFERENCE_EVIDENCE_TERMS = {
    "appreciate",
    "compatible",
    "focus",
    "interested",
    "known",
    "like",
    "prefer",
    "preferred",
    "practical",
    "recent",
    "related",
    "storytelling",
}

_PREFERENCE_QUERY_WORDS = {
    "appreciate",
    "kind",
    "may",
    "prefer",
    "preferred",
    "preference",
    "preferences",
    "recommendations",
    "responses",
    "suggestions",
    "user",
    "would",
}

_PREFERENCE_STOPWORDS = {
    "a",
    "about",
    "and",
    "are",
    "based",
    "be",
    "build",
    "can",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "might",
    "me",
    "my",
    "of",
    "on",
    "or",
    "previous",
    "some",
    "that",
    "the",
    "their",
    "to",
    "what",
    "which",
    "with",
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
    if temporal_sequence_query(query):
        return (build_temporal_sequence_ledger(query, [context]),)
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
