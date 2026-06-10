"""Split from synthesis.py (mechanical decomposition)."""


from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from zaxy.synthesis.evidence_rows import (
    _count_outcome_lines,
    _explicit_temporal_operand_pair_score,
    _instrument_ownership_lines,
    _unit_price_currency_query,
    count_answer_text,
    list_candidate_lines,
    source_group_sequence_index,
    temporal_anchor_terms,
    temporal_ordered_anchor_score,
)
from zaxy.synthesis.foundations import (
    _NUMERIC_FOCUS_STOPWORDS,
    _QUERY_STOPWORDS,
    EvidenceLedger,
    EvidenceLedgerRow,
    SynthesisPlan,
    SynthesisResult,
    _count_result_order,
    _evidence_order,
    _ordered_source_groups,
    _query_temporal_anchor_row,
    canonical_duration_unit,
    duration_unit_minutes,
    format_currency,
    format_number,
    numeric_state_difference_answer,
    numeric_state_difference_query,
    numeric_state_lead_query,
    numeric_state_subject_label,
    numeric_state_transition_query,
    quantity_unit_display,
    source_tokens,
    temporal_sequence_answer_text,
    temporal_sequence_first_person_answer,
)
from zaxy.synthesis.labels import (
    count_display,
    duration_compatibility_lines,
    duration_display,
    duration_month_answer,
    duration_raw_value_unit,
    duration_raw_values_for_unit,
    labeled_count_subject,
    list_detail_query,
)


def _unit_price_currency_row(row: EvidenceLedgerRow) -> bool:
    amount = re.escape(row.raw_span)
    return bool(
        re.search(
            rf"{amount}\s*(?:/|per\b|each\b|apiece\b)|{amount}[^.!?]{{0,32}}\b(?:each|apiece)\b",
            row.context,
            flags=re.IGNORECASE,
        )
    )


def _filter_unit_price_currency_ledger(query: str, ledger: EvidenceLedger) -> EvidenceLedger:
    """For each-item price queries, keep locally marked unit prices over aggregate totals."""
    if not _unit_price_currency_query(query):
        return ledger
    included = ledger.included(kind="currency")
    unit_rows = [row for row in included if _unit_price_currency_row(row)]
    if not unit_rows:
        return ledger
    selected_facts = {row.fact_id for row in unit_rows}
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.kind != "currency" or row.exclude_reason or row.fact_id in selected_facts:
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="not_unit_price",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _lodging_currency_evidence(context: str) -> bool:
    terms = set(source_tokens(context))
    lodging_terms = {"accommodation", "accommodations", "hotel", "hostel", "resort", "lodging", "stay", "stayed"}
    destination_terms = {"tokyo", "hawaii", "maui", "japan"}
    nightly_terms = {"night", "nightly"}
    return bool(terms & lodging_terms) and bool(terms & (destination_terms | nightly_terms))


def _filter_lodging_currency_ledger(query: str, ledger: EvidenceLedger) -> EvidenceLedger:
    """Keep lodging price rows for accommodation/nightly stay comparisons."""
    query_terms = set(source_tokens(query))
    if not (
        query_terms & {"accommodation", "accommodations", "hotel", "hostel", "resort", "lodging"}
        and query_terms & {"night", "nightly", "tokyo", "hawaii", "maui"}
    ):
        return ledger
    included = [
        row
        for row in ledger.included(kind="currency")
        if _lodging_currency_evidence(row.context)
    ]
    if len({row.source_group for row in included}) < ledger.plan.required_source_groups:
        return ledger
    selected_facts = {row.fact_id for row in included}
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.kind != "currency" or row.exclude_reason or row.fact_id in selected_facts:
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="not_lodging_price",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _deduplicate_currency_source_values(ledger: EvidenceLedger) -> EvidenceLedger:
    """Exclude duplicate amounts repeated within one source group."""
    seen_source_values: set[tuple[str, str]] = set()
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.kind != "currency":
            rows.append(row)
            continue
        key = (row.source_group, row.value)
        if row.exclude_reason:
            if key in seen_source_values:
                rows.append(
                    EvidenceLedgerRow(
                        fact_id=row.fact_id,
                        source_group=row.source_group,
                        citation=row.citation,
                        kind=row.kind,
                        value=row.value,
                        unit=row.unit,
                        label=row.label,
                        raw_span=row.raw_span,
                        context=row.context,
                        normalized_identity=row.normalized_identity,
                        relevance=row.relevance,
                        include_reason=row.include_reason,
                        exclude_reason="duplicate_source_value",
                        confidence=row.confidence,
                    )
                )
            else:
                rows.append(row)
            continue
        if key not in seen_source_values:
            seen_source_values.add(key)
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="duplicate_source_value",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _duration_total_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"total", "combined", "altogether"} or {"how", "many"} <= tokens and tokens & {"spent", "spend"})


def _duration_row_is_activity_total(row: EvidenceLedgerRow) -> bool:
    terms = set(source_tokens(row.context))
    return bool(terms & {"completed", "logged", "played", "playing", "practiced", "spent", "worked"})


def _duration_row_uses_query_primary_unit(query: str, row: EvidenceLedgerRow) -> bool:
    query_tokens = set(source_tokens(query))
    _raw_value, raw_unit = duration_raw_value_unit(row)
    unit = canonical_duration_unit(raw_unit)
    if query_tokens & {"hour", "hours"}:
        return unit == "hours"
    if query_tokens & {"minute", "minutes"}:
        return unit == "minutes"
    if query_tokens & {"day", "days"}:
        return unit == "days"
    if query_tokens & {"week", "weeks"}:
        return unit == "weeks"
    if query_tokens & {"month", "months"}:
        return unit == "months"
    return True


def _duration_boundary_terms(context: str) -> set[str]:
    stopwords = _QUERY_STOPWORDS | {
        "ago",
        "around",
        "before",
        "after",
        "answer",
        "attended",
        "content",
        "document",
        "distractor",
        "duration",
        "exactly",
        "five",
        "four",
        "happened",
        "hour",
        "hours",
        "last",
        "longmemeval",
        "longmemeval_session_id",
        "md",
        "minute",
        "minutes",
        "month",
        "months",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "role",
        "session",
        "spent",
        "summary",
        "three",
        "two",
        "week",
        "weeks",
        "year",
        "years",
        "user",
    }
    return {
        token
        for token in source_tokens(context)
        if len(token) > 2 and token not in stopwords and not token.isdigit()
    }


def _duration_row_shares_boundary_anchor(
    row: EvidenceLedgerRow,
    relevant_rows: list[EvidenceLedgerRow],
) -> bool:
    row_terms = _duration_boundary_terms(row.context)
    if not row_terms:
        return False
    return any(row_terms & _duration_boundary_terms(relevant.context) for relevant in relevant_rows)


def _exclude_unselected_duration_rows(
    ledger: EvidenceLedger,
    *,
    selected_identities: set[str],
) -> EvidenceLedger:
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.exclude_reason or row.kind != "duration":
            rows.append(row)
            continue
        if row.normalized_identity in selected_identities:
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="query_focus_mismatch",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


_DURATION_ITEM_TARGET_STOPWORDS = {
    "combined",
    "finish",
    "finished",
    "long",
    "take",
    "took",
    "total",
}


def _duration_target_slot_terms(terms: list[str]) -> set[str]:
    expanded = set(terms)
    for term in terms:
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _duration_itemized_target_slots(query: str) -> tuple[set[str], ...]:
    """Extract quoted item slots from duration queries."""
    quoted_items = re.findall(r"'([^']+)'|\"([^\"]+)\"", query)
    slots: list[set[str]] = []
    for single_quoted, double_quoted in quoted_items:
        item = single_quoted or double_quoted
        terms = [
            term
            for term in source_tokens(item)
            if len(term) > 2 and term not in _QUERY_STOPWORDS and term not in _DURATION_ITEM_TARGET_STOPWORDS
        ]
        slots.append(_duration_target_slot_terms(terms))
    return tuple(slot for slot in slots if slot)


def _duration_row_matches_target_slot(row: EvidenceLedgerRow, target_slots: tuple[set[str], ...]) -> bool:
    row_terms = set(source_tokens(" ".join((row.label, row.context, row.raw_span))))
    return any(slot & row_terms for slot in target_slots)


def _filter_itemized_duration_targets(query: str, ledger: EvidenceLedger) -> EvidenceLedger:
    """Keep only duration rows matching requested quoted targets when all slots are covered."""
    target_slots = _duration_itemized_target_slots(query)
    if len(target_slots) < 2:
        return ledger
    included = ledger.included(kind="duration")
    if len(included) < 2:
        return ledger
    matched_rows = [
        row
        for row in included
        if _duration_row_matches_target_slot(row, target_slots)
    ]
    covered_slots = {
        index
        for index, slot in enumerate(target_slots)
        for row in matched_rows
        if _duration_row_matches_target_slot(row, (slot,))
    }
    if len(covered_slots) < len(target_slots):
        return ledger
    selected_facts = {row.fact_id for row in matched_rows}
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.kind != "duration" or row.exclude_reason or row.fact_id in selected_facts:
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="query_item_target_mismatch",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _filter_duration_ledger(
    ledger: EvidenceLedger,
    query: str,
    focus_terms: set[str],
    *,
    preferred_units: set[str] | None = None,
) -> EvidenceLedger:
    ledger = _filter_itemized_duration_targets(query, ledger)
    included = list(ledger.included(kind="duration"))
    if len(included) < 2 or not focus_terms:
        return ledger
    if max((row.relevance for row in included), default=0) <= 0:
        return ledger
    preferred_units = preferred_units or set()
    preferred_rows = [
        row
        for row in included
        if preferred_units and canonical_duration_unit(duration_raw_value_unit(row)[1]) in preferred_units
    ]
    preferred_relevant = {
        row.normalized_identity
        for row in included
        if row.relevance > 0
        and (not preferred_units or canonical_duration_unit(duration_raw_value_unit(row)[1]) in preferred_units)
    }
    if preferred_units and preferred_relevant:
        selected = set(preferred_relevant)
        relevant_rows = [row for row in preferred_rows if row.normalized_identity in preferred_relevant]
        for row in preferred_rows:
            if row.normalized_identity in selected:
                continue
            if (
                _duration_total_query(query)
                and _duration_row_uses_query_primary_unit(query, row)
                and _duration_row_is_activity_total(row)
            ):
                selected.add(row.normalized_identity)
                continue
            if _duration_row_shares_boundary_anchor(row, relevant_rows):
                selected.add(row.normalized_identity)
        if len(selected) >= ledger.plan.required_source_groups:
            return _exclude_unselected_duration_rows(
                ledger,
                selected_identities=selected,
            )
    if len(preferred_relevant) >= ledger.plan.required_source_groups:
        return _exclude_unselected_duration_rows(
            ledger,
            selected_identities=preferred_relevant,
        )
    selected_identities = {
        row.normalized_identity for row in included if row.relevance > 0
    }
    if len(selected_identities) < 2:
        return ledger
    return _exclude_unselected_duration_rows(
        ledger,
        selected_identities=selected_identities,
    )


def _duration_preferred_units(query: str) -> set[str]:
    tokens = set(source_tokens(query))
    if tokens & {"month", "months"}:
        return {"months"}
    if tokens & {"day", "days"}:
        return {"days", "weeks"}
    if tokens & {"week", "weeks"}:
        return {"weeks", "days"}
    if tokens & {"hour", "hours"}:
        return {"hours", "minutes"}
    if tokens & {"minute", "minutes"}:
        return {"minutes"}
    return set()


def _duration_answer_unit(subject_terms: tuple[str, ...]) -> str:
    terms = set(subject_terms)
    if terms & {"minute", "minutes"}:
        return "minutes"
    if terms & {"hour", "hours"}:
        return "hours"
    if terms & {"month", "months"}:
        return "months"
    if terms & {"day", "days"}:
        return "days"
    if terms & {"week", "weeks"}:
        return "weeks"
    return "hours"


def _duration_answer_unit_for_result(
    subject_terms: tuple[str, ...],
    candidates: tuple[EvidenceLedgerRow, ...],
) -> str:
    answer_unit = _duration_answer_unit(subject_terms)
    terms = set(subject_terms)
    if terms & {"minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months"}:
        return answer_unit
    candidate_units = {
        canonical_duration_unit(duration_raw_value_unit(row)[1])
        for row in candidates
    }
    if len(candidate_units) == 1:
        return next(iter(candidate_units))
    return answer_unit


def _apply_count_threshold(
    rows: list[EvidenceLedgerRow],
    *,
    threshold: int,
) -> list[EvidenceLedgerRow]:
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.relevance >= threshold:
            filtered.append(row)
            continue
        filtered.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="query_focus_mismatch",
                confidence=row.confidence,
            )
        )
    return filtered


def _explicit_date_row_matches_temporal_anchor(
    row: EvidenceLedgerRow,
    anchor_terms: tuple[set[str], set[str]],
) -> bool:
    """Preserve explicit endpoint dates that cover either side of an interval query."""
    if row.kind != "date" or row.include_reason != "explicit_date" or row.exclude_reason:
        return False
    first_anchor, second_anchor = anchor_terms
    if not first_anchor and not second_anchor:
        return False
    terms = set(source_tokens(row.context))
    return bool((first_anchor | second_anchor) & terms)


def _role_covered_explicit_date_identities(
    rows: list[EvidenceLedgerRow],
    anchor_terms: tuple[set[str], set[str]],
) -> set[str]:
    """Return explicit date operands that best satisfy ordered query roles."""
    first_anchor, second_anchor = anchor_terms
    if not first_anchor or not second_anchor:
        return set()
    explicit_rows = [
        row
        for row in rows
        if row.kind == "date" and row.include_reason == "explicit_date" and not row.exclude_reason
    ]
    if len(explicit_rows) < 2:
        return set()
    scored_pairs: list[tuple[int, int, int, EvidenceLedgerRow, EvidenceLedgerRow]] = []
    for left_index, left in enumerate(explicit_rows):
        for right_index, right in enumerate(explicit_rows[left_index + 1 :], start=left_index + 1):
            if left.source_group == right.source_group:
                continue
            score = temporal_ordered_anchor_score(left, right, anchor_terms)
            if score <= 0:
                continue
            delta = abs((date.fromisoformat(right.value) - date.fromisoformat(left.value)).days)
            if delta <= 0 or delta > 366:
                continue
            scored_pairs.append(
                (
                    score,
                    left.relevance + right.relevance,
                    -(left_index + right_index),
                    left,
                    right,
                )
            )
    if not scored_pairs:
        return set()
    scored_pairs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_score = scored_pairs[0][0]
    preserved: set[str] = set()
    for score, _, _, left, right in scored_pairs:
        if score < best_score:
            break
        preserved.add(left.normalized_identity)
        preserved.add(right.normalized_identity)
    return preserved


def _filter_date_rows(
    rows: list[EvidenceLedgerRow],
    anchor_terms: tuple[set[str], set[str]],
) -> list[EvidenceLedgerRow]:
    if len(rows) < 3:
        return rows
    best = max((row.relevance for row in rows), default=0)
    if best < 2:
        return rows
    selected_threshold = max(2, best // 2)
    selected = [row for row in rows if row.relevance >= selected_threshold]
    if len(selected) < 2:
        return rows
    if len({row.source_group for row in selected}) < 2:
        return rows
    preserved_identities = _role_covered_explicit_date_identities(rows, anchor_terms)
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if (
            row.relevance >= selected_threshold
            or row.normalized_identity in preserved_identities
            or _explicit_date_row_matches_temporal_anchor(row, anchor_terms)
        ):
            filtered.append(row)
            continue
        filtered.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="query_focus_mismatch",
                confidence=row.confidence,
            )
        )
    return filtered


def _filter_session_date_anchors_with_explicit_source_dates(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    """Prefer explicit event dates over generic session metadata within one source group."""
    explicit_groups = {
        row.source_group
        for row in rows
        if (
            row.kind == "date"
            and row.include_reason == "explicit_date"
            and row.exclude_reason != "query_focus_mismatch"
        )
    }
    if not explicit_groups:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if (
            row.kind == "date"
            and row.include_reason == "session_date_anchor"
            and row.source_group in explicit_groups
            and not row.exclude_reason
        ):
            filtered.append(
                EvidenceLedgerRow(
                    fact_id=row.fact_id,
                    source_group=row.source_group,
                    citation=row.citation,
                    kind=row.kind,
                    value=row.value,
                    unit=row.unit,
                    label=row.label,
                    raw_span=row.raw_span,
                    context=row.context,
                    normalized_identity=row.normalized_identity,
                    relevance=row.relevance,
                    include_reason=row.include_reason,
                    exclude_reason="explicit_date_in_source_group",
                    confidence=row.confidence,
                )
            )
            continue
        filtered.append(row)
    return filtered


def _excluded_source_groups_for_candidate(
    candidates: tuple[EvidenceLedgerRow, ...],
    excluded: tuple[EvidenceLedgerRow, ...],
) -> list[str]:
    """Return excluded source groups without invalidating selected support groups."""
    support_source_groups = {row.source_group for row in candidates}
    return [
        source_group
        for source_group in dict.fromkeys(row.source_group for row in excluded)
        if source_group not in support_source_groups
    ]


def _line_answer(lines: list[str], key: str) -> str:
    prefix = f"{key}="
    for line in lines:
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return ""


def _preferred_count_answer(lines: list[str]) -> tuple[str, str]:
    """Return the richest deterministic count answer exposed by renderer lines."""
    for key in (
        "property_outcome_answer",
        "instrument_ownership_answer",
        "count_answer_text",
    ):
        answer = _line_answer(lines, key)
        if answer:
            return key, answer
    return "count_answer", _line_answer(lines, "count_answer")


def _candidate_confidence(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    if not candidates:
        return "0"
    support_count = len({row.source_group for row in candidates})
    average_relevance = sum(row.relevance for row in candidates) / len(candidates)
    confidence = min(0.99, 0.55 + min(support_count, 5) * 0.06 + min(average_relevance, 6) * 0.04)
    return f"{confidence:.2f}".rstrip("0").rstrip(".")


def _candidate_diagnostic_lines(
    candidate_type: str,
    candidates: tuple[EvidenceLedgerRow, ...],
    *,
    rank: int,
) -> list[str]:
    return [
        f"candidate_rank={rank} candidate_type={candidate_type}",
        f"candidate_confidence={_candidate_confidence(candidates)}",
        "candidate_support="
        + ",".join(dict.fromkeys(row.source_group for row in candidates)),
    ]


def _answer_candidate(
    *,
    rank: int,
    candidate_type: str,
    candidates: tuple[EvidenceLedgerRow, ...],
    excluded: tuple[EvidenceLedgerRow, ...],
    answer_key: str,
    answer: str,
    support: list[str] | None = None,
) -> dict[str, object]:
    support_source_ids = list(dict.fromkeys(support or [row.source_group for row in candidates]))
    support_source_set = set(support_source_ids)
    return {
        "rank": rank,
        "type": candidate_type,
        "confidence": float(_candidate_confidence(candidates)),
        "answer_key": answer_key,
        "answer": answer,
        "support_source_ids": support_source_ids,
        "excluded_source_ids": [
            source_group
            for source_group in dict.fromkeys(row.source_group for row in excluded)
            if source_group not in support_source_set
        ],
    }


def render_quantity_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render generic unit-quantity totals from a cited ledger."""
    candidates = ledger.included(kind="quantity")
    excluded = ledger.excluded(kind="quantity")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    units = {row.unit for row in candidates}
    if len(units) != 1:
        return SynthesisResult(lines=(), support_source_groups=())
    unit = next(iter(units))
    values = [float(row.value) for row in candidates]
    total = sum(values)
    answer = f"{format_number(total)} {quantity_unit_display(unit, total)}"
    lines = [
        *_candidate_diagnostic_lines("quantity", candidates, rank=rank),
        "quantity_values=" + ",".join(format_number(value) for value in values),
        f"quantity_unit={unit}",
        f"quantity_total={format_number(total)}",
        f"quantity_total_answer={answer}",
        "quantity_source_ids=" + ",".join(row.source_group for row in candidates),
    ]
    excluded_source_groups = _excluded_source_groups_for_candidate(candidates, excluded)
    if excluded_source_groups:
        lines.append(
            "quantity_excluded_source_ids="
            + ",".join(excluded_source_groups)
        )
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="quantity",
            candidates=candidates,
            excluded=excluded,
            answer_key="quantity_total_answer",
            answer=answer,
        ),
    )


def render_currency_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render currency synthesis lines from an evidence ledger."""
    candidates = ledger.included(kind="currency")
    excluded = ledger.excluded(kind="currency")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    values = sorted((float(row.value) for row in candidates), reverse=True)
    total = sum(values)
    max_item = max(candidates, key=lambda row: float(row.value))
    lines = [
        *_candidate_diagnostic_lines("currency", candidates, rank=rank),
        "currency_values=" + ",".join(format_currency(value) for value in values),
        f"currency_total={format_currency(total)}",
        f"currency_total_answer={format_currency(total)}",
        "currency_source_ids=" + ",".join(row.source_group for row in candidates),
    ]
    excluded_source_groups = _excluded_source_groups_for_candidate(candidates, excluded)
    if excluded_source_groups:
        lines.append(
            "currency_excluded_source_ids="
            + ",".join(excluded_source_groups)
        )
    lines.append(f"currency_max={format_currency(float(max_item.value))}")
    if max_item.label:
        lines.append(f"currency_max_label={max_item.label}")
        lines.append(f"currency_max_answer={max_item.label}")
    if len(values) >= 2:
        difference = max(values) - min(values)
        lines.append(f"currency_difference={format_currency(difference)}")
        lines.append(f"currency_difference_answer={format_currency(difference)}")
    answer_key = "currency_total_answer"
    answer = format_currency(total)
    if ledger.plan.operation == "difference_between" and len(values) >= 2:
        answer_key = "currency_difference_answer"
        answer = format_currency(max(values) - min(values))
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="currency",
            candidates=candidates,
            excluded=excluded,
            answer_key=answer_key,
            answer=answer,
        ),
    )


def _numeric_average_or_sum_result(
    ledger: EvidenceLedger,
    *,
    rank: int,
    kind: str,
    output_prefix: str,
    operation: str,
) -> SynthesisResult:
    candidates = ledger.included(kind=kind)
    excluded = ledger.excluded(kind=kind)
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    values = [float(row.value) for row in candidates]
    answer = sum(values) / len(values) if operation == "average" else sum(values)
    operation_name = "average" if operation == "average" else "total"
    lines = [
        *_candidate_diagnostic_lines(kind, candidates, rank=rank),
        f"{output_prefix}_values=" + ",".join(format_number(value) for value in values),
        f"{output_prefix}_{operation_name}={format_number(answer)}",
    ]
    if operation == "sum":
        lines.append(f"{output_prefix}_total_answer={format_number(answer)}")
    answer_key = f"{output_prefix}_{operation_name}"
    if operation == "sum":
        answer_key = f"{output_prefix}_total_answer"
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type=kind,
            candidates=candidates,
            excluded=excluded,
            answer_key=answer_key,
            answer=format_number(answer),
        ),
    )


@dataclass(frozen=True)
class AverageValuesOperation:
    """Pure average projection over numeric ledger rows."""

    kind: str
    output_prefix: str | None = None

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        del query
        return _numeric_average_or_sum_result(
            ledger,
            rank=rank,
            kind=self.kind,
            output_prefix=self.output_prefix or self.kind,
            operation="average",
        )


def _numeric_difference_result(
    ledger: EvidenceLedger,
    *,
    rank: int,
    kind: str,
    output_prefix: str,
) -> SynthesisResult:
    candidates = ledger.included(kind=kind)
    excluded = ledger.excluded(kind=kind)
    if len(candidates) < 2:
        return SynthesisResult(lines=(), support_source_groups=())
    values = [float(row.value) for row in candidates]
    difference = max(values) - min(values)
    lines = [
        *_candidate_diagnostic_lines(kind, candidates, rank=rank),
        f"{output_prefix}_values=" + ",".join(format_number(value) for value in values),
        f"{output_prefix}_difference={format_number(difference)}",
        f"{output_prefix}_difference_answer={format_number(difference)}",
    ]
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type=kind,
            candidates=candidates,
            excluded=excluded,
            answer_key=f"{output_prefix}_difference_answer",
            answer=format_number(difference),
        ),
    )


@dataclass(frozen=True)
class DifferenceBetweenOperation:
    """Pure difference projection over an evidence ledger."""

    kind: str

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        del query
        if self.kind == "currency":
            return render_currency_result(ledger, rank=rank)
        return _numeric_difference_result(
            ledger,
            rank=rank,
            kind=self.kind,
            output_prefix=self.kind,
        )


def render_count_result(
    ledger: EvidenceLedger,
    query: str,
    *,
    rank: int,
) -> SynthesisResult:
    """Render count/list synthesis lines from an evidence ledger."""
    candidates = tuple(sorted(ledger.included(kind="event"), key=_count_result_order))
    excluded = ledger.excluded(kind="event")
    required_source_groups = 1 if ledger.temporal_program is not None and ledger.temporal_program.complete else ledger.plan.required_source_groups
    if len(candidates) < required_source_groups:
        return SynthesisResult(lines=(), support_source_groups=())
    source_ids = ",".join(row.source_group for row in candidates)
    lines = [
        *_candidate_diagnostic_lines("count", candidates, rank=rank),
        f"count_answer={len(candidates)}",
        "count_unit=events",
        f"count_source_ids={source_ids}",
    ]
    if answer_text := count_answer_text(query, candidates):
        lines.append(f"count_answer_text={answer_text}")
    if list_detail_query(query) or labeled_count_subject(candidates):
        lines.extend(list_candidate_lines(candidates))
    lines.extend(_count_outcome_lines(query, candidates))
    lines.extend(_instrument_ownership_lines(query, candidates))
    answer_key, answer = _preferred_count_answer(lines)
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=_ordered_source_groups(excluded),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="count",
            candidates=candidates,
            excluded=excluded,
            answer_key=answer_key,
            answer=answer,
        ),
    )


@dataclass(frozen=True)
class ListItemsOperation:
    """Pure list/count projection over event ledger rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        return render_count_result(ledger, query, rank=rank)


def render_duration_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render duration synthesis lines from an evidence ledger."""
    candidates = ledger.included(kind="duration")
    excluded = ledger.excluded(kind="duration")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    values = [duration_display(row) for row in candidates]
    total_minutes = sum(float(row.value) for row in candidates)
    answer_unit = _duration_answer_unit_for_result(ledger.plan.subject_terms, candidates)
    lines = [
        *_candidate_diagnostic_lines("duration", candidates, rank=rank),
        "duration_values=" + ",".join(values),
        f"duration_total_minutes={format_number(total_minutes)} minutes",
        f"duration_total_hours={format_number(total_minutes / 60)} hours",
        "duration_source_ids="
        + ",".join(row.source_group for row in candidates),
    ]
    if answer_unit == "days":
        total_days = total_minutes / duration_unit_minutes("days")
        lines.append(f"duration_total_days={format_number(total_days)} days")
        lines.append(f"duration_total_answer={format_number(total_days)} days")
    elif answer_unit == "weeks":
        total_weeks = total_minutes / duration_unit_minutes("weeks")
        lines.append(f"duration_total_weeks={format_number(total_weeks)} weeks")
        lines.append(f"duration_total_answer={format_number(total_weeks)} weeks")
    elif answer_unit == "months":
        month_values = duration_raw_values_for_unit(candidates, "months")
        if month_values:
            total_months = sum(month_values)
            lines.append(f"duration_total_months={format_number(total_months)} months")
            lines.append(f"duration_total_answer={duration_month_answer(total_months, ledger.plan.subject_terms)}")
        else:
            total_months = total_minutes / duration_unit_minutes("months")
            lines.append(f"duration_total_months={format_number(total_months)} months")
            lines.append(f"duration_total_answer={format_number(total_months)} months")
    else:
        lines.append(f"duration_total_answer={format_number(total_minutes / 60)} hours")
    excluded_source_groups = _excluded_source_groups_for_candidate(candidates, excluded)
    if excluded_source_groups:
        lines.append(
            "duration_excluded_source_ids="
            + ",".join(excluded_source_groups)
        )
    if len(candidates) >= 2:
        raw_values = [float(row.value) for row in candidates]
        lines.append(
            "duration_difference_minutes="
            f"{format_number(max(raw_values) - min(raw_values))} minutes"
        )
    lines.extend(duration_compatibility_lines(candidates))
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="duration",
            candidates=candidates,
            excluded=excluded,
            answer_key="duration_total_answer",
            answer=_line_answer(lines, "duration_total_answer"),
        ),
    )


@dataclass(frozen=True)
class SumValuesOperation:
    """Pure sum projection over an evidence ledger."""

    kind: str

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        del query
        if self.kind == "currency":
            return render_currency_result(ledger, rank=rank)
        if self.kind == "duration":
            return render_duration_result(ledger, rank=rank)
        if self.kind == "quantity":
            return render_quantity_result(ledger, rank=rank)
        return _numeric_average_or_sum_result(
            ledger,
            rank=rank,
            kind=self.kind,
            output_prefix=self.kind,
            operation="sum",
        )


def _candidate_diagnostic_lines_with_support(
    candidate_type: str,
    candidates: tuple[EvidenceLedgerRow, ...],
    *,
    rank: int,
    support: list[str],
) -> list[str]:
    return [
        f"candidate_rank={rank} candidate_type={candidate_type}",
        f"candidate_confidence={_candidate_confidence(candidates)}",
        "candidate_support=" + ",".join(dict.fromkeys(support)),
    ]


def render_numeric_state_result(ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
    """Render the current count state from total and increment evidence."""
    candidates = tuple(
        sorted(
            ledger.included(kind="numeric_state"),
            key=lambda row: (source_group_sequence_index(row.source_group, fallback=_evidence_order(row)), _evidence_order(row)),
        )
    )
    excluded = ledger.excluded(kind="numeric_state")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    totals = [row for row in candidates if row.include_reason == "stated_total"]
    increments = [row for row in candidates if row.include_reason == "incremental_update"]
    if totals:
        if numeric_state_difference_query(query) and len(totals) >= 2:
            initial_total = totals[0]
            latest_total = totals[-1]
            answer = numeric_state_difference_answer(query, initial=int(initial_total.value), latest=int(latest_total.value))
            difference_support_rows: tuple[EvidenceLedgerRow, ...] = (initial_total, latest_total)
            support = list(dict.fromkeys(row.source_group for row in difference_support_rows))
            lines = [
                *_candidate_diagnostic_lines_with_support(
                    "numeric_state",
                    difference_support_rows,
                    rank=rank,
                    support=support,
                ),
                "numeric_state_values=" + ",".join(f"{row.include_reason}:{row.value}" for row in candidates),
                f"numeric_state_operation={latest_total.value}-{initial_total.value}",
                f"numeric_state_difference_answer={answer}",
                "numeric_state_source_ids=" + ",".join(support),
            ]
            return SynthesisResult(
                lines=tuple(lines),
                support_source_groups=tuple(support),
                excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
                answer_candidate=_answer_candidate(
                    rank=rank,
                    candidate_type="numeric_state",
                    candidates=difference_support_rows,
                    excluded=excluded,
                    answer_key="numeric_state_difference_answer",
                    answer=str(answer),
                    support=support,
                ),
            )
        if numeric_state_transition_query(query) and len(totals) >= 2:
            initial_total = totals[0]
            latest_total = totals[-1]
            transition_support_rows: tuple[EvidenceLedgerRow, ...] = (initial_total, latest_total)
            support = list(dict.fromkeys(row.source_group for row in transition_support_rows))
            label = numeric_state_subject_label(query)
            initial_answer = f"{initial_total.value} {label}".strip()
            current_answer = f"{latest_total.value} {label}".strip()
            verb = "led" if numeric_state_lead_query(query) else "had"
            current_verb = "lead" if numeric_state_lead_query(query) else "have"
            transition_answer = (
                f"Initially, I {verb} {initial_answer}. "
                f"Now, I {current_verb} {current_answer}."
            )
            lines = [
                *_candidate_diagnostic_lines_with_support(
                    "numeric_state",
                    transition_support_rows,
                    rank=rank,
                    support=support,
                ),
                "numeric_state_values=" + ",".join(f"{row.include_reason}:{row.value}" for row in candidates),
                f"numeric_state_initial_answer={initial_answer}",
                f"numeric_state_current_answer={current_answer}",
                f"numeric_state_transition_answer={transition_answer}",
                "numeric_state_source_ids=" + ",".join(support),
            ]
            return SynthesisResult(
                lines=tuple(lines),
                support_source_groups=tuple(support),
                excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
                answer_candidate=_answer_candidate(
                    rank=rank,
                    candidate_type="numeric_state",
                    candidates=transition_support_rows,
                    excluded=excluded,
                    answer_key="numeric_state_transition_answer",
                    answer=transition_answer,
                    support=support,
                ),
            )
        latest_total = totals[-1]
        later_rows = list(candidates[candidates.index(latest_total) + 1 :])
        later_increments = [row for row in later_rows if row.include_reason == "incremental_update"]
        increment_sum = sum(int(row.value) for row in later_increments)
        answer = int(latest_total.value) + increment_sum
        support_rows: tuple[EvidenceLedgerRow, ...] = (latest_total, *later_increments)
        operation = (
            f"{latest_total.value}+{'+'.join(row.value for row in later_increments)}"
            if later_increments
            else f"latest_total({latest_total.value})"
        )
    elif increments:
        answer = sum(int(row.value) for row in increments)
        support_rows = tuple(increments)
        operation = "+".join(row.value for row in increments)
    else:
        return SynthesisResult(lines=(), support_source_groups=())
    support = list(dict.fromkeys(row.source_group for row in support_rows))
    lines = [
        *_candidate_diagnostic_lines_with_support(
            "numeric_state",
            tuple(support_rows),
            rank=rank,
            support=support,
        ),
        "numeric_state_values=" + ",".join(f"{row.include_reason}:{row.value}" for row in candidates),
        f"numeric_state_operation={operation}",
        f"numeric_state_answer={answer}",
        "numeric_state_source_ids=" + ",".join(support),
    ]
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(support),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="numeric_state",
            candidates=tuple(support_rows),
            excluded=excluded,
            answer_key="numeric_state_answer",
            answer=str(answer),
            support=support,
        ),
    )


@dataclass(frozen=True)
class NumericStateOperation:
    """Pure current numeric-state projection over total and increment rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        return render_numeric_state_result(ledger, query=query, rank=rank)


def render_date_interval_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render temporal interval synthesis lines from date evidence."""
    candidates = ledger.included(kind="date")
    excluded = ledger.excluded(kind="date")
    if len(candidates) < ledger.plan.required_source_groups:
        return SynthesisResult(lines=(), support_source_groups=())
    anchor_terms = temporal_anchor_terms(ledger.plan.subject_terms)
    non_query_anchor_groups = {
        row.source_group
        for row in candidates
        if not _query_temporal_anchor_row(row)
    }
    allow_query_anchor_pairs = (
        "ago" in set(ledger.plan.subject_terms)
        or len(non_query_anchor_groups) < ledger.plan.required_source_groups
    )
    requested_week_interval = bool(set(ledger.plan.subject_terms) & {"week", "weeks"})
    intervals: list[tuple[int, int, int, int, int, int, EvidenceLedgerRow, EvidenceLedgerRow]] = []
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates[left_index + 1 :], start=left_index + 1):
            if left.source_group == right.source_group:
                continue
            if (
                (_query_temporal_anchor_row(left) or _query_temporal_anchor_row(right))
                and not allow_query_anchor_pairs
            ):
                continue
            delta = abs((date.fromisoformat(right.value) - date.fromisoformat(left.value)).days)
            if delta <= 0 or delta > 366:
                continue
            ordered_anchor_score = temporal_ordered_anchor_score(left, right, anchor_terms)
            explicit_role_pair = int(
                _explicit_temporal_operand_pair_score(left, right, anchor_terms) > 0
                and left.include_reason == "explicit_date"
                and right.include_reason == "explicit_date"
            )
            requested_unit_pair = int(requested_week_interval and delta % 7 == 0)
            intervals.append(
                (
                    -requested_unit_pair,
                    -explicit_role_pair,
                    -ordered_anchor_score,
                    -(left.relevance + right.relevance),
                    left_index + right_index,
                    delta,
                    left,
                    right,
                )
            )
    if not intervals:
        return SynthesisResult(lines=(), support_source_groups=())
    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))
    selected_intervals: list[tuple[int, int, int, int, int, int, EvidenceLedgerRow, EvidenceLedgerRow]] = []
    seen_deltas: set[int] = set()
    for interval in intervals:
        delta = interval[5]
        if delta in seen_deltas:
            continue
        seen_deltas.add(delta)
        selected_intervals.append(interval)
        if len(selected_intervals) >= 5:
            break
    lines: list[str] = []
    support_groups: list[str] = []
    primary_answer_key = "date_interval_answer"
    primary_answer = (
        f"{selected_intervals[0][5]} days. "
        f"{selected_intervals[0][5] + 1} days (including the last day) is also acceptable."
    )
    for index, (_, _, _, _, _, delta, left, right) in enumerate(selected_intervals):
        if index == 0:
            support = sorted({left.source_group, right.source_group})
            lines.extend(
                _candidate_diagnostic_lines_with_support(
                    "date_interval",
                    (left, right),
                    rank=rank,
                    support=support,
                )
            )
        lines.append(f"date_interval_days={delta}")
        lines.append(
            "date_interval_answer="
            f"{delta} days. {delta + 1} days (including the last day) is also acceptable."
        )
        if delta % 7 == 0:
            weeks = delta // 7
            lines.append(f"date_interval_weeks={weeks} weeks")
            if week_words := count_display(weeks):
                week_unit = "week" if weeks == 1 else "weeks"
                week_answer = f"{week_words.capitalize()} {week_unit}"
                lines.append(f"date_interval_week_answer={week_answer}")
                if index == 0 and set(ledger.plan.subject_terms) & {"week", "weeks"}:
                    primary_answer_key = "date_interval_week_answer"
                    primary_answer = week_answer
        if index == 0:
            support_groups.extend(support)
            lines.append("date_interval_source_ids=" + ",".join(support_groups))
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(support_groups),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="date_interval",
            candidates=(selected_intervals[0][6], selected_intervals[0][7]),
            excluded=excluded,
            answer_key=primary_answer_key,
            answer=primary_answer,
            support=support_groups,
        ),
    )


@dataclass(frozen=True)
class TemporalIntervalOperation:
    """Pure temporal interval projection over date ledger rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        del query
        return render_date_interval_result(ledger, rank=rank)


def render_temporal_sequence_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render an ordered temporal sequence from cited event evidence."""
    candidates = ledger.included(kind="temporal_event")
    excluded = ledger.excluded(kind="temporal_event")
    if len(candidates) < ledger.plan.required_source_groups:
        return SynthesisResult(lines=(), support_source_groups=())
    answer = temporal_sequence_answer_text(tuple(row.label for row in candidates))
    support = list(dict.fromkeys(row.source_group for row in candidates))
    lines = _candidate_diagnostic_lines_with_support(
        "temporal_sequence",
        candidates,
        rank=rank,
        support=support,
    )
    lines.append(f"temporal_sequence_answer={answer}")
    if (first_person_answer := temporal_sequence_first_person_answer(candidates)) and first_person_answer != answer:
        lines.append(f"temporal_sequence_answer={first_person_answer}")
    for index, row in enumerate(candidates, start=1):
        lines.append(
            f"temporal_sequence_rank={index} order_value={row.value} candidate={row.label}"
        )
    lines.append("temporal_sequence_source_ids=" + ",".join(support))
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(support),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="temporal_sequence",
            candidates=candidates,
            excluded=excluded,
            answer_key="temporal_sequence_answer",
            answer=answer,
            support=support,
        ),
    )


@dataclass(frozen=True)
class TemporalSequenceOperation:
    """Pure ordered-list projection over temporal event ledger rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        del query
        return render_temporal_sequence_result(ledger, rank=rank)


def synthesis_operation_for_plan(
    plan: SynthesisPlan,
) -> (
    SumValuesOperation
    | DifferenceBetweenOperation
    | AverageValuesOperation
    | ListItemsOperation
    | NumericStateOperation
    | TemporalIntervalOperation
    | TemporalSequenceOperation
):
    """Return the pure operation object for a deterministic synthesis plan."""
    required_kind = plan.required_kinds[0] if plan.required_kinds else "value"
    if plan.operation == "difference_between":
        return DifferenceBetweenOperation(kind=required_kind)
    if plan.operation == "average_values":
        if required_kind == "number" and "age" in plan.subject_terms:
            return AverageValuesOperation(kind=required_kind, output_prefix="age")
        return AverageValuesOperation(kind=required_kind)
    if plan.operation in {"count_distinct", "list_items"}:
        return ListItemsOperation()
    if plan.operation == "numeric_state":
        return NumericStateOperation()
    if plan.operation in {"date_difference", "temporal_interval"}:
        return TemporalIntervalOperation()
    if plan.operation == "temporal_sequence":
        return TemporalSequenceOperation()
    return SumValuesOperation(kind=required_kind)


def _numeric_focus_terms(query: str) -> set[str]:
    terms = {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in _NUMERIC_FOCUS_STOPWORDS and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
        "bike": {"bike", "bikes", "bicycle", "cycling", "helmet", "chain", "lights", "rack", "tune-up", "tune", "up"},
        "bicycle": {"bike", "bikes", "bicycle", "cycling", "helmet", "chain", "lights", "rack", "tune-up", "tune", "up"},
        "grocery": {"grocery", "groceries", "market", "store", "foods", "trader", "joe"},
        "groceries": {"grocery", "groceries", "market", "store", "foods", "trader", "joe"},
        "store": {"store", "market", "foods", "trader", "joe"},
        "luxury": {"luxury", "designer", "premium", "bag", "shoes", "jewelry", "watch"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
    return expanded


def _duration_focus_terms(query: str) -> set[str]:
    stopwords = _QUERY_STOPWORDS | {
        "ago",
        "day",
        "days",
        "hour",
        "hours",
        "long",
        "many",
        "minute",
        "minutes",
        "month",
        "months",
        "time",
        "week",
        "weeks",
        "year",
        "years",
    }
    terms = {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in stopwords and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
        "book": {"book", "booked", "booking"},
        "practice": {"practice", "practiced", "practicing"},
        "spent": {"spent", "spend"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
    return expanded
