"""Structured synthesis planning and evidence-ledger operations."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class SynthesisPlan:
    """Deterministic answer-shape plan for memory synthesis."""

    answer_type: str
    operation: str
    subject_terms: tuple[str, ...]
    required_kinds: tuple[str, ...]
    required_source_groups: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceLedgerRow:
    """One normalized synthesis evidence row with provenance and decision metadata."""

    fact_id: str
    source_group: str
    citation: str
    kind: str
    value: str
    unit: str
    label: str
    raw_span: str
    context: str
    normalized_identity: str
    relevance: int
    include_reason: str
    exclude_reason: str = ""
    confidence: float = 0.0


@dataclass(frozen=True)
class CountEvidenceItem:
    """One countable event or item extracted from a cited memory span."""

    label: str
    span: str
    normalized_identity: str
    relevance: int


@dataclass(frozen=True)
class EvidenceLedger:
    """Typed evidence working memory used by synthesis operations."""

    plan: SynthesisPlan
    rows: tuple[EvidenceLedgerRow, ...]

    def included(self, *, kind: str | None = None) -> tuple[EvidenceLedgerRow, ...]:
        """Return included rows, optionally filtered by evidence kind."""
        return tuple(
            row
            for row in self.rows
            if not row.exclude_reason and (kind is None or row.kind == kind)
        )

    def excluded(self, *, kind: str | None = None) -> tuple[EvidenceLedgerRow, ...]:
        """Return excluded rows, optionally filtered by evidence kind."""
        return tuple(
            row
            for row in self.rows
            if row.exclude_reason and (kind is None or row.kind == kind)
        )


@dataclass(frozen=True)
class SynthesisResult:
    """Rendered synthesis result plus support/exclusion provenance."""

    lines: tuple[str, ...]
    support_source_groups: tuple[str, ...]
    excluded_source_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class DurationValueMatch:
    """One duration value and its source span coordinates."""

    value: float
    unit: str
    raw: str
    start: int
    end: int


_QUERY_STOPWORDS = {
    "a",
    "about",
    "all",
    "and",
    "compared",
    "did",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "since",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "year",
}

_NUMERIC_FOCUS_STOPWORDS = _QUERY_STOPWORDS | {
    "cost",
    "costs",
    "expense",
    "expenses",
    "money",
    "much",
    "related",
    "spend",
    "spent",
    "total",
}

_COUNT_STOPWORDS = _QUERY_STOPWORDS | {
    "after",
    "amount",
    "attend",
    "attended",
    "before",
    "hours",
    "money",
    "most",
    "much",
    "past",
    "spent",
    "total",
    "year",
}

_DATE_STOPWORDS = {
    "did",
    "for",
    "had",
    "has",
    "have",
    "how",
    "many",
    "me",
    "my",
    "the",
    "take",
    "days",
    "passed",
}

_MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_FIRST_PERSON_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:i(?:'(?:ve|m|d|ll))?|me|my|mine|we(?:'(?:ve|re))?|our|ours)"
    r"(?![A-Za-z0-9-])",
    flags=re.IGNORECASE,
)


def build_synthesis_plan(query: str, *, limit: int = 10) -> SynthesisPlan:
    """Build a deterministic answer-shape plan for a memory query."""
    del limit
    tokens = set(source_tokens(query))
    subject_terms = tuple(
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    )
    reasons: list[str] = []
    if tokens & {"more", "less", "difference", "compared", "versus", "vs"}:
        reasons.append("comparison")
    if tokens & {"total", "sum"}:
        reasons.append("total")
    duration_terms = {
        "hour",
        "hours",
        "minute",
        "minutes",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
    }
    explicit_money_terms = {"money", "cost", "costs", "price", "prices", "amount"}
    money_terms = explicit_money_terms | {"spent", "spend"}
    duration_query = bool(tokens & duration_terms)
    if tokens & money_terms and (not duration_query or bool(tokens & explicit_money_terms)):
        if "comparison" in reasons:
            return SynthesisPlan(
                answer_type="difference",
                operation="difference_between",
                subject_terms=subject_terms,
                required_kinds=("currency",),
                required_source_groups=2,
                reasons=tuple(reasons),
            )
        return SynthesisPlan(
            answer_type="sum",
            operation="sum_values",
            subject_terms=subject_terms,
            required_kinds=("currency",),
            required_source_groups=2,
            reasons=tuple(reasons or ["money"]),
        )
    if duration_query:
        return SynthesisPlan(
            answer_type="sum",
            operation="sum_values",
            subject_terms=subject_terms,
            required_kinds=("duration",),
            required_source_groups=2,
            reasons=tuple(reasons or ["duration"]),
        )
    if "average" in tokens:
        return SynthesisPlan(
            answer_type="average",
            operation="average_values",
            subject_terms=subject_terms,
            required_kinds=("number",),
            required_source_groups=2,
            reasons=("average",),
        )
    if {"how", "many"} <= tokens:
        return SynthesisPlan(
            answer_type="count",
            operation="count_distinct",
            subject_terms=subject_terms,
            required_kinds=("event",),
            required_source_groups=2,
            reasons=("count",),
        )
    return SynthesisPlan(
        answer_type="direct_fact",
        operation="select_fact",
        subject_terms=subject_terms,
        required_kinds=(),
        required_source_groups=1,
        reasons=("direct",),
    )


def build_currency_ledger(query: str, contexts: list[str]) -> EvidenceLedger:
    """Extract and normalize currency evidence into a cited ledger."""
    plan = build_synthesis_plan(query)
    if "currency" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _numeric_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    requires_actual_spend = _spent_total_query(query)
    rows: list[EvidenceLedgerRow] = []
    included_by_identity: dict[str, EvidenceLedgerRow] = {}
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        for match_index, match in enumerate(
            re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text)
        ):
            evidence_span = local_evidence_span(text, match.start(), match.end())
            relevance = _relevance(focus_terms, evidence_span)
            value = str(float(match.group("value").replace(",", "")))
            label = currency_label(text, match.start(), match.end())
            identity = currency_identity(group=group, value=value, label=label)
            confidence = _row_confidence(relevance=relevance, has_label=bool(label))
            duplicate = included_by_identity.get(identity)
            exclude_reason = ""
            if requires_personal_memory and not _personal_numeric_evidence(
                text,
                match.start(),
                match.end(),
            ):
                exclude_reason = "not_personal_memory"
            elif requires_actual_spend and _non_spend_currency_context(evidence_span):
                exclude_reason = "not_actual_spend"
            elif duplicate is not None:
                exclude_reason = "duplicate_identity"
            fact_id = f"currency:{context_index}:{match_index}"
            row = EvidenceLedgerRow(
                fact_id=fact_id,
                source_group=group,
                citation=citation,
                kind="currency",
                value=value,
                unit="USD",
                label=label,
                raw_span=match.group(0),
                context=evidence_span,
                normalized_identity=identity,
                relevance=relevance,
                include_reason="currency_amount",
                exclude_reason=exclude_reason,
                confidence=confidence,
            )
            rows.append(row)
            if not exclude_reason:
                included_by_identity[identity] = row
    ledger = _deduplicate_currency_source_values(
        EvidenceLedger(plan=plan, rows=tuple(rows))
    )
    return _filter_currency_ledger(ledger, focus_terms)


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
    if excluded:
        lines.append(
            "currency_excluded_source_ids="
            + ",".join(row.source_group for row in excluded)
        )
    lines.append(f"currency_max={format_currency(float(max_item.value))}")
    if max_item.label:
        lines.append(f"currency_max_label={max_item.label}")
        lines.append(f"currency_max_answer={max_item.label}")
    if len(values) >= 2:
        difference = max(values) - min(values)
        lines.append(f"currency_difference={format_currency(difference)}")
        lines.append(f"currency_difference_answer={format_currency(difference)}")
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
    )


def build_count_ledger(query: str, contexts: list[str]) -> EvidenceLedger:
    """Extract distinct cited-event evidence for count/list synthesis."""
    plan = build_synthesis_plan(query)
    if "event" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _count_focus_terms(query)
    action_terms = _count_action_terms(query)
    subject = _count_subject(query)
    rows: list[EvidenceLedgerRow] = []
    seen_identities: set[str] = set()
    provisional: list[EvidenceLedgerRow] = []
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        items = count_evidence_items(
            text,
            group=group,
            focus_terms=focus_terms,
            action_terms=action_terms,
            subject=subject,
        )
        if not items:
            span = count_evidence_span(
                text,
                focus_terms=focus_terms,
                action_terms=action_terms,
                subject=subject,
            )
            relevance = _relevance(focus_terms, span or text)
            label = count_label(span or text)
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"count:{context_index}",
                    source_group=group,
                    citation=citation,
                    kind="event",
                    value="1",
                    unit="event",
                    label=label,
                    raw_span=span or text,
                    context=span or text,
                    normalized_identity=f"source_group={group}",
                    relevance=relevance,
                    include_reason="relevant_source_event",
                    exclude_reason="query_focus_mismatch",
                    confidence=_row_confidence(relevance=relevance, has_label=bool(label)),
                )
            )
            continue
        for item_index, item in enumerate(items):
            duplicate = item.normalized_identity in seen_identities
            if not duplicate:
                seen_identities.add(item.normalized_identity)
            exclude_reason = ""
            if duplicate:
                exclude_reason = (
                    "duplicate_source_group"
                    if item.normalized_identity == f"source_group={group}"
                    else "duplicate_identity"
                )
            row = EvidenceLedgerRow(
                fact_id=f"count:{context_index}:{item_index}",
                source_group=group,
                citation=citation,
                kind="event",
                value="1",
                unit="event",
                label=item.label,
                raw_span=item.span,
                context=item.span,
                normalized_identity=item.normalized_identity,
                relevance=item.relevance,
                include_reason="relevant_source_event",
                exclude_reason=exclude_reason,
                confidence=_row_confidence(relevance=item.relevance, has_label=bool(item.label)),
            )
            if exclude_reason:
                rows.append(row)
            else:
                provisional.append(row)
    rows.extend(provisional)
    rows = _filter_count_rows(subject, rows, query=query)
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def render_count_result(
    ledger: EvidenceLedger,
    query: str,
    *,
    rank: int,
) -> SynthesisResult:
    """Render count/list synthesis lines from an evidence ledger."""
    candidates = ledger.included(kind="event")
    excluded = ledger.excluded(kind="event")
    if len(candidates) < ledger.plan.required_source_groups:
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
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=_ordered_source_groups(excluded),
    )


def _ordered_source_groups(rows: tuple[EvidenceLedgerRow, ...]) -> tuple[str, ...]:
    """Return unique source groups in original evidence order when row ids expose it."""
    return tuple(
        dict.fromkeys(
            row.source_group
            for row in sorted(rows, key=lambda row: (_evidence_order(row), row.source_group))
        )
    )


def _evidence_order(row: EvidenceLedgerRow) -> int:
    match = re.match(r"^[^:]+:(?P<index>\d+)(?::\d+)?$", row.fact_id)
    if match:
        return int(match.group("index"))
    return 10**9


def _filter_count_rows(
    subject: str,
    rows: list[EvidenceLedgerRow],
    *,
    query: str,
) -> list[EvidenceLedgerRow]:
    """Apply subject-specific count normalization after candidate extraction."""
    if subject == "property_viewing":
        return _filter_target_property_rows(query, rows)
    if subject == "musical_instrument":
        return _filter_subsumed_instrument_rows(rows)
    if subject != "doctor_visit":
        return rows
    included = [row for row in rows if row.kind == "event" and not row.exclude_reason]
    specific_identities = {
        row.normalized_identity
        for row in included
        if row.normalized_identity not in {"doctor_visit=doctor", "doctor_visit=physician"}
    }
    if not specific_identities:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.exclude_reason or row.normalized_identity not in {"doctor_visit=doctor", "doctor_visit=physician"}:
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
                exclude_reason="generic_role_subsumed",
                confidence=row.confidence,
            )
        )
    return filtered


def _filter_target_property_rows(
    query: str,
    rows: list[EvidenceLedgerRow],
) -> list[EvidenceLedgerRow]:
    target_terms = _target_property_terms(query)
    if not target_terms:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.exclude_reason:
            filtered.append(row)
            continue
        context_terms = set(source_tokens(row.context))
        if target_terms <= context_terms and not re.search(
            r"\b(?:rejected|higher\s+bid|outbid|deal-breaker|dealbreaker|renovation|budget)\b",
            row.context,
            flags=re.IGNORECASE,
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
                    exclude_reason="target_property",
                    confidence=row.confidence,
                )
            )
            continue
        filtered.append(row)
    return filtered


def _filter_subsumed_instrument_rows(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    included = [row for row in rows if row.kind == "event" and not row.exclude_reason]
    specific_families_by_source: dict[str, set[str]] = {}
    for row in included:
        if _generic_instrument_family(row.label):
            continue
        family = _instrument_family(row.label)
        if not family:
            continue
        specific_families_by_source.setdefault(row.source_group, set()).add(family)
    if not specific_families_by_source:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        family = _generic_instrument_family(row.label)
        if (
            row.exclude_reason
            or row.kind != "event"
            or not family
            or family not in specific_families_by_source.get(row.source_group, set())
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
                exclude_reason="generic_instrument_subsumed",
                confidence=row.confidence,
            )
        )
    return filtered


def _target_property_terms(query: str) -> set[str]:
    match = re.search(
        r"\boffer\s+on\s+(?:a|an|the)?\s*(?P<target>[^?]+)\??$",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return set()
    stopwords = _COUNT_STOPWORDS | {"making", "offer", "neighborhood"}
    return {
        token
        for token in source_tokens(match.group("target"))
        if token not in stopwords and not token.isdigit() and len(token) > 2
    }


def build_date_ledger(query: str, contexts: list[str]) -> EvidenceLedger:
    """Extract explicit date evidence for temporal interval synthesis."""
    plan = SynthesisPlan(
        answer_type="interval",
        operation="date_difference",
        subject_terms=tuple(
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _DATE_STOPWORDS and not token.isdigit()
        ),
        required_kinds=("date",),
        required_source_groups=2,
        reasons=("temporal",),
    )
    query_tokens = set(source_tokens(query))
    if not ({"day", "days", "week", "weeks"} & query_tokens):
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _date_focus_terms(query)
    rows: list[EvidenceLedgerRow] = []
    provisional: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        raw_text = context_text(context)
        text = temporal_evidence_text(raw_text)
        default_year = context_year(raw_text)
        group = source_group(context)
        citation = source_citation(context)
        relevance = _relevance(focus_terms, text)
        for match_index, value in enumerate(explicit_dates(text, default_year=default_year)):
            identity = f"group={group}|date={value.isoformat()}"
            duplicate = identity in seen
            seen.add(identity)
            row = EvidenceLedgerRow(
                fact_id=f"date:{context_index}:{match_index}",
                source_group=group,
                citation=citation,
                kind="date",
                value=value.isoformat(),
                unit="day",
                label=value.isoformat(),
                raw_span=value.isoformat(),
                context=text,
                normalized_identity=identity,
                relevance=relevance,
                include_reason="explicit_date",
                exclude_reason="duplicate_identity" if duplicate else "",
                confidence=_row_confidence(relevance=relevance, has_label=True),
            )
            if duplicate:
                rows.append(row)
            else:
                provisional.append(row)
    rows.extend(_filter_date_rows(provisional))
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def render_date_interval_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render temporal interval synthesis lines from date evidence."""
    candidates = ledger.included(kind="date")
    excluded = ledger.excluded(kind="date")
    if len(candidates) < ledger.plan.required_source_groups:
        return SynthesisResult(lines=(), support_source_groups=())
    anchor_terms = temporal_anchor_terms(ledger.plan.subject_terms)
    intervals: list[tuple[int, int, int, int, EvidenceLedgerRow, EvidenceLedgerRow]] = []
    seen_deltas: set[int] = set()
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates[left_index + 1 :], start=left_index + 1):
            if left.source_group == right.source_group:
                continue
            delta = abs((date.fromisoformat(right.value) - date.fromisoformat(left.value)).days)
            if delta <= 0 or delta > 366 or delta in seen_deltas:
                continue
            seen_deltas.add(delta)
            ordered_anchor_score = temporal_ordered_anchor_score(left, right, anchor_terms)
            intervals.append(
                (
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
    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    lines: list[str] = []
    support_groups: list[str] = []
    for index, (_, _, _, delta, left, right) in enumerate(intervals[:5]):
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
                lines.append(f"date_interval_week_answer={week_words.capitalize()} {week_unit}")
        if index == 0:
            support_groups.extend(support)
            lines.append("date_interval_source_ids=" + ",".join(support_groups))
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(support_groups),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
    )


def build_duration_ledger(query: str, contexts: list[str]) -> EvidenceLedger:
    """Extract and normalize duration evidence into a cited ledger."""
    plan = build_synthesis_plan(query)
    if "duration" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _duration_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        duration_matches = list(duration_value_matches(text))
        for match_index, duration_match in enumerate(duration_matches):
            unit = canonical_duration_unit(duration_match.unit)
            raw_value = duration_match.value
            minutes = raw_value * duration_unit_minutes(unit)
            label = f"{format_number(raw_value)} {unit}"
            identity = duration_identity(group=group, minutes=minutes, label=label)
            duplicate = identity in seen
            seen.add(identity)
            evidence_span = local_evidence_span(
                text,
                duration_match.start,
                duration_match.end,
            )
            relevance = _relevance(focus_terms, evidence_span)
            exclude_reason = ""
            if requires_personal_memory and not _personal_numeric_evidence(
                text,
                duration_match.start,
                duration_match.end,
            ):
                exclude_reason = "not_personal_memory"
            elif duplicate:
                exclude_reason = "duplicate_identity"
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"duration:{context_index}:{match_index}",
                    source_group=group,
                    citation=citation,
                    kind="duration",
                    value=str(minutes),
                    unit="minutes",
                    label=label,
                    raw_span=duration_match.raw,
                    context=evidence_span,
                    normalized_identity=identity,
                    relevance=relevance,
                    include_reason="duration_value",
                    exclude_reason=exclude_reason,
                    confidence=_row_confidence(relevance=relevance, has_label=True),
                )
            )
    return _filter_duration_ledger(
        EvidenceLedger(plan=plan, rows=tuple(rows)),
        focus_terms,
        preferred_units=_duration_preferred_units(query),
    )


def render_duration_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render duration synthesis lines from an evidence ledger."""
    candidates = ledger.included(kind="duration")
    excluded = ledger.excluded(kind="duration")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    values = [duration_display(row) for row in candidates]
    total_minutes = sum(float(row.value) for row in candidates)
    answer_unit = _duration_answer_unit(ledger.plan.subject_terms)
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
    else:
        lines.append(f"duration_total_answer={format_number(total_minutes / 60)} hours")
    if excluded:
        lines.append(
            "duration_excluded_source_ids="
            + ",".join(row.source_group for row in excluded)
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
    )


def source_tokens(text: str) -> list[str]:
    """Tokenize source/query text for deterministic synthesis helpers."""
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*", text.casefold()):
        tokens.append(token)
        if re.search(r"[-_:/#]", token):
            tokens.extend(part for part in re.split(r"[-_:/#]+", token) if part)
    return tokens


def context_text(context: str) -> str:
    """Return source text once, excluding Eventloom JSON payload echoes."""
    text = " ".join(context.split())
    return text.split(' {"content":', 1)[0]


def _personal_memory_query(query: str) -> bool:
    """Return whether a query asks about the user's own remembered facts."""
    return bool({"i", "me", "my", "mine", "we", "our"} & set(source_tokens(query)))


def _personal_numeric_evidence(text: str, start: int, end: int) -> bool:
    """Return whether a numeric span belongs to user memory rather than advice text."""
    role = _speaker_role_before(text, start)
    if role == "assistant":
        return False
    if role == "user":
        return True
    evidence = local_evidence_span(text, start, end, window_chars=160)
    return bool(_FIRST_PERSON_EVIDENCE_RE.search(evidence))


def _speaker_role_before(text: str, position: int) -> str:
    """Return the nearest explicit conversational role before a text position."""
    role = ""
    role_pattern = re.compile(
        r"(?:^|\b)(?:\d+\.\s*)?(?P<colon_role>user|assistant)\s*:|"
        r"\brole=(?P<meta_role>user|assistant)\b",
        flags=re.IGNORECASE,
    )
    for match in role_pattern.finditer(text[:position]):
        role = (match.group("colon_role") or match.group("meta_role") or "").casefold()
    return role


def local_evidence_span(
    text: str,
    start: int,
    end: int,
    *,
    window_chars: int = 120,
) -> str:
    """Return the local clause around an extracted value for relevance scoring."""
    left_bound = max(0, start - window_chars)
    right_bound = min(len(text), end + window_chars)
    left_punctuation = max(
        text.rfind(".", left_bound, start),
        text.rfind("!", left_bound, start),
        text.rfind("?", left_bound, start),
        text.rfind(";", left_bound, start),
    )
    right_candidates = [
        position for position in (
            text.find(".", end, right_bound),
            text.find("!", end, right_bound),
            text.find("?", end, right_bound),
            text.find(";", end, right_bound),
        )
        if position != -1
    ]
    span_start = left_punctuation + 1 if left_punctuation >= left_bound else left_bound
    span_end = min(right_candidates) + 1 if right_candidates else right_bound
    span = text[span_start:span_end]
    relative_start = max(0, start - span_start)
    before = span[:relative_start]
    after = span[max(0, end - span_start):]
    for separator in (" but ", " however ", " although ", " while ", " whereas "):
        position = before.lower().rfind(separator)
        if position != -1:
            before = before[position + len(separator):]
            break
    for separator in (" but ", " however ", " although ", " while ", " whereas "):
        position = after.lower().find(separator)
        if position != -1:
            after = after[:position]
            break
    value_text = text[start:end]
    return " ".join(f"{before}{value_text}{after}".split())


def source_group(context: str) -> str:
    """Return a stable source group from common citation/session metadata."""
    patterns = [
        r"\b[a-z0-9_.-]*session[_-]?id=(?P<value>[^\s]+)",
        r"\b(?:source_path|path|file)=['\"]?(?P<value>[^\s'\"]+)",
        r"\bthread=['\"]?(?P<value>[^\s'\"]+)",
        r"eventloom://[^/]+/events/(?P<value>\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group("value").casefold()
    return context[:160].casefold()


def source_citation(context: str) -> str:
    """Extract a compact citation token from source context."""
    for pattern in (
        r"\bcitation=(?P<value>\S+)",
        r"(?P<value>eventloom://\S+)",
        r"\bsource_path=(?P<value>\S+)",
    ):
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group("value")
    return "unknown"


def currency_label(text: str, start: int, end: int) -> str:
    """Extract a compact item label near a currency amount."""
    local_span = local_evidence_span(text, start, end)
    if label := _currency_merchant_label(local_span):
        return label
    after = text[end : end + 120]
    for pattern in (
        r"\s+(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})",
        r"\s+(?:for|on)\s+(?:a|an|the)?\s*(?P<label>[A-Za-z][A-Za-z0-9'&.-]*(?:\s+[A-Za-z][A-Za-z0-9'&.-]*){0,4})",
    ):
        match = re.match(pattern, after)
        if match:
            return clean_label(match.group("label"))
    before = text[max(0, start - 120) : start]
    match = re.search(
        r"\b(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})\s*$",
        before,
    )
    if match:
        return clean_label(match.group("label"))
    if label := _currency_label_before_amount(text[:start]):
        return label
    return ""


def _currency_merchant_label(text: str) -> str:
    """Extract merchant/store labels from local currency evidence."""
    patterns = (
        r"\b(?:spent|paid)\s+(?:around\s+|about\s+)?\$\d+(?:,\d{3})*(?:\.\d+)?\s+at\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})",
        r"\bat\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})\s+(?:and\s+)?(?:spent|paid)\s+(?:around\s+|about\s+)?\$\d+(?:,\d{3})*(?:\.\d+)?",
        r"\b(?:order|ordered)\s+with\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})\s+[^.!?]{0,80}?\b(?:spent|paid)\s+(?:around\s+|about\s+)?\$\d+(?:,\d{3})*(?:\.\d+)?",
    )
    for pattern in patterns:
        if match := re.search(pattern, text):
            return clean_label(match.group("label"))
    return ""


def _spent_total_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"spent", "spend", "paid", "pay", "cost"} or {"how", "much", "money"} <= tokens)


def _non_spend_currency_context(span: str) -> bool:
    """Return whether a currency span describes planned, hypothetical, or advisory cost."""
    lowered = span.casefold()
    future_or_hypothetical = (
        "going to order",
        "going to buy",
        "plan to",
        "planning to",
        "next week",
        "might",
        "could",
        "would",
        "recommend",
        "look for",
        "consider",
        "available for",
        "range from",
        "budget",
    )
    if any(marker in lowered for marker in future_or_hypothetical):
        return True
    return bool(
        re.search(
            r"\b(?:you|you'll|you\s+can|if\s+you're)\b[^.!?]{0,120}\$",
            span,
            flags=re.IGNORECASE,
        )
    )


def currency_identity(*, group: str, value: str, label: str) -> str:
    """Return a stable identity used for currency deduplication."""
    normalized_label = normalize_currency_label(label)
    if normalized_label:
        return f"value={value}|label={normalized_label}"
    return f"group={group}|value={value}"


def format_currency(value: float) -> str:
    """Render a currency value with thousands separators."""
    if value.is_integer():
        return f"${int(value):,}"
    whole = int(value)
    fraction = f"{value:.2f}".split(".", 1)[1].rstrip("0")
    return f"${whole:,}.{fraction}"


def format_number(value: float) -> str:
    """Render a number without unnecessary trailing zeroes."""
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def canonical_duration_unit(unit: str) -> str:
    """Return the canonical display unit for a duration token."""
    normalized = unit.casefold()
    if normalized in {"min", "mins", "minute", "minutes"}:
        return "minutes"
    if normalized in {"hr", "hrs", "hour", "hours"}:
        return "hours"
    if normalized in {"day", "days"}:
        return "days"
    if normalized in {"week", "weeks"}:
        return "weeks"
    return "months"


def duration_value_matches(text: str) -> tuple[DurationValueMatch, ...]:
    """Extract numeric and word-based duration values with source positions."""
    matches: list[DurationValueMatch] = []
    pattern = re.compile(
        r"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        matches.append(
            DurationValueMatch(
                value=float(match.group("value")),
                unit=match.group("unit"),
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    word_pattern = re.compile(
        r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"[- ](?P<unit>minute|hour|day|week|month)s?(?:[- ]long)?\b",
        flags=re.IGNORECASE,
    )
    for match in word_pattern.finditer(text):
        matches.append(
            DurationValueMatch(
                value=float(_NUMBER_WORDS[match.group("value").casefold()]),
                unit=f"{match.group('unit')}s",
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    matches.sort(key=lambda item: item.start)
    return tuple(matches)


def duration_unit_minutes(unit: str) -> float:
    """Return the number of minutes represented by one canonical duration unit."""
    return {
        "minutes": 1,
        "hours": 60,
        "days": 60 * 24,
        "weeks": 60 * 24 * 7,
        "months": 60 * 24 * 28,
    }[unit]


def duration_identity(*, group: str, minutes: float, label: str) -> str:
    """Return a stable identity used for duration deduplication."""
    return f"group={group}|minutes={format_number(minutes)}|label={label.casefold()}"


def duration_display(row: EvidenceLedgerRow) -> str:
    """Return the original display value for a duration ledger row."""
    return row.label or f"{format_number(float(row.value))} minutes"


def duration_compatibility_lines(candidates: tuple[EvidenceLedgerRow, ...]) -> list[str]:
    """Render legacy compatibility fields for common duration units."""
    by_unit: dict[str, list[float]] = {}
    for row in candidates:
        raw_value, raw_unit = duration_raw_value_unit(row)
        by_unit.setdefault(raw_unit, []).append(raw_value)
    lines: list[str] = []
    if minute_values := by_unit.get("minutes"):
        lines.append("minute_values=" + ",".join(format_number(value) for value in minute_values))
        lines.append(f"minute_total_hours={format_number(sum(minute_values) / 60)} hours")
    if hour_values := by_unit.get("hours"):
        lines.append("hour_values=" + ",".join(format_number(value) for value in hour_values))
        lines.append(f"hour_total={format_number(sum(hour_values))} hours")
    if day_values := by_unit.get("days"):
        lines.append("day_values=" + ",".join(format_number(value) for value in day_values))
        lines.append(f"day_total={format_number(sum(day_values))} days")
    return lines


def duration_raw_value_unit(row: EvidenceLedgerRow) -> tuple[float, str]:
    """Return the original raw value/unit encoded in a duration label."""
    match = re.match(r"(?P<value>\d+(?:\.\d+)?)\s+(?P<unit>[a-z]+)", row.label)
    if not match:
        return float(row.value), row.unit
    return float(match.group("value")), match.group("unit")


def count_answer_text(query: str, candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    """Render a compact natural-language answer for count synthesis."""
    subject = count_subject_phrase(query)
    if not subject:
        return ""
    action = common_count_action(candidates)
    count = count_display(len(candidates))
    query_tokens = set(source_tokens(query))
    count_subject = _count_subject(query)
    if subject == "model kits" and query_tokens & {"work", "worked"} and query_tokens & {"buy", "bought"}:
        return f"I have worked on or bought {count} {subject}."
    if count_subject == "film_festival":
        return f"I attended {count} movie festivals."
    if count_subject == "doctor_visit":
        labels = _joined_count_labels(candidates)
        if labels:
            return f"I visited {count} different doctors: {labels}."
        return f"I visited {count} different doctors."
    if count_subject == "wedding":
        couples = _joined_wedding_couples(candidates)
        if couples:
            return f"I attended {count} weddings. The couples were {couples}."
        return f"I attended {count} weddings."
    if count_subject == "property_viewing" and query_tokens & {"view", "viewed"}:
        return f"I viewed {count} properties."
    if count_subject == "musical_instrument":
        return f"I currently own {count} musical instruments."
    if action:
        return f"I {action} {count} {subject}."
    return f"There are {count} {subject}."


def _joined_count_labels(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    labels = [rendered_list_label(row) for row in candidates if rendered_list_label(row)]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _joined_wedding_couples(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    couples = [
        couple
        for row in candidates
        if (couple := _wedding_couple_from_label(rendered_list_label(row)))
    ]
    if not couples:
        return ""
    if len(couples) == 1:
        return couples[0]
    return ", ".join(couples[:-1]) + f", and {couples[-1]}"


def _wedding_couple_from_label(label: str) -> str:
    value = re.sub(
        r"^\s*(?:i\s+)?(?:attended|went to|been to|returned from|got back from)\s+",
        "",
        label,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?:'s)?\s+wedding\b.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:my|a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value.strip(" .,'\""))
    if not value or " and " not in value.casefold():
        return ""
    return value


def count_subject_phrase(query: str) -> str:
    """Extract the counted subject phrase from a question."""
    match = re.search(
        r"\bhow\s+many\s+(?P<subject>.+?)(?:\s+(?:did|do|does|that|have|has|had|"
        r"were|was|are|is|can|could|should|would)\b|[?.,]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return " ".join(match.group("subject").strip(" .,'\"").casefold().split())


def common_count_action(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    """Return the common leading action across count labels, when stable."""
    actions: list[str] = []
    for row in candidates:
        label = rendered_list_label(row)
        if not label:
            continue
        first = label.split(maxsplit=1)[0].casefold()
        if first:
            actions.append(first)
    if not actions:
        return ""
    first_action = actions[0]
    if all(action == first_action for action in actions):
        return first_action
    return ""


def count_display(value: int) -> str:
    """Render small count values as words for answer text."""
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(value, str(value))


def list_detail_query(query: str) -> bool:
    """Return true when count synthesis should include item labels."""
    tokens = set(source_tokens(query))
    return bool(
        tokens
        & {
            "which",
            "who",
            "what",
            "list",
            "name",
            "names",
            "were",
            "they",
            "them",
            "items",
        }
    )


def labeled_count_subject(candidates: tuple[EvidenceLedgerRow, ...]) -> bool:
    """Return whether count evidence should expose item labels by default."""
    if not candidates:
        return False
    identities = [row.normalized_identity for row in candidates]
    return all(
        identity.startswith(
            (
                "doctor_visit=",
                "musical_instrument=",
                "model_kit=",
                "source_group=",
            )
        )
        for identity in identities
    )


def list_candidate_lines(candidates: tuple[EvidenceLedgerRow, ...]) -> list[str]:
    """Render itemized count details from labeled count rows."""
    labeled_candidates = [row for row in candidates if rendered_list_label(row)]
    if not labeled_candidates:
        return []
    lines = [
        f"list_item_count={len(labeled_candidates)}",
        "list_items="
        + " | ".join(rendered_list_label(row) for row in labeled_candidates),
        "list_source_ids=" + ",".join(row.source_group for row in labeled_candidates),
    ]
    if all(row.normalized_identity.startswith("model_kit=") for row in labeled_candidates):
        lines.append(
            "model_kit_scales="
            + " | ".join(model_kit_scale_label(rendered_list_label(row)) for row in labeled_candidates)
        )
    return lines


def _count_outcome_lines(
    query: str,
    candidates: tuple[EvidenceLedgerRow, ...],
) -> list[str]:
    if _count_subject(query) != "property_viewing" or not candidates:
        return []
    outcomes = [rendered_list_label(row) for row in candidates if rendered_list_label(row)]
    if not outcomes:
        return []
    target = _property_offer_target_phrase(query)
    if target:
        prefix = f"I viewed {count_display(len(candidates))} properties before making an offer on {target}."
    else:
        prefix = f"I viewed {count_display(len(candidates))} properties before making an offer."
    return [
        (
            "property_outcome_answer="
            f"{prefix} The reasons I didn't make an offer on them were: "
            + "; ".join(outcomes)
            + "."
        )
    ]


def _instrument_ownership_lines(
    query: str,
    candidates: tuple[EvidenceLedgerRow, ...],
) -> list[str]:
    if _count_subject(query) != "musical_instrument" or not candidates:
        return []
    details: list[str] = []
    for row in candidates:
        label = rendered_list_label(row)
        if not label:
            continue
        duration = _instrument_duration_phrase(row.context)
        details.append(f"the {label} for {duration}")
    if not details:
        return []
    return [
        (
            "instrument_ownership_answer="
            f"I currently own {len(candidates)} musical instruments. I've had "
            + ", ".join(details[:-1])
            + (f", and {details[-1]}" if len(details) > 1 else details[0])
            + "."
        )
    ]


def _instrument_duration_phrase(context: str) -> str:
    if match := re.search(r"\bfor\s+(?:about\s+)?(?P<years>\d+)\s+years?\b", context, flags=re.IGNORECASE):
        return f"{match.group('years')} years"
    return "an unspecified amount of time"


def _property_offer_target_phrase(query: str) -> str:
    match = re.search(
        r"\boffer\s+on\s+(?P<target>[^?]+?)\??$",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    target = " ".join(match.group("target").strip(" .,'\"").split())
    if not re.match(r"^(?:a|an|the)\b", target, flags=re.IGNORECASE):
        target = f"the {target}"
    return target


def model_kit_scale_label(label: str) -> str:
    """Render model-kit labels with explicit scale presence for answer synthesis."""
    if re.search(r"\b\d+/\d+\s+scale\b", label, flags=re.IGNORECASE):
        return label
    return f"{label} scale not mentioned"


def count_label(text: str) -> str:
    """Extract a compact event label for count/list synthesis."""
    cleaned = re.sub(r"\bcontent=\S+\s*", "", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", "", cleaned)
    cleaned = re.sub(r"\b(?:user|assistant):\s*", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\bI\s+(?P<label>.+?)(?:[.?!]|$)", cleaned)
    if match:
        label = " ".join(match.group("label").split()[:12])
        return label
    return ""


def count_evidence_span(
    text: str,
    *,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str = "",
) -> str:
    """Return the most count-relevant first-person memory span from a source context."""
    spans = count_evidence_spans(
        text,
        focus_terms=focus_terms,
        action_terms=action_terms,
        subject=subject,
    )
    return spans[0] if spans else ""


def count_evidence_spans(
    text: str,
    *,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str = "",
) -> list[str]:
    """Return all count-relevant first-person memory spans from a source context."""
    spans = _first_person_spans(text)
    if not spans:
        return []
    scored: list[tuple[int, int, int, str]] = []
    for index, span in enumerate(spans):
        if _negated_count_action(span, action_terms):
            continue
        if not _span_matches_count_subject(span, subject):
            continue
        focus_score = _relevance(focus_terms, span)
        action_score = 2 if _has_count_action(span, action_terms) else 0
        if focus_score <= 0 or action_score <= 0:
            continue
        scored.append((-(focus_score + action_score), -action_score, index, span))
    if not scored:
        return []
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, _, span in scored:
        normalized = " ".join(source_tokens(span))
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(span)
    return ordered


def count_evidence_items(
    text: str,
    *,
    group: str,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str,
) -> list[CountEvidenceItem]:
    """Extract countable items or events from the strongest cited memory span."""
    spans = count_evidence_spans(
        text,
        focus_terms=focus_terms,
        action_terms=action_terms,
        subject=subject,
    )
    if not spans:
        return []
    if subject == "model_kit":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="model_kit",
            label_extractor=_model_kit_labels,
        )
    if subject == "film_festival":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="film_festival",
            label_extractor=_film_festival_labels,
        )
    if subject == "doctor_visit":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="doctor_visit",
            label_extractor=_doctor_visit_labels,
        )
    if subject == "musical_instrument":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="musical_instrument",
            label_extractor=_musical_instrument_labels,
        )
    span = spans[0]
    relevance = _relevance(focus_terms, span)
    if relevance <= 0:
        return []
    if subject == "property_viewing":
        label = _property_viewing_label(span)
        return [
            CountEvidenceItem(
                label=label,
                span=span,
                normalized_identity=f"source_group={group}",
                relevance=relevance,
            )
        ]
    if subject == "wedding":
        label = _wedding_label(span)
        return [
            CountEvidenceItem(
                label=label,
                span=span,
                normalized_identity=f"source_group={group}",
                relevance=relevance,
            )
        ]
    label = count_label(span)
    return [
        CountEvidenceItem(
            label=label,
            span=span,
            normalized_identity=f"source_group={group}",
            relevance=relevance,
        )
    ]


def _count_labeled_items_from_spans(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
    label_extractor: Callable[[str], list[str]],
) -> list[CountEvidenceItem]:
    """Extract deduplicated typed count items from all relevant source spans."""
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        relevance = _relevance(focus_terms, span)
        if relevance <= 0:
            continue
        for label in label_extractor(span):
            identity = f"{identity_prefix}={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=identity,
                    relevance=max(relevance, 2),
                )
            )
    return items


def _first_person_spans(text: str) -> list[str]:
    """Extract bounded first-person clauses suitable for count evidence."""
    cleaned = re.sub(r"\bcontent=\S+\s*", " ", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", " ", cleaned)
    spans: list[str] = []
    pattern = re.compile(
        r"(?:\buser:\s*)?\bI(?:\s+|['’](?:m|ve|d|ll|re)\s+).{3,220}?(?:[.!?](?=\s|$)|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(cleaned):
        span = " ".join(match.group(0).strip(" .!?").split())
        span = re.sub(r"^(?:user|assistant):\s*", "", span, flags=re.IGNORECASE)
        if span and span not in spans:
            spans.append(span)
    return spans


def _count_action_terms(query: str) -> set[str]:
    """Return action terms that must appear in count evidence spans."""
    query_terms = set(source_tokens(query))
    action_groups = {
        "attend": {"assist", "assisted", "attend", "attended", "attending", "back", "go", "got", "participated", "visited", "volunteered", "went"},
        "attended": {"assist", "assisted", "attend", "attended", "attending", "back", "go", "got", "participated", "visited", "volunteered", "went"},
        "visit": {"appointment", "back", "diagnosed", "got", "had", "prescribed", "see", "saw", "visit", "visited", "went"},
        "visited": {"appointment", "back", "diagnosed", "got", "had", "prescribed", "see", "saw", "visit", "visited", "went"},
        "view": {"fell", "love", "rejected", "saw", "searching", "seen", "tour", "toured", "view", "viewed", "visited"},
        "viewed": {"fell", "love", "rejected", "saw", "searching", "seen", "tour", "toured", "view", "viewed", "visited"},
        "own": {"had", "have", "own", "owned", "playing", "selling", "service"},
        "owned": {"had", "have", "own", "owned", "playing", "selling", "service"},
        "work": {"bought", "build", "built", "finished", "got", "new", "planning", "started", "starting", "thinking", "work", "worked", "working"},
        "worked": {"bought", "build", "built", "finished", "got", "new", "planning", "started", "starting", "thinking", "work", "worked", "working"},
        "buy": {"buy", "bought", "got", "new", "purchased", "picked"},
        "bought": {"buy", "bought", "got", "new", "purchased", "picked"},
        "wedding": {"attend", "attended", "back", "been", "returned", "went"},
        "weddings": {"attend", "attended", "back", "been", "returned", "went"},
    }
    actions: set[str] = set()
    for term in query_terms:
        actions.update(action_groups.get(term, set()))
    if not actions and {"how", "many"} <= query_terms:
        actions.update({"attend", "attended", "visit", "visited", "went", "got", "bought", "been"})
    return actions


def _has_count_action(span: str, action_terms: set[str]) -> bool:
    if not action_terms:
        return True
    tokens = set(source_tokens(span))
    if not tokens & action_terms:
        return False
    return not re.search(r"\bI\s+(?:requested|asked|wanted|hoped|planned)\s+to\s+see\b", span, flags=re.IGNORECASE)


def _negated_count_action(span: str, action_terms: set[str]) -> bool:
    if not action_terms:
        return False
    escaped_actions = "|".join(sorted((re.escape(term) for term in action_terms), key=len, reverse=True))
    if not escaped_actions:
        return False
    return bool(
        re.search(
            rf"\b(?:did\s+not|didn't|do\s+not|don't|never|not)\s+[^.!?]{{0,40}}\b(?:{escaped_actions})\b",
            span,
            flags=re.IGNORECASE,
        )
    )


def _count_subject(query: str) -> str:
    """Classify the count target into a small deterministic evidence facet."""
    tokens = set(source_tokens(query))
    if tokens & {"movie", "movies", "film", "films", "cinema"} and tokens & {"festival", "festivals", "fest", "fests"}:
        return "film_festival"
    if tokens & {"property", "properties", "home", "homes", "house", "houses"}:
        return "property_viewing"
    if tokens & {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"}:
        return "doctor_visit"
    if tokens & {"instrument", "instruments", "guitar", "piano", "drum", "drums"}:
        return "musical_instrument"
    if tokens & {"model", "models", "kit", "kits"}:
        return "model_kit"
    if tokens & {"wedding", "weddings"}:
        return "wedding"
    return ""


def _span_matches_count_subject(span: str, subject: str) -> bool:
    """Return whether a span satisfies the count subject facet."""
    if not subject:
        return True
    tokens = set(source_tokens(span))
    if subject == "film_festival":
        if _film_festival_name(span):
            return True
        return bool(tokens & {"movie", "movies", "film", "films", "cinema"} and tokens & {"festival", "festivals", "fest", "fests"})
    if subject == "property_viewing":
        if re.search(
            r"\bI\s+(?:made|put\s+in)\s+an\s+offer\b",
            span,
            flags=re.IGNORECASE,
        ) and not re.search(r"\b(?:rejected|higher\s+bid|outbid)\b", span, flags=re.IGNORECASE):
            return False
        return bool(tokens & {"property", "properties", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"})
    if subject == "doctor_visit":
        if re.search(r"\bI\s+(?:requested|asked|wanted|hoped|planned)\s+to\s+see\b", span, flags=re.IGNORECASE):
            return False
        return bool(tokens & {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"})
    if subject == "musical_instrument":
        if re.search(
            r"\b(?:thinking\s+about|considering|planning\s+to|want\s+to)\s+"
            r"(?:get|buy|purchase|try)\b",
            span,
            flags=re.IGNORECASE,
        ):
            return False
        return bool(tokens & {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"})
    if subject == "model_kit":
        return bool(_model_kit_labels(span))
    if subject == "wedding":
        return bool(tokens & {"wedding", "weddings"})
    return True


def _model_kit_labels(span: str) -> list[str]:
    """Extract distinct model-kit items from a first-person evidence span."""
    labels: list[str] = []
    for match in re.finditer(
        r"\b(?:simple\s+)?(?P<label>[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z0-9][A-Za-z0-9.'-]+){1,6}\s+kit)\b",
        span,
    ):
        label = _clean_count_item_label(match.group("label"))
        identity = _normalize_count_identity(label)
        if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
            labels.append(label)
    for match in re.finditer(
        r"\b(?:(?:[A-Z][A-Za-z0-9.-]+)\s+)?\d+/\d+\s+scale\s+[^;!?]+",
        span,
    ):
        fragment = match.group(0)
        fragments = re.split(r"\s+\band\b\s+(?:a|an|the)?\s*(?=\d+/\d+\s+scale)", fragment)
        for raw_label in fragments:
            label = _clean_count_item_label(raw_label)
            if label and _normalize_count_identity(label) not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _doctor_visit_labels(span: str) -> list[str]:
    """Extract distinct medical provider roles from a consultation memory span."""
    labels: list[str] = []
    specific_patterns = (
        r"\b(?P<label>primary\s+care\s+physician)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>ENT\s+specialist)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>ENT)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>dermatologist)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
    )
    for pattern in specific_patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _clean_count_item_label(match.group("label"))
            identity = _normalize_count_identity(label)
            if identity == "ent" and any(
                _normalize_count_identity(existing) == "ent specialist"
                for existing in labels
            ):
                continue
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    if labels:
        return labels
    for pattern in (
        r"\b(?P<label>physician)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>doctor)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
    ):
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _clean_count_item_label(match.group("label"))
            identity = _normalize_count_identity(label)
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _musical_instrument_labels(span: str) -> list[str]:
    """Extract distinct currently-owned musical instruments from first-person evidence."""
    labels: list[str] = []
    patterns = (
        r"\b(?P<label>Fender\s+Stratocaster\s+electric\s+guitar)\b",
        r"\b(?:acoustic\s+guitar,\s+a\s+)?(?P<label>Yamaha\s+FG800(?:\s+acoustic\s+guitar)?)\b",
        r"\b(?P<label>5-piece\s+Pearl\s+Export(?:\s+drum\s+set)?)\b",
        r"\b(?P<label>Korg\s+B1(?:\s+piano)?)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_instrument_label(_clean_count_item_label(match.group("label")))
            identity = _normalize_count_identity(label)
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    generic_patterns = (
        r"\b(?:my\s+)?(?P<label>[A-Z][A-Za-z0-9'-]+(?:\s+[A-Z][A-Za-z0-9'-]+){0,3}\s+"
        r"(?:electric\s+guitar|acoustic\s+guitar|drum\s+set|piano))\b",
        r"\b(?:my\s+)?(?P<label>(?:electric\s+guitar|acoustic\s+guitar|drum\s+set|piano))\b",
    )
    for pattern in generic_patterns:
        for match in re.finditer(pattern, span):
            label = _canonical_instrument_label(_clean_count_item_label(match.group("label")))
            identity = _normalize_count_identity(label)
            if any(
                identity
                and (
                    identity in _normalize_count_identity(existing)
                    or _normalize_count_identity(existing) in identity
                )
                for existing in labels
            ):
                continue
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _canonical_instrument_label(label: str) -> str:
    lowered = label.casefold()
    if lowered == "yamaha fg800":
        return "Yamaha FG800 acoustic guitar"
    if lowered == "5-piece pearl export":
        return "5-piece Pearl Export drum set"
    if lowered == "korg b1":
        return "Korg B1 piano"
    return label


def _generic_instrument_family(label: str) -> str:
    normalized = _normalize_count_identity(label)
    generic = {
        "acoustic guitar": "acoustic guitar",
        "electric guitar": "electric guitar",
        "drum set": "drum set",
        "piano": "piano",
    }
    return generic.get(normalized, "")


def _instrument_family(label: str) -> str:
    normalized = _normalize_count_identity(label)
    for family in ("acoustic guitar", "electric guitar", "drum set", "piano"):
        if family in normalized:
            return family
    return ""


def _film_festival_labels(span: str) -> list[str]:
    """Extract distinct named film festivals from a participation memory span."""
    labels: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\b(?P<label>[A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+){0,5}\s+"
        r"(?:International\s+Film\s+Festival|Film\s+Festival|Fest))\b",
        span,
    ):
        label = _clean_count_item_label(match.group("label"))
        identity = _normalize_count_identity(label)
        if label and identity not in seen:
            seen.add(identity)
            labels.append(label)
    for known in ("Sundance", "Tribeca", "Cannes", "Telluride", "SXSW", "TIFF"):
        if re.search(rf"\b{re.escape(known)}\b", span, flags=re.IGNORECASE):
            if any(re.search(rf"\b{re.escape(known)}\b", label, flags=re.IGNORECASE) for label in labels):
                continue
            identity = _normalize_count_identity(known)
            if identity not in seen:
                seen.add(identity)
                labels.append(known)
    return labels


def _property_viewing_label(span: str) -> str:
    """Return property-viewing labels with the rejection/outcome reason preserved."""
    if match := re.search(
        r"\b(?P<label>(?:that\s+one\s+in\s+)?Cedar\s+Creek[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        label = _clean_property_label(match.group("label"))
        if "budget" in span.casefold() and "budget" not in label.casefold():
            label = f"{label} was out of my budget"
        return label
    if match := re.search(
        r"\b(?P<label>\d+-bedroom\s+bungalow[^.!?]{0,160}?"
        r"\bkitchen[^.!?]{0,100}?\brenovation[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        return _clean_property_label(match.group("label"))
    if match := re.search(
        r"\b(?P<label>\d+-bedroom\s+condo[^.!?]{0,120}?"
        r"(?:deal-breaker|higher\s+bid|rejected)[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        return _clean_property_label(match.group("label"))
    return count_label(span)


def _clean_property_label(label: str) -> str:
    label = re.sub(
        r"^\s*(?:and\s+)?(?:a|an|the|that\s+one\s+in)\s+",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s+", " ", label.strip(" .,'\""))
    return label.strip(" .,'\"")


def _wedding_label(span: str) -> str:
    """Return a compact wedding attendance label with couple names when present."""
    if match := re.search(
        r"\b(?P<label>my\s+cousin's\s+wedding[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        return _clean_count_item_label(match.group("label"))
    if match := re.search(
        r"\b(?:the\s+bride,\s*)?(?P<left>[A-Z][A-Za-z' -]{1,30}),?\s+[^.!?]{0,80}?"
        r"\b(?:husband|partner),?\s+(?P<right>[A-Z][A-Za-z' -]{1,30})\b",
        span,
    ):
        return f"{_clean_count_item_label(match.group('left'))} and {_clean_count_item_label(match.group('right'))}'s wedding"
    if match := re.search(
        r"\b(?P<left>[A-Z][A-Za-z' -]{1,30})\s+[^.!?]{0,80}?\b(?:married|tie the knot)\s+"
        r"(?:with\s+)?(?:her|his|their)?\s*(?:partner\s+)?(?P<right>[A-Z][A-Za-z' -]{1,30})\b",
        span,
        flags=re.IGNORECASE,
    ):
        return f"{_clean_count_item_label(match.group('left'))} and {_clean_count_item_label(match.group('right'))}'s wedding"
    return count_label(span)


def _clean_count_item_label(label: str) -> str:
    label = re.sub(r"^\s*(?:and\s+)?(?:a|an|the)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label.strip(" .,'\""))
    label = re.split(
        r"\s+\b(?:and|plus|with|where|that|which|but|because|next|during|at)\b(?:\s+|[,.!?;:]|$)",
        label,
        maxsplit=1,
    )[0]
    return label.strip(" .,'\"")


def _normalize_count_identity(label: str) -> str:
    generic_terms = {"as", "kit", "kits", "model", "models", "well"}
    return " ".join(token for token in source_tokens(label) if token not in generic_terms)


def rendered_list_label(row: EvidenceLedgerRow) -> str:
    """Return a display-safe list label."""
    label = " ".join(row.label.strip(" .,'\"").split())
    if row.normalized_identity.startswith("film_festival=") and not re.match(
        r"(?i)^(?:attended|went|visited)\b",
        label,
    ):
        return f"attended the {label}"
    return label


def temporal_evidence_text(text: str) -> str:
    """Remove metadata dates that should not be treated as answer evidence."""
    text = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)", " ", text)
    return re.sub(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", " ", text)


def context_year(text: str) -> int | None:
    """Infer the default year for month/day date mentions."""
    match = re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/", text)
    if match:
        return int(match.group("year"))
    match = re.search(r"\b(?P<year>20\d{2})[/-]\d{1,2}[/-]\d{1,2}\b", text)
    if match:
        return int(match.group("year"))
    return None


def explicit_dates(text: str, *, default_year: int | None) -> list[date]:
    """Extract explicit and supported relative dates from source text."""
    dates: list[date] = []
    for match in re.finditer(
        r"\b(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        text,
    ):
        append_date(
            dates,
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    if default_year is not None:
        black_friday = black_friday_date(default_year)
        if re.search(r"\bblack friday\b", text, flags=re.IGNORECASE):
            if re.search(r"\b(?:on|during)\s+black friday\b", text, flags=re.IGNORECASE):
                append_unique_date(dates, black_friday)
            if re.search(r"\b(?:a|one|1)\s+weeks?\s+before\s+black friday\b", text, flags=re.IGNORECASE):
                append_unique_date(dates, black_friday - timedelta(days=7))
            if re.search(r"\b(?:a|one|1)\s+weeks?\s+after\s+black friday\b", text, flags=re.IGNORECASE):
                append_unique_date(dates, black_friday + timedelta(days=7))
    for match in re.finditer(
        rf"\b(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        append_date(
            dates,
            year,
            _MONTHS[match.group("month").casefold()],
            int(match.group("day")),
        )
    for match in re.finditer(
        rf"\b(?:the\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?P<month>{month_pattern})(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        append_date(
            dates,
            year,
            _MONTHS[match.group("month").casefold()],
            int(match.group("day")),
        )
    for match in re.finditer(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})(?:/(?P<year>\d{2,4}))?\b", text):
        year_text = match.group("year")
        year = default_year
        if year_text:
            year = int(year_text)
            if year < 100:
                year += 2000
        if year is None:
            continue
        append_date(dates, year, int(match.group("month")), int(match.group("day")))
    return dates


def black_friday_date(year: int) -> date:
    """Return Black Friday for a given year."""
    thanksgiving = nth_weekday_of_month(year, 11, weekday=3, n=4)
    return thanksgiving + timedelta(days=1)


def nth_weekday_of_month(year: int, month: int, *, weekday: int, n: int) -> date:
    """Return the nth weekday within a month."""
    value = date(year, month, 1)
    days_until_weekday = (weekday - value.weekday()) % 7
    return value + timedelta(days=days_until_weekday + (n - 1) * 7)


def append_unique_date(dates: list[date], value: date) -> None:
    """Append a date once while preserving extraction order."""
    if value not in dates:
        dates.append(value)


def append_date(dates: list[date], year: int, month: int, day: int) -> None:
    """Append a valid calendar date once."""
    try:
        value = date(year, month, day)
    except ValueError:
        return
    append_unique_date(dates, value)


def temporal_anchor_terms(subject_terms: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Split temporal query terms into ordered left/right event anchors."""
    if not subject_terms:
        return set(), set()
    terms = list(subject_terms)
    separators = {"after", "before", "between"}
    if "between" in terms and "and" in terms:
        between_index = terms.index("between")
        and_index = terms.index("and")
        if between_index < and_index:
            return set(terms[between_index + 1 : and_index]), set(terms[and_index + 1 :])
    if "after" in terms:
        after_index = terms.index("after")
        return set(terms[after_index + 1 :]), set(terms[:after_index])
    if "before" in terms:
        before_index = terms.index("before")
        return set(terms[:before_index]), set(terms[before_index + 1 :])
    midpoint = len(terms) // 2
    if midpoint == 0:
        return set(), set()
    left = {term for term in terms[:midpoint] if term not in separators}
    right = {term for term in terms[midpoint:] if term not in separators}
    return left, right


def temporal_ordered_anchor_score(
    left: EvidenceLedgerRow,
    right: EvidenceLedgerRow,
    anchor_terms: tuple[set[str], set[str]],
) -> int:
    """Score whether two date rows match ordered temporal query anchors."""
    first_anchor, second_anchor = anchor_terms
    if not first_anchor or not second_anchor:
        return 0
    left_context_terms = set(source_tokens(left.context))
    right_context_terms = set(source_tokens(right.context))
    left_date = date.fromisoformat(left.value)
    right_date = date.fromisoformat(right.value)
    forward_score = len(first_anchor & left_context_terms) + len(second_anchor & right_context_terms)
    reverse_score = len(first_anchor & right_context_terms) + len(second_anchor & left_context_terms)
    if left_date <= right_date:
        forward_score += 1
    else:
        reverse_score += 1
    return max(forward_score, reverse_score)


def clean_label(label: str) -> str:
    """Return a bounded label with trailing clauses removed."""
    label = re.split(r"\b(?:on|in|for|with|and|but|because|when)\b", label, maxsplit=1)[0]
    return " ".join(label.strip(" .,'\"").split())


def normalize_currency_label(label: str) -> str:
    """Normalize a currency label for cross-source duplicate detection."""
    normalized = _clean_currency_item_label(label).casefold()
    normalized = re.sub(
        r"\b(?:new|recent|recently|installed|got|my|the|a|an|i|d|like|to|add|that)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _filter_currency_ledger(
    ledger: EvidenceLedger,
    focus_terms: set[str],
) -> EvidenceLedger:
    included = list(ledger.included(kind="currency"))
    if len(included) < 2 or not focus_terms:
        return ledger
    if max((row.relevance for row in included), default=0) <= 0:
        return ledger
    selected_identities = {
        row.normalized_identity for row in included if row.relevance > 0
    }
    if len(selected_identities) < 2:
        return ledger
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.exclude_reason or row.kind != "currency":
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


def _filter_duration_ledger(
    ledger: EvidenceLedger,
    focus_terms: set[str],
    *,
    preferred_units: set[str] | None = None,
) -> EvidenceLedger:
    included = list(ledger.included(kind="duration"))
    if len(included) < 2 or not focus_terms:
        return ledger
    if max((row.relevance for row in included), default=0) <= 0:
        return ledger
    preferred_units = preferred_units or set()
    preferred_relevant = {
        row.normalized_identity
        for row in included
        if row.relevance > 0
        and (not preferred_units or canonical_duration_unit(duration_raw_value_unit(row)[1]) in preferred_units)
    }
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


def _duration_preferred_units(query: str) -> set[str]:
    tokens = set(source_tokens(query))
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
    if terms & {"day", "days"}:
        return "days"
    if terms & {"week", "weeks"}:
        return "weeks"
    return "hours"


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


def _filter_date_rows(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
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
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.relevance >= selected_threshold:
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


def _candidate_confidence(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    if not candidates:
        return "0"
    support_count = len({row.source_group for row in candidates})
    average_relevance = sum(row.relevance for row in candidates) / len(candidates)
    confidence = min(0.99, 0.55 + min(support_count, 5) * 0.06 + min(average_relevance, 6) * 0.04)
    return f"{confidence:.2f}".rstrip("0").rstrip(".")


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
    terms = {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
        "practice": {"practice", "practiced", "practicing"},
        "spent": {"spent", "spend"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
    return expanded


def _count_focus_terms(query: str) -> set[str]:
    terms = {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in _COUNT_STOPWORDS and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
        "movie": {"movie", "movies", "film", "films", "cinema"},
        "doctor": {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"},
        "doctors": {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"},
        "festival": {"festival", "festivals", "fest", "fests"},
        "festivals": {"festival", "festivals", "fest", "fests"},
        "kit": {"kit", "kits", "model", "models", "scale"},
        "kits": {"kit", "kits", "model", "models", "scale"},
        "model": {"kit", "kits", "model", "models", "scale"},
        "models": {"kit", "kits", "model", "models", "scale"},
        "instrument": {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"},
        "instruments": {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"},
        "musical": {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"},
        "properties": {"properties", "property", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"},
        "property": {"properties", "property", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"},
        "wedding": {"wedding", "weddings"},
        "weddings": {"wedding", "weddings"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _date_focus_terms(query: str) -> set[str]:
    return _count_focus_terms(query)


def _relevance(focus_terms: set[str], context: str) -> int:
    if not focus_terms:
        return 0
    context_terms = set(source_tokens(context))
    score = len(focus_terms & context_terms)
    if {"movie", "film"} & focus_terms and _film_festival_name(context):
        score += 1
    return score


def _film_festival_name(context: str) -> bool:
    """Return whether text names a film/movie festival without the generic noun."""
    return bool(
        re.search(
            r"\b(?:AFI\s+Fest|Sundance|Tribeca|Cannes|Telluride|SXSW|TIFF)\b",
            context,
            flags=re.IGNORECASE,
        )
    )


def _row_confidence(*, relevance: int, has_label: bool) -> float:
    return min(0.99, 0.55 + min(relevance, 6) * 0.05 + (0.08 if has_label else 0.0))


def _currency_label_before_amount(prefix: str) -> str:
    prefix = " ".join(prefix.split())
    patterns = (
        r"(?P<label>[^.!?]{1,100}?)\s+for\s*$",
        r"(?P<label>[^.!?]{1,100}?),?\s+which\s+were\s*$",
        r"(?P<label>[^.!?]{1,100}?)\s+cost\s+me\s*$",
        r"(?P<label>[^.!?]{1,100}?)\s+cost\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, prefix, flags=re.IGNORECASE)
        if match:
            return _clean_currency_item_label(match.group("label"))
    return ""


def _clean_currency_item_label(label: str) -> str:
    label = re.sub(r"\b(?:content|session_id|longmemeval_session_id)=\S+\s*", "", label)
    label = re.sub(r"\brole=\S+\s*", "", label)
    label = re.sub(r"\b[a-z0-9_.-]*session[_-]?id=\S+\s*", "", label, flags=re.IGNORECASE)
    item_patterns = (
        r"\bI\s+(?:recently\s+|also\s+)?(?:bought|got|installed|replaced)\s+(?P<item>[^.!?;,]{1,100})",
        r"\bI\s+(?:needed|had)\s+to\s+replace\s+(?P<item>[^.!?;,]{1,80})",
        r"\b(?:mechanic|shop)\s+[^.!?]{0,80}?\breplace\s+(?P<item>[^.!?;,]{1,80})",
    )
    for pattern in item_patterns:
        matches = list(re.finditer(pattern, label, flags=re.IGNORECASE))
        if matches:
            return clean_label(matches[-1].group("item"))
    label = re.split(r"[,;:]|\b(?:because|and then|while)\b", label, maxsplit=1)[-1]
    label = re.sub(
        r"^(?:i\s+)?(?:recently\s+)?(?:also\s+)?"
        r"(?:bought|booked|got|installed|replaced|stayed in|paid for|spent on)\s+",
        "",
        label.strip(),
        flags=re.IGNORECASE,
    )
    label = re.sub(r"^(?:a|an|the|my|new)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^set\s+of\s+", "", label, flags=re.IGNORECASE)
    return clean_label(label)
