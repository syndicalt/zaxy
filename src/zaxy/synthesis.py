"""Structured synthesis planning and evidence-ledger operations."""

from __future__ import annotations

import re
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
    rows: list[EvidenceLedgerRow] = []
    included_by_identity: dict[str, EvidenceLedgerRow] = {}
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        relevance = _relevance(focus_terms, text)
        for match_index, match in enumerate(
            re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text)
        ):
            value = str(float(match.group("value").replace(",", "")))
            label = currency_label(text, match.start(), match.end())
            identity = currency_identity(group=group, value=value, label=label)
            confidence = _row_confidence(relevance=relevance, has_label=bool(label))
            duplicate = included_by_identity.get(identity)
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
                context=text,
                normalized_identity=identity,
                relevance=relevance,
                include_reason="currency_amount",
                exclude_reason="duplicate_identity" if duplicate is not None else "",
                confidence=confidence,
            )
            rows.append(row)
            if duplicate is None:
                included_by_identity[identity] = row
    return _filter_currency_ledger(EvidenceLedger(plan=plan, rows=tuple(rows)), focus_terms)


def render_currency_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render currency synthesis lines from an evidence ledger."""
    candidates = ledger.included(kind="currency")
    excluded = ledger.excluded(kind="currency")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    values = sorted((float(row.value) for row in candidates), reverse=True)
    total = sum(values)
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
    max_item = max(candidates, key=lambda row: float(row.value))
    lines.append(f"currency_max={format_currency(float(max_item.value))}")
    if max_item.label:
        lines.append(f"currency_max_label={max_item.label}")
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
    rows: list[EvidenceLedgerRow] = []
    seen_groups: set[str] = set()
    provisional: list[EvidenceLedgerRow] = []
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        relevance = _relevance(focus_terms, text)
        label = count_label(text)
        duplicate = group in seen_groups
        if relevance > 0 and not duplicate:
            seen_groups.add(group)
        exclude_reason = ""
        if duplicate:
            exclude_reason = "duplicate_source_group"
        elif relevance <= 0:
            exclude_reason = "query_focus_mismatch"
        row = EvidenceLedgerRow(
            fact_id=f"count:{context_index}",
            source_group=group,
            citation=citation,
            kind="event",
            value="1",
            unit="event",
            label=label,
            raw_span=text,
            context=text,
            normalized_identity=f"source_group={group}",
            relevance=relevance,
            include_reason="relevant_source_event",
            exclude_reason=exclude_reason,
            confidence=_row_confidence(relevance=relevance, has_label=bool(label)),
        )
        if exclude_reason:
            rows.append(row)
        else:
            provisional.append(row)
    if not provisional:
        return EvidenceLedger(plan=plan, rows=tuple(rows))
    best = max(row.relevance for row in provisional)
    threshold = 2 if best >= 2 else best
    rows.extend(_apply_count_threshold(provisional, threshold=threshold))
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
    if list_detail_query(query):
        lines.extend(list_candidate_lines(candidates))
    return SynthesisResult(
        lines=tuple(lines),
        support_source_groups=tuple(dict.fromkeys(row.source_group for row in candidates)),
        excluded_source_groups=tuple(dict.fromkeys(row.source_group for row in excluded)),
    )


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
    if "days" not in set(source_tokens(query)):
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
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        relevance = _relevance(focus_terms, text)
        for match_index, match in enumerate(
            re.finditer(
                r"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b",
                text,
                flags=re.IGNORECASE,
            )
        ):
            unit = canonical_duration_unit(match.group("unit"))
            raw_value = float(match.group("value"))
            minutes = raw_value * duration_unit_minutes(unit)
            label = f"{format_number(raw_value)} {unit}"
            identity = duration_identity(group=group, minutes=minutes, label=label)
            duplicate = identity in seen
            seen.add(identity)
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"duration:{context_index}:{match_index}",
                    source_group=group,
                    citation=citation,
                    kind="duration",
                    value=str(minutes),
                    unit="minutes",
                    label=label,
                    raw_span=match.group(0),
                    context=text,
                    normalized_identity=identity,
                    relevance=relevance,
                    include_reason="duration_value",
                    exclude_reason="duplicate_identity" if duplicate else "",
                    confidence=_row_confidence(relevance=relevance, has_label=True),
                )
            )
    return _filter_duration_ledger(EvidenceLedger(plan=plan, rows=tuple(rows)), focus_terms)


def render_duration_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render duration synthesis lines from an evidence ledger."""
    candidates = ledger.included(kind="duration")
    excluded = ledger.excluded(kind="duration")
    if not candidates:
        return SynthesisResult(lines=(), support_source_groups=())
    values = [duration_display(row) for row in candidates]
    total_minutes = sum(float(row.value) for row in candidates)
    lines = [
        *_candidate_diagnostic_lines("duration", candidates, rank=rank),
        "duration_values=" + ",".join(values),
        f"duration_total_minutes={format_number(total_minutes)} minutes",
        f"duration_total_hours={format_number(total_minutes / 60)} hours",
        f"duration_total_answer={format_number(total_minutes / 60)} hours",
        "duration_source_ids="
        + ",".join(row.source_group for row in candidates),
    ]
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
    if action:
        return f"I {action} {count} {subject}."
    return f"There are {count} {subject}."


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


def list_candidate_lines(candidates: tuple[EvidenceLedgerRow, ...]) -> list[str]:
    """Render itemized count details from labeled count rows."""
    labeled_candidates = [row for row in candidates if rendered_list_label(row)]
    if not labeled_candidates:
        return []
    return [
        f"list_item_count={len(labeled_candidates)}",
        "list_items="
        + " | ".join(rendered_list_label(row) for row in labeled_candidates),
        "list_source_ids=" + ",".join(row.source_group for row in labeled_candidates),
    ]


def count_label(text: str) -> str:
    """Extract a compact event label for count/list synthesis."""
    cleaned = re.sub(r"\bcontent=\S+\s*", "", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", "", cleaned)
    match = re.search(r"\bI\s+(?P<label>.+?)(?:[.?!]|$)", cleaned)
    if match:
        return " ".join(match.group("label").split()[:12])
    return ""


def rendered_list_label(row: EvidenceLedgerRow) -> str:
    """Return a display-safe list label."""
    return " ".join(row.label.strip(" .,'\"").split())


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
    normalized = re.sub(r"\b(?:new|recent|recently|installed|got|my|the|a|an)\b", " ", normalized)
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


def _filter_duration_ledger(
    ledger: EvidenceLedger,
    focus_terms: set[str],
) -> EvidenceLedger:
    included = list(ledger.included(kind="duration"))
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
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
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
        "festival": {"festival", "festivals"},
        "festivals": {"festival", "festivals"},
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
    return len(focus_terms & context_terms)


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
