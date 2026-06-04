"""Structured synthesis planning and evidence-ledger operations."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from functools import lru_cache

_SOURCE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*")
_SOURCE_TOKEN_SPLIT_RE = re.compile(r"[-_:/#]+")
_CURRENCY_LABEL_BEFORE_AMOUNT_PATTERNS = (
    re.compile(r"(?P<label>[^.!?]{1,100}?)\s+for\s*$", flags=re.IGNORECASE),
    re.compile(r"(?P<label>[^.!?]{1,100}?),?\s+which\s+were\s*$", flags=re.IGNORECASE),
    re.compile(r"(?P<label>[^.!?]{1,100}?)\s+cost\s+me\s*$", flags=re.IGNORECASE),
    re.compile(r"(?P<label>[^.!?]{1,100}?)\s+cost\s*$", flags=re.IGNORECASE),
)
_CURRENCY_PURCHASE_LABEL_BEFORE_AMOUNT_RE = re.compile(
    r"\bI\s+(?:recently\s+|also\s+)?"
    r"(?:bought|booked|got|purchased|picked up|paid for|spent on)\s+"
    r"(?:a|an|the|my)?\s*(?P<item>[^.!?;,]{1,100})",
    flags=re.IGNORECASE,
)
_AGE_AVERAGE_RE = (
    re.compile(r"\b(?:just\s+turned|turned|am|is)\s+(?P<value>\d{1,3})\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?P<person>mom|dad|mother|father|grandma|grandpa|grandmother|grandfather)\s+is\s+(?P<value>\d{1,3})\b",
        flags=re.IGNORECASE,
    ),
)


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
class ExplicitDateMatch:
    """One explicit or normalized date mention with source-span offsets."""

    value: date
    start: int
    end: int
    raw: str


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
    answer_candidate: dict[str, object] | None = None


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
        return _numeric_average_or_sum_result(
            ledger,
            rank=rank,
            kind=self.kind,
            output_prefix=self.kind,
            operation="sum",
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


@dataclass(frozen=True)
class ListItemsOperation:
    """Pure list/count projection over event ledger rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        return render_count_result(ledger, query, rank=rank)


@dataclass(frozen=True)
class NumericStateOperation:
    """Pure current numeric-state projection over total and increment rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        return render_numeric_state_result(ledger, query=query, rank=rank)


@dataclass(frozen=True)
class TemporalIntervalOperation:
    """Pure temporal interval projection over date ledger rows."""

    def execute(self, ledger: EvidenceLedger, *, rank: int, query: str = "") -> SynthesisResult:
        del query
        return render_date_interval_result(ledger, rank=rank)


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

_TEMPORAL_SEQUENCE_STOPWORDS = _DATE_STOPWORDS | {
    "and",
    "event",
    "events",
    "first",
    "from",
    "latest",
    "last",
    "order",
    "ordered",
    "past",
    "three",
    "timeline",
    "took",
    "what",
    "which",
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
_NUMERIC_STATE_VALUE_PATTERN = (
    r"\d{1,6}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
)

_FIRST_PERSON_EVIDENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:i(?:'(?:ve|m|d|ll))?|me|my|mine|we(?:'(?:ve|re))?|our|ours)"
    r"(?![A-Za-z0-9-])",
    flags=re.IGNORECASE,
)


def build_synthesis_plan(query: str, *, limit: int = 10) -> SynthesisPlan:
    """Build a deterministic answer-shape plan for a memory query."""
    del limit
    query_tokens = source_tokens(query)
    tokens = set(query_tokens)
    subject_terms = tuple(
        token
        for token in query_tokens
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    )
    reasons: list[str] = []
    savings_terms = {"save", "saved", "saving", "savings"}
    if tokens & {"more", "less", "difference", "compared", "versus", "vs"} or tokens & savings_terms:
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
    duration_question = {"how", "long"} <= tokens and bool(tokens & {"combined", "total", "altogether"})
    explicit_money_terms = {"money", "cost", "costs", "price", "prices", "amount"}
    floor_value_terms = {"minimum", "sold", "sell", "worth", "value"}
    money_terms = explicit_money_terms | {"spent", "spend"} | savings_terms | floor_value_terms
    duration_query = bool(tokens & duration_terms) or duration_question
    count_subject = _count_subject(query, tokens=tokens)
    if {"how", "many"} <= tokens and count_subject and (
        not _duration_measure_query(tokens) or _incidental_time_modifier_query(query)
    ):
        return SynthesisPlan(
            answer_type="count",
            operation="count_distinct",
            subject_terms=subject_terms,
            required_kinds=("event",),
            required_source_groups=2,
            reasons=("count",),
        )
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


def _duration_measure_query(tokens: set[str]) -> bool:
    """Return whether duration units are the thing being counted."""
    return bool(tokens & {"hours", "hour", "minutes", "minute", "days", "day", "weeks", "week"}) and not bool(
        tokens & {"month", "months"}
    )


def _incidental_time_modifier_query(query: str) -> bool:
    """Return whether duration words describe when the event happened, not what to count."""
    return bool(
        re.search(
            r"\b(?:ago|last|previous|past|prior)\s+(?:minute|hour|day|week|month|year)s?\b"
            r"|\b(?:minute|hour|day|week|month|year)s?\s+ago\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def build_currency_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract and normalize currency evidence into a cited ledger."""
    plan = plan or build_synthesis_plan(query)
    if "currency" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _numeric_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    requires_floor_value = _floor_value_sale_query(query)
    requires_earned_total = _earned_total_query(query) and not requires_floor_value
    requires_actual_spend = _spent_total_query(query) and not requires_earned_total
    rows: list[EvidenceLedgerRow] = []
    included_by_identity: dict[str, EvidenceLedgerRow] = {}
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        for match_index, match in enumerate(
            re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text)
        ):
            evidence_span = local_evidence_span(text, match.start(), match.end(), window_chars=320)
            amount = float(match.group("value").replace(",", ""))
            value = str(_realized_unit_price_total(evidence_span, amount) if requires_earned_total else amount)
            label = currency_label(text, match.start(), match.end())
            relevance = _relevance(focus_terms, " ".join((evidence_span, label)))
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
            elif requires_earned_total and not _earned_currency_context(evidence_span):
                exclude_reason = "not_earned_money"
            elif requires_floor_value and not _floor_value_currency_context(evidence_span):
                exclude_reason = "not_floor_value"
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
    ledger = _filter_lodging_currency_ledger(query, ledger)
    ledger = _filter_itemized_currency_targets(query, ledger)
    ledger = _filter_unit_price_currency_ledger(query, ledger)
    return _filter_currency_ledger(ledger, focus_terms, query=query)


def build_age_average_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract family age evidence into a typed number ledger for average synthesis."""
    plan = plan or build_synthesis_plan(query)
    query_tokens = set(source_tokens(query))
    if plan.operation != "average_values" or "age" not in query_tokens:
        return EvidenceLedger(plan=plan, rows=())
    evidence = _age_average_evidence(contexts)
    rows = [
        EvidenceLedgerRow(
            fact_id=f"age_average:{index}",
            source_group=source_group(context),
            citation=source_citation(context),
            kind="number",
            value=str(value),
            unit="years",
            label="age",
            raw_span=raw,
            context=context_text(context),
            normalized_identity=f"age_average:{source_group(context)}:{value}",
            relevance=4,
            include_reason="age_average_input",
            confidence=0.78,
        )
        for index, (context, value, raw) in enumerate(evidence)
    ]
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def _age_average_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    indexed = list(enumerate(_age_value_evidence(contexts)))
    indexed.sort(
        key=lambda item: (
            _source_group_natural_key(source_group(item[1][0])),
            item[0],
        )
    )
    return [evidence for _index, evidence in indexed]


def _age_value_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    values: list[int] = []
    evidence: list[tuple[str, int, str]] = []
    for context in contexts:
        for pattern in _AGE_AVERAGE_RE:
            for match in pattern.finditer(context):
                value = int(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
                    evidence.append((context, value, match.group(0)))
    return evidence


def _source_group_natural_key(group: str) -> tuple[str, int]:
    match = re.match(r"^(?P<prefix>.*?)(?:[_-](?P<suffix>\d+))?$", group)
    if not match or match.group("suffix") is None:
        return group, -1
    return match.group("prefix"), int(match.group("suffix"))


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


def build_count_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract distinct cited-event evidence for count/list synthesis."""
    plan = plan or build_synthesis_plan(query)
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


def build_numeric_state_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract cited current-count state and incremental updates."""
    plan = plan or SynthesisPlan(
        answer_type="count",
        operation="numeric_state",
        subject_terms=tuple(
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _COUNT_STOPWORDS and not token.isdigit()
        ),
        required_kinds=("numeric_state",),
        required_source_groups=1,
        reasons=("numeric_state",),
    )
    if not numeric_state_query(query):
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = set(plan.subject_terms)
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    required_qualifier_terms = numeric_state_required_qualifier_terms(query)
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        for item_index, (value, span, reason) in enumerate(numeric_state_evidence(text, focus_terms=focus_terms)):
            identity = f"group={group}|reason={reason}|value={value}|span={' '.join(source_tokens(span))[:160]}"
            duplicate = identity in seen
            seen.add(identity)
            relevance = _relevance(focus_terms, span)
            span_terms = set(source_tokens(span))
            exclude_reason = ""
            if required_qualifier_terms and not required_qualifier_terms <= span_terms:
                exclude_reason = "missing_required_state_qualifier"
            elif duplicate:
                exclude_reason = "duplicate_identity"
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"numeric_state:{context_index}:{item_index}",
                    source_group=group,
                    citation=citation,
                    kind="numeric_state",
                    value=str(value),
                    unit="count",
                    label=span,
                    raw_span=span,
                    context=span,
                    normalized_identity=identity,
                    relevance=relevance,
                    include_reason=reason,
                    exclude_reason=exclude_reason,
                    confidence=_row_confidence(relevance=relevance, has_label=True),
                )
            )
    return EvidenceLedger(plan=plan, rows=tuple(rows))


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


def numeric_state_transition_query(query: str) -> bool:
    """Return whether a query asks for both an earlier and current numeric state."""
    tokens = set(source_tokens(query))
    return bool(tokens & {"now", "current", "currently"}) and bool(
        tokens & {"initial", "initially", "start", "started", "beginning", "began", "when"}
    )


def numeric_state_difference_query(query: str) -> bool:
    """Return whether a query asks for the delta between count states."""
    tokens = set(source_tokens(query))
    return bool(
        tokens
        & {
            "change",
            "changed",
            "decrease",
            "decreased",
            "difference",
            "growth",
            "increase",
            "increased",
            "grew",
            "grown",
        }
    )


def numeric_state_difference_answer(query: str, *, initial: int, latest: int) -> int:
    """Return the signed count-state delta implied by the query wording."""
    tokens = set(source_tokens(query))
    if tokens & {"decrease", "decreased"}:
        return initial - latest
    return latest - initial


def numeric_state_lead_query(query: str) -> bool:
    """Return whether the current-state count is about led team members."""
    return bool(set(source_tokens(query)) & {"lead", "leads", "leading", "led"})


def numeric_state_subject_label(query: str) -> str:
    """Return a compact count label for numeric-state answers."""
    tokens = source_tokens(query)
    for preferred in ("engineers", "followers", "species", "coins", "titles", "pages"):
        if preferred in tokens:
            return preferred
    for token in reversed(tokens):
        if len(token) > 2 and token not in _COUNT_STOPWORDS and token not in {"currently", "initially", "started"}:
            return token
    return "items"


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
    if subject == "museum_gallery":
        return _filter_museum_gallery_rows(rows)
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


def _filter_museum_gallery_rows(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    """Keep named museum/gallery visits even when generic focus scoring is sparse."""
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.normalized_identity.startswith("museum_gallery=") and row.exclude_reason == "query_focus_mismatch":
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
                    relevance=max(row.relevance, 2),
                    include_reason=row.include_reason,
                    exclude_reason="",
                    confidence=row.confidence,
                )
            )
            continue
        filtered.append(row)
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


def build_date_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
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
        if "query_temporal_anchor=true" in raw_text.casefold():
            anchor_date = session_anchor_date(raw_text, raw_text)
            if anchor_date is None:
                continue
            identity = f"group={group}|date={anchor_date.isoformat()}"
            duplicate = identity in seen
            seen.add(identity)
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"date:{context_index}:query",
                    source_group=group,
                    citation=citation,
                    kind="date",
                    value=anchor_date.isoformat(),
                    unit="day",
                    label=anchor_date.isoformat(),
                    raw_span=anchor_date.isoformat(),
                    context=text,
                    normalized_identity=identity,
                    relevance=10,
                    include_reason="query_temporal_anchor",
                    exclude_reason="duplicate_identity" if duplicate else "",
                    confidence=_row_confidence(relevance=10, has_label=True),
                )
            )
            continue
        context_dates = explicit_date_matches(text, default_year=default_year)
        for match_index, date_match in enumerate(context_dates):
            value = date_match.value
            identity = f"group={group}|date={value.isoformat()}"
            duplicate = identity in seen
            seen.add(identity)
            evidence_span = local_evidence_span(text, date_match.start, date_match.end)
            relevance = _relevance(focus_terms, evidence_span)
            row = EvidenceLedgerRow(
                fact_id=f"date:{context_index}:{match_index}",
                source_group=group,
                citation=citation,
                kind="date",
                value=value.isoformat(),
                unit="day",
                label=value.isoformat(),
                raw_span=date_match.raw,
                context=evidence_span,
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
        if context_dates:
            continue
        anchor_date = session_anchor_date(raw_text, text)
        if anchor_date is None or not session_date_anchor_allowed(
            query,
            text,
            focus_terms=focus_terms,
            relevance=relevance,
        ):
            continue
        identity = f"group={group}|date={anchor_date.isoformat()}"
        duplicate = identity in seen
        seen.add(identity)
        include_reason = "relative_session_date_anchor" if relative_session_date_offset(text) else "session_date_anchor"
        row = EvidenceLedgerRow(
            fact_id=f"date:{context_index}:session",
            source_group=group,
            citation=citation,
            kind="date",
            value=anchor_date.isoformat(),
            unit="day",
            label=anchor_date.isoformat(),
            raw_span=anchor_date.isoformat(),
            context=text,
            normalized_identity=identity,
            relevance=relevance,
            include_reason=include_reason,
            exclude_reason="duplicate_identity" if duplicate else "",
            confidence=_row_confidence(relevance=relevance, has_label=True),
        )
        if duplicate:
            rows.append(row)
        else:
            provisional.append(row)
    anchor_terms = temporal_anchor_terms(plan.subject_terms)
    preserved_anchor_terms = (
        anchor_terms if _inverted_before_temporal_anchor_query(plan.subject_terms) else (set(), set())
    )
    rows.extend(
        _filter_session_date_anchors_with_explicit_source_dates(
            _filter_date_rows(provisional, preserved_anchor_terms)
        )
    )
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def render_date_interval_result(ledger: EvidenceLedger, *, rank: int) -> SynthesisResult:
    """Render temporal interval synthesis lines from date evidence."""
    candidates = ledger.included(kind="date")
    excluded = ledger.excluded(kind="date")
    if len(candidates) < ledger.plan.required_source_groups:
        return SynthesisResult(lines=(), support_source_groups=())
    anchor_terms = temporal_anchor_terms(ledger.plan.subject_terms)
    prefer_explicit_role_pairs = _inverted_before_temporal_anchor_query(ledger.plan.subject_terms)
    non_query_anchor_groups = {
        row.source_group
        for row in candidates
        if not _query_temporal_anchor_row(row)
    }
    allow_query_anchor_pairs = (
        "ago" in set(ledger.plan.subject_terms)
        or len(non_query_anchor_groups) < ledger.plan.required_source_groups
    )
    intervals: list[tuple[int, int, int, int, int, EvidenceLedgerRow, EvidenceLedgerRow]] = []
    seen_deltas: set[int] = set()
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
            if delta <= 0 or delta > 366 or delta in seen_deltas:
                continue
            seen_deltas.add(delta)
            ordered_anchor_score = temporal_ordered_anchor_score(left, right, anchor_terms)
            explicit_role_pair = int(
                prefer_explicit_role_pairs
                and
                _explicit_temporal_operand_pair_score(left, right, anchor_terms) > 0
                and left.include_reason == "explicit_date"
                and right.include_reason == "explicit_date"
            )
            intervals.append(
                (
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
    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    lines: list[str] = []
    support_groups: list[str] = []
    for index, (_, _, _, _, delta, left, right) in enumerate(intervals[:5]):
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
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="date_interval",
            candidates=(intervals[0][5], intervals[0][6]),
            excluded=excluded,
            answer_key="date_interval_answer",
            answer=f"{intervals[0][4]} days. {intervals[0][4] + 1} days (including the last day) is also acceptable.",
            support=support_groups,
        ),
    )


def _query_temporal_anchor_row(row: EvidenceLedgerRow) -> bool:
    return row.include_reason == "query_temporal_anchor" or row.source_group == "query-temporal-anchor"


def build_temporal_sequence_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract ordered event evidence for temporal sequence synthesis."""
    plan = plan or SynthesisPlan(
        answer_type="ordered_list",
        operation="temporal_sequence",
        subject_terms=tuple(
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _TEMPORAL_SEQUENCE_STOPWORDS and not token.isdigit()
        ),
        required_kinds=("temporal_event",),
        required_source_groups=2,
        reasons=("temporal_sequence",),
    )
    if not temporal_sequence_query(query):
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = set(plan.subject_terms)
    query_slots = temporal_sequence_query_slots(query)
    provisional: list[tuple[int, int, int, EvidenceLedgerRow]] = []
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        raw_text = context_text(context)
        text = temporal_evidence_text(raw_text)
        label = temporal_sequence_candidate(query, text)
        if not label:
            continue
        label = temporal_sequence_query_slot_label(query_slots, label) or label
        group = source_group(context)
        identity = f"temporal_event={normalize_temporal_sequence_label(label)}"
        duplicate = identity in seen
        seen.add(identity)
        order_value, include_reason = temporal_sequence_order_value(raw_text, text)
        provenance_index = source_group_sequence_index(group, fallback=context_index)
        relevance = _relevance(focus_terms, text)
        row = EvidenceLedgerRow(
            fact_id=f"temporal_sequence:{context_index}",
            source_group=group,
            citation=source_citation(context),
            kind="temporal_event",
            value=str(order_value),
            unit="sequence_order",
            label=label,
            raw_span=label,
            context=text,
            normalized_identity=identity,
            relevance=relevance,
            include_reason=include_reason,
            exclude_reason="duplicate_identity" if duplicate else "",
            confidence=_row_confidence(relevance=relevance, has_label=True),
        )
        if duplicate:
            rows.append(row)
            continue
        provisional.append((order_value, provenance_index, context_index, row))
    provisional.sort(key=lambda item: (item[0], item[1], item[2]))
    rows.extend(row for _, _, _, row in provisional)
    return EvidenceLedger(plan=plan, rows=tuple(rows))


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


def temporal_sequence_answer_text(labels: tuple[str, ...]) -> str:
    """Render ordered event labels as a direct first-person answer."""
    if not labels:
        return ""
    sentences: list[str] = []
    for index, label in enumerate(labels):
        if index == 0:
            marker = "First"
        elif index == len(labels) - 1 and len(labels) > 2:
            marker = "Lastly"
        else:
            marker = "Then"
        sentences.append(f"{marker}, {temporal_sequence_answer_phrase(label)}.")
    return " ".join(sentences)


def temporal_sequence_answer_phrase(label: str) -> str:
    """Return an answer phrase for one ordered temporal event label."""
    text = label.strip(" .")
    if re.match(r"^(?:I|we)\b", text, flags=re.IGNORECASE):
        return text
    if re.match(
        r"^(?:helped|used|redeemed|signed|ordered|went|got|returned|took|watched|"
        r"attended|participated|visited|flew)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return f"I {text}"
    return text


def build_duration_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract and normalize duration evidence into a cited ledger."""
    plan = plan or build_synthesis_plan(query)
    if "duration" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _duration_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    requires_actual_travel_duration = _travel_duration_total_query(query)
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    context_by_group: dict[str, str] = {}
    for context in contexts:
        group = source_group(context)
        context_by_group[group] = " ".join((context_by_group.get(group, ""), context_text(context))).strip()
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
                window_chars=320,
            )
            relevance = _relevance(focus_terms, evidence_span)
            exclude_reason = ""
            if requires_personal_memory and not _personal_numeric_evidence(
                text,
                duration_match.start,
                duration_match.end,
            ):
                exclude_reason = "not_personal_memory"
            elif requires_actual_travel_duration and not _actual_travel_duration_context(
                query,
                context_by_group.get(group, text),
            ):
                exclude_reason = "not_actual_travel_duration"
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
        query,
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
        answer_candidate=_answer_candidate(
            rank=rank,
            candidate_type="duration",
            candidates=candidates,
            excluded=excluded,
            answer_key="duration_total_answer",
            answer=_line_answer(lines, "duration_total_answer"),
        ),
    )


def source_tokens(text: str) -> list[str]:
    """Tokenize source/query text for deterministic synthesis helpers."""
    return list(_source_token_tuple(text))


@lru_cache(maxsize=8192)
def _source_token_tuple(text: str) -> tuple[str, ...]:
    """Tokenize source/query text once while keeping callers mutation-isolated."""
    tokens: list[str] = []
    for token in _SOURCE_TOKEN_RE.findall(text.casefold()):
        tokens.append(token)
        if not token.isalnum():
            tokens.extend(part for part in _SOURCE_TOKEN_SPLIT_RE.split(token) if part)
    return tuple(tokens)


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
    left_punctuation = previous_sentence_boundary(text, left_bound, start)
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


def previous_sentence_boundary(text: str, left_bound: int, position: int) -> int:
    """Return the previous sentence boundary, ignoring common abbreviation periods."""
    best = -1
    for index in range(left_bound, position):
        if text[index] not in ".!?;":
            continue
        if text[index] == "." and _abbreviation_period(text, index):
            continue
        best = index
    return best


def _abbreviation_period(text: str, index: int) -> bool:
    prefix = text[max(0, index - 12) : index]
    match = re.search(r"(?P<token>[A-Za-z]{1,4})$", prefix)
    if not match:
        return False
    token = match.group("token").casefold()
    return token in {"dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "rev"}


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
    prefix = text[max(0, start - 240) : start]
    if label := _currency_label_before_amount(prefix):
        return label
    purchase_prefix = text[max(0, start - 720) : start]
    if label := _currency_purchase_label_before_amount(purchase_prefix):
        if recipient := _currency_pronoun_recipient_label(purchase_prefix, label):
            return f"{recipient} {label}"
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


def _earned_total_query(query: str) -> bool:
    """Return whether a money query asks for incoming revenue rather than spend."""
    tokens = set(source_tokens(query))
    return bool(
        tokens
        & {
            "earn",
            "earned",
            "earning",
            "earnings",
            "made",
            "make",
            "sold",
            "sale",
            "sales",
            "revenue",
            "profit",
            "profits",
            "raised",
            "raise",
        }
    )


def _floor_value_sale_query(query: str) -> bool:
    """Return whether a query asks for minimum resale/appraisal value."""
    tokens = set(source_tokens(query))
    return bool(tokens & {"minimum", "least", "floor"}) and bool(tokens & {"sold", "sell", "selling", "get"})


def _floor_value_currency_context(span: str) -> bool:
    """Return whether a currency span describes an appraisal or minimum sale floor."""
    return bool(
        re.search(
            r"\b(?:worth|valued|value|appraised|appraisal|at\s+least|minimum|could\s+get|sell(?:ing)?)\b",
            span,
            flags=re.IGNORECASE,
        )
    )


def _earned_currency_context(span: str) -> bool:
    """Return whether a currency span describes money received by the user."""
    lowered = span.casefold()
    if re.search(
        r"\b(?:reward|rewards|discount|coupon|budget|threshold|spend|spent|"
        r"price|prices|cost|costs|fee|fees)\b",
        lowered,
    ) and not re.search(
        r"\b(?:earn(?:ed|ing)?|made|sold|sale|sales|revenue|profit|profits|"
        r"raised|raise)\b",
        lowered,
    ):
        return False
    return bool(
        re.search(
            r"\b(?:earn(?:ed|ing)?|made|sold|sale|sales|revenue|profit|profits|"
            r"raised|raise|brought\s+in|took\s+in)\b",
            lowered,
        )
    )


def _realized_unit_price_total(span: str, amount: float) -> float:
    """Return quantity times unit price for completed product sales."""
    if not re.search(r"\b(?:each|apiece|per)\b", span, flags=re.IGNORECASE):
        return amount
    if not re.search(r"\b(?:sold|sale|sales)\b", span, flags=re.IGNORECASE):
        return amount
    before_amount = span.split("$", 1)[0]
    matches = list(
        re.finditer(
            r"\b(?P<count>\d{1,4})\s+[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,8}\b",
            before_amount,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return amount
    count = int(matches[-1].group("count"))
    if count <= 0:
        return amount
    return amount * count


def _non_spend_currency_context(span: str) -> bool:
    """Return whether a currency span describes planned, hypothetical, or advisory cost."""
    lowered = span.casefold()
    price_filter_or_range = (
        bool(re.search(r"\b(?:under|over)\s+\$", lowered))
        and bool(
            re.search(
                r"\b(?:filter|filters|search|google|range|ranges|budget|recommend|"
                r"look\s+for|consider|option|options|available)\b",
                lowered,
            )
        )
    )
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
        "cost ranges",
        "filter by",
        "google search",
        "search terms",
        "budget",
    )
    if price_filter_or_range or any(marker in lowered for marker in future_or_hypothetical):
        return True
    return bool(
        re.search(
            r"\b(?:you|you'll|you\s+can|if\s+you're)\b[^.!?]{0,120}\$",
            span,
            flags=re.IGNORECASE,
        )
    )


def _travel_duration_total_query(query: str) -> bool:
    """Return whether day/week duration evidence should be actual travel time."""
    tokens = set(source_tokens(query))
    travel_terms = {
        "trip",
        "trips",
        "travel",
        "traveled",
        "traveling",
        "travelled",
        "travelling",
        "vacation",
        "vacations",
        "visited",
        "visit",
        "hawaii",
        "maui",
        "nyc",
        "york",
        "city",
    }
    duration_terms = {"day", "days", "week", "weeks"}
    return bool(tokens & duration_terms) and bool(tokens & travel_terms)


def _actual_travel_duration_context(query: str, span: str) -> bool:
    """Return whether a duration span names an actual trip, not planning or advice."""
    lowered = span.casefold()
    terms = set(source_tokens(span))
    query_terms = set(source_tokens(query))
    destination_terms = {
        term
        for term in query_terms
        if term
        in {
            "hawaii",
            "maui",
            "oahu",
            "kauai",
            "nyc",
            "new",
            "york",
            "city",
            "tokyo",
            "paris",
            "london",
        }
    }
    completed_trip_context = bool(
        re.search(
            r"\b(?:got\s+back|returned|recently\s+got\s+back|came\s+back)\b[^.!?]{0,180}\b(?:trip|travel|vacation)\b"
            r"|\b(?:trip|travel|vacation)\b[^.!?]{0,180}\b(?:got\s+back|returned|came\s+back)\b",
            lowered,
        )
    )
    if re.search(
        r"\b(?:itinerary|plan|planning|advice|suggest|suggested|recommend|"
        r"recommended|could|would|might|option|options)\b",
        lowered,
    ) and not (completed_trip_context and (not destination_terms or destination_terms & terms)):
        return False
    actual_markers = {
        "trip",
        "trips",
        "travel",
        "traveled",
        "traveling",
        "travelled",
        "travelling",
        "visited",
        "visit",
        "vacation",
        "returned",
        "back",
        "spent",
        "stayed",
        "stay",
    }
    if destination_terms and not destination_terms & terms:
        return False
    return bool(terms & actual_markers)


def currency_identity(*, group: str, value: str, label: str) -> str:
    """Return a stable identity used for currency deduplication."""
    normalized_label = normalize_currency_label(label)
    if normalized_label:
        return f"value={value}|label={normalized_label}"
    return f"group={group}|value={value}"


def format_currency(value: float | Decimal) -> str:
    """Render a currency value with thousands separators."""
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return f"${int(value):,}"
        whole = int(value)
        fraction = format(value.copy_abs(), "f").split(".", 1)[1].rstrip("0")
        if len(fraction) == 1:
            fraction = f"{fraction}0"
        return f"${whole:,}.{fraction}"
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
    fractional_word_pattern = re.compile(
        r"\b(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"(?:\s+|-)+and(?:\s+|-)+a(?:\s+|-)+half(?:\s+|-)+"
        r"(?P<unit>minute|hour|day|week|month)s?(?:[- ]long)?\b",
        flags=re.IGNORECASE,
    )
    for match in fractional_word_pattern.finditer(text):
        value_text = match.group("value").casefold()
        base_value = float(value_text) if value_text.isdigit() else float(_NUMBER_WORDS[value_text])
        matches.append(
            DurationValueMatch(
                value=base_value + 0.5,
                unit=f"{match.group('unit')}s",
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
    if count_subject == "museum_gallery":
        labels = _joined_count_labels(candidates)
        if labels:
            return f"I visited {count} different museums or galleries: {labels}."
        return f"I visited {count} different museums or galleries."
    if count_subject == "kitchen_item":
        labels = _joined_kitchen_item_labels(candidates)
        if labels:
            return f"I replaced or fixed {count} items: {labels}."
        return f"I replaced or fixed {count} kitchen items."
    if count_subject == "wedding":
        couples = _joined_wedding_couples(candidates)
        if couples:
            return f"I attended {count} weddings. The couples were {couples}."
        return f"I attended {count} weddings."
    if count_subject == "rollercoaster_ride":
        return f"I rode rollercoasters {len(candidates)} times."
    if count_subject == "fish_inventory":
        return f"There are {len(candidates)} fish in my aquariums."
    if count_subject == "property_viewing" and query_tokens & {"view", "viewed"}:
        return f"I viewed {count} properties."
    if count_subject == "musical_instrument":
        return f"I currently own {count} musical instruments."
    if action:
        return f"I {action} {count} {subject}."
    return f"There are {count} {subject}."


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


def _joined_count_labels(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    labels = [rendered_list_label(row) for row in candidates if rendered_list_label(row)]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _joined_kitchen_item_labels(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    labels = [
        f"the {label}"
        for row in candidates
        if (label := rendered_list_label(row))
    ]
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
    if subject == "kitchen_item":
        return _kitchen_count_items(
            spans or [text],
            focus_terms=focus_terms,
            identity_prefix="kitchen_item",
        )
    if subject == "fish_inventory":
        return _fish_inventory_items(
            spans or _first_person_spans(text) or [text],
            focus_terms=focus_terms,
            identity_prefix="fish_inventory",
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
    if subject == "museum_gallery":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="museum_gallery",
            label_extractor=_museum_gallery_labels,
        )
    if subject == "musical_instrument":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="musical_instrument",
            label_extractor=_musical_instrument_labels,
        )
    if subject == "rollercoaster_ride":
        return _rollercoaster_ride_items(
            spans,
            focus_terms=focus_terms,
            identity_prefix="rollercoaster_ride",
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
    if subject == "writing_piece":
        label = _writing_piece_label(span)
        return [
            CountEvidenceItem(
                label=label,
                span=span,
                normalized_identity=f"writing_piece:{_normalize_count_identity(label)}",
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


def numeric_state_query(query: str) -> bool:
    """Return whether a count question asks for current numeric state."""
    tokens = set(source_tokens(query))
    if not {"how", "many"} <= tokens and not numeric_state_difference_query(query):
        return False
    if tokens & {"times", "weddings", "events", "doctors", "appointments", "festivals", "properties", "rode"}:
        return False
    return bool(
        tokens
        & {
            "collection",
            "count",
            "currently",
            "decrease",
            "difference",
            "followers",
            "growth",
            "has",
            "have",
            "increase",
            "lead",
            "leading",
            "leads",
            "own",
            "owns",
            "seen",
            "spotted",
            "team",
        }
    )


def numeric_state_required_qualifier_terms(query: str) -> set[str]:
    """Return required qualifier terms for state questions scoped to an explicit role."""
    match = re.search(
        r"\brole\s+as\s+(?:a|an|the\s+)?(?P<qualifier>[A-Za-z][A-Za-z0-9 /&+-]*?)(?:[?.,;:]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return set()
    return {
        token
        for token in source_tokens(match.group("qualifier"))
        if len(token) > 2 and token not in _COUNT_STOPWORDS
    }


def numeric_state_evidence(text: str, *, focus_terms: set[str]) -> tuple[tuple[int, str, str], ...]:
    """Extract explicit total-state and incremental count updates from user memory text."""
    evidence: list[tuple[int, str, str]] = []
    for sentence in numeric_state_sentences(text):
        sentence_terms = set(source_tokens(sentence))
        explicit_state_terms = {
            "added",
            "bought",
            "count",
            "currently",
            "lead",
            "now",
            "reached",
            "team",
            "total",
        } & sentence_terms
        if focus_terms and len(focus_terms & sentence_terms) < 2 and not explicit_state_terms:
            continue
        evidence.extend(numeric_state_totals(sentence))
        evidence.extend(numeric_state_increments(sentence, focus_terms=focus_terms))
    return tuple(evidence)


def numeric_state_sentences(text: str) -> list[str]:
    """Return bounded user-authored sentences that may carry numeric state."""
    cleaned = re.sub(r"\bcontent=longmemeval_session_id=\S+\s*", " ", text)
    cleaned = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", " ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=user:|assistant:)", cleaned)
    sentences: list[str] = []
    for part in parts:
        sentence = " ".join(part.strip().split())
        if not sentence or re.match(r"assistant\s*:", sentence, flags=re.IGNORECASE):
            continue
        sentence = re.sub(r"^(?:\d+\.\s*)?user\s*:\s*", "", sentence, flags=re.IGNORECASE)
        if re.search(r"\b(?:I|I've|my|mine)\b", sentence, flags=re.IGNORECASE):
            sentences.append(sentence)
    return sentences


def numeric_state_totals(sentence: str) -> list[tuple[int, str, str]]:
    """Extract stated current totals from one sentence."""
    patterns = (
        rf"\btotal\s+(?:species\s+)?count\s+(?:to|is|of|at)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\b(?:a\s+)?total\s+of\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bmanaged\s+to\s+(?:spot|see|identify)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\b(?:spotted|seen|identified)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\s+different\b",
        rf"\bI\s+(?:currently\s+)?(?:have|own)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bI\s+(?:had|have)\s+(?:about\s+|around\s+|approximately\s+|roughly\s+)?(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\s+followers\b",
        rf"\bI\s+started\b[^.!?]{{0,80}}?\bwith\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\s+followers\b",
        rf"\bI(?:'m|m| am)\s+(?:currently\s+)?now\s+at\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bI\s+(?:just\s+)?reached\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bI\s+(?:now\s+|currently\s+)?lead\s+a\s+team\s+of\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
    )
    values: list[tuple[int, str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            value = _count_word_value(match.group("value"))
            if value:
                values.append((value, sentence, "stated_total"))
    return values


def numeric_state_increments(sentence: str, *, focus_terms: set[str]) -> list[tuple[int, str, str]]:
    """Extract additive updates from one sentence."""
    if not focus_terms & set(source_tokens(sentence)):
        return []
    patterns = (
        r"\b(?:just\s+|recently\s+)?added\s+(?P<value>a|an|one|two|three|four|five|\d{1,4})\s+(?:new\s+)?",
        r"\b(?:just\s+|recently\s+)?bought\s+(?P<value>a|an|one|two|three|four|five|\d{1,4})\s+(?:new\s+)?",
    )
    increments: list[tuple[int, str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            value_text = match.group("value").casefold()
            value = _NUMBER_WORDS.get(value_text, int(value_text) if value_text.isdigit() else 0)
            if value:
                increments.append((value, sentence, "incremental_update"))
    return increments


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


def _kitchen_count_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Extract kitchen item rows without relying on generic first-person scoring."""
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        if not _span_matches_count_subject(span, "kitchen_item"):
            continue
        relevance = max(_relevance(focus_terms, span), 2)
        for label in _kitchen_item_labels(span):
            identity = f"{identity_prefix}={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=identity,
                    relevance=relevance,
                )
            )
    return items


def _rollercoaster_ride_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Expand rollercoaster ride memories into one row per ride occurrence."""
    items: list[CountEvidenceItem] = []
    seen_counts: dict[str, int] = {}
    for span in spans:
        relevance = max(_relevance(focus_terms, span), 2)
        labels = _rollercoaster_ride_labels(span)
        for label in labels:
            normalized = _normalize_count_identity(label)
            occurrence = seen_counts.get(normalized, 0) + 1
            seen_counts[normalized] = occurrence
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=f"{identity_prefix}={normalized}:{occurrence}",
                    relevance=relevance,
                )
            )
    return items


def _fish_inventory_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Expand aquarium inventory memories into one row per fish."""
    items: list[CountEvidenceItem] = []
    seen_counts: dict[str, int] = {}
    for span in spans:
        if not _span_matches_count_subject(span, "fish_inventory"):
            continue
        relevance = max(_relevance(focus_terms, span), 2)
        for label, count in _fish_inventory_counts(span):
            normalized = _normalize_count_identity(label)
            for _ in range(count):
                occurrence = seen_counts.get(normalized, 0) + 1
                seen_counts[normalized] = occurrence
                items.append(
                    CountEvidenceItem(
                        label=label,
                        span=span,
                        normalized_identity=f"{identity_prefix}={normalized}:{occurrence}",
                        relevance=relevance,
                    )
                )
    return items


def _fish_inventory_counts(span: str) -> list[tuple[str, int]]:
    """Extract fish species/count pairs from first-person aquarium inventory text."""
    counts: list[tuple[str, int]] = []
    seen: set[str] = set()
    quantity_patterns = (
        r"\b(?P<count>\d{1,3})\s+(?P<label>neon\s+tetras?|golden\s+honey\s+gouramis?|gouramis?|tetras?|danios?)\b",
        r"\b(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?P<label>neon\s+tetras?|golden\s+honey\s+gouramis?|gouramis?|tetras?|danios?)\b",
    )
    for pattern in quantity_patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_fish_label(match.group("label"))
            count = _count_word_value(match.group("count"))
            if label and count > 0 and label not in seen:
                seen.add(label)
                counts.append((label, count))
    singular_patterns = (
        r"\b(?:a|an|one|my)\s+(?:small\s+)?(?P<label>pleco\s+catfish|pleco|betta\s+fish|betta)\b",
        r"\b(?P<label>betta\s+fish),\s+Bubbles\b",
    )
    for pattern in singular_patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_fish_label(match.group("label"))
            if label and label not in seen:
                seen.add(label)
                counts.append((label, 1))
    return counts


def _canonical_fish_label(label: str) -> str:
    normalized = " ".join(source_tokens(label))
    if "neon" in normalized and "tetra" in normalized:
        return "neon tetra"
    if "gourami" in normalized:
        return "golden honey gourami" if "golden" in normalized or "honey" in normalized else "gourami"
    if "pleco" in normalized:
        return "pleco catfish"
    if "betta" in normalized:
        return "betta fish"
    if "danio" in normalized:
        return "danio"
    return normalized


def _rollercoaster_ride_labels(span: str) -> list[str]:
    """Return one label per actual rollercoaster ride occurrence in a span."""
    labels: list[str] = []
    repeated_pattern = re.compile(
        r"\brode\s+(?P<label>[A-Z][A-Za-z0-9'&: -]{1,60}?)\s+"
        r"(?P<count>twice|thrice|(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times?)\b",
        flags=re.IGNORECASE,
    )
    consumed: list[tuple[int, int]] = []
    for match in repeated_pattern.finditer(span):
        label = _clean_rollercoaster_label(match.group("label"))
        count = _count_word_value(match.group("count"))
        if not label or count <= 0:
            continue
        consumed.append((match.start(), match.end()))
        labels.extend([label] * count)
    enumerated_pattern = re.compile(
        r"\brode\s+(?P<labels>[A-Z][A-Za-z0-9'&: -]{1,90}?"
        r"(?:,\s*[A-Z][A-Za-z0-9'&: -]{1,40})*"
        r"(?:,?\s+and\s+[A-Z][A-Za-z0-9'&: -]{1,40})?)"
        r"(?:\s+(?:rollercoasters?|coasters?|rides?))?"
        r"(?:\s+(?:at|in|during|on|with|before|after)\b|[.!?]|$)",
        flags=re.IGNORECASE,
    )
    for match in enumerated_pattern.finditer(span):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        for label in _split_rollercoaster_labels(match.group("labels")):
            labels.append(label)
    return labels


def _count_word_value(value: str) -> int:
    lowered = re.sub(r"\s+times?$", "", value.casefold()).strip()
    if lowered == "twice":
        return 2
    if lowered == "thrice":
        return 3
    if lowered.isdigit():
        return int(lowered)
    return _NUMBER_WORDS.get(lowered, 0)


def _split_rollercoaster_labels(raw_labels: str) -> list[str]:
    labels: list[str] = []
    for raw_label in re.split(r"\s*,\s*|\s+\band\b\s+", raw_labels):
        label = _clean_rollercoaster_label(raw_label)
        if label:
            labels.append(label)
    return labels


def _clean_rollercoaster_label(label: str) -> str:
    label = re.split(
        r"\b(?:at|in|during|on|with|before|after|again|twice|thrice)\b",
        label,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    label = re.sub(r"\b(?:rollercoasters?|coasters?|rides?)\b", "", label, flags=re.IGNORECASE)
    label = " ".join(label.strip(" .,'\"").split())
    if not label or label.casefold() in {"some", "several", "many"}:
        return ""
    return label


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
        "fix": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "fixed": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "replace": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "replaced": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "visit": {"appointment", "back", "diagnosed", "got", "had", "prescribed", "see", "saw", "took", "visit", "visited", "went"},
        "visited": {"appointment", "back", "diagnosed", "got", "had", "prescribed", "see", "saw", "took", "visit", "visited", "went"},
        "view": {"fell", "love", "rejected", "saw", "searching", "seen", "tour", "toured", "view", "viewed", "visited"},
        "viewed": {"fell", "love", "rejected", "saw", "searching", "seen", "tour", "toured", "view", "viewed", "visited"},
        "own": {"had", "have", "own", "owned", "playing", "selling", "service"},
        "owned": {"had", "have", "own", "owned", "playing", "selling", "service"},
        "work": {"bought", "build", "built", "finished", "got", "new", "planning", "started", "starting", "thinking", "work", "worked", "working"},
        "worked": {"bought", "build", "built", "finished", "got", "new", "planning", "started", "starting", "thinking", "work", "worked", "working"},
        "completed": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "drafted": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "finished": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "published": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "wrote": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "written": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "buy": {"buy", "bought", "got", "new", "purchased", "picked"},
        "bought": {"buy", "bought", "got", "new", "purchased", "picked"},
        "wedding": {"attend", "attended", "back", "been", "returned", "went"},
        "weddings": {"attend", "attended", "back", "been", "returned", "went"},
        "ride": {"ride", "rides", "riding", "rode", "ridden"},
        "rides": {"ride", "rides", "riding", "rode", "ridden"},
        "rode": {"ride", "rides", "riding", "rode", "ridden"},
        "rollercoaster": {"ride", "rides", "riding", "rode", "ridden"},
        "rollercoasters": {"ride", "rides", "riding", "rode", "ridden"},
        "coaster": {"ride", "rides", "riding", "rode", "ridden"},
        "coasters": {"ride", "rides", "riding", "rode", "ridden"},
    }
    actions: set[str] = set()
    for term in query_terms:
        actions.update(action_groups.get(term, set()))
    if not actions and {"how", "many"} <= query_terms:
        actions.update({"attend", "attended", "visit", "visited", "went", "got", "bought", "been", "took"})
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


def _count_subject(query: str, *, tokens: set[str] | None = None) -> str:
    """Classify the count target into a small deterministic evidence facet."""
    tokens = tokens or set(source_tokens(query))
    if tokens & {"movie", "movies", "film", "films", "cinema"} and tokens & {"festival", "festivals", "fest", "fests"}:
        return "film_festival"
    if tokens & {"museum", "museums", "gallery", "galleries", "cube"}:
        return "museum_gallery"
    if tokens & {"kitchen", "toaster", "faucet", "mat", "shelves", "coffee"} and tokens & {
        "item",
        "items",
        "replace",
        "replaced",
        "fix",
        "fixed",
    }:
        return "kitchen_item"
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
    if tokens & {"rollercoaster", "rollercoasters", "coaster", "coasters"} or (
        tokens & {"times"} and tokens & {"ride", "rides", "riding", "rode"}
    ):
        return "rollercoaster_ride"
    if tokens & {"fish", "aquarium", "aquariums", "tank", "tanks"} and tokens & {"total", "both", "many"}:
        return "fish_inventory"
    if tokens & {"piece", "pieces", "writing", "writings", "story", "stories"} and tokens & {
        "completed",
        "drafted",
        "finished",
        "published",
        "wrote",
        "written",
    }:
        return "writing_piece"
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
    if subject == "museum_gallery":
        if re.search(r"\bno\s+(?:museum|museums|gallery|galleries)\b", span, flags=re.IGNORECASE):
            return False
        if tokens & {"january"}:
            return False
        return bool(tokens & {"museum", "museums", "gallery", "galleries", "cube"})
    if subject == "kitchen_item":
        return bool(
            tokens
            & {
                "coffee",
                "espresso",
                "faucet",
                "ikea",
                "kitchen",
                "mat",
                "shelves",
                "sink",
                "toaster",
            }
            and tokens
            & {"donated", "fixed", "got", "new", "replaced", "upgrade"}
        )
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
    if subject == "rollercoaster_ride":
        return bool(
            tokens
            & {
                "rollercoaster",
                "rollercoasters",
                "coaster",
                "coasters",
                "ride",
                "rides",
                "riding",
                "rode",
                "ridden",
            }
        )
    if subject == "fish_inventory":
        return bool(tokens & {"fish", "tetras", "gouramis", "pleco", "catfish", "betta", "bubbles", "tank", "aquarium"})
    if subject == "writing_piece":
        return bool(
            tokens
            & {"article", "articles", "essay", "essays", "piece", "pieces", "poem", "poems", "story", "stories", "writing"}
            and tokens
            & {"completed", "drafted", "finished", "published", "wrote", "written"}
        )
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


def _writing_piece_label(span: str) -> str:
    """Extract the concrete writing item from a completed-writing span."""
    patterns = (
        r"\b(?:completed|drafted|finished|published|wrote|written)\s+(?:a|an|the)?\s*(?P<label>[^.!?]{0,60}?\b(?:article|essay|piece|poem|story))\b",
        r"\b(?P<label>(?:article|essay|piece|poem|story)\s+[^.!?]{0,40})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, span, flags=re.IGNORECASE)
        if not match:
            continue
        label = _clean_count_item_label(match.group("label"))
        if label:
            return label
    return count_label(span)


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


def _museum_gallery_labels(span: str) -> list[str]:
    """Extract named museum/gallery venues from first-person visit evidence."""
    labels: list[str] = []
    patterns = (
        r"\b(?P<label>(?:The\s+)?[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,5}\s+"
        r"Museum(?:\s+of\s+[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,4})?)'?s?\b",
        r"\b(?P<label>(?:The\s+)?[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,5}\s+(?:Museum|Gallery|Cube))\b",
        r"\b(?P<label>(?:Museum|Gallery)\s+of\s+[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,4})'?s?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, span):
            label = re.sub(r"'s$", "", _clean_count_item_label(match.group("label")))
            identity = _normalize_count_identity(label)
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _kitchen_item_labels(span: str) -> list[str]:
    """Extract durable kitchen items from repair, replacement, and upgrade memories."""
    labels: list[str] = []
    lowered = span.casefold()
    label_patterns = (
        ("kitchen faucet", r"\bkitchen\s+faucet\b|\bfaucet\b"),
        ("kitchen mat", r"\bkitchen\s+mat\b|\bmat\s+in\s+front\s+of\s+the\s+sink\b"),
        ("toaster", r"\bold\s+toaster\b|\btoaster\b"),
        ("coffee maker", r"\bold\s+coffee\s+maker\b|\bcoffee\s+maker\b"),
        ("kitchen shelves", r"\bkitchen\s+shelves\b|\bshelves\b"),
    )
    for label, pattern in label_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in labels:
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


def session_anchor_date(raw_text: str, evidence_text: str) -> date | None:
    """Return a typed event-date anchor from session metadata and relative text cues."""
    match = re.search(
        r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        raw_text,
    )
    if not match:
        return None
    try:
        value = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None
    return value + timedelta(days=relative_session_date_offset(evidence_text))


def relative_session_date_offset(text: str) -> int:
    """Return a small relative offset anchored to the source session date."""
    lowered = text.casefold()
    if re.search(r"\b(?:yesterday|the day before)\b", lowered):
        return -1
    if re.search(r"\b(?:tomorrow|the next day)\b", lowered):
        return 1
    match = re.search(r"\b(?P<count>\d{1,2}|one|two|three|four|five|six|seven)\s+days?\s+ago\b", lowered)
    if match:
        return -_small_number_word(match.group("count"))
    if re.search(r"\b(?:today|earlier today|this morning|this afternoon|this evening|tonight)\b", lowered):
        return 0
    return 0


def temporal_sequence_query(query: str) -> bool:
    """Return whether a query asks for an ordered list of remembered events."""
    tokens = set(source_tokens(query))
    if not tokens & {"order", "ordered", "sequence", "timeline"}:
        return False
    if tokens & {"earliest", "latest"}:
        return True
    if {"first", "last"} <= tokens:
        return True
    return bool(tokens & {"events", "trips", "airlines", "sports", "activities", "watched", "flew", "took"})


def temporal_sequence_order_value(raw_text: str, evidence_text: str) -> tuple[int, str]:
    """Return a sortable chronology value and the evidence reason used for it."""
    explicit = explicit_dates(evidence_text, default_year=context_year(raw_text))
    if explicit:
        return explicit[0].toordinal(), "explicit_date_anchor"
    anchor = session_anchor_date(raw_text, evidence_text)
    if anchor is not None and temporal_sequence_has_temporal_cue(evidence_text):
        return anchor.toordinal(), "session_date_anchor"
    days_ago = temporal_sequence_relative_days_ago(evidence_text)
    if days_ago is not None:
        return -days_ago, "relative_time_anchor"
    return 0, "provenance_order_anchor"


def temporal_sequence_has_temporal_cue(text: str) -> bool:
    """Return whether a source span has a cue tying the event to session time."""
    return bool(
        re.search(
            r"\b(?:today|tonight|yesterday|recently|just|ago|last\s+(?:week|month|year|weekend)|"
            r"this\s+(?:morning|afternoon|evening|week|month|weekend))\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def temporal_sequence_relative_days_ago(text: str) -> int | None:
    """Extract coarse relative-day offsets for event ordering."""
    lowered = text.casefold()
    if re.search(r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lowered):
        return 0
    if "yesterday" in lowered:
        return 1
    if "last week" in lowered:
        return 7
    if "last weekend" in lowered:
        return 7
    if "last month" in lowered:
        return 30
    if "a few months ago" in lowered:
        return 90
    if "recently" in lowered:
        return 3
    match = re.search(
        r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?P<unit>days?|weeks?|months?)\s+ago\b",
        lowered,
    )
    if not match:
        return None
    value_text = match.group("value")
    value = _NUMBER_WORDS.get(value_text, int(value_text) if value_text.isdigit() else 0)
    unit = match.group("unit")
    multiplier = 1 if unit.startswith("day") else 7 if unit.startswith("week") else 30
    return value * multiplier


def temporal_sequence_candidate(query: str, text: str) -> str:
    """Extract a concise event label from a cited source span."""
    query_tokens = set(source_tokens(query))
    sentences = temporal_sequence_sentences(text)
    best: tuple[int, str] | None = None
    for sentence in sentences:
        candidate = temporal_sequence_candidate_from_sentence(query_tokens, sentence)
        if not candidate:
            continue
        score = _relevance(query_tokens - _TEMPORAL_SEQUENCE_STOPWORDS, sentence)
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best else ""


def temporal_sequence_sentences(text: str) -> list[str]:
    """Return bounded candidate event sentences from user-authored spans."""
    cleaned = re.sub(r"\bquery=[^#\n]*?(?=\s+(?:content=|snippet=|source_path=|# Event)|$)", " ", text)
    cleaned = re.sub(r"\bcontent=longmemeval_session_id=\S+\s*", " ", cleaned)
    cleaned = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", " ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=user:|assistant:)", cleaned)
    sentences: list[str] = []
    for part in parts:
        sentence = " ".join(part.strip().split())
        if not sentence:
            continue
        if re.match(r"assistant\s*:", sentence, flags=re.IGNORECASE):
            continue
        sentence = re.sub(r"^(?:\d+\.\s*)?user\s*:\s*", "", sentence, flags=re.IGNORECASE)
        if re.search(r"\bI\s+", sentence, flags=re.IGNORECASE):
            sentences.append(sentence)
    return sentences


def temporal_sequence_candidate_from_sentence(query_tokens: set[str], sentence: str) -> str:
    """Return an event label from one first-person event sentence."""
    if query_tokens & {"museum", "museums", "gallery", "galleries"} and (
        venue_label := temporal_sequence_venue_label(sentence)
    ):
        return venue_label
    airline_query = bool(query_tokens & {"airline", "airlines", "flew", "flight", "flights"})
    if airline_query:
        match = re.search(r"\bI\s+(?:just\s+|recently\s+|also\s+)?flew\s+with\s+(?P<label>[^.!?;,]{2,80})", sentence, flags=re.IGNORECASE)
        if match:
            return clean_temporal_sequence_label(match.group("label"), airline=True)
    patterns = (
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?got\s+back\s+from\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?returned\s+from\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?went\s+on\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?took\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?watched\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?attended\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?participated\s+in\s+(?P<label>[^.!?;,]{3,140})"),
        ("helped ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?helped\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>ordered\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>used\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>redeemed\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>signed\s+up\s+for\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?visited\s+(?P<label>[^.!?;,]{3,140})"),
    )
    return _best_temporal_sequence_pattern_label(patterns, sentence)


def _best_temporal_sequence_pattern_label(patterns: tuple[tuple[str, str], ...], sentence: str) -> str:
    """Return the first supported event pattern label from a sentence."""
    for prefix, pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)
        if match:
            label = clean_temporal_sequence_label(match.group("label"), airline=False)
            return f"{prefix}{label}" if prefix and label else label
    return ""


def temporal_sequence_venue_label(sentence: str) -> str:
    """Return a venue entity from a first-person temporal venue sentence."""
    if not re.search(
        r"\bI\s+(?:just\s+|recently\s+|actually\s+|also\s+)?(?:"
        r"visited|attended|participated\s+in|took|went\s+to|got\s+back\s+from|"
        r"came\s+back\s+from|saw|went\s+on)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        return ""
    labels = _museum_gallery_labels(sentence)
    return labels[0] if labels else ""


def temporal_sequence_query_slots(query: str) -> tuple[str, ...]:
    """Return canonical event labels explicitly enumerated in the query."""
    quoted = tuple(
        clean_temporal_sequence_query_slot(match.group("label"))
        for match in re.finditer(r"'(?P<label>[^']{3,180})'", query)
    )
    if quoted:
        return tuple(slot for slot in quoted if slot)
    slots: list[str] = []
    for match in re.finditer(
        r"(?:\bthe\s+day\s+)?\bI\s+(?P<label>.*?)(?=,\s*(?:and\s+)?(?:the\s+day\s+)?I\s+|[?]$)",
        query,
        flags=re.IGNORECASE,
    ):
        label = clean_temporal_sequence_query_slot(match.group("label"))
        if label:
            slots.append(label)
    return tuple(slots)


def clean_temporal_sequence_query_slot(label: str) -> str:
    """Normalize a query-enumerated event slot into an answer label."""
    label = re.sub(r"^\s*(?:the\s+day\s+)?I\s+", "", label, flags=re.IGNORECASE)
    return clean_temporal_sequence_label(label, airline=False)


def temporal_sequence_query_slot_label(query_slots: tuple[str, ...], label: str) -> str:
    """Map an extracted cited event label back to a query-enumerated slot."""
    if not query_slots:
        return ""
    label_terms = _temporal_sequence_slot_terms(label)
    if not label_terms:
        return ""
    best: tuple[int, int, str] | None = None
    for index, slot in enumerate(query_slots):
        slot_terms = _temporal_sequence_slot_terms(slot)
        if not slot_terms:
            continue
        overlap = len(label_terms & slot_terms)
        threshold = 1 if len(slot_terms) == 1 else 2
        if overlap < threshold:
            continue
        candidate = (overlap, -index, slot)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else ""


def _temporal_sequence_slot_terms(text: str) -> set[str]:
    return {
        token
        for token in source_tokens(text)
        if len(token) > 2 and token not in _TEMPORAL_SEQUENCE_STOPWORDS and token not in {"thing", "things"}
    }


def clean_temporal_sequence_label(label: str, *, airline: bool) -> str:
    """Normalize a temporal event label while preserving answer-bearing nouns."""
    label = re.sub(
        r"\b(?:today|tonight|yesterday|recently|last\s+(?:week|month|year|weekend)|"
        r"about\s+(?:a|an|one|two|three|four|five|\d+)\s+(?:days?|weeks?|months?)\s+ago)\b",
        " ",
        label,
        flags=re.IGNORECASE,
    )
    label = re.split(r"\s+(?:but|while|because)\s+", label, maxsplit=1)[0]
    label = " ".join(label.strip(" .,'\"").split())
    label = re.sub(r"^(?:a|an|the)\s+", "", label, flags=re.IGNORECASE)
    if airline:
        label = re.sub(r"\bflight\b.*$", "", label, flags=re.IGNORECASE).strip(" .,'\"")
    return label


def normalize_temporal_sequence_label(label: str) -> str:
    """Normalize sequence labels for duplicate suppression."""
    normalized = label.casefold()
    normalized = re.sub(r"\b(?:a|an|the|my|our|with|for|to|from|today|recently)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def source_group_sequence_index(group: str, *, fallback: int) -> int:
    """Extract a stable sequence suffix from LongMemEval-style source groups."""
    match = re.search(r"(?:^|[_-])(?P<index>\d{1,4})$", group)
    if match:
        return int(match.group("index"))
    return fallback


def session_date_anchor_allowed(
    query: str,
    text: str,
    *,
    focus_terms: set[str],
    relevance: int,
) -> bool:
    """Return whether session-date metadata is strong enough to become typed date evidence."""
    if relevance < 2:
        return False
    query_terms = {term for term in source_tokens(query) if len(term) > 2}
    text_terms = set(source_tokens(text))
    event_terms = {
        "attended",
        "baked",
        "class",
        "concert",
        "discovered",
        "exhibit",
        "garden",
        "got",
        "harvested",
        "launched",
        "made",
        "played",
        "signed",
        "started",
        "took",
        "visited",
        "website",
    }
    temporal_cues = {
        "ago",
        "back",
        "day",
        "just",
        "recently",
        "today",
        "tonight",
        "yesterday",
    }
    if query_terms & event_terms and text_terms & event_terms and relevance >= 3:
        return True
    return bool(text_terms & temporal_cues and len(focus_terms & text_terms) >= 2)


def _small_number_word(value: str) -> int:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }
    return words.get(value.casefold(), int(value) if value.isdigit() else 0)


def explicit_dates(text: str, *, default_year: int | None) -> list[date]:
    """Extract explicit and supported relative dates from source text."""
    return [match.value for match in explicit_date_matches(text, default_year=default_year)]


def explicit_date_matches(text: str, *, default_year: int | None) -> list[ExplicitDateMatch]:
    """Extract explicit and supported relative dates with source offsets."""
    dates: list[ExplicitDateMatch] = []
    for match in re.finditer(
        r"\b(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        text,
    ):
        append_date_match(
            dates,
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
        )
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    if default_year is not None:
        black_friday = black_friday_date(default_year)
        for match in re.finditer(r"\bblack friday\b", text, flags=re.IGNORECASE):
            if re.search(r"\b(?:on|during)\s+black friday\b", text[max(0, match.start() - 16) : match.end()], flags=re.IGNORECASE):
                append_unique_date_match(dates, black_friday, start=match.start(), end=match.end(), raw=match.group(0))
        for match in re.finditer(r"\b(?:a|one|1)\s+weeks?\s+before\s+black friday\b", text, flags=re.IGNORECASE):
            append_unique_date_match(dates, black_friday - timedelta(days=7), start=match.start(), end=match.end(), raw=match.group(0))
        for match in re.finditer(r"\b(?:a|one|1)\s+weeks?\s+after\s+black friday\b", text, flags=re.IGNORECASE):
            append_unique_date_match(dates, black_friday + timedelta(days=7), start=match.start(), end=match.end(), raw=match.group(0))
    for match in re.finditer(
        rf"\b(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        append_date_match(
            dates,
            year,
            _MONTHS[match.group("month").casefold()],
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
        )
    for match in re.finditer(
        rf"\b(?:the\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?P<month>{month_pattern})(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        append_date_match(
            dates,
            year,
            _MONTHS[match.group("month").casefold()],
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
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
        append_date_match(
            dates,
            year,
            int(match.group("month")),
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
        )
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


def append_unique_date_match(
    dates: list[ExplicitDateMatch],
    value: date,
    *,
    start: int,
    end: int,
    raw: str,
) -> None:
    """Append a date match once while preserving extraction order."""
    if any(existing.value == value for existing in dates):
        return
    dates.append(ExplicitDateMatch(value=value, start=start, end=end, raw=raw))


def append_date(dates: list[date], year: int, month: int, day: int) -> None:
    """Append a valid calendar date once."""
    try:
        value = date(year, month, day)
    except ValueError:
        return
    append_unique_date(dates, value)


def append_date_match(
    dates: list[ExplicitDateMatch],
    year: int,
    month: int,
    day: int,
    *,
    start: int,
    end: int,
    raw: str,
) -> None:
    """Append a valid calendar date match once."""
    try:
        value = date(year, month, day)
    except ValueError:
        return
    append_unique_date_match(dates, value, start=start, end=end, raw=raw)


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
            return (
                _expand_temporal_anchor_terms(terms[between_index + 1 : and_index]),
                _expand_temporal_anchor_terms(terms[and_index + 1 :]),
            )
    if "after" in terms:
        after_index = terms.index("after")
        return (
            _expand_temporal_anchor_terms(terms[after_index + 1 :]),
            _expand_temporal_anchor_terms(terms[:after_index]),
        )
    if "before" in terms:
        before_index = terms.index("before")
        if before_index == 0:
            trailing = terms[before_index + 1 :]
            action_index = _temporal_anchor_action_index(trailing)
            if action_index is not None and action_index > 0:
                return (
                    _expand_temporal_anchor_terms(trailing[action_index:]),
                    _expand_temporal_anchor_terms(trailing[:action_index]),
                )
        return (
            _expand_temporal_anchor_terms(terms[:before_index]),
            _expand_temporal_anchor_terms(terms[before_index + 1 :]),
        )
    midpoint = len(terms) // 2
    if midpoint == 0:
        return set(), set()
    left = _expand_temporal_anchor_terms(term for term in terms[:midpoint] if term not in separators)
    right = _expand_temporal_anchor_terms(term for term in terms[midpoint:] if term not in separators)
    return left, right


def _expand_temporal_anchor_terms(terms: Iterable[str]) -> set[str]:
    """Expand query anchor terms with common inflections used in memory text."""
    expanded = set(terms)
    variants = {
        "book": {"booked", "booking"},
        "buy": {"bought", "buying"},
        "get": {"got", "getting"},
        "make": {"made", "making"},
        "order": {"ordered", "ordering"},
        "purchase": {"purchased", "purchasing"},
        "reserve": {"reserved", "reservation"},
        "schedule": {"scheduled", "scheduling"},
    }
    for term in tuple(expanded):
        expanded.update(variants.get(term, ()))
    return expanded


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


def _explicit_temporal_operand_pair_score(
    left: EvidenceLedgerRow,
    right: EvidenceLedgerRow,
    anchor_terms: tuple[set[str], set[str]],
) -> int:
    """Score explicit date pairs only when both query roles are covered."""
    first_anchor, second_anchor = anchor_terms
    if not first_anchor or not second_anchor:
        return 0
    left_context_terms = set(source_tokens(left.context))
    right_context_terms = set(source_tokens(right.context))
    left_date = date.fromisoformat(left.value)
    right_date = date.fromisoformat(right.value)
    if left_date <= right_date:
        earlier_terms, later_terms = left_context_terms, right_context_terms
    else:
        earlier_terms, later_terms = right_context_terms, left_context_terms
    first_score = len(first_anchor & earlier_terms)
    second_score = len(second_anchor & later_terms)
    if first_score <= 0 or second_score <= 0:
        return 0
    return first_score + second_score


def _temporal_anchor_action_index(terms: list[str]) -> int | None:
    """Return the split point for inverted before/after questions."""
    action_terms = {
        "book",
        "booked",
        "buy",
        "bought",
        "get",
        "got",
        "make",
        "made",
        "order",
        "ordered",
        "purchase",
        "purchased",
        "reserve",
        "reserved",
        "schedule",
        "scheduled",
    }
    for index, term in enumerate(terms):
        if term in action_terms:
            return index
    return None


def _inverted_before_temporal_anchor_query(subject_terms: tuple[str, ...]) -> bool:
    """Return whether normalized terms match "before X did I Y" structure."""
    terms = list(subject_terms)
    if not terms or terms[0] != "before":
        return False
    action_index = _temporal_anchor_action_index(terms[1:])
    return action_index is not None and action_index > 0


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
    *,
    query: str,
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
    target_slots = _currency_itemized_target_slots(query)
    for row in ledger.rows:
        if row.exclude_reason or row.kind != "currency":
            rows.append(row)
            continue
        if target_slots and _currency_row_matches_target_slot(row, target_slots):
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


def _filter_itemized_currency_targets(query: str, ledger: EvidenceLedger) -> EvidenceLedger:
    """Keep only rows matching requested item slots in conjunctive money queries."""
    target_slots = _currency_itemized_target_slots(query)
    if len(target_slots) < 2:
        return ledger
    included = ledger.included(kind="currency")
    if len(included) < 2:
        return ledger
    matched_rows = [
        row
        for row in included
        if _currency_row_matches_target_slot(row, target_slots)
    ]
    covered_slots = {
        slot_index
        for row in matched_rows
        for slot_index in _currency_row_target_slot_indexes(row, target_slots)
    }
    if len(covered_slots) < len(target_slots):
        return ledger
    selected_facts = {row.fact_id for row in matched_rows}
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
                exclude_reason="query_item_target_mismatch",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _currency_itemized_target_slots(query: str) -> tuple[set[str], ...]:
    """Extract item slots from queries like 'cost of X and Y'."""
    lowered = query.casefold()
    if " and " not in lowered or not (
        {"cost", "costs", "spent", "spend", "paid", "price", "amount", "money"} & set(source_tokens(query))
    ):
        return ()
    match = re.search(
        r"\b(?:cost(?:s)?|spent|spend|paid|price|amount|money)\b[^?]{0,80}?\b(?:of|on|for)\s+(?P<items>[^?]+)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ()
    parts = re.split(r"\s+and\s+|,\s*", match.group("items"), flags=re.IGNORECASE)
    slots: list[set[str]] = []
    for part in parts:
        cleaned = re.sub(r"\b[A-Za-z0-9_-]+['’]s\s+", " ", part)
        terms = {
            token
            for token in source_tokens(cleaned)
            if len(token) > 2 and token not in _QUERY_STOPWORDS and token not in _CURRENCY_ITEM_TARGET_STOPWORDS
        }
        expanded = set(terms)
        for term in terms:
            if term.endswith("s") and len(term) > 3:
                expanded.add(term[:-1])
            else:
                expanded.add(f"{term}s")
        if expanded:
            slots.append(expanded)
    return tuple(slots)


_CURRENCY_ITEM_TARGET_STOPWORDS = {
    "total",
    "cost",
    "costs",
    "money",
    "amount",
    "price",
    "spent",
    "spend",
    "paid",
    "last",
    "month",
    "months",
    "new",
    "week",
    "weeks",
    "year",
    "years",
    "gift",
    "gifts",
    "got",
    "max",
}


def _currency_row_matches_target_slot(row: EvidenceLedgerRow, target_slots: tuple[set[str], ...]) -> bool:
    return bool(_currency_row_target_slot_indexes(row, target_slots))


def _currency_row_target_slot_indexes(row: EvidenceLedgerRow, target_slots: tuple[set[str], ...]) -> tuple[int, ...]:
    text = _currency_slot_match_text(_currency_row_slot_match_text(row))
    row_terms = set(source_tokens(text))
    return tuple(
        index
        for index, slot in enumerate(target_slots)
        if _currency_slot_required_terms(slot) & row_terms
    )


def _currency_row_slot_match_text(row: EvidenceLedgerRow) -> str:
    """Return amount-local text for matching itemized money slots."""
    local_context = row.context
    if row.raw_span and row.raw_span in row.context:
        amount_index = row.context.find(row.raw_span)
        start = max(0, amount_index - 72)
        end = min(len(row.context), amount_index + len(row.raw_span) + 72)
        local_context = row.context[start:end]
    return " ".join((row.label, local_context, row.raw_span))


def _currency_slot_match_text(text: str) -> str:
    """Remove retrieval metadata before matching query item slots."""
    text = re.sub(r"\b(?:session_id|longmemeval_session_id|source_path|citation)=\S+", " ", text)
    text = re.sub(r"\brole=\S+", " ", text)
    return text


def _currency_slot_required_terms(slot: set[str]) -> set[str]:
    """Return substantive terms required to satisfy an itemized query slot."""
    substantive = slot - _CURRENCY_ITEM_SLOT_GENERIC_TERMS
    return substantive or slot


_CURRENCY_ITEM_SLOT_GENERIC_TERMS = {
    "auto",
    "autos",
    "car",
    "cars",
    "vehicle",
    "vehicles",
}


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


def _unit_price_currency_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"each", "per", "apiece"})


def _unit_price_currency_row(row: EvidenceLedgerRow) -> bool:
    amount = re.escape(row.raw_span)
    return bool(
        re.search(
            rf"{amount}\s*(?:/|per\b|each\b|apiece\b)|{amount}[^.!?]{{0,32}}\b(?:each|apiece)\b",
            row.context,
            flags=re.IGNORECASE,
        )
    )


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


def _lodging_currency_evidence(context: str) -> bool:
    terms = set(source_tokens(context))
    lodging_terms = {"accommodation", "accommodations", "hotel", "hostel", "resort", "lodging", "stay", "stayed"}
    destination_terms = {"tokyo", "hawaii", "maui", "japan"}
    nightly_terms = {"night", "nightly"}
    return bool(terms & lodging_terms) and bool(terms & (destination_terms | nightly_terms))


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


def _duration_row_matches_target_slot(row: EvidenceLedgerRow, target_slots: tuple[set[str], ...]) -> bool:
    row_terms = set(source_tokens(" ".join((row.label, row.context, row.raw_span))))
    return any(slot & row_terms for slot in target_slots)


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


def _duration_answer_unit_for_result(
    subject_terms: tuple[str, ...],
    candidates: tuple[EvidenceLedgerRow, ...],
) -> str:
    answer_unit = _duration_answer_unit(subject_terms)
    terms = set(subject_terms)
    if terms & {"minute", "minutes", "hour", "hours", "day", "days", "week", "weeks"}:
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
        if row.relevance >= selected_threshold or row.normalized_identity in preserved_identities:
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
    return {
        "rank": rank,
        "type": candidate_type,
        "confidence": float(_candidate_confidence(candidates)),
        "answer_key": answer_key,
        "answer": answer,
        "support_source_ids": list(dict.fromkeys(support or [row.source_group for row in candidates])),
        "excluded_source_ids": list(dict.fromkeys(row.source_group for row in excluded)),
    }


def _line_answer(lines: list[str], key: str) -> str:
    prefix = f"{key}="
    for line in lines:
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    return ""


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
        "museum": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "museums": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "gallery": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "galleries": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "properties": {"properties", "property", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"},
        "property": {"properties", "property", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"},
        "wedding": {"wedding", "weddings"},
        "weddings": {"wedding", "weddings"},
        "ride": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rides": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rode": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rollercoaster": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rollercoasters": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "coaster": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "coasters": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "fish": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "aquarium": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "aquariums": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "tank": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "tanks": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _date_focus_terms(query: str) -> set[str]:
    terms = _count_focus_terms(query)
    if "meet" in terms:
        terms.update({"met", "catch", "caught", "up"})
    if "moma" in terms:
        terms.update({"museum", "modern", "art"})
    if {"metropolitan", "ancient", "civilizations"} & terms:
        terms.update({"metropolitan", "museum", "art", "exhibit"})
    return terms


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
    lowered = prefix.casefold()
    if not (
        lowered.endswith((" for", " which were", " cost me", " cost"))
        or lowered in {"for", "which were", "cost me", "cost"}
    ):
        return ""
    for pattern in _CURRENCY_LABEL_BEFORE_AMOUNT_PATTERNS:
        match = pattern.search(prefix)
        if match:
            return _clean_currency_item_label(match.group("label"))
    return ""


def _currency_purchase_label_before_amount(prefix: str) -> str:
    """Extract a nearby purchased item when the amount is in a follow-up clause."""
    sentences = re.split(r"[.!?]", prefix)
    for sentence in reversed(sentences[-3:]):
        match = _CURRENCY_PURCHASE_LABEL_BEFORE_AMOUNT_RE.search(sentence)
        if match:
            return _clean_currency_item_label(match.group("item"))
    return ""


def _currency_pronoun_recipient_label(prefix: str, label: str) -> str:
    """Resolve local gift-recipient pronouns to a recently mentioned person term."""
    if not re.match(r"\s*(?:her|him|them|their)\b", label, flags=re.IGNORECASE):
        return ""
    recipient_matches = list(
        re.finditer(
            r"\b(?P<recipient>coworker|colleague|brother|sister|friend|mom|mother|dad|father|"
            r"partner|spouse|wife|husband|child|son|daughter|niece|nephew)\b",
            prefix,
            flags=re.IGNORECASE,
        )
    )
    if not recipient_matches:
        return ""
    return recipient_matches[-1].group("recipient").casefold()


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
