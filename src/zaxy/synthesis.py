"""Structured synthesis planning and evidence-ledger operations."""

from __future__ import annotations

import re
from dataclasses import dataclass


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
