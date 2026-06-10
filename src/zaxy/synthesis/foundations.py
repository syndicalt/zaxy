"""Split from synthesis.py (mechanical decomposition)."""


from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from functools import lru_cache

from zaxy.evidence_program import (
    TemporalEvidenceProgramResult,
)

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
class SynthesisResult:
    """Rendered synthesis result plus support/exclusion provenance."""

    lines: tuple[str, ...]
    support_source_groups: tuple[str, ...]
    excluded_source_groups: tuple[str, ...] = ()
    answer_candidate: dict[str, object] | None = None


@dataclass(frozen=True)
class DurationValueMatch:
    """One duration value and its source span coordinates."""

    value: float
    unit: str
    raw: str
    start: int
    end: int


@dataclass(frozen=True)
class QuantityValueMatch:
    """One scalar unit quantity and its source span coordinates."""

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
    "got",
    "back",
    "latest",
    "last",
    "order",
    "ordered",
    "past",
    "returned",
    "started",
    "three",
    "timeline",
    "took",
    "went",
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


def _duration_measure_query(tokens: set[str]) -> bool:
    """Return whether duration units are the thing being counted."""
    return bool(tokens & {"hours", "hour", "minutes", "minute", "days", "day", "weeks", "week"}) and not bool(
        tokens & {"month", "months"}
    )


def _incidental_time_modifier_query(query: str) -> bool:
    """Return whether duration words describe when the event happened, not what to count."""
    return bool(
        re.search(
            r"\b(?:ago|last|previous|past|prior)\s+"
            r"(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)?"
            r"(?:minute|hour|day|week|month|year)s?\b"
            r"|\b(?:minute|hour|day|week|month|year)s?\s+ago\b",
            query,
            flags=re.IGNORECASE,
        )
    )


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


def month_only_date(text: str, *, default_year: int | None) -> date | None:
    """Return a coarse date for month-only event mentions when a year is known."""
    if default_year is None:
        return None
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    match = re.search(
        rf"\b(?:in|during|since|from|back\s+in)\s+(?P<month>{month_pattern})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return date(default_year, _MONTHS[match.group("month").casefold()], 1)


def explicit_date_match_is_calendar_operand(text: str, match: ExplicitDateMatch) -> bool:
    """Return whether an explicit date match is not a numeric setting or measurement."""
    if "/" not in match.raw:
        return True
    before = text[max(0, match.start - 8) : match.start]
    after = text[match.end : min(len(text), match.end + 8)]
    if after.startswith(("°", "%")):
        return False
    if re.match(r"^\s*(?:deg(?:ree)?s?|degrees?|mm|cm|in|inch|inches|psi|bar|lb|lbs|kg|ft|feet)\b", after, flags=re.IGNORECASE):
        return False
    return not re.search(r"\b(?:toe|camber|damping|ratio|setting|settings)\s*$", before, flags=re.IGNORECASE)


def temporal_sequence_named_graduation_answer(labels: tuple[str, ...]) -> str:
    """Render ordered named graduation events in a compact actor-order form."""
    names: list[str] = []
    for label in labels:
        match = re.match(r"^\s*(?P<name>[A-Z][A-Za-z'-]{1,40})\s+graduated\b", label)
        if not match:
            return ""
        names.append(match.group("name"))
    if len(names) == 2:
        return f"{names[0]} graduated first, followed by {names[1]}."
    if len(names) >= 3:
        return f"{names[0]} graduated first, followed by {', '.join(names[1:-1])} and then {names[-1]}."
    return f"{names[0]} graduated first." if names else ""


def temporal_sequence_sports_answer_phrase(label: str) -> str:
    """Render bare sports event labels as first-person action phrases."""
    normalized = label.casefold()
    if re.search(r"\b(?:playoffs?|game|match)\b", normalized):
        return f"I watched {label}"
    if re.search(r"\b(?:triathlon|run|race|marathon)\b", normalized):
        return f"I completed {label}"
    if re.search(r"\b(?:tournament)\b", normalized):
        return f"I participated in {label}"
    return ""


def temporal_sequence_answer_phrase(label: str) -> str:
    """Return an answer phrase for one ordered temporal event label."""
    text = label.strip(" .")
    if re.match(r"^(?:I|we)\b", text, flags=re.IGNORECASE):
        return text
    if sports_phrase := temporal_sequence_sports_answer_phrase(text):
        return sports_phrase
    if re.match(
        r"^(?:helped|used|redeemed|signed|ordered|went|got|returned|took|watched|"
        r"attended|completed|finished|participate|participated|visited|started|flew)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return f"I {text}"
    return text


def temporal_sequence_answer_text(labels: tuple[str, ...]) -> str:
    """Render ordered event labels as a direct first-person answer."""
    if not labels:
        return ""
    if graduation_answer := temporal_sequence_named_graduation_answer(labels):
        return graduation_answer
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


def temporal_sequence_first_person_phrase(label: str, context: str) -> str:
    """Return the first-person action phrase supported by a row's local evidence."""
    escaped = re.escape(label)
    action_patterns = (
        ("went on", rf"\bwent\s+on\s+(?P<article>a|an|the)?\s*{escaped}\b"),
        ("got back from", rf"\bgot\s+back\s+from\s+(?P<article>a|an|the)?\s*{escaped}\b"),
        ("returned from", rf"\breturned\s+from\s+(?P<article>a|an|the)?\s*{escaped}\b"),
        ("started", rf"\bstarted\s+(?P<article>a|an|the)?\s*{escaped}\b"),
    )
    for action, pattern in action_patterns:
        if match := re.search(pattern, context, flags=re.IGNORECASE):
            article = match.group("article") or ""
            article_prefix = f"{article.lower()} " if article else ""
            if action == "started":
                return f"I started {article_prefix}{label}"
            return f"I {action} {article_prefix}{label}"
    if (sports_phrase := temporal_sequence_sports_answer_phrase(label)) and re.search(
        re.escape(label),
        context,
        flags=re.IGNORECASE,
    ):
        return sports_phrase
    if re.search(rf"\bparticipate(?:d)?\s+in\s+(?:the\s+)?{escaped}\b", context, flags=re.IGNORECASE):
        return f"I participated in {label}"
    if re.search(rf"\bcompleted\s+(?:the\s+)?{escaped}\b", context, flags=re.IGNORECASE):
        return f"I completed {label}"
    if re.search(rf"\bfinished\b[^.!?;,]{{0,80}}\bat\s+(?:the\s+)?{escaped}\b", context, flags=re.IGNORECASE):
        return f"I finished {label}"
    return temporal_sequence_answer_phrase(label)


@lru_cache(maxsize=8192)
def _source_token_tuple(text: str) -> tuple[str, ...]:
    """Tokenize source/query text once while keeping callers mutation-isolated."""
    tokens: list[str] = []
    for token in _SOURCE_TOKEN_RE.findall(text.casefold()):
        tokens.append(token)
        if not token.isalnum():
            tokens.extend(part for part in _SOURCE_TOKEN_SPLIT_RE.split(token) if part)
    return tuple(tokens)


def source_tokens(text: str) -> list[str]:
    """Tokenize source/query text for deterministic synthesis helpers."""
    return list(_source_token_tuple(text))


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


def _temporal_count_constraint(query: str) -> tuple[str, set[str]] | None:
    match = re.search(
        r"\b(?P<direction>before|after)\s+(?:the|a|an)?\s*(?P<target>[^?]+?)\??$",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    target = re.sub(r"\b(?:event|events|meeting|meetup|conference|appointment)\b$", "", match.group("target"), flags=re.IGNORECASE)
    target_terms = {
        token
        for token in source_tokens(target)
        if len(token) > 2 and token not in _COUNT_STOPWORDS and not token.isdigit()
    }
    if not target_terms:
        return None
    return match.group("direction").casefold(), target_terms


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


def context_text(context: str) -> str:
    """Return source text once, excluding Eventloom JSON payload echoes."""
    text = " ".join(context.split())
    return text.split(' {"content":', 1)[0]


def _personal_memory_query(query: str) -> bool:
    """Return whether a query asks about the user's own remembered facts."""
    return bool({"i", "me", "my", "mine", "we", "our"} & set(source_tokens(query)))


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


def _abbreviation_period(text: str, index: int) -> bool:
    prefix = text[max(0, index - 12) : index]
    match = re.search(r"(?P<token>[A-Za-z]{1,4})$", prefix)
    if not match:
        return False
    token = match.group("token").casefold()
    return token in {"dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "rev"}


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


def _personal_numeric_evidence(text: str, start: int, end: int) -> bool:
    """Return whether a numeric span belongs to user memory rather than advice text."""
    role = _speaker_role_before(text, start)
    if role == "assistant":
        return False
    if role == "user":
        return True
    evidence = local_evidence_span(text, start, end, window_chars=160)
    return bool(_FIRST_PERSON_EVIDENCE_RE.search(evidence))


def source_group(context: str) -> str:
    """Return a stable source group from common citation/session metadata."""
    patterns = [
        r"\b[a-z0-9_.-]*session[_-]?id=(?P<value>[^\s]+)",
        r"\bsource[_-]?id=(?P<value>[^\s]+)",
        r"\b(?:source_path|path|file)=['\"]?(?P<value>[^\s'\"]+)",
        r"\bthread=['\"]?(?P<value>[^\s'\"]+)",
        r"eventloom://[^/]+/events/(?P<value>\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, context, flags=re.IGNORECASE)
        if match:
            return match.group("value").casefold()
    return context[:160].casefold()


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
    temporal_program: TemporalEvidenceProgramResult | None = None

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


def _age_average_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    indexed = list(enumerate(_age_value_evidence(contexts)))
    indexed.sort(
        key=lambda item: (
            _source_group_natural_key(source_group(item[1][0])),
            item[0],
        )
    )
    return [evidence for _index, evidence in indexed]


def _evidence_order(row: EvidenceLedgerRow) -> int:
    match = re.match(r"^[^:]+:(?P<index>\d+)(?::\d+)?$", row.fact_id)
    if match:
        return int(match.group("index"))
    return 10**9


def _ordered_source_groups(rows: tuple[EvidenceLedgerRow, ...]) -> tuple[str, ...]:
    """Return unique source groups in original evidence order when row ids expose it."""
    return tuple(
        dict.fromkeys(
            row.source_group
            for row in sorted(rows, key=lambda row: (_evidence_order(row), row.source_group))
        )
    )


def _count_result_order(row: EvidenceLedgerRow) -> tuple[int, int, str]:
    """Return a stable source-oriented order for unordered count/list answers."""
    match = re.search(r"(?:^|[-_/.:])(?P<ordinal>\d{1,8})$", row.source_group)
    if match:
        return (0, int(match.group("ordinal")), row.source_group)
    return (1, _evidence_order(row), row.source_group)


def _count_row_action(row: EvidenceLedgerRow) -> str:
    for token in source_tokens(row.raw_span):
        if token in {"attended", "joined", "started", "hosted", "visited", "bought", "purchased", "booked", "completed"}:
            return token
    return ""


def _temporal_count_target_row(
    dated_rows: list[tuple[EvidenceLedgerRow, date]],
    target_terms: set[str],
) -> tuple[EvidenceLedgerRow, date] | None:
    scored: list[tuple[int, int, EvidenceLedgerRow, date]] = []
    for row, value in dated_rows:
        row_terms = set(source_tokens(f"{row.label} {row.raw_span}"))
        score = len(target_terms & row_terms)
        if score <= 0:
            continue
        scored.append((score, row.relevance, row, value))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2], scored[0][3]


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


def _date_row_dedupe_sort_key(row: EvidenceLedgerRow) -> tuple[int, int, int, float, int]:
    return (
        1 if row.exclude_reason else 0,
        0 if row.include_reason == "explicit_date" else 1,
        -row.relevance,
        -row.confidence,
        len(row.context),
    )


def _dedupe_filtered_date_rows(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    """Keep the best surviving date row per source/date after relevance filters run."""
    grouped: dict[str, list[EvidenceLedgerRow]] = {}
    ordered_identities: list[str] = []
    for row in rows:
        identity = row.normalized_identity
        if identity not in grouped:
            grouped[identity] = []
            ordered_identities.append(identity)
        grouped[identity].append(row)

    deduped: list[EvidenceLedgerRow] = []
    for identity in ordered_identities:
        group = grouped[identity]
        if len(group) == 1:
            deduped.extend(group)
            continue
        ranked = sorted(group, key=_date_row_dedupe_sort_key)
        winner = ranked[0]
        deduped.append(winner)
        for row in ranked[1:]:
            if row.exclude_reason:
                deduped.append(row)
            else:
                deduped.append(replace(row, exclude_reason="duplicate_identity"))
    return deduped


def _query_temporal_anchor_row(row: EvidenceLedgerRow) -> bool:
    return row.include_reason == "query_temporal_anchor" or row.source_group == "query-temporal-anchor"


def temporal_sequence_exclude_unanchored_when_answerable(
    rows: list[EvidenceLedgerRow],
    *,
    required_events: int,
) -> list[EvidenceLedgerRow]:
    """Exclude undated sequence candidates when anchored evidence can answer the query."""
    answerable_reasons = {
        "explicit_date_anchor",
        "relative_session_date_anchor",
        "relative_time_anchor",
        "session_date_anchor",
    }
    included = [row for row in rows if not row.exclude_reason]
    anchored = [row for row in included if row.include_reason in answerable_reasons]
    if len(anchored) < max(2, required_events):
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if not row.exclude_reason and row.include_reason == "provenance_order_anchor":
            filtered.append(replace(row, exclude_reason="unanchored_temporal_candidate"))
        else:
            filtered.append(row)
    included_filtered = [row for row in filtered if not row.exclude_reason]
    excluded = [row for row in filtered if row.exclude_reason]
    included_filtered.sort(key=lambda row: int(row.value) if row.value.lstrip("-").isdigit() else 0)
    return [*included_filtered, *excluded]


def temporal_sequence_first_person_answer(rows: tuple[EvidenceLedgerRow, ...]) -> str:
    """Render an action-form sequence when every cited row exposes a local verb."""
    phrases: list[str] = []
    for row in rows:
        phrase = temporal_sequence_first_person_phrase(row.label, row.context)
        if not phrase:
            return ""
        phrases.append(phrase)
    return temporal_sequence_answer_text(tuple(phrases))


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
    unit_first_fractional_pattern = re.compile(
        r"\b(?:(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"(?:\s+|-)+)?"
        r"(?P<unit>minute|hour|day|week|month)s?"
        r"(?:\s+|-)+and(?:\s+|-)+a(?:\s+|-)+half\b",
        flags=re.IGNORECASE,
    )
    fractional_spans = {(match.start(), match.end()) for match in fractional_word_pattern.finditer(text)}
    for match in unit_first_fractional_pattern.finditer(text):
        if (match.start(), match.end()) in fractional_spans:
            continue
        value_text = (match.group("value") or "one").casefold()
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
        if any(start <= match.start() and match.end() <= end for start, end in fractional_spans):
            continue
        if any(match.start() == item.start and match.end() <= item.end for item in matches):
            continue
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


def quantity_total_query(query: str, tokens: set[str] | None = None) -> bool:
    """Return whether a query asks for a summed non-duration unit quantity."""
    tokens = tokens or set(source_tokens(query))
    if tokens & {"cost", "costs", "price", "prices", "money", "spent", "spend"}:
        return False
    if (
        tokens & {"hour", "hours", "minute", "minutes", "day", "days", "week", "weeks", "month", "months"}
        and not _incidental_time_modifier_query(query)
    ):
        return False
    return bool(
        tokens & {"total", "combined", "altogether", "sum"}
        or re.search(r"\btotal\s+(?:weight|distance|pages?)\b", query, flags=re.IGNORECASE)
    )


def quantity_query_units(query: str) -> set[str]:
    """Return generic quantity units requested by an aggregate query."""
    tokens = set(source_tokens(query))
    if not quantity_total_query(query, tokens):
        return set()
    units: set[str] = set()
    if tokens & {"pound", "pounds", "lb", "lbs", "weight", "feed", "feeds", "grain", "grains"}:
        units.add("pounds")
    if tokens & {"mile", "miles", "distance"}:
        units.add("miles")
    if tokens & {"page", "pages"}:
        units.add("pages")
    return units


def quantity_value_matches(text: str) -> tuple[QuantityValueMatch, ...]:
    """Extract simple scalar unit quantities with source positions."""
    matches: list[QuantityValueMatch] = []
    pattern = re.compile(
        r"\b(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\s*[- ](?P<unit>pounds?|lbs?|miles?|pages?)\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        matches.append(
            QuantityValueMatch(
                value=float(match.group("value").replace(",", "")),
                unit=match.group("unit"),
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )
    matches.sort(key=lambda item: item.start)
    return tuple(matches)


def canonical_quantity_unit(unit: str) -> str:
    """Return canonical display unit for generic quantities."""
    normalized = unit.casefold()
    if normalized in {"pound", "pounds", "lb", "lbs"}:
        return "pounds"
    if normalized in {"mile", "miles"}:
        return "miles"
    return "pages"


def quantity_unit_display(unit: str, value: float) -> str:
    """Return singular or plural display for a canonical quantity unit."""
    singular = {"pounds": "pound", "miles": "mile", "pages": "page"}[unit]
    if value == 1:
        return singular
    return unit


def quantity_identity_signature(evidence_span: str) -> str:
    """Return a projection-stable local signature for a generic quantity operand."""
    text = evidence_span
    role_markers = list(
        re.finditer(
            r"(?:^|\b)(?:\d+\.\s*)?(?:user|assistant)\s*:|\brole=(?:user|assistant)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if role_markers:
        text = text[role_markers[-1].end():]
    text = re.split(
        r"\s+(?:end_line|source_path|source_start_line|source_end_line|source_event_seq|"
        r"source_event_hash|path|sha256|start_line|turn_index|summary)=",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[-1]
    text = re.sub(r"\blongmemeval_[a-z_]+=[^\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsession_id=[^\s]+", " ", text, flags=re.IGNORECASE)
    signature_terms = [
        term
        for term in source_tokens(text)
        if term not in _QUERY_STOPWORDS
        and term not in {"pound", "pounds", "lb", "lbs", "mile", "miles", "page", "pages"}
    ]
    return " ".join(signature_terms[:20])


def quantity_identity(*, group: str, value: float, unit: str, evidence_span: str) -> str:
    """Return projection-stable identity for generic quantity rows."""
    return (
        f"group={group}|value={format_number(value)}|unit={unit}"
        f"|context={quantity_identity_signature(evidence_span)}"
    )


def quantity_match_is_rate_or_guideline(text: str, start: int, end: int) -> bool:
    """Return whether a quantity is a rate, range endpoint, or generic guideline."""
    before = text[max(0, start - 48):start].casefold()
    after = text[end:min(len(text), end + 80)].casefold()
    if re.search(r"[-–—]\s*$", before) or re.match(r"\s*[-–—]\s*\d", after):
        return True
    if re.match(r"\s*(?:per|a|each|/)\b", after):
        return True
    if re.match(r"\s+per\s+\w+\b", after):
        return True
    return bool(
        re.search(
            r"\b(?:recommend(?:ed|ation)?|guidelines?|provide|offer|feed\s+them|"
            r"give\s+each|serving|daily|per\s+day)\b",
            before + " " + after,
        )
    )


def _quantity_focus_terms(query: str) -> set[str]:
    """Return focus terms for generic unit-quantity aggregation."""
    terms = {
        token
        for token in source_tokens(query)
        if len(token) > 2
        and token not in _NUMERIC_FOCUS_STOPWORDS
        and token not in {"weight", "pound", "pounds", "total", "past", "months"}
        and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
        "feed": {"feed", "feeds", "layer", "scratch", "grain", "grains", "chicken", "chickens", "hens"},
        "feeds": {"feed", "feeds", "layer", "scratch", "grain", "grains", "chicken", "chickens", "hens"},
        "grain": {"feed", "feeds", "layer", "scratch", "grain", "grains", "chicken", "chickens", "hens"},
        "grains": {"feed", "feeds", "layer", "scratch", "grain", "grains", "chicken", "chickens", "hens"},
        "purchased": {"purchased", "bought", "got", "ordered"},
        "purchase": {"purchased", "bought", "got", "ordered"},
        "bought": {"purchased", "bought", "got", "ordered"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _duration_query_accepts_relative_time_anchor(query: str) -> bool:
    """Return whether relative-time duration spans are the requested answer."""
    normalized = " ".join(query.casefold().split())
    return bool(
        re.search(
            r"\bhow\s+(?:many|long)\s+"
            r"(?:minutes?|hours?|days?|weeks?|months?|years?)\s+ago\b",
            normalized,
        )
        or re.search(
            r"\bhow\s+long\s+ago\b",
            normalized,
        )
        or re.search(
            r"\bhow\s+(?:many|long)\s+"
            r"(?:minutes?|hours?|days?|weeks?|months?|years?)\s+(?:before|after)\b",
            normalized,
        )
    )


def _duration_match_is_relative_time_anchor(text: str, start: int, end: int) -> bool:
    """Return whether a duration span locates an event in time rather than measuring it."""
    before = text[max(0, start - 48):start].casefold()
    after = text[end:min(len(text), end + 48)].casefold()
    if re.match(r"\s*(?:ago|earlier|later)\b", after):
        return True
    if re.match(r"\s+(?:before|after)\s+(?:the\s+|a\s+|an\s+)?[a-z0-9]", after):
        return True
    if re.match(r"\s+in\s+advance\b", after):
        return True
    return bool(
        re.search(
            r"\b(?:last|previous|past|prior|next)\s+$"
            r"|\b(?:in|during|over|within)\s+(?:the\s+)?(?:last|previous|past|prior|next)\s+$",
            before,
        )
    )


def _duration_match_is_recurring_cadence(text: str, start: int, end: int) -> bool:
    """Return whether a duration span is a recurrence period rather than work done."""
    before = text[max(0, start - 96):start].casefold()
    after = text[end:min(len(text), end + 48)].casefold()
    clause_before = re.split(r"[.;!?]\s*", before)[-1]
    if re.search(
        r"\b(?:per|every|each)\s+$"
        r"|\b(?:once|twice|\d+\s+times?|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?:or\s+\w+\s+)?(?:times?|sessions?|classes?|practices?|workouts?)\s+$",
        clause_before,
    ):
        return True
    if re.search(
        r"\b(?:times?|sessions?|classes?|practices?|workouts?)\s+$",
        clause_before,
    ) and re.search(
        r"\b(?:once|twice|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        clause_before,
    ):
        return True
    return bool(re.match(r"\s*(?:cadence|schedule|routine|frequency)\b", after))


def _duration_match_is_habitual_per_occurrence(text: str, start: int, end: int) -> bool:
    """Return whether a duration measures one occurrence in a habitual routine."""
    before = text[max(0, start - 128):start].casefold()
    clause_before = re.split(r"[.;!?]\s*", before)[-1]
    return bool(
        re.search(r"\b(?:each|every)\s+time\s+(?:for\s+)?$", clause_before)
        and re.search(
            r"\b(?:used\s+to|usually|typically|normally|routine|habit|"
            r"times?\s+a\s+\w+|sessions?\s+a\s+\w+|classes?\s+a\s+\w+|"
            r"practices?\s+a\s+\w+|workouts?\s+a\s+\w+)\b",
            clause_before,
        )
    )


def duration_unit_minutes(unit: str) -> float:
    """Return the number of minutes represented by one canonical duration unit."""
    return {
        "minutes": 1,
        "hours": 60,
        "days": 60 * 24,
        "weeks": 60 * 24 * 7,
        "months": 60 * 24 * 28,
    }[unit]


def duration_identity(
    *,
    group: str,
    minutes: float,
    label: str,
    evidence_signature: str = "",
    occurrence_index: int = 0,
) -> str:
    """Return a stable identity used for duration deduplication."""
    return (
        f"group={group}|minutes={format_number(minutes)}|label={label.casefold()}"
        f"|occurrence={occurrence_index}|context={evidence_signature}"
    )
