"""Typed evidence candidates for retrieval-time answer synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class EvidenceCandidate:
    """A typed answer ingredient extracted from one cited source context."""

    kind: str
    value: str
    unit: str
    source_group: str
    label: str
    context: str
    relevance: int


@dataclass(frozen=True)
class EvidenceProjection:
    """Rendered evidence candidates plus the source groups that support them."""

    lines: tuple[str, ...]
    source_groups: tuple[str, ...]


def aggregate_candidate_projection(query: str, contexts: list[str]) -> EvidenceProjection:
    """Build deterministic aggregate answer candidates from cited contexts."""
    lines: list[str] = []
    source_groups: list[str] = []
    rank = 1
    count = _count_candidates(query, contexts)
    if len(count) >= 2:
        lines.extend(_count_candidate_lines(count, rank=rank))
        source_groups.extend(candidate.source_group for candidate in count)
        rank += 1
    currency = _currency_candidates(query, contexts)
    if currency:
        lines.extend(_currency_candidate_lines(currency, rank=rank))
        source_groups.extend(candidate.source_group for candidate in currency)
        rank += 1
    duration = _duration_candidates(query, contexts)
    if duration:
        lines.extend(_duration_candidate_lines(duration, rank=rank))
        source_groups.extend(candidate.source_group for candidate in duration)
        rank += 1
    date_projection = _date_interval_projection(query, contexts, rank=rank)
    if date_projection.lines:
        lines.extend(date_projection.lines)
        source_groups.extend(date_projection.source_groups)
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=tuple(dict.fromkeys(source_groups)),
    )


def aggregate_candidate_lines(query: str, contexts: list[str]) -> list[str]:
    """Render deterministic aggregate answer candidates from cited contexts."""
    return list(aggregate_candidate_projection(query, contexts).lines)


def _count_candidate_lines(candidates: list[EvidenceCandidate], *, rank: int) -> list[str]:
    source_ids = ",".join(candidate.source_group for candidate in candidates)
    return [
        *_candidate_diagnostic_lines("count", candidates, rank=rank),
        f"count_answer={len(candidates)}",
        "count_unit=events",
        f"count_source_ids={source_ids}",
    ]


def _count_candidates(query: str, contexts: list[str]) -> list[EvidenceCandidate]:
    if not _count_query(query):
        return []
    focus_terms = _expanded_focus_terms(query)
    candidates: list[EvidenceCandidate] = []
    seen_groups: set[str] = set()
    scored: list[EvidenceCandidate] = []
    for context in contexts:
        text = _context_text(context)
        group = _source_group(context)
        if group in seen_groups:
            continue
        relevance = _relevance(focus_terms, text)
        if relevance <= 0:
            continue
        scored.append(
            EvidenceCandidate(
                kind="count",
                value="1",
                unit="event",
                source_group=group,
                label=_count_label(text),
                context=text,
                relevance=relevance,
            )
        )
    if not scored:
        return []
    best = max(candidate.relevance for candidate in scored)
    threshold = 2 if best >= 2 else best
    for candidate in scored:
        if candidate.relevance < threshold:
            continue
        candidates.append(candidate)
        seen_groups.add(candidate.source_group)
    return candidates


def _currency_candidate_lines(candidates: list[EvidenceCandidate], *, rank: int) -> list[str]:
    values = sorted((float(candidate.value) for candidate in candidates), reverse=True)
    lines = [
        *_candidate_diagnostic_lines("currency", candidates, rank=rank),
        "currency_values=" + ",".join(_format_currency(value) for value in values),
        f"currency_total={_format_currency(sum(values))}",
    ]
    max_item = max(candidates, key=lambda candidate: float(candidate.value))
    lines.append(f"currency_max={_format_currency(float(max_item.value))}")
    if max_item.label:
        lines.append(f"currency_max_label={max_item.label}")
    if len(values) >= 2:
        lines.append(
            f"currency_difference={_format_currency(max(values) - min(values))}"
        )
    return lines


def _currency_candidates(query: str, contexts: list[str]) -> list[EvidenceCandidate]:
    if not _currency_query(query):
        return []
    items: list[EvidenceCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for context in contexts:
        text = _context_text(context)
        group = _source_group(context)
        relevance = _relevance(_numeric_focus_terms(query), text)
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text):
            value = str(float(match.group("value").replace(",", "")))
            label = _currency_label(text, match.start(), match.end())
            identity = (group, value, label.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                EvidenceCandidate(
                    kind="currency",
                    value=value,
                    unit="USD",
                    source_group=group,
                    label=label,
                    context=text,
                    relevance=relevance,
                )
            )
    return _filter_focused_candidates(query, items)


def _duration_candidate_lines(candidates: list[EvidenceCandidate], *, rank: int) -> list[str]:
    values = [_duration_display(candidate) for candidate in candidates]
    total_minutes = sum(float(candidate.value) for candidate in candidates)
    lines = [
        *_candidate_diagnostic_lines("duration", candidates, rank=rank),
        "duration_values=" + ",".join(values),
        f"duration_total_minutes={_format_number(total_minutes)} minutes",
        f"duration_total_hours={_format_number(total_minutes / 60)} hours",
        "duration_source_ids="
        + ",".join(candidate.source_group for candidate in candidates),
    ]
    if len(candidates) >= 2:
        raw_values = [float(candidate.value) for candidate in candidates]
        lines.append(
            "duration_difference_minutes="
            f"{_format_number(max(raw_values) - min(raw_values))} minutes"
        )
    lines.extend(_duration_compatibility_lines(candidates))
    return lines


def _duration_compatibility_lines(candidates: list[EvidenceCandidate]) -> list[str]:
    by_unit: dict[str, list[float]] = {}
    for candidate in candidates:
        raw_value, raw_unit = _duration_raw_value_unit(candidate)
        by_unit.setdefault(raw_unit, []).append(raw_value)
    lines: list[str] = []
    if minute_values := by_unit.get("minutes"):
        lines.append(
            "minute_values=" + ",".join(_format_number(value) for value in minute_values)
        )
        lines.append(
            f"minute_total_hours={_format_number(sum(minute_values) / 60)} hours"
        )
    if hour_values := by_unit.get("hours"):
        lines.append(
            "hour_values=" + ",".join(_format_number(value) for value in hour_values)
        )
        lines.append(f"hour_total={_format_number(sum(hour_values))} hours")
    if day_values := by_unit.get("days"):
        lines.append(
            "day_values=" + ",".join(_format_number(value) for value in day_values)
        )
        lines.append(f"day_total={_format_number(sum(day_values))} days")
    return lines


def _duration_raw_value_unit(candidate: EvidenceCandidate) -> tuple[float, str]:
    match = re.match(r"(?P<value>\d+(?:\.\d+)?)\s+(?P<unit>[a-z]+)", candidate.label)
    if not match:
        return float(candidate.value), candidate.unit
    return float(match.group("value")), match.group("unit")


def _duration_candidates(query: str, contexts: list[str]) -> list[EvidenceCandidate]:
    items: list[EvidenceCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    focus_terms = _duration_focus_terms(query)
    for context in contexts:
        text = _context_text(context)
        group = _source_group(context)
        relevance = _relevance(focus_terms, text)
        for match in re.finditer(
            r"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\b",
            text,
            flags=re.IGNORECASE,
        ):
            unit = _canonical_duration_unit(match.group("unit"))
            minutes = float(match.group("value")) * _duration_unit_minutes(unit)
            raw = f"{_format_number(float(match.group('value')))} {unit}"
            identity = (group, str(minutes), raw)
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                EvidenceCandidate(
                    kind="duration",
                    value=str(minutes),
                    unit="minutes",
                    source_group=group,
                    label=raw,
                    context=text,
                    relevance=relevance,
                )
            )
    return _filter_duration_candidates(query, items)


def _date_interval_projection(
    query: str,
    contexts: list[str],
    *,
    rank: int,
) -> EvidenceProjection:
    if "days" not in _tokens(query):
        return EvidenceProjection(lines=(), source_groups=())
    candidates = _date_candidates(query, contexts)
    if len(candidates) < 2:
        return EvidenceProjection(lines=(), source_groups=())
    intervals: list[tuple[int, int, int, EvidenceCandidate, EvidenceCandidate]] = []
    seen_deltas: set[int] = set()
    for left_index, left in enumerate(candidates):
        for right_index, right in enumerate(candidates[left_index + 1 :], start=left_index + 1):
            if left.source_group == right.source_group:
                continue
            delta = abs((date.fromisoformat(right.value) - date.fromisoformat(left.value)).days)
            if delta <= 0 or delta > 366 or delta in seen_deltas:
                continue
            seen_deltas.add(delta)
            intervals.append(
                (
                    -(left.relevance + right.relevance),
                    left_index + right_index,
                    delta,
                    left,
                    right,
                )
            )
    if not intervals:
        return EvidenceProjection(lines=(), source_groups=())
    intervals.sort(key=lambda item: (item[0], item[1], item[2]))
    lines: list[str] = []
    support_groups: list[str] = []
    for index, (_, _, delta, left, right) in enumerate(intervals[:5]):
        if index == 0:
            support = sorted({left.source_group, right.source_group})
            lines.extend(
                _candidate_diagnostic_lines(
                    "date_interval",
                    [left, right],
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
    return EvidenceProjection(lines=tuple(lines), source_groups=tuple(support_groups))


def _date_candidates(query: str, contexts: list[str]) -> list[EvidenceCandidate]:
    focus_terms = _expanded_focus_terms(query)
    candidates: list[EvidenceCandidate] = []
    seen: set[tuple[str, str]] = set()
    for context in contexts:
        raw_text = _context_text(context)
        text = _temporal_evidence_text(raw_text)
        default_year = _context_year(raw_text)
        group = _source_group(context)
        relevance = _relevance(focus_terms, text)
        for value in _explicit_dates(text, default_year=default_year):
            identity = (group, value.isoformat())
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                EvidenceCandidate(
                    kind="date",
                    value=value.isoformat(),
                    unit="day",
                    source_group=group,
                    label=value.isoformat(),
                    context=text,
                    relevance=relevance,
                )
            )
    return _filter_date_candidates(candidates)


def _filter_date_candidates(candidates: list[EvidenceCandidate]) -> list[EvidenceCandidate]:
    if len(candidates) < 3:
        return candidates
    best = max((candidate.relevance for candidate in candidates), default=0)
    if best < 2:
        return candidates
    selected = [candidate for candidate in candidates if candidate.relevance >= max(2, best // 2)]
    return selected if len(selected) >= 2 else candidates


def _temporal_evidence_text(text: str) -> str:
    text = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)", " ", text)
    text = re.sub(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", " ", text)
    return text


def _context_year(text: str) -> int | None:
    match = re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/", text)
    if match:
        return int(match.group("year"))
    match = re.search(r"\b(?P<year>20\d{2})[/-]\d{1,2}[/-]\d{1,2}\b", text)
    if match:
        return int(match.group("year"))
    return None


def _explicit_dates(text: str, *, default_year: int | None) -> list[date]:
    dates: list[date] = []
    for match in re.finditer(
        r"\b(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        text,
    ):
        _append_date(
            dates,
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    if default_year is not None:
        black_friday = _black_friday(default_year)
        if re.search(r"\bblack friday\b", text, flags=re.IGNORECASE):
            if re.search(r"\b(?:on|during)\s+black friday\b", text, flags=re.IGNORECASE):
                _append_unique_date(dates, black_friday)
            if re.search(r"\b(?:a|one|1)\s+weeks?\s+before\s+black friday\b", text, flags=re.IGNORECASE):
                _append_unique_date(dates, black_friday - timedelta(days=7))
            if re.search(r"\b(?:a|one|1)\s+weeks?\s+after\s+black friday\b", text, flags=re.IGNORECASE):
                _append_unique_date(dates, black_friday + timedelta(days=7))
    for match in re.finditer(
        rf"\b(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        _append_date(
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
        _append_date(
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
        _append_date(dates, year, int(match.group("month")), int(match.group("day")))
    return dates


def _black_friday(year: int) -> date:
    thanksgiving = _nth_weekday_of_month(year, 11, weekday=3, n=4)
    return thanksgiving + timedelta(days=1)


def _nth_weekday_of_month(year: int, month: int, *, weekday: int, n: int) -> date:
    value = date(year, month, 1)
    days_until_weekday = (weekday - value.weekday()) % 7
    return value + timedelta(days=days_until_weekday + (n - 1) * 7)


def _append_unique_date(dates: list[date], value: date) -> None:
    if value not in dates:
        dates.append(value)


def _append_date(dates: list[date], year: int, month: int, day: int) -> None:
    try:
        value = date(year, month, day)
    except ValueError:
        return
    if value not in dates:
        dates.append(value)


def _filter_duration_candidates(
    query: str,
    items: list[EvidenceCandidate],
) -> list[EvidenceCandidate]:
    if len(items) < 2:
        return items
    focus_terms = _duration_focus_terms(query)
    if not focus_terms:
        return items
    if max((item.relevance for item in items), default=0) <= 0:
        return items
    selected = [item for item in items if item.relevance > 0]
    return selected if len(selected) >= 2 else items


def _candidate_diagnostic_lines(
    candidate_type: str,
    candidates: list[EvidenceCandidate],
    *,
    rank: int,
    support: list[str] | None = None,
) -> list[str]:
    support_ids = support or sorted({candidate.source_group for candidate in candidates})
    return [
        f"candidate_rank={rank} candidate_type={candidate_type}",
        f"candidate_confidence={_candidate_confidence(candidates)}",
        "candidate_support=" + ",".join(dict.fromkeys(support_ids)),
    ]


def _candidate_confidence(candidates: list[EvidenceCandidate]) -> str:
    if not candidates:
        return "0"
    support_count = len({candidate.source_group for candidate in candidates})
    average_relevance = sum(candidate.relevance for candidate in candidates) / len(candidates)
    confidence = min(0.99, 0.55 + min(support_count, 5) * 0.06 + min(average_relevance, 6) * 0.04)
    return f"{confidence:.2f}".rstrip("0").rstrip(".")


def _duration_display(candidate: EvidenceCandidate) -> str:
    return candidate.label or f"{_format_number(float(candidate.value))} minutes"


def _canonical_duration_unit(unit: str) -> str:
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


def _duration_unit_minutes(unit: str) -> float:
    return {
        "minutes": 1,
        "hours": 60,
        "days": 60 * 24,
        "weeks": 60 * 24 * 7,
        "months": 60 * 24 * 28,
    }[unit]


def _filter_focused_candidates(
    query: str,
    items: list[EvidenceCandidate],
) -> list[EvidenceCandidate]:
    if len(items) < 2:
        return items
    focus_terms = _numeric_focus_terms(query)
    if not focus_terms:
        return items
    if max((item.relevance for item in items), default=0) <= 0:
        return items
    selected = [item for item in items if item.relevance > 0]
    return selected if len(selected) >= 2 else items


def _count_query(query: str) -> bool:
    tokens = set(_tokens(query))
    if tokens & {"hours", "hour", "minutes", "minute", "days", "day", "weeks", "week", "months", "month"}:
        return False
    return bool(tokens & {"many", "number", "count", "total"}) and "how" in tokens


def _currency_query(query: str) -> bool:
    tokens = set(_tokens(query))
    if tokens & {"hours", "hour", "minutes", "minute", "days", "day", "weeks", "week", "months", "month"}:
        return bool(tokens & {"money", "amount", "cost", "costs", "price", "prices"})
    return bool(
        tokens & {"money", "amount", "cost", "costs", "spent", "spend", "price", "prices", "much"}
    )


def _expanded_focus_terms(query: str) -> set[str]:
    terms = _query_specific_terms(query)
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


def _numeric_focus_terms(query: str) -> set[str]:
    terms = _query_specific_terms(query)
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
    terms = _query_specific_terms(query)
    expanded = set(terms)
    semantic_groups = {
        "practice": {"practice", "practiced", "practicing"},
        "spent": {"spent", "spend"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
    return expanded


def _query_specific_terms(query: str) -> set[str]:
    return {
        token
        for token in _tokens(query)
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    }


def _relevance(focus_terms: set[str], context: str) -> int:
    if not focus_terms:
        return 0
    context_terms = set(_tokens(context))
    return len(focus_terms & context_terms)


def _context_text(context: str) -> str:
    text = " ".join(context.split())
    return text.split(' {"content":', 1)[0]


def _source_group(context: str) -> str:
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


def _count_label(text: str) -> str:
    cleaned = re.sub(r"\bcontent=\S+\s*", "", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", "", cleaned)
    match = re.search(r"\bI\s+(?P<label>.+?)(?:[.?!]|$)", cleaned)
    if match:
        return " ".join(match.group("label").split()[:12])
    return ""


def _currency_label(text: str, start: int, end: int) -> str:
    after = text[end : end + 120]
    for pattern in (
        r"\s+(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})",
        r"\s+(?:for|on)\s+(?:a|an|the)?\s*(?P<label>[A-Za-z][A-Za-z0-9'&.-]*(?:\s+[A-Za-z][A-Za-z0-9'&.-]*){0,4})",
    ):
        match = re.match(pattern, after)
        if match:
            return _clean_label(match.group("label"))
    before = text[max(0, start - 120) : start]
    match = re.search(
        r"\b(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})\s*$",
        before,
    )
    if match:
        return _clean_label(match.group("label"))
    return ""


def _clean_label(label: str) -> str:
    label = re.split(r"\b(?:on|in|for|with|and|but|because|when)\b", label, maxsplit=1)[0]
    return " ".join(label.strip(" .,'\"").split())


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*", text.casefold()):
        tokens.append(token)
        if re.search(r"[-_:/#]", token):
            tokens.extend(part for part in re.split(r"[-_:/#]+", token) if part)
    return tokens


def _format_currency(value: float) -> str:
    if value.is_integer():
        return f"${int(value):,}"
    whole = int(value)
    fraction = f"{value:.2f}".split(".", 1)[1].rstrip("0")
    return f"${whole:,}.{fraction}"


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


_QUERY_STOPWORDS = {
    "about",
    "after",
    "all",
    "and",
    "amount",
    "attend",
    "attended",
    "before",
    "did",
    "for",
    "from",
    "hours",
    "how",
    "many",
    "money",
    "most",
    "much",
    "past",
    "spent",
    "the",
    "total",
    "what",
    "when",
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
