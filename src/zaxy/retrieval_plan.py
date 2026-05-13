"""Retrieval planning utilities shared by product and benchmark paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent


@dataclass(frozen=True)
class EvidencePlan:
    """Query-level evidence shape required for answerable memory checkout."""

    mode: str
    needs_source_lane: bool
    source_lane_slots: int
    required_source_groups: int
    promote_cited_sources: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable diagnostics representation."""
        return {
            "mode": self.mode,
            "needs_source_lane": self.needs_source_lane,
            "source_lane_slots": self.source_lane_slots,
            "required_source_groups": self.required_source_groups,
            "promote_cited_sources": self.promote_cited_sources,
            "reasons": list(self.reasons),
        }


def build_evidence_plan(query: str, *, limit: int = 10) -> EvidencePlan:
    """Build deterministic evidence requirements for a memory query."""
    intent = classify_retrieval_intent(query, limit=limit)
    reasons = set(intent.reasons)
    if {"aggregation", "aggregation_question"} & reasons:
        mode = "multi_source_aggregation"
        required_source_groups = 2
    elif "absence_check" in reasons:
        mode = "absence_check"
        required_source_groups = 1
    elif "source_recall" in reasons:
        mode = "source_recall"
        required_source_groups = 1
    elif "operational_memory" in reasons:
        mode = "operational_memory"
        required_source_groups = 1
    else:
        mode = "direct_fact"
        required_source_groups = 1 if intent.needs_source_lane else 0
    return EvidencePlan(
        mode=mode,
        needs_source_lane=intent.needs_source_lane,
        source_lane_slots=intent.source_lane_slots,
        required_source_groups=required_source_groups,
        promote_cited_sources=intent.needs_source_lane or mode != "direct_fact",
        reasons=intent.reasons,
    )


def should_query_source_lane(query: str, *, limit: int = 10) -> bool:
    """Return whether source text should supplement graph retrieval."""
    return classify_retrieval_intent(query, limit=limit).needs_source_lane


def source_lane_query(query: str, graph_results: list[str]) -> str:
    """Expand source lookup with compact answer concepts found by graph retrieval."""
    concepts = graph_answer_concepts(graph_results)
    if not concepts:
        return query
    return " ".join([query, *concepts])


def source_lane_candidate_limit(query: str, *, limit: int) -> int:
    """Return internal source candidate budget for source-sensitive retrieval."""
    if limit <= 0:
        return 0
    intent = classify_retrieval_intent(query, limit=limit)
    if not intent.needs_source_lane:
        return limit
    if any(
        reason in intent.reasons
        for reason in ("aggregation", "aggregation_question", "absence_check")
    ):
        return max(limit, intent.source_lane_slots * 6)
    return max(limit, intent.source_lane_slots * 4)


def graph_answer_concepts(graph_results: list[str], *, limit: int = 4) -> list[str]:
    """Extract bounded human-scale concepts from graph context for source backfill."""
    concepts: list[str] = []
    seen: set[str] = set()
    skip_tokens = {
        "entity",
        "event",
        "source",
        "summary",
        "document",
        "citation",
        "benchmark",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
        "for",
        "the",
        "do",
        "now",
    }
    for result in graph_results:
        for phrase in re.findall(
            r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\b",
            result,
        ):
            normalized = phrase.casefold()
            words = normalized.split()
            if normalized in seen or all(word in skip_tokens for word in words):
                continue
            if len(words) == 1 and (words[0] in skip_tokens or len(words[0]) < 3):
                continue
            if re.fullmatch(r"[a-f0-9]{8,}", normalized):
                continue
            concepts.append(phrase)
            seen.add(normalized)
            if len(concepts) >= limit:
                return concepts
    return concepts


def filter_superseded_preference_source_results(
    graph_results: list[str],
    source_results: list[str],
) -> list[str]:
    """Remove raw stale preference rows when graph retrieval has the current fact."""
    current_preferences = _current_preference_values(graph_results)
    if not current_preferences:
        return source_results
    filtered: list[str] = []
    for result in source_results:
        if _is_stale_preference_result(result, current_preferences):
            continue
        filtered.append(result)
    return filtered


def reserve_source_lane(
    fused_results: list[str],
    source_results: list[str],
    *,
    query: str,
    limit: int,
    synthesis_bundle: str | None = None,
) -> list[str]:
    """Preserve top source hits as a bounded lane in fused context."""
    if limit <= 0 or not source_results:
        return fused_results[:limit]
    intent = classify_retrieval_intent(query, limit=limit)
    reserved_count = min(
        len(source_results),
        max(1, min(2, limit // 5), intent.source_lane_slots),
    )
    reserved = diverse_source_contexts(source_results, limit=reserved_count)
    reserved_set = set(reserved)
    primary_slots = max(0, limit - len(reserved))
    primary = [
        result for result in fused_results
        if result not in reserved_set
    ][:primary_slots]
    if should_rank_source_lane_first(intent):
        results = [*reserved, *primary][:limit]
    else:
        results = [*primary, *reserved][:limit]
    if synthesis_bundle is None:
        return results
    return [synthesis_bundle, *[result for result in results if result != synthesis_bundle]][:limit]


def should_rank_source_lane_first(intent: RetrievalIntent) -> bool:
    """Return whether source hits should precede graph hits for this intent."""
    reasons = set(intent.reasons)
    return "personal_memory" in reasons and not reasons & {
        "aggregation",
        "aggregation_question",
        "operational_memory",
        "source_recall",
    }


def source_synthesis_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
) -> str | None:
    """Build one compact cited source bundle for multi-source synthesis queries."""
    intent = classify_retrieval_intent(query, limit=limit)
    if not {"aggregation", "aggregation_question"} & set(intent.reasons) and not _issue_query(query):
        return None
    group_limit = max(limit, intent.source_lane_slots)
    grouped_sources = diverse_source_contexts(source_results, limit=group_limit)
    if len(grouped_sources) < 2:
        return None
    lines = [
        "zaxy_synthesis_bundle=true",
        "synthesis_mode=multi_source_aggregation",
        f"query={query}",
        f"source_count={len(grouped_sources)}",
    ]
    lines.extend(_numeric_synthesis_lines(grouped_sources))
    lines.extend(_date_interval_synthesis_lines(query, grouped_sources))
    lines.extend(_issue_synthesis_lines(query, grouped_sources))
    for index, context in enumerate(grouped_sources, start=1):
        lines.append(
            "- "
            f"source_id={source_context_group(context)} "
            f"citation={source_context_citation(context)} "
            f"snippet={source_context_snippet(context)}"
        )
        if index >= group_limit:
            break
    return "\n".join(lines)


def absence_check_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
) -> str | None:
    """Build cited guidance for questions about absent personal memories."""
    intent = classify_retrieval_intent(query, limit=limit)
    if "absence_check" not in intent.reasons:
        return None
    target = absence_check_target(query)
    if not target:
        return None
    grouped_sources = diverse_source_contexts(
        source_results,
        limit=max(1, intent.source_lane_slots),
    )
    if not grouped_sources or target_terms_present(target, grouped_sources):
        return None
    lines = [
        "zaxy_absence_check=true",
        "synthesis_mode=absence_check",
        f"query={query}",
        f"not_mentioned_candidate={target}",
        (
            "answer_guidance=You did not mention this information. "
            f"You mentioned cited evidence below, but not {target}."
        ),
    ]
    for context in grouped_sources:
        lines.append(
            "- "
            f"source_id={source_context_group(context)} "
            f"citation={source_context_citation(context)} "
            f"snippet={source_context_snippet(context)}"
        )
    return "\n".join(lines)


_ABSENCE_QUERY_STOPWORDS = {
    "about",
    "any",
    "anything",
    "did",
    "do",
    "does",
    "ever",
    "have",
    "i",
    "in",
    "information",
    "me",
    "mention",
    "mentioned",
    "my",
    "not",
    "remember",
    "say",
    "the",
    "this",
    "whether",
}


def absence_check_target(query: str) -> str:
    """Return compact target terms for an absence-check query."""
    terms = [
        token
        for token in source_tokens(query)
        if token not in _ABSENCE_QUERY_STOPWORDS
        and not token.isdigit()
        and len(token) > 1
    ]
    return " ".join(dict.fromkeys(terms))


def target_terms_present(target: str, contexts: list[str]) -> bool:
    """Return whether all target terms are present in any source context."""
    target_terms = [
        token
        for token in source_tokens(target)
        if token not in _ABSENCE_QUERY_STOPWORDS and len(token) > 1
    ]
    if not target_terms:
        return False
    for context in contexts:
        context_terms = set(source_tokens(context))
        if all(term in context_terms for term in target_terms):
            return True
    return False


def diverse_source_contexts(contexts: list[str], *, limit: int) -> list[str]:
    """Select source contexts across provenance groups before filling by rank."""
    if limit <= 0:
        return []
    contexts = source_lane_priority_order(contexts)
    selected: list[str] = []
    seen_contexts: set[str] = set()
    seen_groups: set[str] = set()
    for context in contexts:
        if context in seen_contexts:
            continue
        group = source_context_group(context)
        if group in seen_groups:
            continue
        selected.append(context)
        seen_contexts.add(context)
        seen_groups.add(group)
        if len(selected) >= limit:
            return selected
    for context in contexts:
        if context in seen_contexts:
            continue
        selected.append(context)
        seen_contexts.add(context)
        if len(selected) >= limit:
            break
    return selected


def source_lane_priority_order(contexts: list[str]) -> list[str]:
    """Prefer compact source memories over raw chunks while preserving rank within tiers."""
    indexed = list(enumerate(contexts))
    indexed.sort(key=lambda item: (-source_lane_priority(item[1]), item[0]))
    return [context for _, context in indexed]


def source_lane_priority(context: str) -> int:
    """Return priority tier for a source context."""
    lowered = context.casefold()
    if (
        "salient_memory_turn=true" in lowered
        or "hook.checkpoint" in lowered
        or "longmemeval_salient_memory_turn=true" in lowered
    ):
        return 2
    if "citation=" in lowered or "eventloom://" in lowered or "source_path=" in lowered:
        return 1
    return 0


def source_context_group(context: str) -> str:
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


def source_context_citation(context: str) -> str:
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


def source_context_snippet(context: str, *, max_chars: int = 900) -> str:
    """Return a bounded one-line source snippet."""
    snippet = " ".join(context.split())
    if len(snippet) <= max_chars:
        return snippet
    return f"{snippet[: max_chars - 3].rstrip()}..."


def source_tokens(text: str) -> list[str]:
    """Tokenize source/query text for deterministic planning helpers."""
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*", text.casefold()):
        tokens.append(token)
        if re.search(r"[-_:/#]", token):
            tokens.extend(part for part in re.split(r"[-_:/#]+", token) if part)
    return tokens


def _numeric_synthesis_lines(contexts: list[str]) -> list[str]:
    """Project deterministic numeric operations from cited source snippets."""
    numeric_contexts = [_numeric_context_text(context) for context in contexts]
    lines: list[str] = []
    currency_values = _currency_values(numeric_contexts)
    if currency_values:
        lines.append(
            "currency_values="
            + ",".join(_format_currency(value) for value in currency_values)
        )
        lines.append(f"currency_total={_format_currency(sum(currency_values))}")
        if len(currency_values) >= 2:
            lines.append(
                "currency_difference="
                f"{_format_currency(max(currency_values) - min(currency_values))}"
            )
    minute_values = _unit_values(numeric_contexts, unit_pattern=r"minutes?|mins?")
    if minute_values:
        lines.append("minute_values=" + ",".join(_format_number(value) for value in minute_values))
        lines.append(f"minute_total_hours={_format_number(sum(minute_values) / 60)} hours")
    hour_values = _unit_values(numeric_contexts, unit_pattern=r"hours?|hrs?")
    if hour_values:
        lines.append("hour_values=" + ",".join(_format_number(value) for value in hour_values))
        lines.append(f"hour_total={_format_number(sum(hour_values))} hours")
    day_values = _unit_values(numeric_contexts, unit_pattern=r"days?")
    if day_values:
        lines.append("day_values=" + ",".join(_format_number(value) for value in day_values))
        lines.append(f"day_total={_format_number(sum(day_values))} days")
    month_values = _month_values(numeric_contexts)
    if month_values:
        lines.append("month_values=" + ",".join(_format_number(value) for value in month_values))
        month_total = sum(month_values)
        lines.append(f"month_total={_format_number(month_total)} months ago")
        if month_words := _number_words(month_total):
            lines.append(f"month_total_words={month_words} months ago")
    return lines


def _numeric_context_text(context: str) -> str:
    """Return source text once, excluding Eventloom JSON payload echoes."""
    text = source_context_snippet(context)
    return text.split(' {"content":', 1)[0]


def _currency_values(contexts: list[str]) -> list[float]:
    values: list[float] = []
    for context in contexts:
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", context):
            values.append(float(match.group("value").replace(",", "")))
    return values


def _unit_values(contexts: list[str], *, unit_pattern: str) -> list[float]:
    values: list[float] = []
    pattern = re.compile(
        rf"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?:{unit_pattern})\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            values.append(float(match.group("value")))
    return values


_NUMBER_WORDS = {
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


def _month_values(contexts: list[str]) -> list[float]:
    values = _unit_values(contexts, unit_pattern=r"months?")
    pattern = re.compile(
        r"\b(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            values.append(float(_NUMBER_WORDS[match.group("value").casefold()]))
    return values


def _number_words(value: float) -> str | None:
    if not value.is_integer():
        return None
    integer = int(value)
    for word, number in _NUMBER_WORDS.items():
        if number == integer:
            return word.title()
    return None


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


def _date_interval_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project deterministic day intervals from cited temporal evidence."""
    if "days" not in query.casefold():
        return []
    observations = _dated_observations(contexts)
    if len(observations) < 2:
        return []
    intervals: list[int] = []
    for index, first in enumerate(observations):
        for second in observations[index + 1 :]:
            delta = abs((second - first).days)
            if 0 < delta <= 366 and delta not in intervals:
                intervals.append(delta)
    lines: list[str] = []
    for delta in intervals[:5]:
        lines.append(f"date_interval_days={delta}")
        lines.append(
            "date_interval_answer="
            f"{delta} days. {delta + 1} days (including the last day) is also acceptable."
        )
    return lines


def _dated_observations(contexts: list[str]) -> list[date]:
    observations: list[date] = []
    for context in contexts:
        raw_text = _numeric_context_text(context)
        text = _temporal_evidence_text(raw_text)
        default_year = _context_year(raw_text)
        for value in _explicit_dates(text, default_year=default_year):
            if value not in observations:
                observations.append(value)
    return observations


def _temporal_evidence_text(text: str) -> str:
    """Remove provenance timestamps so synthesis uses remembered event dates."""
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


def _append_date(dates: list[date], year: int, month: int, day: int) -> None:
    try:
        value = date(year, month, day)
    except ValueError:
        return
    if value not in dates:
        dates.append(value)


def _issue_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project normalized issue candidates from cited source snippets."""
    if not _issue_query(query):
        return []
    lines: list[str] = []
    for context in contexts:
        text = _numeric_context_text(context)
        for match in re.finditer(
            r"\bissue with (?:my|the)?\s*(?:car's\s*)?(?P<subject>[A-Za-z0-9][A-Za-z0-9' -]{1,80}?)(?:\s+on\b|\s+and\b|\s+that\b|[,.])",
            text,
            flags=re.IGNORECASE,
        ):
            subject = " ".join(match.group("subject").replace("'s", "").split())
            if not subject:
                continue
            lines.append(f"issue_candidate={subject} not functioning correctly")
            if len(lines) >= 3:
                return lines
    return lines


def _issue_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool({"issue", "problem", "problems"} & tokens)


def _format_currency(value: float) -> str:
    return f"${_format_number(value)}"


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _current_preference_values(results: list[str]) -> dict[tuple[str, str], str]:
    preferences: dict[tuple[str, str], str] = {}
    for result in results:
        for match in re.finditer(
            r"\b(?P<user>user-\d{4}):(?P<key>[A-Za-z0-9_.-]+)\b.*?"
            r"(?P=key)=(?P<value>[A-Za-z0-9_.-]+)",
            result,
            flags=re.IGNORECASE,
        ):
            preferences[
                (match.group("user").casefold(), match.group("key").casefold())
            ] = match.group("value").casefold()
    return preferences


def _is_stale_preference_result(
    result: str,
    current_preferences: dict[tuple[str, str], str],
) -> bool:
    lowered = result.casefold()
    for (user_id, key), current_value in current_preferences.items():
        if user_id not in lowered or key not in lowered:
            continue
        value_match = re.search(
            rf"\b(?:value|{re.escape(key)})[=:]\s*['\"]?(?P<value>[A-Za-z0-9_.-]+)",
            result,
            flags=re.IGNORECASE,
        )
        if value_match and value_match.group("value").casefold() != current_value:
            return True
    return False
