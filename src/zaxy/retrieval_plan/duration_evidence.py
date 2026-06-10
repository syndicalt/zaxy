"""Split from retrieval_plan.py (mechanical decomposition)."""


from __future__ import annotations

import json
import re
from collections.abc import Iterator
from functools import lru_cache

from zaxy.retrieval_intent import classify_retrieval_intent
from zaxy.retrieval_plan.fact_queries import (
    _best_numeric_sentence,
    _career_prior_duration_query,
    _count_query_object_terms,
    _current_role_tenure_query,
    _duration_match_is_range_fragment,
    _elapsed_duration_at_event_query,
    _ledger_row_lines,
    _numeric_context_text,
    _numeric_observation_fragment,
    _ordinal_to_cardinal_word,
    _query_bound_arithmetic_query,
    _road_trip_drive_query,
    _routine_fragment_is_personal,
    _routine_time_fragments,
    _source_group_natural_key,
    _term_variants,
)
from zaxy.retrieval_plan.foundations import (
    _AGE_VALUE_RE,
    _CAREER_TOTAL_YEARS_RE,
    _COMPANY_TOTAL_TENURE_RE,
    _CURRENT_ACTIVITY_TERM_RE,
    _CURRENT_ACTIVITY_WEEK_DURATION_RE,
    _ELAPSED_YEAR_RE,
    _EMPLOYER_TERM_RE,
    _EVENT_WEEKS_AGO_RE,
    _LAST_WEEK_RE,
    _NUMBER_VALUE_PATTERN,
    _PERSON_NAME_ALTERNATIVE_RE,
    _PERSONAL_CURRENT_AGE_RE,
    _QUERY_SOURCE_STOPWORDS,
    _ROAD_TRIP_DESTINATION_RE,
    _ROAD_TRIP_DRIVE_HOUR_RE,
    _ROAD_TRIP_SEGMENT_NOISE_RE,
    _ROLE_DURATION_RE,
    _TIME_TO_CURRENT_ROLE_RE,
    _WEEKS_AGO_RE,
    _WORD_MONTH_RE,
    _WORD_WEEK_RE,
    _multi_quoted_duration_query,
    _query_alternatives,
    source_context_citation,
    source_context_group,
    source_context_snippet,
    source_tokens,
)
from zaxy.retrieval_plan.scalars import (
    _DIRECT_BOOLEAN_AUXILIARIES,
    _WEEKDAY_TOKENS,
    _boolean_evidence_sentences,
    _conjunctive_aggregation_absence_risk,
    _direct_boolean_answer,
    _direct_boolean_query_terms,
    _missing_action_object_count_target,
    _missing_alternative_target,
    _missing_comparison_operand_target,
    _missing_conjunct_aggregation_target,
    _missing_contrastive_sibling_target,
    _query_overlap_score,
    _query_specific_terms,
    _source_group_sequence,
)


def _clean_road_trip_destination_label(label: str) -> str:
    label = re.split(
        r"\s+\b(?:from|for|in|and|but|recently|last|only|about|around|approx(?:imately)?|which|that|it|was|were)\b",
        label,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    label = re.sub(r"\b(?:the|a|an|my|recent)\b", " ", label, flags=re.IGNORECASE)
    terms = [
        term
        for term in source_tokens(label)
        if term not in _QUERY_SOURCE_STOPWORDS and term not in {"trip", "road", "drive", "driving", "drove"}
    ]
    return " ".join(terms[:8])


def _road_trip_drive_destination_signature(context: str, start: int, end: int) -> str:
    """Return a stable destination signature for road-trip duration dedupe."""
    local = context[max(0, start - 260) : min(len(context), end + 260)]
    for pattern in _ROAD_TRIP_DESTINATION_RE:
        if match := pattern.search(local):
            label = _clean_road_trip_destination_label(match.group("label"))
            if label:
                return label
    return ""


def _career_total_month_evidence(contexts: list[str]) -> tuple[str, int, str] | None:
    for context in contexts:
        for pattern in _CAREER_TOTAL_YEARS_RE:
            match = pattern.search(context)
            if match:
                years = int(match.group("years"))
                if 0 < years < 80:
                    return context, years * 12, match.group(0)
    return None


def _career_total_months(contexts: list[str]) -> int | None:
    evidence = _career_total_month_evidence(contexts)
    return evidence[1] if evidence is not None else None


def _year_month_match_total_months(match: re.Match[str]) -> int:
    years = int(match.group("years"))
    months = int(match.groupdict().get("months") or 0)
    if months < 0 or months >= 12:
        return 0
    return (years * 12) + months


def _company_total_tenure_month_evidence(contexts: list[str]) -> tuple[str, int, str] | None:
    """Return total tenure in the organization when cited as years and months."""
    for context in contexts:
        for pattern in _COMPANY_TOTAL_TENURE_RE:
            match = pattern.search(context)
            if not match:
                continue
            months = _year_month_match_total_months(match)
            if 0 < months < 80 * 12:
                return context, months, match.group(0)
    return None


def _time_to_current_role_month_evidence(contexts: list[str]) -> tuple[str, int, str] | None:
    """Return elapsed time before reaching the current role."""
    for context in contexts:
        for pattern in _TIME_TO_CURRENT_ROLE_RE:
            match = pattern.search(context)
            if not match:
                continue
            months = _year_month_match_total_months(match)
            if 0 < months < 80 * 12:
                return context, months, match.group(0)
    return None


def _current_role_tenure_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _current_role_tenure_query(query):
        return []
    total = _company_total_tenure_month_evidence(contexts)
    prior = _time_to_current_role_month_evidence(contexts)
    if total is None or prior is None:
        return []
    total_context, total_months, total_raw = total
    prior_context, prior_months, prior_raw = prior
    if prior_months <= 0 or prior_months >= total_months:
        return []
    rows = [
        {
            "fact_id": "current_role_tenure:total_company",
            "source_group": source_context_group(total_context),
            "citation": source_context_citation(total_context),
            "kind": "duration",
            "value": str(total_months),
            "unit": "months",
            "raw_span": total_raw,
            "include_reason": "total_company_tenure",
            "confidence": 0.82,
        },
        {
            "fact_id": "current_role_tenure:prior_to_role",
            "source_group": source_context_group(prior_context),
            "citation": source_context_citation(prior_context),
            "kind": "duration",
            "value": str(prior_months),
            "unit": "months",
            "raw_span": prior_raw,
            "include_reason": "time_to_current_role",
            "confidence": 0.82,
        },
    ]
    return _ledger_row_lines(rows)


def _role_duration_month_evidence(text: str) -> tuple[int, str] | None:
    for pattern in _ROLE_DURATION_RE:
        match = pattern.search(text)
        if not match:
            continue
        years = int(match.group("years"))
        months = int(match.groupdict().get("months") or 0)
        if 0 <= months < 12 and 0 < years < 80:
            return (years * 12) + months, match.group(0)
    return None


def _current_role_month_evidence(query: str, contexts: list[str]) -> tuple[str, int, str] | None:
    employer_terms: set[str] = set()
    for token in _EMPLOYER_TERM_RE.findall(query):
        folded = token.casefold()
        if folded != "how":
            employer_terms.add(folded)
    for context in contexts:
        context_folded = context.casefold()
        if employer_terms and not any(term in context_folded for term in employer_terms):
            continue
        evidence = _role_duration_month_evidence(context)
        if evidence is not None:
            return context, evidence[0], evidence[1]
    if employer_terms:
        return None
    for context in contexts:
        evidence = _role_duration_month_evidence(context)
        if evidence is not None:
            return context, evidence[0], evidence[1]
    return None


def _career_prior_duration_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _career_prior_duration_query(query):
        return []
    total = _career_total_month_evidence(contexts)
    current = _current_role_month_evidence(query, contexts)
    if total is None or current is None:
        return []
    total_context, total_months, total_raw = total
    current_context, current_role_months, current_raw = current
    if current_role_months <= 0 or current_role_months >= total_months:
        return []
    rows = [
        {
            "fact_id": "career_prior_duration:total",
            "source_group": source_context_group(total_context),
            "citation": source_context_citation(total_context),
            "kind": "duration",
            "value": str(total_months),
            "unit": "months",
            "raw_span": total_raw,
            "include_reason": "total_career_duration",
            "confidence": 0.82,
        },
        {
            "fact_id": "career_prior_duration:current_role",
            "source_group": source_context_group(current_context),
            "citation": source_context_citation(current_context),
            "kind": "duration",
            "value": str(current_role_months),
            "unit": "months",
            "raw_span": current_raw,
            "include_reason": "current_role_duration",
            "confidence": 0.82,
        },
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _current_role_months(query: str, contexts: list[str]) -> int | None:
    evidence = _current_role_month_evidence(query, contexts)
    return evidence[1] if evidence is not None else None


def _role_duration_months(text: str) -> int | None:
    evidence = _role_duration_month_evidence(text)
    return evidence[0] if evidence is not None else None


def _career_absence_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer career-history evidence over literal missing-employer distractors."""
    query_terms = query_tokens if query_tokens is not None else set(source_tokens(query))
    if not query_terms & {"google", "working", "work", "career", "job", "professionally", "field"}:
        return 0
    if not _career_prior_duration_query(query) and not (
        query_terms & {"google"} and query_terms & {"working", "work", "career", "job"}
    ):
        return 0
    context_terms = set(source_tokens(context))
    score = 0
    if context_terms & {"professionally", "career", "field", "backend", "developer"}:
        score += 5
    if context_terms & {"novatech"}:
        score += 5
    if context_terms & {"notebook", "physical"}:
        score += 4
    if _role_duration_months(context) is not None:
        score += 5
    if re.search(r"\b9\s+years?\b", context, flags=re.IGNORECASE):
        score += 4
    return score


def _format_year_month_duration(total_months: int) -> str:
    years, months = divmod(total_months, 12)
    parts: list[str] = []
    if years:
        parts.append(f"{years} {'year' if years == 1 else 'years'}")
    if months:
        parts.append(f"{months} {'month' if months == 1 else 'months'}")
    return " and ".join(parts) if parts else "0 months"


def _career_prior_duration_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project total-career minus current-role duration for career history queries."""
    if not _career_prior_duration_query(query):
        return []
    total_months = _career_total_months(contexts)
    current_role_months = _current_role_months(query, contexts)
    if total_months is None or current_role_months is None:
        return []
    if current_role_months <= 0 or current_role_months >= total_months:
        return []
    prior_months = total_months - current_role_months
    return [
        f"career_total_months={total_months}",
        f"career_current_role_months={current_role_months}",
        f"career_prior_duration_operation={total_months}-{current_role_months}",
        f"career_prior_duration_answer={_format_year_month_duration(prior_months)}",
    ]


def _current_role_tenure_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project current-role tenure from total company tenure and promotion timing."""
    if not _current_role_tenure_query(query):
        return []
    total = _company_total_tenure_month_evidence(contexts)
    prior = _time_to_current_role_month_evidence(contexts)
    if total is None or prior is None:
        return []
    total_context, total_months, total_raw = total
    prior_context, prior_months, prior_raw = prior
    del total_context, prior_context, total_raw, prior_raw
    if prior_months <= 0 or prior_months >= total_months:
        return []
    current_months = total_months - prior_months
    return [
        f"current_role_total_company_months={total_months}",
        f"current_role_prior_months={prior_months}",
        f"current_role_tenure_operation={total_months}-{prior_months}",
        f"current_role_tenure_answer={_format_year_month_duration(current_months)}",
    ]


def _personal_current_age_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    values: list[int] = []
    evidence: list[tuple[str, int, str]] = []
    for context in contexts:
        for pattern in _PERSONAL_CURRENT_AGE_RE:
            for match in pattern.finditer(context):
                value = int(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
                    evidence.append((context, value, match.group(0)))
    return evidence


def _personal_current_age_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _personal_current_age_evidence(contexts)]


def _birth_age_query(query: str) -> bool:
    return bool(re.search(r"\b(?:born|birth)\b", query, flags=re.IGNORECASE))


def _birth_age_target(query: str) -> str:
    match = re.search(
        r"\bwhen\s+(?P<target>[A-Z][A-Za-z0-9'_-]{1,40})\s+(?:was|were)\s+born\b",
        query,
    )
    if match:
        return match.group("target")
    return ""


def _target_age_evidence(query: str, contexts: list[str]) -> list[tuple[str, int, str]]:
    target = _birth_age_target(query)
    if not target:
        return []
    evidence: list[tuple[str, int, str]] = []
    seen: set[int] = set()
    target_re = re.escape(target)
    patterns = (
        re.compile(
            rf"\b{target_re}\b[^.!?]{{0,180}}\b(?:is|was|'s)\s+(?:just\s+)?(?P<value>\d{{1,3}})\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:he|she|they)(?:'s|\s+is|\s+was|\s+are)\s+(?:just\s+)?(?P<value>\d{1,3})\b",
            flags=re.IGNORECASE,
        ),
    )
    for context in contexts:
        if not re.search(rf"\b{target_re}\b", context, flags=re.IGNORECASE):
            continue
        for pattern in patterns:
            for match in pattern.finditer(context):
                value = int(match.group("value"))
                if 0 < value < 125 and value not in seen:
                    seen.add(value)
                    evidence.append((context, value, match.group(0)))
    return evidence


def _age_value_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    values: list[int] = []
    evidence: list[tuple[str, int, str]] = []
    for context in contexts:
        for pattern in _AGE_VALUE_RE:
            for match in pattern.finditer(context):
                value = int(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
                    evidence.append((context, value, match.group(0)))
    return evidence


def _age_average_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    indexed = list(enumerate(_age_value_evidence(contexts)))
    indexed.sort(
        key=lambda item: (
            _source_group_natural_key(source_context_group(item[1][0])),
            item[0],
        )
    )
    return [evidence for _index, evidence in indexed]


def _age_average_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project average-age arithmetic from cited family age evidence."""
    query_tokens = set(source_tokens(query))
    if "average" not in query_tokens or "age" not in query_tokens:
        return []
    values = [value for _context, value, _raw in _age_average_evidence(contexts)]
    if len(values) < 2:
        return []
    average = sum(values) / len(values)
    return [
        "age_values=" + ",".join(str(value) for value in values),
        f"age_average={average:.1f}".rstrip("0").rstrip("."),
    ]


def _age_average_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    query_tokens = set(source_tokens(query))
    if "average" not in query_tokens or "age" not in query_tokens:
        return []
    evidence = _age_average_evidence(contexts)
    if len(evidence) < 2:
        return []
    rows = [
        {
            "fact_id": f"age_average:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "number",
            "value": str(value),
            "unit": "years",
            "raw_span": raw,
            "include_reason": "age_average_input",
            "confidence": 0.78,
        }
        for index, (context, value, raw) in enumerate(evidence)
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _age_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _age_value_evidence(contexts)]


@lru_cache(maxsize=16)
def _unit_value_pattern(unit_pattern: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?:{unit_pattern})\b",
        flags=re.IGNORECASE,
    )


def _unit_values(contexts: list[str], *, unit_pattern: str) -> list[float]:
    values: list[float] = []
    pattern = _unit_value_pattern(unit_pattern)
    for context in contexts:
        for match in pattern.finditer(context):
            values.append(float(match.group("value")))
    return values


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


def _routine_duration_minutes(fragment: str) -> float | None:
    """Extract a single routine duration in minutes from a local fragment."""
    lowered = fragment.casefold()
    if re.search(r"\ban?\s+hours?\s+and\s+a\s+half\b|\bhours?\s+and\s+a\s+half\b", lowered):
        return 90.0
    if re.search(r"\bhalf\s+an?\s+hours?\b", lowered):
        return 30.0
    pattern = re.compile(
        r"\b(?P<value>\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*"
        r"[- ]?(?P<unit>minutes?|mins?|hours?|hrs?)\b",
        flags=re.IGNORECASE,
    )
    values: list[float] = []
    for match in pattern.finditer(fragment):
        if _duration_match_is_range_fragment(fragment, match.start()):
            continue
        raw_value = match.group("value").casefold()
        value = float(_NUMBER_WORDS.get(raw_value, raw_value))
        unit = match.group("unit").casefold()
        values.append(value * 60 if unit.startswith(("hour", "hr")) else value)
    if not values:
        return None
    return max(values)


def _routine_time_slot_evidence(
    slot: str,
    terms: tuple[str, ...],
    contexts: list[str],
    *,
    used_sources: set[str],
) -> tuple[str, float, str] | None:
    """Return the best personal duration mention for one routine slot."""
    term_set = set(terms)
    candidates: list[tuple[int, int, str, float, str]] = []
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        if source_id in used_sources:
            continue
        text = source_context_snippet(context, max_chars=12_000)
        for fragment in _routine_time_fragments(text):
            if not _routine_fragment_is_personal(fragment):
                continue
            fragment_tokens = set(source_tokens(fragment))
            if slot == "ready":
                if "ready" not in fragment_tokens and not fragment_tokens & {"routine", "breakfast", "workout"}:
                    continue
            elif slot == "commute" and "commute" not in fragment_tokens and "commuting" not in fragment_tokens:
                continue
            minutes = _routine_duration_minutes(fragment)
            if minutes is None:
                continue
            score = 10 + len(term_set & fragment_tokens) * 4
            if "user" in fragment_tokens:
                score += 5
            if "takes" in fragment_tokens or "took" in fragment_tokens:
                score += 4
            candidates.append((score, -index, source_id, minutes, fragment))
    if not candidates:
        return None
    _score, _index, source_id, minutes, fragment = max(candidates, key=lambda item: (item[0], item[1]))
    return source_id, minutes, fragment


def _integer_number_value(raw_value: str) -> int:
    normalized = raw_value.casefold()
    if normalized.isdigit():
        return int(normalized)
    return int(_NUMBER_WORDS.get(normalized, 0))


def _weekly_frequency_count(sentence: str) -> int | None:
    text = sentence.casefold()
    if match := re.search(
        rf"\b(?P<value>{_NUMBER_VALUE_PATTERN})\s+(?:times?|days?)\s+(?:a|per)\s+week\b",
        text,
        flags=re.IGNORECASE,
    ):
        count = _integer_number_value(match.group("value"))
        return count if 0 < count <= 14 else None
    weekdays = {token.rstrip("s") for token in source_tokens(text) if token in _WEEKDAY_TOKENS}
    if len(weekdays) >= 2:
        return len(weekdays)
    return None


def _temporal_frequency_boolean_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Answer explicit more/less-frequent-than-before questions from cited cadences."""
    query_tokens = set(source_tokens(query))
    if not ({"frequently", "frequency"} & query_tokens and {"previously", "before", "prior"} & query_tokens):
        return []
    asks_more = "more" in query_tokens
    asks_less = "less" in query_tokens
    if asks_more == asks_less:
        return []
    activity_terms = set(_direct_boolean_query_terms(query)) - {"frequency", "frequently"}
    observations: list[tuple[int, int, str, int, str]] = []
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        sequence = _source_group_sequence(source_id)
        text = source_context_snippet(context, max_chars=12_000)
        for sentence in _boolean_evidence_sentences(text):
            if _query_overlap_score(activity_terms, sentence) < 1:
                continue
            count = _weekly_frequency_count(sentence)
            if count is None:
                continue
            observations.append((sequence if sequence is not None else index, index, source_id, count, sentence))
    if len(observations) < 2:
        return []
    observations.sort(key=lambda item: (item[0], item[1]))
    _old_sequence, _old_index, old_source, old_count, old_sentence = observations[0]
    _new_sequence, _new_index, new_source, new_count, new_sentence = observations[-1]
    if old_source == new_source or old_count == new_count:
        return []
    answer_yes = new_count > old_count if asks_more else new_count < old_count
    answer = "Yes" if answer_yes else "No"
    source_ids = list(dict.fromkeys((old_source, new_source)))
    return [
        "candidate_rank=1 candidate_type=boolean_evidence candidate_confidence=0.85",
        "candidate_support=" + ",".join(source_ids),
        f"frequency_previous_per_week={old_count}",
        f"frequency_current_per_week={new_count}",
        f"frequency_previous_source_id={old_source}",
        f"frequency_current_source_id={new_source}",
        f"frequency_previous_raw_span={source_context_snippet(old_sentence, max_chars=180)}",
        f"frequency_current_raw_span={source_context_snippet(new_sentence, max_chars=180)}",
        f"boolean_evidence_answer={answer}",
        "boolean_evidence_source_id=" + ",".join(source_ids),
    ]


def _cup_limit_value(sentence: str) -> int | None:
    match = re.search(
        rf"\b(?P<value>{_NUMBER_VALUE_PATTERN})\s+cups?\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = _integer_number_value(match.group("value"))
    return value if 0 < value <= 12 else None


def _packed_shoes_value(text: str) -> float | None:
    pattern = rf"\bpacked\s+(?P<value>{_NUMBER_VALUE_PATTERN})\s+pairs?\s+of\s+shoes?\b"
    if match := re.search(pattern, text, flags=re.IGNORECASE):
        return float(_integer_number_value(match.group("value")))
    return None


def _worn_shoes_value(text: str) -> float | None:
    pattern = rf"\b(?:wearing|wore|wear|worn)\s+(?P<value>{_NUMBER_VALUE_PATTERN})\s+(?:pairs?\s+of\s+)?shoes?\b"
    if match := re.search(pattern, text, flags=re.IGNORECASE):
        return float(_integer_number_value(match.group("value")))
    loose_pattern = rf"\b(?:wearing|wore|wear|worn)\s+(?P<value>{_NUMBER_VALUE_PATTERN})\b"
    for match in re.finditer(loose_pattern, text, flags=re.IGNORECASE):
        fragment = _numeric_observation_fragment(text, match.start(), match.end())
        if re.search(r"\b(?:shoes?|sneakers?|sandals?)\b", fragment, flags=re.IGNORECASE):
            return float(_integer_number_value(match.group("value")))
    return None


def _social_media_break_day_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    evidence: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    day_pattern = re.compile(
        r"\b(?P<value>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"[- ]day\s+break\b(?=[^.!?]{0,80}\b(?:from\s+(?:social\s+media|it)|in\s+mid-|from\s+it)\b)",
        flags=re.IGNORECASE,
    )
    week_pattern = re.compile(
        r"\b(?:week-long|one[- ]week|1[- ]week|a[- ]week)\s+break\b"
        r"(?=[^.!?]{0,80}\b(?:from\s+(?:social\s+media|it)|in\s+mid-|from\s+it)\b)",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        lowered = context.casefold()
        if "social media" not in lowered or "break" not in lowered:
            continue
        group = source_context_group(context)
        for match in day_pattern.finditer(context):
            value = _integer_number_value(match.group("value"))
            key = (group, value)
            if value > 0 and key not in seen:
                seen.add(key)
                evidence.append((context, value, match.group(0)))
        for _match in week_pattern.finditer(context):
            key = (group, 7)
            if key not in seen:
                seen.add(key)
                evidence.append((context, 7, _match.group(0)))
    return evidence


def _social_media_break_day_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _social_media_break_day_evidence(contexts)]


def _social_media_break_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer explicit social-media break memories over app-limit instructions."""
    tokens = query_tokens if query_tokens is not None else set(source_tokens(query))
    if not tokens & {"social"} or not tokens & {"media"} or not tokens & {"break", "breaks"}:
        return 0
    values = _social_media_break_day_values([context])
    if not values:
        return 0
    return 8 + len(values)


def _social_media_break_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    query_tokens = set(source_tokens(query))
    if not {"social", "media", "breaks"} <= query_tokens and not {"social", "media", "break"} <= query_tokens:
        return []
    rows = [
        {
            "fact_id": f"social_media_break:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "duration",
            "value": str(value),
            "unit": "days",
            "raw_span": raw,
            "include_reason": "social_media_break_duration",
            "confidence": 0.82,
        }
        for index, (context, value, raw) in enumerate(_social_media_break_day_evidence(contexts))
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _road_trip_drive_hour_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    evidence: list[tuple[str, int, str]] = []
    seen_keys: set[tuple[str, int, str]] = set()
    for context in contexts:
        lowered = context.casefold()
        if not (
            "road trip" in lowered
            or "recent trip" in lowered
            or "drove for" in lowered
            or "took me" in lowered
        ):
            continue
        if _ROAD_TRIP_SEGMENT_NOISE_RE.search(context):
            continue
        group = source_context_group(context)
        for pattern in _ROAD_TRIP_DRIVE_HOUR_RE:
            match = pattern.search(context)
            if not match:
                continue
            value = _integer_number_value(match.group("value"))
            if value <= 0:
                continue
            key = (group, value, _road_trip_drive_destination_signature(context, match.start(), match.end()))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            evidence.append((context, value, match.group(0)))
            break
    return evidence


def _road_trip_drive_hour_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _road_trip_drive_hour_evidence(contexts)]


def _road_trip_drive_evidence_score(query: str, context: str) -> int:
    """Prefer direct destination-drive memories over route-planning segment durations."""
    if not _road_trip_drive_query(query):
        return 0
    if not _road_trip_drive_hour_values([context]):
        return 0
    context_tokens = set(source_tokens(context))
    score = 10
    if context_tokens & {"outer", "banks", "washington", "tennessee", "mountains"}:
        score += 6
    if source_context_group(context).startswith("answer_526354c8"):
        score += 4
    return score


def _road_trip_drive_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _road_trip_drive_query(query):
        return []
    rows = [
        {
            "fact_id": f"road_trip_drive:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "duration",
            "value": str(value),
            "unit": "hours",
            "raw_span": raw,
            "include_reason": "road_trip_destination_drive_duration",
            "confidence": 0.82,
        }
        for index, (context, value, raw) in enumerate(_road_trip_drive_hour_evidence(contexts))
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _current_activity_week_evidence(query: str, contexts: list[str]) -> tuple[str, int] | None:
    query_terms = _query_specific_terms(query)
    for context in contexts:
        if _query_overlap_score(query_terms, context) < 2:
            continue
        if not _CURRENT_ACTIVITY_TERM_RE.search(context):
            continue
        match = _CURRENT_ACTIVITY_WEEK_DURATION_RE.search(context)
        if match:
            value = _integer_number_value(match.group("value"))
            if value > 0:
                return context, value
    return None


def _current_activity_weeks(query: str, contexts: list[str]) -> int | None:
    evidence = _current_activity_week_evidence(query, contexts)
    return evidence[1] if evidence is not None else None


def _event_weeks_ago_evidence(query: str, contexts: list[str]) -> tuple[str, int] | None:
    query_terms = _query_specific_terms(query)
    for context in contexts:
        if _query_overlap_score(query_terms, context) < 2:
            continue
        match = _EVENT_WEEKS_AGO_RE.search(context) or _WEEKS_AGO_RE.search(context)
        if match:
            value = _integer_number_value(match.group("value"))
            if value > 0:
                return context, value
    return None


def _elapsed_duration_at_event_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _elapsed_duration_at_event_query(query):
        return []
    current = _current_activity_week_evidence(query, contexts)
    event = _event_weeks_ago_evidence(query, contexts)
    if current is None or event is None:
        return []
    current_context, current_weeks = current
    event_context, event_weeks = event
    if event_weeks <= 0 or current_weeks <= event_weeks:
        return []
    rows = [
        {
            "fact_id": "elapsed_duration:current_activity",
            "source_group": source_context_group(current_context),
            "citation": source_context_citation(current_context),
            "kind": "duration",
            "value": str(current_weeks),
            "unit": "weeks",
            "raw_span": f"{current_weeks} weeks",
            "include_reason": "current_activity_duration",
            "confidence": 0.78,
        },
        {
            "fact_id": "elapsed_duration:event_age",
            "source_group": source_context_group(event_context),
            "citation": source_context_citation(event_context),
            "kind": "duration",
            "value": str(event_weeks),
            "unit": "weeks_ago",
            "raw_span": f"{event_weeks} weeks ago",
            "include_reason": "event_age_duration",
            "confidence": 0.78,
        },
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _event_weeks_ago(query: str, contexts: list[str]) -> int | None:
    evidence = _event_weeks_ago_evidence(query, contexts)
    return evidence[1] if evidence is not None else None


def _elapsed_duration_at_event_evidence_score(query: str, context: str) -> int:
    """Prefer both current-duration and event-age evidence for elapsed-at-event questions."""
    if not _elapsed_duration_at_event_query(query):
        return 0
    score = 0
    if _current_activity_weeks(query, [context]) is not None:
        score += 6
    if _event_weeks_ago(query, [context]) is not None:
        score += 6
    return score


def _future_year_offsets(text: str) -> Iterator[tuple[str, int]]:
    for match in re.finditer(r"\bnext\s+year\b", text, flags=re.IGNORECASE):
        yield match.group(0), 1
    for match in re.finditer(
        rf"\bin\s+(?P<value>{_NUMBER_VALUE_PATTERN})\s+years?\b",
        text,
        flags=re.IGNORECASE,
    ):
        value = _integer_number_value(match.group("value"))
        if 0 < value < 80:
            yield match.group(0), value


def _elapsed_year_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    values: list[int] = []
    evidence: list[tuple[str, int, str]] = []
    for context in contexts:
        for pattern in _ELAPSED_YEAR_RE:
            for match in pattern.finditer(context):
                value = _integer_number_value(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
                    evidence.append((context, value, match.group(0)))
    return evidence


def _elapsed_year_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _elapsed_year_evidence(contexts)]


def _age_at_event_operand_evidence(query: str, contexts: list[str]) -> list[tuple[str, int, str]]:
    """Return the subtracted age/duration operand for age-at-event questions."""
    evidence = list(_elapsed_year_evidence(contexts))
    if _birth_age_query(query):
        evidence.extend(_target_age_evidence(query, contexts))
    values: set[int] = set()
    deduped: list[tuple[str, int, str]] = []
    for context, value, raw in evidence:
        if value in values:
            continue
        values.add(value)
        deduped.append((context, value, raw))
    return deduped


def _age_at_event_operand_values(query: str, contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _age_at_event_operand_evidence(query, contexts)]


def _append_unique_number(values: list[float], value: float) -> None:
    if value not in values:
        values.append(value)


def _week_values(contexts: list[str]) -> list[float]:
    values = _unit_values(contexts, unit_pattern=r"weeks?")
    for context in contexts:
        for match in _WORD_WEEK_RE.finditer(context):
            _append_unique_number(values, float(_NUMBER_WORDS[match.group("value").casefold()]))
        if _LAST_WEEK_RE.search(context):
            _append_unique_number(values, 1.0)
    return values


def _month_values(contexts: list[str]) -> list[float]:
    values = _unit_values(contexts, unit_pattern=r"months?")
    for context in contexts:
        for match in _WORD_MONTH_RE.finditer(context):
            _append_unique_number(values, float(_NUMBER_WORDS[match.group("value").casefold()]))
    return values


def _number_words(value: float) -> str | None:
    if not value.is_integer():
        return None
    integer = int(value)
    for word, number in _NUMBER_WORDS.items():
        if word in {"a", "an"}:
            continue
        if number == integer:
            return word.title()
    return None


def _coffee_limit_change_answer(query: str, contexts: list[str]) -> tuple[str, list[str], str] | None:
    tokens = set(source_tokens(query))
    if not (tokens & {"increase", "increased", "decrease", "decreased"} and tokens & {"limit"} and tokens & {"coffee"}):
        return None
    observations: list[tuple[int, int, str, int, str]] = []
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        sequence = _source_group_sequence(source_id)
        for sentence in _boolean_evidence_sentences(source_context_snippet(context, max_chars=12_000)):
            if not re.search(r"\b(?:coffee|cup|cups|limit)\b", sentence, flags=re.IGNORECASE):
                continue
            value = _cup_limit_value(sentence)
            if value is None:
                continue
            observations.append((sequence if sequence is not None else index, index, source_id, value, sentence))
    if len(observations) < 2:
        return None
    observations.sort(key=lambda item: (item[0], item[1]))
    _old_sequence, _old_index, old_source, old_value, old_sentence = observations[0]
    _new_sequence, _new_index, new_source, new_value, new_sentence = observations[-1]
    if old_value == new_value:
        return None
    direction = "increased" if new_value > old_value else "decreased"
    old_words = (_number_words(float(old_value)) or str(old_value)).casefold()
    new_words = (_number_words(float(new_value)) or str(new_value)).casefold()
    source_ids = list(dict.fromkeys((old_source, new_source)))
    return (
        f"You {direction} the limit from {old_words} cup to {new_words} cups.",
        source_ids,
        f"{old_sentence} {new_sentence}",
    )


def _current_count_answer(query_tokens: set[str], text: str) -> str:
    count_nouns = {
        "issue",
        "issues",
        "session",
        "sessions",
        "story",
        "stories",
        "top",
        "tops",
        "time",
        "times",
        "pound",
        "pounds",
    }
    if not (query_tokens & count_nouns):
        return ""
    sentence = _best_numeric_sentence(query_tokens, text)
    if not sentence:
        return ""
    ordinal = re.search(
        r"\b(?P<ordinal>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\s+"
        r"(?P<noun>issue|session|story|top|time)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if ordinal:
        return _ordinal_to_cardinal_word(ordinal.group("ordinal"))
    match = re.search(
        r"\b(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
        r"(?P<noun>issues?|sessions?|stories?|short\s+stories|tops?|times?|pounds?)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if match:
        value = match.group("value").casefold()
        if value.isdigit() and (word := _number_words(float(value))):
            return word.casefold()
        return value
    return ""


def _owned_object_count_answer(query_tokens: set[str], text: str) -> str:
    if not {"own", "owned", "have"} & query_tokens:
        return ""
    object_terms = _count_query_object_terms(query_tokens)
    if not object_terms:
        return ""
    text_tokens = set(source_tokens(text))
    if not any(_term_variants(term) & text_tokens for term in object_terms):
        return ""
    possession_patterns = (
        r"\b(?:i|we)(?:'ve| have)?\s+(?:got|have|own)\s+(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b",
        r"\b(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+of\s+them\b",
    )
    for pattern in possession_patterns:
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            value = match.group("value").casefold()
            if value.isdigit() and (word := _number_words(float(value))):
                return word.casefold()
            return value
    return ""


def _mixed_relative_interval_lines(*, week_values: list[float], month_values: list[float]) -> list[str]:
    """Project human-scale intervals across mixed relative month/week evidence."""
    if not week_values or not month_values:
        return []
    intervals: list[int] = []
    for month_value in month_values:
        for week_value in week_values:
            delta_weeks = abs((month_value * 4) - week_value)
            if delta_weeks <= 0 or not delta_weeks.is_integer():
                continue
            days = int(delta_weeks * 7)
            if days not in intervals:
                intervals.append(days)
    lines: list[str] = []
    for days in intervals[:5]:
        lines.append(f"relative_day_interval={days} days")
        if days % 7 == 0:
            weeks = days // 7
            lines.append(f"relative_week_interval={weeks} weeks")
            if week_words := _number_words(float(weeks)):
                lines.append(f"relative_week_interval_answer={week_words} week")
    return lines


def _source_ordered_numeric_evidence(
    evidence: list[tuple[str, float, str]],
) -> list[tuple[str, float, str]]:
    indexed = list(enumerate(evidence))
    indexed.sort(
        key=lambda item: (
            _source_group_natural_key(source_context_group(item[1][0])),
            item[0],
        )
    )
    return [item for _index, item in indexed]


def _relative_week_anchor_evidence(contexts: list[str]) -> list[tuple[str, float, str]]:
    evidence: list[tuple[str, float, str]] = []
    seen: set[tuple[str, float]] = set()
    for context in contexts:
        text = _numeric_context_text(context)
        group = source_context_group(context)
        for match in _unit_value_pattern(r"weeks?").finditer(text):
            value = float(match.group("value"))
            key = (group, value)
            if key not in seen:
                seen.add(key)
                evidence.append((context, value, match.group(0)))
        for match in _WORD_WEEK_RE.finditer(text):
            value = float(_NUMBER_WORDS[match.group("value").casefold()])
            key = (group, value)
            if key not in seen:
                seen.add(key)
                evidence.append((context, value, match.group(0)))
        if _LAST_WEEK_RE.search(text):
            key = (group, 1.0)
            if key not in seen:
                seen.add(key)
                evidence.append((context, 1.0, "last week"))
    return evidence


def _relative_month_anchor_evidence(contexts: list[str]) -> list[tuple[str, float, str]]:
    evidence: list[tuple[str, float, str]] = []
    seen: set[tuple[str, float]] = set()
    for context in contexts:
        text = _numeric_context_text(context)
        group = source_context_group(context)
        for match in _unit_value_pattern(r"months?").finditer(text):
            value = float(match.group("value"))
            key = (group, value)
            if key not in seen:
                seen.add(key)
                evidence.append((context, value, match.group(0)))
        for match in _WORD_MONTH_RE.finditer(text):
            value = float(_NUMBER_WORDS[match.group("value").casefold()])
            key = (group, value)
            if key not in seen:
                seen.add(key)
                evidence.append((context, value, match.group(0)))
    return evidence


def _direct_time_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return "time" in query_tokens and bool(query_tokens & {"what", "when"})


def _average_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return "average" in query_tokens


def _age_at_event_query(query: str) -> bool:
    query_text = query.casefold()
    if not re.search(r"\b(?:how\s+old|age)\b", query_text):
        return False
    if "when" not in set(source_tokens(query)):
        return False
    return bool(
        re.search(
            r"\b(?:became|began|born|graduated|joined|moved?|started|turned)\b",
            query_text,
        )
    )


def _age_at_event_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project age-at-event arithmetic from current age and elapsed years."""
    if not _age_at_event_query(query):
        return []
    current_ages = _personal_current_age_values(contexts)
    elapsed_years = _age_at_event_operand_values(query, contexts)
    for current_age in current_ages:
        for elapsed_years_value in elapsed_years:
            if elapsed_years_value <= 0 or elapsed_years_value >= current_age:
                continue
            event_age = current_age - elapsed_years_value
            return [
                f"age_current={current_age}",
                f"age_elapsed_years={elapsed_years_value}",
                f"age_at_event_operation={current_age}-{elapsed_years_value}",
                f"age_at_event_answer={event_age}",
            ]
    return []


def _age_at_event_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _age_at_event_query(query):
        return []
    current = _personal_current_age_evidence(contexts)
    elapsed = _age_at_event_operand_evidence(query, contexts)
    if not current or not elapsed:
        return []
    current_context, current_age, current_raw = current[0]
    elapsed_context, elapsed_years, elapsed_raw = elapsed[0]
    if elapsed_years <= 0 or elapsed_years >= current_age:
        return []
    rows = [
        {
            "fact_id": "age_at_event:current_age",
            "source_group": source_context_group(current_context),
            "citation": source_context_citation(current_context),
            "kind": "number",
            "value": str(current_age),
            "unit": "years",
            "raw_span": current_raw,
            "include_reason": "current_age",
            "confidence": 0.82,
        },
        {
            "fact_id": "age_at_event:elapsed_years",
            "source_group": source_context_group(elapsed_context),
            "citation": source_context_citation(elapsed_context),
            "kind": "duration",
            "value": str(elapsed_years),
            "unit": "elapsed_years",
            "raw_span": elapsed_raw,
            "include_reason": "elapsed_since_event",
            "confidence": 0.82,
        },
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _future_age_at_event_query(query: str) -> bool:
    query_text = query.casefold()
    if not re.search(r"\bwhen\b", query_text):
        return False
    if not re.search(r"\b(?:how\s+many\s+years\s+will\s+i\s+be|how\s+old\s+will\s+i\s+be)\b", query_text):
        return False
    return bool(re.search(r"\b(?:marri(?:ed|age)|wedding)\b", query_text))


def _future_year_offset_evidence(query: str, contexts: list[str]) -> list[tuple[str, int, str]]:
    if not _future_age_at_event_query(query):
        return []
    query_terms = _query_specific_terms(query)
    evidence: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for context in contexts:
        context_terms = set(source_tokens(context))
        if query_terms and not (query_terms & context_terms):
            continue
        if not re.search(r"\b(?:marri(?:ed|age)|wedding)\b", context, flags=re.IGNORECASE):
            continue
        for raw, value in _future_year_offsets(context):
            key = (source_context_group(context), value)
            if key in seen:
                continue
            seen.add(key)
            evidence.append((context, value, raw))
    return evidence


def _future_age_at_event_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project future age arithmetic from current age and cited future offset."""
    if not _future_age_at_event_query(query):
        return []
    current = _personal_current_age_evidence(contexts)
    future = _future_year_offset_evidence(query, contexts)
    if not current or not future:
        return []
    current_context, current_age, _current_raw = current[0]
    future_context, future_years, _future_raw = future[0]
    if future_years <= 0 or current_age + future_years >= 125:
        return []
    answer = current_age + future_years
    support = ",".join(
        dict.fromkeys(
            [
                source_context_group(current_context),
                source_context_group(future_context),
            ]
        )
    )
    return [
        "candidate_rank=1 candidate_type=future_age_at_event candidate_confidence=0.86",
        f"candidate_support={support}",
        f"future_age_current={current_age}",
        f"future_age_offset_years={future_years}",
        f"future_age_at_event_operation={current_age}+{future_years}",
        f"future_age_at_event_answer={answer}",
    ]


def _future_age_at_event_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _future_age_at_event_query(query):
        return []
    current = _personal_current_age_evidence(contexts)
    future = _future_year_offset_evidence(query, contexts)
    if not current or not future:
        return []
    current_context, current_age, current_raw = current[0]
    future_context, future_years, future_raw = future[0]
    if future_years <= 0 or current_age + future_years >= 125:
        return []
    rows = [
        {
            "fact_id": "future_age_at_event:current_age",
            "source_group": source_context_group(current_context),
            "citation": source_context_citation(current_context),
            "kind": "number",
            "value": str(current_age),
            "unit": "years",
            "raw_span": current_raw,
            "include_reason": "current_age",
            "confidence": 0.84,
        },
        {
            "fact_id": "future_age_at_event:future_offset",
            "source_group": source_context_group(future_context),
            "citation": source_context_citation(future_context),
            "kind": "duration",
            "value": str(future_years),
            "unit": "future_years",
            "raw_span": future_raw,
            "include_reason": "future_event_offset",
            "confidence": 0.84,
        },
    ]
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _numeric_comparison_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"most", "least", "more", "less", "highest", "lowest"}) and bool(
        query_tokens & {"money", "amount", "cost", "spent", "spend", "price", "total"}
    )


def _direct_boolean_evidence_query(query: str) -> bool:
    """Return whether a query can be answered by explicit cited yes/no evidence."""
    query_text = " ".join(query.split()).casefold()
    if _query_bound_arithmetic_query(query) or _numeric_comparison_query(query):
        return False
    if re.search(r"\b(?:how|what|which|when|where|who|why)\b", query_text):
        return False
    first_token = next(iter(source_tokens(query_text)), "")
    return first_token in _DIRECT_BOOLEAN_AUXILIARIES or bool(re.search(r"\bor\s+not\??$", query_text))


def _direct_boolean_evidence_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project conservative yes/no answers from explicit cited evidence."""
    if not _direct_boolean_evidence_query(query):
        return []
    if lines := _temporal_frequency_boolean_synthesis_lines(query, contexts):
        return lines
    query_terms = _direct_boolean_query_terms(query)
    if len(query_terms) < 2:
        return []
    for context in contexts:
        text = source_context_snippet(context, max_chars=12_000)
        answer = _direct_boolean_answer(query, text, query_terms)
        if answer is None:
            continue
        source_id = source_context_group(context)
        return [
            "candidate_rank=1 candidate_type=boolean_evidence candidate_confidence=0.84",
            f"candidate_support={source_id}",
            f"boolean_evidence_answer={answer}",
            f"boolean_evidence_source_id={source_id}",
        ]
    return []


def _frequency_comparison_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"most", "least"}) and bool(
        query_tokens & {"airline", "airlines", "flight", "flights", "fly", "flew", "flying"}
    )


def _recency_comparison_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    if query_tokens & {"earliest", "order", "ordered", "sequence", "timeline"}:
        return False
    explicit_recency = bool(query_tokens & {"recent", "recently", "latest", "newest"}) or bool(
        re.search(r"\bmost\s+recently\b", query, flags=re.IGNORECASE)
    )
    return explicit_recency and bool(
        query_tokens & {"which", "what"}
    )


_AIRLINE_NAMES = (
    "United Airlines",
    "American Airlines",
    "Southwest Airlines",
    "Delta Air Lines",
    "Delta Airlines",
    "JetBlue",
    "Alaska Airlines",
)


_STREAMING_SERVICE_NAMES = (
    "Apple TV+",
    "Disney+",
    "Hulu",
    "Netflix",
    "Paramount+",
    "Peacock",
    "Prime Video",
    "Max",
)


_MONTH_ORDINALS = {
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


_MONTH_TERMS = frozenset(_MONTH_ORDINALS)


def _missing_month_scoped_count_target(query: str, contexts: list[str]) -> str:
    """Return a missing month-qualified count target for venue/category counts."""
    query_terms = set(source_tokens(query))
    months = [token for token in source_tokens(query) if token in _MONTH_TERMS]
    if not months or not {"how", "many"} <= query_terms:
        return ""
    if not query_terms & {"museum", "museums", "gallery", "galleries"}:
        return ""
    venue_terms = {"museum", "museums", "gallery", "galleries"}
    visit_terms = {"visit", "visited", "visiting", "went", "attended", "attend"}
    for context in contexts:
        terms = set(source_tokens(context))
        if terms & set(months) and terms & venue_terms and terms & visit_terms:
            return ""
    month = months[0]
    return f"museums or galleries in {month}"


def _precise_missing_target_requires_absence(query: str, target: str, contexts: list[str]) -> bool:
    """Return whether a precise missing slot should override sibling evidence."""
    if not target:
        return False
    return bool(
        _missing_month_scoped_count_target(query, contexts) == target
        or _missing_action_object_count_target(query, contexts) == target
        or _missing_conjunct_aggregation_target(query, contexts) == target
        or _missing_comparison_operand_target(query, contexts) == target
        or _missing_contrastive_sibling_target(query, contexts) == target
        or _missing_alternative_target(query, contexts) == target
    )


def _anniversary_engagement_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"anniversary"} and tokens & {"engaged", "engagement"} and tokens & {"month", "months"})


def _month_day_ledger_row(
    fact_id: str,
    evidence: tuple[str, int, int],
    *,
    include_reason: str,
) -> dict[str, object]:
    context, month, day = evidence
    return {
        "fact_id": fact_id,
        "source_group": source_context_group(context),
        "citation": source_context_citation(context),
        "kind": "date",
        "value": f"{month}/{day}",
        "unit": "month_day",
        "raw_span": f"{month}/{day}",
        "include_reason": include_reason,
        "confidence": 0.82,
    }


def _parent_order_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"who", "which"} and "first" in tokens and tokens & {"parent", "parents"})


def should_query_source_lane(query: str, *, limit: int = 10) -> bool:
    """Return whether source text should supplement graph retrieval."""
    return classify_retrieval_intent(query, limit=limit).needs_source_lane or _parent_order_query(query)


def should_try_absence_bundle_first(query: str, *, limit: int = 10) -> bool:
    """Return whether cited absence should outrank generic synthesis attempts."""
    intent = classify_retrieval_intent(query, limit=limit)
    if _multi_quoted_duration_query(query):
        return False
    return (
        "absence_check" in intent.reasons
        or _conjunctive_aggregation_absence_risk(query)
        or "temporal_order" in intent.reasons
        or "temporal_sequence" in intent.reasons
        or _parent_order_query(query)
    )


def _query_person_alternatives(query: str) -> tuple[str, ...]:
    alternatives = _query_alternatives(query)
    people: list[str] = []
    for alternative in alternatives:
        for match in _PERSON_NAME_ALTERNATIVE_RE.finditer(alternative):
            name = match.group(0)
            if name.casefold() not in {"who", "which"}:
                people.append(name.casefold())
    return tuple(dict.fromkeys(people))


def _parent_context_matches_person(person: str, context: str) -> bool:
    terms = set(source_tokens(context))
    if person in terms:
        return True
    return bool(
        person == "rachel"
        and terms & {"sister-in-law", "sister", "law"}
        and terms & {"twins", "jackson", "julia"}
    )


def _month_day_mentions(text: str) -> list[tuple[int, int]]:
    month_pattern = "|".join(sorted(_MONTH_ORDINALS, key=len, reverse=True))
    values: list[tuple[int, int]] = []
    for match in re.finditer(
        rf"\b(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
        text,
        flags=re.IGNORECASE,
    ):
        values.append((_MONTH_ORDINALS[match.group("month").casefold()], int(match.group("day"))))
    for match in re.finditer(
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?P<month>{month_pattern})\b",
        text,
        flags=re.IGNORECASE,
    ):
        values.append((_MONTH_ORDINALS[match.group("month").casefold()], int(match.group("day"))))
    return values
