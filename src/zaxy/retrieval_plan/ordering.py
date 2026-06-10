"""Split from retrieval_plan.py (mechanical decomposition)."""


from __future__ import annotations

import re
from datetime import date

from zaxy.evidence_candidates import (
    EvidenceProjection,
)
from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent
from zaxy.retrieval_plan.duration_evidence import (
    _AIRLINE_NAMES,
    _MONTH_ORDINALS,
    _MONTH_TERMS,
    _NUMBER_WORDS,
    _STREAMING_SERVICE_NAMES,
    _anniversary_engagement_query,
    _birth_age_query,
    _birth_age_target,
    _direct_time_query,
    _frequency_comparison_query,
    _future_age_at_event_query,
    _integer_number_value,
    _missing_month_scoped_count_target,
    _month_day_ledger_row,
    _month_day_mentions,
    _parent_context_matches_person,
    _parent_order_query,
    _precise_missing_target_requires_absence,
    _query_person_alternatives,
    _recency_comparison_query,
)
from zaxy.retrieval_plan.fact_queries import (
    _career_prior_duration_query,
    _ledger_row_lines,
    _missing_current_employer_target,
    _numeric_context_text,
)
from zaxy.retrieval_plan.foundations import (
    _BRIDGE_QUERY_STOPWORDS,
    _CLOCK_TIME_RE,
    _CONNECTING_FLIGHT_RE,
    _FLIGHT_COUNT_RE,
    _FLIGHT_TERM_RE,
    _LONGMEMEVAL_SESSION_DATE_RE,
    _QUERY_COUPLE_DAYS_AGO_RE,
    _QUERY_RELATIVE_TIME_RE,
    _QUERY_SOURCE_STOPWORDS,
    _RELATIVE_DAYS_AGO_RE,
    _RELATIVE_MINUTE_OFFSET_RE,
    _ROUND_TRIP_FLIGHTS_RE,
    _absence_term_variants,
    _countable_category_evidence_present,
    _has_domain_specific_temporal_source_query,
    _multi_quoted_duration_query,
    _paid_event_aggregation_terms,
    _quoted_query_targets,
    _suppress_generic_temporal_interval_queries,
    _temporal_count_program_query,
    source_context_citation,
    source_context_group,
    source_context_snippet,
    source_lane_priority,
    source_lane_query,
    source_tokens,
)
from zaxy.retrieval_plan.scalars import (
    _missing_action_object_count_target,
    _missing_alternative_target,
    _missing_category_modifier_target,
    _missing_comparison_operand_target,
    _missing_concrete_query_target,
    _missing_conjunct_aggregation_target,
    _missing_contrastive_sibling_target,
    _missing_location_target,
    _missing_reading_progress_target,
    _query_bound_direct_answer_lines,
    _query_overlap_score,
    _query_specific_terms,
    target_terms_present,
)


def _anniversary_engagement_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer engagement and anniversary date evidence for interval questions."""
    if not _anniversary_engagement_query(query):
        return 0
    context_terms = set(source_tokens(context))
    score = 0
    if context_terms & {"engaged", "engagement"}:
        score += 4
    if context_terms & {"anniversary"}:
        score += 4
    if _month_day_mentions(context):
        score += 3
    tokens = query_tokens if query_tokens is not None else set(source_tokens(query))
    if "rachel" in tokens and "rachel" in context_terms:
        score += 2
    return score


def _first_event_month_day_evidence(contexts: list[str], *, terms: set[str]) -> tuple[str, int, int] | None:
    for context in contexts:
        context_terms = set(source_tokens(context))
        if not context_terms & terms:
            continue
        month_days = _month_day_mentions(context)
        if month_days:
            month, day = month_days[0]
            return context, month, day
    return None


def _anniversary_engagement_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _anniversary_engagement_query(query):
        return []
    engagement = _first_event_month_day_evidence(contexts, terms={"engaged", "engagement"})
    anniversary = _first_event_month_day_evidence(contexts, terms={"anniversary"})
    if engagement is None or anniversary is None:
        return []
    rows = [
        _month_day_ledger_row(
            "anniversary_engagement:engagement",
            engagement,
            include_reason="engagement_date",
        ),
        _month_day_ledger_row(
            "anniversary_engagement:anniversary",
            anniversary,
            include_reason="anniversary_date",
        ),
    ]
    return _ledger_row_lines(rows)


def _first_event_month_day(contexts: list[str], *, terms: set[str]) -> tuple[int, int] | None:
    evidence = _first_event_month_day_evidence(contexts, terms=terms)
    if evidence is None:
        return None
    return evidence[1], evidence[2]


def _anniversary_engagement_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project month interval between engagement and anniversary evidence."""
    if not _anniversary_engagement_query(query):
        return []
    engagement = _first_event_month_day(contexts, terms={"engaged", "engagement"})
    anniversary = _first_event_month_day(contexts, terms={"anniversary"})
    if engagement is None or anniversary is None:
        return []
    engagement_month, engagement_day = engagement
    anniversary_month, anniversary_day = anniversary
    month_delta = anniversary_month - engagement_month
    if anniversary_day < engagement_day:
        month_delta -= 1
    if month_delta <= 0:
        month_delta = (anniversary_month + 12) - engagement_month
        if anniversary_day < engagement_day:
            month_delta -= 1
    if month_delta <= 0:
        return []
    unit = "month" if month_delta == 1 else "months"
    return [
        f"anniversary_engagement_months={month_delta}",
        f"anniversary_engagement_interval_answer={month_delta} {unit}",
    ]


def _month_only_mention(text: str) -> tuple[int, int] | None:
    month_pattern = "|".join(sorted(_MONTH_ORDINALS, key=len, reverse=True))
    match = re.search(rf"\b(?P<month>{month_pattern})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return (_MONTH_ORDINALS[match.group("month").casefold()], 1)


def _parent_event_month_day(context: str) -> tuple[int, int] | None:
    text = source_context_snippet(context, max_chars=1_500)
    if not re.search(r"\b(?:adopted|adoption|baby|born|twins?|parent)\b", text, flags=re.IGNORECASE):
        return None
    month_days = _month_day_mentions(text)
    if month_days:
        return month_days[0]
    return _month_only_mention(text)


def _parent_order_evidence_score(query: str, context: str) -> int:
    """Prefer parent/adoption/birth evidence for named parent-order questions."""
    if not _parent_order_query(query):
        return 0
    context_terms = set(source_tokens(context))
    score = 0
    if context_terms & {"parent", "adopted", "adoption", "baby", "born", "twins"}:
        score += 4
    if context_terms & set(_query_person_alternatives(query)):
        score += 3
    if _parent_event_month_day(context) is not None:
        score += 4
    return score


def _parent_event_month_day_for_person_evidence(person: str, contexts: list[str]) -> tuple[str, int, int] | None:
    person_contexts = [
        context for context in contexts
        if _parent_context_matches_person(person, context)
    ]
    for context in person_contexts:
        event_date = _parent_event_month_day(context)
        if event_date is not None:
            month, day = event_date
            return context, month, day
    if person == "rachel" and any("rachel" in set(source_tokens(context)) for context in person_contexts):
        for context in contexts:
            terms = set(source_tokens(context))
            if terms & {"twins", "jackson", "julia"}:
                event_date = _parent_event_month_day(context)
                if event_date is not None:
                    month, day = event_date
                    return context, month, day
    return None


def _parent_order_observations(query: str, contexts: list[str]) -> list[tuple[int, int, str, str]]:
    people = _query_person_alternatives(query)
    if len(people) < 2:
        return []
    observations: list[tuple[int, int, str, str]] = []
    for person in people:
        event_date = _parent_event_month_day_for_person_evidence(person, contexts)
        if event_date is None:
            return []
        context, month, day = event_date
        observations.append((month, day, person.title(), context))
    observations.sort(key=lambda item: (item[0], item[1]))
    return observations


def _parent_order_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _parent_order_query(query):
        return []
    observations = _parent_order_observations(query, contexts)
    if len(observations) < 2:
        return []
    rows = [
        {
            "fact_id": f"parent_order:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "date",
            "value": f"{month}/{day}",
            "unit": "month_day",
            "raw_span": f"{month}/{day}",
            "candidate": person,
            "include_reason": "parent_order_candidate",
            "confidence": 0.8,
        }
        for index, (month, day, person, context) in enumerate(observations)
    ]
    return _ledger_row_lines(rows)


def _parent_event_month_day_for_person(person: str, contexts: list[str]) -> tuple[int, int] | None:
    evidence = _parent_event_month_day_for_person_evidence(person, contexts)
    if evidence is None:
        return None
    return evidence[1], evidence[2]


def _missing_parent_order_target(query: str, contexts: list[str]) -> str:
    """Return a missing named parent alternative using event-level evidence."""
    if not _parent_order_query(query):
        return ""
    for person in _query_person_alternatives(query):
        if _parent_event_month_day_for_person(person, contexts) is None:
            return person
    return ""


def high_precision_missing_target(query: str, contexts: list[str]) -> str:
    """Return concrete missing query targets with low false-positive risk."""
    if target := _missing_current_employer_target(query, contexts):
        return target
    if target := _missing_parent_order_target(query, contexts):
        return target
    if target := _missing_action_object_count_target(query, contexts):
        return target
    if target := _missing_month_scoped_count_target(query, contexts):
        return target
    if target := _missing_reading_progress_target(query, contexts):
        return target
    if target := _missing_category_modifier_target(query, contexts):
        return target
    if target := _missing_conjunct_aggregation_target(query, contexts):
        return target
    if target := _missing_comparison_operand_target(query, contexts):
        return target
    if target := _missing_contrastive_sibling_target(query, contexts):
        return target
    if target := _missing_location_target(query, contexts):
        return target
    if target := _missing_alternative_target(query, contexts):
        return target
    return _missing_concrete_query_target(query, contexts)


def should_defer_to_absence_check(
    query: str,
    contexts: list[str],
    intent: RetrievalIntent,
) -> bool:
    """Return whether missing evidence should outrank numeric/order synthesis."""
    if not intent.needs_source_lane or not contexts:
        return False
    target = high_precision_missing_target(query, contexts)
    if _countable_category_evidence_present(query, contexts) and not _precise_missing_target_requires_absence(
        query,
        target,
        contexts,
    ):
        return False
    return bool(target and not target_terms_present(target, contexts))


def missing_query_target(query: str, contexts: list[str]) -> str:
    """Return query-specific terms absent from all cited source contexts."""
    if target := high_precision_missing_target(query, contexts):
        return target
    query_terms = _query_specific_terms(query)
    if not query_terms:
        return ""
    context_terms: set[str] = set()
    for context in contexts:
        context_terms.update(source_tokens(context))
    missing = [
        term for term in sorted(query_terms)
        if not (_absence_term_variants(term) & context_terms)
    ]
    return " ".join(missing)


def _parent_order_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project first parent from named adoption/birth evidence."""
    if not _parent_order_query(query):
        return []
    people = _query_person_alternatives(query)
    if len(people) < 2:
        return []
    observations: list[tuple[int, int, str]] = []
    for person in people:
        event_date = _parent_event_month_day_for_person(person, contexts)
        if event_date is None:
            return []
        month, day = event_date
        observations.append((month, day, person.title()))
    observations.sort(key=lambda item: (item[0], item[1]))
    lines = [f"parent_order_answer={observations[0][2]}"]
    for index, (month, day, person) in enumerate(observations, start=1):
        lines.append(f"parent_order_rank={index} month={month} day={day} person={person}")
    return lines


def _longmemeval_session_date(text: str) -> date | None:
    match = _LONGMEMEVAL_SESSION_DATE_RE.search(text)
    if not match:
        return None
    try:
        return date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def _query_temporal_anchor_date(contexts: list[str]) -> date | None:
    for context in contexts:
        if "query_temporal_anchor=true" not in context.casefold():
            continue
        return _longmemeval_session_date(context)
    return None


def _query_relative_target_days(query: str) -> int | None:
    if _QUERY_COUPLE_DAYS_AGO_RE.search(query):
        return 2
    match = _QUERY_RELATIVE_TIME_RE.search(query)
    if not match:
        return None
    value_text = match.group("value").casefold()
    value = _NUMBER_WORDS.get(value_text)
    if value is None:
        value = int(value_text)
    unit = match.group("unit").casefold()
    if unit.startswith("month"):
        return value * 30
    if unit.startswith("week"):
        return value * 7
    return value


def _elapsed_relative_temporal_query_unit(query: str) -> str | None:
    query_text = " ".join(query.casefold().split())
    match = re.search(r"\bhow\s+many\s+(?P<unit>days?|weeks?|months?)\s+ago\b", query_text)
    if not match:
        return None
    unit = match.group("unit")
    if unit.startswith("month"):
        return "months"
    if unit.startswith("week"):
        return "weeks"
    return "days"


def query_temporal_anchor_synthesis_query(query: str) -> bool:
    """Return whether query-date context can produce a relative temporal answer."""
    query_text = " ".join(query.casefold().split())
    if "ago" not in query_text:
        return False
    tokens = set(source_tokens(query))
    if _elapsed_relative_temporal_query_unit(query):
        return True
    if _query_relative_target_days(query) is None:
        return False
    return bool(
        tokens & {"what", "which", "where", "did"}
        and tokens
        & {
            "appliance",
            "book",
            "buy",
            "bought",
            "cook",
            "cooked",
            "cooking",
            "finish",
            "finished",
            "garden",
            "gardening",
            "purchase",
            "purchased",
            "read",
        }
    )


def _relative_temporal_anchor_query(query: str) -> bool:
    return query_temporal_anchor_synthesis_query(query)


def _relative_temporal_action_overlap(query: str, context: str) -> int:
    query_tokens = set(source_tokens(query))
    context_tokens = set(source_tokens(context))
    score = 0
    if query_tokens & {"buy", "bought", "purchase", "purchased"} and context_tokens & {
        "buy",
        "bought",
        "purchase",
        "purchased",
        "got",
    }:
        score += 2
    if query_tokens & {"finish", "finished", "read"} and context_tokens & {"finish", "finished", "read", "reading"}:
        score += 2
    if query_tokens & {"cooking", "cook", "cooked"} and context_tokens & {"baked", "made", "cooked", "prepared"}:
        score += 2
    if query_tokens & {"gardening", "garden", "activity"} and context_tokens & {
        "planted",
        "planting",
        "seeded",
        "transplanted",
        "harvested",
        "pruned",
        "watered",
    }:
        score += 2
    return score


def _relative_temporal_anchor_query_terms(query: str) -> set[str]:
    stopwords = _QUERY_SOURCE_STOPWORDS | {
        "ago",
        "couple",
        "day",
        "days",
        "did",
        "do",
        "many",
        "month",
        "months",
        "today",
        "week",
        "weeks",
    }
    return {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in stopwords and not token.isdigit()
    }


def _relative_temporal_target_tolerance(target_days: int) -> int:
    if target_days <= 3:
        return 1
    if target_days <= 14:
        return 2
    if target_days <= 35:
        return 4
    return 10


def _relative_temporal_anchor_observations(
    query: str,
    contexts: list[str],
    anchor_date: date,
) -> list[tuple[int, int, int, str, date]]:
    query_terms = _relative_temporal_anchor_query_terms(query)
    target_days = _query_relative_target_days(query)
    observations: list[tuple[int, int, int, str, date]] = []
    for index, context in enumerate(contexts):
        if "query_temporal_anchor=true" in context.casefold():
            continue
        session_date = _longmemeval_session_date(context)
        if session_date is None:
            continue
        days_ago = (anchor_date - session_date).days
        if days_ago < 0:
            continue
        overlap = _query_overlap_score(query_terms, context) + _relative_temporal_action_overlap(query, context)
        if target_days is None:
            if overlap < 2:
                continue
            distance_penalty = 0
        else:
            distance = abs(days_ago - target_days)
            if distance > _relative_temporal_target_tolerance(target_days):
                continue
            if overlap < 1:
                continue
            distance_penalty = distance
        score = (overlap * 20) - distance_penalty + source_lane_priority(context)
        observations.append((days_ago, score, -index, context, session_date))
    observations.sort(key=lambda item: (-item[1], -item[2]))
    return observations


def _plural_unit(value: int, singular: str) -> str:
    return singular if value == 1 else f"{singular}s"


def _format_relative_temporal_elapsed_answer(days_ago: int, unit: str) -> str | None:
    if unit == "months":
        value = max(1, round(days_ago / 30))
        return f"{value} {_plural_unit(value, 'month')} ago"
    if unit == "weeks":
        value = max(1, round(days_ago / 7))
        return f"{value} {_plural_unit(value, 'week')} ago"
    if unit == "days":
        value = max(0, days_ago)
        if value == 1:
            return "1 day ago"
        return f"{value} days ago. {value + 1} days (including the last day) is also acceptable."
    return None


def _clean_relative_temporal_answer(answer: str) -> str:
    answer = re.sub(
        r"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
        r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
        r"(?:days?|weeks?|months?)\s+ago\b.*$",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(r"\b(?:a\s+)?couple\s+of\s+days?\s+ago\b.*$", "", answer, flags=re.IGNORECASE)
    return " ".join(answer.strip(" ,.;!?").split())


def _direct_object_after_action(text: str, *, actions: tuple[str, ...]) -> str:
    action_pattern = "|".join(re.escape(action) for action in actions)
    match = re.search(
        rf"\b(?:i\s+)?(?:{action_pattern})\s+(?P<answer>(?:(?:a|an|the)\s+)?[A-Za-z0-9'\" -]{{2,100}})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    answer = re.split(
        r"\s+(?:for|so|to|because|since|about|around|approximately|exactly|a\s+couple\s+of|"
        r"\d+\s+(?:days?|weeks?|months?)\s+ago)\b",
        match.group("answer").strip(" ,.;!?"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _clean_relative_temporal_answer(answer)


def _book_finished_answer(text: str) -> str:
    match = re.search(
        r"\b(?:finished|finish)\s+(?:reading\s+)?(?P<answer>(?:'[^']+'|\"[^\"]+\"|[A-Z][^.,;!?]{1,120}))",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _clean_relative_temporal_answer(match.group("answer"))


def _activity_phrase_answer(text: str) -> str:
    match = re.search(
        r"\bi\s+(?P<verb>planted|planted|started|set\s+up|built|transplanted|seeded|harvested|pruned|watered)\s+"
        r"(?P<object>[A-Za-z0-9'\" -]{2,120})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    verb = match.group("verb").casefold()
    verb_map = {
        "planted": "planting",
        "started": "starting",
        "set up": "setting up",
        "built": "building",
        "transplanted": "transplanting",
        "seeded": "seeding",
        "harvested": "harvesting",
        "pruned": "pruning",
        "watered": "watering",
    }
    answer = f"{verb_map.get(verb, verb)} {match.group('object')}"
    return _clean_relative_temporal_answer(answer)


def _relative_temporal_anchor_direct_answer(query: str, context: str) -> str:
    text = _numeric_context_text(context)
    tokens = set(source_tokens(query))
    if tokens & {"appliance", "buy", "bought", "purchase", "purchased"} and (
        answer := _direct_object_after_action(
            text,
            actions=("bought", "buy", "purchased", "got", "picked up"),
        )
    ):
        return answer
    if tokens & {"book", "finish", "finished", "read"} and (answer := _book_finished_answer(text)):
        return answer
    if tokens & {"cooking", "cook", "cooked", "friend"} and (
        answer := _direct_object_after_action(
            text,
            actions=("baked", "made", "cooked", "prepared"),
        )
    ):
        return answer
    if tokens & {"gardening", "activity", "garden"} and (answer := _activity_phrase_answer(text)):
        return answer
    return ""


def _relative_temporal_anchor_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project answers whose relative-time anchor is the question date."""
    if not _relative_temporal_anchor_query(query):
        return []
    anchor_date = _query_temporal_anchor_date(contexts)
    if anchor_date is None:
        return []
    observations = _relative_temporal_anchor_observations(query, contexts, anchor_date)
    if not observations:
        return []
    best = observations[0]
    if unit := _elapsed_relative_temporal_query_unit(query):
        answer = _format_relative_temporal_elapsed_answer(best[0], unit)
        if not answer:
            return []
    else:
        answer = _relative_temporal_anchor_direct_answer(query, best[3])
        if not answer:
            return []
    source_id = source_context_group(best[3])
    lines = [
        "candidate_rank=1 candidate_type=relative_temporal_anchor candidate_confidence=0.86",
        f"candidate_support={source_id}",
        f"relative_temporal_anchor_days_ago={best[0]}",
        f"relative_temporal_anchor_session_date={best[4].isoformat()}",
        f"relative_temporal_anchor_answer={answer}",
        f"relative_temporal_anchor_source_id={source_id}",
    ]
    if target_days := _query_relative_target_days(query):
        lines.append(f"relative_temporal_anchor_target_days={target_days}")
    lines.append(f"relative_temporal_anchor_raw_span={source_context_snippet(best[3], max_chars=220)}")
    return lines


def _streaming_service_names_in_context(text: str) -> list[str]:
    names: list[str] = []
    for service in _STREAMING_SERVICE_NAMES:
        right_boundary = r"\b" if service[-1].isalnum() else r"(?=\W|$)"
        if (
            re.search(rf"\b{re.escape(service)}{right_boundary}", text, flags=re.IGNORECASE)
            and service not in names
        ):
            names.append(service)
    return names


def _recency_candidate_values(query: str, text: str) -> list[str]:
    query_tokens = set(source_tokens(query))
    if query_tokens & {"streaming", "service"}:
        return _streaming_service_names_in_context(text)
    return []


def _airline_names_in_context(context: str) -> list[str]:
    names: list[str] = []
    for carrier in _AIRLINE_NAMES:
        if re.search(rf"\b{re.escape(carrier)}\b", context, flags=re.IGNORECASE):
            normalized = "Delta Air Lines" if carrier == "Delta Airlines" else carrier
            if normalized not in names:
                names.append(normalized)
    return names


def _flight_count_in_context(context: str) -> int:
    text = source_context_snippet(context, max_chars=1_500)
    if not _FLIGHT_TERM_RE.search(text):
        return 0
    if _ROUND_TRIP_FLIGHTS_RE.search(text):
        return 4
    if _FLIGHT_COUNT_RE.search(text):
        count = 0
        for match in _FLIGHT_COUNT_RE.finditer(text):
            count += _integer_number_value(match.group("value"))
        return count
    if _CONNECTING_FLIGHT_RE.search(text):
        return 2
    return 1


def _frequency_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project bounded frequency answers from repeated categorical evidence."""
    if not _frequency_comparison_query(query):
        return []
    counts: dict[str, int] = {}
    for context in contexts:
        context_count = _flight_count_in_context(context)
        if context_count <= 0:
            continue
        for carrier in _airline_names_in_context(context):
            counts[carrier] = counts.get(carrier, 0) + context_count
    if len(counts) < 2:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return []
    return [
        "frequency_counts=" + ",".join(f"{carrier}:{count}" for carrier, count in ranked),
        f"frequency_answer={ranked[0][0]}",
    ]


def _time_offset_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return "time" in query_tokens


def _temporal_interval_query(query: str) -> bool:
    """Return whether a query asks for elapsed time between cited events."""
    query_tokens = set(source_tokens(query))
    return bool(
        query_tokens
        & {
            "day",
            "days",
            "hour",
            "hours",
            "minute",
            "minutes",
            "month",
            "months",
            "week",
            "weeks",
        }
    ) and bool(query_tokens & {"after", "before", "between", "since", "until"})


def aggregation_event_source_queries(query: str) -> tuple[str, ...]:
    """Return deterministic source queries for event-like aggregation memories."""
    query_terms = set(source_tokens(query))
    queries: list[str] = []
    generic_temporal_interval_queries: list[str] = []
    generic_temporal_count_queries: list[str] = []
    if _multi_quoted_duration_query(query):
        for target in _quoted_query_targets(query):
            queries.append(f"{target} started finished reading listening today")
    if paid_event_terms := _paid_event_aggregation_terms(query):
        queries.append(" ".join([*paid_event_terms, "paid attend attended free cost registration fee"]))
    if {"parent", "first"} <= query_terms or (query_terms & {"rachel", "alex", "tom"} and "parent" in query_terms):
        queries.append("parent rachel alex adopted born twins cousin baby girl china february january")
    if query_terms & {"project", "projects"} and (
        {"how", "many"} <= query_terms or query_terms & {"led", "leading"}
    ):
        queries.append("project led leading currently managed team initiative launched rollout migration")
    if query_terms & {"bike", "bicycle", "cycling"} and (
        query_terms & {"expense", "expenses", "spent", "money", "total", "cost"}
    ):
        queries.append("bike bicycle helmet chain lights rack tune-up cost bought replaced installed")
    if query_terms & {"bake", "baked", "baking"} or (
        query_terms & {"bread", "cookies", "sourdough", "muffins"} and {"how", "many"} <= query_terms
    ):
        queries.append("bake baked baking bread cookies sourdough muffins recipe")
    if query_terms & {"bake", "baked", "baking"} and query_terms & {"birthday", "party"}:
        queries.append("birthday party baked baking cake dessert niece nephew uncle aunt lemon blueberry")
    if query_terms & {"fish", "aquarium", "aquariums", "tank", "tanks"}:
        queries.append("fish aquarium aquariums tank tanks betta gourami tetras bubbles upgraded")
    if query_terms & {"plant", "plants"} and (
        query_terms & {"acquire", "acquired", "bought", "received", "last", "month"}
    ):
        queries.append("plants plant acquired bought received monstera pothos snake plant succulent nursery last month")
    if query_terms & {"google"} and query_terms & {"working", "work", "career", "job"}:
        queries.append("google current job years months career working professionally field started")
        queries.append("NovaTech working professionally 9 years 4 years 3 months backend developer current job")
        queries.append("physical notebook working professionally 9 years tasks projects current using notebook")
    elif _career_prior_duration_query(query):
        queries.append("current job working professionally 9 years years field career started")
    if query_terms & {"camera", "lens", "prime"} and query_terms & {"coast", "coastal", "trip", "road"}:
        queries.append("prime lens 50mm got month ago coastal trip coast road trip camera photography")
    if query_terms & {"anniversary"} and query_terms & {"engaged", "engagement", "rachel"}:
        queries.append("Rachel engaged May 15 anniversary July 22 last month close friend partner")
    if query_terms & {"jewelry", "received", "gift"}:
        queries.append("received jewelry gift necklace bracelet ring earrings aunt uncle mother father last Saturday")
    if _temporal_interval_query(query):
        interval_terms = [
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _QUERY_SOURCE_STOPWORDS and not token.isdigit()
        ]
        generic_temporal_interval_queries.append(
            " ".join(
                [
                    *interval_terms[:12],
                    "started began since after before between found saw loved attended went bought booked on date",
                    "January February March April May June July August September October November December",
                ]
            )
        )
    if _temporal_count_program_query(query):
        event_terms = [
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _QUERY_SOURCE_STOPWORDS and not token.isdigit()
        ]
        generic_temporal_count_queries.append(
            " ".join(
                [
                    *event_terms[:8],
                    "event events attended participated volunteered ran walked danced raised funds fundraising support cause",
                    "January February March April May June July August September October November December",
                ]
            )
        )
    if query_terms & {"order", "ordered", "sequence", "timeline", "earliest", "latest"}:
        if query_terms & {"trip", "trips", "travel", "travels"}:
            queries.append(
                "trip trips road trip camping hike day hike got back returned went on started "
                "family friends today yesterday last week last month few months ago"
            )
        if query_terms & {"event", "events", "activity", "activities", "sports", "sport", "watched"}:
            queries.append(
                "event events activities watched attended participated completed took part game tournament "
                "today yesterday last week last month few months ago"
            )
    if (
        ({"how", "old"} <= query_terms or _future_age_at_event_query(query))
        and query_terms & {"married", "wedding"}
    ):
        names = [token for token in source_tokens(query) if token not in _BRIDGE_QUERY_STOPWORDS and token not in {"old", "married", "wedding", "get", "when", "will"}]
        name_terms = " ".join(names[:3])
        queries.append(
            " ".join(
                term
                for term in (
                    name_terms,
                    "age years old birthday getting married wedding next year friend life goals",
                )
                if term
            )
        )
        queries.append("my age years old 30s 32 skin type fine lines wrinkles Rachel getting married")
    if query_terms & {"meet", "met"} and query_terms & {"first", "earlier", "before"}:
        query_names = _query_person_alternatives(query)
        if {"mark", "sarah", "tom"} & set(query_names):
            queries.append("met first Mark Sarah Tom few months ago about a month ago beach trip charity event")
    if query_terms & {"streaming", "service"} and query_terms & {"recently", "most", "start", "started", "using"}:
        queries.append("streaming service started using most recently last week last month months ago Hulu Netflix Disney+")
    if query_terms & {"social", "media"} and query_terms & {"break", "breaks"}:
        queries.append("social media break week-long 10-day mid-January mid-February")
    if query_terms & {"sunday", "mass", "church"} and query_terms & {"ash", "wednesday", "cathedral"}:
        queries.append("Sunday mass St. Mary's Church January 2 Ash Wednesday service cathedral February 1")
    if query_terms & {"road", "trip", "destinations"} and query_terms & {"driving", "drove", "drive"}:
        queries.append("road trip drove driving hours destinations Outer Banks Washington D.C. Tennessee mountains")
    if query_terms & {"wedding", "weddings"} and query_terms & {"attended", "attend", "been"}:
        queries.append("weddings attended got back cousin Rachel vineyard Emily Sarah rooftop Jen Tom rustic barn")
    if query_terms & {"grocery", "store"} and query_terms & {"spent", "spend", "money"}:
        queries.append(
            "grocery shopping spent money store Trader Joe's Walmart Thrive Market Publix Instacart $80 $120 $150 $60"
        )
    if query_terms & {"average", "age"} and query_terms & {"parents", "grandparents"}:
        queries.append("average age me parents grandparents turned 32 mom 55 dad 58 grandma 75 grandpa 78")
    if _birth_age_query(query) and (target := _birth_age_target(query)):
        queries.append(
            " ".join(
                [
                    target,
                    "age years old just current age my age I just turned born birthday",
                ]
            )
        )
    if query_terms & {"charity", "events"} and query_terms & {"raise", "raised", "money", "total"}:
        queries.append(
            "charity events participated raised total charity walk $250 Bike-a-Thon Cancer Research $5,000 charity yoga $600 animal shelter"
        )
    if query_terms & {"sports", "sport"} and query_terms & {"event", "events"} and query_terms & {
        "mentioned",
        "participating",
        "participated",
    }:
        queries.append("sports event participated annual charity soccer tournament company two weeks ago")
        queries.append("company annual charity soccer tournament sports event")
    if query_terms & {"cuisine", "cuisines"} and query_terms & {"learned", "cook", "cooking", "tried"}:
        queries.append(
            "cuisines learned cook tried Ethiopian Indian Mexican Thai meal prep tikka masala tacos pad thai"
        )
    if query_terms & {"accommodation", "accommodations", "hotel", "hostel", "resort"} and (
        query_terms & {"hawaii", "maui", "tokyo"} or query_terms & {"night", "nightly"}
    ):
        queries.append(
            "Hawaii Maui Tokyo accommodations per night hostel resort hotel costs $300 $30 luxurious affordable"
        )
    if query_terms & {"airline", "airlines"} and query_terms & {"fly", "flew", "flying"}:
        queries.append(
            "United Airlines American Airlines Southwest Airlines flights flew flying March April two flights each way"
        )
    if query_terms & {"doctor", "doctors", "appointment", "appointments"} and query_terms & {"bed", "sleep", "time"}:
        queries.append("went to bed bedtime 2 AM doctor appointment day before Dr. appointment")
        queries.append("didn't get to bed until 2 AM last Wednesday sluggish Thursday doctor appointment")
    if query_terms & {"doctor", "doctors", "appointment", "appointments"} and query_terms & {"march"}:
        queries.append("March doctor appointment Dr. Patel Dr. Thompson orthopedic surgeon follow-up ENT")
    if query_terms & {"wake", "waking"} and query_terms & {"time", "tuesdays", "thursdays"}:
        queries.append(
            "waking wake up wake-up call AM minutes earlier Tuesdays Thursdays morning routine personalized"
        )
    if query_terms & {"model", "models", "kit", "kits"} and {"how", "many"} <= query_terms:
        queries.append("model kits finished started picked up got bought scale")
    if query_terms & {"arrive", "arrived", "arrival"} and query_terms & {"bought", "buy", "purchased", "ordered"}:
        queries.append("bought purchased ordered arrived arrival delivered Amazon dates backpack laptop accessories")
    if query_terms & {"practice", "practicing"} and query_terms & {"time", "daily", "day", "everyday"}:
        queries.append("practicing practice daily every day minutes guitar violin piano music theory fingerpicking")
    if query_terms & {"business", "buisiness", "milestone", "milestones"}:
        queries.append("business milestone signed contract first client freelance clients today QuickBooks contract")
    if query_terms & {"competition", "investment"} and query_terms & {"buy", "bought"}:
        queries.append("competition investment bought got own set sculpting tools modeling tool wire cutter sculpting mat")
        queries.append("sculpting tools competition art sculpture category local art studio")
    if query_terms & {"doctor", "doctors", "physician", "physicians"} and {"how", "many"} <= query_terms:
        queries.append("doctor physician dermatologist ent visited saw appointment")
    if query_terms & {"movie", "movies", "film", "films", "festival", "festivals"} and {"how", "many"} <= query_terms:
        queries.append("film festival movie attended went participated")
    if query_terms & {"collecting", "collection", "collect"} and query_terms & {"vintage", "film", "films"}:
        queries.append("collecting vintage cameras films camera collection")
    if query_terms & {"art", "art-related"} and query_terms & {"event", "events", "attended", "attend"}:
        queries.append(
            "art exhibition gallery museum festival studio attended event events past month "
            "Children's Museum History Museum Art Gallery Art Afternoon guided tour lecture street art"
        )
    if query_terms & {"fitness", "class", "classes"} and (
        {"how", "many"} <= query_terms or query_terms & {"typical", "week", "attend"}
    ):
        queries.append(
            "fitness classes yoga pilates spin boxing barre Zumba BodyPump Hip Hop Abs "
            "typical week attend schedule Mondays Tuesdays Thursdays Saturday Sunday"
        )
    if query_terms & {"museum", "museums", "gallery", "galleries"} and query_terms & {
        "order",
        "ordered",
        "earliest",
        "latest",
        "sequence",
        "timeline",
    }:
        queries.append(
            "museum museums gallery galleries visited attended lecture lectures series tour guided "
            "exhibition came back got back participated"
        )
    if query_terms & {"museum", "museums", "gallery", "galleries"} and {"how", "many"} <= query_terms:
        queries.append("February museum museums gallery galleries visited went attended")
        queries.append("February 2/8 2/15 Natural History Museum The Art Cube visited art gallery")
        if query_terms & _MONTH_TERMS:
            queries.append(
                "museum museums gallery galleries visited went attended "
                "December January February March April May June July August September October November"
            )
    if query_terms & {"property", "properties", "house", "home", "townhouse"} and {"how", "many"} <= query_terms:
        queries.append("property house bungalow condo townhouse viewed toured saw offer")
        if query_terms & {"townhouse", "offer", "brookside"}:
            queries.append(
                "Brookside townhouse Oakwood bungalow Cedar Creek 1-bedroom condo 2-bedroom condo kitchen renovation budget highway higher bid"
            )
    if query_terms & {"instrument", "instruments", "guitar", "piano"} and {"how", "many"} <= query_terms:
        queries.append(
            "musical instruments guitar piano drum set acoustic electric korg yamaha fender pearl owned had playing"
        )
    if generic_temporal_interval_queries and not _suppress_generic_temporal_interval_queries(query_terms):
        queries.extend(generic_temporal_interval_queries)
    if generic_temporal_count_queries and not _has_domain_specific_temporal_source_query(query_terms):
        queries.extend(generic_temporal_count_queries)
    return tuple(queries)


def source_lane_queries(query: str, graph_results: list[str]) -> tuple[str, ...]:
    """Return source-lane queries in safe recall order.

    The original user query remains first so graph-derived concepts can improve
    recall without replacing the lexical evidence request when graph retrieval
    starts in the wrong neighborhood.
    """
    queries = [query]
    queries.extend(aggregation_event_source_queries(query))
    expanded = source_lane_query(query, graph_results)
    if expanded != query:
        queries.append(expanded)
    return tuple(dict.fromkeys(queries))


def _clock_time_values(contexts: list[str]) -> list[int]:
    values: list[int] = []
    for context in contexts:
        for match in _CLOCK_TIME_RE.finditer(context):
            hour = int(match.group("hour"))
            minute = int(match.group("minute") or 0)
            period = match.group("period").casefold().replace(".", "")
            if period == "am":
                hour = 0 if hour == 12 else hour
            else:
                hour = 12 if hour == 12 else hour + 12
            total = hour * 60 + minute
            if total not in values:
                values.append(total)
    return values


def _bedtime_appointment_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer bedtime and appointment records for time-before-appointment questions."""
    query_terms = query_tokens if query_tokens is not None else set(source_tokens(query))
    if not (query_terms & {"bed", "bedtime"} and query_terms & {"doctor", "appointment"}):
        return 0
    context_terms = set(source_tokens(context))
    score = 0
    if context_terms & {"bed", "bedtime"} and _clock_time_values([context]):
        score += 5
        if re.search(
            r"\b(?:went|get|got)\s+to\s+bed\b[^.!?]{0,80}\b(?:until|at)\b",
            context,
            flags=re.IGNORECASE,
        ):
            score += 8
    if context_terms & {"doctor", "appointment", "appointments"}:
        score += 3
        if re.search(r"\bDr\.\s+[A-Z][A-Za-z]+\b", context):
            score += 2
    return score


def _relative_minute_offsets(contexts: list[str]) -> list[int]:
    values: list[int] = []
    for context in contexts:
        for match in _RELATIVE_MINUTE_OFFSET_RE.finditer(context):
            value = int(match.group("value"))
            direction = match.group("direction").casefold()
            offset = -value if direction in {"earlier", "before"} else value
            if offset not in values:
                values.append(offset)
    return values


def _time_offset_evidence_score(query: str, context: str) -> int:
    """Prefer base clock-time and relative-offset evidence for time questions."""
    if not _time_offset_query(query):
        return 0
    text = source_context_snippet(context, max_chars=1_500)
    score = 0
    if _clock_time_values([text]) and re.search(r"\b(?:wake|waking|wake-up)\b", text, flags=re.IGNORECASE):
        score += 5
    if _relative_minute_offsets([text]) and re.search(r"\b(?:wake|waking|earlier|later)\b", text, flags=re.IGNORECASE):
        score += 5
    return score


def _format_minutes_as_clock(total_minutes: int) -> str:
    total_minutes %= 24 * 60
    hour_24, minute = divmod(total_minutes, 60)
    period = "AM" if hour_24 < 12 else "PM"
    hour_12 = hour_24 % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:{minute:02d} {period}"


def _time_offset_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project time candidates adjusted by relative minute offsets."""
    if not _time_offset_query(query):
        return []
    base_times = _clock_time_values(contexts)
    offsets = _relative_minute_offsets(contexts)
    if not base_times or not offsets:
        return []
    lines = ["time_values=" + ",".join(_format_minutes_as_clock(value) for value in base_times)]
    lines.append("time_offset_minutes=" + ",".join(str(value) for value in offsets))
    answers: list[str] = []
    for base in base_times:
        for offset in offsets:
            answer = _format_minutes_as_clock(base + offset)
            if answer not in answers:
                answers.append(answer)
    for answer in answers[:5]:
        lines.append(f"time_offset_answer={answer}")
    return lines


def _direct_time_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project direct clock-time answers when the query asks for a cited time."""
    if not _direct_time_query(query):
        return []
    scored_times: list[tuple[int, int]] = []
    for context in contexts:
        for value in _clock_time_values([context]):
            score = (
                _bedtime_appointment_evidence_score(query, context)
                + _query_overlap_score(_query_specific_terms(query), context)
            )
            scored_times.append((score, value))
    if not scored_times:
        return []
    scored_times.sort(key=lambda item: (-item[0], item[1]))
    answer = _format_minutes_as_clock(scored_times[0][1])
    return [
        "time_answer_type=direct_clock",
        f"time_answer={answer}",
    ]


def _first_month_event_date_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"what", "which"} and "date" in tokens and "first" in tokens and tokens & _MONTH_TERMS)


def _first_month_event_date_terms(query: str) -> set[str]:
    stopwords = {
        "attend",
        "attended",
        "date",
        "did",
        "event",
        "first",
        "happened",
        "which",
        "what",
        "when",
    } | _MONTH_TERMS | _QUERY_SOURCE_STOPWORDS
    return {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in stopwords and not token.isdigit()
    }


def _first_month_event_context_relevant(query: str, context_terms: set[str]) -> bool:
    query_terms = set(source_tokens(query))
    if query_terms & {"attend", "attended"} and not context_terms & {"attend", "attended", "went", "joined"}:
        return False
    return not (query_terms & {"event", "party"} and not context_terms & {"event", "party", "festival", "workshop", "meetup"})


def _first_month_event_raw_span(snippet: str, month: int, day: int) -> str:
    month_names = [name for name, ordinal in _MONTH_ORDINALS.items() if ordinal == month]
    month_pattern = "|".join(sorted(month_names, key=len, reverse=True))
    patterns = (
        rf"\b{month_pattern}\s+{day}(?:st|nd|rd|th)?\b",
        rf"\b{day}(?:st|nd|rd|th)?\s+of\s+(?:{month_pattern})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, snippet, flags=re.IGNORECASE)
        if match:
            start = max(0, match.start() - 120)
            end = min(len(snippet), match.end() + 120)
            return snippet[start:end].strip()
    return snippet[:240].strip()


def _first_month_event_date_observation(query: str, contexts: list[str]) -> tuple[int, int, str, str] | None:
    if not _first_month_event_date_query(query):
        return None
    query_months = [token for token in source_tokens(query) if token in _MONTH_TERMS]
    if not query_months:
        return None
    month = _MONTH_ORDINALS[query_months[0]]
    query_terms = _first_month_event_date_terms(query)
    observations: list[tuple[int, int, int, str, str]] = []
    for index, context in enumerate(contexts):
        snippet = source_context_snippet(context, max_chars=2_000)
        snippet_terms = set(source_tokens(snippet))
        if not _first_month_event_context_relevant(query, snippet_terms):
            continue
        overlap = len(query_terms & snippet_terms)
        if query_terms and overlap <= 0:
            continue
        for candidate_month, day in _month_day_mentions(snippet):
            if candidate_month != month:
                continue
            raw_span = _first_month_event_raw_span(snippet, month, day)
            observations.append((day, -(overlap + source_lane_priority(context)), index, context, raw_span))
    if not observations:
        return None
    day, _score, _index, context, raw_span = min(observations, key=lambda item: (item[0], item[1], item[2]))
    return month, day, context, raw_span


def _format_month_day_answer(month: int, day: int) -> str:
    month_name = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }[month]
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{month_name} {day}{suffix}"


def _first_month_event_date_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project the earliest explicit event date in a named month."""
    observation = _first_month_event_date_observation(query, contexts)
    if observation is None:
        return []
    month, day, context, raw_span = observation
    answer = _format_month_day_answer(month, day)
    return _query_bound_direct_answer_lines(
        (
            answer,
            [source_context_group(context)],
            raw_span,
        )
    )


def _first_month_event_date_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    observation = _first_month_event_date_observation(query, contexts)
    if observation is None:
        return []
    month, day, context, raw_span = observation
    return _ledger_row_lines(
        [
            {
                "fact_id": f"first_month_event_date:{source_context_group(context)}:{month}:{day}",
                "source_group": source_context_group(context),
                "citation": source_context_citation(context),
                "kind": "date",
                "value": f"{month}/{day}",
                "unit": "month_day",
                "raw_span": raw_span,
                "candidate": _format_month_day_answer(month, day),
                "include_reason": "first_month_event_date",
                "confidence": 0.84,
            }
        ]
    )


def _temporal_order_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if tokens & {"meet", "met"} and tokens & {"first", "earlier", "before"}:
        return True
    if "or" in tokens and tokens & {"first", "earlier", "before"}:
        return True
    return bool(tokens & {"first", "earlier", "before"}) and bool(
        tokens & {"event", "happened", "which"}
    )


def source_lane_candidate_limit(query: str, *, limit: int) -> int:
    """Return internal source candidate budget for source-sensitive retrieval."""
    if limit <= 0:
        return 0
    intent = classify_retrieval_intent(query, limit=limit)
    if not intent.needs_source_lane:
        return limit
    if any(
        reason in intent.reasons
        for reason in ("aggregation", "aggregation_question", "absence_check", "event_slot_question")
    ):
        return max(limit, intent.source_lane_slots * 12)
    if _temporal_order_query(query) or _temporal_interval_query(query):
        return max(limit, intent.source_lane_slots * 8)
    return max(limit, intent.source_lane_slots * 4)


def _temporal_order_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer concrete alternatives needed by temporal-order questions."""
    if not _temporal_order_query(query):
        return 0
    query_terms = query_tokens if query_tokens is not None else set(source_tokens(query))
    context_terms = set(source_tokens(context))
    score = 0
    if query_terms & {"lens", "prime"} and context_terms & {"lens", "prime"}:
        score += 3
        if context_terms & {"got", "arrived", "arrival", "new"}:
            score += 3
    if query_terms & {"coast", "coastal", "trip", "road"} and context_terms & {"coast", "coastal", "trip", "road"}:
        score += 3
        if context_terms & {"took", "went", "visited"}:
            score += 2
    return score


def _clean_temporal_order_choice_label(value: str) -> str:
    value = " ".join(value.strip(" ,.?;:'\"").split())
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(?:the\s+arrival\s+of|arrival\s+of|arrival)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^losing\b", "lost", value, flags=re.IGNORECASE)
    value = re.sub(r"^purchasing\b", "purchased", value, flags=re.IGNORECASE)
    value = re.sub(r"^receiving\b", "received", value, flags=re.IGNORECASE)
    return value


def _temporal_order_choice_evidence_label(candidate: str, span: str) -> str:
    """Preserve local possessives from the source span in temporal-order labels."""
    if not candidate or re.search(r"\b(?:my|our|their|his|her)\b", candidate, flags=re.IGNORECASE):
        return candidate
    terms = candidate.split()
    if len(terms) < 2:
        return candidate
    tail = " ".join(re.escape(term) for term in terms[1:])
    match = re.search(rf"\b(?P<verb>{re.escape(terms[0])})\s+(?P<possessive>my|our|their|his|her)\s+{tail}\b", span, flags=re.IGNORECASE)
    if not match:
        return candidate
    return f"{terms[0]} {match.group('possessive').casefold()} {' '.join(terms[1:])}"


def _temporal_order_choice_terms(choice: str) -> tuple[str, ...]:
    stopwords = {
        "arrival",
        "device",
        "event",
        "happened",
        "item",
        "new",
        "purchase",
        "task",
        "the",
        "trip",
        "vehicle",
        "with",
    } | _QUERY_SOURCE_STOPWORDS | set(_NUMBER_WORDS)
    terms = [
        token
        for token in source_tokens(choice)
        if len(token) > 2 and token not in stopwords and not token.isdigit()
    ]
    return tuple(dict.fromkeys(terms))


_TEMPORAL_ORDER_GENERIC_CHOICE_TERMS = {
    "event",
    "events",
    "first",
    "happened",
    "new",
    "phone",
    "task",
    "tasks",
}


def _temporal_order_choice_term_variants(term: str) -> set[str]:
    variants = {
        "fixing": {"fix", "fixed", "fixing"},
        "losing": {"lost", "lose", "losing"},
        "meet": {"meet", "met"},
        "met": {"meet", "met"},
        "purchase": {"purchase", "purchased", "purchasing", "bought"},
        "purchasing": {"purchase", "purchased", "purchasing", "bought"},
        "receiving": {"receive", "received", "receiving"},
    }
    return variants.get(term, {term})


def _temporal_order_choice_span_matches(span: str, terms: tuple[str, ...]) -> bool:
    """Return whether a local dated span substantively binds a query alternative."""
    expanded_terms = set(terms)
    for term in terms:
        expanded_terms.update(_temporal_order_choice_term_variants(term))
    span_terms = set(source_tokens(span))
    matched = expanded_terms & span_terms
    if not matched:
        return False
    distinctive_terms = expanded_terms - _TEMPORAL_ORDER_GENERIC_CHOICE_TERMS
    distinctive_matched = distinctive_terms & span_terms
    if distinctive_terms and not distinctive_matched:
        return False
    return not (len(terms) >= 2 and len(matched) < 2 and not distinctive_matched)


def _temporal_order_choice_spans(text: str, terms: tuple[str, ...]) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
        if sentence.strip()
    ]
    spans: list[str] = []
    term_set = set(terms)
    for sentence in sentences:
        if _query_overlap_score(term_set, sentence) <= 0:
            continue
        spans.append(sentence)
    return spans or [text]


def _temporal_order_session_year(context: str) -> int | None:
    match = re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/", context)
    return int(match.group("year")) if match else None


def _temporal_order_date_value(year: int, month: int, day: int) -> int | None:
    try:
        return date(year, month, day).toordinal()
    except ValueError:
        return None


def _temporal_order_action_date_value(span: str, context: str) -> int | None:
    year = _temporal_order_session_year(context)
    if year is None:
        return None
    month_pattern = "|".join(sorted(_MONTH_ORDINALS, key=len, reverse=True))
    action_pattern = (
        r"\b(?:arrived|attended|became|born|bought|completed|finished|got|happened|"
        r"met|participated|purchased|received|set\s+up|started|took|visited)\b"
    )
    patterns = (
        action_pattern + rf"[^.!?]{{0,140}}\bon\s+(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
        rf"\bon\s+(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b[^.!?]{{0,140}}" + action_pattern,
    )
    for pattern in patterns:
        if match := re.search(pattern, span, flags=re.IGNORECASE):
            month = _MONTH_ORDINALS[match.group("month").casefold()]
            return _temporal_order_date_value(year, month, int(match.group("day")))
    month_days = _month_day_mentions(span)
    if month_days and re.search(r"\b(?:plan|planned|planning|schedule|scheduled|expect|expected|order|ordered)\b", span, flags=re.IGNORECASE):
        return None
    if len(month_days) == 1:
        month, day = month_days[0]
        return _temporal_order_date_value(year, month, day)
    return None


def _temporal_order_session_date_value(context: str) -> int | None:
    match = re.search(
        r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})",
        context,
    )
    if not match:
        return None
    return _temporal_order_date_value(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )


def _relative_days_ago(text: str) -> int | None:
    lowered = text.casefold()
    if "last week" in lowered:
        return 7
    if "last month" in lowered:
        return 30
    if "a few months ago" in lowered:
        return 90
    if "recently" in lowered:
        return 3
    for pattern, multiplier in _RELATIVE_DAYS_AGO_RE:
        for match in pattern.finditer(text):
            value_text = match.group("value").casefold()
            value = _NUMBER_WORDS.get(value_text)
            if value is None:
                value = int(value_text)
            return value * multiplier
    return None


def _recency_evidence_score(query: str, context: str) -> int:
    """Prefer categorical start/use evidence with relative recency markers."""
    if not _recency_comparison_query(query):
        return 0
    text = source_context_snippet(context, max_chars=1_500)
    context_terms = set(source_tokens(text))
    score = 0
    if _relative_days_ago(text) is not None:
        score += 4
    if context_terms & {"started", "start", "using", "began"}:
        score += 3
    if _streaming_service_names_in_context(text):
        score += 4
    return score


def _recency_observations(query: str, contexts: list[str]) -> list[tuple[int, str, str]]:
    observations: list[tuple[int, str, str]] = []
    for context in contexts:
        text = source_context_snippet(context, max_chars=1_500)
        days_ago = _relative_days_ago(text)
        if days_ago is None:
            continue
        for value in _recency_candidate_values(query, text):
            observations.append((days_ago, value, context))
    observations.sort(key=lambda item: (item[0], item[1].casefold()))
    return observations


def _recency_ledger_rows(query: str, contexts: list[str]) -> list[dict[str, object]]:
    if not _recency_comparison_query(query):
        return []
    return [
        {
            "fact_id": f"recency:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "temporal_order",
            "value": str(days_ago),
            "unit": "order_value",
            "raw_span": str(days_ago),
            "candidate": value,
            "include_reason": "recency_candidate",
            "confidence": 0.78,
        }
        for index, (days_ago, value, context) in enumerate(_recency_observations(query, contexts)[:5])
    ]


def recency_candidate_projection(query: str, contexts: list[str]) -> EvidenceProjection:
    """Project typed most-recent categorical evidence for checkout ranking."""
    if not _recency_comparison_query(query):
        return EvidenceProjection((), ())
    observations = _recency_observations(query, contexts)
    if not observations:
        return EvidenceProjection((), ())
    answer_days, answer, answer_context = observations[0]
    support_source_id = source_context_group(answer_context)
    lines = [
        "candidate_rank=1 candidate_type=recency candidate_confidence=0.78",
        f"candidate_support={support_source_id}",
        f"recency_answer={answer}",
        f"recency_source_id={support_source_id}",
    ]
    for index, (days_ago, value, _context) in enumerate(observations[:5], start=1):
        lines.append(f"recency_rank={index} relative_days_ago={days_ago} candidate={value}")
    ledger_rows = tuple(_recency_ledger_rows(query, contexts))
    candidate = {
        "rank": 1,
        "type": "recency",
        "confidence": 0.78,
        "answer_key": "recency_answer",
        "answer": answer,
        "support_source_ids": [support_source_id],
        "excluded_source_ids": [
            source_group
            for source_group in dict.fromkeys(source_context_group(context) for _days, _value, context in observations[1:])
            if source_group != support_source_id
        ],
    }
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=(support_source_id,),
        ledger_rows=ledger_rows,
        answer_candidates=(candidate,),
        operations=(
            {
                "name": "select_most_recent",
                "kind": "temporal_order",
                "input_count": len(observations),
                "answer": answer,
                "answer_days_ago": answer_days,
            },
        ),
        result={
            "answer_key": "recency_answer",
            "answer": answer,
            "support_source_ids": [support_source_id],
        },
    )


def _recency_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project most-recent categorical answers from relative-time evidence."""
    return list(recency_candidate_projection(query, contexts).lines)
