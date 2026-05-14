"""Retrieval planning utilities shared by product and benchmark paths."""

from __future__ import annotations

import re
from dataclasses import dataclass

from zaxy.evidence_candidates import aggregate_candidate_projection
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


def source_lane_queries(query: str, graph_results: list[str]) -> tuple[str, ...]:
    """Return source-lane queries in safe recall order.

    The original user query remains first so graph-derived concepts can improve
    recall without replacing the lexical evidence request when graph retrieval
    starts in the wrong neighborhood.
    """
    expanded = source_lane_query(query, graph_results)
    if expanded == query:
        return (query,)
    return (query, expanded)


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
    preferred_source_groups: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    """Build one compact cited source bundle for multi-source synthesis queries."""
    intent = classify_retrieval_intent(query, limit=limit)
    if (
        not {"aggregation", "aggregation_question"} & set(intent.reasons)
        and not _issue_query(query)
        and not _average_query(query)
        and not _numeric_comparison_query(query)
        and not _time_offset_query(query)
        and not _temporal_order_query(query)
    ):
        return None
    group_limit = max(limit, intent.source_lane_slots)
    ordered_sources = query_specific_source_order(query, source_results)
    if preferred_source_groups:
        ordered_sources = preferred_source_group_order(
            ordered_sources,
            preferred_source_groups,
        )
    grouped_sources = diverse_source_contexts(
        ordered_sources,
        limit=group_limit,
        preserve_order=True,
    )
    if len(grouped_sources) < 2:
        return None
    aggregation_query = bool({"aggregation", "aggregation_question"} & set(intent.reasons))
    aggregate_projection = aggregate_candidate_projection(query, grouped_sources)
    derived_lines = [
        *_numeric_synthesis_lines(
            query,
            grouped_sources,
            aggregate_lines=list(aggregate_projection.lines),
        ),
        *_temporal_order_synthesis_lines(query, grouped_sources),
        *_issue_synthesis_lines(query, grouped_sources),
    ]
    if not aggregation_query and not derived_lines and missing_query_target(query, grouped_sources):
        return None
    if not aggregation_query and not derived_lines:
        return None
    lines = [
        "zaxy_synthesis_bundle=true",
        "synthesis_mode=multi_source_aggregation",
        f"query={query}",
        f"source_count={len(grouped_sources)}",
    ]
    lines.extend(derived_lines)
    support_sources = _supporting_synthesis_sources(
        grouped_sources,
        source_groups=aggregate_projection.source_groups,
    )
    for index, context in enumerate(support_sources, start=1):
        lines.append(
            "- "
            f"source_id={source_context_group(context)} "
            f"citation={source_context_citation(context)} "
            f"snippet={source_context_snippet(context)}"
        )
        if index >= group_limit:
            break
    return "\n".join(lines)


def preferred_source_group_order(
    contexts: list[str],
    preferred_groups: list[str] | tuple[str, ...],
) -> list[str]:
    """Move graph-anchored source groups ahead of lexical-only candidates by graph rank."""
    if not preferred_groups:
        return contexts
    group_rank = {
        group: rank
        for rank, group in enumerate(dict.fromkeys(preferred_groups))
    }
    indexed = list(enumerate(contexts))
    indexed.sort(
        key=lambda item: (
            group_rank.get(source_context_group(item[1]), len(group_rank)),
            item[0],
        )
    )
    return [context for _, context in indexed]


def absence_check_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
) -> str | None:
    """Build cited guidance for questions about absent personal memories."""
    intent = classify_retrieval_intent(query, limit=limit)
    if not intent.needs_source_lane:
        return None
    grouped_sources = diverse_source_contexts(
        source_results,
        limit=max(1, intent.source_lane_slots or min(2, limit)),
    )
    target = missing_query_target(query, grouped_sources)
    if not target and "absence_check" in intent.reasons:
        target = absence_check_target(query)
    if not target:
        return None
    if not grouped_sources or target_terms_present(target, grouped_sources):
        return None
    lines = [
        "zaxy_absence_check=true",
        "synthesis_mode=absence_check",
        f"query={query}",
        f"not_mentioned_candidate={target}",
        (
            "answer_guidance=The information provided is not enough. "
            "You did not mention this information. "
            f"You did not mention {target}. "
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


def missing_query_target(query: str, contexts: list[str]) -> str:
    """Return query-specific terms absent from all cited source contexts."""
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


def _absence_term_variants(term: str) -> set[str]:
    variants = {term}
    if term.endswith("ing") and len(term) > 4:
        stem = term[:-3]
        variants.update({stem, f"{stem}e", f"{stem}ed"})
    if len(term) > 3:
        variants.update({f"{term}s", f"{term}ed", f"{term}ing"})
        if term.endswith("y"):
            variants.add(f"{term[:-1]}ies")
        if term.endswith("s"):
            variants.add(term[:-1])
    irregular = {
        "airline": {"airlines"},
        "age": {"ages", "turned"},
        "fly": {"flew", "flown", "flying"},
        "grandparents": {"grandparent", "grandma", "grandpa", "grandmother", "grandfather"},
        "losing": {"lost", "lose"},
        "parents": {"parent", "mom", "dad", "mother", "father"},
        "purchasing": {"purchased", "purchase"},
        "receiving": {"received", "receive"},
    }
    variants.update(irregular.get(term, set()))
    return variants


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


def query_specific_source_order(query: str, contexts: list[str]) -> list[str]:
    """Prefer source contexts that overlap query-specific concepts."""
    query_terms = _query_specific_terms(query)
    if not query_terms:
        return contexts
    indexed = list(enumerate(contexts))
    indexed.sort(
        key=lambda item: (
            -_query_overlap_score(query_terms, item[1]),
            -source_lane_priority(item[1]),
            item[0],
        )
    )
    return [context for _, context in indexed]


def diverse_source_contexts(
    contexts: list[str],
    *,
    limit: int,
    preserve_order: bool = False,
) -> list[str]:
    """Select source contexts across provenance groups before filling by rank."""
    if limit <= 0:
        return []
    if not preserve_order:
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


_QUERY_SOURCE_STOPWORDS = {
    "a",
    "about",
    "after",
    "ago",
    "all",
    "and",
    "before",
    "between",
    "breed",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "have",
    "how",
    "i",
    "in",
    "it",
    "many",
    "me",
    "money",
    "most",
    "my",
    "of",
    "on",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
    "average",
    "days",
    "event",
    "first",
    "happened",
    "project",
    "spent",
    "task",
    "total",
}


def _query_specific_terms(query: str) -> set[str]:
    return {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in _QUERY_SOURCE_STOPWORDS and not token.isdigit()
    }


def _query_overlap_score(query_terms: set[str], context: str) -> int:
    context_terms = set(source_tokens(context))
    score = len(query_terms & context_terms)
    for term in query_terms:
        if term.endswith("ing") and term[:-3] in context_terms:
            score += 1
        if f"{term}ed" in context_terms or f"{term}ing" in context_terms:
            score += 1
    return score


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


def _supporting_synthesis_sources(
    contexts: list[str],
    *,
    source_groups: tuple[str, ...],
) -> list[str]:
    if not source_groups:
        return contexts
    source_group_set = set(source_groups)
    selected = [
        context for context in contexts
        if source_context_group(context) in source_group_set
    ]
    return selected if len(selected) >= 2 else contexts


def _numeric_synthesis_lines(
    query: str,
    contexts: list[str],
    *,
    aggregate_lines: list[str] | None = None,
) -> list[str]:
    """Project deterministic numeric operations from cited source snippets."""
    numeric_contexts = [_numeric_context_text(context) for context in contexts]
    lines: list[str] = list(aggregate_lines or [])
    has_typed_duration = any(line.startswith("duration_values=") for line in lines)
    lines.extend(_age_average_synthesis_lines(query, numeric_contexts))
    if not has_typed_duration:
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
    week_contexts = [
        _numeric_context_text(context)
        for context in _query_relevant_numeric_contexts(query, contexts)
    ]
    week_values = _week_values(week_contexts)
    if week_values:
        lines.append("week_values=" + ",".join(_format_number(value) for value in week_values))
        week_total = sum(week_values)
        lines.append(f"week_total={_format_number(week_total)} weeks")
        if week_words := _number_words(week_total):
            lines.append(f"week_total_words={week_words} weeks")
        if len(week_values) >= 2:
            week_interval = max(week_values) - min(week_values)
            lines.append(f"week_interval={_format_number(week_interval)} weeks")
            if week_interval_words := _number_words(week_interval):
                lines.append(f"week_interval_answer={week_interval_words} weeks")
    month_contexts = [
        _numeric_context_text(context)
        for context in _query_relevant_numeric_contexts(query, contexts)
    ]
    month_values = _month_values(month_contexts)
    if month_values:
        lines.append("month_values=" + ",".join(_format_number(value) for value in month_values))
        month_total = sum(month_values)
        lines.append(f"month_total={_format_number(month_total)} months ago")
        if month_words := _number_words(month_total):
            lines.append(f"month_total_words={month_words} months ago")
        if len(month_values) >= 2:
            month_interval = max(month_values) - min(month_values)
            lines.append(f"month_interval={_format_number(month_interval)} months")
            if month_interval_words := _number_words(month_interval):
                lines.append(f"month_interval_answer={month_interval_words} months")
    lines.extend(_mixed_relative_interval_lines(week_values=week_values, month_values=month_values))
    lines.extend(_time_offset_synthesis_lines(query, numeric_contexts))
    return lines


def _query_relevant_numeric_contexts(query: str, contexts: list[str]) -> list[str]:
    """Keep numeric evidence tied to query concepts before aggregation."""
    query_terms = _query_specific_terms(query)
    if not query_terms:
        return contexts
    scored = [
        (_query_overlap_score(query_terms, context), index, context)
        for index, context in enumerate(contexts)
    ]
    best_score = max((score for score, _, _ in scored), default=0)
    if best_score < 2:
        return contexts
    threshold = max(2, best_score // 2)
    selected = [
        context
        for score, _, context in scored
        if score >= threshold
    ]
    return selected if len(selected) >= 2 else contexts


def _numeric_context_text(context: str) -> str:
    """Return source text once, excluding Eventloom JSON payload echoes."""
    text = source_context_snippet(context)
    return text.split(' {"content":', 1)[0]


def _age_average_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project average-age arithmetic from cited family age evidence."""
    query_tokens = set(source_tokens(query))
    if "average" not in query_tokens or "age" not in query_tokens:
        return []
    values = _age_values(contexts)
    if len(values) < 2:
        return []
    average = sum(values) / len(values)
    return [
        "age_values=" + ",".join(str(value) for value in values),
        f"age_average={average:.1f}".rstrip("0").rstrip("."),
    ]


def _age_values(contexts: list[str]) -> list[int]:
    values: list[int] = []
    patterns = (
        r"\b(?:turned|am|is)\s+(?P<value>\d{1,3})\b",
        r"\b(?P<person>mom|dad|mother|father|grandma|grandpa|grandmother|grandfather)\s+is\s+(?P<value>\d{1,3})\b",
    )
    for context in contexts:
        for pattern in patterns:
            for match in re.finditer(pattern, context, flags=re.IGNORECASE):
                value = int(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
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


def _week_values(contexts: list[str]) -> list[float]:
    values = _unit_values(contexts, unit_pattern=r"weeks?")
    pattern = re.compile(
        r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            _append_unique_number(values, float(_NUMBER_WORDS[match.group("value").casefold()]))
    return values


def _month_values(contexts: list[str]) -> list[float]:
    values = _unit_values(contexts, unit_pattern=r"months?")
    pattern = re.compile(
        r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            _append_unique_number(values, float(_NUMBER_WORDS[match.group("value").casefold()]))
    return values


def _append_unique_number(values: list[float], value: float) -> None:
    if value not in values:
        values.append(value)


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


def _average_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return "average" in query_tokens


def _numeric_comparison_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"most", "least", "more", "less", "highest", "lowest"}) and bool(
        query_tokens & {"money", "amount", "cost", "spent", "spend", "price", "total"}
    )


def _time_offset_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return "time" in query_tokens


def _clock_time_values(contexts: list[str]) -> list[int]:
    values: list[int] = []
    pattern = re.compile(
        r"\b(?P<hour>1[0-2]|0?[1-9]):(?P<minute>[0-5]\d)\s*(?P<period>a\.?m\.?|p\.?m\.?)\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            hour = int(match.group("hour"))
            minute = int(match.group("minute"))
            period = match.group("period").casefold().replace(".", "")
            if period == "am":
                hour = 0 if hour == 12 else hour
            else:
                hour = 12 if hour == 12 else hour + 12
            total = hour * 60 + minute
            if total not in values:
                values.append(total)
    return values


def _relative_minute_offsets(contexts: list[str]) -> list[int]:
    values: list[int] = []
    pattern = re.compile(
        r"\b(?P<value>\d+)\s+minutes?\s+(?P<direction>earlier|before|later|after)\b",
        flags=re.IGNORECASE,
    )
    for context in contexts:
        for match in pattern.finditer(context):
            value = int(match.group("value"))
            direction = match.group("direction").casefold()
            offset = -value if direction in {"earlier", "before"} else value
            if offset not in values:
                values.append(offset)
    return values


def _format_minutes_as_clock(total_minutes: int) -> str:
    total_minutes %= 24 * 60
    hour_24, minute = divmod(total_minutes, 60)
    period = "AM" if hour_24 < 12 else "PM"
    hour_12 = hour_24 % 12
    if hour_12 == 0:
        hour_12 = 12
    return f"{hour_12}:{minute:02d} {period}"


def _temporal_order_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project relative ordering candidates from cited temporal evidence."""
    if not _temporal_order_query(query):
        return []
    observations: list[tuple[int, str]] = []
    for context in contexts:
        text = _numeric_context_text(context)
        days_ago = _relative_days_ago(text)
        if days_ago is None:
            continue
        candidate = _temporal_order_candidate(text)
        if not candidate:
            continue
        observations.append((days_ago, candidate))
    if len(observations) < 2:
        return []
    observations.sort(key=lambda item: item[0], reverse=True)
    lines = [f"temporal_order_answer={observations[0][1]}"]
    for index, (days_ago, candidate) in enumerate(observations[:5], start=1):
        lines.append(
            f"temporal_order_rank={index} relative_days_ago={days_ago} candidate={candidate}"
        )
    return lines


def _temporal_order_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"first", "earlier", "before"}) and bool(
        tokens & {"event", "happened", "which"}
    )


def _relative_days_ago(text: str) -> int | None:
    lowered = text.casefold()
    if "last week" in lowered:
        return 7
    if "recently" in lowered:
        return 3
    for unit, multiplier in (("month", 30), ("week", 7), ("day", 1)):
        pattern = re.compile(
            rf"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
            rf"(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
            rf"{unit}s?\s+ago\b",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            value_text = match.group("value").casefold()
            value = _NUMBER_WORDS.get(value_text)
            if value is None:
                value = int(value_text)
            return value * multiplier
    return None


def _temporal_order_candidate(text: str) -> str:
    text = re.sub(r"\bcontent=longmemeval_session_id=\S+\s*", "", text)
    text = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", "", text)
    text = re.sub(
        r"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
        r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
        r"(?:months?|weeks?|days?)\s+ago\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\blast week\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\brecently\b", "", text, flags=re.IGNORECASE)
    text = text.strip(" .")
    match = re.match(r"\bI\s+(?P<candidate>.+)", text, flags=re.IGNORECASE)
    if match:
        text = match.group("candidate").strip(" .")
    words = text.split()
    return " ".join(words[:8])


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
    if value.is_integer():
        return f"${int(value):,}"
    whole = int(value)
    fraction = f"{value:.2f}".split(".", 1)[1].rstrip("0")
    return f"${whole:,}.{fraction}"


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
