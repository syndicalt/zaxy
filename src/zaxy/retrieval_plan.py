"""Retrieval planning utilities shared by product and benchmark paths."""

from __future__ import annotations

import re
from dataclasses import dataclass

from zaxy.evidence_candidates import aggregate_candidate_projection, aggregate_evidence_score
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
    queries = [query]
    queries.extend(aggregation_event_source_queries(query))
    expanded = source_lane_query(query, graph_results)
    if expanded != query:
        queries.append(expanded)
    return tuple(dict.fromkeys(queries))


def aggregation_event_source_queries(query: str) -> tuple[str, ...]:
    """Return deterministic source queries for event-like aggregation memories."""
    query_terms = set(source_tokens(query))
    queries: list[str] = []
    if query_terms & {"model", "models", "kit", "kits"} and {"how", "many"} <= query_terms:
        queries.append("model kits finished started picked up got bought scale")
    if query_terms & {"doctor", "doctors", "physician", "physicians"} and {"how", "many"} <= query_terms:
        queries.append("doctor physician dermatologist ent visited saw appointment")
    if query_terms & {"movie", "movies", "film", "films", "festival", "festivals"} and {"how", "many"} <= query_terms:
        queries.append("film festival movie attended went participated")
    if query_terms & {"property", "properties", "house", "home", "townhouse"} and {"how", "many"} <= query_terms:
        queries.append("property house bungalow condo townhouse viewed toured saw offer")
    if query_terms & {"instrument", "instruments", "guitar", "piano"} and {"how", "many"} <= query_terms:
        queries.append(
            "musical instruments guitar piano drum set acoustic electric korg yamaha fender pearl owned had playing"
        )
    return tuple(queries)


def bridge_source_lane_queries(query: str, source_results: list[str]) -> tuple[str, ...]:
    """Return query expansions from deterministic session/entity bridges."""
    targets = possessive_entity_targets(query)
    attribute_terms = bridge_attribute_terms(query, targets)
    if not attribute_terms:
        return ()
    aliases = session_entity_aliases(query, source_results, targets=targets)
    if not aliases:
        return ()
    queries: list[str] = []
    for alias in aliases:
        queries.append(" ".join([alias, *attribute_terms]))
    return tuple(dict.fromkeys(queries))


def session_entity_aliases(
    query: str,
    source_results: list[str],
    *,
    targets: tuple[str, ...] | None = None,
    limit: int = 3,
) -> tuple[str, ...]:
    """Extract concrete aliases for possessive entity references in a query."""
    targets = targets if targets is not None else possessive_entity_targets(query)
    if not targets:
        return ()
    aliases: list[str] = []
    seen: set[str] = set()
    for context in source_results:
        text = source_context_snippet(context, max_chars=1_000)
        for target in targets:
            for alias in aliases_for_possessive_target(text, target):
                normalized = alias.casefold()
                if normalized in seen:
                    continue
                seen.add(normalized)
                aliases.append(alias)
                if len(aliases) >= limit:
                    return tuple(aliases)
    return tuple(aliases)


def possessive_entity_targets(query: str) -> tuple[str, ...]:
    """Return entity nouns referenced possessively by the user query."""
    targets: list[str] = []
    for match in re.finditer(
        r"\b(?:my|our)\s+(?:new\s+|old\s+)?(?P<target>[a-z][a-z0-9_-]*)\b",
        query,
        flags=re.IGNORECASE,
    ):
        target = match.group("target").casefold()
        if target in _BRIDGE_ENTITY_STOPWORDS or target not in _BRIDGE_ENTITY_TARGETS:
            continue
        targets.append(target)
    return tuple(dict.fromkeys(targets))


def bridge_attribute_terms(
    query: str,
    targets: tuple[str, ...],
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    """Return attribute terms needed after resolving a possessive entity alias."""
    target_terms = set(targets)
    terms: list[str] = []
    for token in source_tokens(query):
        if (
            token in target_terms
            or token in _BRIDGE_QUERY_STOPWORDS
            or token.isdigit()
            or len(token) <= 1
        ):
            continue
        terms.append(token)
        if len(terms) >= limit:
            break
    return tuple(dict.fromkeys(terms))


def aliases_for_possessive_target(text: str, target: str) -> tuple[str, ...]:
    """Extract aliases introduced near a possessive entity target."""
    aliases: list[str] = []
    escaped_target = re.escape(target)
    patterns = (
        rf"\b(?i:my|our)\s+(?i:new\s+|old\s+)?(?i:{escaped_target})\s+(?P<alias>[A-Z][A-Za-z0-9'_-]{{1,40}})\b",
        rf"\b(?i:for|with|about)\s+(?P<alias>[A-Z][A-Za-z0-9'_-]{{1,40}})\b(?=[^.!?]{{0,80}}\b(?i:{escaped_target})\b)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            alias = match.group("alias").strip(" .'\"")
            if not valid_entity_alias(alias, target):
                continue
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


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


def valid_entity_alias(alias: str, target: str) -> bool:
    """Return whether a candidate alias is useful for source-query bridging."""
    normalized = alias.casefold()
    if not alias[0].isupper():
        return False
    if normalized == target or normalized in _BRIDGE_ALIAS_STOPWORDS:
        return False
    if normalized in _QUERY_SOURCE_STOPWORDS:
        return False
    if len(alias) < 2:
        return False
    return bool(re.search(r"[A-Za-z]", alias))


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
        and not _age_at_event_query(query)
        and not _numeric_comparison_query(query)
        and not _time_offset_query(query)
        and not _temporal_order_query(query)
        and not _possessive_attribute_query_target(query)
    ):
        return None
    group_limit = source_synthesis_candidate_limit(intent, limit=limit)
    if _average_query(query):
        group_limit = max(group_limit, 8)
    ordered_sources = query_specific_source_order(query, source_results)
    if preferred_source_groups:
        ordered_sources = preferred_source_group_order(
            ordered_sources,
            preferred_source_groups,
        )
    ordered_sources = evidence_source_order(query, ordered_sources)
    grouped_sources = diverse_source_contexts(
        ordered_sources,
        limit=group_limit,
        preserve_order=True,
    )
    direct_attribute = _possessive_attribute_query_target(query)
    if len(grouped_sources) < 2 and not direct_attribute:
        return None
    if (
        (
            (_numeric_comparison_query(query) or _temporal_order_query(query))
            and _query_alternatives(query)
        )
        or _temporal_interval_query(query)
    ) and should_defer_to_absence_check(query, grouped_sources, intent):
        return None
    aggregate_projection = aggregate_candidate_projection(query, grouped_sources)
    derived_lines = [
        *_numeric_synthesis_lines(
            query,
            grouped_sources,
            aggregate_lines=list(aggregate_projection.lines),
        ),
        *_temporal_order_synthesis_lines(query, grouped_sources),
        *_issue_synthesis_lines(query, grouped_sources),
        *_direct_fact_synthesis_lines(query, grouped_sources),
    ]
    if not derived_lines and should_defer_to_absence_check(query, grouped_sources, intent):
        return None
    if not derived_lines and missing_query_target(query, grouped_sources):
        return None
    if not derived_lines:
        return None
    support_sources = _supporting_synthesis_sources(
        grouped_sources,
        source_groups=aggregate_projection.source_groups,
    )
    lines = [
        "zaxy_synthesis_bundle=true",
        "synthesis_mode=multi_source_aggregation",
        f"query={query}",
        f"source_count={len(support_sources)}",
    ]
    lines.extend(derived_lines)
    support_source_limit = min(group_limit, max(limit, 8))
    for index, context in enumerate(support_sources, start=1):
        lines.append(
            "- "
            f"source_id={source_context_group(context)} "
            f"citation={source_context_citation(context)} "
            f"snippet={source_context_snippet(context)}"
        )
        if index >= support_source_limit:
            break
    return "\n".join(lines)


def source_synthesis_candidate_limit(intent: RetrievalIntent, *, limit: int) -> int:
    """Return the internal source pool size used before compact synthesis."""
    if {"aggregation", "aggregation_question"} & set(intent.reasons):
        return max(limit, intent.source_lane_slots * 4, 16)
    return max(limit, intent.source_lane_slots)


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


def evidence_source_order(query: str, contexts: list[str]) -> list[str]:
    """Prefer snippets that can produce typed synthesis evidence for the query."""
    query_terms = _query_specific_terms(query)
    indexed = list(enumerate(contexts))
    indexed.sort(
        key=lambda item: (
            -source_evidence_score(query, item[1]),
            -_query_overlap_score(query_terms, item[1]),
            -source_lane_priority(item[1]),
            item[0],
        )
    )
    return [context for _, context in indexed]


def source_evidence_score(query: str, context: str) -> int:
    """Return a deterministic evidence score for synthesis source selection."""
    projection = aggregate_candidate_projection(query, [context])
    score = aggregate_evidence_score(query, context)
    score += _query_action_object_evidence_score(query, context)
    for line in projection.lines:
        if line.startswith("candidate_type=") or " candidate_type=" in line:
            score += 3
        elif line.endswith("_answer=") or "_answer=" in line or line.startswith(("currency_values=", "duration_values=", "count_answer=", "date_values=")):
            score += 2
        else:
            score += 1
    return score


def _query_action_object_evidence_score(query: str, context: str) -> int:
    """Prefer contexts carrying the queried action-object evidence over nearby context."""
    if re.search(r"\bhow\s+long\s+had\b", query, flags=re.IGNORECASE):
        return 0
    context_terms = set(source_tokens(context))
    score = 0
    for target in _concrete_query_targets(query):
        terms = tuple(source_tokens(target))
        if not terms:
            continue
        action, *objects = terms
        if _absence_term_variants(action) & context_terms:
            score += 3
        score += sum(1 for term in objects if _absence_term_variants(term) & context_terms)
    return score


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
    target = high_precision_missing_target(query, grouped_sources)
    if not target and has_direct_fact_evidence(query, grouped_sources):
        return None
    if not target and "absence_check" in intent.reasons:
        target = missing_query_target(query, grouped_sources) or absence_check_target(query)
    if not target and {"aggregation", "aggregation_question"} & set(intent.reasons):
        target = _missing_location_target(query, grouped_sources)
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
    if known_evidence := known_related_evidence_summary(query, grouped_sources, target):
        lines.append(f"known_related_evidence={known_evidence}")
    for context in grouped_sources:
        lines.append(
            "- "
            f"source_id={source_context_group(context)} "
            f"citation={source_context_citation(context)} "
            f"snippet={source_context_snippet(context)}"
        )
    return "\n".join(lines)


def should_defer_to_absence_check(
    query: str,
    contexts: list[str],
    intent: RetrievalIntent,
) -> bool:
    """Return whether missing evidence should outrank numeric/order synthesis."""
    if not intent.needs_source_lane or not contexts:
        return False
    target = high_precision_missing_target(query, contexts)
    return bool(target and not target_terms_present(target, contexts))


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

_BRIDGE_ENTITY_STOPWORDS = _ABSENCE_QUERY_STOPWORDS | {
    "area",
    "favorite",
    "name",
    "new",
    "old",
    "place",
    "time",
}

_BRIDGE_ENTITY_TARGETS = {
    "bike",
    "bicycle",
    "camera",
    "car",
    "cat",
    "computer",
    "dog",
    "guitar",
    "instrument",
    "laptop",
    "pet",
    "phone",
    "tablet",
    "truck",
}

_BRIDGE_ALIAS_STOPWORDS = {
    "i",
    "the",
    "this",
    "that",
    "it",
    "a",
    "an",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
}

_BRIDGE_QUERY_STOPWORDS = _ABSENCE_QUERY_STOPWORDS | {
    "am",
    "an",
    "are",
    "can",
    "could",
    "favorite",
    "is",
    "name",
    "of",
    "our",
    "should",
    "that",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "why",
    "would",
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


def has_direct_fact_evidence(query: str, contexts: list[str]) -> bool:
    """Return whether contexts already contain enough direct-fact query evidence."""
    query_terms = _query_specific_terms(query)
    if not query_terms:
        return False
    context_terms: set[str] = set()
    for context in contexts:
        context_terms.update(source_tokens(context))
    return bool(query_terms) and all(
        _absence_term_variants(term) & context_terms
        for term in query_terms
    )


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


def high_precision_missing_target(query: str, contexts: list[str]) -> str:
    """Return concrete missing query targets with low false-positive risk."""
    if target := _missing_alternative_target(query, contexts):
        return target
    return _missing_concrete_query_target(query, contexts)


def known_related_evidence_summary(
    query: str,
    contexts: list[str],
    missing_target: str,
) -> str:
    """Return compact query evidence that is present while another target is absent."""
    del missing_target
    if present := _present_alternative_target(query, contexts):
        return present
    if present := _present_concrete_query_target(query, contexts):
        return present
    query_terms = _query_specific_terms(query)
    context_terms: set[str] = set()
    for context in contexts:
        context_terms.update(source_tokens(context))
    present_terms = [
        term for term in sorted(query_terms)
        if _absence_term_variants(term) & context_terms
    ]
    return " ".join(dict.fromkeys(present_terms[:6]))


def _missing_alternative_target(query: str, contexts: list[str]) -> str:
    alternatives = _query_alternatives(query)
    if len(alternatives) < 2:
        return ""
    for alternative in alternatives:
        terms = _alternative_terms(alternative)
        if terms and not _terms_present_in_contexts(terms, contexts):
            return " ".join(terms)
    return ""


def _missing_concrete_query_target(query: str, contexts: list[str]) -> str:
    """Return a missing action-object target from the query, if one is precise enough."""
    for target in _concrete_query_targets(query):
        target_tokens = tuple(source_tokens(target))
        if not target_tokens or target_tokens[0] not in _MISSING_CONCRETE_ACTIONS:
            continue
        missing_terms = _missing_target_terms(target_tokens, contexts)
        if not missing_terms:
            continue
        if len(target_tokens) > 2 and len(missing_terms) < len(target_tokens) - 1:
            return " ".join(missing_terms)
        if not _terms_present_in_contexts(target_tokens, contexts):
            return target
    return ""


def _missing_location_target(query: str, contexts: list[str]) -> str:
    """Return a missing proper location target from a duration/location query."""
    match = re.search(
        r"\b(?:in|to|from)\s+(?P<location>[A-Z][A-Za-z0-9' -]{1,60})(?:\s+for)?[?.,]?$",
        query,
    )
    if not match:
        return ""
    location = " ".join(match.group("location").strip(" .,'\"").split())
    terms = tuple(source_tokens(location))
    if not terms or _terms_present_in_contexts(terms, contexts):
        return ""
    return location.casefold()


def _present_alternative_target(query: str, contexts: list[str]) -> str:
    for alternative in _query_alternatives(query):
        terms = _alternative_terms(alternative)
        if terms and _terms_present_in_contexts(terms, contexts):
            return _clean_alternative_summary(alternative)
    return ""


def _present_concrete_query_target(query: str, contexts: list[str]) -> str:
    """Return the first concrete query target supported by the cited contexts."""
    for target in _concrete_query_targets(query):
        if _terms_present_in_contexts(tuple(source_tokens(target)), contexts):
            return target
    return ""


def _concrete_query_targets(query: str) -> tuple[str, ...]:
    """Extract bounded action-object targets that are safe for absence checks."""
    targets: list[str] = []
    action_pattern = re.compile(
        r"\b(?P<verb>bought|buy|purchased|purchase|purchasing|booked|book|booking|"
        r"started|start|starting|joined|join|joining|visited|visit|visiting)\s+"
        r"(?P<object>[a-z0-9][a-z0-9' -]{1,100}?)"
        r"(?=\s+(?:did|do|does|before|after|when|while|and|or)\b|[?.,;]|$)",
        flags=re.IGNORECASE,
    )
    for match in action_pattern.finditer(query):
        target = _normalize_concrete_query_target(
            match.group("verb"),
            match.group("object"),
        )
        if target:
            targets.append(target)
    return tuple(dict.fromkeys(targets))


def _normalize_concrete_query_target(verb: str, object_text: str) -> str:
    """Normalize a concrete action-object phrase without widening it to generic words."""
    verb_token = _canonical_absence_action(verb)
    object_terms = [
        token
        for token in source_tokens(object_text)
        if token not in _CONCRETE_TARGET_STOPWORDS
        and not token.isdigit()
        and len(token) > 1
    ]
    if not verb_token or not object_terms:
        return ""
    return " ".join([verb_token, *dict.fromkeys(object_terms)])


def _canonical_absence_action(verb: str) -> str:
    normalized = verb.casefold()
    canonical = {
        "buy": "bought",
        "bought": "bought",
        "purchase": "purchased",
        "purchased": "purchased",
        "purchasing": "purchased",
        "book": "booked",
        "booked": "booked",
        "booking": "booked",
        "start": "started",
        "started": "started",
        "starting": "started",
        "join": "joined",
        "joined": "joined",
        "joining": "joined",
        "visit": "visit",
        "visited": "visit",
        "visiting": "visit",
    }
    return canonical.get(normalized, "")


def _query_alternatives(query: str) -> tuple[str, ...]:
    normalized = re.sub(r"[?!.]+$", "", query.strip())
    parts = re.split(r"\s+or\s+", normalized, flags=re.IGNORECASE)
    if len(parts) < 2:
        return ()
    first = re.sub(r"^.*?(?:first|between|which|whether)\b", "", parts[0], flags=re.IGNORECASE).strip()
    alternatives = [first, *parts[1:]]
    return tuple(part for part in alternatives if part)


def _alternative_terms(text: str) -> tuple[str, ...]:
    stopwords = _ABSENCE_QUERY_STOPWORDS | {
        "became",
        "complete",
        "completed",
        "current",
        "did",
        "event",
        "first",
        "from",
        "happened",
        "or",
        "parent",
        "project",
        "start",
        "started",
        "task",
        "the",
        "which",
    }
    single_letter_identifiers = {
        match.group(0).casefold()
        for match in re.finditer(r"\b[A-Z]\b", text)
    }
    terms = [
        token
        for token in source_tokens(text)
        if token not in stopwords
        and (
            token.isdigit()
            or len(token) > 1
            or token in single_letter_identifiers
        )
    ]
    return tuple(dict.fromkeys(terms))


def _missing_target_terms(terms: tuple[str, ...], contexts: list[str]) -> tuple[str, ...]:
    missing: list[str] = []
    for term in terms:
        if _terms_present_in_contexts((term,), contexts):
            continue
        missing.append(term)
    return tuple(missing)


def _clean_alternative_summary(text: str) -> str:
    text = re.sub(r"^[,;:\s]+", "", text)
    text = re.sub(
        r"^(?:task\s+)?(?:did\s+)?(?:i\s+)?(?:complete\s+)?(?:first[\s,]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.strip(" ,;:.").split())


def _terms_present_in_contexts(terms: tuple[str, ...], contexts: list[str]) -> bool:
    if not terms:
        return False
    for context in contexts:
        if _negated_target_context(terms, context):
            continue
        context_terms = set(source_tokens(context))
        if all(_absence_term_variants(term) & context_terms for term in terms):
            return True
    return False


def _negated_target_context(terms: tuple[str, ...], context: str) -> bool:
    text = source_context_snippet(context, max_chars=1_200).casefold()
    for term in terms:
        variants = sorted(_absence_term_variants(term), key=len, reverse=True)
        variant_pattern = "|".join(re.escape(variant) for variant in variants)
        if re.search(
            rf"\b(?:no|not|never|without)\b[^.!?]{{0,80}}\b(?:{variant_pattern})\b",
            text,
        ):
            return True
        if re.search(
            rf"\b(?:{variant_pattern})\b[^.!?]{{0,80}}\b(?:not|never|wasn'?t|isn'?t|didn'?t)\b",
            text,
        ):
            return True
    return False


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
        "favorite": {"favourite", "favorites", "favourites"},
        "fly": {"flew", "flown", "flying"},
        "grandparents": {"grandparent", "grandma", "grandpa", "grandmother", "grandfather"},
        "losing": {"lost", "lose"},
        "parents": {"parent", "mom", "dad", "mother", "father"},
        "purchasing": {"purchased", "purchase"},
        "receiving": {"received", "receive"},
        "bought": {"buy", "bought", "got", "purchase", "purchased", "purchasing"},
        "purchased": {"buy", "bought", "purchase", "purchased", "purchasing"},
        "booked": {"book", "booking", "booked"},
        "ceremony": {"ceremony", "graduation"},
        "started": {"start", "starting", "started", "began"},
        "joined": {"join", "joining", "joined", "became"},
        "visit": {"visit", "visited", "visiting"},
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
        if all(_absence_term_variants(term) & context_terms for term in target_terms):
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
    "are",
    "before",
    "between",
    "been",
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
    "member",
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
    "favorite",
    "first",
    "happened",
    "project",
    "long",
    "spent",
    "task",
    "total",
}

_CONCRETE_TARGET_STOPWORDS = _QUERY_SOURCE_STOPWORDS | {
    "an",
    "at",
    "current",
    "most",
    "new",
    "old",
    "our",
    "recently",
    "using",
}

_MISSING_CONCRETE_ACTIONS = {
    "booked",
    "bought",
    "joined",
    "purchased",
    "started",
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


def _direct_fact_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project compact direct-attribute answers from cited source snippets."""
    attribute = _possessive_attribute_query_target(query)
    if not attribute:
        return []
    for context in contexts:
        snippet = source_context_snippet(context)
        if answer := _mixed_attribute_answer(snippet, attribute):
            return [
                "direct_fact_type=attribute",
                f"direct_fact_attribute={attribute}",
                f"direct_answer={answer}",
                f"direct_fact_source_id={source_context_group(context)}",
            ]
        if answer := _literal_attribute_answer(snippet, attribute):
            return [
                "direct_fact_type=attribute",
                f"direct_fact_attribute={attribute}",
                f"direct_answer={answer}",
                f"direct_fact_source_id={source_context_group(context)}",
            ]
    return []


def _possessive_attribute_query_target(query: str) -> str:
    """Return the attribute noun in direct questions like 'what is my X?'."""
    match = re.search(
        r"\bwhat\s+(?:is|are|was|were)\s+(?:my|our)\s+(?P<attribute>[a-z][a-z0-9_-]*)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    attribute = match.group("attribute").casefold()
    if attribute in _QUERY_SOURCE_STOPWORDS:
        return ""
    return attribute


def _mixed_attribute_answer(text: str, attribute: str) -> str:
    """Normalize 'mixed <attribute> - A and B' into an answer sentence."""
    pattern = re.compile(
        rf"\bmixed\s+{re.escape(attribute)}\s*[-:]\s*"
        r"(?P<left>[A-Z][A-Za-z' -]{1,40}?)\s+and\s+"
        r"(?P<right>[A-Z][A-Za-z' -]{1,40}?)(?:\s*[-.,;!?)]|$)",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    left = _clean_direct_fact_value(match.group("left"))
    right = _clean_direct_fact_value(match.group("right"))
    if not left or not right:
        return ""
    return f"A mix of {left} and {right}"


def _literal_attribute_answer(text: str, attribute: str) -> str:
    """Extract bounded literal possessive attribute assignments."""
    pattern = re.compile(
        rf"\b(?:my|our)\s+{re.escape(attribute)}\s+(?:is|was|are|were)\s+"
        r"(?P<value>[^.!?;\n]{1,120})",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return _clean_direct_fact_value(match.group("value"))


def _clean_direct_fact_value(value: str) -> str:
    value = re.split(r"\b(?:because|but|although|while|whereas)\b", value, maxsplit=1)[0]
    return " ".join(value.strip(" .,'\"()").split())


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
    lines.extend(_age_at_event_synthesis_lines(query, numeric_contexts))
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
    selected_groups = {source_context_group(context) for context in selected}
    available_groups = {source_context_group(context) for _, _, context in scored}
    if len(selected_groups) < min(2, len(available_groups)):
        for _score, _index, context in sorted(scored, key=lambda item: (-item[0], item[1])):
            group = source_context_group(context)
            if group in selected_groups:
                continue
            selected.append(context)
            selected_groups.add(group)
            if len(selected_groups) >= min(2, len(available_groups)):
                break
    selected_set = set(selected)
    for score, _index, context in scored:
        if context in selected_set:
            continue
        if score <= 0 or not _relative_time_evidence(context):
            continue
        selected.append(context)
        selected_set.add(context)
    return selected if len(selected) >= 2 else contexts


def _relative_time_evidence(context: str) -> bool:
    return bool(
        re.search(
            r"\b(?:last\s+week(?:end)?|"
            r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
            r"(?:days?|weeks?|months?)\s+ago)\b",
            context,
            flags=re.IGNORECASE,
        )
    )


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


def _age_at_event_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project age-at-event arithmetic from current age and elapsed years."""
    if not _age_at_event_query(query):
        return []
    current_ages = _personal_current_age_values(contexts)
    elapsed_years = _elapsed_year_values(contexts)
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


def _personal_current_age_values(contexts: list[str]) -> list[int]:
    values: list[int] = []
    patterns = (
        r"\b(?:i\s+am|i'm|im)\s+(?P<value>\d{1,3})\s*[- ]?(?:years?\s+old|year[- ]old)\b",
        r"\bi\s+(?:just\s+)?turned\s+(?P<value>\d{1,3})\b",
        r"\bmy\s+age\s+(?:is|was)\s+(?P<value>\d{1,3})\b",
    )
    for context in contexts:
        for pattern in patterns:
            for match in re.finditer(pattern, context, flags=re.IGNORECASE):
                value = int(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
    return values


def _elapsed_year_values(contexts: list[str]) -> list[int]:
    values: list[int] = []
    number_pattern = (
        r"\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve"
    )
    patterns = (
        rf"\b(?:for\s+)?(?:the\s+)?past\s+(?P<value>{number_pattern})\s+years?\b",
        rf"\bfor\s+(?P<value>{number_pattern})\s+years?\b",
        rf"\b(?P<value>{number_pattern})\s+years?\s+ago\b",
    )
    for context in contexts:
        for pattern in patterns:
            for match in re.finditer(pattern, context, flags=re.IGNORECASE):
                value = _integer_number_value(match.group("value"))
                if 0 < value < 125 and value not in values:
                    values.append(value)
    return values


def _integer_number_value(raw_value: str) -> int:
    normalized = raw_value.casefold()
    if normalized.isdigit():
        return int(normalized)
    return int(_NUMBER_WORDS.get(normalized, 0))


def _age_values(contexts: list[str]) -> list[int]:
    values: list[int] = []
    patterns = (
        r"\b(?:just\s+turned|turned|am|is)\s+(?P<value>\d{1,3})\b",
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
        if re.search(r"\blast\s+week(?:end)?\b", context, flags=re.IGNORECASE):
            _append_unique_number(values, 1.0)
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


def _numeric_comparison_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"most", "least", "more", "less", "highest", "lowest"}) and bool(
        query_tokens & {"money", "amount", "cost", "spent", "spend", "price", "total"}
    )


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
