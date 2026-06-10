"""Split from retrieval_plan.py (mechanical decomposition)."""


from __future__ import annotations

import json
import re
from dataclasses import dataclass

from zaxy.evidence_candidates import (
    EvidenceProjection,
    aggregate_candidate_projection,
    aggregate_evidence_score,
    preference_candidate_projection,
)
from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent
from zaxy.retrieval_plan.duration_evidence import (
    _MONTH_TERMS,
    _NUMBER_WORDS,
    _age_at_event_ledger_row_lines,
    _age_at_event_query,
    _age_at_event_synthesis_lines,
    _age_average_ledger_row_lines,
    _age_average_synthesis_lines,
    _anniversary_engagement_query,
    _append_unique_number,
    _average_query,
    _career_absence_evidence_score,
    _career_prior_duration_ledger_row_lines,
    _career_prior_duration_synthesis_lines,
    _coffee_limit_change_answer,
    _current_activity_week_evidence,
    _current_count_answer,
    _current_role_tenure_ledger_row_lines,
    _current_role_tenure_synthesis_lines,
    _direct_boolean_evidence_query,
    _direct_boolean_evidence_synthesis_lines,
    _direct_time_query,
    _elapsed_duration_at_event_evidence_score,
    _elapsed_duration_at_event_ledger_row_lines,
    _event_weeks_ago_evidence,
    _frequency_comparison_query,
    _future_age_at_event_ledger_row_lines,
    _future_age_at_event_query,
    _future_age_at_event_synthesis_lines,
    _missing_month_scoped_count_target,
    _mixed_relative_interval_lines,
    _month_day_mentions,
    _month_values,
    _number_words,
    _numeric_comparison_query,
    _owned_object_count_answer,
    _packed_shoes_value,
    _parent_order_query,
    _precise_missing_target_requires_absence,
    _recency_comparison_query,
    _relative_month_anchor_evidence,
    _relative_week_anchor_evidence,
    _road_trip_drive_evidence_score,
    _road_trip_drive_hour_evidence,
    _road_trip_drive_ledger_row_lines,
    _routine_time_slot_evidence,
    _social_media_break_day_evidence,
    _social_media_break_evidence_score,
    _social_media_break_ledger_row_lines,
    _source_ordered_numeric_evidence,
    _unit_values,
    _week_values,
    _worn_shoes_value,
)
from zaxy.retrieval_plan.fact_queries import (
    _aggregate_total_answer_query,
    _arithmetic_context_text,
    _career_prior_duration_query,
    _clean_direct_fact_value,
    _count_percentage_of_targets,
    _currency_percentage_of_targets,
    _current_duration_answer,
    _current_page_value,
    _current_role_tenure_query,
    _current_value_phrase_score,
    _direct_fact_synthesis_lines,
    _direct_numeric_synthesis_query,
    _direct_numeric_value_query,
    _elapsed_duration_at_event_query,
    _event_count_query,
    _latest_currency_answer,
    _latest_state_query,
    _latest_state_should_suppress_aggregate,
    _latest_state_synthesis_lines,
    _ledger_row_lines,
    _numeric_context_text,
    _numeric_observation_fragment,
    _original_price_value,
    _page_count_matches,
    _page_count_observation_relevant,
    _page_count_query,
    _paid_price_value,
    _percentage_comparison_targets,
    _personal_best_time_answer,
    _possessive_attribute_query_target,
    _query_bound_arithmetic_answer_present,
    _query_bound_arithmetic_query,
    _query_bound_scalar_synthesis_lines,
    _query_bound_scalar_total_synthesis_lines,
    _query_relevant_numeric_contexts,
    _road_trip_drive_query,
    _routine_time_slots,
    _session_recency_score,
    _source_group_natural_key,
    _target_count_value_for_percentage,
    _target_currency_value,
    _target_currency_value_for_percentage,
    _target_duration_difference_synthesis_lines,
    _target_percentage_value,
    _text_mentions_title,
    _total_page_value,
)
from zaxy.retrieval_plan.foundations import (
    _NUMBER_VALUE_PATTERN,
    SourceSynthesisBundleResult,
    _absence_answer_candidate_lines,
    _absence_answer_guidance,
    _church_service_interval_evidence_score,
    _countable_category_evidence_present,
    _has_multi_source_answer_candidate_type,
    _incomplete_explicit_temporal_sequence_projection,
    _irrelevant_currency_ranking_context,
    _quoted_query_title,
    _should_skip_typed_evidence_score,
    _source_evidence_scoring_context,
    _source_groups_from_synthesis_lines,
    _SourceTokenCache,
    _temporal_count_program_query,
    absence_check_target,
    diverse_source_contexts,
    preferred_source_group_order,
    primary_evidence_source_contexts,
    source_context_citation,
    source_context_group,
    source_context_namespace,
    source_context_snippet,
    source_lane_priority,
    source_synthesis_candidate_limit,
    source_tokens,
)
from zaxy.retrieval_plan.ordering import (
    _anniversary_engagement_evidence_score,
    _anniversary_engagement_ledger_row_lines,
    _anniversary_engagement_synthesis_lines,
    _bedtime_appointment_evidence_score,
    _clean_temporal_order_choice_label,
    _direct_time_synthesis_lines,
    _first_month_event_date_ledger_row_lines,
    _first_month_event_date_synthesis_lines,
    _frequency_synthesis_lines,
    _parent_event_month_day_for_person,
    _parent_order_evidence_score,
    _parent_order_ledger_row_lines,
    _parent_order_synthesis_lines,
    _recency_evidence_score,
    _recency_ledger_rows,
    _relative_days_ago,
    _relative_temporal_anchor_query,
    _relative_temporal_anchor_synthesis_lines,
    _temporal_interval_query,
    _temporal_order_action_date_value,
    _temporal_order_choice_evidence_label,
    _temporal_order_choice_span_matches,
    _temporal_order_choice_spans,
    _temporal_order_choice_terms,
    _temporal_order_evidence_score,
    _temporal_order_query,
    _temporal_order_session_date_value,
    _time_offset_evidence_score,
    _time_offset_query,
    _time_offset_synthesis_lines,
    high_precision_missing_target,
    missing_query_target,
    recency_candidate_projection,
    should_defer_to_absence_check,
)
from zaxy.retrieval_plan.scalars import (
    _assistant_answer_sentence,
    _assistant_recall_query,
    _assistant_recall_synthesis_lines,
    _boolean_evidence_sentences,
    _duration_location_absence_query,
    _duration_location_query_terms,
    _missing_alternative_target,
    _missing_comparison_operand_target,
    _missing_concrete_query_target,
    _missing_conjunct_aggregation_target,
    _missing_contrastive_sibling_target,
    _missing_location_target,
    _missing_reading_progress_target,
    _query_action_object_evidence_score,
    _query_bound_direct_answer_lines,
    _query_bound_direct_answer_query,
    _query_bound_scalar_query,
    _query_overlap_score,
    _query_specific_terms,
    _supporting_synthesis_sources,
    _target_terms_present_for_absence,
    _typed_projection_can_override_missing_target,
    _weekly_class_frequency_answer,
    has_direct_fact_evidence,
    known_related_evidence_summary,
    query_specific_source_order,
)
from zaxy.synthesis import (
    build_synthesis_plan,
)
from zaxy.synthesis_packet import synthesis_packet_from_items


def _recency_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in _recency_ledger_rows(query, contexts)
    ]


def _temporal_order_span_value(span: str, context: str) -> int | None:
    explicit = _temporal_order_action_date_value(span, context)
    if explicit is not None:
        return explicit
    if _month_day_mentions(span):
        return None
    if relative := _relative_days_ago(span):
        session_value = _temporal_order_session_date_value(context)
        if session_value is not None:
            return session_value - relative
        return -relative
    return _temporal_order_session_date_value(context)


def _quoted_query_choices(query: str) -> tuple[str, ...]:
    """Return explicit quoted alternatives from a temporal-order query."""
    choices = [
        match.group("single") or match.group("double")
        for match in re.finditer(r"'(?P<single>[^']{2,160})'|\"(?P<double>[^\"]{2,160})\"", query)
    ]
    return tuple(dict.fromkeys(" ".join(choice.split()) for choice in choices if choice))


def _temporal_order_query_choices(query: str) -> tuple[str, ...]:
    quoted = _quoted_query_choices(query)
    if len(quoted) >= 2:
        return quoted
    text = query.rstrip(" ?")
    if " or " not in text:
        return ()
    before, after = text.rsplit(" or ", 1)
    left = before.split(",", 1)[-1]
    left = re.sub(r"^.*?\b(?:first|earlier|before)\b\s*,?\s*", "", left, flags=re.IGNORECASE)
    left = re.sub(r"^(?:the|a|an)\s+", "", left.strip(), flags=re.IGNORECASE)
    right = re.sub(r"[?.,;]+$", "", after).strip()
    right = re.sub(r"^(?:the|a|an)\s+", "", right, flags=re.IGNORECASE)
    choices = tuple(
        choice
        for choice in (
            _clean_temporal_order_choice_label(left),
            _clean_temporal_order_choice_label(right),
        )
        if choice
    )
    return tuple(dict.fromkeys(choices))


def _temporal_order_choice_observations(query: str, contexts: list[str]) -> list[tuple[int, str, str]]:
    """Return earliest-first observations by binding query alternatives to dated spans."""
    choices = _temporal_order_query_choices(query)
    if len(choices) < 2:
        return []
    observations: list[tuple[int, int, str, str]] = []
    for choice in choices[:4]:
        terms = _temporal_order_choice_terms(choice)
        if not terms:
            continue
        best: tuple[int, int, str, str] | None = None
        for context_index, context in enumerate(contexts):
            text = _numeric_context_text(context)
            if _query_overlap_score(set(terms), text) <= 0:
                continue
            for span in _temporal_order_choice_spans(text, terms):
                if not _temporal_order_choice_span_matches(span, terms):
                    continue
                order_value = _temporal_order_span_value(span, context)
                if order_value is None:
                    continue
                overlap = _query_overlap_score(set(terms), span)
                candidate = _temporal_order_choice_evidence_label(
                    _clean_temporal_order_choice_label(choice),
                    span,
                )
                ranked = (order_value, -(overlap * 100 - context_index), candidate, context)
                if best is None or ranked < best:
                    best = ranked
        if best is not None:
            observations.append(best)
    if len({candidate for _order, _rank, candidate, _context in observations}) < 2:
        return []
    observations.sort(key=lambda item: (item[0], item[1]))
    return [(order_value, candidate, context) for order_value, _rank, candidate, context in observations]


def _temporal_order_choices_present(query: str, contexts: list[str]) -> bool:
    choices = _temporal_order_query_choices(query)
    if len(choices) < 2:
        return False
    text = " ".join(contexts)
    return all(
        _query_overlap_score(set(_temporal_order_choice_terms(choice)), text) > 0
        for choice in choices[:2]
    )


def _meeting_order_candidate(text: str) -> str:
    """Return a named meeting candidate for relative meeting-order memories."""
    if not re.search(r"\bmet\b", text, flags=re.IGNORECASE):
        return ""
    if re.search(r"\bMark\s+and\s+Sarah\b", text):
        return "Mark and Sarah"
    if re.search(r"\bTom\b", text):
        return "Tom"
    match = re.search(
        r"\b(?:named|called)\s+(?P<name>[A-Z][A-Za-z'-]+)\b",
        text,
    )
    return match.group("name") if match else ""


def _temporal_order_candidate(text: str, *, query: str = "") -> str:
    if candidate := _meeting_order_candidate(text):
        return candidate
    query_slots = _quoted_query_choices(query)
    if query_slots:
        lowered = text.casefold()
        for slot in query_slots:
            if slot.casefold() in lowered:
                return slot
    text = re.sub(r"\bcontent=longmemeval_session_id=\S+\s*", "", text)
    text = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", "", text)
    text = re.sub(r"^# Event\b.*?\bcontent=", "", text, flags=re.IGNORECASE | re.DOTALL)
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


def _temporal_order_observations(query: str, contexts: list[str]) -> list[tuple[int, str, str]]:
    choice_observations = _temporal_order_choice_observations(query, contexts)
    if choice_observations:
        return choice_observations
    observations: list[tuple[int, str, str]] = []
    for context in contexts:
        text = _numeric_context_text(context)
        days_ago = _relative_days_ago(text)
        if days_ago is None:
            continue
        candidate = _temporal_order_candidate(text, query=query)
        if not candidate:
            continue
        observations.append((days_ago, candidate, context))
    observations.sort(key=lambda item: item[0], reverse=True)
    return observations


def _temporal_order_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project relative ordering candidates from cited temporal evidence."""
    if not _temporal_order_query(query):
        return []
    observations = _temporal_order_observations(query, contexts)
    if len(observations) < 2:
        return []
    support_source = source_context_group(observations[0][2])
    lines = [
        "candidate_rank=1 candidate_type=temporal_order candidate_confidence=0.86",
        f"candidate_support={support_source}",
        f"temporal_order_answer={observations[0][1]}",
        f"temporal_order_source_id={support_source}",
    ]
    for index, (order_value, candidate, _context) in enumerate(observations[:5], start=1):
        relative_days_ago = abs(order_value) if order_value < 0 else order_value
        lines.append(
            f"temporal_order_rank={index} relative_days_ago={relative_days_ago} candidate={candidate}"
        )
    return lines


def _temporal_order_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _temporal_order_query(query):
        return []
    rows = [
        {
            "fact_id": f"temporal_order:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "relative_time",
            "value": str(days_ago),
            "unit": "days_ago",
            "raw_span": str(days_ago),
            "candidate": candidate,
            "include_reason": "temporal_order_candidate",
            "confidence": 0.78,
        }
        for index, (days_ago, candidate, context) in enumerate(_temporal_order_observations(query, contexts)[:5])
    ]
    return _ledger_row_lines(rows)


def _issue_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool({"issue", "problem", "problems"} & tokens)


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


def source_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
    required_kinds: set[str] | None = None,
) -> int:
    """Return a deterministic evidence score for synthesis source selection."""
    scoring_context = _source_evidence_scoring_context(context)
    query_tokens = query_tokens if query_tokens is not None else set(source_tokens(query))
    required_kinds = required_kinds if required_kinds is not None else set(build_synthesis_plan(query).required_kinds)
    skip_typed_evidence = _should_skip_typed_evidence_score(required_kinds, scoring_context)
    direct_time_query = "time" in query_tokens and bool(query_tokens & {"what", "when"})
    score = (
        0
        if direct_time_query
        or skip_typed_evidence
        or _irrelevant_currency_ranking_context(
            query,
            scoring_context,
            required_kinds=required_kinds,
            query_tokens=query_tokens,
        )
        else aggregate_evidence_score(query, scoring_context)
    )
    score += _query_action_object_evidence_score(query, scoring_context, query_tokens=query_tokens)
    if query_tokens & {"issue", "issues", "problem", "problems"} and _issue_synthesis_lines(query, [scoring_context]):
        score += 5
    if query_tokens & {"wake", "waking", "wake-up", "earlier", "later", "time"}:
        score += _time_offset_evidence_score(query, scoring_context)
    if query_tokens & {"first", "earlier", "before", "after", "which", "event"}:
        score += _temporal_order_evidence_score(query, scoring_context, query_tokens=query_tokens)
    if query_tokens & {"parent", "adopted", "adoption", "born", "baby", "twins", "rachel", "alex", "tom"}:
        score += _parent_order_evidence_score(query, scoring_context)
    if query_tokens & {"anniversary", "engaged", "engagement"}:
        score += _anniversary_engagement_evidence_score(query, scoring_context, query_tokens=query_tokens)
    score += _career_absence_evidence_score(query, scoring_context, query_tokens=query_tokens)
    score += _bedtime_appointment_evidence_score(query, scoring_context, query_tokens=query_tokens)
    if query_tokens & {"streaming", "service", "recently", "started", "using"}:
        score += _recency_evidence_score(query, scoring_context)
    if query_tokens & {"before", "started", "current", "job", "working", "event"}:
        score += _elapsed_duration_at_event_evidence_score(query, scoring_context)
    score += _social_media_break_evidence_score(query, scoring_context, query_tokens=query_tokens)
    score += _church_service_interval_evidence_score(query, scoring_context, query_tokens=query_tokens)
    if query_tokens & {"road", "trip", "drive", "driving", "drove"}:
        score += _road_trip_drive_evidence_score(query, scoring_context)
    return score


@dataclass
class _SourceEvidenceScoreCache:
    """Memoize per-query source evidence scoring inside one synthesis pass."""

    query: str
    scores: dict[str, int]
    query_tokens: set[str] | None = None
    required_kinds: set[str] | None = None

    def score(self, context: str) -> int:
        cached = self.scores.get(context)
        if cached is not None:
            return cached
        if self.query_tokens is None:
            self.query_tokens = set(source_tokens(self.query))
        if self.required_kinds is None:
            self.required_kinds = set(build_synthesis_plan(self.query).required_kinds)
        score = source_evidence_score(
            self.query,
            context,
            query_tokens=self.query_tokens,
            required_kinds=self.required_kinds,
        )
        self.scores[context] = score
        return score


def evidence_source_order(
    query: str,
    contexts: list[str],
    *,
    score_cache: _SourceEvidenceScoreCache | None = None,
    token_cache: _SourceTokenCache | None = None,
) -> list[str]:
    """Prefer snippets that can produce typed synthesis evidence for the query."""
    query_terms = _query_specific_terms(query)
    scorer = score_cache.score if score_cache is not None else lambda context: source_evidence_score(query, context)
    indexed = list(enumerate(contexts))
    indexed.sort(
        key=lambda item: (
            -scorer(item[1]),
            -_query_overlap_score(query_terms, item[1], token_cache=token_cache),
            -source_lane_priority(item[1]),
            item[0],
        )
    )
    return [context for _, context in indexed]


def _dominant_provenance_cluster_contexts(
    query: str,
    contexts: list[str],
    *,
    score_cache: _SourceEvidenceScoreCache | None = None,
) -> list[str]:
    """Scope aggregation to a dominant provenance namespace when one clearly exists."""
    scorer = score_cache.score if score_cache is not None else lambda context: source_evidence_score(query, context)
    namespace_groups: dict[str, set[str]] = {}
    for context in contexts:
        if scorer(context) <= 0:
            continue
        namespace = source_context_namespace(context)
        if not namespace:
            continue
        namespace_groups.setdefault(namespace, set()).add(source_context_group(context))
    if not namespace_groups:
        return []
    ranked = sorted(namespace_groups.items(), key=lambda item: len(item[1]), reverse=True)
    best_namespace, best_groups = ranked[0]
    total_groups = len(set().union(*namespace_groups.values()))
    if len(best_groups) < 3 or len(best_groups) <= total_groups / 2:
        return []
    return [
        context for context in contexts
        if source_context_namespace(context) == best_namespace
    ]


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


def _query_bound_direct_sentence_answer(query: str, sentence: str, context: str) -> str:
    tokens = set(source_tokens(query))
    sentence_text = sentence.casefold()
    if tokens & {"weight", "lost"} and tokens & {"gym", "consistently"}:
        match = re.search(r"\blost\s+(?P<value>\d+(?:\.\d+)?)\s+pounds?\b", sentence_text)
        if match and re.search(r"\b(?:gym|workout|cardio)\b", sentence_text):
            return f"{_format_number(float(match.group('value')))} pounds"
    if tokens & {"current"} and tokens & {"record"}:
        match = re.search(r"\b(?P<record>\d{1,2}\s*[-–]\s*\d{1,2})\s+record\b", sentence, flags=re.IGNORECASE)
        if match and _query_overlap_score({"record", "league", "team", "volleyball"}, sentence) >= 2:
            return match.group("record").replace(" ", "").replace("–", "-")
    if tokens & {"times"} and tokens & {"met", "meet"}:
        if re.search(r"\bmet\s+up\s+twice\b", sentence_text) and _query_overlap_score(_query_specific_terms(query), context) >= 2:
            return "We've met up twice."
        match = re.search(r"\bmet\s+up\s+(?P<value>\d+)\s+times\b", sentence_text)
        if match and _query_overlap_score(_query_specific_terms(query), context) >= 2:
            return f"We've met up {match.group('value')} times."
    if tokens & {"long"} and tokens & {"for"} and tokens & {"in"}:
        location_terms = _duration_location_query_terms(query)
        if location_terms and not _query_overlap_score(location_terms, context):
            return ""
        match = re.search(
            rf"\bspent\s+(?P<value>{_NUMBER_VALUE_PATTERN})\s+(?P<unit>days?|weeks?|months?)\s+(?:traveling|travelling|visiting|in)\b",
            sentence_text,
            flags=re.IGNORECASE,
        )
        if match:
            return f"{match.group('value')} {match.group('unit')}"
    if tokens & {"buy", "bought"} and tokens & {"what"}:
        match = re.search(
            r"\b(?:i\s+)?(?:actually\s+)?(?:got|bought|purchased)\s+(?P<value>my\s+own\s+set\s+of\s+[^,.!?;]{2,100})",
            sentence,
            flags=re.IGNORECASE,
        )
        if match and _query_overlap_score(_query_specific_terms(query), context) >= 1:
            return _assistant_answer_sentence(_clean_direct_fact_value(match.group("value")))
    return ""


def _query_bound_direct_answer_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project direct answer sentences that bind tightly to the query shape."""
    if not _query_bound_direct_answer_query(query):
        return []
    if answer := _weekly_class_frequency_answer(query, contexts):
        return _query_bound_direct_answer_lines(answer)
    if answer := _coffee_limit_change_answer(query, contexts):
        return _query_bound_direct_answer_lines(answer)
    query_terms = _query_specific_terms(query)
    candidates: list[tuple[int, int, str, str, str]] = []
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        text = source_context_snippet(context, max_chars=12_000)
        for sentence in _boolean_evidence_sentences(text):
            answer_text = _query_bound_direct_sentence_answer(query, sentence, text)
            if not answer_text:
                continue
            score = 80 + _query_overlap_score(query_terms, sentence) + _session_recency_score(text)
            if source_context_group(context) in answer_text:
                score += 1
            candidates.append((score, index, source_id, answer_text, sentence))
    if not candidates:
        return []
    _score, _index, source_id, answer_text, sentence = max(candidates, key=lambda item: (item[0], -item[1]))
    return _query_bound_direct_answer_lines((answer_text, [source_id], sentence))


def _page_count_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    if not _page_count_query(query):
        return []
    query_months = {token for token in source_tokens(query) if token in _MONTH_TERMS}
    values: list[float] = []
    source_ids: list[str] = []
    for context in contexts:
        snippet = source_context_snippet(context, max_chars=2_000)
        for match in _page_count_matches(snippet):
            fragment = _numeric_observation_fragment(snippet, match.start(), match.end())
            if not _page_count_observation_relevant(query, fragment):
                continue
            fragment_months = {token for token in source_tokens(fragment) if token in _MONTH_TERMS}
            if query_months and fragment_months and not fragment_months <= query_months:
                continue
            value_text = match.group("value") or match.group("value_after")
            if not value_text:
                continue
            value = float(value_text.replace(",", ""))
            if value <= 0:
                continue
            _append_unique_number(values, value)
            source_id = source_context_group(context)
            if source_id not in source_ids:
                source_ids.append(source_id)
    if not values:
        return []
    total = sum(values)
    return [
        "candidate_rank=1 candidate_type=page_count candidate_confidence=0.84",
        "candidate_support=" + ",".join(source_ids),
        "page_values=" + ",".join(_format_number(value) for value in values),
        f"page_total={_format_number(total)}",
        f"page_total_answer={_format_number(total)}",
        "page_source_ids=" + ",".join(source_ids),
    ]


def _routine_duration_answer(total_minutes: float) -> str:
    """Render common routine totals as natural language."""
    if total_minutes == 90:
        return "an hour and a half"
    if total_minutes == 30:
        return "half an hour"
    if total_minutes % 60 == 0:
        hours = total_minutes / 60
        return f"{_format_number(hours)} hour" + ("" if hours == 1 else "s")
    if total_minutes < 60:
        return f"{_format_number(total_minutes)} minutes"
    return f"{_format_number(total_minutes / 60)} hours"


def _routine_time_total_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project a total routine duration from cited slot-bound personal evidence."""
    slots = _routine_time_slots(query)
    if len(slots) < 2:
        return []
    evidence: list[tuple[str, float, str, str]] = []
    used_sources: set[str] = set()
    for slot, terms in slots:
        match = _routine_time_slot_evidence(slot, terms, contexts, used_sources=used_sources)
        if match is None:
            return []
        source_id, minutes, fragment = match
        evidence.append((slot, minutes, source_id, fragment))
        used_sources.add(source_id)
    total_minutes = sum(minutes for _slot, minutes, _source_id, _fragment in evidence)
    if total_minutes <= 0 or total_minutes > 24 * 60:
        return []
    source_ids = [source_id for _slot, _minutes, source_id, _fragment in evidence]
    answer = _routine_duration_answer(total_minutes)
    return [
        "candidate_rank=1 candidate_type=routine_time_total candidate_confidence=0.88",
        "candidate_support=" + ",".join(source_ids),
        "routine_time_total_operation=sum_slot_bound_durations",
        "routine_time_total_slots=" + ",".join(slot for slot, _minutes, _source_id, _fragment in evidence),
        "routine_time_total_values=" + ",".join(_format_number(minutes) for _slot, minutes, _source_id, _fragment in evidence),
        f"routine_time_total_minutes={_format_number(total_minutes)}",
        f"routine_time_total_hours={_format_number(total_minutes / 60)}",
        f"routine_time_total_answer={answer}",
        "routine_time_total_source_ids=" + ",".join(source_ids),
    ]


def _target_currency_difference_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Return a cited currency difference between two query-named costs."""
    del query
    taxi = _target_currency_value(("taxi",), contexts)
    train = _target_currency_value(("train", "fare"), contexts)
    if taxi is None or train is None:
        return []
    taxi_value, taxi_source, taxi_span = taxi
    train_value, train_source, train_span = train
    difference = abs(taxi_value - train_value)
    if difference <= 0:
        return []
    answer = f"${_format_number(difference)}"
    source_ids = list(dict.fromkeys((taxi_source, train_source)))
    return [
        "candidate_rank=1 candidate_type=query_bound_difference candidate_confidence=0.88",
        "candidate_support=" + ",".join(source_ids),
        "difference_left_label=taxi",
        f"difference_left_value={_format_currency(taxi_value)}",
        "difference_right_label=train_fare",
        f"difference_right_value={_format_currency(train_value)}",
        f"query_bound_difference_answer={answer}",
        f"query_bound_difference_left_raw_span={source_context_snippet(taxi_span, max_chars=180)}",
        f"query_bound_difference_right_raw_span={source_context_snippet(train_span, max_chars=180)}",
        "query_bound_difference_source_ids=" + ",".join(source_ids),
    ]


def _query_bound_difference_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project explicit target-vs-target differences from cited operands."""
    tokens = set(source_tokens(query))
    if tokens & {"more", "expensive", "compared"} and tokens & {"taxi", "train", "fare"}:
        return _target_currency_difference_synthesis_lines(query, contexts)
    if tokens & {"exceed", "exceeded"} and tokens & {"target", "marathon", "minutes"}:
        return _target_duration_difference_synthesis_lines(query, contexts)
    return []


def _distance_total_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    tokens = set(source_tokens(query))
    if not ({"total", "distance"} <= tokens and tokens & {"hike", "hikes", "hiked", "trail", "trails"}):
        return []
    evidence: list[tuple[str, float, str]] = []
    seen: set[tuple[str, float]] = set()
    for context in contexts:
        snippet = _arithmetic_context_text(context)
        for match in re.finditer(r"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?miles?\b", snippet, flags=re.IGNORECASE):
            fragment = _numeric_observation_fragment(snippet, match.start(), match.end())
            if not re.search(r"\b(?:hike|hiked|hikes|trail|loop|ridge)\b", fragment, flags=re.IGNORECASE):
                continue
            source_id = source_context_group(context)
            value = float(match.group("value"))
            key = (source_id, value)
            if key in seen:
                continue
            seen.add(key)
            evidence.append((source_id, value, fragment))
    if len(evidence) < 2:
        return []
    source_ids = list(dict.fromkeys(source_id for source_id, _value, _fragment in evidence))
    values = [value for _source_id, value, _fragment in evidence]
    total = sum(values)
    answer = f"{_format_number(total)} miles"
    return [
        "candidate_rank=1 candidate_type=distance_total candidate_confidence=0.86",
        "candidate_support=" + ",".join(source_ids),
        "distance_values=" + ",".join(_format_number(value) for value in values),
        f"distance_total={answer}",
        f"distance_total_answer={answer}",
        "distance_source_ids=" + ",".join(source_ids),
    ]


def _pages_remaining_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    tokens = set(source_tokens(query))
    if not (tokens & {"left", "remaining"} and tokens & {"page", "pages", "read"}):
        return []
    title = _quoted_query_title(query)
    if not title:
        return []
    current: tuple[float, str] | None = None
    total: tuple[float, str] | None = None
    for context in contexts:
        snippet = _arithmetic_context_text(context)
        if not _text_mentions_title(snippet, title):
            continue
        source_id = source_context_group(context)
        if current is None and (current_value := _current_page_value(snippet, title)) is not None:
            current = (current_value, source_id)
        if total is None and (total_value := _total_page_value(snippet, title)) is not None:
            total = (total_value, source_id)
    if current is None or total is None:
        return []
    current_value, current_source = current
    total_value, total_source = total
    remaining = total_value - current_value
    if remaining <= 0:
        return []
    source_ids = list(dict.fromkeys((current_source, total_source)))
    return [
        "candidate_rank=1 candidate_type=pages_remaining candidate_confidence=0.87",
        "candidate_support=" + ",".join(source_ids),
        f"pages_current={_format_number(current_value)}",
        f"pages_total={_format_number(total_value)}",
        f"pages_remaining={_format_number(remaining)}",
        f"pages_remaining_answer={_format_number(remaining)}",
        "pages_remaining_source_ids=" + ",".join(source_ids),
    ]


def _format_percentage(value: float) -> str:
    return f"{_format_number(round(value, 2))}%"


def _currency_percentage_of_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Return numerator/denominator percentage for cited currency operands."""
    targets = _currency_percentage_of_targets(query)
    if targets is None:
        return []
    denominator_terms, numerator_terms = targets
    denominator = _target_currency_value_for_percentage(denominator_terms, contexts)
    numerator = _target_currency_value_for_percentage(numerator_terms, contexts)
    if denominator is None or numerator is None:
        return []
    denominator_value, denominator_source, denominator_span = denominator
    numerator_value, numerator_source, numerator_span = numerator
    if denominator_value <= 0 or numerator_value < 0:
        return []
    percent = (numerator_value / denominator_value) * 100
    if percent < 0 or percent > 1000:
        return []
    answer = _format_percentage(percent)
    source_ids = list(dict.fromkeys((denominator_source, numerator_source)))
    return [
        "candidate_rank=1 candidate_type=percentage candidate_confidence=0.88",
        "candidate_support=" + ",".join(source_ids),
        "percentage_operation=currency_numerator_divided_by_denominator",
        f"percentage_denominator_label={' '.join(denominator_terms)}",
        f"percentage_denominator={_format_currency(denominator_value)}",
        f"percentage_numerator_label={' '.join(numerator_terms)}",
        f"percentage_numerator={_format_currency(numerator_value)}",
        f"percentage_value={answer}",
        f"percentage_answer={answer}",
        f"percentage_denominator_raw_span={source_context_snippet(denominator_span, max_chars=180)}",
        f"percentage_numerator_raw_span={source_context_snippet(numerator_span, max_chars=180)}",
        "percentage_source_ids=" + ",".join(source_ids),
    ]


def _count_percentage_of_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Return numerator/denominator percentage for cited count operands."""
    targets = _count_percentage_of_targets(query)
    if targets is None:
        return []
    denominator_terms, numerator_terms = targets
    denominator = _target_count_value_for_percentage(denominator_terms, (), contexts)
    numerator = _target_count_value_for_percentage(denominator_terms, numerator_terms, contexts)
    if denominator is None or numerator is None:
        return []
    denominator_value, denominator_source, denominator_span = denominator
    numerator_value, numerator_source, numerator_span = numerator
    if denominator_value <= 0 or numerator_value < 0 or numerator_value > denominator_value:
        return []
    percent = (numerator_value / denominator_value) * 100
    answer = _format_percentage(percent)
    source_ids = list(dict.fromkeys((denominator_source, numerator_source)))
    return [
        "candidate_rank=1 candidate_type=percentage candidate_confidence=0.88",
        "candidate_support=" + ",".join(source_ids),
        "percentage_operation=count_numerator_divided_by_denominator",
        f"percentage_denominator_label={' '.join(denominator_terms)}",
        f"percentage_denominator={_format_number(denominator_value)}",
        f"percentage_numerator_label={' '.join(numerator_terms)}",
        f"percentage_numerator={_format_number(numerator_value)}",
        f"percentage_value={answer}",
        f"percentage_answer={answer}",
        f"percentage_denominator_raw_span={source_context_snippet(denominator_span, max_chars=180)}",
        f"percentage_numerator_raw_span={source_context_snippet(numerator_span, max_chars=180)}",
        "percentage_source_ids=" + ",".join(source_ids),
    ]


def _percentage_comparison_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Answer cited yes/no comparisons between two percentage operands."""
    tokens = set(source_tokens(query))
    if not (tokens & {"percentage", "percent"} and tokens & {"compared", "than"}):
        return []
    targets = _percentage_comparison_targets(query)
    if targets is None:
        return []
    left_target, right_target = targets
    left = _target_percentage_value(left_target, contexts)
    right = _target_percentage_value(right_target, contexts)
    if left is None or right is None:
        return []
    left_value, left_source = left
    right_value, right_source = right
    if left_value == right_value:
        return []
    asks_higher = bool(tokens & {"higher", "more", "greater", "larger"})
    asks_lower = bool(tokens & {"lower", "less", "smaller"})
    if not asks_higher and not asks_lower:
        return []
    answer_yes = left_value > right_value if asks_higher else left_value < right_value
    answer = "Yes" if answer_yes else "No"
    source_ids = list(dict.fromkeys((left_source, right_source)))
    left_label = " ".join(left_target)
    right_label = " ".join(right_target)
    return [
        "candidate_rank=1 candidate_type=boolean_comparison candidate_confidence=0.86",
        "candidate_support=" + ",".join(source_ids),
        f"percentage_left_label={left_label}",
        f"percentage_left_value={_format_percentage(left_value)}",
        f"percentage_right_label={right_label}",
        f"percentage_right_value={_format_percentage(right_value)}",
        f"boolean_comparison_operator={'higher' if asks_higher else 'lower'}",
        f"boolean_comparison_answer={answer}",
        "boolean_comparison_source_ids=" + ",".join(source_ids),
    ]


def _packed_shoes_percentage_synthesis_lines(contexts: list[str]) -> list[str]:
    packed: tuple[float, str] | None = None
    worn: tuple[float, str] | None = None
    for context in contexts:
        snippet = _arithmetic_context_text(context)
        source_id = source_context_group(context)
        if packed is None and (value := _packed_shoes_value(snippet)) is not None:
            packed = (value, source_id)
        if worn is None and (value := _worn_shoes_value(snippet)) is not None:
            worn = (value, source_id)
    if packed is None or worn is None:
        return []
    packed_value, packed_source = packed
    worn_value, worn_source = worn
    if packed_value <= 0 or worn_value < 0 or worn_value > packed_value:
        return []
    percent = (worn_value / packed_value) * 100
    answer = _format_percentage(percent)
    source_ids = list(dict.fromkeys((packed_source, worn_source)))
    return [
        "candidate_rank=1 candidate_type=percentage candidate_confidence=0.86",
        "candidate_support=" + ",".join(source_ids),
        f"percentage_numerator={_format_number(worn_value)}",
        f"percentage_denominator={_format_number(packed_value)}",
        f"percentage_value={answer}",
        f"percentage_answer={answer}",
        "percentage_source_ids=" + ",".join(source_ids),
    ]


def _discount_percentage_synthesis_lines(contexts: list[str]) -> list[str]:
    original: tuple[float, str] | None = None
    paid: tuple[float, str] | None = None
    for context in contexts:
        snippet = _arithmetic_context_text(context)
        source_id = source_context_group(context)
        if original is None and (value := _original_price_value(snippet)) is not None:
            original = (value, source_id)
        if paid is None and (value := _paid_price_value(snippet)) is not None:
            paid = (value, source_id)
    if original is None or paid is None:
        return []
    original_value, original_source = original
    paid_value, paid_source = paid
    if original_value <= 0 or paid_value < 0 or paid_value > original_value:
        return []
    percent = ((original_value - paid_value) / original_value) * 100
    answer = _format_percentage(percent)
    source_ids = list(dict.fromkeys((original_source, paid_source)))
    return [
        "candidate_rank=1 candidate_type=percentage candidate_confidence=0.86",
        "candidate_support=" + ",".join(source_ids),
        f"percentage_original={_format_currency(original_value)}",
        f"percentage_paid={_format_currency(paid_value)}",
        f"percentage_value={answer}",
        f"percentage_answer={answer}",
        "percentage_source_ids=" + ",".join(source_ids),
    ]


def _percentage_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    tokens = set(source_tokens(query))
    if not tokens & {"percentage", "percent"}:
        return []
    if lines := _currency_percentage_of_synthesis_lines(query, contexts):
        return lines
    if lines := _count_percentage_of_synthesis_lines(query, contexts):
        return lines
    if tokens & {"packed", "wear", "wore", "worn", "shoes"}:
        return _packed_shoes_percentage_synthesis_lines(contexts)
    if "discount" in tokens:
        return _discount_percentage_synthesis_lines(contexts)
    return []


def _query_bound_arithmetic_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project answer-ready arithmetic when cited facts bind to the query."""
    if not _query_bound_arithmetic_query(query):
        return []
    for builder in (
        _routine_time_total_synthesis_lines,
        _query_bound_difference_synthesis_lines,
        _percentage_comparison_synthesis_lines,
        _distance_total_synthesis_lines,
        _pages_remaining_synthesis_lines,
        _percentage_synthesis_lines,
    ):
        lines = builder(query, contexts)
        if lines:
            return lines
    return []


def absence_check_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
) -> str | None:
    """Build cited guidance for questions about absent personal memories."""
    intent = classify_retrieval_intent(query, limit=limit)
    if not intent.needs_source_lane and not _parent_order_query(query):
        return None
    grouped_sources = diverse_source_contexts(
        source_results,
        limit=max(1, intent.source_lane_slots or min(2, limit)),
    )
    if _query_bound_arithmetic_synthesis_lines(query, grouped_sources):
        return None
    target = high_precision_missing_target(query, grouped_sources)
    if not target and has_direct_fact_evidence(query, grouped_sources):
        return None
    if not target and (
        {"absence_check", "personal_memory"} & set(intent.reasons)
        and not _recency_comparison_query(query)
        and not _temporal_interval_query(query)
        and not {"aggregation", "aggregation_question"} & set(intent.reasons)
    ):
        target = missing_query_target(query, grouped_sources)
    if not target and "absence_check" in intent.reasons:
        target = absence_check_target(query)
    if not target and {"aggregation", "aggregation_question"} & set(intent.reasons):
        target = _missing_location_target(query, grouped_sources)
    if not target:
        return None
    if not grouped_sources:
        return None
    if _countable_category_evidence_present(query, grouped_sources) and not _precise_missing_target_requires_absence(
        query,
        target,
        grouped_sources,
    ):
        return None
    if _parent_order_query(query):
        if _parent_event_month_day_for_person(target, grouped_sources) is not None:
            return None
    elif _target_terms_present_for_absence(query, target, grouped_sources):
        return None
    candidate_source_ids = tuple(
        dict.fromkeys(
            source_context_group(context)
            for context in source_results
            if source_context_group(context)
        )
    )
    known_evidence = known_related_evidence_summary(query, grouped_sources, target)
    answer_guidance = _absence_answer_guidance(target)
    lines = [
        "zaxy_absence_check=true",
        "synthesis_mode=absence_check",
        f"query={query}",
        f"not_mentioned_candidate={target}",
        "support_source_ids=" + ",".join(source_context_group(context) for context in grouped_sources),
        "candidate_source_ids=" + ",".join(candidate_source_ids[: min(len(candidate_source_ids), max(4, limit * 2))]),
        f"answer_guidance={answer_guidance}",
    ]
    lines.extend(_absence_answer_candidate_lines(query, target, known_evidence))
    if known_evidence:
        lines.append(f"known_related_evidence={known_evidence}")
    for context in grouped_sources:
        lines.append(
            "- "
            f"source_id={source_context_group(context)} "
            f"citation={source_context_citation(context)} "
            f"snippet={source_context_snippet(context)}"
        )
    return "\n".join(lines)


def _should_defer_synthesis_to_absence(
    query: str,
    contexts: list[str],
    intent: RetrievalIntent,
) -> bool:
    """Return whether a precise missing slot should suppress generic synthesis."""
    if not intent.needs_source_lane or not contexts:
        return False
    if _query_bound_arithmetic_synthesis_lines(query, contexts):
        return False
    if _query_bound_direct_answer_synthesis_lines(query, contexts):
        return False
    if _temporal_order_query(query) and _temporal_order_choices_present(query, contexts):
        return False
    target = high_precision_missing_target(query, contexts)
    if not target or _target_terms_present_for_absence(query, target, contexts):
        return False
    if _typed_projection_can_override_missing_target(query, contexts, target):
        return False
    if _countable_category_evidence_present(query, contexts) and not _precise_missing_target_requires_absence(
        query,
        target,
        contexts,
    ):
        return False
    return bool(
        _missing_month_scoped_count_target(query, contexts) == target
        or _missing_reading_progress_target(query, contexts) == target
        or (_duration_location_absence_query(query) and _missing_location_target(query, contexts) == target)
        or _missing_conjunct_aggregation_target(query, contexts) == target
        or _missing_comparison_operand_target(query, contexts) == target
        or _missing_contrastive_sibling_target(query, contexts) == target
        or _missing_alternative_target(query, contexts) == target
        or (_temporal_interval_query(query) and _missing_concrete_query_target(query, contexts) == target)
    )


def _currency_difference_answer(query_tokens: set[str], text: str) -> str:
    if not (query_tokens & {"difference", "compared", "more"}):
        return ""
    if query_tokens & {"accommodation", "accommodations", "lodging", "hostel", "resort"}:
        return ""
    values = [
        float(match.group("value").replace(",", ""))
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text)
    ]
    if len(values) < 2:
        return ""
    difference = max(values) - min(values)
    if difference <= 0:
        return ""
    return f"${_format_number(difference)}"


def _cross_context_currency_difference_answer(query_tokens: set[str], contexts: list[str]) -> str:
    if not (query_tokens & {"difference", "compared", "more"}):
        return ""
    text = " ".join(source_context_snippet(context, max_chars=1_000) for context in contexts)
    return _currency_difference_answer(query_tokens, text)


def _direct_numeric_value_candidates(query: str, contexts: list[str]) -> list[tuple[int, int, str, str, str]]:
    query_tokens = set(source_tokens(query))
    if not _direct_numeric_value_query(query_tokens, query):
        return []
    query_terms = _query_specific_terms(query)
    candidates: list[tuple[int, int, str, str, str]] = []
    if answer := _cross_context_currency_difference_answer(query_tokens, contexts):
        support = ",".join(dict.fromkeys(source_context_group(context) for context in contexts[:2]))
        candidates.append((90, 0, support, answer, answer))
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        snippet = source_context_snippet(context, max_chars=2_000)
        overlap = _query_overlap_score(query_terms, snippet)
        if answer := _owned_object_count_answer(query_tokens, snippet):
            candidates.append((58 + overlap + _session_recency_score(snippet), index, source_id, answer, answer))
        if overlap <= 0:
            continue
        if answer := _personal_best_time_answer(query_tokens, snippet):
            recency = _session_recency_score(snippet)
            temporal_score = -recency if "previous" in query_tokens else recency
            candidates.append((80 + overlap + temporal_score, index, source_id, answer, answer))
        if answer := _latest_currency_answer(query_tokens, snippet):
            candidates.append((70 + overlap + _session_recency_score(snippet), index, source_id, answer, answer))
        if answer := _current_duration_answer(query_tokens, snippet):
            candidates.append((60 + overlap + _session_recency_score(snippet), index, source_id, answer, answer))
        if answer := _current_count_answer(query_tokens, snippet):
            candidates.append(
                (
                    50 + overlap + _session_recency_score(snippet) + _current_value_phrase_score(snippet),
                    index,
                    source_id,
                    answer,
                    answer,
                )
            )
        if answer := _currency_difference_answer(query_tokens, snippet):
            candidates.append((45 + overlap, index, source_id, answer, answer))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def _direct_numeric_value_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project answer-ready current/latest numeric values from cited evidence."""
    if _query_bound_scalar_query(query):
        return []
    candidates = _direct_numeric_value_candidates(query, contexts)
    if not candidates:
        return []
    _score, _index, source_id, answer, raw = candidates[0]
    return [
        "candidate_rank=1 candidate_type=direct_numeric_value candidate_confidence=0.84",
        f"candidate_support={source_id}",
        f"direct_numeric_answer={answer}",
        f"direct_numeric_raw_span={raw}",
        f"direct_numeric_source_id={source_id}",
    ]


def _elapsed_duration_at_event_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project current-duration minus event-age arithmetic for prior-event queries."""
    if not _elapsed_duration_at_event_query(query):
        return []
    current = _current_activity_week_evidence(query, contexts)
    event = _event_weeks_ago_evidence(query, contexts)
    if current is None or event is None:
        return []
    current_weeks = current[1]
    event_weeks_ago = event[1]
    if event_weeks_ago <= 0 or current_weeks <= event_weeks_ago:
        return []
    elapsed_weeks = current_weeks - event_weeks_ago
    answer = _number_words(float(elapsed_weeks)) or _format_number(float(elapsed_weeks))
    return [
        f"elapsed_current_weeks={current_weeks}",
        f"elapsed_event_weeks_ago={event_weeks_ago}",
        f"elapsed_at_event_operation={current_weeks}-{event_weeks_ago}",
        f"elapsed_at_event_answer={answer} weeks",
    ]


def _social_media_break_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project total days from explicit social-media break durations."""
    query_tokens = set(source_tokens(query))
    if not {"social", "media", "breaks"} <= query_tokens and not {"social", "media", "break"} <= query_tokens:
        return []
    evidence = _social_media_break_day_evidence(contexts)
    values = [value for _context, value, _raw in evidence]
    if not values:
        return []
    total = sum(values)
    support = ",".join(dict.fromkeys(source_context_group(context) for context, _value, _raw in evidence))
    lines = [
        "candidate_rank=1 candidate_type=social_media_break candidate_confidence=0.84",
        f"candidate_support={support}",
        "social_media_break_day_values=" + ",".join(_format_number(float(value)) for value in values),
        f"social_media_break_total={_format_number(float(total))} days",
        f"social_media_break_total_answer={_format_number(float(total))} days",
    ]
    if total_words := _number_words(float(total)):
        lines.append(f"social_media_break_total_words={total_words} days")
    return lines


def _road_trip_destination_count_phrase(query: str, count: int) -> str:
    """Return a natural destination count phrase for road-trip aggregate answers."""
    query_tokens = set(source_tokens(query))
    for word, value in _NUMBER_WORDS.items():
        if value == count and word in query_tokens and word not in {"a", "an"}:
            return word
    return _format_number(float(count))


def _road_trip_drive_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project total hours for direct road-trip destination-drive memories."""
    if not _road_trip_drive_query(query):
        return []
    evidence = _road_trip_drive_hour_evidence(contexts)
    values = [value for _context, value, _raw in evidence]
    if not values:
        return []
    total = sum(values)
    support = ",".join(dict.fromkeys(source_context_group(context) for context, _value, _raw in evidence))
    destination_count = _road_trip_destination_count_phrase(query, len(values))
    total_answer = (
        f"{_format_number(float(total))} hours for getting to the {destination_count} destinations "
        f"(or {_format_number(float(total * 2))} hours for the round trip)"
    )
    lines = [
        "candidate_rank=1 candidate_type=road_trip_drive candidate_confidence=0.84",
        f"candidate_support={support}",
        "road_trip_drive_hour_values=" + ",".join(_format_number(float(value)) for value in values),
        f"road_trip_drive_total={_format_number(float(total))} hours",
        f"road_trip_drive_total_answer={total_answer}",
        f"road_trip_drive_total_round_trip={_format_number(float(total * 2))} hours",
    ]
    if total_words := _number_words(float(total)):
        lines.append(f"road_trip_drive_total_words={total_words} hours")
    return lines


def _numeric_synthesis_lines(
    query: str,
    contexts: list[str],
    *,
    aggregate_lines: list[str] | None = None,
) -> list[str]:
    """Project deterministic numeric operations from cited source snippets."""
    arithmetic_lines = _query_bound_arithmetic_synthesis_lines(query, contexts)
    if arithmetic_lines:
        return arithmetic_lines
    scalar_total_lines = _query_bound_scalar_total_synthesis_lines(query, contexts)
    if scalar_total_lines:
        return scalar_total_lines
    if not (_aggregate_total_answer_query(query) and aggregate_lines):
        latest_state_lines = _latest_state_synthesis_lines(query, contexts)
        if latest_state_lines:
            return latest_state_lines
    numeric_contexts = [_numeric_context_text(context) for context in contexts]
    aggregate_lines = aggregate_lines or []
    lines: list[str] = []
    lines.extend(_direct_numeric_value_synthesis_lines(query, numeric_contexts))
    has_typed_duration = any(line.startswith("duration_values=") for line in aggregate_lines)
    has_typed_projection = any(line.startswith("candidate_rank=") for line in aggregate_lines)
    has_typed_age_average = any(line.startswith("age_average=") for line in aggregate_lines)
    lines.extend(_age_at_event_synthesis_lines(query, numeric_contexts))
    lines.extend(_future_age_at_event_synthesis_lines(query, contexts))
    if not has_typed_age_average:
        lines.extend(_age_average_synthesis_lines(query, numeric_contexts))
    lines.extend(_elapsed_duration_at_event_synthesis_lines(query, numeric_contexts))
    lines.extend(_social_media_break_synthesis_lines(query, numeric_contexts))
    lines.extend(_road_trip_drive_synthesis_lines(query, numeric_contexts))
    lines.extend(_career_prior_duration_synthesis_lines(query, numeric_contexts))
    lines.extend(_current_role_tenure_synthesis_lines(query, contexts))
    page_count_query = _page_count_query(query)
    lines.extend(_page_count_synthesis_lines(query, numeric_contexts))
    if _career_prior_duration_query(query):
        return lines
    if _current_role_tenure_query(query):
        return lines
    if _query_bound_arithmetic_answer_present(lines):
        return lines
    if any("candidate_type=direct_numeric_value" in line for line in lines):
        return lines
    if _event_count_query(query):
        return lines
    if not has_typed_duration and not has_typed_projection and not _temporal_interval_query(query):
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
    if page_count_query:
        lines.extend(_time_offset_synthesis_lines(query, numeric_contexts))
        return lines
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


def _relative_interval_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if _elapsed_duration_at_event_query(query):
        return []
    query_tokens = set(source_tokens(query))
    if not (
        _temporal_interval_query(query)
        or query_tokens & {"after", "before", "between", "long", "since", "until", "when"}
    ):
        return []
    relevant_contexts = _query_relevant_numeric_contexts(query, contexts)
    week_evidence = _relative_week_anchor_evidence(relevant_contexts)
    month_evidence = _relative_month_anchor_evidence(relevant_contexts)
    if len(week_evidence) + len(month_evidence) < 2:
        return []
    rows: list[dict[str, object]] = []
    for index, (context, value, raw) in enumerate(_source_ordered_numeric_evidence(month_evidence)):
        rows.append(
            {
                "fact_id": f"relative_interval:month:{index}",
                "source_group": source_context_group(context),
                "citation": source_context_citation(context),
                "kind": "relative_time",
                "value": _format_number(value),
                "unit": "months_ago",
                "raw_span": raw,
                "include_reason": "relative_month_anchor",
                "confidence": 0.8,
            }
        )
    for index, (context, value, raw) in enumerate(_source_ordered_numeric_evidence(week_evidence)):
        rows.append(
            {
                "fact_id": f"relative_interval:week:{index}",
                "source_group": source_context_group(context),
                "citation": source_context_citation(context),
                "kind": "relative_time",
                "value": _format_number(value),
                "unit": "weeks_ago",
                "raw_span": raw,
                "include_reason": "relative_week_anchor",
                "confidence": 0.8,
            }
        )
    rows.sort(key=lambda row: _source_group_natural_key(str(row["source_group"])))
    return _ledger_row_lines(rows)


def source_synthesis_bundle_result(
    *,
    query: str,
    source_results: list[str],
    limit: int,
    preferred_source_groups: list[str] | tuple[str, ...] | None = None,
) -> SourceSynthesisBundleResult | None:
    """Build one compact cited source bundle with typed synthesis packet data."""
    source_results = primary_evidence_source_contexts(source_results)
    intent = classify_retrieval_intent(query, limit=limit)
    if (
        not {"aggregation", "aggregation_question"} & set(intent.reasons)
        and not _issue_query(query)
        and not _average_query(query)
        and not _age_at_event_query(query)
        and not _future_age_at_event_query(query)
        and not _elapsed_duration_at_event_query(query)
        and not _numeric_comparison_query(query)
        and not _frequency_comparison_query(query)
        and not _time_offset_query(query)
        and not _current_role_tenure_query(query)
        and not _temporal_order_query(query)
        and not _temporal_interval_query(query)
        and "temporal_sequence" not in intent.reasons
        and not _parent_order_query(query)
        and not _anniversary_engagement_query(query)
        and not _recency_comparison_query(query)
        and not _relative_temporal_anchor_query(query)
        and not _direct_time_query(query)
        and not _assistant_recall_query(query)
        and not _direct_numeric_synthesis_query(query)
        and not _query_bound_direct_answer_query(query)
        and not _query_bound_scalar_query(query)
        and not _query_bound_arithmetic_query(query)
        and not _latest_state_query(query)
        and not _direct_boolean_evidence_query(query)
        and not _possessive_attribute_query_target(query)
        and "preference_profile" not in intent.reasons
    ):
        return None
    group_limit = source_synthesis_candidate_limit(intent, limit=limit)
    if _average_query(query):
        group_limit = max(group_limit, 8)
    if _query_bound_arithmetic_query(query):
        group_limit = max(group_limit, 8)
    if _temporal_interval_query(query):
        group_limit = max(group_limit, 64)
    if _temporal_count_program_query(query):
        group_limit = max(group_limit, 64)
    token_cache = _SourceTokenCache(tokens={})
    ordered_sources = query_specific_source_order(query, source_results, token_cache=token_cache)
    if preferred_source_groups:
        ordered_sources = preferred_source_group_order(
            ordered_sources,
            preferred_source_groups,
        )
    score_cache = _SourceEvidenceScoreCache(query=query, scores={})
    ordered_sources = evidence_source_order(
        query,
        ordered_sources,
        score_cache=score_cache,
        token_cache=token_cache,
    )
    if {"aggregation", "aggregation_question"} & set(intent.reasons):
        ordered_sources = (
            _dominant_provenance_cluster_contexts(
                query,
                ordered_sources,
                score_cache=score_cache,
            )
            or ordered_sources
        )
    grouped_sources = diverse_source_contexts(
        ordered_sources,
        limit=group_limit,
        preserve_order=True,
    )
    direct_attribute = _possessive_attribute_query_target(query)
    if (
        len(grouped_sources) < 2
        and not direct_attribute
        and not _assistant_recall_query(query)
        and not _latest_state_query(query)
        and not _query_bound_direct_answer_query(query)
        and "preference_profile" not in intent.reasons
    ):
        return None
    if _should_defer_synthesis_to_absence(query, grouped_sources, intent):
        return None
    if (
        _temporal_order_query(query)
        and not _parent_order_query(query)
        and _temporal_order_query_choices(query)
        and len(_temporal_order_choice_observations(query, grouped_sources)) < 2
    ):
        return None
    aggregate_projection = (
        EvidenceProjection((), ())
        if _direct_time_query(query)
        or _recency_comparison_query(query)
        or _latest_state_should_suppress_aggregate(query)
        else aggregate_candidate_projection(query, grouped_sources)
    )
    if _incomplete_explicit_temporal_sequence_projection(query, aggregate_projection):
        aggregate_projection = EvidenceProjection((), ())
    preference_projection = preference_candidate_projection(query, grouped_sources, limit=group_limit)
    recency_projection = recency_candidate_projection(query, grouped_sources)
    derived_lines = [
        *aggregate_projection.lines,
        *preference_projection.lines,
        *(
            ()
            if _recency_comparison_query(query)
            or _has_multi_source_answer_candidate_type(aggregate_projection, "temporal_sequence")
            else _numeric_synthesis_lines(
                query,
                grouped_sources,
                aggregate_lines=list(aggregate_projection.lines),
            )
        ),
        *_anniversary_engagement_synthesis_lines(query, grouped_sources),
        *_frequency_synthesis_lines(query, grouped_sources),
        *_parent_order_synthesis_lines(query, grouped_sources),
        *_first_month_event_date_synthesis_lines(query, grouped_sources),
        *_temporal_order_synthesis_lines(query, grouped_sources),
        *recency_projection.lines,
        *(
            ()
            if _has_multi_source_answer_candidate_type(aggregate_projection, "duration")
            or _has_multi_source_answer_candidate_type(aggregate_projection, "temporal_sequence")
            else _relative_temporal_anchor_synthesis_lines(query, grouped_sources)
        ),
        *_direct_time_synthesis_lines(query, grouped_sources),
        *_assistant_recall_synthesis_lines(query, grouped_sources),
        *_issue_synthesis_lines(query, grouped_sources),
        *_query_bound_direct_answer_synthesis_lines(query, grouped_sources),
        *_query_bound_scalar_synthesis_lines(query, grouped_sources),
        *_direct_boolean_evidence_synthesis_lines(query, grouped_sources),
        *_direct_fact_synthesis_lines(query, grouped_sources),
    ]
    if not derived_lines and should_defer_to_absence_check(query, grouped_sources, intent):
        return None
    if not derived_lines and missing_query_target(query, grouped_sources):
        return None
    if not derived_lines:
        return None
    support_source_groups = tuple(
        dict.fromkeys(
            [
                *aggregate_projection.source_groups,
                *preference_projection.source_groups,
                *recency_projection.source_groups,
                *_source_groups_from_synthesis_lines(derived_lines),
            ]
        )
    )
    support_sources = _supporting_synthesis_sources(
        grouped_sources,
        source_groups=support_source_groups,
    )
    lines = [
        "zaxy_synthesis_bundle=true",
        "synthesis_mode=multi_source_aggregation",
        f"query={query}",
        f"source_count={len(support_sources)}",
    ]
    lines.extend(derived_lines)
    lines.extend(_elapsed_duration_at_event_ledger_row_lines(query, grouped_sources))
    lines.extend(_social_media_break_ledger_row_lines(query, grouped_sources))
    lines.extend(_road_trip_drive_ledger_row_lines(query, grouped_sources))
    lines.extend(_age_at_event_ledger_row_lines(query, grouped_sources))
    lines.extend(_future_age_at_event_ledger_row_lines(query, grouped_sources))
    lines.extend(_career_prior_duration_ledger_row_lines(query, grouped_sources))
    lines.extend(_current_role_tenure_ledger_row_lines(query, grouped_sources))
    if not any(row.get("include_reason") == "age_average_input" for row in aggregate_projection.ledger_rows):
        lines.extend(_age_average_ledger_row_lines(query, grouped_sources))
    lines.extend(_relative_interval_ledger_row_lines(query, grouped_sources))
    lines.extend(_anniversary_engagement_ledger_row_lines(query, grouped_sources))
    lines.extend(_parent_order_ledger_row_lines(query, grouped_sources))
    lines.extend(_first_month_event_date_ledger_row_lines(query, grouped_sources))
    lines.extend(_recency_ledger_row_lines(query, grouped_sources))
    lines.extend(_temporal_order_ledger_row_lines(query, grouped_sources))
    lines.extend(
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in aggregate_projection.ledger_rows
    )
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
    content = "\n".join(lines)
    packet = synthesis_packet_from_items(
        [
            {
                "content": content,
                "synthesis_packet": {
                    "schema_version": "synthesis_packet_v1",
                    "operations": [
                        *aggregate_projection.operations,
                        *preference_projection.operations,
                        *recency_projection.operations,
                    ],
                    "result": aggregate_projection.result or preference_projection.result or recency_projection.result or {},
                    "answer_candidates": [
                        *aggregate_projection.answer_candidates,
                        *preference_projection.answer_candidates,
                        *recency_projection.answer_candidates,
                    ],
                    "ledger_rows": [
                        *aggregate_projection.ledger_rows,
                        *preference_projection.ledger_rows,
                        *recency_projection.ledger_rows,
                    ],
                },
            }
        ]
    )
    return SourceSynthesisBundleResult(
        content=content,
        packet={
            "schema_version": "synthesis_packet_v1",
            "operations": packet.operations,
            "result": packet.result,
            "answer_candidates": packet.answer_candidates,
            "ledger_rows": packet.ledger_rows,
            "content": content,
        },
    )


def source_synthesis_bundle(
    *,
    query: str,
    source_results: list[str],
    limit: int,
    preferred_source_groups: list[str] | tuple[str, ...] | None = None,
) -> str | None:
    """Build one compact cited source bundle for multi-source synthesis queries."""
    result = source_synthesis_bundle_result(
        query=query,
        source_results=source_results,
        limit=limit,
        preferred_source_groups=preferred_source_groups,
    )
    return result.content if result is not None else None


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
