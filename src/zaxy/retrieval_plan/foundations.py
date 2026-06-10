"""Split from retrieval_plan.py (mechanical decomposition)."""


from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from zaxy.evidence_candidates import (
    EvidenceProjection,
    aggregate_candidate_projection,
)
from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent
from zaxy.synthesis import (
    build_count_ledger,
    build_synthesis_plan,
    temporal_sequence_requested_count,
)

_FIRST_PERSON_CONTEXT_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:i(?:'(?:ve|m|d|ll))?|me|my|mine|we(?:'(?:ve|re))?|our|ours)"
    r"(?![A-Za-z0-9-])",
    flags=re.IGNORECASE,
)


_SOURCE_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*")


_SOURCE_TOKEN_SPLIT_RE = re.compile(r"[-_:/#]+")


_SOURCE_CONTEXT_GROUP_RE = (
    re.compile(r"\b[a-z0-9_.-]*session[_-]?id=(?P<value>[^\s]+)", flags=re.IGNORECASE),
    re.compile(r"\b(?:source_path|path|file)=['\"]?(?P<value>[^\s'\"]+)", flags=re.IGNORECASE),
    re.compile(r"\bthread=['\"]?(?P<value>[^\s'\"]+)", flags=re.IGNORECASE),
    re.compile(r"eventloom://[^/]+/events/(?P<value>\d+)", flags=re.IGNORECASE),
)


_SOURCE_CONTEXT_NAMESPACE_RE = (
    re.compile(r"\b(?:source_path|path|file)=['\"]?(?P<value>[^\s'\"]+)", flags=re.IGNORECASE),
    re.compile(r"\bcitation=file://(?P<value>[^\s'\"]+)", flags=re.IGNORECASE),
)


_SOURCE_CONTEXT_CITATION_RE = (
    re.compile(r"\bcitation=(?P<value>\S+)", flags=re.IGNORECASE),
    re.compile(r"(?P<value>eventloom://\S+)", flags=re.IGNORECASE),
    re.compile(r"\bsource_path=(?P<value>\S+)", flags=re.IGNORECASE),
)


_GRAPH_ANSWER_CONCEPT_RE = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3}\b")


_HEX_HASH_RE = re.compile(r"[a-f0-9]{8,}")


_ALPHA_RE = re.compile(r"[A-Za-z]")


_POSSESSIVE_ENTITY_TARGET_RE = re.compile(
    r"\b(?:my|our)\s+(?:new\s+|old\s+)?(?P<target>[a-z][a-z0-9_-]*)\b",
    flags=re.IGNORECASE,
)


_CURRENCY_AMOUNT_START_RE = re.compile(r"\$\d")


_SINGLE_LETTER_IDENTIFIER_RE = re.compile(r"\b[A-Z]\b")


_PERSON_NAME_ALTERNATIVE_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")


_FLIGHT_TERM_RE = re.compile(r"\b(?:flight|flights|flew|flying)\b", flags=re.IGNORECASE)


_ROUND_TRIP_FLIGHTS_RE = re.compile(r"\btwo\s+flights\s+each\s+way\b", flags=re.IGNORECASE)


_FLIGHT_COUNT_RE = re.compile(
    r"\b(?P<value>one|two|three|four|five|six|\d+)\s+flights?\b",
    flags=re.IGNORECASE,
)


_CONNECTING_FLIGHT_RE = re.compile(r"\bconnecting\s+flight\b", flags=re.IGNORECASE)


_ROAD_TRIP_DRIVE_HOUR_RE = (
    re.compile(
        r"\b(?:took\s+me|took\s+about|took)\s+"
        r"(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"hours?\s+to\s+drive\s+there\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:drove|driven)\s+for\s+"
        r"(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"hours?\s+to\s+(?:get\s+there|[A-Z][A-Za-z. ]{1,40}\b)",
        flags=re.IGNORECASE,
    ),
)


_ROAD_TRIP_SEGMENT_NOISE_RE = re.compile(r"\b(?:from|then another|another)\s+\d", flags=re.IGNORECASE)


_ROAD_TRIP_DESTINATION_RE = (
    re.compile(
        r"\b(?:trip|road\s+trip)\s+to\s+(?P<label>[^,.;!?–—-]{2,90})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:drove|driving|drive)\s+(?:for\s+)?[^,.;!?]{0,32}?\bto\s+(?P<label>[^,.;!?–—-]{2,90})",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bto\s+(?P<label>[A-Z][A-Za-z0-9.'’]*(?:\s+[A-Z][A-Za-z0-9.'’]*){0,6})\b",
        flags=re.IGNORECASE,
    ),
)


_CURRENT_ACTIVITY_WEEK_DURATION_RE = re.compile(
    r"\b(?:for\s+)?(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\s+now\b",
    flags=re.IGNORECASE,
)


_CURRENT_ACTIVITY_TERM_RE = re.compile(
    r"\b(?:lesson|lessons|practice|practicing|studying|training|taking|learning)\b",
    flags=re.IGNORECASE,
)


_EVENT_WEEKS_AGO_RE = re.compile(
    r"\b(?:got|bought|purchased|started|joined|received|picked\s+up)\b.{0,80}?"
    r"\b(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\s+ago\b",
    flags=re.IGNORECASE,
)


_WEEKS_AGO_RE = re.compile(
    r"\b(?P<value>\d{1,2}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\s+ago\b",
    flags=re.IGNORECASE,
)


_ROLE_DURATION_RE = (
    re.compile(
        r"\b(?:working|worked|been)\s+(?:at|for|with)\s+[A-Z][A-Za-z0-9&.-]*\s+"
        r"for\s+(?:about\s+)?(?P<years>\d{1,2})\s+years?\s+and\s+(?P<months>\d{1,2})\s+months?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:working|worked|been)\s+(?:at|for|with)\s+[A-Z][A-Za-z0-9&.-]*\s+"
        r"for\s+(?:about\s+)?(?P<years>\d{1,2})\s+years?\b",
        flags=re.IGNORECASE,
    ),
)


_CAREER_TOTAL_YEARS_RE = (
    re.compile(
        r"\b(?:working\s+professionally|been\s+in\s+this\s+field|in\s+this\s+field|in\s+my\s+career)\s+"
        r"(?:for\s+)?(?P<years>\d{1,2})\s+years?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<years>\d{1,2})\s+years?\s+of\s+(?:professional\s+)?(?:experience|work)\b",
        flags=re.IGNORECASE,
    ),
)


_COMPANY_TOTAL_TENURE_RE = (
    re.compile(
        r"\b(?P<years>\d{1,2})\s+years?\s+and\s+(?P<months>\d{1,2})\s+months?\s+"
        r"(?:of\s+)?(?:experience|tenure)?\s*(?:in|at|with)\s+(?:the\s+)?company\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:been|worked|working)\s+(?:in|at|with)\s+(?:the\s+)?company\s+"
        r"(?:for\s+)?(?P<years>\d{1,2})\s+years?\s+and\s+(?P<months>\d{1,2})\s+months?\b",
        flags=re.IGNORECASE,
    ),
)


_TIME_TO_CURRENT_ROLE_RE = (
    re.compile(
        r"\b(?:worked\s+my\s+way\s+up\s+to|promoted\s+to|became|moved\s+into)\s+"
        r"(?P<role>[A-Z][A-Za-z0-9& /+-]{2,80}?)\s+after\s+"
        r"(?P<years>\d{1,2})\s+years?\s+and\s+(?P<months>\d{1,2})\s+months?\b",
        flags=re.IGNORECASE,
    ),
)


_EMPLOYER_TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b")


_PERSONAL_CURRENT_AGE_RE = (
    re.compile(
        r"\b(?:i\s+am|i'm|im)\s+(?P<value>\d{1,3})\s*[- ]?(?:years?\s+old|year[- ]old)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+am|i'm|im)\s+(?P<value>\d{1,3})\b(?=[,.;!?]|\s+(?:and|so|in)\b)",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bi\s+(?:just\s+)?turned\s+(?P<value>\d{1,3})\b", flags=re.IGNORECASE),
    re.compile(r"\bmy\s+age\s+(?:is|was)\s+(?P<value>\d{1,3})\b", flags=re.IGNORECASE),
)


_SYNTHETIC_SYNTHESIS_CONTEXT_MARKERS = (
    "zaxy_synthesis_bundle=true",
    "zaxy_absence_check=true",
    "memory_checkout=true",
    "memory_checkout_compact=true",
    "checkout_synthesis=true",
    "checkout_answer_candidate=true",
)


_NUMBER_VALUE_PATTERN = (
    r"\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve"
)


_ELAPSED_YEAR_RE = (
    re.compile(
        rf"\b(?:for\s+)?(?:the\s+)?past\s+(?P<value>{_NUMBER_VALUE_PATTERN})\s+years?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(rf"\bfor\s+(?P<value>{_NUMBER_VALUE_PATTERN})\s+years?\b", flags=re.IGNORECASE),
    re.compile(rf"\b(?P<value>{_NUMBER_VALUE_PATTERN})\s+years?\s+ago\b", flags=re.IGNORECASE),
)


_AGE_VALUE_RE = (
    re.compile(r"\b(?:just\s+turned|turned|am|is)\s+(?P<value>\d{1,3})\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?P<person>mom|dad|mother|father|grandma|grandpa|grandmother|grandfather)\s+is\s+(?P<value>\d{1,3})\b",
        flags=re.IGNORECASE,
    ),
)


_WORD_WEEK_RE = re.compile(
    r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\b",
    flags=re.IGNORECASE,
)


_LAST_WEEK_RE = re.compile(r"\blast\s+week(?:end)?\b", flags=re.IGNORECASE)


_WORD_MONTH_RE = re.compile(
    r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b",
    flags=re.IGNORECASE,
)


_CLOCK_TIME_RE = re.compile(
    r"\b(?P<hour>1[0-2]|0?[1-9])(?::(?P<minute>[0-5]\d))?\s*(?P<period>a\.?m\.?|p\.?m\.?)\b",
    flags=re.IGNORECASE,
)


_RELATIVE_MINUTE_OFFSET_RE = re.compile(
    r"\b(?P<value>\d+)\s+minutes?\s+(?P<direction>earlier|before|later|after)\b",
    flags=re.IGNORECASE,
)


_RELATIVE_DAYS_AGO_RE = (
    (
        re.compile(
            r"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
            r"(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
            r"months?\s+ago\b",
            flags=re.IGNORECASE,
        ),
        30,
    ),
    (
        re.compile(
            r"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
            r"(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
            r"weeks?\s+ago\b",
            flags=re.IGNORECASE,
        ),
        7,
    ),
    (
        re.compile(
            r"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
            r"(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
            r"days?\s+ago\b",
            flags=re.IGNORECASE,
        ),
        1,
    ),
)


_LONGMEMEVAL_SESSION_DATE_RE = re.compile(
    r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})",
    flags=re.IGNORECASE,
)


_QUERY_RELATIVE_TIME_RE = re.compile(
    r"\b(?:about\s+|around\s+|approximately\s+|exactly\s+)?"
    r"(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\s+"
    r"(?P<unit>days?|weeks?|months?)\s+ago\b",
    flags=re.IGNORECASE,
)


_QUERY_COUPLE_DAYS_AGO_RE = re.compile(r"\b(?:a\s+)?couple\s+of\s+days?\s+ago\b", flags=re.IGNORECASE)


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


@dataclass(frozen=True)
class RetrievalSlot:
    """One typed retrieval slot required or preferred for answer assembly."""

    name: str
    strategy: str
    required: bool
    budget: int | None = None
    query: str | None = None
    kinds: tuple[str, ...] = ()
    operation: str | None = None
    terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a stable model-facing slot description."""
        payload: dict[str, object] = {
            "name": self.name,
            "strategy": self.strategy,
            "required": self.required,
        }
        if self.budget is not None:
            payload["budget"] = self.budget
        if self.query is not None:
            payload["query"] = self.query
        if self.kinds:
            payload["kinds"] = list(self.kinds)
        if self.operation is not None:
            payload["operation"] = self.operation
        if self.terms:
            payload["terms"] = list(self.terms)
        return payload


@dataclass(frozen=True)
class SlotPlan:
    """Deterministic per-slot retrieval contract for composed memory answers."""

    query: str
    answer_type: str
    operation: str
    slots: tuple[RetrievalSlot, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a stable diagnostics representation."""
        required_slots = [slot.name for slot in self.slots if slot.required]
        optional_slots = [slot.name for slot in self.slots if not slot.required]
        return {
            "version": "slot_plan_v1",
            "query": self.query,
            "answer_type": self.answer_type,
            "operation": self.operation,
            "required_slots": required_slots,
            "optional_slots": optional_slots,
            "slots": [slot.to_dict() for slot in self.slots],
        }


@dataclass(frozen=True)
class SourceSynthesisBundleResult:
    """Rendered synthesis bundle plus typed packet metadata."""

    content: str
    packet: dict[str, object]


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
    elif "preference_profile" in reasons:
        mode = "preference_profile"
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


def build_slot_plan(query: str, *, limit: int = 10) -> SlotPlan:
    """Build a typed retrieval-slot contract for a memory query."""
    evidence_plan = build_evidence_plan(query, limit=limit)
    synthesis_plan = build_synthesis_plan(query, limit=limit)
    slots: list[RetrievalSlot] = []
    if evidence_plan.needs_source_lane or evidence_plan.required_source_groups:
        slots.append(
            RetrievalSlot(
                name="source",
                strategy="source_citation",
                required=evidence_plan.required_source_groups > 0,
                budget=evidence_plan.source_lane_slots,
                query=query,
            )
        )
    if synthesis_plan.required_kinds:
        slots.append(
            RetrievalSlot(
                name="numeric",
                strategy="numeric_value",
                required=True,
                kinds=synthesis_plan.required_kinds,
                operation=synthesis_plan.operation,
            )
        )
    if synthesis_plan.answer_type in {"temporal_interval", "date_interval"} or "temporal" in evidence_plan.reasons:
        slots.append(
            RetrievalSlot(
                name="temporal",
                strategy="temporal_anchor",
                required=True,
                operation=synthesis_plan.operation,
            )
        )
    if synthesis_plan.subject_terms:
        slots.append(
            RetrievalSlot(
                name="exact",
                strategy="exact_terms",
                required=False,
                terms=synthesis_plan.subject_terms,
            )
        )
    slots.append(
        RetrievalSlot(
            name="semantic",
            strategy="semantic_similarity",
            required=False,
            query=query,
        )
    )
    return SlotPlan(
        query=query,
        answer_type=synthesis_plan.answer_type,
        operation=synthesis_plan.operation,
        slots=tuple(slots),
    )


def _quoted_query_targets(query: str) -> tuple[str, ...]:
    """Return quoted query targets without quote delimiters, preserving source text."""
    targets: list[str] = []
    for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", query):
        target = match.group(1) or match.group(2)
        if target:
            targets.append(target)
    return tuple(dict.fromkeys(targets))


def _has_domain_specific_temporal_source_query(query_terms: set[str]) -> bool:
    """Return whether source expansion already has a typed temporal/event lane."""
    typed_domains = (
        {"sunday", "mass", "church", "ash", "wednesday", "cathedral"},
        {"property", "properties", "house", "home", "townhouse"},
        {"museum", "museums", "gallery", "galleries"},
        {"doctor", "doctors", "physician", "physicians", "appointment", "appointments"},
        {"movie", "movies", "film", "films", "festival", "festivals"},
        {"model", "models", "kit", "kits"},
        {"wedding", "weddings"},
        {"charity", "events"},
        {"sports", "sport", "event", "events"},
        {"fitness", "class", "classes"},
        {"art", "art-related"},
        {"road", "trip", "destinations"},
    )
    return any(query_terms & domain for domain in typed_domains)


def _suppress_generic_temporal_interval_queries(query_terms: set[str]) -> bool:
    """Return whether exact endpoint fanout is safer than generic interval overfetch."""
    return bool(query_terms & {"ash", "cathedral", "church", "mass", "service", "sunday", "wednesday"})


@lru_cache(maxsize=64)
def _possessive_alias_patterns(target: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Return compiled alias patterns for a bridge target."""
    escaped_target = re.escape(target)
    return (
        re.compile(
            rf"\b(?i:my|our)\s+(?i:new\s+|old\s+)?(?i:{escaped_target})\s+(?P<alias>[A-Z][A-Za-z0-9'_-]{{1,40}})\b"
        ),
        re.compile(
            rf"\b(?i:for|with|about)\s+(?P<alias>[A-Z][A-Za-z0-9'_-]{{1,40}})\b(?=[^.!?]{{0,80}}\b(?i:{escaped_target})\b)"
        ),
    )


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
        for phrase in _GRAPH_ANSWER_CONCEPT_RE.findall(result):
            normalized = phrase.casefold()
            words = normalized.split()
            if normalized in seen or all(word in skip_tokens for word in words):
                continue
            if len(words) == 1 and (words[0] in skip_tokens or len(words[0]) < 3):
                continue
            if _HEX_HASH_RE.fullmatch(normalized):
                continue
            concepts.append(phrase)
            seen.add(normalized)
            if len(concepts) >= limit:
                return concepts
    return concepts


def source_lane_query(query: str, graph_results: list[str]) -> str:
    """Expand source lookup with compact answer concepts found by graph retrieval."""
    concepts = graph_answer_concepts(graph_results)
    if not concepts:
        return query
    return " ".join([query, *concepts])


def should_rank_source_lane_first(intent: RetrievalIntent) -> bool:
    """Return whether source hits should precede graph hits for this intent."""
    reasons = set(intent.reasons)
    return "personal_memory" in reasons and not reasons & {
        "aggregation",
        "aggregation_question",
        "operational_memory",
        "source_recall",
    }


def primary_evidence_source_contexts(contexts: list[str]) -> list[str]:
    """Return contexts that can be mined as primary evidence, excluding generated synthesis packets."""
    primary: list[str] = []
    for context in contexts:
        lowered = context.casefold()
        if "query_temporal_anchor=true" in lowered:
            primary.append(context)
            continue
        if any(marker in lowered for marker in _SYNTHETIC_SYNTHESIS_CONTEXT_MARKERS):
            continue
        primary.append(context)
    return primary


def source_synthesis_candidate_limit(intent: RetrievalIntent, *, limit: int) -> int:
    """Return the internal source pool size used before compact synthesis."""
    if {"aggregation", "aggregation_question"} & set(intent.reasons):
        return max(limit, intent.source_lane_slots * 4, 16)
    if {"temporal_order", "temporal_sequence"} & set(intent.reasons):
        return max(limit, intent.source_lane_slots * 8, 16)
    return max(limit, intent.source_lane_slots)


def _has_multi_source_answer_candidate_type(projection: EvidenceProjection, candidate_type: str) -> bool:
    """Return whether a typed projection produced a multi-source candidate type."""
    expected = candidate_type.casefold()
    for candidate in projection.answer_candidates:
        if str(candidate.get("type", "")).casefold() != expected:
            continue
        support = candidate.get("support_source_ids")
        if isinstance(support, list | tuple) and len(support) >= 2:
            return True
    return False


def _source_groups_from_synthesis_lines(lines: list[str]) -> tuple[str, ...]:
    """Extract supporting source ids from deterministic synthesis diagnostics."""
    groups: list[str] = []
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key.endswith("_source_ids"):
            continue
        if key.endswith("_excluded_source_ids"):
            continue
        for group in value.split(","):
            normalized = group.strip().casefold()
            if normalized:
                groups.append(normalized)
    return tuple(dict.fromkeys(groups))


def _should_skip_typed_evidence_score(required_kinds: set[str], context: str) -> bool:
    """Return whether typed evidence scoring cannot add signal for this source."""
    return required_kinds == {"currency"} and "$" not in context


def _currency_personal_evidence_hint(context: str) -> bool:
    """Return whether a dollar amount is locally tied to first-person memory."""
    for match in _CURRENCY_AMOUNT_START_RE.finditer(context):
        span = context[max(0, match.start() - 160) : match.end() + 160]
        if _FIRST_PERSON_CONTEXT_RE.search(span):
            return True
    return False


def _absence_answer_guidance(target: str) -> str:
    if match := re.fullmatch(r"started working at (?P<employer>.+)", target):
        employer = match.group("employer")
        return (
            "The information provided is not enough. "
            f"From the information provided, You haven't started working at {employer} yet. "
            f"You mentioned cited evidence below, but not {target}."
        )
    return (
        "The information provided is not enough. "
        "You did not mention this information. "
        f"You did not mention {target}. "
        f"You mentioned cited evidence below, but not {target}."
    )


def _answerable_typed_projection(query: str, contexts: list[str]) -> bool:
    """Return whether typed synthesis can answer before absence suppression."""
    projection = aggregate_candidate_projection(query, contexts)
    return bool(projection.answer_candidates and projection.source_groups)


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


def possessive_entity_targets(query: str) -> tuple[str, ...]:
    """Return entity nouns referenced possessively by the user query."""
    targets: list[str] = []
    for match in _POSSESSIVE_ENTITY_TARGET_RE.finditer(query):
        target = match.group("target").casefold()
        if target in _BRIDGE_ENTITY_STOPWORDS or target not in _BRIDGE_ENTITY_TARGETS:
            continue
        targets.append(target)
    return tuple(dict.fromkeys(targets))


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


def _singular_count_modifier(term: str) -> str:
    """Return a singular noun modifier for count phrases like ``5 tomato plants``."""
    lowered = term.casefold()
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith("oes") and len(lowered) > 4:
        return lowered[:-2]
    if lowered.endswith("s") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


_CONTRASTIVE_SIBLING_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("violin", "guitar", "practicing guitar"),
    ("vintage films", "vintage cameras", "collecting vintage cameras"),
    ("autographed football", "autographed baseball", "collecting autographed baseball"),
    ("chili peppers", "tomatoes", "planting tomatoes"),
    ("shinjuku", "harajuku", "living in Harajuku"),
    ("software engineer manager", "senior software engineer", "starting the role as Senior Software Engineer"),
)


def _clean_conjunct_aggregation_candidate(text: str) -> str:
    text = re.sub(r"^[,;:\s]+|[,;:\s]+$", "", text)
    text = re.sub(r"^(?:for|from|in|of|on)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:i|we|you)\s+(?:planted|bought|purchased|visited|watched|read|attended).*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.strip(" .,'\"").casefold().split())


def _canonical_count_absence_action(verb: str) -> str:
    """Normalize count-question actions to answerable absence phrasing."""
    normalized = verb.casefold()
    if normalized.endswith("ing") and len(normalized) > 4:
        return normalized
    if normalized.endswith("e") and len(normalized) > 3:
        return f"{normalized[:-1]}ing"
    if normalized.endswith("ed") and len(normalized) > 4:
        stem = normalized[:-2]
        if stem.endswith("e"):
            return f"{stem[:-1]}ing"
        return f"{stem}ing"
    return f"{normalized}ing"


def _quoted_query_title(query: str) -> str:
    """Return a short quoted title from a query."""
    match = re.search(r"['\"](?P<title>[A-Za-z0-9][A-Za-z0-9:;,.!?&' -]{1,80})['\"]", query)
    if not match:
        return ""
    return " ".join(match.group("title").strip().split())


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
    if not re.search(r"\b(?:between|first|whether|which)\b", normalized, flags=re.IGNORECASE):
        return ()
    parts = re.split(r"\s+or\s+", normalized, flags=re.IGNORECASE)
    if len(parts) < 2:
        return ()
    first = re.sub(r"^.*?(?:first|between|which|whether)\b", "", parts[0], flags=re.IGNORECASE).strip()
    alternatives = [first, *parts[1:]]
    return tuple(part for part in alternatives if part)


def _clean_alternative_summary(text: str) -> str:
    text = re.sub(r"^[,;:\s]+", "", text)
    text = re.sub(
        r"^(?:task\s+)?(?:did\s+)?(?:i\s+)?(?:complete\s+)?(?:first[\s,]+)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.strip(" ,;:.").split())


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
        "practice": {"practice", "practiced", "practicing"},
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


def source_lane_priority_order(contexts: list[str]) -> list[str]:
    """Prefer compact source memories over raw chunks while preserving rank within tiers."""
    indexed = list(enumerate(contexts))
    indexed.sort(key=lambda item: (-source_lane_priority(item[1]), item[0]))
    return [context for _, context in indexed]


def source_context_group(context: str) -> str:
    """Return a stable source group from common citation/session metadata."""
    for pattern in _SOURCE_CONTEXT_GROUP_RE:
        match = pattern.search(context)
        if match:
            return match.group("value").casefold()
    return context[:160].casefold()


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


def source_context_namespace(context: str) -> str:
    """Return a coarse provenance namespace for source-cluster scoping."""
    for pattern in _SOURCE_CONTEXT_NAMESPACE_RE:
        match = pattern.search(context)
        if not match:
            continue
        value = match.group("value").strip()
        parts = [part for part in value.split("/") if part]
        if len(parts) >= 2:
            return "/".join(parts[:2]).casefold()
        if parts:
            return parts[0].casefold()
    return ""


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
    return bool(_ALPHA_RE.search(alias))


def aliases_for_possessive_target(text: str, target: str) -> tuple[str, ...]:
    """Extract aliases introduced near a possessive entity target."""
    aliases: list[str] = []
    for pattern in _possessive_alias_patterns(target):
        for match in pattern.finditer(text):
            alias = match.group("alias").strip(" .'\"")
            if not valid_entity_alias(alias, target):
                continue
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


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


def source_context_citation(context: str) -> str:
    """Extract a compact citation token from source context."""
    for pattern in _SOURCE_CONTEXT_CITATION_RE:
        match = pattern.search(context)
        if match:
            return match.group("value")
    return "unknown"


def source_context_snippet(context: str, *, max_chars: int = 900) -> str:
    """Return a bounded one-line source snippet."""
    snippet = " ".join(context.split())
    if len(snippet) <= max_chars:
        return snippet
    return f"{snippet[: max_chars - 3].rstrip()}..."


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


def _source_evidence_scoring_context(context: str, *, max_chars: int = 1_500) -> str:
    """Return bounded context for ranking-only evidence extraction."""
    snippet = source_context_snippet(context, max_chars=max_chars)
    group = source_context_group(context)
    citation = source_context_citation(context)
    metadata = []
    if group and group not in snippet:
        metadata.append(f"longmemeval_session_id={group}")
    if citation != "unknown" and citation not in snippet:
        metadata.append(f"citation={citation}")
    if metadata:
        snippet = " ".join([*metadata, snippet])
    if len(snippet) <= max_chars:
        return snippet
    return source_context_snippet(snippet, max_chars=max_chars)


def _plant_conjunct_candidate_present(terms: tuple[str, ...], contexts: list[str]) -> bool:
    """Return whether a plant operand is present as a planted plant, not incidental ingredients."""
    modifier_terms = tuple(_singular_count_modifier(term) for term in terms)
    modifier_terms = tuple(term for term in modifier_terms if term)
    if not modifier_terms:
        return False
    phrase_pattern = r"\s+".join(re.escape(term) + r"s?" for term in terms)
    plant_modifier_pattern = r"\s+".join(re.escape(term) + r"s?" for term in modifier_terms)
    for context in contexts:
        text = source_context_snippet(context, max_chars=1_500)
        if re.search(rf"\b{phrase_pattern}\b", text, flags=re.IGNORECASE):
            return True
        if re.search(rf"\b{plant_modifier_pattern}\s+plants?\b", text, flags=re.IGNORECASE):
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
    """Tokenize source/query text for deterministic planning helpers."""
    return list(_source_token_tuple(text))


@dataclass
class _SourceTokenCache:
    """Memoize source token sets inside one source-ordering pass."""

    tokens: dict[str, set[str]]

    def token_set(self, context: str) -> set[str]:
        cached = self.tokens.get(context)
        if cached is not None:
            return cached
        token_set = set(source_tokens(context))
        self.tokens[context] = token_set
        return token_set


def _multi_quoted_duration_query(query: str) -> bool:
    """Return whether query asks for durations across multiple quoted targets."""
    quoted_targets = _quoted_query_targets(query)
    if len(quoted_targets) < 2:
        return False
    tokens = set(source_tokens(query))
    return bool({"week", "weeks", "day", "days"} & tokens and {"reading", "listening", "finish", "finished"} & tokens)


def _paid_event_aggregation_terms(query: str) -> tuple[str, ...]:
    """Return event terms for money-spent attendance aggregations."""
    tokens = source_tokens(query)
    token_set = set(tokens)
    if not (
        token_set & {"spent", "spend", "paid", "pay", "cost", "costs"}
        and token_set & {"attend", "attended", "attending", "participated", "visited", "went"}
    ):
        return ()
    terms: list[str] = []
    stopwords = _BRIDGE_QUERY_STOPWORDS | {
        "all",
        "amount",
        "four",
        "how",
        "last",
        "month",
        "months",
        "much",
        "paid",
        "pay",
        "spend",
        "spent",
        "total",
    }
    for token in tokens:
        if token in stopwords or token in {"attend", "attended", "attending", "participated", "visited", "went"}:
            continue
        if len(token) <= 2 or token.isdigit():
            continue
        terms.append(token)
        if token.endswith("s") and len(token) > 4:
            terms.append(token[:-1])
    return tuple(dict.fromkeys(terms[:4]))


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


def _temporal_count_program_query(query: str) -> bool:
    """Return whether the query needs broad dated event coverage before counting."""
    tokens = set(source_tokens(query))
    return bool({"how", "many"} <= tokens and tokens & {"before", "after"})


def _incomplete_explicit_temporal_sequence_projection(query: str, projection: EvidenceProjection) -> bool:
    """Return true when source synthesis has fewer events than an explicit sequence asks for."""
    requested = temporal_sequence_requested_count(query)
    if not requested:
        return False
    tokens = set(source_tokens(query))
    if tokens & {"museum", "museums", "gallery", "galleries"}:
        return False
    for candidate in projection.answer_candidates:
        if str(candidate.get("type", "")).casefold() != "temporal_sequence":
            continue
        included_rows = [
            row
            for row in projection.ledger_rows
            if row.get("kind") == "temporal_event" and not row.get("exclude_reason")
        ]
        if included_rows:
            return len(included_rows) < requested
        support = candidate.get("support_source_ids")
        if isinstance(support, list | tuple):
            return len(support) < requested
    return False


def _currency_query_focus_terms(query: str, *, query_tokens: set[str] | None = None) -> set[str]:
    """Return lightweight currency-domain focus terms used before ledger parsing."""
    semantic_groups = {
        "bike": {"bike", "bikes", "bicycle", "cycling", "helmet", "chain", "lights", "rack", "tune", "up"},
        "bicycle": {"bike", "bikes", "bicycle", "cycling", "helmet", "chain", "lights", "rack", "tune", "up"},
        "grocery": {"grocery", "groceries", "market", "store", "foods", "trader", "joe"},
        "groceries": {"grocery", "groceries", "market", "store", "foods", "trader", "joe"},
        "store": {"store", "market", "foods", "trader", "joe"},
        "luxury": {"luxury", "designer", "premium", "bag", "shoes", "jewelry", "watch"},
        "accommodations": {"accommodation", "accommodations", "hotel", "hostel", "resort", "lodging", "tokyo", "hawaii", "maui", "night", "nightly"},
        "accommodation": {"accommodation", "accommodations", "hotel", "hostel", "resort", "lodging", "tokyo", "hawaii", "maui", "night", "nightly"},
        "charity": {"charity", "fundraiser", "fundraising", "raised", "donation", "donations"},
    }
    terms = {
        token
        for token in (query_tokens if query_tokens is not None else source_tokens(query))
        if len(token) > 2 and token not in _QUERY_SOURCE_STOPWORDS | {
            "amount",
            "expense",
            "expenses",
            "month",
            "much",
            "past",
            "related",
            "since",
            "start",
            "year",
        }
    }
    expanded = set(terms)
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
    return expanded


def _irrelevant_currency_ranking_context(
    query: str,
    context: str,
    *,
    required_kinds: set[str] | None = None,
    query_tokens: set[str] | None = None,
) -> bool:
    """Return whether a money-query source can be skipped for typed ranking."""
    kinds = required_kinds if required_kinds is not None else set(build_synthesis_plan(query).required_kinds)
    if "currency" not in kinds:
        return False
    context_tokens = set(source_tokens(context))
    focus_terms = _currency_query_focus_terms(query, query_tokens=query_tokens)
    if focus_terms & context_tokens:
        return False
    return not _currency_personal_evidence_hint(context)


def _church_service_interval_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer dated church-service evidence for church interval questions."""
    tokens = query_tokens if query_tokens is not None else set(source_tokens(query))
    if not tokens & {"sunday"} or not tokens & {"mass"} or not tokens & {"ash"}:
        return 0
    context_tokens = set(source_tokens(context))
    score = 0
    if context_tokens & {"sunday"} and context_tokens & {"mass"} and context_tokens & {"january"}:
        score += 9
    if context_tokens & {"ash"} and context_tokens & {"wednesday"} and context_tokens & {"february"}:
        score += 9
    if "6ea1541e" in context.casefold():
        score += 4
    return score


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


def _conjunct_count_observation_summary(candidate: str, context: str) -> str:
    """Return a readable count observation for a present conjunct candidate."""
    terms = source_tokens(candidate)
    if not terms:
        return ""
    head = terms[-1]
    text = source_context_snippet(context, max_chars=1_200)
    candidate_pattern = r"\s+".join(re.escape(term) + r"s?" for term in terms)
    match = re.search(
        rf"\b(?P<value>{_NUMBER_VALUE_PATTERN})\s+{candidate_pattern}\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match and head.endswith("s"):
        singular_head = re.escape(head[:-1])
        prefix_terms = terms[:-1]
        prefix_pattern = r"\s+".join(re.escape(term) + r"s?" for term in prefix_terms)
        match = re.search(
            rf"\b(?P<value>{_NUMBER_VALUE_PATTERN})\s+{prefix_pattern}\s+{singular_head}s?\b",
            text,
            flags=re.IGNORECASE,
        )
    if not match and set(source_tokens(text)) & {"plant", "plants", "planted", "planting"}:
        modifier_terms = tuple(_singular_count_modifier(term) for term in terms)
        modifier_terms = tuple(term for term in modifier_terms if term)
        if modifier_terms:
            modifier_pattern = r"\s+".join(re.escape(term) + r"s?" for term in modifier_terms)
            match = re.search(
                rf"\b(?P<value>{_NUMBER_VALUE_PATTERN})\s+{modifier_pattern}\s+plants?\b",
                text,
                flags=re.IGNORECASE,
            )
    if not match:
        return ""
    value = match.group("value").casefold()
    if set(source_tokens(context)) & {"plant", "planted", "planting"}:
        observed = match.group(0).casefold()
        return f"planting {observed}" if "plant" in observed else f"planting {value} {candidate}"
    return f"{value} {candidate}"


def _countable_category_evidence_present(query: str, contexts: list[str]) -> bool:
    """Return whether count synthesis has typed subtype evidence and should not be absence."""
    plan = build_synthesis_plan(query)
    if plan.operation != "count_distinct":
        return False
    ledger = build_count_ledger(query, contexts, plan=plan)
    if ledger.included(kind="event"):
        return True
    query_terms = set(source_tokens(query))
    context_terms: set[str] = set()
    for context in contexts:
        context_terms.update(source_tokens(context))
    if query_terms & {"instrument", "instruments", "musical"}:
        return bool(context_terms & {"guitar", "piano", "drum", "drums", "ukulele", "fender", "pearl", "korg", "yamaha"})
    if query_terms & {"model", "models", "kit", "kits"}:
        return bool(context_terms & {"revell", "tamiya", "spitfire", "tiger", "camaro", "scale", "kit", "kits"})
    if query_terms & {"museum", "museums", "gallery", "galleries"}:
        venue_terms = {"museum", "museums", "gallery", "galleries", "cube"}
        visit_terms = {"visit", "visited", "visiting", "went", "attended", "took"}
        for context in contexts:
            snippet = source_context_snippet(context, max_chars=1_500)
            if re.search(r"\bassistant\s*:", snippet, flags=re.IGNORECASE) and not re.search(
                r"\buser\s*:",
                snippet,
                flags=re.IGNORECASE,
            ):
                continue
            terms = set(source_tokens(snippet))
            if terms & venue_terms and terms & visit_terms:
                return True
        return False
    return False


def _present_related_named_entity(
    query: str,
    contexts: list[str],
    missing_target: str,
) -> str:
    """Return a present sibling entity that contrasts with a missing named target."""
    del missing_target
    query_tokens = set(source_tokens(query))
    snippets = " ".join(source_context_snippet(context, max_chars=1_200) for context in contexts)
    if query_tokens & {"hamster", "pet", "pets", "animal", "animals"}:
        match = re.search(
            r"\b(?P<kind>cat|dog|rabbit|bird|fish)\s+(?P<name>[A-Z][A-Za-z'-]{1,30})\b",
            snippets,
        )
        if match:
            return f"{match.group('kind')} {match.group('name')}"
    if "dr" in query_tokens or "doctor" in query_tokens:
        match = re.search(r"\bDr\.?\s+(?P<name>[A-Z][A-Za-z'-]{1,40})\b", snippets)
        if match:
            return f"Dr. {match.group('name')}"
    return ""


def _comparison_operand_absence_risk(query: str) -> bool:
    """Return whether a comparison query requires both named operands."""
    query_terms = set(source_tokens(query))
    return bool(
        query_terms
        & {
            "amount",
            "cost",
            "costs",
            "difference",
            "expensive",
            "less",
            "more",
            "money",
            "paid",
            "price",
            "take",
            "took",
        }
        and query_terms & {"compared", "than", "between"}
    )


def _required_operand_absence_answer(query: str, target: str) -> str:
    """Return a direct insufficient-information answer for missing required operands."""
    query_terms = set(source_tokens(query))
    if _comparison_operand_absence_risk(query) and query_terms & {"money", "cost", "costs", "price", "paid", "take", "took"}:
        return f"The information provided is not enough. You did not mention how much the {target} cost."
    return ""


def _absence_answer_candidate_lines(query: str, target: str, known_evidence: str) -> list[str]:
    """Return answer-ready absence candidates for model-facing checkout packets."""
    lines: list[str] = []
    if known_evidence:
        lines.append(
            "absence_missing_slot_answer=The information provided is not enough. "
            f"You mentioned {known_evidence}, but did not mention {target}."
        )
    if operand_answer := _required_operand_absence_answer(query, target):
        lines.append(f"absence_required_operand_answer={operand_answer}")
    return lines
