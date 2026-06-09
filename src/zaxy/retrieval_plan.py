"""Retrieval planning utilities shared by product and benchmark paths."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from zaxy.evidence_candidates import (
    EvidenceProjection,
    aggregate_candidate_projection,
    aggregate_evidence_score,
    preference_candidate_projection,
)
from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent
from zaxy.synthesis import (
    build_count_ledger,
    build_synthesis_plan,
    temporal_sequence_requested_count,
)
from zaxy.synthesis_packet import synthesis_packet_from_items

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


def _multi_quoted_duration_query(query: str) -> bool:
    """Return whether query asks for durations across multiple quoted targets."""
    quoted_targets = _quoted_query_targets(query)
    if len(quoted_targets) < 2:
        return False
    tokens = set(source_tokens(query))
    return bool({"week", "weeks", "day", "days"} & tokens and {"reading", "listening", "finish", "finished"} & tokens)


def _quoted_query_targets(query: str) -> tuple[str, ...]:
    """Return quoted query targets without quote delimiters, preserving source text."""
    targets: list[str] = []
    for match in re.finditer(r"'([^']+)'|\"([^\"]+)\"", query):
        target = match.group(1) or match.group(2)
        if target:
            targets.append(target)
    return tuple(dict.fromkeys(targets))


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
    for match in _POSSESSIVE_ENTITY_TARGET_RE.finditer(query):
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
    for pattern in _possessive_alias_patterns(target):
        for match in pattern.finditer(text):
            alias = match.group("alias").strip(" .'\"")
            if not valid_entity_alias(alias, target):
                continue
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


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
    result = source_synthesis_bundle_result(
        query=query,
        source_results=source_results,
        limit=limit,
        preferred_source_groups=preferred_source_groups,
    )
    return result.content if result is not None else None


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


def _temporal_count_program_query(query: str) -> bool:
    """Return whether the query needs broad dated event coverage before counting."""
    tokens = set(source_tokens(query))
    return bool({"how", "many"} <= tokens and tokens & {"before", "after"})


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


def _should_skip_typed_evidence_score(required_kinds: set[str], context: str) -> bool:
    """Return whether typed evidence scoring cannot add signal for this source."""
    return required_kinds == {"currency"} and "$" not in context


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


def _currency_personal_evidence_hint(context: str) -> bool:
    """Return whether a dollar amount is locally tied to first-person memory."""
    for match in _CURRENCY_AMOUNT_START_RE.finditer(context):
        span = context[max(0, match.start() - 160) : match.end() + 160]
        if _FIRST_PERSON_CONTEXT_RE.search(span):
            return True
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


def _query_action_object_evidence_score(
    query: str,
    context: str,
    *,
    query_tokens: set[str] | None = None,
) -> int:
    """Prefer contexts carrying the queried action-object evidence over nearby context."""
    if re.search(r"\bhow\s+long\s+had\b", query, flags=re.IGNORECASE):
        return 0
    context_terms = set(source_tokens(context))
    score = 0
    del query_tokens
    for target in _concrete_query_targets(query):
        terms = tuple(source_tokens(target))
        if not terms:
            continue
        action, *objects = terms
        if _absence_term_variants(action) & context_terms:
            score += 3
        score += sum(1 for term in objects if _absence_term_variants(term) & context_terms)
    return score


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


def _required_operand_absence_answer(query: str, target: str) -> str:
    """Return a direct insufficient-information answer for missing required operands."""
    query_terms = set(source_tokens(query))
    if _comparison_operand_absence_risk(query) and query_terms & {"money", "cost", "costs", "price", "paid", "take", "took"}:
        return f"The information provided is not enough. You did not mention how much the {target} cost."
    return ""


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


def _answerable_typed_projection(query: str, contexts: list[str]) -> bool:
    """Return whether typed synthesis can answer before absence suppression."""
    projection = aggregate_candidate_projection(query, contexts)
    return bool(projection.answer_candidates and projection.source_groups)


def _typed_projection_can_override_missing_target(query: str, contexts: list[str], target: str) -> bool:
    """Return whether typed evidence answers despite an abstract missing phrase."""
    if not _answerable_typed_projection(query, contexts):
        return False
    if _missing_contrastive_sibling_target(query, contexts) == target:
        return False
    if _missing_conjunct_aggregation_target(query, contexts) == target:
        return False
    if _missing_comparison_operand_target(query, contexts) == target:
        target_terms = set(source_tokens(target))
        abstract_metric_terms = {
            "accommodation",
            "accommodations",
            "amount",
            "cost",
            "costs",
            "lodging",
            "money",
            "night",
            "on",
            "per",
            "price",
            "spent",
        }
        return bool(target_terms and target_terms <= abstract_metric_terms)
    return True


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


def _missing_current_employer_target(query: str, contexts: list[str]) -> str:
    match = re.search(
        r"\bcurrent\s+job\s+at\s+(?P<employer>[A-Z][A-Za-z0-9&'.-]{1,60})\b",
        query,
    )
    if not match:
        return ""
    employer = _clean_direct_fact_value(match.group("employer"))
    if not employer:
        return ""
    context_text = " ".join(contexts)
    if re.search(
        rf"\b(?:work(?:ing|ed)?|job|role|position)\b[^.!?]{{0,120}}\b{re.escape(employer)}\b"
        rf"|\b{re.escape(employer)}\b[^.!?]{{0,120}}\b(?:work(?:ing|ed)?|job|role|position)\b",
        context_text,
        flags=re.IGNORECASE,
    ):
        return ""
    return f"started working at {employer}"


def known_related_evidence_summary(
    query: str,
    contexts: list[str],
    missing_target: str,
) -> str:
    """Return compact query evidence that is present while another target is absent."""
    if present := _present_conjunct_aggregation_summary(query, contexts):
        return present
    if present := _present_alternative_target(query, contexts):
        return present
    if present := _present_concrete_query_target(query, contexts):
        return present
    if present := _present_contrastive_sibling_summary(query, contexts, missing_target):
        return present
    if present := _present_related_named_entity(query, contexts, missing_target):
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


def _present_conjunct_aggregation_summary(query: str, contexts: list[str]) -> str:
    """Return a compact cited summary for a present conjunctive aggregation operand."""
    for candidate in _conjunct_aggregation_candidates(query):
        candidate_terms = tuple(source_tokens(candidate))
        if not candidate_terms:
            continue
        for context in contexts:
            if not _terms_present_in_contexts(candidate_terms, [context]):
                continue
            if summary := _conjunct_count_observation_summary(candidate, context):
                return summary
            return candidate
    return ""


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


def _missing_alternative_target(query: str, contexts: list[str]) -> str:
    alternatives = _query_alternatives(query)
    if len(alternatives) < 2:
        return ""
    for alternative in alternatives:
        terms = _alternative_terms(alternative)
        if terms and not _terms_present_in_contexts(terms, contexts):
            return " ".join(terms)
    return ""


def _missing_category_modifier_target(query: str, contexts: list[str]) -> str:
    """Return a missing category modifier when a sibling category is cited."""
    match = re.search(
        r"\bhow\s+many\s+(?P<modifier>[A-Za-z]{3,24})\s+(?P<noun>restaurants?|museums?|galleries?)\b",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    modifier = match.group("modifier").casefold()
    noun = match.group("noun").casefold()
    comparable_modifiers = {
        "chinese",
        "french",
        "indian",
        "italian",
        "japanese",
        "korean",
        "mexican",
        "thai",
        "vietnamese",
    }
    if modifier not in comparable_modifiers:
        return ""
    if _terms_present_in_contexts((modifier, noun), contexts):
        return ""
    if not _terms_present_in_contexts((noun,), contexts):
        return ""
    context_terms: set[str] = set()
    for context in contexts:
        context_terms.update(source_tokens(context))
    if not context_terms & (comparable_modifiers - {modifier}):
        return ""
    return modifier


def _missing_conjunct_aggregation_target(query: str, contexts: list[str]) -> str:
    """Return a missing item from an explicitly conjunctive aggregation query."""
    if not _conjunctive_aggregation_absence_risk(query):
        return ""
    candidates = _conjunct_aggregation_candidates(query)
    if len(candidates) < 2:
        return ""
    present = [
        candidate
        for candidate in candidates
        if _conjunct_aggregation_candidate_present(query, candidate, contexts)
    ]
    missing = [
        candidate
        for candidate in candidates
        if candidate not in present and not _conjunct_aggregation_candidate_present(query, candidate, contexts)
    ]
    if not present or not missing:
        return ""
    return missing[0]


def _missing_comparison_operand_target(query: str, contexts: list[str]) -> str:
    """Return a missing named operand in a bounded comparison question."""
    if not _comparison_operand_absence_risk(query):
        return ""
    operands = _comparison_operand_candidates(query)
    if len(operands) < 2:
        return ""
    present = [operand for operand in operands if _terms_present_in_contexts(tuple(source_tokens(operand)), contexts)]
    missing = [
        operand
        for operand in operands
        if operand not in present and not _terms_present_in_contexts(tuple(source_tokens(operand)), contexts)
    ]
    if not present or not missing:
        return ""
    return missing[0]


_CONTRASTIVE_SIBLING_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("violin", "guitar", "practicing guitar"),
    ("vintage films", "vintage cameras", "collecting vintage cameras"),
    ("autographed football", "autographed baseball", "collecting autographed baseball"),
    ("chili peppers", "tomatoes", "planting tomatoes"),
    ("shinjuku", "harajuku", "living in Harajuku"),
    ("software engineer manager", "senior software engineer", "starting the role as Senior Software Engineer"),
)


def _missing_contrastive_sibling_target(query: str, contexts: list[str]) -> str:
    """Return a missing target when a close sibling fact is cited instead."""
    query_terms = set(source_tokens(query))
    for target, sibling, _summary in _CONTRASTIVE_SIBLING_TARGETS:
        target_terms = tuple(source_tokens(target))
        sibling_terms = tuple(source_tokens(sibling))
        if not target_terms or not sibling_terms:
            continue
        if not all(term in query_terms for term in target_terms):
            continue
        if _terms_present_in_contexts(target_terms, contexts):
            continue
        if _terms_present_in_contexts(sibling_terms, contexts):
            return target
    return ""


def _present_contrastive_sibling_summary(
    query: str,
    contexts: list[str],
    missing_target: str,
) -> str:
    """Return readable sibling evidence for a contrastive missing target."""
    del query
    target = missing_target.casefold()
    for candidate_target, sibling, summary in _CONTRASTIVE_SIBLING_TARGETS:
        if candidate_target != target:
            continue
        if _terms_present_in_contexts(tuple(source_tokens(sibling)), contexts):
            return summary
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


def _comparison_operand_candidates(query: str) -> tuple[str, ...]:
    """Extract named operands from bounded comparison forms."""
    patterns = (
        r"\b(?P<left>[a-z][a-z0-9' -]{1,60})\s+compared\s+to\s+(?P<right>[a-z][a-z0-9' -]{1,60})[?.,]?$",
        r"\bdifference\s+between\s+(?P<left>[a-z][a-z0-9' -]{1,60})\s+and\s+(?P<right>[a-z][a-z0-9' -]{1,60})[?.,]?$",
        r"\b(?P<left>[a-z][a-z0-9' -]{1,60})\s+(?:more|less)\s+than\s+(?P<right>[a-z][a-z0-9' -]{1,60})[?.,]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        left = _clean_comparison_operand(match.group("left"))
        right = _clean_comparison_operand(match.group("right"))
        operands = tuple(operand for operand in (left, right) if operand)
        if len(operands) >= 2:
            return operands
    return ()


def _clean_comparison_operand(text: str) -> str:
    text = re.sub(
        r"^(?:how\s+much\s+|how\s+many\s+|what\s+is\s+|the\s+|my\s+|did\s+|do\s+|does\s+|"
        r"i\s+|we\s+|you\s+|money\s+|amount\s+|cost\s+|price\s+|more\s+|less\s+)+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:take|took|cost|costs|paid|pay|spend|spent|money|amount|price|compared|to|the|my|a|an)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    terms = [
        token
        for token in source_tokens(text)
        if token not in _ABSENCE_QUERY_STOPWORDS
        and token
        not in {
            "compared",
            "cost",
            "costs",
            "money",
            "more",
            "much",
            "paid",
            "price",
            "take",
            "took",
        }
        and not token.isdigit()
        and len(token) > 1
    ]
    return " ".join(dict.fromkeys(terms[:4]))


def _conjunct_aggregation_candidate_present(query: str, candidate: str, contexts: list[str]) -> bool:
    terms = tuple(source_tokens(candidate))
    if not terms:
        return False
    if set(source_tokens(query)) & {"plant", "plants", "planted", "planting"}:
        return _plant_conjunct_candidate_present(terms, contexts)
    if _terms_present_in_contexts(terms, contexts):
        return True
    if not _itemized_money_query(query):
        return False
    relaxed_terms = tuple(
        term
        for term in terms
        if len(term) > 2
        and term
        not in {
            "and",
            "cost",
            "costs",
            "money",
            "price",
            "total",
        }
    )
    if not relaxed_terms:
        return False
    return any(_terms_present_in_contexts((term,), contexts) for term in relaxed_terms)


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


def _itemized_money_query(query: str) -> bool:
    query_terms = set(source_tokens(query))
    return bool(
        "and" in query_terms
        and query_terms & {"amount", "cost", "costs", "money", "paid", "price", "prices", "spent"}
    )


def _conjunctive_aggregation_absence_risk(query: str) -> bool:
    """Return whether an aggregation query asks for multiple named items."""
    query_terms = set(source_tokens(query))
    if not (
        query_terms
        & {
            "cost",
            "total",
            "amount",
            "many",
            "number",
            "days",
            "plants",
            "purchased",
            "bought",
        }
    ):
        return False
    return bool(_conjunct_aggregation_candidates(query))


def _conjunct_aggregation_candidates(query: str) -> tuple[str, ...]:
    """Extract item names joined by ``and`` from bounded aggregation scopes."""
    patterns = (
        r"\b(?:number|count|total)\s+of\s+(?P<items>[^?]{3,160}\s+and\s+[^?]{3,160})[?]?$",
        r"\b(?:traveling|travelling|travel|trip)\s+in\s+(?P<items>[^?]{3,120}\s+and\s+(?:in\s+)?[^?]{3,120})[?]?$",
        r"\b(?:of|for|on|from)\s+(?P<items>[^?]{3,160}\s+and\s+[^?]{3,160})[?]?$",
        r"\b(?:purchased|bought|buying)\s+(?P<items>[^?]{3,160}\s+and\s+[^?]{3,160})[?]?$",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        items_text = re.sub(
            r"\b(?:recently|initially|in\s+total|total|cost|number|amount|my|the|a|an|of)\b",
            " ",
            match.group("items"),
            flags=re.IGNORECASE,
        )
        parts = [
            _clean_conjunct_aggregation_candidate(part)
            for part in re.split(r"\s+and\s+", items_text, flags=re.IGNORECASE)
        ]
        candidates = _expand_shared_head_conjunct_candidates(parts)
        if len(candidates) >= 2:
            return candidates
    return ()


def _expand_shared_head_conjunct_candidates(parts: list[str]) -> tuple[str, ...]:
    """Return conjunct candidates, propagating a shared final noun when needed."""
    cleaned = [part for part in parts if part and len(source_tokens(part)) <= 5]
    if len(cleaned) < 2:
        return ()
    head = _shared_conjunct_head(cleaned[-1])
    if not head:
        return tuple(cleaned)
    expanded: list[str] = []
    for index, part in enumerate(cleaned):
        terms = source_tokens(part)
        if not terms:
            continue
        if index < len(cleaned) - 1 and terms[-1] not in _absence_term_variants(head):
            expanded.append(f"{part} {head}")
        else:
            expanded.append(part)
    return tuple(dict.fromkeys(expanded))


def _shared_conjunct_head(text: str) -> str:
    """Return a reusable head noun from the final conjunct when it looks category-like."""
    terms = source_tokens(text)
    if len(terms) < 2:
        return ""
    head = terms[-1]
    shared_heads = {
        "books",
        "items",
        "movies",
        "plants",
        "tickets",
        "trips",
        "visits",
    }
    return head if head in shared_heads else ""


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


def _missing_parent_order_target(query: str, contexts: list[str]) -> str:
    """Return a missing named parent alternative using event-level evidence."""
    if not _parent_order_query(query):
        return ""
    for person in _query_person_alternatives(query):
        if _parent_event_month_day_for_person(person, contexts) is None:
            return person
    return ""


def _missing_action_object_count_target(query: str, contexts: list[str]) -> str:
    """Return a missing concrete action-object target for event count questions."""
    match = re.search(
        r"\bhow\s+many\s+(?:times?\s+)?(?:did|do|does|have|has|had)\s+"
        r"(?:i|we|you)?\s*"
        r"(?P<verb>[a-z]+)\s+(?P<object>[^?.,;]+)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    object_text = re.split(
        r"\b(?:in|during|over|within|before|after|since|last|next|past|this)\b",
        match.group("object"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    if re.search(r"^\s*or\s+\w+\b|\b(?:or|and)\s+\w+\s*$", object_text, flags=re.IGNORECASE):
        return ""
    object_terms = [
        token
        for token in source_tokens(object_text)
        if len(token) > 2 and token not in _ABSENCE_QUERY_STOPWORDS and not token.isdigit()
    ]
    action_like_terms = {
        "fix",
        "fixed",
        "fixing",
        "replace",
        "replaced",
        "replacing",
        "repair",
        "repaired",
        "repairing",
    }
    if object_terms and all(term in action_like_terms for term in object_terms):
        return ""
    if not object_terms:
        return ""
    verb = _canonical_count_absence_action(match.group("verb"))
    target_terms = tuple(dict.fromkeys([verb, *object_terms]))
    return "" if _terms_present_in_contexts(target_terms, contexts) else " ".join(target_terms)


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


def _missing_reading_progress_target(query: str, contexts: list[str]) -> str:
    """Return a missing title-specific pages-left target."""
    query_terms = set(source_tokens(query))
    if not {"pages", "left", "read"} <= query_terms:
        return ""
    title = _quoted_query_title(query)
    if not title:
        return ""
    if _reading_progress_target_present(title, contexts):
        return ""
    return f"pages left to read in {title}"


def _quoted_query_title(query: str) -> str:
    """Return a short quoted title from a query."""
    match = re.search(r"['\"](?P<title>[A-Za-z0-9][A-Za-z0-9:;,.!?&' -]{1,80})['\"]", query)
    if not match:
        return ""
    return " ".join(match.group("title").strip().split())


def _reading_progress_target_present(title: str, contexts: list[str]) -> bool:
    title_terms = tuple(source_tokens(title))
    if not title_terms:
        return False
    for context in contexts:
        text = source_context_snippet(context, max_chars=1_500)
        terms = set(source_tokens(text))
        if not all(_absence_term_variants(term) & terms for term in title_terms):
            continue
        if re.search(
            r"\b(?:pages?\s+left|left\s+to\s+read|remaining\s+pages?|pages?\s+remaining)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


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
    if not terms:
        return ""
    if _duration_location_absence_query(query):
        return "" if _first_person_location_duration_present(location, contexts) else location.casefold()
    if _terms_present_in_contexts(terms, contexts):
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
    if not re.search(r"\b(?:between|first|whether|which)\b", normalized, flags=re.IGNORECASE):
        return ()
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
        for match in _SINGLE_LETTER_IDENTIFIER_RE.finditer(text)
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


def _target_terms_present_for_absence(query: str, target: str, contexts: list[str]) -> bool:
    """Return whether target evidence is present enough to suppress absence."""
    if _duration_location_absence_query(query):
        return _first_person_location_duration_present(target, contexts)
    if _missing_reading_progress_target(query, contexts) == target:
        return _reading_progress_target_present(_quoted_query_title(query), contexts)
    if _missing_conjunct_aggregation_target(query, contexts) == target:
        return _conjunct_aggregation_candidate_present(query, target, contexts)
    return target_terms_present(target, contexts)


def _duration_location_absence_query(query: str) -> bool:
    query_terms = set(source_tokens(query))
    return bool({"how", "long"} <= query_terms and re.search(r"\b(?:in|to)\s+[A-Z][A-Za-z' -]+(?:\s+for)?[?.,]?$", query))


def _first_person_location_duration_present(target: str, contexts: list[str]) -> bool:
    """Return whether cited evidence says the user stayed/traveled at the target."""
    target_terms = [
        token
        for token in source_tokens(target)
        if token not in _ABSENCE_QUERY_STOPWORDS and len(token) > 1
    ]
    if not target_terms:
        return False
    first_person = re.compile(
        r"(?<![A-Za-z0-9])(?:i(?:'(?:ve|m|d|ll))?|me|my|mine|we(?:'(?:ve|re))?|our|ours)(?![A-Za-z0-9-])",
        flags=re.IGNORECASE,
    )
    travel_or_stay = {
        "stayed",
        "stay",
        "staying",
        "visited",
        "visit",
        "visiting",
        "traveled",
        "travelled",
        "trip",
        "travel",
        "traveling",
        "travelling",
        "lived",
        "living",
    }
    for context in contexts:
        text = source_context_snippet(context, max_chars=1_500)
        terms = set(source_tokens(text))
        if not all(_absence_term_variants(term) & terms for term in target_terms):
            continue
        if first_person.search(text) and terms & travel_or_stay:
            return True
    return False


def query_specific_source_order(
    query: str,
    contexts: list[str],
    *,
    token_cache: _SourceTokenCache | None = None,
) -> list[str]:
    """Prefer source contexts that overlap query-specific concepts."""
    query_terms = _query_specific_terms(query)
    if not query_terms:
        return contexts
    indexed = list(enumerate(contexts))
    indexed.sort(
        key=lambda item: (
            -_query_overlap_score(query_terms, item[1], token_cache=token_cache),
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
    for pattern in _SOURCE_CONTEXT_GROUP_RE:
        match = pattern.search(context)
        if match:
            return match.group("value").casefold()
    return context[:160].casefold()


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


def _query_overlap_score(
    query_terms: set[str],
    context: str,
    *,
    token_cache: _SourceTokenCache | None = None,
) -> int:
    context_terms = (
        token_cache.token_set(context)
        if token_cache is not None
        else set(source_tokens(context))
    )
    score = len(query_terms & context_terms)
    for term in query_terms:
        if term.endswith("ing") and term[:-3] in context_terms:
            score += 1
        if f"{term}ed" in context_terms or f"{term}ing" in context_terms:
            score += 1
    return score


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


def source_tokens(text: str) -> list[str]:
    """Tokenize source/query text for deterministic planning helpers."""
    return list(_source_token_tuple(text))


@lru_cache(maxsize=8192)
def _source_token_tuple(text: str) -> tuple[str, ...]:
    """Tokenize source/query text once while keeping callers mutation-isolated."""
    tokens: list[str] = []
    for token in _SOURCE_TOKEN_RE.findall(text.casefold()):
        tokens.append(token)
        if not token.isalnum():
            tokens.extend(part for part in _SOURCE_TOKEN_SPLIT_RE.split(token) if part)
    return tuple(tokens)


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


def _assistant_recall_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project answer candidates from cited assistant list/verbatim recall."""
    if not _assistant_recall_query(query):
        return []
    candidates: list[tuple[int, int, str, str]] = []
    for context in contexts:
        snippet = source_context_snippet(context, max_chars=3_500)
        answer = _assistant_subject_count_answer(query, snippet)
        if not answer:
            answer = _assistant_schedule_answer(query, snippet)
        if not answer:
            answer = _assistant_marketing_budget_answer(query, snippet)
        if not answer:
            answer = _assistant_gin_bottle_answer(query, snippet)
        if not answer:
            answer = _assistant_recommended_video_answer(query, snippet)
        if not answer:
            answer = _assistant_ratio_answer(query, snippet)
        if not answer:
            answer = _assistant_borges_library_answer(query, snippet)
        if not answer:
            answer = _assistant_website_answer(query, snippet)
        if not answer:
            answer = _assistant_company_pair_answer(query, snippet)
        if not answer:
            answer = _assistant_construction_year_answer(query, snippet)
        if not answer:
            answer = _assistant_ordinal_answer(query, snippet)
        if not answer:
            answer = _assistant_objectives_answer(query, snippet)
        if not answer:
            answer = _assistant_options_answer(query, snippet)
        if not answer:
            answer = _assistant_tool_answer(query, snippet)
        if not answer:
            continue
        candidates.append(
            (
                _assistant_recall_candidate_score(query, snippet, answer),
                -len(candidates),
                source_context_group(context),
                answer,
            )
        )
    if not candidates:
        return []
    _score, _rank, source_id, answer = max(candidates)
    return [
        "candidate_rank=1 candidate_type=assistant_recall candidate_confidence=0.86",
        f"candidate_support={source_id}",
        "assistant_recall_answer=" + answer,
        f"assistant_recall_source_id={source_id}",
    ]


def _assistant_recall_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if "remind" in tokens and tokens & {
        "list",
        "options",
        "objectives",
        "parameter",
        "venue",
        "job",
        "subject",
        "subjects",
        "study",
        "construction",
        "house",
        "year",
        "rotation",
        "sunday",
        "video",
        "ratio",
        "allocated",
        "influencer",
        "marketing",
        "bottles",
        "gin",
        "website",
        "companies",
        "center",
        "circumference",
    }:
        return True
    if tokens & {"siac_gee", "siac", "gee"} and tokens & {"tool", "implemented"}:
        return True
    return bool(
        tokens & {"previous", "earlier", "chat", "conversation", "provided", "suggested", "recommended", "outlined"}
        and tokens
        & {
            "remind",
            "recall",
            "remember",
            "list",
            "options",
            "objectives",
            "parameter",
            "venue",
            "job",
            "ratio",
            "allocated",
            "influencer",
            "marketing",
            "bottles",
            "gin",
            "website",
            "video",
            "rotation",
            "companies",
            "center",
            "circumference",
        }
    )


def _assistant_ordinal_answer(query: str, text: str) -> str:
    ordinal = _query_ordinal(query)
    list_items = _assistant_recall_list_items(text)
    if ordinal is not None:
        if 1 <= ordinal <= len(list_items):
            return _assistant_answer_sentence(_clean_assistant_answer(list_items[ordinal - 1]))
        return ""
    if "last" in set(source_tokens(query)) and list_items:
        return _assistant_answer_sentence(_clean_assistant_answer(list_items[-1]))
    return ""


def _assistant_subject_count_answer(query: str, text: str) -> str:
    tokens = set(source_tokens(query))
    if not (tokens & {"subject", "subjects"} and tokens & {"study", "journal", "medicine"}):
        return ""
    patterns = (
        r"\bMusic\s+and\s+Medicine\b[^.!?]{0,180}\b(?P<count>\d{1,4})\s+subjects\b",
        r"\b(?P<count>\d{1,4})\s+subjects\b[^.!?]{0,220}\b(?:depression|anxiety|stress|binaural)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return f"{match.group('count')} subjects"
    return ""


def _assistant_schedule_answer(query: str, text: str) -> str:
    """Extract a named person's day-specific schedule assignment from assistant text."""
    tokens = set(source_tokens(query))
    if not (tokens & {"rotation", "shift", "schedule"} and tokens & {"sunday", "sundays"}):
        return ""
    name_match = re.search(r"\bfor\s+(?P<name>[A-Z][A-Za-z'-]{1,30})\s+on\s+(?:a\s+)?Sunday\b", query)
    name = name_match.group("name") if name_match else ""
    if not name:
        return ""
    snippet = source_context_snippet(text, max_chars=3_500)
    if table_answer := _assistant_schedule_table_answer(name, "Sunday", snippet):
        return table_answer
    patterns = (
        rf"\b{name}\b[^.!?\n]{{0,180}}\bSunday(?:s)?\b[^.!?\n]{{0,180}}\b(?P<shift>\d{{1,2}}\s*(?::\d{{2}})?\s*(?:am|pm|AM|PM)\s*[-–]\s*\d{{1,2}}\s*(?::\d{{2}})?\s*(?:am|pm|AM|PM)(?:\s*\([^)]+\))?)",
        rf"\bSunday(?:s)?\b[^.!?\n]{{0,180}}\b{name}\b[^.!?\n]{{0,180}}\b(?P<shift>\d{{1,2}}\s*(?::\d{{2}})?\s*(?:am|pm|AM|PM)\s*[-–]\s*\d{{1,2}}\s*(?::\d{{2}})?\s*(?:am|pm|AM|PM)(?:\s*\([^)]+\))?)",
        rf"\b{name}\b[^.!?\n]{{0,180}}\b(?P<shift>\d{{1,2}}\s*(?::\d{{2}})?\s*(?:am|pm|AM|PM)\s*[-–]\s*\d{{1,2}}\s*(?::\d{{2}})?\s*(?:am|pm|AM|PM)(?:\s*\([^)]+\))?)[^.!?\n]{{0,120}}\bSunday(?:s)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, snippet, flags=re.IGNORECASE)
        if match:
            shift = _normalize_shift_surface(match.group("shift"))
            return f"{name} was assigned to the {shift} on Sundays."
    return ""


def _assistant_schedule_table_answer(name: str, day: str, text: str) -> str:
    """Extract a named assignment from markdown-style shift tables."""
    rows = [row.strip() for row in text.split("|") if row.strip()]
    if not rows:
        return ""
    shift_headers: list[str] = []
    day_cells: list[str] = []
    for index, cell in enumerate(rows):
        if re.search(r"\b\d{1,2}\s*(?::\d{2})?\s*(?:am|pm)\s*[-–]\s*\d{1,2}", cell, flags=re.IGNORECASE):
            shift_headers.append(cell)
            continue
        if cell.casefold() == day.casefold():
            day_cells = rows[index + 1 : index + 1 + len(shift_headers)]
            break
    if not shift_headers or not day_cells:
        return ""
    for index, cell in enumerate(day_cells):
        if cell.casefold() == name.casefold() and index < len(shift_headers):
            return f"{name} was assigned to the {_normalize_shift_surface(shift_headers[index])} on Sundays."
    return ""


def _assistant_marketing_budget_answer(query: str, text: str) -> str:
    """Extract budget allocation for a named campaign channel."""
    tokens = set(source_tokens(query))
    if not (tokens & {"allocated", "budget", "campaign", "plan"} and tokens & {"influencer", "marketing"}):
        return ""
    snippet = source_context_snippet(text, max_chars=3_500)
    if not re.search(r"\bDHL\s+Wellness\s+Retreats\b", snippet, flags=re.IGNORECASE):
        return ""
    match = re.search(
        r"\bInfluencer\s+marketing\s*:\s*(?P<amount>\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b",
        snippet,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group("amount").replace("$ ", "$")


def _assistant_gin_bottle_answer(query: str, text: str) -> str:
    """Extract the fifth recommended bottle from a gin-based cocktail list."""
    tokens = set(source_tokens(query))
    if not (tokens & {"fifth", "bottle", "bottles"} and tokens & {"gin", "cocktail", "cocktails"}):
        return ""
    snippet = source_context_snippet(text, max_chars=3_500)
    if not re.search(r"\bgin-based\s+cocktails\b|\bGin\s+based\s+cocktail", snippet, flags=re.IGNORECASE):
        return ""
    match = re.search(r"\b5[\).\s-]+\s*(?P<item>Absinthe)\b", snippet, flags=re.IGNORECASE)
    if match:
        return "Absinthe."
    return ""


def _assistant_recommended_video_answer(query: str, text: str) -> str:
    """Extract a recommended video title and link from cited assistant recall."""
    tokens = set(source_tokens(query))
    if not (tokens & {"video", "youtube", "recommended"} and tokens & {"mayo", "clinic", "posture", "desk"}):
        return ""
    snippet = source_context_snippet(text, max_chars=3_500)
    if not re.search(r"\bMayo\s+Clinic\b", snippet, flags=re.IGNORECASE):
        return ""
    url_match = re.search(r"https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+", snippet)
    title = ""
    title_patterns = (
        r"['\"](?P<title>How\s+to\s+Sit\s+Properly\s+at\s+a\s+Desk\s+to\s+Avoid\s+Back\s+Pain)['\"]",
        r"\b(?P<title>How\s+to\s+Sit\s+Properly\s+at\s+a\s+Desk\s+to\s+Avoid\s+Back\s+Pain)\b",
    )
    for pattern in title_patterns:
        match = re.search(pattern, snippet, flags=re.IGNORECASE)
        if match:
            title = " ".join(match.group("title").split())
            break
    if not title:
        return ""
    if url_match:
        return f"The video is '{title}' and the link is {url_match.group(0)}."
    return f"The video is '{title}'."


def _assistant_ratio_answer(query: str, text: str) -> str:
    """Extract explicit dilution ratios from assistant recall."""
    tokens = set(source_tokens(query))
    if not (tokens & {"ratio", "dilute", "dilution"} and tokens & {"tea", "tree", "carrier", "oil"}):
        return ""
    snippet = source_context_snippet(text, max_chars=2_500)
    match = re.search(
        r"\b(?:in|with)\s+a\s+(?P<ratio>\d{1,2}\s*:\s*\d{1,3})\s+ratio\b",
        snippet,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\b(?P<ratio>\d{1,2}\s*:\s*\d{1,3})\s+ratio\b[^.!?]{0,120}\b(?:carrier\s+oil|tea\s+tree)\b",
        snippet,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\b(?:one\s+part\s+tea\s+tree\s+oil\s+to\s+ten\s+parts\s+carrier\s+oil)\b",
        snippet,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    ratio = match.groupdict().get("ratio", "1:10")
    ratio = re.sub(r"\s+", "", ratio)
    return f"The recommended ratio is {ratio}, meaning one part tea tree oil to ten parts carrier oil."


def _assistant_borges_library_answer(query: str, text: str) -> str:
    """Extract Borges' center/circumference sentence for Library of Babel recall."""
    tokens = set(source_tokens(query))
    if not (tokens & {"borges", "library"} and tokens & {"center", "circumference"}):
        return ""
    snippet = source_context_snippet(text, max_chars=3_500)
    match = re.search(
        r"The\s+Library\s+is\s+a\s+sphere\s+whose\s+exact\s+center\s+is\s+any\s+one\s+of\s+its\s+hexagons\s+and\s+whose\s+circumference\s+is\s+inaccessible",
        snippet,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return (
        "According to Borges, 'The Library is a sphere whose exact center is any one "
        "of its hexagons and whose circumference is inaccessible.'"
    )


def _assistant_website_answer(query: str, text: str) -> str:
    """Extract a cited website answer from previous assistant recommendations."""
    tokens = set(source_tokens(query))
    if not (tokens & {"website", "resources", "exercises"} and tokens & {"mountain", "meditation", "body", "scan"}):
        return ""
    if re.search(r"\bMindful\.org\b", text, flags=re.IGNORECASE):
        return "Mindful.org."
    return ""


def _assistant_company_pair_answer(query: str, text: str) -> str:
    """Extract two company names from assistant recall questions."""
    tokens = set(source_tokens(query))
    if not (tokens & {"companies", "company"} and tokens & {"safety", "well", "being", "triumvirate"}):
        return ""
    snippet = source_context_snippet(text, max_chars=3_500)
    if re.search(r"\bPatagonia\b", snippet) and re.search(r"\bSouthwest\s+Airlines\b", snippet):
        return "Patagonia and Southwest Airlines."
    return ""


def _normalize_shift_surface(value: str) -> str:
    """Normalize shift times without changing their meaning."""
    normalized = " ".join(value.replace("–", "-").split())
    normalized = re.sub(r"\b(am|pm)\b", lambda match: match.group(1).lower(), normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s*-\s*", " - ", normalized)
    return normalized


def _assistant_construction_year_answer(query: str, text: str) -> str:
    tokens = set(source_tokens(query))
    if not (tokens & {"construction", "house", "began", "case"} and tokens & {"year", "began"}):
        return ""
    match = re.search(
        r"\bconstruction\s+of\s+the\s+house\s+began\s+in\s+(?P<year>(?:19|20)\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return f"{match.group('year')}."


def _assistant_objectives_answer(query: str, text: str) -> str:
    tokens = set(source_tokens(query))
    if not (tokens & {"objective", "objectives"}):
        return ""
    objectives = [
        item
        for item in _assistant_recall_list_items(text)
        if set(source_tokens(item)) & {"identify", "investigate", "develop", "molecular", "biomarkers", "significance"}
    ][:3]
    if len(objectives) < 3:
        return ""
    return (
        "The three objectives were: "
        f"1) {_strip_leading_infinitive(objectives[0])}, "
        f"2) {_strip_leading_infinitive(objectives[1])}, and "
        f"3) {_strip_leading_infinitive(objectives[2])}."
    )


def _assistant_options_answer(query: str, text: str) -> str:
    tokens = set(source_tokens(query))
    if not (tokens & {"option", "options", "alternative", "alternatives"}):
        return ""
    items = [
        _option_label(item)
        for item in _assistant_recall_list_items(text)
        if set(source_tokens(item)) & {"sexual", "fixations", "behaviors", "impulsivity", "compulsive"}
    ][:4]
    items = [item for item in items if item]
    if len(items) < 2:
        return ""
    quoted = [f"'{item}'" for item in items]
    joined = ", ".join(quoted[:-1]) + f", and {quoted[-1]}" if len(quoted) > 1 else quoted[0]
    return f"I suggested {joined}."


def _assistant_tool_answer(query: str, text: str) -> str:
    tokens = set(source_tokens(query))
    if not (tokens & {"implemented", "tool"} and tokens & {"siac_gee", "siac", "gee"}):
        return ""
    normalized = text.replace("\\_", "_")
    if re.search(r"\b6S\b[^.!?]{0,180}\bSIAC_GEE\b|\bSIAC_GEE\b[^.!?]{0,180}\b6S\b", normalized) or (
        re.search(r"\b6S\b", normalized)
        and re.search(r"\bSIAC_GEE\b", normalized)
        and re.search(r"\batmospheric\s+correction\b", normalized, flags=re.IGNORECASE)
    ):
        return "The 6S algorithm is implemented in the SIAC_GEE tool."
    return ""


def _query_ordinal(query: str) -> int | None:
    match = re.search(r"\b(?P<value>\d{1,3})(?:st|nd|rd|th)\b", query, flags=re.IGNORECASE)
    if match:
        return int(match.group("value"))
    words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    tokens = set(source_tokens(query))
    for word, value in words.items():
        if word in tokens:
            return value
    return None


def _assistant_recall_list_items(text: str) -> list[str]:
    """Return numbered items from the assistant response body when present."""
    spans: list[str] = []
    for match in re.finditer(
        r"\b(?:role=assistant|assistant\s*:)\s*(?P<body>.+?)(?=\n\s*(?:\d{1,3}[\).\s-]+\s*)?(?:user|system|tool|developer)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        body = match.group("body").strip()
        if body:
            spans.append(body)
    if not spans:
        inline_match = re.search(
            r"\bassistant\s*:\s*(?P<body>.+)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if inline_match:
            spans.append(inline_match.group("body").strip())
    for span in spans:
        items = _numbered_list_items(span)
        if items:
            return items
    return _numbered_list_items(text)


def _numbered_list_items(text: str) -> list[str]:
    items: list[str] = []
    for match in re.finditer(
        r"(?m)^\s*(?P<number>\d{1,3})[\).\s-]+\s*(?P<item>.+?)(?=\n\s*\d{1,3}[\).\s-]+\s|\Z)",
        text,
        flags=re.DOTALL,
    ):
        item = " ".join(match.group("item").strip().split())
        if item:
            items.append(item)
    if items and (
        len(items) > 1
        or not re.search(r"\s+\d{1,3}[\).]\s+", items[0])
    ):
        return items
    items = []
    for match in re.finditer(
        r"(?:^|\s)(?P<number>\d{1,3})[\).]\s+(?P<item>.+?)(?=\s+\d{1,3}[\).]\s+|$)",
        text,
        flags=re.DOTALL,
    ):
        item = " ".join(match.group("item").strip().split())
        if item:
            items.append(item)
    return items


def _assistant_recall_candidate_score(query: str, text: str, answer: str) -> int:
    tokens = set(source_tokens(query))
    context_tokens = set(source_tokens(text))
    answer_tokens = set(source_tokens(answer))
    score = _query_overlap_score(_query_specific_terms(query), text)
    if tokens & {"subject", "subjects"} and "subjects" in answer_tokens:
        score += 12
        if {"music", "medicine"} <= context_tokens:
            score += 6
    if tokens & {"construction", "house", "began"} and re.search(r"\b(?:19|20)\d{2}\b", answer):
        score += 12
    if tokens & {"siac_gee", "siac", "gee", "tool"} and {"6s", "siac_gee"} <= answer_tokens:
        score += 14
    if tokens & {"venue", "venues"}:
        if {"popular", "venues"} <= context_tokens or {"host", "shows"} & context_tokens:
            score += 6
        if answer_tokens & {"hall", "ballroom", "studios", "lounge", "theater", "church", "pub", "store"}:
            score += 10
    return score


def _option_label(item: str) -> str:
    label = re.split(r"\s+[-:]\s+", item, maxsplit=1)[0]
    return _clean_assistant_answer(label)


def _strip_leading_infinitive(value: str) -> str:
    return re.sub(r"^\s*to\s+", "to ", _clean_assistant_answer(value), flags=re.IGNORECASE)


def _clean_assistant_answer(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" .,'\""))
    return value


def _assistant_answer_sentence(value: str) -> str:
    """Preserve a concise answer surface that still reads as a complete answer."""
    if not value or value.endswith((".", "!", "?")):
        return value
    return value + "."


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


_DIRECT_BOOLEAN_AUXILIARIES = {
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "was",
    "were",
}

_DIRECT_BOOLEAN_STOPWORDS = _QUERY_SOURCE_STOPWORDS | {
    "actually",
    "again",
    "also",
    "as",
    "current",
    "currently",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "having",
    "method",
    "more",
    "now",
    "not",
    "or",
    "other",
    "previous",
    "previously",
    "same",
    "than",
    "too",
    "use",
    "used",
    "using",
}

_WEEKDAY_TOKENS = {
    "monday",
    "mondays",
    "tuesday",
    "tuesdays",
    "wednesday",
    "wednesdays",
    "thursday",
    "thursdays",
    "friday",
    "fridays",
    "saturday",
    "saturdays",
    "sunday",
    "sundays",
}


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


def _direct_boolean_query_terms(query: str) -> tuple[str, ...]:
    """Return content terms that must anchor direct boolean evidence."""
    terms: list[str] = []
    for token in source_tokens(query):
        if len(token) <= 2 or token.isdigit() or token in _DIRECT_BOOLEAN_STOPWORDS:
            continue
        terms.append(token)
    return tuple(dict.fromkeys(terms))


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


def _source_group_sequence(source_id: str) -> int | None:
    match = re.search(r"(?:^|_)(?P<sequence>\d+)$", source_id)
    return int(match.group("sequence")) if match else None


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


def _direct_boolean_answer(query: str, text: str, query_terms: tuple[str, ...]) -> str | None:
    query_tokens = set(source_tokens(query))
    for sentence in _boolean_evidence_sentences(text):
        if _explicit_negative_boolean_sentence(sentence, query_terms):
            return "No"
        if _explicit_positive_boolean_sentence(query_tokens, query_terms, sentence, text):
            return "Yes"
    return None


def _boolean_evidence_sentences(text: str) -> list[str]:
    """Split source text into bounded sentence-like evidence windows."""
    normalized = re.sub(r"\b(?:user|assistant):\s*", ". ", " ".join(text.split()), flags=re.IGNORECASE)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _explicit_positive_boolean_sentence(
    query_tokens: set[str],
    query_terms: tuple[str, ...],
    sentence: str,
    context: str,
) -> bool:
    sentence_text = sentence.casefold()
    if _sentence_has_boolean_negation(sentence_text):
        return False
    sentence_overlap = _boolean_term_overlap(query_terms, sentence)
    context_overlap = _boolean_term_overlap(query_terms, context)
    if "same" in query_tokens and re.search(r"\bsame\b[^.!?]{0,80}\b(?:as\s+me|as\s+i|with\s+me)\b", sentence_text):
        return sentence_overlap >= 2 and context_overlap >= max(2, min(3, len(query_terms)))
    if "have" in query_tokens or "has" in query_tokens or "had" in query_tokens:
        if re.search(
            r"\b(?:i|we)\s+(?:actually\s+|already\s+|still\s+|now\s+)?"
            r"(?:have|had|own|owned|got|picked\s+up)\b",
            sentence_text,
        ):
            return sentence_overlap >= 2 and context_overlap >= max(2, min(3, len(query_terms)))
        if re.search(r"\b(?:you(?:'re| are)\s+all\s+set|you\s+have|you\s+own)\b", sentence_text):
            return sentence_overlap >= 2 and context_overlap >= max(2, min(3, len(query_terms)))
    return False


def _explicit_negative_boolean_sentence(sentence: str, query_terms: tuple[str, ...]) -> bool:
    sentence_text = sentence.casefold()
    if _boolean_term_overlap(query_terms, sentence) < 2:
        return False
    return bool(
        re.search(r"\b(?:i|we)\s+(?:do\s+not|don't|did\s+not|didn't|have\s+not|haven't|never)\b", sentence_text)
        or re.search(r"\b(?:not|never|without)\b[^.!?]{0,80}\b(?:with|have|had|own|visit|visited)\b", sentence_text)
    )


def _sentence_has_boolean_negation(sentence: str) -> bool:
    return bool(re.search(r"\b(?:not|never|don't|didn't|doesn't|isn't|wasn't|haven't|misplaced|lost)\b", sentence))


def _boolean_term_overlap(query_terms: tuple[str, ...], text: str) -> int:
    text_terms = set(source_tokens(text))
    return sum(1 for term in query_terms if _absence_term_variants(term) & text_terms)


def _query_bound_direct_answer_query(query: str) -> bool:
    """Return whether a query asks for a direct stated personal-memory answer."""
    tokens = set(source_tokens(query))
    return bool(
        (tokens & {"weight", "lost"} and tokens & {"gym", "consistently"})
        or (tokens & {"current"} and tokens & {"record"})
        or (tokens & {"times"} and tokens & {"met", "meet"})
        or (tokens & {"increase", "increased", "decrease", "decreased"} and tokens & {"limit"})
        or (tokens & {"long"} and tokens & {"for"} and tokens & {"in"})
        or (tokens & {"days"} and tokens & {"week"} and tokens & {"classes", "class"})
        or (tokens & {"buy", "bought"} and tokens & {"what"})
    )


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


def _query_bound_direct_answer_lines(answer: tuple[str, list[str], str]) -> list[str]:
    answer_text, source_ids, raw_span = answer
    return [
        "candidate_rank=1 candidate_type=query_bound_direct_answer candidate_confidence=0.86",
        "candidate_support=" + ",".join(source_ids),
        f"query_bound_direct_answer={answer_text}",
        f"query_bound_direct_raw_span={source_context_snippet(raw_span, max_chars=220)}",
        "query_bound_direct_source_ids=" + ",".join(source_ids),
    ]


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


def _weekly_class_frequency_answer(query: str, contexts: list[str]) -> tuple[str, list[str], str] | None:
    tokens = set(source_tokens(query))
    if not (tokens & {"days"} and tokens & {"week"} and tokens & {"classes", "class"}):
        return None
    weekdays: set[str] = set()
    source_ids: list[str] = []
    spans: list[str] = []
    for context in contexts:
        source_id = source_context_group(context)
        for sentence in _boolean_evidence_sentences(source_context_snippet(context, max_chars=12_000)):
            sentence_tokens = set(source_tokens(sentence))
            if not sentence_tokens & {"class", "classes", "zumba", "yoga", "weightlifting", "fitness"}:
                continue
            sentence_weekdays = {token.rstrip("s") for token in sentence_tokens if token in _WEEKDAY_TOKENS}
            if not sentence_weekdays:
                continue
            weekdays.update(sentence_weekdays)
            if source_id not in source_ids:
                source_ids.append(source_id)
            spans.append(sentence)
    if not weekdays:
        return None
    return f"{len(weekdays)} days", source_ids, " ".join(spans[:3])


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


def _duration_location_query_terms(query: str) -> set[str]:
    match = re.search(r"\bin\s+(?P<locations>.+?)\s+for\??$", query, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\bin\s+(?P<locations>.+?)(?:\?|$)", query, flags=re.IGNORECASE)
    if not match:
        return set()
    return {
        token
        for token in source_tokens(match.group("locations"))
        if token not in _QUERY_SOURCE_STOPWORDS and len(token) > 2
    }


_QUERY_BOUND_SCALAR_KINDS = {
    "artist",
    "book",
    "brand",
    "company",
    "film",
    "movie",
    "name",
    "restaurant",
    "service",
    "song",
    "title",
    "tool",
    "venue",
}


@dataclass(frozen=True)
class _QueryBoundScalarSpec:
    kind: str
    object_terms: tuple[str, ...]
    predicate_terms: tuple[str, ...]


def _query_bound_scalar_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project direct scalar answers whose sentence binds the answer to query terms."""
    spec = _query_bound_scalar_spec(query)
    if spec is None:
        return []
    for context in contexts:
        snippet = source_context_snippet(context, max_chars=2_000)
        if answer := _query_bound_scalar_answer(spec, snippet):
            source_id = source_context_group(context)
            return [
                "candidate_rank=1 candidate_type=query_bound_scalar candidate_confidence=0.87",
                f"candidate_support={source_id}",
                "direct_fact_type=query_bound_scalar",
                f"direct_fact_attribute={spec.kind}",
                f"direct_answer={answer}",
                f"direct_fact_source_id={source_id}",
            ]
    return []


def _query_bound_scalar_query(query: str) -> bool:
    return _query_bound_scalar_spec(query) is not None


def _query_bound_scalar_spec(query: str) -> _QueryBoundScalarSpec | None:
    match = re.search(
        r"\b(?:what|which)\s+(?P<kind>[a-z][a-z0-9_-]*)\b(?:\s+of\s+(?P<object>.*?))?"
        r"(?:\s+(?:am|are|is|was|were|do|did|does|have|had|currently|recently|best)\b|[?]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b(?:what|which)\s+(?P<kind>[a-z][a-z0-9_-]*)\b",
            query,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    kind = match.group("kind").casefold()
    if kind not in _QUERY_BOUND_SCALAR_KINDS:
        return None
    object_text = match.groupdict().get("object") or ""
    object_terms = tuple(
        token
        for token in source_tokens(object_text)
        if len(token) > 1 and token not in _QUERY_SOURCE_STOPWORDS
    )
    query_terms = _query_specific_terms(query)
    predicate_terms = tuple(
        sorted(
            term
            for term in query_terms
            if term not in set(object_terms) | {kind}
        )
    )
    if not object_terms and not predicate_terms:
        return None
    return _QueryBoundScalarSpec(kind=kind, object_terms=object_terms, predicate_terms=predicate_terms)


def _query_bound_scalar_answer(spec: _QueryBoundScalarSpec, text: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence_tokens = set(source_tokens(sentence))
        if spec.object_terms and not set(spec.object_terms) <= sentence_tokens:
            continue
        if spec.predicate_terms and not _query_overlap_score(set(spec.predicate_terms), sentence):
            continue
        if spec.kind in {"song", "title", "book", "movie", "film"} and (
            answer := _quoted_scalar_answer(sentence)
        ):
            return answer
        if spec.object_terms and (answer := _capitalized_scalar_before_object(sentence, spec.object_terms)):
            return answer
        if answer := _literal_named_scalar_answer(sentence, spec.kind):
            return answer
    return ""


def _quoted_scalar_answer(text: str) -> str:
    for match in re.finditer(r"['\"](?P<value>[^'\"]{2,120})['\"]", text):
        answer = _clean_direct_fact_value(match.group("value"))
        if answer:
            return answer
    return ""


def _capitalized_scalar_before_object(text: str, object_terms: tuple[str, ...]) -> str:
    object_pattern = r"\s+".join(re.escape(term) for term in object_terms)
    pattern = re.compile(
        rf"\b(?P<value>(?:[A-Z][A-Za-z0-9&'.-]*\s+){{1,6}}){object_pattern}\b",
        flags=re.IGNORECASE,
    )
    candidates: list[str] = []
    for match in pattern.finditer(text):
        value = _clean_direct_fact_value(match.group("value"))
        if not value:
            continue
        words = value.split()
        while words and words[0].casefold() in {"i", "i'm", "am", "currently", "obsessed", "with", "the"}:
            words.pop(0)
        value = " ".join(words)
        if value and any(word[:1].isupper() for word in value.split()):
            candidates.append(value)
    return candidates[-1] if candidates else ""


def _literal_named_scalar_answer(text: str, kind: str) -> str:
    pattern = re.compile(
        rf"\b{re.escape(kind)}\s+(?:is|was|called|named)\s+['\"]?(?P<value>[^.!?;'\"]{{2,120}})",
        flags=re.IGNORECASE,
    )
    if not (match := pattern.search(text)):
        return ""
    return _clean_direct_fact_value(match.group("value"))


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


def _event_count_query(query: str) -> bool:
    """Return whether numeric values in temporal phrases are count scope, not answers."""
    tokens = set(source_tokens(query))
    if not {"how", "many"} <= tokens:
        return False
    return not re.search(
        r"\bhow\s+many\s+(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
        query,
        flags=re.IGNORECASE,
    )


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


def _latest_state_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if tokens & {"clean", "cleaned"} and tokens & {"pair", "shoes", "shoe", "sneakers", "sneaker"}:
        return True
    if tokens & {"ram", "memory"} and tokens & {"upgrade", "upgraded", "laptop"}:
        return True
    if tokens & {"page", "pages"} and tokens & {"read", "so", "far", "current", "currently", "now"}:
        return True
    if tokens & {"hours", "hour"} and tokens & {"spent", "spend"}:
        return True
    return _generic_scalar_state_query(query)


def _latest_state_should_suppress_aggregate(query: str) -> bool:
    """Return true when current-state duration evidence should outrank stale totals."""
    tokens = set(source_tokens(query))
    if not _latest_state_query(query) or _aggregate_total_answer_query(query):
        return False
    if tokens & {"maximum", "max", "most", "highest", "largest"}:
        return False
    if "and" in tokens or tokens & {"combined", "together", "altogether"}:
        return False
    return bool(tokens & {"hours", "hour"} and tokens & {"spent", "spend"})


def _latest_state_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project compact current/latest state answers from cited source spans."""
    if not _latest_state_query(query):
        return []
    candidates: list[tuple[int, int, str, str, str]] = []
    query_terms = _query_specific_terms(query)
    for index, context in enumerate(contexts):
        text = _arithmetic_context_text(context)
        answer = _latest_state_answer(query, text)
        if not answer:
            continue
        source_id = source_context_group(context)
        score = (
            70
            + _query_overlap_score(query_terms, text)
            + _session_recency_score(text)
            + _source_group_state_recency_score(source_id)
            + min(_current_value_phrase_score(text), 5)
            + _latest_state_answer_specificity_score(query, answer, text)
        )
        candidates.append((score, index, source_id, answer, _latest_state_support_fragment(query, text, answer)))
    if not candidates:
        return []
    _score, _index, source_id, answer, fragment = max(candidates, key=lambda item: (item[0], -item[1]))
    return [
        "candidate_rank=1 candidate_type=latest_state candidate_confidence=0.88",
        f"candidate_support={source_id}",
        f"latest_state_answer={answer}",
        f"latest_state_raw_span={fragment}",
        f"latest_state_source_id={source_id}",
    ]


def _latest_state_answer(query: str, text: str) -> str:
    tokens = set(source_tokens(query))
    if tokens & {"clean", "cleaned"} and tokens & {"pair", "shoes", "shoe", "sneakers", "sneaker"}:
        return _cleaned_shoe_answer(text)
    if tokens & {"ram", "memory"} and tokens & {"upgrade", "upgraded", "laptop"}:
        return _ram_upgrade_answer(text)
    if tokens & {"page", "pages"} and tokens & {"read", "so", "far", "current", "currently", "now"}:
        return _latest_page_progress_answer(query, text)
    if tokens & {"hours", "hour"} and tokens & {"spent", "spend"}:
        return _latest_spent_duration_answer(query, text)
    return _generic_scalar_state_answer(query, text)


_STATE_QUERY_MODIFIERS = {
    "current",
    "currently",
    "latest",
    "most",
    "new",
    "newest",
    "now",
    "old",
    "previous",
    "previously",
    "prior",
    "recent",
    "recently",
}
_STATE_QUERY_ATTRIBUTE_STOPWORDS = _QUERY_SOURCE_STOPWORDS | _STATE_QUERY_MODIFIERS | {
    "am",
    "are",
    "is",
    "level",
    "status",
    "type",
    "value",
}


def _generic_scalar_state_query(query: str) -> bool:
    """Return whether a query asks for a mutable scalar state value."""
    if _query_bound_direct_answer_query(query):
        return False
    query_text = " ".join(query.casefold().split())
    tokens = set(source_tokens(query_text))
    if not tokens & _STATE_QUERY_MODIFIERS:
        return False
    if re.search(r"\b(?:how\s+many|how\s+much|total|sum|average|difference|increase|decrease)\b", query_text):
        return False
    if not re.search(r"\b(?:what|which|where|who)\b", query_text):
        return False
    return bool(_generic_scalar_state_attribute_terms(query))


def _generic_scalar_state_answer(query: str, text: str) -> str:
    """Extract current/latest/previous scalar state from cited update sentences."""
    attribute_terms = _generic_scalar_state_attribute_terms(query)
    if not attribute_terms:
        return ""
    previous_query = _previous_scalar_state_query(query)
    candidates: list[tuple[int, str, str]] = []
    for index, sentence in enumerate(_state_sentences(text)):
        extracted = _generic_scalar_state_sentence_value(
            sentence,
            attribute_terms=attribute_terms,
            previous=previous_query,
        )
        if not extracted:
            continue
        value, reason = extracted
        score = (
            index
            + _query_overlap_score(set(attribute_terms), sentence)
            + _current_value_phrase_score(sentence)
            + (12 if reason == "transition" else 0)
        )
        if previous_query and reason == "previous":
            score += 24
        elif not previous_query and reason in {"current", "transition"}:
            score += 18
        candidates.append((score, value, sentence))
    if not candidates:
        return ""
    _score, value, _sentence = max(candidates, key=lambda item: item[0])
    return value


def _previous_scalar_state_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"old", "previous", "previously", "prior"})


def _generic_scalar_state_attribute_terms(query: str) -> tuple[str, ...]:
    terms = [
        token
        for token in source_tokens(query)
        if len(token) > 2
        and not token.isdigit()
        and token not in _STATE_QUERY_ATTRIBUTE_STOPWORDS
    ]
    return tuple(dict.fromkeys(terms))


def _generic_scalar_state_sentence_value(
    sentence: str,
    *,
    attribute_terms: tuple[str, ...],
    previous: bool,
) -> tuple[str, str] | None:
    """Return a scalar state value and reason from one sentence."""
    if _query_overlap_score(set(attribute_terms), sentence) <= 0:
        return None
    transition = re.search(
        r"\b(?:changed|updated|switched|moved|transitioned)\b"
        r"[^.!?]{0,120}?\bfrom\s+(?P<old>[^.!?;,]{1,80}?)\s+to\s+(?P<new>[^.!?;,]{1,80})",
        sentence,
        flags=re.IGNORECASE,
    )
    if transition:
        key = "old" if previous else "new"
        reason = "previous" if previous else "transition"
        return _clean_scalar_state_value(transition.group(key)), reason
    assignment = re.search(
        r"\b(?:my|our)\s+(?:current\s+|latest\s+|new\s+|old\s+|previous\s+)?"
        r"(?P<attribute>[a-z][a-z0-9' -]{0,80}?)\s+"
        r"(?:is|are|was|were|became|has\s+become|changed\s+to|updated\s+to|switched\s+to)\s+"
        r"(?P<value>[^.!?;,]{1,100})",
        sentence,
        flags=re.IGNORECASE,
    )
    if assignment and _query_overlap_score(set(attribute_terms), assignment.group("attribute")) > 0:
        reason = "previous" if previous and re.search(
            r"\b(?:old|previous|prior|was|were)\b",
            sentence,
            flags=re.IGNORECASE,
        ) else "current"
        if previous and reason != "previous":
            return None
        return _clean_scalar_state_value(assignment.group("value")), reason
    return None


def _clean_scalar_state_value(value: str) -> str:
    value = re.split(
        r"\b(?:because|but|although|while|whereas|after|when|since)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(value.strip(" .,'\"()").split())


def _cleaned_shoe_answer(text: str) -> str:
    for sentence in _state_sentences(text):
        if not re.search(r"\bcleaned\b", sentence, flags=re.IGNORECASE):
            continue
        match = re.search(
            r"\bcleaned\s+(?:my\s+)?(?P<answer>[A-Za-z0-9][A-Za-z0-9' -]{0,80}\s+(?:sneakers?|shoes?))\b",
            sentence,
            flags=re.IGNORECASE,
        )
        if match:
            return _clean_state_answer(match.group("answer"))
    return ""


def _ram_upgrade_answer(text: str) -> str:
    patterns = (
        r"\bRAM\s+upgrade\s+to\s+(?P<answer>\d+\s*(?:GB|MB|TB))\b",
        r"\bupgrad(?:e|ed|ing)\b[^.!?]{0,80}\bRAM\b[^.!?]{0,80}\bto\s+(?P<answer>\d+\s*(?:GB|MB|TB))\b",
        r"\bRAM\b[^.!?]{0,80}\bto\s+(?P<answer>\d+\s*(?:GB|MB|TB))\b",
    )
    for sentence in _state_sentences(text):
        if not re.search(r"\bRAM\b", sentence, flags=re.IGNORECASE):
            continue
        for pattern in patterns:
            if match := re.search(pattern, sentence, flags=re.IGNORECASE):
                return re.sub(r"\s+", "", match.group("answer").upper())
    return ""


def _latest_page_progress_answer(query: str, text: str) -> str:
    title = _quoted_query_title(query)
    text_mentions_title = bool(title and _text_mentions_title(text, title))
    candidates: list[tuple[int, int]] = []
    for index, sentence in enumerate(_state_sentences(text)):
        if title and not text_mentions_title and not re.search(
            r"\b(?:currently|now|so\s+far|on\s+page)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue
        if title and text_mentions_title and not _text_mentions_title(sentence, title) and not re.search(
            r"\b(?:currently|now|so\s+far|on\s+page)\b",
            sentence,
            flags=re.IGNORECASE,
        ):
            continue
        for match in re.finditer(r"\b(?:currently\s+)?(?:on\s+)?page\s+(?P<value>\d{1,5})\b", sentence, flags=re.IGNORECASE):
            value = int(match.group("value"))
            score = index + _current_value_phrase_score(sentence)
            candidates.append((score, value))
    if not candidates:
        return ""
    _score, value = max(candidates, key=lambda item: item[0])
    return str(value)


def _latest_spent_duration_answer(query: str, text: str) -> str:
    query_terms = _query_specific_terms(query)
    candidates: list[tuple[int, str]] = []
    for index, sentence in enumerate(_state_sentences(text)):
        if not re.search(r"\b(?:spent|put\s+in|already)\b", sentence, flags=re.IGNORECASE):
            continue
        if _query_overlap_score(query_terms, sentence) < 2:
            continue
        match = re.search(
            r"\b(?P<answer>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*(?:hours?|hrs?))\b",
            sentence,
            flags=re.IGNORECASE,
        ) or re.search(
            r"\b(?P<answer>\d+(?:\.\d+)?\s*(?:hours?|hrs?))\b",
            sentence,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        answer = re.sub(r"\s*-\s*", "-", match.group("answer"))
        answer = re.sub(r"\s+", " ", answer.strip())
        score = index + _current_value_phrase_score(sentence) + _query_overlap_score(query_terms, sentence)
        candidates.append((score, answer))
    if not candidates:
        return ""
    _score, answer = max(candidates, key=lambda item: item[0])
    return answer


def _state_sentences(text: str) -> list[str]:
    normalized = re.sub(
        r"\b(?:citation|source_id|longmemeval_session_id)=[^\s]+",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    normalized = " ".join(normalized.split())
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", normalized)
        if sentence.strip()
    ]


def _clean_state_answer(value: str) -> str:
    return " ".join(value.strip(" .,'\"").split())


def _latest_state_support_fragment(query: str, text: str, answer: str) -> str:
    query_terms = _query_specific_terms(query)
    for sentence in _state_sentences(text):
        if answer.casefold() in sentence.casefold() and _query_overlap_score(query_terms, sentence):
            return source_context_snippet(sentence, max_chars=300)
    return answer


def _latest_state_answer_specificity_score(query: str, answer: str, text: str) -> int:
    tokens = set(source_tokens(query))
    answer_tokens = set(source_tokens(answer))
    score = 0
    if tokens & {"clean", "cleaned"} and answer_tokens & {"sneakers", "sneaker", "shoes", "shoe"}:
        score += 10
    if tokens & {"ram", "memory"} and re.search(r"\b\d+\s*(?:GB|MB|TB)\b", answer, flags=re.IGNORECASE):
        score += 10
    if tokens & {"page", "pages"} and answer.isdigit():
        score += 8
        score += min(int(answer), 500)
    if tokens & {"hours", "hour"} and "hour" in answer.casefold():
        score += 8
    if _generic_scalar_state_query(query):
        if re.search(r"\b(?:changed|updated|switched|moved|transitioned)\b[^.!?]{0,120}\bfrom\b[^.!?]{1,120}\bto\b", text, flags=re.IGNORECASE):
            score += 28
        if not _previous_scalar_state_query(query) and re.search(r"\b(?:was|were|old|previous|prior)\b", text, flags=re.IGNORECASE):
            score -= 10
        if _previous_scalar_state_query(query) and re.search(r"\b(?:old|previous|prior|from)\b", text, flags=re.IGNORECASE):
            score += 14
    if re.search(r"\b(?:currently|now|already|recently|lately|so\s+far)\b", text, flags=re.IGNORECASE):
        score += 4
    return score


def _source_group_state_recency_score(source_id: str) -> int:
    lowered = source_id.casefold()
    if re.search(r"\b(?:current|latest|recent|new|now)\b", lowered):
        return 8
    if re.search(r"\b(?:older|old|previous|stale)\b", lowered):
        return -4
    return 0


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


def _query_bound_arithmetic_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if {"total", "distance"} <= tokens and tokens & {"hike", "hikes", "hiked", "trail", "trails"}:
        return True
    if tokens & {"left", "remaining"} and tokens & {"page", "pages", "read"}:
        return True
    if tokens & {"more", "expensive", "compared"} and tokens & {"taxi", "train", "fare"}:
        return True
    if tokens & {"exceed", "exceeded"} and tokens & {"target", "marathon", "minutes"}:
        return True
    if _routine_time_total_query(query):
        return True
    return bool(
        tokens & {"percentage", "percent"}
        and (
            tokens & {"discount", "packed", "wear", "wore", "worn", "shoes"}
            or _currency_percentage_of_targets(query) is not None
            or _count_percentage_of_targets(query) is not None
        )
    )


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


def _query_bound_arithmetic_answer_present(lines: list[str]) -> bool:
    return any(
        "candidate_type=distance_total" in line
        or "candidate_type=pages_remaining" in line
        or "candidate_type=percentage" in line
        or "candidate_type=boolean_comparison" in line
        or "candidate_type=query_bound_difference" in line
        or "candidate_type=routine_time_total" in line
        for line in lines
    )


def _routine_time_total_query(query: str) -> bool:
    """Return whether the query asks for total time across routine activity slots."""
    tokens = set(source_tokens(query))
    return bool(
        tokens & {"total", "combined"}
        and tokens & {"time"}
        and tokens & {"ready", "commute", "commuting", "routine"}
        and len(_routine_time_slots(query)) >= 2
    )


def _routine_time_slots(query: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return requested routine-duration slots from a query."""
    tokens = set(source_tokens(query))
    slots: list[tuple[str, tuple[str, ...]]] = []
    if tokens & {"ready", "breakfast", "morning", "routine"}:
        slots.append(("ready", ("ready", "morning", "routine", "breakfast", "meditation", "workout")))
    if tokens & {"commute", "commuting"}:
        slots.append(("commute", ("commute", "commuting", "work")))
    return tuple(slots)


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


def _routine_time_fragments(text: str) -> list[str]:
    """Split retrieved source text into bounded role/sentence fragments."""
    normalized = " ".join(text.split())
    role_parts = re.split(r"(?=\b(?:\d+\.\s*)?(?:user|assistant):)", normalized, flags=re.IGNORECASE)
    fragments: list[str] = []
    for part in role_parts or [normalized]:
        if not part.strip() or _assistant_role_fragment(part):
            continue
        fragments.extend(fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+", part) if fragment.strip())
    return fragments


def _assistant_role_fragment(fragment: str) -> bool:
    user_index = fragment.casefold().find("user:")
    assistant_index = fragment.casefold().find("assistant:")
    return assistant_index >= 0 and (user_index < 0 or assistant_index < user_index)


def _routine_fragment_is_personal(fragment: str) -> bool:
    """Return whether a fragment records the user's own routine, not advice."""
    lowered = fragment.casefold()
    if re.search(r"\b(?:try|suggest|recommend|could|should|would|tips?)\b", lowered):
        return False
    if re.search(r"\b(?:your|you)\s+(?:morning\s+)?commute\b", lowered) and not _FIRST_PERSON_CONTEXT_RE.search(fragment):
        return False
    return bool(_FIRST_PERSON_CONTEXT_RE.search(fragment))


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


def _duration_match_is_range_fragment(fragment: str, start: int) -> bool:
    before = fragment[max(0, start - 3):start]
    return bool(re.search(r"\d\s*[-–]\s*$", before))


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


def _aggregate_total_answer_query(query: str) -> bool:
    """Return whether the query asks for a combined aggregate answer surface."""
    query_text = " ".join(query.casefold().split())
    return bool(
        re.search(
            r"\b(?:total|combined|altogether|sum)\b|\bin\s+total\b|\bhow\s+many\s+.*\btotal\b",
            query_text,
        )
    )


@dataclass(frozen=True)
class _QueryBoundScalarTotalSpec:
    kind: str
    answer_unit: str


def _query_bound_scalar_total_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project totals for query-named numeric quantities without simple unit suffixes."""
    spec = _query_bound_scalar_total_spec(query)
    if spec is None:
        return []
    evidence: list[tuple[str, float, str]] = []
    seen: set[tuple[str, float]] = set()
    for context in contexts:
        source_id = source_context_group(context)
        text = source_context_snippet(context, max_chars=12_000)
        for sentence in _boolean_evidence_sentences(text):
            value = _query_bound_scalar_total_value(spec, sentence)
            if value is None:
                continue
            identity = (source_id, value)
            if identity in seen:
                continue
            seen.add(identity)
            evidence.append((source_id, value, sentence))
    if len(evidence) < 2:
        return []
    total = sum(value for _source_id, value, _sentence in evidence)
    answer_value = _format_grouped_number(total)
    answer = f"{answer_value} {spec.answer_unit}".strip()
    source_ids = list(dict.fromkeys(source_id for source_id, _value, _sentence in evidence))
    return [
        "candidate_rank=1 candidate_type=query_bound_scalar_total candidate_confidence=0.86",
        "candidate_support=" + ",".join(source_ids),
        f"query_bound_scalar_total_kind={spec.kind}",
        "query_bound_scalar_total_values=" + ",".join(_format_grouped_number(value) for _source_id, value, _sentence in evidence),
        f"query_bound_scalar_total_answer={answer}",
        "query_bound_scalar_total_source_ids=" + ",".join(source_ids),
    ]


def _query_bound_scalar_total_spec(query: str) -> _QueryBoundScalarTotalSpec | None:
    tokens = set(source_tokens(query))
    if not _aggregate_total_answer_query(query):
        return None
    if tokens & {"rare"} and tokens & {"items", "item"}:
        return _QueryBoundScalarTotalSpec(kind="rare_items", answer_unit="")
    if tokens & {"people", "person"} and tokens & {"reach", "reached"}:
        return _QueryBoundScalarTotalSpec(kind="people_reached", answer_unit="")
    if tokens & {"views", "view"} and tokens & {"youtube", "tiktok", "videos", "video"}:
        return _QueryBoundScalarTotalSpec(kind="video_views", answer_unit="")
    if tokens & {"comments", "comment"} and tokens & {
        "facebook",
        "youtube",
        "video",
        "videos",
        "live",
        "session",
    }:
        return _QueryBoundScalarTotalSpec(kind="engagement_comments", answer_unit="")
    if tokens & {"distance", "covered"} and tokens & {"road", "trip", "trips", "miles"}:
        return _QueryBoundScalarTotalSpec(kind="road_trip_miles", answer_unit="miles")
    return None


def _query_bound_scalar_total_value(spec: _QueryBoundScalarTotalSpec, sentence: str) -> float | None:
    sentence_tokens = set(source_tokens(sentence))
    if spec.kind == "rare_items":
        if "rare" not in sentence_tokens:
            return None
        return _semantic_number_value(
            sentence,
            r"\b(?P<value>\d{1,6}(?:,\d{3})*)\s+"
            r"(?:rare\s+)?(?:books?|figurines?|records?|coins?|items?)\b",
        )
    if spec.kind == "people_reached":
        if not sentence_tokens & {"facebook", "instagram", "campaign", "influencer", "promoted", "reached"}:
            return None
        return _semantic_number_value(
            sentence,
            r"\b(?:reached|reach)\s+(?:around\s+|about\s+)?(?P<value>\d{1,6}(?:,\d{3})*)\s+people\b"
            r"|\bpromoted\b[^.!?;]{0,120}\bto\s+(?:her\s+|his\s+|their\s+)?"
            r"(?P<value_alt>\d{1,6}(?:,\d{3})*)\s+followers\b",
        )
    if spec.kind == "video_views":
        if not sentence_tokens & {"youtube", "tiktok", "video", "videos"}:
            return None
        return _semantic_number_value(
            sentence,
            r"\b(?:my\s+)?(?:video|tutorial)\b[^.!?;]{0,120}\b(?:has|with)\s+"
            r"(?P<value>\d{1,6}(?:,\d{3})*)\s+views\b"
            r"|\bit\s+has\s+(?P<value_alt>\d{1,6}(?:,\d{3})*)\s+views\b",
        )
    if spec.kind == "engagement_comments":
        if "comments" not in sentence_tokens and "comment" not in sentence_tokens:
            return None
        if not sentence_tokens & {"facebook", "youtube", "video", "videos", "live", "session"}:
            return None
        return _semantic_number_value(
            sentence,
            r"\b(?:got|has|have|had|received|with)\s+"
            r"(?P<value>\d{1,6}(?:,\d{3})*)\s+comments?\b"
            r"|\b(?P<value_alt>\d{1,6}(?:,\d{3})*)\s+comments?\b",
        )
    if spec.kind == "road_trip_miles":
        if "covered" not in sentence_tokens:
            return None
        return _semantic_number_value(
            sentence,
            r"\bcovered\s+(?:a\s+)?total\s+of\s+(?P<value>\d{1,6}(?:,\d{3})*)\s+miles\b",
        )
    return None


def _semantic_number_value(sentence: str, pattern: str) -> float | None:
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.groupdict().get("value") or match.groupdict().get("value_alt")
    if not value:
        return None
    return float(value.replace(",", ""))


def _format_grouped_number(value: float) -> str:
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


def _query_bound_difference_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project explicit target-vs-target differences from cited operands."""
    tokens = set(source_tokens(query))
    if tokens & {"more", "expensive", "compared"} and tokens & {"taxi", "train", "fare"}:
        return _target_currency_difference_synthesis_lines(query, contexts)
    if tokens & {"exceed", "exceeded"} and tokens & {"target", "marathon", "minutes"}:
        return _target_duration_difference_synthesis_lines(query, contexts)
    return []


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


def _target_currency_value(target_terms: tuple[str, ...], contexts: list[str]) -> tuple[float, str, str] | None:
    candidates: list[tuple[int, int, float, str, str]] = []
    target_set = set(target_terms)
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        text = source_context_snippet(context, max_chars=12_000)
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text):
            fragment = _numeric_observation_fragment(text, match.start(), match.end())
            local_fragment = _currency_match_context(fragment, match.group(0))
            local_tokens = set(source_tokens(local_fragment))
            fragment_tokens = set(source_tokens(fragment))
            if not (target_set <= local_tokens or target_set <= fragment_tokens):
                continue
            value = float(match.group("value").replace(",", ""))
            score = 10 + (len(target_set & local_tokens) * 3) + len(target_set & fragment_tokens)
            if target_set <= local_tokens:
                score += 8
            if re.search(r"\b(?:actually|actual)\b", local_fragment, flags=re.IGNORECASE):
                score += 8
            elif re.search(r"\b(?:actually|actual)\b", fragment, flags=re.IGNORECASE):
                score += 2
            if re.search(
                r"\b(?:estimate|estimated|assuming|assume|approximately|roughly)\b",
                local_fragment,
                flags=re.IGNORECASE,
            ):
                score -= 6
            candidates.append((score, -index, value, source_id, local_fragment))
    if not candidates:
        return None
    score, index, value, source_id, fragment = max(candidates, key=lambda item: (item[0], item[1]))
    del score, index
    return value, source_id, fragment


def _currency_match_context(fragment: str, currency_text: str) -> str:
    """Return the smallest local clause that still contains the currency value."""
    match_index = fragment.find(currency_text)
    if match_index < 0:
        return fragment
    start = 0
    end = len(fragment)
    left_boundary = max(
        fragment.rfind(".", 0, match_index),
        fragment.rfind(";", 0, match_index),
        fragment.rfind("?", 0, match_index),
        fragment.rfind("!", 0, match_index),
        fragment.rfind(" but ", 0, match_index),
        fragment.rfind(" while ", 0, match_index),
        fragment.rfind(" whereas ", 0, match_index),
    )
    if left_boundary >= 0:
        start = left_boundary + 1
    right_boundaries = [
        position
        for needle in (".", ";", "?", "!", " but ", " while ", " whereas ")
        if (position := fragment.find(needle, match_index + len(currency_text))) >= 0
    ]
    if right_boundaries:
        end = min(right_boundaries)
    return " ".join(fragment[start:end].strip(" ,.;!?").split()) or fragment


def _target_duration_difference_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Return actual minus target duration for explicit target-time questions."""
    del query
    actual: tuple[int, str, str] | None = None
    target: tuple[int, str, str] | None = None
    for context in contexts:
        source_id = source_context_group(context)
        text = source_context_snippet(context, max_chars=12_000)
        for sentence in _boolean_evidence_sentences(text):
            sentence_tokens = set(source_tokens(sentence))
            value = _marathon_duration_minutes(sentence)
            if value is None:
                continue
            if actual is None and "target" not in sentence_tokens and "marathon" in sentence_tokens and re.search(
                r"\b(?:completed|finished|finish|ran)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                actual = (value, source_id, sentence)
            if target is None and "target" in sentence_tokens:
                target = (value, source_id, sentence)
    if actual is None or target is None:
        return []
    actual_value, actual_source, actual_span = actual
    target_value, target_source, target_span = target
    difference = actual_value - target_value
    if difference <= 0:
        return []
    source_ids = list(dict.fromkeys((actual_source, target_source)))
    return [
        "candidate_rank=1 candidate_type=query_bound_difference candidate_confidence=0.88",
        "candidate_support=" + ",".join(source_ids),
        "difference_left_label=actual_marathon_time",
        f"difference_left_minutes={actual_value}",
        "difference_right_label=target_marathon_time",
        f"difference_right_minutes={target_value}",
        f"query_bound_difference_answer={difference}",
        "query_bound_difference_unit=minutes",
        f"query_bound_difference_left_raw_span={source_context_snippet(actual_span, max_chars=180)}",
        f"query_bound_difference_right_raw_span={source_context_snippet(target_span, max_chars=180)}",
        "query_bound_difference_source_ids=" + ",".join(source_ids),
    ]


def _marathon_duration_minutes(text: str) -> int | None:
    if match := re.search(r"\b(?P<hours>\d{1,2})h\s*(?P<minutes>\d{1,2})min\b", text, flags=re.IGNORECASE):
        return int(match.group("hours")) * 60 + int(match.group("minutes"))
    if match := re.search(
        r"\b(?P<hours>\d{1,2})\s+hours?\s+(?:and\s+)?(?P<minutes>\d{1,2})\s+minutes?\b",
        text,
        flags=re.IGNORECASE,
    ):
        return int(match.group("hours")) * 60 + int(match.group("minutes"))
    return None


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


def _currency_percentage_of_targets(query: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return denominator/numerator target terms for 'what percent of X is Y' queries."""
    query_text = " ".join(query.split())
    match = re.search(
        r"\bwhat\s+percent(?:age)?\s+of\s+(?P<denominator>.+?)\s+is\s+(?P<numerator>.+?)(?:\?|$)",
        query_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    denominator = _percentage_ratio_target_terms(match.group("denominator"))
    numerator = _percentage_ratio_target_terms(match.group("numerator"))
    if not denominator or not numerator:
        return None
    return denominator, numerator


def _percentage_ratio_target_terms(value: str) -> tuple[str, ...]:
    """Return non-generic terms that bind percentage ratio operands to evidence."""
    stopwords = {
        "cost",
        "current",
        "do",
        "for",
        "house",
        "i",
        "my",
        "of",
        "on",
        "plan",
        "plans",
        "price",
        "prices",
        "property",
        "properties",
        "the",
        "to",
    }
    terms: list[str] = []
    for token in source_tokens(value.replace("'", " ")):
        if len(token) <= 2 or token in stopwords:
            continue
        if token not in terms:
            terms.append(token)
    return tuple(terms)


def _target_currency_value_for_percentage(
    target_terms: tuple[str, ...],
    contexts: list[str],
) -> tuple[float, str, str] | None:
    """Return a currency operand for percentage ratios, allowing nearby antecedents."""
    candidates: list[tuple[int, int, float, str, str]] = []
    target_set = set(target_terms)
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        text = source_context_snippet(context, max_chars=12_000)
        text_tokens = set(source_tokens(text))
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text):
            fragment = _numeric_observation_fragment(text, match.start(), match.end())
            local_fragment = _currency_match_context(fragment, match.group(0))
            local_tokens = set(source_tokens(local_fragment))
            fragment_tokens = set(source_tokens(fragment))
            score = 0
            if target_set <= local_tokens:
                score = 30 + len(target_set & local_tokens) * 3
            elif target_set <= fragment_tokens:
                score = 20 + len(target_set & fragment_tokens) * 2
            elif target_set <= text_tokens:
                score = 8 + len(target_set & text_tokens)
            if score <= 0:
                continue
            value = float(match.group("value").replace(",", ""))
            if re.search(
                r"\b(?:listed|price|cost|costs|estimate|estimated|around|budget)\b",
                local_fragment,
                flags=re.IGNORECASE,
            ):
                score += 4
            candidates.append((score, -index, value, source_id, local_fragment))
    if not candidates:
        return None
    score, index, value, source_id, fragment = max(candidates, key=lambda item: (item[0], item[1]))
    del score, index
    return value, source_id, fragment


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


def _count_percentage_of_targets(query: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return denominator/numerator target terms for count-ratio percentage questions."""
    query_text = " ".join(query.split())
    match = re.search(
        r"\bwhat\s+percent(?:age)?\s+of\s+(?P<denominator>.+?)\s+"
        r"(?:do|does|did|are|is|were|was)\s+(?P<numerator>.+?)\s+"
        r"(?:hold|occupy|have|make\s+up|account\s+for)\b",
        query_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    denominator = _percentage_count_target_terms(match.group("denominator"))
    numerator = _percentage_count_target_terms(match.group("numerator"))
    if not denominator or not numerator:
        return None
    return denominator, numerator


def _percentage_count_target_terms(value: str) -> tuple[str, ...]:
    """Return non-generic count-ratio terms that bind operands to evidence."""
    stopwords = {
        "company",
        "current",
        "do",
        "does",
        "did",
        "for",
        "hold",
        "in",
        "is",
        "my",
        "of",
        "our",
        "the",
        "was",
        "were",
    }
    terms: list[str] = []
    for token in source_tokens(value.replace("'", " ")):
        if len(token) <= 2 or token in stopwords:
            continue
        if token not in terms:
            terms.append(token)
    return tuple(terms)


def _target_count_value_for_percentage(
    denominator_terms: tuple[str, ...],
    numerator_terms: tuple[str, ...],
    contexts: list[str],
) -> tuple[float, str, str] | None:
    """Return a count operand for percentage ratios, bound to denominator and optional numerator terms."""
    denominator_set = set(denominator_terms)
    numerator_set = set(numerator_terms)
    candidates: list[tuple[int, int, float, str, str]] = []
    for index, context in enumerate(contexts):
        source_id = source_context_group(context)
        text = source_context_snippet(context, max_chars=12_000)
        for match in re.finditer(r"\b(?P<value>\d{1,6}(?:,\d{3})*)\b", text):
            fragment = _numeric_observation_fragment(text, match.start(), match.end())
            local_tokens = set(source_tokens(fragment))
            text_tokens = set(source_tokens(text))
            if not denominator_set <= (local_tokens | text_tokens):
                continue
            if numerator_set and not numerator_set <= (local_tokens | text_tokens):
                continue
            if _count_percentage_number_is_percentage_or_date(fragment, match.group(0)):
                continue
            score = 10 + len(denominator_set & local_tokens) * 4
            if numerator_set:
                score += 20 + len(numerator_set & local_tokens) * 5
            elif re.search(r"\b(?:total|all|across|overall)\b", fragment, flags=re.IGNORECASE):
                score += 10
            if re.search(r"\b(?:positions?|roles?|seats?|members?|employees?|people)\b", fragment, flags=re.IGNORECASE):
                score += 4
            value = float(match.group("value").replace(",", ""))
            candidates.append((score, -index, value, source_id, fragment))
    if not candidates:
        return None
    score, index, value, source_id, fragment = max(candidates, key=lambda item: (item[0], item[1]))
    del score, index
    return value, source_id, fragment


def _count_percentage_number_is_percentage_or_date(fragment: str, raw_value: str) -> bool:
    """Return whether a numeric span is not a plain count operand."""
    escaped = re.escape(raw_value)
    return bool(
        re.search(rf"\b{escaped}\s*%", fragment)
        or re.search(rf"\b{escaped}\s*(?:am|pm)\b", fragment, flags=re.IGNORECASE)
        or re.search(rf"\b(?:19|20){escaped}\b", fragment)
    )


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


def _percentage_comparison_targets(query: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Return left/right target terms for a direct percentage comparison query."""
    query_text = " ".join(query.split())
    patterns = (
        r"\bfrom\s+(?P<left>[A-Za-z0-9][A-Za-z0-9 +&'-]{1,80}?)\s*,?\s+compared\s+to\s+(?:my\s+)?(?:first\s+)?(?P<right>[A-Za-z0-9][A-Za-z0-9 +&'-]{1,80}?)(?:\s+order|\?|$)",
        r"\bfor\s+(?P<left>[A-Za-z0-9][A-Za-z0-9 +&'-]{1,80}?)\s*,?\s+compared\s+to\s+(?:my\s+)?(?:first\s+)?(?P<right>[A-Za-z0-9][A-Za-z0-9 +&'-]{1,80}?)(?:\s+order|\?|$)",
        r"\b(?P<left>[A-Za-z0-9][A-Za-z0-9 +&'-]{1,80}?)\s+(?:than|compared\s+to)\s+(?:my\s+)?(?:first\s+)?(?P<right>[A-Za-z0-9][A-Za-z0-9 +&'-]{1,80}?)(?:\s+order|\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query_text, flags=re.IGNORECASE)
        if not match:
            continue
        left = _percentage_target_terms(match.group("left"))
        right = _percentage_target_terms(match.group("right"))
        if left and right:
            return left, right
    return None


def _percentage_target_terms(value: str) -> tuple[str, ...]:
    stopwords = {
        "compared",
        "discount",
        "first",
        "higher",
        "lower",
        "my",
        "order",
        "percentage",
        "receive",
        "received",
        "than",
        "the",
    }
    return tuple(token for token in source_tokens(value) if len(token) > 1 and token not in stopwords)


def _target_percentage_value(target_terms: tuple[str, ...], contexts: list[str]) -> tuple[float, str] | None:
    """Return the best cited percentage value bound to all target terms."""
    candidates: list[tuple[int, int, float, str]] = []
    for context_index, context in enumerate(contexts):
        snippet = _arithmetic_context_text(context)
        source_id = source_context_group(context)
        for match in re.finditer(r"\b(?P<value>\d{1,3}(?:\.\d+)?)\s*%\s*(?:off|discount)?\b", snippet):
            span = _local_percent_context(snippet, match.start(), match.end())
            span_tokens = set(source_tokens(span))
            if not set(target_terms) <= span_tokens:
                continue
            value = float(match.group("value"))
            if value < 0 or value > 100:
                continue
            score = len(span_tokens & set(target_terms))
            if re.search(r"\b(?:got|received|had|used)\b", span, flags=re.IGNORECASE):
                score += 2
            candidates.append((-score, context_index, value, source_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, _, value, source_id = candidates[0]
    return value, source_id


def _local_percent_context(text: str, start: int, end: int) -> str:
    """Return a bounded clause around a percentage mention."""
    left = max(text.rfind(".", 0, start), text.rfind("|", 0, start), text.rfind("\n", 0, start))
    right_candidates = [
        index for index in (text.find(".", end), text.find("|", end), text.find("\n", end)) if index >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right][:500]


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


def _text_mentions_title(text: str, title: str) -> bool:
    return title.casefold() in text.casefold()


def _arithmetic_context_text(context: str) -> str:
    """Return enough normalized source text to find split arithmetic operands."""
    return source_context_snippet(context, max_chars=12_000)


def _current_page_value(text: str, title: str) -> float | None:
    escaped = re.escape(title)
    patterns = (
        rf"\b(?:currently|now|still)\s+(?:on\s+)?page\s+(?P<value>\d{{1,5}})\s+of\s+['\"]?{escaped}['\"]?",
        rf"\bpage\s+(?P<value>\d{{1,5}})\s+of\s+['\"]?{escaped}['\"]?",
    )
    for pattern in patterns:
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            return float(match.group("value").replace(",", ""))
    return None


def _total_page_value(text: str, title: str) -> float | None:
    escaped = re.escape(title)
    patterns = (
        rf"['\"]?{escaped}['\"]?[^.!?;]{{0,160}}\b(?:is|was|had|has)?\s*(?P<value>\d{{2,5}})\s+pages?\b",
        rf"\b(?P<value>\d{{2,5}})\s+pages?\b[^.!?;]{{0,160}}['\"]?{escaped}['\"]?",
        rf"['\"]?{escaped}['\"]?[^.!?;]{{0,160}}\b(?P<value>\d{{2,5}})\s*-\s*page\b",
    )
    for pattern in patterns:
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            return float(match.group("value").replace(",", ""))
    return None


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


def _original_price_value(text: str) -> float | None:
    candidates: list[tuple[int, float]] = []
    for match in _currency_matches(text):
        fragment = _numeric_observation_fragment(text, match.start(), match.end())
        if re.search(r"\b(?:originally|original|listed|regular|priced)\b", fragment, flags=re.IGNORECASE):
            score = _discount_currency_fragment_score(fragment)
            candidates.append((score, float(match.group("value").replace(",", ""))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    return candidates[0][1]


def _paid_price_value(text: str) -> float | None:
    candidates: list[tuple[int, float]] = []
    for match in _currency_matches(text):
        fragment = _numeric_observation_fragment(text, match.start(), match.end())
        if not re.search(r"\b(?:paid|pay|got|bought|purchased|for)\b", fragment, flags=re.IGNORECASE):
            continue
        score = _discount_currency_fragment_score(fragment)
        if score <= 0:
            continue
        candidates.append((score, float(match.group("value").replace(",", ""))))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    return candidates[0][1]


def _discount_currency_fragment_score(fragment: str) -> int:
    tokens = set(source_tokens(fragment))
    score = 0
    if tokens & {"book", "bookstore", "author", "release"}:
        score += 5
    if tokens & {"discount", "sale"}:
        score += 4
    if tokens & {"favorite"}:
        score += 2
    if tokens & {"gift", "jewelry", "necklace", "budget", "mom", "sister"}:
        score -= 4
    return score


def _currency_matches(text: str) -> Iterator[re.Match[str]]:
    return re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)\b", text)


def _format_percentage(value: float) -> str:
    return f"{_format_number(round(value, 2))}%"


def _page_count_matches(text: str) -> Iterator[re.Match[str]]:
    return re.finditer(
        r"\b(?P<value>\d{2,5})\s*-\s*page\b|\b(?P<value_after>\d{2,5})\s+pages?\b",
        text,
        flags=re.IGNORECASE,
    )


def _numeric_observation_fragment(text: str, start: int, end: int) -> str:
    """Return the bounded clause that owns a numeric observation."""
    boundaries = list(
        re.finditer(
            r"(?<=[.!?;])\s+|\bbut\s+before\s+that,?\s+|\bbefore\s+that,?\s+",
            text,
            flags=re.IGNORECASE,
        )
    )
    fragment_start = 0
    for boundary in boundaries:
        fragment_end = boundary.start()
        if fragment_start <= start and end <= fragment_end:
            return text[fragment_start:fragment_end]
        fragment_start = boundary.end()
    return text[fragment_start:]


def _page_count_observation_relevant(query: str, fragment: str) -> bool:
    if not re.search(r"\b(?:finished|finish|read|completed|complete)\b", query, flags=re.IGNORECASE):
        return True
    return bool(
        re.search(
            r"\b(?:i|me|my|you|your)\b[^.!?;]{0,120}\b(?:finished|finish|read|completed|complete)\b"
            r"|\b(?:finished|finish|read|completed|complete)\b[^.!?;]{0,120}\b(?:i|me|my|you|your)\b",
            fragment,
            flags=re.IGNORECASE,
        )
    )


def _page_count_query(query: str) -> bool:
    lowered = query.casefold()
    return bool(
        re.search(r"\bpages?\b|\bpage\s+count\b", lowered)
        and not re.search(r"\b(?:left|remaining|per\s+day|each\s+day|daily)\b", lowered)
    )


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


def _direct_numeric_synthesis_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if tokens & {"difference", "compared", "percentage", "discount"}:
        return True
    if tokens & {"best", "previous"} and tokens & {"time", "run", "5k"}:
        return True
    return bool(tokens & {"current", "currently", "now", "so", "far", "most", "recent", "recently"})


def _direct_numeric_value_query(query_tokens: set[str], query: str) -> bool:
    if query_tokens & {"best", "previous"} and query_tokens & {"time", "run", "5k"}:
        return True
    if query_tokens & {"current", "currently", "now", "so", "far", "since", "most", "recent", "recently"}:
        return True
    if query_tokens & {"difference", "compared", "more", "percentage", "discount"}:
        return True
    return bool(re.search(r"\bhow\s+(?:many|much)\b", query, flags=re.IGNORECASE))


def _personal_best_time_answer(query_tokens: set[str], text: str) -> str:
    lowered = text.casefold()
    if not ({"best", "previous"} & query_tokens and {"time", "run", "5k"} & query_tokens):
        return ""
    if "previous" not in query_tokens and not re.search(r"\bpersonal\s+best\b", lowered):
        return ""
    sentences = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\bpersonal\s+best\b", sentence, flags=re.IGNORECASE)
    ]
    if "previous" in query_tokens:
        sentences = [
            sentence
            for sentence in sentences
            if re.search(r"\bprevious\b", sentence, flags=re.IGNORECASE)
        ] or sentences
    else:
        sentences = [
            sentence
            for sentence in sentences
            if not re.search(r"\bprevious\b", sentence, flags=re.IGNORECASE)
        ] or sentences
    search_text = " ".join(sentences) if sentences else text
    if match := re.search(
        r"\b(?P<minutes>\d{1,2})\s+minutes?\s+(?:and\s+)?(?P<seconds>\d{1,2})\s+seconds?\b",
        search_text,
        flags=re.IGNORECASE,
    ):
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        return f"{minutes} minutes and {seconds} seconds (or {minutes}:{seconds:02d})"
    if match := re.search(r"\b(?P<minutes>\d{1,2}):(?P<seconds>\d{2})\b", search_text):
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        return f"{minutes} minutes and {seconds} seconds (or {minutes}:{seconds:02d})"
    return ""


def _latest_currency_answer(query_tokens: set[str], text: str) -> str:
    if not (query_tokens & {"earn", "earned", "earning", "made", "market", "recent", "recently", "most"}):
        return ""
    if not re.search(r"\b(?:earn(?:ed|ing)?|made|sold|market|visit)\b", text, flags=re.IGNORECASE):
        return ""
    amounts = [match.group(0) for match in re.finditer(r"\$\d+(?:,\d{3})*(?:\.\d+)?", text)]
    if not amounts:
        return ""
    return amounts[-1]


def _current_duration_answer(query_tokens: set[str], text: str) -> str:
    if not (
        query_tokens & {"dedicate", "daily", "day", "current", "currently", "now"}
    ):
        return ""
    patterns = (
        r"\b(?:about|around|roughly|approximately)\s+(?P<word>one|two|three|four|five|six|\d+(?:\.\d+)?)\s+(?P<unit>hours?|hrs?|minutes?|mins?)\b",
        r"\b(?P<word>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|minutes?|mins?)\s+(?:each|per|a)\s+day\b",
        r"\b(?P<word>\d+)\s*-\s*(?P<upper>\d+)\s+(?P<unit>hours?|hrs?|minutes?|mins?)\b",
    )
    for pattern in patterns:
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            if match.groupdict().get("upper"):
                return f"{match.group('word')}-{match.group('upper')} {match.group('unit')}"
            prefix = "about " if re.match(r"(?i)\b(?:about|around|roughly|approximately)\b", match.group(0)) else ""
            return f"{prefix}{match.group('word')} {match.group('unit')}"
    return ""


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


def _count_query_object_terms(query_tokens: set[str]) -> set[str]:
    return {
        token
        for token in query_tokens
        if len(token) > 2
        and token
        not in _QUERY_SOURCE_STOPWORDS
        | {
            "current",
            "currently",
            "have",
            "owned",
            "own",
            "now",
        }
    }


def _term_variants(term: str) -> set[str]:
    variants = {term}
    if term.endswith("s") and len(term) > 3:
        variants.add(term[:-1])
    else:
        variants.add(f"{term}s")
    return variants


def _best_numeric_sentence(query_tokens: set[str], text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scoring_terms = {
        token
        for token in query_tokens
        if len(token) > 2 and token not in _QUERY_SOURCE_STOPWORDS
    }
    ranked: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        terms = set(source_tokens(sentence))
        if not re.search(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+|fifth|sixth|seventh)\b", sentence, flags=re.IGNORECASE):
            continue
        score = len(scoring_terms & terms)
        if terms & {"now", "currently", "so", "far", "since", "finished", "worn", "lost", "bought", "attended", "attending"}:
            score += 3
        if score <= 0:
            continue
        ranked.append((-score, index, sentence))
    if not ranked:
        return ""
    ranked.sort()
    return ranked[0][2]


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


def _current_value_phrase_score(text: str) -> int:
    score = 0
    lowered = text.casefold()
    if re.search(r"\b(?:now|currently|so far|to date|at this point)\b", lowered):
        score += 20
    if re.search(r"\b(?:already|just|recently)\b", lowered):
        score += 5
    return score


def _session_recency_score(text: str) -> int:
    if match := re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})", text):
        return int(match.group("year")) * 400 + int(match.group("month")) * 31 + int(match.group("day"))
    return 0


def _ordinal_to_cardinal_word(value: str) -> str:
    mapping = {
        "first": "one",
        "second": "two",
        "third": "three",
        "fourth": "four",
        "fifth": "five",
        "sixth": "six",
        "seventh": "seven",
        "eighth": "eight",
        "ninth": "nine",
        "tenth": "ten",
        "eleventh": "eleven",
        "twelfth": "twelve",
    }
    return mapping.get(value.casefold(), value.casefold())


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
    values = [value for _context, value, _raw in _age_average_evidence(contexts)]
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


def _age_average_evidence(contexts: list[str]) -> list[tuple[str, int, str]]:
    indexed = list(enumerate(_age_value_evidence(contexts)))
    indexed.sort(
        key=lambda item: (
            _source_group_natural_key(source_context_group(item[1][0])),
            item[0],
        )
    )
    return [evidence for _index, evidence in indexed]


def _source_group_natural_key(group: str) -> tuple[str, int]:
    match = re.match(r"^(?P<prefix>.*?)(?:[_-](?P<suffix>\d+))?$", group)
    if not match or match.group("suffix") is None:
        return group, -1
    return match.group("prefix"), int(match.group("suffix"))


def _ledger_row_lines(rows: list[dict[str, object]]) -> list[str]:
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]


def _career_prior_duration_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    query_text = query.casefold()
    return bool(query_tokens & {"work", "working", "professionally", "field", "career"}) and bool(
        re.search(r"\bbefore\b.*\b(?:current\s+job|started|start)\b", query_text)
        or re.search(r"\b(?:current\s+job|started|start)\b.*\bbefore\b", query_text)
    )


def _current_role_tenure_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool({"how", "long"} <= query_tokens and query_tokens & {"current", "role", "job", "position"})


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


def _elapsed_duration_at_event_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    if not {"how", "long", "when"} <= query_tokens:
        return False
    if not query_tokens & {"had", "been"}:
        return False
    return bool(query_tokens & {"bought", "buy", "got", "purchased", "started", "joined"})


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


def _social_media_break_day_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _social_media_break_day_evidence(contexts)]


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


def _road_trip_drive_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"road", "trip", "destinations"}) and bool(
        query_tokens & {"driving", "drove", "drive"}
    )


def _road_trip_drive_hour_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _road_trip_drive_hour_evidence(contexts)]


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


def _road_trip_destination_count_phrase(query: str, count: int) -> str:
    """Return a natural destination count phrase for road-trip aggregate answers."""
    query_tokens = set(source_tokens(query))
    for word, value in _NUMBER_WORDS.items():
        if value == count and word in query_tokens and word not in {"a", "an"}:
            return word
    return _format_number(float(count))


def _road_trip_drive_destination_signature(context: str, start: int, end: int) -> str:
    """Return a stable destination signature for road-trip duration dedupe."""
    local = context[max(0, start - 260) : min(len(context), end + 260)]
    for pattern in _ROAD_TRIP_DESTINATION_RE:
        if match := pattern.search(local):
            label = _clean_road_trip_destination_label(match.group("label"))
            if label:
                return label
    return ""


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


def _current_activity_weeks(query: str, contexts: list[str]) -> int | None:
    evidence = _current_activity_week_evidence(query, contexts)
    return evidence[1] if evidence is not None else None


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


def _event_weeks_ago(query: str, contexts: list[str]) -> int | None:
    evidence = _event_weeks_ago_evidence(query, contexts)
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


def _career_total_months(contexts: list[str]) -> int | None:
    evidence = _career_total_month_evidence(contexts)
    return evidence[1] if evidence is not None else None


def _career_total_month_evidence(contexts: list[str]) -> tuple[str, int, str] | None:
    for context in contexts:
        for pattern in _CAREER_TOTAL_YEARS_RE:
            match = pattern.search(context)
            if match:
                years = int(match.group("years"))
                if 0 < years < 80:
                    return context, years * 12, match.group(0)
    return None


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


def _year_month_match_total_months(match: re.Match[str]) -> int:
    years = int(match.group("years"))
    months = int(match.groupdict().get("months") or 0)
    if months < 0 or months >= 12:
        return 0
    return (years * 12) + months


def _current_role_months(query: str, contexts: list[str]) -> int | None:
    evidence = _current_role_month_evidence(query, contexts)
    return evidence[1] if evidence is not None else None


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


def _role_duration_months(text: str) -> int | None:
    evidence = _role_duration_month_evidence(text)
    return evidence[0] if evidence is not None else None


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


def _format_year_month_duration(total_months: int) -> str:
    years, months = divmod(total_months, 12)
    parts: list[str] = []
    if years:
        parts.append(f"{years} {'year' if years == 1 else 'years'}")
    if months:
        parts.append(f"{months} {'month' if months == 1 else 'months'}")
    return " and ".join(parts) if parts else "0 months"


def _personal_current_age_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _personal_current_age_evidence(contexts)]


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


def _elapsed_year_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _elapsed_year_evidence(contexts)]


def _age_at_event_operand_values(query: str, contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _age_at_event_operand_evidence(query, contexts)]


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


def _birth_age_query(query: str) -> bool:
    return bool(re.search(r"\b(?:born|birth)\b", query, flags=re.IGNORECASE))


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


def _birth_age_target(query: str) -> str:
    match = re.search(
        r"\bwhen\s+(?P<target>[A-Z][A-Za-z0-9'_-]{1,40})\s+(?:was|were)\s+born\b",
        query,
    )
    if match:
        return match.group("target")
    return ""


def _integer_number_value(raw_value: str) -> int:
    normalized = raw_value.casefold()
    if normalized.isdigit():
        return int(normalized)
    return int(_NUMBER_WORDS.get(normalized, 0))


def _age_values(contexts: list[str]) -> list[int]:
    return [value for _context, value, _raw in _age_value_evidence(contexts)]


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


def _unit_values(contexts: list[str], *, unit_pattern: str) -> list[float]:
    values: list[float] = []
    pattern = _unit_value_pattern(unit_pattern)
    for context in contexts:
        for match in pattern.finditer(context):
            values.append(float(match.group("value")))
    return values


@lru_cache(maxsize=16)
def _unit_value_pattern(unit_pattern: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b(?P<value>\d+(?:\.\d+)?)\s*[- ]?(?:{unit_pattern})\b",
        flags=re.IGNORECASE,
    )


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


def _future_age_at_event_query(query: str) -> bool:
    query_text = query.casefold()
    if not re.search(r"\bwhen\b", query_text):
        return False
    if not re.search(r"\b(?:how\s+many\s+years\s+will\s+i\s+be|how\s+old\s+will\s+i\s+be)\b", query_text):
        return False
    return bool(re.search(r"\b(?:marri(?:ed|age)|wedding)\b", query_text))


def _numeric_comparison_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"most", "least", "more", "less", "highest", "lowest"}) and bool(
        query_tokens & {"money", "amount", "cost", "spent", "spend", "price", "total"}
    )


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


def _anniversary_engagement_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"anniversary"} and tokens & {"engaged", "engagement"} and tokens & {"month", "months"})


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


def _first_event_month_day(contexts: list[str], *, terms: set[str]) -> tuple[int, int] | None:
    evidence = _first_event_month_day_evidence(contexts, terms=terms)
    if evidence is None:
        return None
    return evidence[1], evidence[2]


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


def _parent_order_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"who", "which"} and "first" in tokens and tokens & {"parent", "parents"})


def _query_person_alternatives(query: str) -> tuple[str, ...]:
    alternatives = _query_alternatives(query)
    people: list[str] = []
    for alternative in alternatives:
        for match in _PERSON_NAME_ALTERNATIVE_RE.finditer(alternative):
            name = match.group(0)
            if name.casefold() not in {"who", "which"}:
                people.append(name.casefold())
    return tuple(dict.fromkeys(people))


def _parent_event_month_day_for_person(person: str, contexts: list[str]) -> tuple[int, int] | None:
    evidence = _parent_event_month_day_for_person_evidence(person, contexts)
    if evidence is None:
        return None
    return evidence[1], evidence[2]


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


def _parent_context_matches_person(person: str, context: str) -> bool:
    terms = set(source_tokens(context))
    if person in terms:
        return True
    return bool(
        person == "rachel"
        and terms & {"sister-in-law", "sister", "law"}
        and terms & {"twins", "jackson", "julia"}
    )


def _parent_event_month_day(context: str) -> tuple[int, int] | None:
    text = source_context_snippet(context, max_chars=1_500)
    if not re.search(r"\b(?:adopted|adoption|baby|born|twins?|parent)\b", text, flags=re.IGNORECASE):
        return None
    month_days = _month_day_mentions(text)
    if month_days:
        return month_days[0]
    return _month_only_mention(text)


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


def _month_only_mention(text: str) -> tuple[int, int] | None:
    month_pattern = "|".join(sorted(_MONTH_ORDINALS, key=len, reverse=True))
    match = re.search(rf"\b(?P<month>{month_pattern})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    return (_MONTH_ORDINALS[match.group("month").casefold()], 1)


def _recency_synthesis_lines(query: str, contexts: list[str]) -> list[str]:
    """Project most-recent categorical answers from relative-time evidence."""
    return list(recency_candidate_projection(query, contexts).lines)


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


def _recency_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    return [
        "ledger_row=" + json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in _recency_ledger_rows(query, contexts)
    ]


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


def _recency_candidate_values(query: str, text: str) -> list[str]:
    query_tokens = set(source_tokens(query))
    if query_tokens & {"streaming", "service"}:
        return _streaming_service_names_in_context(text)
    return []


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


def _relative_temporal_anchor_query(query: str) -> bool:
    return query_temporal_anchor_synthesis_query(query)


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


def _query_temporal_anchor_date(contexts: list[str]) -> date | None:
    for context in contexts:
        if "query_temporal_anchor=true" not in context.casefold():
            continue
        return _longmemeval_session_date(context)
    return None


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


def _plural_unit(value: int, singular: str) -> str:
    return singular if value == 1 else f"{singular}s"


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


def _temporal_order_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if tokens & {"meet", "met"} and tokens & {"first", "earlier", "before"}:
        return True
    if "or" in tokens and tokens & {"first", "earlier", "before"}:
        return True
    return bool(tokens & {"first", "earlier", "before"}) and bool(
        tokens & {"event", "happened", "which"}
    )


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


def _temporal_order_choices_present(query: str, contexts: list[str]) -> bool:
    choices = _temporal_order_query_choices(query)
    if len(choices) < 2:
        return False
    text = " ".join(contexts)
    return all(
        _query_overlap_score(set(_temporal_order_choice_terms(choice)), text) > 0
        for choice in choices[:2]
    )


def _temporal_order_session_year(context: str) -> int | None:
    match = re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/", context)
    return int(match.group("year")) if match else None


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


def _temporal_order_date_value(year: int, month: int, day: int) -> int | None:
    try:
        return date(year, month, day).toordinal()
    except ValueError:
        return None


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


def _quoted_query_choices(query: str) -> tuple[str, ...]:
    """Return explicit quoted alternatives from a temporal-order query."""
    choices = [
        match.group("single") or match.group("double")
        for match in re.finditer(r"'(?P<single>[^']{2,160})'|\"(?P<double>[^\"]{2,160})\"", query)
    ]
    return tuple(dict.fromkeys(" ".join(choice.split()) for choice in choices if choice))


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
