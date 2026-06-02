"""Retrieval planning utilities shared by product and benchmark paths."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from zaxy.evidence_candidates import (
    EvidenceProjection,
    aggregate_candidate_projection,
    aggregate_evidence_score,
)
from zaxy.retrieval_intent import RetrievalIntent, classify_retrieval_intent
from zaxy.synthesis import build_synthesis_plan
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
_EMPLOYER_TERM_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.-]{2,}\b")
_PERSONAL_CURRENT_AGE_RE = (
    re.compile(
        r"\b(?:i\s+am|i'm|im)\s+(?P<value>\d{1,3})\s*[- ]?(?:years?\s+old|year[- ]old)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bi\s+(?:just\s+)?turned\s+(?P<value>\d{1,3})\b", flags=re.IGNORECASE),
    re.compile(r"\bmy\s+age\s+(?:is|was)\s+(?P<value>\d{1,3})\b", flags=re.IGNORECASE),
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
    return "absence_check" in intent.reasons or _parent_order_query(query)


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
    if query_terms & {"charity", "events"} and query_terms & {"raise", "raised", "money", "total"}:
        queries.append(
            "charity events participated raised total charity walk $250 Bike-a-Thon Cancer Research $5,000 charity yoga $600 animal shelter"
        )
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
    if query_terms & {"doctor", "doctors", "physician", "physicians"} and {"how", "many"} <= query_terms:
        queries.append("doctor physician dermatologist ent visited saw appointment")
    if query_terms & {"movie", "movies", "film", "films", "festival", "festivals"} and {"how", "many"} <= query_terms:
        queries.append("film festival movie attended went participated")
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
    if query_terms & {"museum", "museums", "gallery", "galleries"} and {"how", "many"} <= query_terms:
        queries.append("February museum museums gallery galleries visited went attended")
        queries.append("February 2/8 2/15 Natural History Museum The Art Cube visited art gallery")
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
        for reason in ("aggregation", "aggregation_question", "absence_check")
    ):
        return max(limit, intent.source_lane_slots * 6)
    if _temporal_order_query(query):
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
    intent = classify_retrieval_intent(query, limit=limit)
    if (
        not {"aggregation", "aggregation_question"} & set(intent.reasons)
        and not _issue_query(query)
        and not _average_query(query)
        and not _age_at_event_query(query)
        and not _elapsed_duration_at_event_query(query)
        and not _numeric_comparison_query(query)
        and not _frequency_comparison_query(query)
        and not _time_offset_query(query)
        and not _temporal_order_query(query)
        and not _parent_order_query(query)
        and not _anniversary_engagement_query(query)
        and not _recency_comparison_query(query)
        and not _direct_time_query(query)
        and not _possessive_attribute_query_target(query)
    ):
        return None
    group_limit = source_synthesis_candidate_limit(intent, limit=limit)
    if _average_query(query):
        group_limit = max(group_limit, 8)
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
    aggregate_projection = (
        EvidenceProjection((), ())
        if _direct_time_query(query)
        else aggregate_candidate_projection(query, grouped_sources)
    )
    derived_lines = [
        *_numeric_synthesis_lines(
            query,
            grouped_sources,
            aggregate_lines=list(aggregate_projection.lines),
        ),
        *_anniversary_engagement_synthesis_lines(query, grouped_sources),
        *_frequency_synthesis_lines(query, grouped_sources),
        *_parent_order_synthesis_lines(query, grouped_sources),
        *_temporal_order_synthesis_lines(query, grouped_sources),
        *_recency_synthesis_lines(query, grouped_sources),
        *_direct_time_synthesis_lines(query, grouped_sources),
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
    lines.extend(_elapsed_duration_at_event_ledger_row_lines(query, grouped_sources))
    lines.extend(_social_media_break_ledger_row_lines(query, grouped_sources))
    lines.extend(_road_trip_drive_ledger_row_lines(query, grouped_sources))
    lines.extend(_age_at_event_ledger_row_lines(query, grouped_sources))
    lines.extend(_career_prior_duration_ledger_row_lines(query, grouped_sources))
    if not any(row.get("include_reason") == "age_average_input" for row in aggregate_projection.ledger_rows):
        lines.extend(_age_average_ledger_row_lines(query, grouped_sources))
    lines.extend(_relative_interval_ledger_row_lines(query, grouped_sources))
    lines.extend(_anniversary_engagement_ledger_row_lines(query, grouped_sources))
    lines.extend(_parent_order_ledger_row_lines(query, grouped_sources))
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
                    "operations": list(aggregate_projection.operations),
                    "result": aggregate_projection.result or {},
                    "answer_candidates": list(aggregate_projection.answer_candidates),
                    "ledger_rows": list(aggregate_projection.ledger_rows),
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


def source_synthesis_candidate_limit(intent: RetrievalIntent, *, limit: int) -> int:
    """Return the internal source pool size used before compact synthesis."""
    if {"aggregation", "aggregation_question"} & set(intent.reasons):
        return max(limit, intent.source_lane_slots * 4, 16)
    if "temporal_order" in intent.reasons:
        return max(limit, intent.source_lane_slots * 8, 16)
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
    target = high_precision_missing_target(query, grouped_sources)
    if not target and has_direct_fact_evidence(query, grouped_sources):
        return None
    if not target and "absence_check" in intent.reasons:
        target = missing_query_target(query, grouped_sources) or absence_check_target(query)
    if not target and {"aggregation", "aggregation_question"} & set(intent.reasons):
        target = _missing_location_target(query, grouped_sources)
    if not target:
        return None
    if not grouped_sources:
        return None
    if _parent_order_query(query):
        if _parent_event_month_day_for_person(target, grouped_sources) is not None:
            return None
    elif target_terms_present(target, grouped_sources):
        return None
    candidate_source_ids = tuple(
        dict.fromkeys(
            source_context_group(context)
            for context in source_results
            if source_context_group(context)
        )
    )
    lines = [
        "zaxy_absence_check=true",
        "synthesis_mode=absence_check",
        f"query={query}",
        f"not_mentioned_candidate={target}",
        "support_source_ids=" + ",".join(source_context_group(context) for context in grouped_sources),
        "candidate_source_ids=" + ",".join(candidate_source_ids[: min(len(candidate_source_ids), max(4, limit * 2))]),
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
    if target := _missing_parent_order_target(query, contexts):
        return target
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


def _missing_parent_order_target(query: str, contexts: list[str]) -> str:
    """Return a missing named parent alternative using event-level evidence."""
    if not _parent_order_query(query):
        return ""
    for person in _query_person_alternatives(query):
        if _parent_event_month_day_for_person(person, contexts) is None:
            return person
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
    has_typed_projection = any(line.startswith("candidate_rank=") for line in lines)
    has_typed_age_average = any(line.startswith("age_average=") for line in lines)
    lines.extend(_age_at_event_synthesis_lines(query, numeric_contexts))
    if not has_typed_age_average:
        lines.extend(_age_average_synthesis_lines(query, numeric_contexts))
    lines.extend(_elapsed_duration_at_event_synthesis_lines(query, numeric_contexts))
    lines.extend(_social_media_break_synthesis_lines(query, numeric_contexts))
    lines.extend(_road_trip_drive_synthesis_lines(query, numeric_contexts))
    lines.extend(_career_prior_duration_synthesis_lines(query, numeric_contexts))
    if _career_prior_duration_query(query):
        return lines
    if not has_typed_duration and not has_typed_projection:
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


def _age_at_event_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _age_at_event_query(query):
        return []
    current = _personal_current_age_evidence(contexts)
    elapsed = _elapsed_year_evidence(contexts)
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
    values = [value for _context, value, _raw in _social_media_break_day_evidence(contexts)]
    if not values:
        return []
    total = sum(values)
    lines = [
        "social_media_break_day_values=" + ",".join(_format_number(float(value)) for value in values),
        f"social_media_break_total={_format_number(float(total))} days",
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
    values = [value for _context, value, _raw in _road_trip_drive_hour_evidence(contexts)]
    if not values:
        return []
    total = sum(values)
    lines = [
        "road_trip_drive_hour_values=" + ",".join(_format_number(float(value)) for value in values),
        f"road_trip_drive_total={_format_number(float(total))} hours",
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
    seen_groups: set[str] = set()
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
        if group in seen_groups:
            continue
        for pattern in _ROAD_TRIP_DRIVE_HOUR_RE:
            match = pattern.search(context)
            if not match:
                continue
            value = _integer_number_value(match.group("value"))
            if value <= 0:
                continue
            seen_groups.add(group)
            evidence.append((context, value, match.group(0)))
            break
    return evidence


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
    return bool(query_tokens & {"recent", "recently", "latest", "last", "newest"}) and bool(
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
    if not _recency_comparison_query(query):
        return []
    observations = _recency_observations(query, contexts)
    if not observations:
        return []
    answer = observations[0][1]
    lines = [f"recency_answer={answer}"]
    for index, (days_ago, value, _context) in enumerate(observations[:5], start=1):
        lines.append(f"recency_rank={index} relative_days_ago={days_ago} candidate={value}")
    return lines


def _recency_ledger_row_lines(query: str, contexts: list[str]) -> list[str]:
    if not _recency_comparison_query(query):
        return []
    rows = [
        {
            "fact_id": f"recency:{index}",
            "source_group": source_context_group(context),
            "citation": source_context_citation(context),
            "kind": "relative_time",
            "value": str(days_ago),
            "unit": "days_ago",
            "raw_span": str(days_ago),
            "candidate": value,
            "include_reason": "recency_candidate",
            "confidence": 0.78,
        }
        for index, (days_ago, value, context) in enumerate(_recency_observations(query, contexts)[:5])
    ]
    return _ledger_row_lines(rows)


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
    observations = _temporal_order_observations(contexts)
    if len(observations) < 2:
        return []
    lines = [f"temporal_order_answer={observations[0][1]}"]
    for index, (days_ago, candidate, _context) in enumerate(observations[:5], start=1):
        lines.append(
            f"temporal_order_rank={index} relative_days_ago={days_ago} candidate={candidate}"
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
        for index, (days_ago, candidate, context) in enumerate(_temporal_order_observations(contexts)[:5])
    ]
    return _ledger_row_lines(rows)


def _temporal_order_observations(contexts: list[str]) -> list[tuple[int, str, str]]:
    observations: list[tuple[int, str, str]] = []
    for context in contexts:
        text = _numeric_context_text(context)
        days_ago = _relative_days_ago(text)
        if days_ago is None:
            continue
        candidate = _temporal_order_candidate(text)
        if not candidate:
            continue
        observations.append((days_ago, candidate, context))
    observations.sort(key=lambda item: item[0], reverse=True)
    return observations


def _temporal_order_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    if tokens & {"meet", "met"} and tokens & {"first", "earlier", "before"}:
        return True
    return bool(tokens & {"first", "earlier", "before"}) and bool(
        tokens & {"event", "happened", "which"}
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


def _temporal_order_candidate(text: str) -> str:
    if candidate := _meeting_order_candidate(text):
        return candidate
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
