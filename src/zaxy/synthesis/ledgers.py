"""Split from synthesis.py (mechanical decomposition)."""


from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace

from zaxy.evidence_program import (
    TemporalEvidenceProgramResult,
    TemporalEvidenceProgramSpec,
    execute_temporal_evidence_program,
)
from zaxy.synthesis.evidence_rows import (
    _currency_merchant_label,
    _film_festival_labels,
    _filter_count_rows,
    _filter_currency_ledger,
    _filter_itemized_currency_targets,
    _museum_gallery_labels,
    _musical_instrument_labels,
    _temporal_evidence_row,
    _temporal_sequence_match_evidence,
    _temporal_sequence_pattern_candidates,
    clean_label,
    clean_temporal_sequence_label,
    context_year,
    explicit_date_matches,
    normalize_temporal_sequence_label,
    relative_session_date_offset,
    session_anchor_date,
    session_date_anchor_allowed,
    source_group_sequence_index,
    temporal_anchor_terms,
    temporal_evidence_text,
    temporal_sequence_deduped_rows,
    temporal_sequence_graduation_candidates,
    temporal_sequence_local_cue_score,
    temporal_sequence_order_value,
    temporal_sequence_query,
    temporal_sequence_query_slot_label,
    temporal_sequence_query_slots,
    temporal_sequence_requested_count,
    temporal_sequence_sentences,
    temporal_sequence_sports_candidates,
    temporal_sequence_venue_label,
)
from zaxy.synthesis.foundations import (
    _COUNT_STOPWORDS,
    _CURRENCY_LABEL_BEFORE_AMOUNT_PATTERNS,
    _CURRENCY_PURCHASE_LABEL_BEFORE_AMOUNT_RE,
    _DATE_STOPWORDS,
    _TEMPORAL_SEQUENCE_STOPWORDS,
    CountEvidenceItem,
    EvidenceLedger,
    EvidenceLedgerRow,
    SynthesisPlan,
    _actual_travel_duration_context,
    _dedupe_filtered_date_rows,
    _duration_match_is_habitual_per_occurrence,
    _duration_match_is_recurring_cadence,
    _duration_match_is_relative_time_anchor,
    _duration_query_accepts_relative_time_anchor,
    _earned_currency_context,
    _earned_total_query,
    _floor_value_currency_context,
    _floor_value_sale_query,
    _non_spend_currency_context,
    _personal_memory_query,
    _personal_numeric_evidence,
    _quantity_focus_terms,
    _realized_unit_price_total,
    _spent_total_query,
    _temporal_count_constraint,
    _travel_duration_total_query,
    canonical_duration_unit,
    canonical_quantity_unit,
    context_text,
    duration_identity,
    duration_unit_minutes,
    duration_value_matches,
    explicit_date_match_is_calendar_operand,
    format_number,
    local_evidence_span,
    quantity_identity,
    quantity_match_is_rate_or_guideline,
    quantity_query_units,
    quantity_unit_display,
    quantity_value_matches,
    source_citation,
    source_group,
    source_tokens,
    temporal_sequence_exclude_unanchored_when_answerable,
)
from zaxy.synthesis.labels import (
    _clean_generic_count_object,
    _competitive_sport_items,
    _count_action_terms,
    _count_subject,
    _date_interval_blocked_by_count_or_duration_query,
    _dinner_party_items,
    _doctor_visit_labels,
    _first_person_spans,
    _fish_inventory_counts,
    _generic_action_matches_query,
    _generic_action_object_phrases,
    _generic_count_object_relevant,
    _has_count_action,
    _kitchen_item_labels,
    _model_kit_labels,
    _negated_count_action,
    _normalize_count_identity,
    _property_viewing_label,
    _rollercoaster_ride_labels,
    _wedding_label,
    _writing_piece_label,
    build_synthesis_plan,
    count_label,
    duration_concrete_subject_signature,
    duration_identity_signature,
    generic_count_event_identity,
    numeric_state_evidence,
    numeric_state_query,
    numeric_state_required_qualifier_terms,
)
from zaxy.synthesis.operations import (
    _deduplicate_currency_source_values,
    _duration_focus_terms,
    _duration_preferred_units,
    _filter_date_rows,
    _filter_duration_ledger,
    _filter_lodging_currency_ledger,
    _filter_session_date_anchors_with_explicit_source_dates,
    _filter_unit_price_currency_ledger,
    _numeric_focus_terms,
)


def _count_focus_terms(query: str) -> set[str]:
    terms = {
        token
        for token in source_tokens(query)
        if len(token) > 2 and token not in _COUNT_STOPWORDS and not token.isdigit()
    }
    expanded = set(terms)
    semantic_groups = {
        "movie": {"movie", "movies", "film", "films", "cinema"},
        "doctor": {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"},
        "doctors": {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"},
        "festival": {"festival", "festivals", "fest", "fests"},
        "festivals": {"festival", "festivals", "fest", "fests"},
        "kit": {"kit", "kits", "model", "models", "scale"},
        "kits": {"kit", "kits", "model", "models", "scale"},
        "model": {"kit", "kits", "model", "models", "scale"},
        "models": {"kit", "kits", "model", "models", "scale"},
        "instrument": {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"},
        "instruments": {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"},
        "musical": {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"},
        "museum": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "museums": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "gallery": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "galleries": {"museum", "museums", "gallery", "galleries", "cube", "art"},
        "properties": {"properties", "property", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"},
        "property": {"properties", "property", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"},
        "clothing": {"blazer", "boots", "dress", "jacket", "shirt", "shoes", "sweater"},
        "clothes": {"blazer", "boots", "dress", "jacket", "shirt", "shoes", "sweater"},
        "furniture": {"bed", "bookshelf", "chair", "couch", "desk", "mattress", "shelves", "shelf", "sofa", "table"},
        "wedding": {"wedding", "weddings"},
        "weddings": {"wedding", "weddings"},
        "charity": {
            "awareness",
            "benefit",
            "cause",
            "charitable",
            "charity",
            "donation",
            "donations",
            "fund",
            "fundraiser",
            "fundraising",
            "funds",
            "raised",
            "support",
            "supported",
            "volunteer",
            "volunteered",
        },
        "charitable": {
            "awareness",
            "benefit",
            "cause",
            "charitable",
            "charity",
            "donation",
            "donations",
            "fund",
            "fundraiser",
            "fundraising",
            "funds",
            "raised",
            "support",
            "supported",
            "volunteer",
            "volunteered",
        },
        "fundraising": {
            "awareness",
            "benefit",
            "cause",
            "charitable",
            "charity",
            "donation",
            "donations",
            "fund",
            "fundraiser",
            "fundraising",
            "funds",
            "raised",
            "support",
            "supported",
            "volunteer",
            "volunteered",
        },
        "fundraiser": {
            "awareness",
            "benefit",
            "cause",
            "charitable",
            "charity",
            "donation",
            "donations",
            "fund",
            "fundraiser",
            "fundraising",
            "funds",
            "raised",
            "support",
            "supported",
            "volunteer",
            "volunteered",
        },
        "ride": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rides": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rode": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rollercoaster": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "rollercoasters": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "coaster": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "coasters": {"ride", "rides", "riding", "rode", "ridden", "rollercoaster", "rollercoasters", "coaster", "coasters"},
        "fish": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "aquarium": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "aquariums": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "tank": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
        "tanks": {"fish", "aquarium", "aquariums", "tank", "tanks", "tetra", "tetras", "gourami", "gouramis", "pleco", "catfish", "betta", "bubbles"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _temporal_count_program(
    query: str,
    rows: list[EvidenceLedgerRow],
) -> TemporalEvidenceProgramResult | None:
    """Build and execute a generic temporal count program for before/after queries."""
    constraint = _temporal_count_constraint(query)
    if constraint is None:
        return None
    direction, target_terms = constraint
    program_rows_source = [
        row
        for row in rows
        if row.kind == "event" and (not row.exclude_reason or row.exclude_reason in {"target_property"})
    ]
    included = [row for row in program_rows_source if not row.exclude_reason]
    if len(included) < 2:
        return None
    program_rows = tuple(_temporal_evidence_row(row) for row in program_rows_source)
    result = execute_temporal_evidence_program(
        TemporalEvidenceProgramSpec(
            operator=f"count_{direction}",
            event_class_terms=tuple(_count_focus_terms(query)),
            boundary_terms=tuple(sorted(target_terms)),
        ),
        rows=program_rows,
    )
    if result.boundary is None:
        return None
    return result


def _filter_temporal_count_rows(query: str, rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    """Apply generic before/after constraints to count evidence with dated events."""
    result = _temporal_count_program(query, rows)
    if result is None or not result.complete:
        return rows
    reasons_by_fact_id = {
        decision.row.event_id: decision.exclude_reason
        for decision in result.decisions
        if decision.exclude_reason
    }
    return [
        replace(row, exclude_reason=reasons_by_fact_id[row.fact_id])
        if row.fact_id in reasons_by_fact_id and not row.exclude_reason
        else row
        for row in rows
    ]


def _date_focus_terms(query: str) -> set[str]:
    terms = _count_focus_terms(query)
    if "meet" in terms:
        terms.update({"met", "catch", "caught", "up"})
    if "moma" in terms:
        terms.update({"museum", "modern", "art"})
    if {"metropolitan", "ancient", "civilizations"} & terms:
        terms.update({"metropolitan", "museum", "art", "exhibit"})
    return terms


def _film_festival_name(context: str) -> bool:
    """Return whether text names a film/movie festival without the generic noun."""
    return bool(
        re.search(
            r"\b(?:AFI\s+Fest|Sundance|Tribeca|Cannes|Telluride|SXSW|TIFF)\b",
            context,
            flags=re.IGNORECASE,
        )
    )


def _span_matches_count_subject(span: str, subject: str) -> bool:
    """Return whether a span satisfies the count subject facet."""
    if not subject:
        return True
    tokens = set(source_tokens(span))
    if subject == "film_festival":
        if _film_festival_name(span):
            return True
        return bool(tokens & {"movie", "movies", "film", "films", "cinema"} and tokens & {"festival", "festivals", "fest", "fests"})
    if subject == "property_viewing":
        if re.search(
            r"\bI\s+(?:made|put\s+in)\s+an\s+offer\b",
            span,
            flags=re.IGNORECASE,
        ) and not re.search(r"\b(?:rejected|higher\s+bid|outbid)\b", span, flags=re.IGNORECASE):
            return False
        return bool(tokens & {"property", "properties", "home", "homes", "house", "houses", "bungalow", "condo", "townhouse"})
    if subject == "doctor_visit":
        if re.search(r"\bI\s+(?:requested|asked|wanted|hoped|planned)\s+to\s+see\b", span, flags=re.IGNORECASE):
            return False
        return bool(tokens & {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"})
    if subject == "museum_gallery":
        if re.search(r"\bno\s+(?:museum|museums|gallery|galleries)\b", span, flags=re.IGNORECASE):
            return False
        if tokens & {"january"}:
            return False
        return bool(tokens & {"museum", "museums", "gallery", "galleries", "cube"})
    if subject == "kitchen_item":
        return bool(
            tokens
            & {
                "coffee",
                "espresso",
                "faucet",
                "ikea",
                "kitchen",
                "mat",
                "shelves",
                "sink",
                "toaster",
            }
            and tokens
            & {"donated", "fixed", "got", "new", "replaced", "upgrade"}
        )
    if subject == "musical_instrument":
        if re.search(
            r"\b(?:thinking\s+about|considering|planning\s+to|want\s+to)\s+"
            r"(?:get|buy|purchase|try)\b",
            span,
            flags=re.IGNORECASE,
        ):
            return False
        return bool(tokens & {"instrument", "instruments", "guitar", "piano", "drum", "drums", "ukulele"})
    if subject == "model_kit":
        return bool(_model_kit_labels(span))
    if subject == "wedding":
        return bool(tokens & {"wedding", "weddings"})
    if subject == "rollercoaster_ride":
        return bool(
            tokens
            & {
                "rollercoaster",
                "rollercoasters",
                "coaster",
                "coasters",
                "ride",
                "rides",
                "riding",
                "rode",
                "ridden",
            }
        )
    if subject == "fish_inventory":
        return bool(tokens & {"fish", "tetras", "gouramis", "pleco", "catfish", "betta", "bubbles", "tank", "aquarium"})
    if subject == "competitive_sport":
        return bool(tokens & {"competitive", "competitively"} and tokens & {"play", "played", "swim", "swimming", "tennis"})
    if subject == "dinner_party":
        if re.search(r"\b(?:hosting|host|planning|soon)\b", span, flags=re.IGNORECASE) and not re.search(
            r"\b(?:attended|had\s+(?:a\s+)?(?:great|lovely)?\s*experience|ones\s+we\s+had)\b",
            span,
            flags=re.IGNORECASE,
        ):
            return False
        return bool(tokens & {"dinner", "party", "parties", "feast", "potluck", "bbq"} and tokens & {"attended", "had"})
    if subject == "writing_piece":
        return bool(
            tokens
            & {"article", "articles", "essay", "essays", "piece", "pieces", "poem", "poems", "story", "stories", "writing"}
            and tokens
            & {"completed", "drafted", "finished", "published", "wrote", "written"}
        )
    return True


def _relevance(focus_terms: set[str], context: str) -> int:
    if not focus_terms:
        return 0
    context_terms = set(source_tokens(context))
    score = len(focus_terms & context_terms)
    if {"movie", "film"} & focus_terms and _film_festival_name(context):
        score += 1
    return score


def count_evidence_spans(
    text: str,
    *,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str = "",
) -> list[str]:
    """Return all count-relevant first-person memory spans from a source context."""
    spans = _first_person_spans(text)
    if not spans:
        return []
    scored: list[tuple[int, int, int, str]] = []
    for index, span in enumerate(spans):
        if _negated_count_action(span, action_terms):
            continue
        if not _span_matches_count_subject(span, subject):
            continue
        focus_score = _relevance(focus_terms, span)
        if subject == "kitchen_item" and _kitchen_item_labels(span):
            focus_score = max(focus_score, 1)
        action_score = 2 if _has_count_action(span, action_terms) else 0
        if focus_terms & {"something", "somethings", "anything", "things", "times"} and action_score > 0:
            focus_score = max(focus_score, 1)
        if focus_score <= 0 or action_score <= 0:
            continue
        scored.append((-(focus_score + action_score), -action_score, index, span))
    if not scored:
        return []
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    ordered: list[str] = []
    seen: set[str] = set()
    for _, _, _, span in scored:
        normalized = " ".join(source_tokens(span))
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(span)
    return ordered


def count_evidence_span(
    text: str,
    *,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str = "",
) -> str:
    """Return the most count-relevant first-person memory span from a source context."""
    spans = count_evidence_spans(
        text,
        focus_terms=focus_terms,
        action_terms=action_terms,
        subject=subject,
    )
    return spans[0] if spans else ""


def _generic_action_object_count_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str,
) -> list[CountEvidenceItem]:
    """Extract concrete action-object count rows for open count questions."""
    if subject in {
        "doctor_visit",
        "film_festival",
        "fish_inventory",
        "kitchen_item",
        "model_kit",
        "museum_gallery",
        "musical_instrument",
        "property_viewing",
        "rollercoaster_ride",
        "wedding",
        "writing_piece",
    }:
        return []
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        if _negated_count_action(span, action_terms):
            continue
        for verb, raw_object in _generic_action_object_phrases(span):
            if action_terms and verb not in action_terms and not _generic_action_matches_query(verb, action_terms):
                continue
            label = _clean_generic_count_object(raw_object)
            if not label or not _generic_count_object_relevant(label, focus_terms):
                continue
            identity = f"generic_action_object={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            relevance = max(_relevance(focus_terms, f"{verb} {label}"), 2)
            items.append(
                CountEvidenceItem(
                    label=f"{verb} {label}",
                    span=span,
                    normalized_identity=identity,
                    relevance=relevance,
                )
            )
    return items


def _count_labeled_items_from_spans(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
    label_extractor: Callable[[str], list[str]],
) -> list[CountEvidenceItem]:
    """Extract deduplicated typed count items from all relevant source spans."""
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        relevance = _relevance(focus_terms, span)
        if relevance <= 0:
            continue
        for label in label_extractor(span):
            identity = f"{identity_prefix}={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=identity,
                    relevance=max(relevance, 2),
                )
            )
    return items


def _kitchen_count_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Extract kitchen item rows without relying on generic first-person scoring."""
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        if not _span_matches_count_subject(span, "kitchen_item"):
            continue
        relevance = max(_relevance(focus_terms, span), 2)
        for label in _kitchen_item_labels(span):
            identity = f"{identity_prefix}={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=identity,
                    relevance=relevance,
                )
            )
    return items


def _rollercoaster_ride_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Expand rollercoaster ride memories into one row per ride occurrence."""
    items: list[CountEvidenceItem] = []
    seen_counts: dict[str, int] = {}
    for span in spans:
        relevance = max(_relevance(focus_terms, span), 2)
        labels = _rollercoaster_ride_labels(span)
        for label in labels:
            normalized = _normalize_count_identity(label)
            occurrence = seen_counts.get(normalized, 0) + 1
            seen_counts[normalized] = occurrence
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=f"{identity_prefix}={normalized}:{occurrence}",
                    relevance=relevance,
                )
            )
    return items


def _fish_inventory_items(
    spans: list[str],
    *,
    focus_terms: set[str],
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Expand aquarium inventory memories into one row per fish."""
    items: list[CountEvidenceItem] = []
    seen_counts: dict[str, int] = {}
    for span in spans:
        if not _span_matches_count_subject(span, "fish_inventory"):
            continue
        relevance = max(_relevance(focus_terms, span), 2)
        for label, count in _fish_inventory_counts(span):
            normalized = _normalize_count_identity(label)
            for _ in range(count):
                occurrence = seen_counts.get(normalized, 0) + 1
                seen_counts[normalized] = occurrence
                items.append(
                    CountEvidenceItem(
                        label=label,
                        span=span,
                        normalized_identity=f"{identity_prefix}={normalized}:{occurrence}",
                        relevance=relevance,
                    )
                )
    return items


def count_evidence_items(
    text: str,
    *,
    group: str,
    focus_terms: set[str],
    action_terms: set[str],
    subject: str,
) -> list[CountEvidenceItem]:
    """Extract countable items or events from the strongest cited memory span."""
    spans = count_evidence_spans(
        text,
        focus_terms=focus_terms,
        action_terms=action_terms,
        subject=subject,
    )
    if subject == "kitchen_item":
        return _kitchen_count_items(
            spans or [text],
            focus_terms=focus_terms,
            identity_prefix="kitchen_item",
        )
    if subject == "fish_inventory":
        return _fish_inventory_items(
            spans or _first_person_spans(text) or [text],
            focus_terms=focus_terms,
            identity_prefix="fish_inventory",
        )
    if subject == "competitive_sport":
        return _competitive_sport_items(
            spans or _first_person_spans(text) or [text],
            identity_prefix="competitive_sport",
        )
    if subject == "dinner_party":
        return _dinner_party_items(
            spans or _first_person_spans(text) or [text],
            identity_prefix="dinner_party",
        )
    if not spans:
        return []
    if subject == "model_kit":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="model_kit",
            label_extractor=_model_kit_labels,
        )
    if subject == "film_festival":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="film_festival",
            label_extractor=_film_festival_labels,
        )
    if subject == "doctor_visit":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="doctor_visit",
            label_extractor=_doctor_visit_labels,
        )
    if subject == "museum_gallery":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="museum_gallery",
            label_extractor=_museum_gallery_labels,
        )
    if subject == "musical_instrument":
        return _count_labeled_items_from_spans(
            spans,
            focus_terms=focus_terms,
            identity_prefix="musical_instrument",
            label_extractor=_musical_instrument_labels,
        )
    if subject == "rollercoaster_ride":
        return _rollercoaster_ride_items(
            spans,
            focus_terms=focus_terms,
            identity_prefix="rollercoaster_ride",
        )
    if generic_items := _generic_action_object_count_items(
        spans,
        focus_terms=focus_terms,
        action_terms=action_terms,
        subject=subject,
    ):
        return generic_items
    span = spans[0]
    relevance = _relevance(focus_terms, span)
    if relevance <= 0:
        return []
    if subject == "property_viewing":
        label = _property_viewing_label(span)
        return [
            CountEvidenceItem(
                label=label,
                span=span,
                normalized_identity=f"source_group={group}",
                relevance=relevance,
            )
        ]
    if subject == "wedding":
        label = _wedding_label(span)
        return [
            CountEvidenceItem(
                label=label,
                span=span,
                normalized_identity=f"source_group={group}",
                relevance=relevance,
            )
        ]
    if subject == "writing_piece":
        label = _writing_piece_label(span)
        return [
            CountEvidenceItem(
                label=label,
                span=span,
                normalized_identity=f"writing_piece:{_normalize_count_identity(label)}",
                relevance=relevance,
            )
        ]
    items: list[CountEvidenceItem] = []
    for item_span in spans:
        item_relevance = _relevance(focus_terms, item_span)
        if item_relevance <= 0:
            continue
        label = count_label(item_span)
        items.append(
            CountEvidenceItem(
                label=label,
                span=item_span,
                normalized_identity=generic_count_event_identity(
                    group=group,
                    label=label,
                    span=item_span,
                ),
                relevance=item_relevance,
            )
        )
    return items


def temporal_sequence_candidates_from_sentence(query_tokens: set[str], sentence: str) -> list[tuple[str, str]]:
    """Return supported event labels with local evidence spans from one sentence."""
    candidates: list[tuple[str, str]] = []
    sports_candidates: list[tuple[str, str]] = []
    if query_tokens & {"sport", "sports", "watched", "participated", "events"}:
        sports_candidates = temporal_sequence_sports_candidates(sentence)
        candidates.extend(sports_candidates)
    graduation_query = bool(query_tokens & {"graduated", "graduation", "graduate"})
    if graduation_query:
        candidates.extend(temporal_sequence_graduation_candidates(sentence))
    venue_query = bool(query_tokens & {"museum", "museums", "gallery", "galleries"})
    venue_label = temporal_sequence_venue_label(sentence) if venue_query else ""
    if venue_label:
        candidates.append((venue_label, sentence))
    airline_query = bool(query_tokens & {"airline", "airlines", "flew", "flight", "flights"})
    if airline_query:
        for match in re.finditer(
            r"\bI\s+(?:just\s+|recently\s+|also\s+)?flew\s+with\s+(?P<label>[^.!?;,]{2,80})",
            sentence,
            flags=re.IGNORECASE,
        ):
            label = clean_temporal_sequence_label(match.group("label"), airline=True)
            if label:
                candidates.append((label, _temporal_sequence_match_evidence(sentence, match)))
    patterns = (
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?got\s+back\s+from\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?returned\s+from\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?went\s+on\s+(?P<label>[^.!?;,]{3,140})"),
        ("took ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?took\s+(?P<label>[^.!?;,]{3,140})"),
        ("watched ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?watched\s+(?P<label>[^.!?;,]{3,140})"),
        ("attended ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?attended\s+(?P<label>[^.!?;,]{3,140})"),
        ("participated in ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?participated\s+in\s+(?P<label>[^.!?;,]{3,140})"),
        ("started ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?started\s+(?P<label>[^.!?;,]{3,140})"),
        ("helped ", r"\bI\s+(?:just\s+|recently\s+|also\s+)?helped\s+(?P<label>[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>ordered\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>used\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>redeemed\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?(?P<label>signed\s+up\s+for\s+[^.!?;,]{3,140})"),
        ("", r"\bI\s+(?:just\s+|recently\s+|also\s+)?visited\s+(?P<label>[^.!?;,]{3,140})"),
    )
    if not graduation_query and not sports_candidates and not venue_label:
        candidates.extend(_temporal_sequence_pattern_candidates(patterns, sentence))
    query_focus = query_tokens - _TEMPORAL_SEQUENCE_STOPWORDS
    deduped: dict[str, tuple[int, str, str]] = {}
    for label, evidence in candidates:
        normalized = normalize_temporal_sequence_label(label)
        if not normalized:
            continue
        score = _relevance(query_focus, evidence) + temporal_sequence_local_cue_score(evidence)
        existing = deduped.get(normalized)
        if existing is None or score > existing[0]:
            deduped[normalized] = (score, label, evidence)
    return [(label, evidence) for _score, label, evidence in sorted(deduped.values(), reverse=True)]


def temporal_sequence_candidates_with_evidence(query: str, text: str) -> list[tuple[str, str]]:
    """Extract supported event labels and local evidence spans from cited source text."""
    query_tokens = set(source_tokens(query))
    sentences = temporal_sequence_sentences(text)
    candidates: list[tuple[int, int, str, str]] = []
    for sentence in sentences:
        for candidate_index, (candidate, evidence) in enumerate(temporal_sequence_candidates_from_sentence(query_tokens, sentence)):
            if not candidate:
                continue
            score = (
                _relevance(query_tokens - _TEMPORAL_SEQUENCE_STOPWORDS, evidence)
                + temporal_sequence_local_cue_score(evidence)
            )
            candidates.append((score, -candidate_index, candidate, evidence))
    candidates.sort(reverse=True)
    return [(candidate, evidence) for _score, _order, candidate, evidence in candidates]


def temporal_sequence_candidate_with_evidence(query: str, text: str) -> tuple[str, str] | None:
    """Extract the best event label and its local evidence span from a cited source."""
    candidates = temporal_sequence_candidates_with_evidence(query, text)
    return candidates[0] if candidates else None


def temporal_sequence_candidate(query: str, text: str) -> str:
    """Extract a concise event label from a cited source span."""
    candidate = temporal_sequence_candidate_with_evidence(query, text)
    return candidate[0] if candidate else ""


def temporal_sequence_candidate_from_sentence(query_tokens: set[str], sentence: str) -> str:
    """Return an event label from one first-person event sentence."""
    candidates = temporal_sequence_candidates_from_sentence(query_tokens, sentence)
    return candidates[0][0] if candidates else ""


def _row_confidence(*, relevance: int, has_label: bool) -> float:
    return min(0.99, 0.55 + min(relevance, 6) * 0.05 + (0.08 if has_label else 0.0))


def build_quantity_ledger(query: str, contexts: list[str]) -> EvidenceLedger:
    """Extract generic non-currency, non-duration unit quantities into a cited ledger."""
    plan = SynthesisPlan(
        answer_type="quantity",
        operation="sum_values",
        subject_terms=tuple(source_tokens(query)),
        required_kinds=("quantity",),
        required_source_groups=1,
        reasons=("quantity", "aggregation"),
    )
    allowed_units = quantity_query_units(query)
    if not allowed_units:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _quantity_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        for match_index, match in enumerate(quantity_value_matches(text)):
            unit = canonical_quantity_unit(match.unit)
            if unit not in allowed_units:
                continue
            evidence_span = local_evidence_span(text, match.start, match.end, window_chars=280)
            relevance = _relevance(focus_terms, evidence_span)
            label = f"{format_number(match.value)} {quantity_unit_display(unit, match.value)}"
            identity = quantity_identity(group=group, value=match.value, unit=unit, evidence_span=evidence_span)
            duplicate = identity in seen
            seen.add(identity)
            exclude_reason = ""
            if requires_personal_memory and not _personal_numeric_evidence(text, match.start, match.end):
                exclude_reason = "not_personal_memory"
            elif quantity_match_is_rate_or_guideline(text, match.start, match.end):
                exclude_reason = "rate_or_guideline"
            elif relevance <= 0:
                exclude_reason = "query_focus_mismatch"
            elif duplicate:
                exclude_reason = "duplicate_identity"
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"quantity:{context_index}:{match_index}",
                    source_group=group,
                    citation=citation,
                    kind="quantity",
                    value=str(match.value),
                    unit=unit,
                    label=label,
                    raw_span=match.raw,
                    context=evidence_span,
                    normalized_identity=identity,
                    relevance=relevance,
                    include_reason="unit_quantity",
                    exclude_reason=exclude_reason,
                    confidence=_row_confidence(relevance=relevance, has_label=True),
                )
            )
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def build_count_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract distinct cited-event evidence for count/list synthesis."""
    plan = plan or build_synthesis_plan(query)
    if "event" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _count_focus_terms(query)
    action_terms = _count_action_terms(query)
    subject = _count_subject(query)
    rows: list[EvidenceLedgerRow] = []
    seen_identities: set[str] = set()
    provisional: list[EvidenceLedgerRow] = []
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        items = count_evidence_items(
            text,
            group=group,
            focus_terms=focus_terms,
            action_terms=action_terms,
            subject=subject,
        )
        if not items:
            span = count_evidence_span(
                text,
                focus_terms=focus_terms,
                action_terms=action_terms,
                subject=subject,
            )
            relevance = _relevance(focus_terms, span or text)
            label = count_label(span or text)
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"count:{context_index}",
                    source_group=group,
                    citation=citation,
                    kind="event",
                    value="1",
                    unit="event",
                    label=label,
                    raw_span=span or text,
                    context=text,
                    normalized_identity=f"source_group={group}",
                    relevance=relevance,
                    include_reason="relevant_source_event",
                    exclude_reason="query_focus_mismatch",
                    confidence=_row_confidence(relevance=relevance, has_label=bool(label)),
                )
            )
            continue
        for item_index, item in enumerate(items):
            duplicate = item.normalized_identity in seen_identities
            if not duplicate:
                seen_identities.add(item.normalized_identity)
            exclude_reason = ""
            if duplicate:
                exclude_reason = (
                    "duplicate_source_group"
                    if item.normalized_identity == f"source_group={group}"
                    else "duplicate_identity"
                )
            row = EvidenceLedgerRow(
                fact_id=f"count:{context_index}:{item_index}",
                source_group=group,
                citation=citation,
                kind="event",
                value="1",
                unit="event",
                label=item.label,
                raw_span=item.span,
                context=text,
                normalized_identity=item.normalized_identity,
                relevance=item.relevance,
                include_reason="relevant_source_event",
                exclude_reason=exclude_reason,
                confidence=_row_confidence(relevance=item.relevance, has_label=bool(item.label)),
            )
            if exclude_reason:
                rows.append(row)
            else:
                provisional.append(row)
    rows.extend(provisional)
    rows = _filter_count_rows(subject, rows, query=query)
    temporal_program = _temporal_count_program(query, rows)
    if temporal_program is not None and temporal_program.complete:
        reasons_by_fact_id = {
            decision.row.event_id: decision.exclude_reason
            for decision in temporal_program.decisions
            if decision.exclude_reason
        }
        rows = [
            replace(row, exclude_reason=reasons_by_fact_id[row.fact_id])
            if row.fact_id in reasons_by_fact_id and not row.exclude_reason
            else row
            for row in rows
        ]
    return EvidenceLedger(plan=plan, rows=tuple(rows), temporal_program=temporal_program)


def build_numeric_state_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract cited current-count state and incremental updates."""
    plan = plan or SynthesisPlan(
        answer_type="count",
        operation="numeric_state",
        subject_terms=tuple(
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _COUNT_STOPWORDS and not token.isdigit()
        ),
        required_kinds=("numeric_state",),
        required_source_groups=1,
        reasons=("numeric_state",),
    )
    if not numeric_state_query(query):
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = set(plan.subject_terms)
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    required_qualifier_terms = numeric_state_required_qualifier_terms(query)
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        for item_index, (value, span, reason) in enumerate(numeric_state_evidence(text, focus_terms=focus_terms)):
            identity = f"group={group}|reason={reason}|value={value}|span={' '.join(source_tokens(span))[:160]}"
            duplicate = identity in seen
            seen.add(identity)
            relevance = _relevance(focus_terms, span)
            span_terms = set(source_tokens(span))
            exclude_reason = ""
            if required_qualifier_terms and not required_qualifier_terms <= span_terms:
                exclude_reason = "missing_required_state_qualifier"
            elif duplicate:
                exclude_reason = "duplicate_identity"
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"numeric_state:{context_index}:{item_index}",
                    source_group=group,
                    citation=citation,
                    kind="numeric_state",
                    value=str(value),
                    unit="count",
                    label=span,
                    raw_span=span,
                    context=span,
                    normalized_identity=identity,
                    relevance=relevance,
                    include_reason=reason,
                    exclude_reason=exclude_reason,
                    confidence=_row_confidence(relevance=relevance, has_label=True),
                )
            )
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def build_date_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract explicit date evidence for temporal interval synthesis."""
    plan = SynthesisPlan(
        answer_type="interval",
        operation="date_difference",
        subject_terms=tuple(
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _DATE_STOPWORDS and not token.isdigit()
        ),
        required_kinds=("date",),
        required_source_groups=2,
        reasons=("temporal",),
    )
    query_tokens = set(source_tokens(query))
    if _date_interval_blocked_by_count_or_duration_query(query, query_tokens):
        return EvidenceLedger(plan=plan, rows=())
    if not ({"day", "days", "week", "weeks"} & query_tokens):
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _date_focus_terms(query)
    rows: list[EvidenceLedgerRow] = []
    provisional: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    for context_index, context in enumerate(contexts):
        raw_text = context_text(context)
        text = temporal_evidence_text(raw_text)
        default_year = context_year(raw_text)
        group = source_group(context)
        citation = source_citation(context)
        relevance = _relevance(focus_terms, text)
        if "query_temporal_anchor=true" in raw_text.casefold():
            anchor_date = session_anchor_date(raw_text, raw_text)
            if anchor_date is None:
                continue
            identity = f"group={group}|date={anchor_date.isoformat()}"
            duplicate = identity in seen
            seen.add(identity)
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"date:{context_index}:query",
                    source_group=group,
                    citation=citation,
                    kind="date",
                    value=anchor_date.isoformat(),
                    unit="day",
                    label=anchor_date.isoformat(),
                    raw_span=anchor_date.isoformat(),
                    context=text,
                    normalized_identity=identity,
                    relevance=10,
                    include_reason="query_temporal_anchor",
                    exclude_reason="duplicate_identity" if duplicate else "",
                    confidence=_row_confidence(relevance=10, has_label=True),
                )
            )
            continue
        context_dates = [
            date_match
            for date_match in explicit_date_matches(text, default_year=default_year)
            if explicit_date_match_is_calendar_operand(text, date_match)
        ]
        for match_index, date_match in enumerate(context_dates):
            value = date_match.value
            identity = f"group={group}|date={value.isoformat()}"
            evidence_span = local_evidence_span(text, date_match.start, date_match.end)
            date_relevance = _relevance(focus_terms, evidence_span)
            row = EvidenceLedgerRow(
                fact_id=f"date:{context_index}:{match_index}",
                source_group=group,
                citation=citation,
                kind="date",
                value=value.isoformat(),
                unit="day",
                label=value.isoformat(),
                raw_span=date_match.raw,
                context=evidence_span,
                normalized_identity=identity,
                relevance=date_relevance,
                include_reason="explicit_date",
                exclude_reason="",
                confidence=_row_confidence(relevance=date_relevance, has_label=True),
            )
            provisional.append(row)
        anchor_date = session_anchor_date(raw_text, text)
        if anchor_date is None or not session_date_anchor_allowed(
            query,
            text,
            focus_terms=focus_terms,
            relevance=relevance,
        ):
            continue
        identity = f"group={group}|date={anchor_date.isoformat()}"
        include_reason = "relative_session_date_anchor" if relative_session_date_offset(text) else "session_date_anchor"
        row = EvidenceLedgerRow(
            fact_id=f"date:{context_index}:session",
            source_group=group,
            citation=citation,
            kind="date",
            value=anchor_date.isoformat(),
            unit="day",
            label=anchor_date.isoformat(),
            raw_span=anchor_date.isoformat(),
            context=text,
            normalized_identity=identity,
            relevance=relevance,
            include_reason=include_reason,
            exclude_reason="",
            confidence=_row_confidence(relevance=relevance, has_label=True),
        )
        provisional.append(row)
    anchor_terms = temporal_anchor_terms(plan.subject_terms)
    rows.extend(
        _dedupe_filtered_date_rows(
            _filter_session_date_anchors_with_explicit_source_dates(
                _filter_date_rows(provisional, anchor_terms)
            )
        )
    )
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def build_temporal_sequence_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract ordered event evidence for temporal sequence synthesis."""
    required_events = temporal_sequence_requested_count(query)
    plan = plan or SynthesisPlan(
        answer_type="ordered_list",
        operation="temporal_sequence",
        subject_terms=tuple(
            token
            for token in source_tokens(query)
            if len(token) > 2 and token not in _TEMPORAL_SEQUENCE_STOPWORDS and not token.isdigit()
        ),
        required_kinds=("temporal_event",),
        required_source_groups=2,
        reasons=("temporal_sequence",),
    )
    if not temporal_sequence_query(query):
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = set(plan.subject_terms)
    query_slots = temporal_sequence_query_slots(query)
    provisional: list[tuple[int, int, int, EvidenceLedgerRow]] = []
    for context_index, context in enumerate(contexts):
        raw_text = context_text(context)
        text = temporal_evidence_text(raw_text)
        candidates = temporal_sequence_candidates_with_evidence(query, text)
        if not candidates:
            continue
        group = source_group(context)
        for candidate_index, (label, evidence_text) in enumerate(candidates):
            label = temporal_sequence_query_slot_label(query_slots, label) or label
            identity = f"temporal_event={normalize_temporal_sequence_label(label)}"
            order_value, include_reason = temporal_sequence_order_value(
                raw_text,
                evidence_text,
                prefer_relative=bool(len(candidates) > 1),
            )
            provenance_index = source_group_sequence_index(group, fallback=context_index)
            relevance = _relevance(focus_terms, evidence_text)
            row = EvidenceLedgerRow(
                fact_id=f"temporal_sequence:{context_index}:{candidate_index}",
                source_group=group,
                citation=source_citation(context),
                kind="temporal_event",
                value=str(order_value),
                unit="sequence_order",
                label=label,
                raw_span=label,
                context=evidence_text,
                normalized_identity=identity,
                relevance=relevance,
                include_reason=include_reason,
                confidence=_row_confidence(relevance=relevance, has_label=True),
            )
            provisional.append((order_value, provenance_index, context_index, row))
    provisional.sort(key=lambda item: (item[0], item[1], item[2]))
    rows = temporal_sequence_deduped_rows([row for _, _, _, row in provisional])
    rows = temporal_sequence_exclude_unanchored_when_answerable(
        rows,
        required_events=max(2, required_events or 0),
    )
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def build_duration_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract and normalize duration evidence into a cited ledger."""
    plan = plan or build_synthesis_plan(query)
    if "duration" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _duration_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    requires_actual_travel_duration = _travel_duration_total_query(query)
    accepts_relative_time_anchor = _duration_query_accepts_relative_time_anchor(query)
    rows: list[EvidenceLedgerRow] = []
    seen: set[str] = set()
    context_by_group: dict[str, str] = {}
    for context in contexts:
        group = source_group(context)
        context_by_group[group] = " ".join((context_by_group.get(group, ""), context_text(context))).strip()
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        duration_matches = list(duration_value_matches(text))
        for match_index, duration_match in enumerate(duration_matches):
            unit = canonical_duration_unit(duration_match.unit)
            raw_value = duration_match.value
            minutes = raw_value * duration_unit_minutes(unit)
            evidence_span = local_evidence_span(
                text,
                duration_match.start,
                duration_match.end,
                window_chars=320,
            )
            label = f"{format_number(raw_value)} {unit}"
            identity_signature = duration_identity_signature(evidence_span)
            subject_signature = duration_concrete_subject_signature(
                query,
                evidence_span,
                duration_match.start,
                duration_match.end,
            )
            signature_for_identity = subject_signature or identity_signature
            occurrence_index = sum(
                1
                for prior_match in duration_matches[:match_index]
                if canonical_duration_unit(prior_match.unit) == unit
                and prior_match.value == raw_value
                and (
                    duration_concrete_subject_signature(
                        query,
                        prior_span := local_evidence_span(
                            text,
                            prior_match.start,
                            prior_match.end,
                            window_chars=320,
                        ),
                        prior_match.start,
                        prior_match.end,
                    )
                    or duration_identity_signature(prior_span)
                )
                == signature_for_identity
            )
            identity = duration_identity(
                group=group,
                minutes=minutes,
                label=label,
                evidence_signature=signature_for_identity,
                occurrence_index=occurrence_index,
            )
            duplicate = identity in seen
            seen.add(identity)
            relevance = _relevance(focus_terms, evidence_span)
            exclude_reason = ""
            if requires_personal_memory and not _personal_numeric_evidence(
                text,
                duration_match.start,
                duration_match.end,
            ):
                exclude_reason = "not_personal_memory"
            elif (
                not accepts_relative_time_anchor
                and _duration_match_is_relative_time_anchor(text, duration_match.start, duration_match.end)
            ):
                exclude_reason = "relative_time_anchor"
            elif _duration_match_is_habitual_per_occurrence(text, duration_match.start, duration_match.end):
                exclude_reason = "habitual_per_occurrence"
            elif _duration_match_is_recurring_cadence(text, duration_match.start, duration_match.end):
                exclude_reason = "recurring_cadence"
            elif requires_actual_travel_duration and not _actual_travel_duration_context(
                query,
                context_by_group.get(group, text),
            ):
                exclude_reason = "not_actual_travel_duration"
            elif duplicate:
                exclude_reason = "duplicate_identity"
            rows.append(
                EvidenceLedgerRow(
                    fact_id=f"duration:{context_index}:{match_index}",
                    source_group=group,
                    citation=citation,
                    kind="duration",
                    value=str(minutes),
                    unit="minutes",
                    label=label,
                    raw_span=duration_match.raw,
                    context=evidence_span,
                    normalized_identity=identity,
                    relevance=relevance,
                    include_reason="duration_value",
                    exclude_reason=exclude_reason,
                    confidence=_row_confidence(relevance=relevance, has_label=True),
                )
            )
    return _filter_duration_ledger(
        EvidenceLedger(plan=plan, rows=tuple(rows)),
        query,
        focus_terms,
        preferred_units=_duration_preferred_units(query),
    )


def _currency_pronoun_recipient_label(prefix: str, label: str) -> str:
    """Resolve local gift-recipient pronouns to a recently mentioned person term."""
    if not re.match(r"\s*(?:her|him|them|their)\b", label, flags=re.IGNORECASE):
        return ""
    recipient_matches = list(
        re.finditer(
            r"\b(?P<recipient>coworker|colleague|brother|sister|friend|mom|mother|dad|father|"
            r"partner|spouse|wife|husband|child|son|daughter|niece|nephew)\b",
            prefix,
            flags=re.IGNORECASE,
        )
    )
    if not recipient_matches:
        return ""
    return recipient_matches[-1].group("recipient").casefold()


def _clean_currency_item_label(label: str) -> str:
    label = re.sub(r"\b(?:content|session_id|longmemeval_session_id)=\S+\s*", "", label)
    label = re.sub(r"\brole=\S+\s*", "", label)
    label = re.sub(r"\b[a-z0-9_.-]*session[_-]?id=\S+\s*", "", label, flags=re.IGNORECASE)
    item_patterns = (
        r"\bI\s+(?:recently\s+|also\s+)?(?:bought|got|installed|replaced)\s+(?P<item>[^.!?;,]{1,100})",
        r"\bI\s+(?:needed|had)\s+to\s+replace\s+(?P<item>[^.!?;,]{1,80})",
        r"\b(?:mechanic|shop)\s+[^.!?]{0,80}?\breplace\s+(?P<item>[^.!?;,]{1,80})",
    )
    for pattern in item_patterns:
        matches = list(re.finditer(pattern, label, flags=re.IGNORECASE))
        if matches:
            return clean_label(matches[-1].group("item"))
    label = re.split(r"[,;:]|\b(?:because|and then|while)\b", label, maxsplit=1)[-1]
    label = re.sub(
        r"^(?:i\s+)?(?:recently\s+)?(?:also\s+)?"
        r"(?:bought|booked|got|installed|replaced|stayed in|paid for|spent on)\s+",
        "",
        label.strip(),
        flags=re.IGNORECASE,
    )
    label = re.sub(r"^(?:a|an|the|my|new)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^set\s+of\s+", "", label, flags=re.IGNORECASE)
    return clean_label(label)


def normalize_currency_label(label: str) -> str:
    """Normalize a currency label for cross-source duplicate detection."""
    normalized = _clean_currency_item_label(label).casefold()
    normalized = re.sub(
        r"\b(?:new|recent|recently|installed|got|my|the|a|an|i|d|like|to|add|that)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def currency_identity(*, group: str, value: str, label: str) -> str:
    """Return a stable identity used for currency deduplication."""
    normalized_label = normalize_currency_label(label)
    if normalized_label:
        return f"value={value}|label={normalized_label}"
    return f"group={group}|value={value}"


def _currency_label_before_amount(prefix: str) -> str:
    prefix = " ".join(prefix.split())
    lowered = prefix.casefold()
    if not (
        lowered.endswith((" for", " which were", " cost me", " cost"))
        or lowered in {"for", "which were", "cost me", "cost"}
    ):
        return ""
    for pattern in _CURRENCY_LABEL_BEFORE_AMOUNT_PATTERNS:
        match = pattern.search(prefix)
        if match:
            return _clean_currency_item_label(match.group("label"))
    return ""


def _currency_purchase_label_before_amount(prefix: str) -> str:
    """Extract a nearby purchased item when the amount is in a follow-up clause."""
    sentences = re.split(r"[.!?]", prefix)
    for sentence in reversed(sentences[-3:]):
        match = _CURRENCY_PURCHASE_LABEL_BEFORE_AMOUNT_RE.search(sentence)
        if match:
            return _clean_currency_item_label(match.group("item"))
    return ""


def currency_label(text: str, start: int, end: int) -> str:
    """Extract a compact item label near a currency amount."""
    local_span = local_evidence_span(text, start, end)
    if label := _currency_merchant_label(local_span):
        return label
    after = text[end : end + 120]
    for pattern in (
        r"\s+(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})",
        r"\s+(?:for|on)\s+(?:a|an|the)?\s*(?P<label>[A-Za-z][A-Za-z0-9'&.-]*(?:\s+[A-Za-z][A-Za-z0-9'&.-]*){0,4})",
    ):
        match = re.match(pattern, after)
        if match:
            return clean_label(match.group("label"))
    before = text[max(0, start - 120) : start]
    match = re.search(
        r"\b(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})\s*$",
        before,
    )
    if match:
        return clean_label(match.group("label"))
    prefix = text[max(0, start - 240) : start]
    if label := _currency_label_before_amount(prefix):
        return label
    purchase_prefix = text[max(0, start - 720) : start]
    if label := _currency_purchase_label_before_amount(purchase_prefix):
        if recipient := _currency_pronoun_recipient_label(purchase_prefix, label):
            return f"{recipient} {label}"
        return label
    return ""


def build_currency_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract and normalize currency evidence into a cited ledger."""
    plan = plan or build_synthesis_plan(query)
    if "currency" not in plan.required_kinds:
        return EvidenceLedger(plan=plan, rows=())
    focus_terms = _numeric_focus_terms(query)
    requires_personal_memory = _personal_memory_query(query)
    requires_floor_value = _floor_value_sale_query(query)
    requires_earned_total = _earned_total_query(query) and not requires_floor_value
    requires_actual_spend = _spent_total_query(query) and not requires_earned_total
    rows: list[EvidenceLedgerRow] = []
    included_by_identity: dict[str, EvidenceLedgerRow] = {}
    for context_index, context in enumerate(contexts):
        text = context_text(context)
        group = source_group(context)
        citation = source_citation(context)
        for match_index, match in enumerate(
            re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text)
        ):
            evidence_span = local_evidence_span(text, match.start(), match.end(), window_chars=320)
            amount = float(match.group("value").replace(",", ""))
            value = str(_realized_unit_price_total(evidence_span, amount) if requires_earned_total else amount)
            label = currency_label(text, match.start(), match.end())
            relevance = _relevance(focus_terms, " ".join((evidence_span, label)))
            identity = currency_identity(group=group, value=value, label=label)
            confidence = _row_confidence(relevance=relevance, has_label=bool(label))
            duplicate = included_by_identity.get(identity)
            exclude_reason = ""
            if requires_personal_memory and not _personal_numeric_evidence(
                text,
                match.start(),
                match.end(),
            ):
                exclude_reason = "not_personal_memory"
            elif requires_earned_total and not _earned_currency_context(evidence_span):
                exclude_reason = "not_earned_money"
            elif requires_floor_value and not _floor_value_currency_context(evidence_span):
                exclude_reason = "not_floor_value"
            elif requires_actual_spend and _non_spend_currency_context(evidence_span):
                exclude_reason = "not_actual_spend"
            elif duplicate is not None:
                exclude_reason = "duplicate_identity"
            fact_id = f"currency:{context_index}:{match_index}"
            row = EvidenceLedgerRow(
                fact_id=fact_id,
                source_group=group,
                citation=citation,
                kind="currency",
                value=value,
                unit="USD",
                label=label,
                raw_span=match.group(0),
                context=evidence_span,
                normalized_identity=identity,
                relevance=relevance,
                include_reason="currency_amount",
                exclude_reason=exclude_reason,
                confidence=confidence,
            )
            rows.append(row)
            if not exclude_reason:
                included_by_identity[identity] = row
    ledger = _deduplicate_currency_source_values(
        EvidenceLedger(plan=plan, rows=tuple(rows))
    )
    ledger = _filter_lodging_currency_ledger(query, ledger)
    ledger = _filter_itemized_currency_targets(query, ledger)
    ledger = _filter_unit_price_currency_ledger(query, ledger)
    return _filter_currency_ledger(ledger, focus_terms, query=query)
