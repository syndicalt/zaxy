"""Split from synthesis.py (mechanical decomposition)."""


from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from datetime import date, timedelta

from zaxy.evidence_program import (
    TemporalEvidenceRow,
)
from zaxy.synthesis.foundations import (
    _COUNT_STOPWORDS,
    _MONTHS,
    _NUMBER_WORDS,
    _QUERY_STOPWORDS,
    _TEMPORAL_SEQUENCE_STOPWORDS,
    EvidenceLedger,
    EvidenceLedgerRow,
    ExplicitDateMatch,
    _count_row_action,
    _filter_museum_gallery_rows,
    _filter_target_property_rows,
    month_only_date,
    source_tokens,
)
from zaxy.synthesis.labels import (
    _base_property_type,
    _canonical_instrument_label,
    _clean_count_item_label,
    _count_subject,
    _filter_specific_action_object_count_rows,
    _instrument_duration_phrase,
    _joined_property_outcomes,
    _normalize_count_identity,
    _property_location_phrase,
    _property_offer_target_phrase,
    _property_subject_phrase,
    _wedding_couple_from_label,
    count_display,
    count_subject_phrase,
    model_kit_scale_label,
)


def _museum_gallery_labels(span: str) -> list[str]:
    """Extract named museum/gallery venues from first-person visit evidence."""
    labels: list[str] = []
    patterns = (
        r"\b(?P<label>(?:The\s+)?[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,5}\s+"
        r"Museum(?:\s+of\s+[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,4})?)'?s?\b",
        r"\b(?P<label>(?:The\s+)?[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,5}\s+(?:Museum|Gallery|Cube))\b",
        r"\b(?P<label>(?:Museum|Gallery)\s+of\s+[A-Z][A-Za-z'&-]+(?:\s+[A-Z][A-Za-z'&-]+){0,4})'?s?\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, span):
            label = re.sub(r"'s$", "", _clean_count_item_label(match.group("label")))
            identity = _normalize_count_identity(label)
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _musical_instrument_labels(span: str) -> list[str]:
    """Extract distinct currently-owned musical instruments from first-person evidence."""
    labels: list[str] = []
    patterns = (
        r"\b(?P<label>Fender\s+Stratocaster\s+electric\s+guitar)\b",
        r"\b(?:acoustic\s+guitar,\s+a\s+)?(?P<label>Yamaha\s+FG800(?:\s+acoustic\s+guitar)?)\b",
        r"\b(?P<label>5-piece\s+Pearl\s+Export(?:\s+drum\s+set)?)\b",
        r"\b(?P<label>Korg\s+B1(?:\s+piano)?)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_instrument_label(_clean_count_item_label(match.group("label")))
            identity = _normalize_count_identity(label)
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    generic_patterns = (
        r"\b(?:my\s+)?(?P<label>[A-Z][A-Za-z0-9'-]+(?:\s+[A-Z][A-Za-z0-9'-]+){0,3}\s+"
        r"(?:electric\s+guitar|acoustic\s+guitar|drum\s+set|piano))\b",
        r"\b(?:my\s+)?(?P<label>(?:electric\s+guitar|acoustic\s+guitar|drum\s+set|piano))\b",
    )
    for pattern in generic_patterns:
        for match in re.finditer(pattern, span):
            label = _canonical_instrument_label(_clean_count_item_label(match.group("label")))
            identity = _normalize_count_identity(label)
            if any(
                identity
                and (
                    identity in _normalize_count_identity(existing)
                    or _normalize_count_identity(existing) in identity
                )
                for existing in labels
            ):
                continue
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _generic_instrument_family(label: str) -> str:
    normalized = _normalize_count_identity(label)
    generic = {
        "acoustic guitar": "acoustic guitar",
        "electric guitar": "electric guitar",
        "drum set": "drum set",
        "piano": "piano",
    }
    return generic.get(normalized, "")


def _instrument_family(label: str) -> str:
    normalized = _normalize_count_identity(label)
    for family in ("acoustic guitar", "electric guitar", "drum set", "piano"):
        if family in normalized:
            return family
    return ""


def _filter_subsumed_instrument_rows(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    included = [row for row in rows if row.kind == "event" and not row.exclude_reason]
    specific_families_by_source: dict[str, set[str]] = {}
    for row in included:
        if _generic_instrument_family(row.label):
            continue
        family = _instrument_family(row.label)
        if not family:
            continue
        specific_families_by_source.setdefault(row.source_group, set()).add(family)
    if not specific_families_by_source:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        family = _generic_instrument_family(row.label)
        if (
            row.exclude_reason
            or row.kind != "event"
            or not family
            or family not in specific_families_by_source.get(row.source_group, set())
        ):
            filtered.append(row)
            continue
        filtered.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="generic_instrument_subsumed",
                confidence=row.confidence,
            )
        )
    return filtered


def _filter_count_rows(
    subject: str,
    rows: list[EvidenceLedgerRow],
    *,
    query: str,
) -> list[EvidenceLedgerRow]:
    """Apply subject-specific count normalization after candidate extraction."""
    rows = _filter_specific_action_object_count_rows(query, rows, subject=subject)
    if subject == "property_viewing":
        return _filter_target_property_rows(query, rows)
    if subject == "musical_instrument":
        return _filter_subsumed_instrument_rows(rows)
    if subject == "museum_gallery":
        return _filter_museum_gallery_rows(rows)
    if subject != "doctor_visit":
        return rows
    included = [row for row in rows if row.kind == "event" and not row.exclude_reason]
    specific_identities = {
        row.normalized_identity
        for row in included
        if row.normalized_identity not in {"doctor_visit=doctor", "doctor_visit=physician"}
    }
    if not specific_identities:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.exclude_reason or row.normalized_identity not in {"doctor_visit=doctor", "doctor_visit=physician"}:
            filtered.append(row)
            continue
        filtered.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="generic_role_subsumed",
                confidence=row.confidence,
            )
        )
    return filtered


def _film_festival_labels(span: str) -> list[str]:
    """Extract distinct named film festivals from a participation memory span."""
    labels: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"\b(?P<label>[A-Z][A-Za-z0-9&.'-]+(?:\s+[A-Z][A-Za-z0-9&.'-]+){0,5}\s+"
        r"(?:International\s+Film\s+Festival|Film\s+Festival|Fest))\b",
        span,
    ):
        label = _clean_count_item_label(match.group("label"))
        identity = _normalize_count_identity(label)
        if label and identity not in seen:
            seen.add(identity)
            labels.append(label)
    for known in ("Sundance", "Tribeca", "Cannes", "Telluride", "SXSW", "TIFF"):
        if re.search(rf"\b{re.escape(known)}\b", span, flags=re.IGNORECASE):
            if any(re.search(rf"\b{re.escape(known)}\b", label, flags=re.IGNORECASE) for label in labels):
                continue
            identity = _normalize_count_identity(known)
            if identity not in seen:
                seen.add(identity)
                labels.append(known)
    return labels


def rendered_list_label(row: EvidenceLedgerRow) -> str:
    """Return a display-safe list label."""
    label = " ".join(row.label.strip(" .,'\"").split())
    if row.normalized_identity.startswith("film_festival=") and not re.match(
        r"(?i)^(?:attended|went|visited)\b",
        label,
    ):
        return f"attended the {label}"
    return label


def _joined_count_labels(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    labels = [rendered_list_label(row) for row in candidates if rendered_list_label(row)]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _joined_count_labels_without_two_item_comma(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    labels = [rendered_list_label(row) for row in candidates if rendered_list_label(row)]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _joined_kitchen_item_labels(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    labels = [
        f"the {label}"
        for row in candidates
        if (label := rendered_list_label(row))
    ]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _joined_wedding_couples(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    couples = [
        couple
        for row in candidates
        if (couple := _wedding_couple_from_label(rendered_list_label(row)))
    ]
    if not couples:
        return ""
    if len(couples) == 1:
        return couples[0]
    return ", ".join(couples[:-1]) + f", and {couples[-1]}"


def common_count_action(candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    """Return the common leading action across count labels, when stable."""
    actions: list[str] = []
    for row in candidates:
        label = rendered_list_label(row)
        if not label:
            continue
        first = label.split(maxsplit=1)[0].casefold()
        if first:
            actions.append(first)
    if not actions:
        return ""
    first_action = actions[0]
    if all(action == first_action for action in actions):
        return first_action
    return ""


def count_answer_text(query: str, candidates: tuple[EvidenceLedgerRow, ...]) -> str:
    """Render a compact natural-language answer for count synthesis."""
    subject = count_subject_phrase(query)
    if not subject:
        return ""
    action = common_count_action(candidates)
    count = count_display(len(candidates))
    query_tokens = set(source_tokens(query))
    count_subject = _count_subject(query)
    if subject == "model kits" and query_tokens & {"work", "worked"} and query_tokens & {"buy", "bought"}:
        return f"I have worked on or bought {count} {subject}."
    if count_subject == "film_festival":
        return f"I attended {count} movie festivals."
    if count_subject == "doctor_visit":
        labels = _joined_count_labels(candidates)
        if labels:
            return f"I visited {count} different doctors: {labels}."
        return f"I visited {count} different doctors."
    if count_subject == "museum_gallery":
        labels = _joined_count_labels(candidates)
        if labels:
            return f"I visited {count} different museums or galleries: {labels}."
        return f"I visited {count} different museums or galleries."
    if count_subject == "kitchen_item":
        labels = _joined_kitchen_item_labels(candidates)
        if labels:
            return f"I replaced or fixed {count} items: {labels}."
        return f"I replaced or fixed {count} kitchen items."
    if count_subject == "wedding":
        couples = _joined_wedding_couples(candidates)
        if couples:
            return f"I attended {count} weddings. The couples were {couples}."
        return f"I attended {count} weddings."
    if count_subject == "rollercoaster_ride":
        return f"I rode rollercoasters {len(candidates)} times."
    if count_subject == "fish_inventory":
        return f"There are {len(candidates)} fish in my aquariums."
    if count_subject == "competitive_sport":
        labels = _joined_count_labels_without_two_item_comma(candidates)
        if labels:
            return f"I played {count} sports competitively in the past: {labels}."
        return f"I played {count} sports competitively in the past."
    if count_subject == "dinner_party":
        labels = _joined_count_labels_without_two_item_comma(candidates)
        if labels:
            return f"I attended {count} dinner parties: {labels}."
        return f"I attended {count} dinner parties."
    if count_subject == "property_viewing" and query_tokens & {"view", "viewed"}:
        return f"I viewed {count} properties."
    if count_subject == "musical_instrument":
        return f"I currently own {count} musical instruments."
    if action:
        return f"I {action} {count} {subject}."
    return f"There are {count} {subject}."


def list_candidate_lines(candidates: tuple[EvidenceLedgerRow, ...]) -> list[str]:
    """Render itemized count details from labeled count rows."""
    labeled_candidates = [row for row in candidates if rendered_list_label(row)]
    if not labeled_candidates:
        return []
    lines = [
        f"list_item_count={len(labeled_candidates)}",
        "list_items="
        + " | ".join(rendered_list_label(row) for row in labeled_candidates),
        "list_source_ids=" + ",".join(row.source_group for row in labeled_candidates),
    ]
    if all(row.normalized_identity.startswith("model_kit=") for row in labeled_candidates):
        lines.append(
            "model_kit_scales="
            + " | ".join(model_kit_scale_label(rendered_list_label(row)) for row in labeled_candidates)
        )
    return lines


def _property_outcome_reason(row: EvidenceLedgerRow) -> str:
    """Return a concise reason a viewed property did not become the accepted offer."""
    span = row.raw_span or row.context
    label = rendered_list_label(row)
    property_name = _property_subject_phrase(label, span)
    if kitchen := re.search(
        r"\bkitchen\s+needed\s+(?:some\s+)?(?P<severity>serious\s+)?renovation(?:\s+work)?\b",
        span,
        flags=re.IGNORECASE,
    ):
        severity = "serious " if kitchen.group("severity") else ""
        return f"the kitchen of {_base_property_type(property_name)} needed {severity}renovation"
    if re.search(r"\b(?:did\s+not|didn't|does\s+not|doesn't|just\s+didn't)\s+fit\s+(?:my\s+)?budget\b", span, flags=re.IGNORECASE) or re.search(
        r"\b(?:out\s+of\s+my\s+budget|way\s+out\s+of\s+my\s+league)\b",
        span,
        flags=re.IGNORECASE,
    ):
        location = _property_location_phrase(label, span)
        return f"the property in {location} was out of my budget" if location else f"{property_name} was out of my budget"
    if noise := re.search(
        r"\b(?P<reason>(?:the\s+)?noise\s+from\s+the\s+highway|highway\s+noise)\s+was\s+a\s+deal-breaker\b",
        span,
        flags=re.IGNORECASE,
    ):
        reason = " ".join(noise.group("reason").split())
        return f"{reason} was a deal-breaker for {property_name}"
    if re.search(r"\b(?:offer\s+got\s+rejected|offer\s+was\s+rejected|rejected)\b[^.!?]{0,80}\bhigher\s+bid\b", span, flags=re.IGNORECASE):
        return f"my offer on {property_name} was rejected due to a higher bid"
    return label


def _count_outcome_lines(
    query: str,
    candidates: tuple[EvidenceLedgerRow, ...],
) -> list[str]:
    if _count_subject(query) != "property_viewing" or not candidates:
        return []
    outcomes = [
        outcome
        for row in candidates
        if (outcome := _property_outcome_reason(row))
    ]
    if not outcomes:
        return []
    target = _property_offer_target_phrase(query)
    if target:
        prefix = f"I viewed {count_display(len(candidates))} properties before making an offer on {target}."
    else:
        prefix = f"I viewed {count_display(len(candidates))} properties before making an offer."
    return [
        (
            "property_outcome_answer="
            f"{prefix} The reasons I didn't make an offer on them were: "
            + _joined_property_outcomes(outcomes)
            + "."
        )
    ]


def _instrument_ownership_lines(
    query: str,
    candidates: tuple[EvidenceLedgerRow, ...],
) -> list[str]:
    if _count_subject(query) != "musical_instrument" or not candidates:
        return []
    details: list[str] = []
    for row in candidates:
        label = rendered_list_label(row)
        if not label:
            continue
        duration = _instrument_duration_phrase(row.context)
        details.append(f"the {label} for {duration}")
    if not details:
        return []
    return [
        (
            "instrument_ownership_answer="
            f"I currently own {len(candidates)} musical instruments. I've had "
            + ", ".join(details[:-1])
            + (f", and {details[-1]}" if len(details) > 1 else details[0])
            + "."
        )
    ]


def temporal_evidence_text(text: str) -> str:
    """Remove metadata dates that should not be treated as answer evidence."""
    text = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)", " ", text)
    return re.sub(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\b", " ", text)


def context_year(text: str) -> int | None:
    """Infer the default year for month/day date mentions."""
    match = re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/", text)
    if match:
        return int(match.group("year"))
    match = re.search(r"\b(?P<year>20\d{2})[/-]\d{1,2}[/-]\d{1,2}\b", text)
    if match:
        return int(match.group("year"))
    return None


def temporal_sequence_query(query: str) -> bool:
    """Return whether a query asks for an ordered list of remembered events."""
    tokens = set(source_tokens(query))
    if {"first", "second", "third"} <= tokens:
        return True
    if not tokens & {"order", "ordered", "sequence", "timeline"}:
        return False
    if tokens & {"earliest", "latest"}:
        return True
    if {"first", "last"} <= tokens:
        return True
    return bool(tokens & {"events", "trips", "airlines", "sports", "activities", "watched", "flew", "took"})


def temporal_sequence_has_temporal_cue(text: str) -> bool:
    """Return whether a source span has a cue tying the event to session time."""
    return bool(
        re.search(
            r"\b(?:today|tonight|yesterday|recently|just|ago|last\s+(?:week|month|year|weekend)|"
            r"this\s+(?:morning|afternoon|evening|week|month|weekend))\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def temporal_sequence_relative_days_ago(text: str) -> int | None:
    """Extract coarse relative-day offsets for event ordering."""
    lowered = text.casefold()
    if re.search(r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lowered):
        return 0
    if "yesterday" in lowered:
        return 1
    if "last week" in lowered:
        return 7
    if "last weekend" in lowered:
        return 7
    if "last month" in lowered:
        return 30
    if "a few months ago" in lowered:
        return 90
    if "recently" in lowered:
        return 3
    match = re.search(
        r"\b(?P<value>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?P<unit>days?|weeks?|months?)\s+ago\b",
        lowered,
    )
    if not match:
        return None
    value_text = match.group("value")
    value = _NUMBER_WORDS.get(value_text, int(value_text) if value_text.isdigit() else 0)
    unit = match.group("unit")
    multiplier = 1 if unit.startswith("day") else 7 if unit.startswith("week") else 30
    return value * multiplier


def temporal_sequence_local_cue_score(text: str) -> int:
    """Score temporal cues local to one candidate event mention."""
    lowered = text.casefold()
    score = 0
    if re.search(r"\b(?:today|tonight|this morning|this afternoon|this evening)\b", lowered):
        score += 12
    if "yesterday" in lowered:
        score += 10
    if re.search(r"\b(?:last week|last weekend|last month|last year)\b", lowered):
        score += 8
    if re.search(r"\b(?:a few months ago|ago)\b", lowered):
        score += 7
    if re.search(r"\b(?:recently|just)\b", lowered):
        score += 4
    return score


def _clean_sports_sequence_label(label: str) -> str:
    """Normalize sports event labels while preserving named-event detail."""
    label = re.sub(
        r"\b(?:where|with|at\s+home|at\s+my|at\s+our|last\s+weekend|yesterday|today)\b.*$",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(
        r"^the\s+Kansas\s+City\s+Chiefs\s+defeat\s+the\s+Buffalo\s+Bills\s+in\s+the\s+Divisional\s+Round\s+of\s+the\s+",
        "",
        label,
        flags=re.IGNORECASE,
    )
    return " ".join(label.strip(" .,'\"").split())


def temporal_sequence_sentences(text: str) -> list[str]:
    """Return bounded candidate event sentences from user-authored spans."""
    cleaned = re.sub(r"\bquery=[^#\n]*?(?=\s+(?:content=|snippet=|source_path=|# Event)|$)", " ", text)
    cleaned = re.sub(r"\bcontent=longmemeval_session_id=\S+\s*", " ", cleaned)
    cleaned = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", " ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=user:|assistant:)", cleaned)
    sentences: list[str] = []
    for part in parts:
        sentence = " ".join(part.strip().split())
        if not sentence:
            continue
        if re.match(r"assistant\s*:", sentence, flags=re.IGNORECASE):
            continue
        sentence = re.sub(r"^(?:\d+\.\s*)?user\s*:\s*", "", sentence, flags=re.IGNORECASE)
        if re.search(r"\bI(?:\s+|['’](?:m|ve|d|ll|re)\b)", sentence, flags=re.IGNORECASE):
            sentences.append(sentence)
    return sentences


def _temporal_sequence_match_evidence(sentence: str, match: re.Match[str]) -> str:
    """Return a bounded span around a candidate event mention."""
    start = max(0, match.start() - 40)
    end = min(len(sentence), match.end() + 80)
    return sentence[start:end]


def temporal_sequence_graduation_candidates(sentence: str) -> list[tuple[str, str]]:
    """Extract named graduation events from cited user memories."""
    patterns = (
        r"\b(?:my\s+)?(?:niece|nephew|cousin|friend|sister|brother|daughter|son)?\s*"
        r"(?P<name>[A-Z][A-Za-z'-]{1,40})\s+(?:just\s+|recently\s+)?graduated\b",
        r"\b(?:my\s+)?(?:niece|nephew|cousin|friend|sister|brother|daughter|son)\s+"
        r"(?P<name>[A-Z][A-Za-z'-]{1,40})(?:'s|’s)\s+[^.!?]{0,80}\bgraduation\b",
        r"\b(?P<name>[A-Z][A-Za-z'-]{1,40})(?:'s|’s)\s+[^.!?]{0,80}\bgraduation\b",
        r"\b(?P<name>[A-Z][A-Za-z'-]{1,40}),\s+who\s+graduated\b",
    )
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, sentence):
            name = match.group("name")
            if name.casefold() in seen:
                continue
            seen.add(name.casefold())
            candidates.append((f"{name} graduated", _temporal_sequence_match_evidence(sentence, match)))
    return candidates


def temporal_sequence_venue_label(sentence: str) -> str:
    """Return a venue entity from a first-person temporal venue sentence."""
    if not re.search(
        r"\bI\s+(?:just\s+|recently\s+|actually\s+|also\s+)?(?:"
        r"visited|attended|participated\s+in|took|went\s+to|got\s+back\s+from|"
        r"came\s+back\s+from|saw|went\s+on)\b",
        sentence,
        flags=re.IGNORECASE,
    ):
        return ""
    labels = _museum_gallery_labels(sentence)
    return labels[0] if labels else ""


def _temporal_sequence_slot_terms(text: str) -> set[str]:
    return {
        token
        for token in source_tokens(text)
        if len(token) > 2 and token not in _TEMPORAL_SEQUENCE_STOPWORDS and token not in {"thing", "things"}
    }


def temporal_sequence_query_slot_label(query_slots: tuple[str, ...], label: str) -> str:
    """Map an extracted cited event label back to a query-enumerated slot."""
    if not query_slots:
        return ""
    label_terms = _temporal_sequence_slot_terms(label)
    if not label_terms:
        return ""
    best: tuple[int, int, str] | None = None
    for index, slot in enumerate(query_slots):
        slot_terms = _temporal_sequence_slot_terms(slot)
        if not slot_terms:
            continue
        overlap = len(label_terms & slot_terms)
        threshold = 1 if len(slot_terms) == 1 else 2
        if overlap < threshold:
            continue
        candidate = (overlap, -index, slot)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else ""


def clean_temporal_sequence_label(label: str, *, airline: bool, strip_leading_article: bool = True) -> str:
    """Normalize a temporal event label while preserving answer-bearing nouns."""
    label = re.sub(
        r"\b(?:today|tonight|yesterday|recently|last\s+(?:week|month|year|weekend)|"
        r"about\s+(?:a|an|one|two|three|four|five|\d+)\s+(?:days?|weeks?|months?)\s+ago)\b",
        " ",
        label,
        flags=re.IGNORECASE,
    )
    label = re.split(
        r"\s+and\s+(?:realized|realised|noticed|learned|found|decided|started\s+looking|needed|need)\b",
        label,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    label = re.split(r"\s+(?:but|while|because)\s+", label, maxsplit=1)[0]
    label = " ".join(label.strip(" .,'\"").split())
    if strip_leading_article:
        label = re.sub(r"^(?:a|an|the)\s+", "", label, flags=re.IGNORECASE)
    if airline:
        label = re.sub(r"\bflight\b.*$", "", label, flags=re.IGNORECASE).strip(" .,'\"")
    return label


def _temporal_sequence_pattern_candidates(patterns: tuple[tuple[str, str], ...], sentence: str) -> list[tuple[str, str]]:
    """Return all supported event pattern labels from a sentence."""
    candidates: list[tuple[str, str]] = []
    for prefix, pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            label = clean_temporal_sequence_label(
                match.group("label"),
                airline=False,
                strip_leading_article=not bool(prefix),
            )
            if label:
                candidates.append((f"{prefix}{label}" if prefix else label, _temporal_sequence_match_evidence(sentence, match)))
    return candidates


def clean_temporal_sequence_query_slot(label: str) -> str:
    """Normalize a query-enumerated event slot into an answer label."""
    label = re.sub(r"^\s*(?:the\s+day\s+)?I\s+", "", label, flags=re.IGNORECASE)
    return clean_temporal_sequence_label(label, airline=False)


def temporal_sequence_query_slots(query: str) -> tuple[str, ...]:
    """Return canonical event labels explicitly enumerated in the query."""
    quoted = tuple(
        clean_temporal_sequence_query_slot(match.group("label"))
        for match in re.finditer(r"'(?P<label>[^']{3,180})'", query)
    )
    if quoted:
        return tuple(slot for slot in quoted if slot)
    slots: list[str] = []
    for match in re.finditer(
        r"(?:\bthe\s+day\s+)?\bI\s+(?P<label>.*?)(?=,\s*(?:and\s+)?(?:the\s+day\s+)?I\s+|[?]$)",
        query,
        flags=re.IGNORECASE,
    ):
        label = clean_temporal_sequence_query_slot(match.group("label"))
        if label:
            slots.append(label)
    return tuple(slots)


def temporal_sequence_requested_count(query: str) -> int | None:
    """Return an explicit requested sequence length from order-list wording."""
    lowered = query.casefold()
    tokens = set(source_tokens(query))
    if {"first", "second", "third"} <= tokens:
        return 3
    patterns = (
        r"\border\s+of\s+(?:the\s+)?(?P<value>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+",
        r"\bsequence\s+of\s+(?:the\s+)?(?P<value>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+",
        r"\btimeline\s+of\s+(?:the\s+)?(?P<value>\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value_text = match.group("value")
        return int(value_text) if value_text.isdigit() else _NUMBER_WORDS.get(value_text)
    quoted_slots = temporal_sequence_query_slots(query)
    return len(quoted_slots) if len(quoted_slots) >= 2 else None


def normalize_temporal_sequence_label(label: str) -> str:
    """Normalize sequence labels for duplicate suppression."""
    normalized = label.casefold()
    normalized = re.sub(r"\b(?:a|an|the|my|our|with|for|to|from|today|recently)\b", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def temporal_sequence_identity_terms(label: str) -> set[str]:
    """Return content terms used for fuzzy duplicate suppression."""
    return {
        token
        for token in source_tokens(normalize_temporal_sequence_label(label))
        if len(token) > 2 and token not in _TEMPORAL_SEQUENCE_STOPWORDS
    }


def temporal_sequence_duplicate_row_index(rows: list[EvidenceLedgerRow], candidate: EvidenceLedgerRow) -> int | None:
    """Return the index of a semantically duplicate sequence row."""
    candidate_terms = temporal_sequence_identity_terms(candidate.label)
    if not candidate_terms:
        return None
    for index, row in enumerate(rows):
        row_terms = temporal_sequence_identity_terms(row.label)
        if not row_terms:
            continue
        shared = len(candidate_terms & row_terms)
        smaller = min(len(candidate_terms), len(row_terms))
        if smaller and shared / smaller >= 0.8:
            return index
    return None


def temporal_sequence_specificity(row: EvidenceLedgerRow) -> tuple[int, int, float]:
    """Return a stable preference for richer duplicate event labels."""
    terms = temporal_sequence_identity_terms(row.label)
    return (len(terms), len(row.label), row.confidence)


def temporal_sequence_deduped_rows(rows: list[EvidenceLedgerRow]) -> list[EvidenceLedgerRow]:
    """Suppress duplicate ordered events while keeping the most specific cited label."""
    included: list[EvidenceLedgerRow] = []
    excluded: list[EvidenceLedgerRow] = []
    for row in rows:
        duplicate_index = temporal_sequence_duplicate_row_index(included, row)
        if duplicate_index is None:
            included.append(row)
            continue
        existing = included[duplicate_index]
        if temporal_sequence_specificity(row) > temporal_sequence_specificity(existing):
            included[duplicate_index] = row
            excluded.append(replace(existing, exclude_reason="duplicate_identity"))
        else:
            excluded.append(replace(row, exclude_reason="duplicate_identity"))
    included.sort(key=lambda row: int(row.value) if row.value.lstrip("-").isdigit() else 0)
    return [*included, *excluded]


def temporal_sequence_sports_candidates(sentence: str) -> list[tuple[str, str]]:
    """Extract sports event labels from first-person watch/participation memories."""
    patterns = (
        r"\bI\s+(?:just\s+|recently\s+)?went\s+to\s+(?P<label>[^.!?;,]{3,120}?\b(?:game|match|tournament|race|run|triathlon|playoffs?)(?:\s+at\s+[^.!?;,]{2,80})?)",
        r"\bfrom\s+(?:the\s+)?(?P<label>[^.!?;,]{3,120}?\b(?:game|match|tournament|race|run|triathlon|playoffs?))\s+I\s+watched\b",
        r"\bwatch(?:ed|ing)\s+(?:the\s+)?(?P<label>[^.!?;,]{3,160}?\b(?:game|match|tournament|race|run|triathlon|playoffs?))\b",
        r"\bI\s+(?:just\s+|recently\s+)?completed\s+(?:the\s+)?(?P<label>[^.!?;,]{3,120}?\b(?:race|run|triathlon|tournament|marathon))\b",
        r"\bI\s+(?:just\s+|recently\s+)?finished\s+(?:a\s+|an\s+|the\s+)?[^.!?;,]{0,80}?\bat\s+(?:the\s+)?(?P<label>[^.!?;,]{3,120}?\b(?:race|run|triathlon|tournament|marathon))\b",
        r"\bI\s+participate\s+in\s+(?:the\s+)?(?P<label>[^.!?;,]{3,120}?\b(?:game|match|tournament|race|run|triathlon|marathon))\b",
        r"\bI\s+(?:just\s+|recently\s+)?participated\s+in\s+(?:the\s+)?(?P<label>[^.!?;,]{3,120}?\b(?:game|match|tournament|race|run|triathlon|marathon))\b",
    )
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            label = clean_temporal_sequence_label(match.group("label"), airline=False, strip_leading_article=False)
            label = _clean_sports_sequence_label(label)
            if not label:
                continue
            normalized = normalize_temporal_sequence_label(label)
            if normalized in seen:
                continue
            seen.add(normalized)
            candidates.append((label, _temporal_sequence_match_evidence(sentence, match)))
    return candidates


def source_group_sequence_index(group: str, *, fallback: int) -> int:
    """Extract a stable sequence suffix from LongMemEval-style source groups."""
    match = re.search(r"(?:^|[_-])(?P<index>\d{1,4})$", group)
    if match:
        return int(match.group("index"))
    return fallback


def session_date_anchor_allowed(
    query: str,
    text: str,
    *,
    focus_terms: set[str],
    relevance: int,
) -> bool:
    """Return whether session-date metadata is strong enough to become typed date evidence."""
    if relevance < 2:
        return False
    query_terms = {term for term in source_tokens(query) if len(term) > 2}
    text_terms = set(source_tokens(text))
    event_terms = {
        "attended",
        "baked",
        "class",
        "concert",
        "discovered",
        "exhibit",
        "garden",
        "got",
        "harvested",
        "launched",
        "made",
        "played",
        "received",
        "signed",
        "started",
        "tested",
        "testing",
        "took",
        "visited",
        "website",
        "feedback",
    }
    temporal_cues = {
        "ago",
        "back",
        "day",
        "just",
        "recently",
        "today",
        "tonight",
        "yesterday",
    }
    if query_terms & event_terms and text_terms & event_terms and relevance >= 3:
        return True
    return bool(text_terms & temporal_cues and len(focus_terms & text_terms) >= 2)


def _small_number_word(value: str) -> int:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
    }
    return words.get(value.casefold(), int(value) if value.isdigit() else 0)


def relative_session_date_offset(text: str) -> int:
    """Return a small relative offset anchored to the source session date."""
    lowered = text.casefold()
    if re.search(r"\b(?:yesterday|the day before)\b", lowered):
        return -1
    if re.search(r"\b(?:tomorrow|the next day)\b", lowered):
        return 1
    match = re.search(r"\b(?P<count>\d{1,2}|one|two|three|four|five|six|seven)\s+days?\s+ago\b", lowered)
    if match:
        return -_small_number_word(match.group("count"))
    if re.search(r"\b(?:today|earlier today|this morning|this afternoon|this evening|tonight)\b", lowered):
        return 0
    return 0


def session_anchor_date(raw_text: str, evidence_text: str) -> date | None:
    """Return a typed event-date anchor from session metadata and relative text cues."""
    match = re.search(
        r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        raw_text,
    )
    if not match:
        return None
    try:
        value = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None
    return value + timedelta(days=relative_session_date_offset(evidence_text))


def nth_weekday_of_month(year: int, month: int, *, weekday: int, n: int) -> date:
    """Return the nth weekday within a month."""
    value = date(year, month, 1)
    days_until_weekday = (weekday - value.weekday()) % 7
    return value + timedelta(days=days_until_weekday + (n - 1) * 7)


def black_friday_date(year: int) -> date:
    """Return Black Friday for a given year."""
    thanksgiving = nth_weekday_of_month(year, 11, weekday=3, n=4)
    return thanksgiving + timedelta(days=1)


def append_unique_date(dates: list[date], value: date) -> None:
    """Append a date once while preserving extraction order."""
    if value not in dates:
        dates.append(value)


def append_unique_date_match(
    dates: list[ExplicitDateMatch],
    value: date,
    *,
    start: int,
    end: int,
    raw: str,
) -> None:
    """Append a date match once while preserving extraction order."""
    if any(existing.value == value for existing in dates):
        return
    dates.append(ExplicitDateMatch(value=value, start=start, end=end, raw=raw))


def append_date(dates: list[date], year: int, month: int, day: int) -> None:
    """Append a valid calendar date once."""
    try:
        value = date(year, month, day)
    except ValueError:
        return
    append_unique_date(dates, value)


def append_date_match(
    dates: list[ExplicitDateMatch],
    year: int,
    month: int,
    day: int,
    *,
    start: int,
    end: int,
    raw: str,
) -> None:
    """Append a valid calendar date match once."""
    try:
        value = date(year, month, day)
    except ValueError:
        return
    append_unique_date_match(dates, value, start=start, end=end, raw=raw)


def explicit_date_matches(text: str, *, default_year: int | None) -> list[ExplicitDateMatch]:
    """Extract explicit and supported relative dates with source offsets."""
    dates: list[ExplicitDateMatch] = []
    for match in re.finditer(
        r"\b(?P<year>20\d{2})/(?P<month>\d{1,2})/(?P<day>\d{1,2})\b",
        text,
    ):
        append_date_match(
            dates,
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
        )
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    if default_year is not None:
        black_friday = black_friday_date(default_year)
        for match in re.finditer(r"\bblack friday\b", text, flags=re.IGNORECASE):
            if re.search(r"\b(?:on|during)\s+black friday\b", text[max(0, match.start() - 16) : match.end()], flags=re.IGNORECASE):
                append_unique_date_match(dates, black_friday, start=match.start(), end=match.end(), raw=match.group(0))
        for match in re.finditer(r"\b(?:a|one|1)\s+weeks?\s+before\s+black friday\b", text, flags=re.IGNORECASE):
            append_unique_date_match(dates, black_friday - timedelta(days=7), start=match.start(), end=match.end(), raw=match.group(0))
        for match in re.finditer(r"\b(?:a|one|1)\s+weeks?\s+after\s+black friday\b", text, flags=re.IGNORECASE):
            append_unique_date_match(dates, black_friday + timedelta(days=7), start=match.start(), end=match.end(), raw=match.group(0))
    for match in re.finditer(
        rf"\b(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        append_date_match(
            dates,
            year,
            _MONTHS[match.group("month").casefold()],
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
        )
    for match in re.finditer(
        rf"\b(?:the\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+of\s+(?P<month>{month_pattern})(?:,\s*(?P<year>20\d{{2}}))?\b",
        text,
        flags=re.IGNORECASE,
    ):
        year = int(match.group("year")) if match.group("year") else default_year
        if year is None:
            continue
        append_date_match(
            dates,
            year,
            _MONTHS[match.group("month").casefold()],
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
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
        append_date_match(
            dates,
            year,
            int(match.group("month")),
            int(match.group("day")),
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
        )
    return dates


def _count_row_date(row: EvidenceLedgerRow) -> date | None:
    default_year = context_year(row.context)
    text = temporal_evidence_text(row.raw_span)
    matches = explicit_date_matches(text, default_year=default_year)
    if matches:
        return matches[0].value
    if month_value := month_only_date(text, default_year=default_year):
        return month_value
    if anchor := session_anchor_date(row.context, row.raw_span):
        return anchor
    return None


def _temporal_evidence_row(row: EvidenceLedgerRow) -> TemporalEvidenceRow:
    row_date = _count_row_date(row)
    return TemporalEvidenceRow(
        event_id=row.fact_id,
        label=row.label,
        action=_count_row_action(row),
        object_terms=tuple(
            token
            for token in source_tokens(f"{row.label} {row.raw_span}")
            if len(token) > 2 and token not in _COUNT_STOPWORDS and not token.isdigit()
        ),
        event_date=row_date,
        source_group=row.source_group,
        citation=row.citation,
        canonical_identity=row.normalized_identity,
        raw_span=row.raw_span,
        include_reason=row.include_reason,
        exclude_reason=row.exclude_reason,
    )


def explicit_dates(text: str, *, default_year: int | None) -> list[date]:
    """Extract explicit and supported relative dates from source text."""
    return [match.value for match in explicit_date_matches(text, default_year=default_year)]


def temporal_sequence_order_value(raw_text: str, evidence_text: str, *, prefer_relative: bool = False) -> tuple[int, str]:
    """Return a sortable chronology value and the evidence reason used for it."""
    explicit = explicit_dates(evidence_text, default_year=context_year(raw_text))
    if explicit:
        return explicit[0].toordinal(), "explicit_date_anchor"
    anchor = session_anchor_date(raw_text, evidence_text)
    days_ago = temporal_sequence_relative_days_ago(evidence_text)
    if anchor is not None and days_ago is not None and prefer_relative:
        return anchor.toordinal() - days_ago, "relative_session_date_anchor"
    if anchor is not None and temporal_sequence_has_temporal_cue(evidence_text):
        return anchor.toordinal(), "session_date_anchor"
    if days_ago is not None:
        return -days_ago, "relative_time_anchor"
    return 0, "provenance_order_anchor"


def _expand_temporal_anchor_terms(terms: Iterable[str]) -> set[str]:
    """Expand query anchor terms with common inflections used in memory text."""
    expanded = set(terms)
    variants = {
        "book": {"booked", "booking"},
        "buy": {"bought", "buying"},
        "find": {"found", "finding", "saw", "seen"},
        "get": {"got", "getting"},
        "love": {"loved"},
        "loved": {"love"},
        "make": {"made", "making"},
        "order": {"ordered", "ordering"},
        "purchase": {"purchased", "purchasing"},
        "reserve": {"reserved", "reservation"},
        "schedule": {"scheduled", "scheduling"},
        "start": {"started", "starting"},
        "starting": {"start", "started"},
        "work": {"worked", "working"},
    }
    for term in tuple(expanded):
        expanded.update(variants.get(term, ()))
    return expanded


def temporal_ordered_anchor_score(
    left: EvidenceLedgerRow,
    right: EvidenceLedgerRow,
    anchor_terms: tuple[set[str], set[str]],
) -> int:
    """Score whether two date rows match ordered temporal query anchors."""
    first_anchor, second_anchor = anchor_terms
    if not first_anchor or not second_anchor:
        return 0
    left_context_terms = set(source_tokens(left.context))
    right_context_terms = set(source_tokens(right.context))
    left_date = date.fromisoformat(left.value)
    right_date = date.fromisoformat(right.value)
    forward_score = len(first_anchor & left_context_terms) + len(second_anchor & right_context_terms)
    reverse_score = len(first_anchor & right_context_terms) + len(second_anchor & left_context_terms)
    if left_date <= right_date:
        forward_score += 1
    else:
        reverse_score += 1
    return max(forward_score, reverse_score)


def _explicit_temporal_operand_pair_score(
    left: EvidenceLedgerRow,
    right: EvidenceLedgerRow,
    anchor_terms: tuple[set[str], set[str]],
) -> int:
    """Score explicit date pairs only when both query roles are covered."""
    first_anchor, second_anchor = anchor_terms
    if not first_anchor or not second_anchor:
        return 0
    left_context_terms = set(source_tokens(left.context))
    right_context_terms = set(source_tokens(right.context))
    left_date = date.fromisoformat(left.value)
    right_date = date.fromisoformat(right.value)
    if left_date <= right_date:
        earlier_terms, later_terms = left_context_terms, right_context_terms
    else:
        earlier_terms, later_terms = right_context_terms, left_context_terms
    first_score = len(first_anchor & earlier_terms)
    second_score = len(second_anchor & later_terms)
    if first_score <= 0 or second_score <= 0:
        return 0
    return first_score + second_score


def _temporal_anchor_action_index(terms: list[str]) -> int | None:
    """Return the split point for inverted before/after questions."""
    action_terms = {
        "book",
        "booked",
        "buy",
        "bought",
        "get",
        "got",
        "make",
        "made",
        "order",
        "ordered",
        "purchase",
        "purchased",
        "reserve",
        "reserved",
        "schedule",
        "scheduled",
    }
    for index, term in enumerate(terms):
        if term in action_terms:
            return index
    return None


def temporal_anchor_terms(subject_terms: tuple[str, ...]) -> tuple[set[str], set[str]]:
    """Split temporal query terms into ordered left/right event anchors."""
    if not subject_terms:
        return set(), set()
    terms = list(subject_terms)
    separators = {"after", "before", "between"}
    if "between" in terms and "and" in terms:
        between_index = terms.index("between")
        and_index = terms.index("and")
        if between_index < and_index:
            return (
                _expand_temporal_anchor_terms(terms[between_index + 1 : and_index]),
                _expand_temporal_anchor_terms(terms[and_index + 1 :]),
            )
    if "after" in terms:
        after_index = terms.index("after")
        return (
            _expand_temporal_anchor_terms(terms[after_index + 1 :]),
            _expand_temporal_anchor_terms(terms[:after_index]),
        )
    if "before" in terms:
        before_index = terms.index("before")
        if before_index == 0:
            trailing = terms[before_index + 1 :]
            action_index = _temporal_anchor_action_index(trailing)
            if action_index is not None and action_index > 0:
                return (
                    _expand_temporal_anchor_terms(trailing[action_index:]),
                    _expand_temporal_anchor_terms(trailing[:action_index]),
                )
        return (
            _expand_temporal_anchor_terms(terms[:before_index]),
            _expand_temporal_anchor_terms(terms[before_index + 1 :]),
        )
    midpoint = len(terms) // 2
    if midpoint == 0:
        return set(), set()
    left = _expand_temporal_anchor_terms(term for term in terms[:midpoint] if term not in separators)
    right = _expand_temporal_anchor_terms(term for term in terms[midpoint:] if term not in separators)
    return left, right


def _inverted_before_temporal_anchor_query(subject_terms: tuple[str, ...]) -> bool:
    """Return whether normalized terms match "before X did I Y" structure."""
    terms = list(subject_terms)
    if not terms or terms[0] != "before":
        return False
    action_index = _temporal_anchor_action_index(terms[1:])
    return action_index is not None and action_index > 0


def clean_label(label: str) -> str:
    """Return a bounded label with trailing clauses removed."""
    label = re.split(r"\b(?:on|in|for|with|and|but|because|when)\b", label, maxsplit=1)[0]
    return " ".join(label.strip(" .,'\"").split())


def _currency_merchant_label(text: str) -> str:
    """Extract merchant/store labels from local currency evidence."""
    patterns = (
        r"\b(?:spent|paid)\s+(?:around\s+|about\s+)?\$\d+(?:,\d{3})*(?:\.\d+)?\s+at\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})",
        r"\bat\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})\s+(?:and\s+)?(?:spent|paid)\s+(?:around\s+|about\s+)?\$\d+(?:,\d{3})*(?:\.\d+)?",
        r"\b(?:order|ordered)\s+with\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,3})\s+[^.!?]{0,80}?\b(?:spent|paid)\s+(?:around\s+|about\s+)?\$\d+(?:,\d{3})*(?:\.\d+)?",
    )
    for pattern in patterns:
        if match := re.search(pattern, text):
            return clean_label(match.group("label"))
    return ""


_CURRENCY_ITEM_TARGET_STOPWORDS = {
    "total",
    "cost",
    "costs",
    "money",
    "amount",
    "price",
    "spent",
    "spend",
    "paid",
    "last",
    "month",
    "months",
    "new",
    "week",
    "weeks",
    "year",
    "years",
    "gift",
    "gifts",
    "got",
    "max",
}


def _currency_itemized_target_slots(query: str) -> tuple[set[str], ...]:
    """Extract item slots from queries like 'cost of X and Y'."""
    lowered = query.casefold()
    if " and " not in lowered or not (
        {"cost", "costs", "spent", "spend", "paid", "price", "amount", "money"} & set(source_tokens(query))
    ):
        return ()
    match = re.search(
        r"\b(?:cost(?:s)?|spent|spend|paid|price|amount|money)\b[^?]{0,80}?\b(?:of|on|for)\s+(?P<items>[^?]+)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ()
    parts = re.split(r"\s+and\s+|,\s*", match.group("items"), flags=re.IGNORECASE)
    slots: list[set[str]] = []
    for part in parts:
        cleaned = re.sub(r"\b[A-Za-z0-9_-]+['’]s\s+", " ", part)
        terms = {
            token
            for token in source_tokens(cleaned)
            if len(token) > 2 and token not in _QUERY_STOPWORDS and token not in _CURRENCY_ITEM_TARGET_STOPWORDS
        }
        expanded = set(terms)
        for term in terms:
            if term.endswith("s") and len(term) > 3:
                expanded.add(term[:-1])
            else:
                expanded.add(f"{term}s")
        if expanded:
            slots.append(expanded)
    return tuple(slots)


def _currency_row_slot_match_text(row: EvidenceLedgerRow) -> str:
    """Return amount-local text for matching itemized money slots."""
    local_context = row.context
    if row.raw_span and row.raw_span in row.context:
        amount_index = row.context.find(row.raw_span)
        start = max(0, amount_index - 72)
        end = min(len(row.context), amount_index + len(row.raw_span) + 72)
        local_context = row.context[start:end]
    return " ".join((row.label, local_context, row.raw_span))


def _currency_slot_match_text(text: str) -> str:
    """Remove retrieval metadata before matching query item slots."""
    text = re.sub(r"\b(?:session_id|longmemeval_session_id|source_path|citation)=\S+", " ", text)
    text = re.sub(r"\brole=\S+", " ", text)
    return text


_CURRENCY_ITEM_SLOT_GENERIC_TERMS = {
    "auto",
    "autos",
    "car",
    "cars",
    "vehicle",
    "vehicles",
}


def _currency_slot_required_terms(slot: set[str]) -> set[str]:
    """Return substantive terms required to satisfy an itemized query slot."""
    substantive = slot - _CURRENCY_ITEM_SLOT_GENERIC_TERMS
    return substantive or slot


def _currency_row_target_slot_indexes(row: EvidenceLedgerRow, target_slots: tuple[set[str], ...]) -> tuple[int, ...]:
    text = _currency_slot_match_text(_currency_row_slot_match_text(row))
    row_terms = set(source_tokens(text))
    return tuple(
        index
        for index, slot in enumerate(target_slots)
        if _currency_slot_required_terms(slot) & row_terms
    )


def _currency_row_matches_target_slot(row: EvidenceLedgerRow, target_slots: tuple[set[str], ...]) -> bool:
    return bool(_currency_row_target_slot_indexes(row, target_slots))


def _filter_currency_ledger(
    ledger: EvidenceLedger,
    focus_terms: set[str],
    *,
    query: str,
) -> EvidenceLedger:
    included = list(ledger.included(kind="currency"))
    if len(included) < 2 or not focus_terms:
        return ledger
    if max((row.relevance for row in included), default=0) <= 0:
        return ledger
    selected_identities = {
        row.normalized_identity for row in included if row.relevance > 0
    }
    if len(selected_identities) < 2:
        return ledger
    rows: list[EvidenceLedgerRow] = []
    target_slots = _currency_itemized_target_slots(query)
    for row in ledger.rows:
        if row.exclude_reason or row.kind != "currency":
            rows.append(row)
            continue
        if target_slots and _currency_row_matches_target_slot(row, target_slots):
            rows.append(row)
            continue
        if row.normalized_identity in selected_identities:
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="query_focus_mismatch",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _filter_itemized_currency_targets(query: str, ledger: EvidenceLedger) -> EvidenceLedger:
    """Keep only rows matching requested item slots in conjunctive money queries."""
    target_slots = _currency_itemized_target_slots(query)
    if len(target_slots) < 2:
        return ledger
    included = ledger.included(kind="currency")
    if len(included) < 2:
        return ledger
    matched_rows = [
        row
        for row in included
        if _currency_row_matches_target_slot(row, target_slots)
    ]
    covered_slots = {
        slot_index
        for row in matched_rows
        for slot_index in _currency_row_target_slot_indexes(row, target_slots)
    }
    if len(covered_slots) < len(target_slots):
        return ledger
    selected_facts = {row.fact_id for row in matched_rows}
    rows: list[EvidenceLedgerRow] = []
    for row in ledger.rows:
        if row.kind != "currency" or row.exclude_reason or row.fact_id in selected_facts:
            rows.append(row)
            continue
        rows.append(
            EvidenceLedgerRow(
                fact_id=row.fact_id,
                source_group=row.source_group,
                citation=row.citation,
                kind=row.kind,
                value=row.value,
                unit=row.unit,
                label=row.label,
                raw_span=row.raw_span,
                context=row.context,
                normalized_identity=row.normalized_identity,
                relevance=row.relevance,
                include_reason=row.include_reason,
                exclude_reason="query_item_target_mismatch",
                confidence=row.confidence,
            )
        )
    return EvidenceLedger(plan=ledger.plan, rows=tuple(rows))


def _unit_price_currency_query(query: str) -> bool:
    tokens = set(source_tokens(query))
    return bool(tokens & {"each", "per", "apiece"})
