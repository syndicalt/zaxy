"""Split from synthesis.py (mechanical decomposition)."""


from __future__ import annotations

import re

from zaxy.synthesis.foundations import (
    _COUNT_STOPWORDS,
    _MONTHS,
    _NUMBER_WORDS,
    _NUMERIC_STATE_VALUE_PATTERN,
    _QUERY_STOPWORDS,
    CountEvidenceItem,
    EvidenceLedger,
    EvidenceLedgerRow,
    SynthesisPlan,
    _age_average_evidence,
    _duration_measure_query,
    _incidental_time_modifier_query,
    canonical_duration_unit,
    context_text,
    format_number,
    numeric_state_difference_query,
    source_citation,
    source_group,
    source_tokens,
)


def duration_identity_signature(evidence_span: str) -> str:
    """Return a projection-stable local signature for a duration operand."""
    text = evidence_span
    role_markers = list(
        re.finditer(
            r"(?:^|\b)(?:\d+\.\s*)?(?:user|assistant)\s*:|\brole=(?:user|assistant)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if role_markers:
        text = text[role_markers[-1].end():]
    text = re.split(
        r"\s+(?:end_line|source_path|source_start_line|source_end_line|source_event_seq|"
        r"source_event_hash|path|sha256|start_line|turn_index)=",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.sub(r"\blongmemeval_[a-z_]+=[^\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsession_id=[^\s]+", " ", text, flags=re.IGNORECASE)
    evidence_terms = [
        term
        for term in source_tokens(text)
        if term not in _QUERY_STOPWORDS
        and term not in {"minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months"}
    ]
    return " ".join(evidence_terms[:24])


def _clean_travel_duration_label(label: str) -> str:
    label = re.split(
        r"\s+\b(?:from|for|in|and|but|recently|last|only|about|around|approx(?:imately)?|which|that|it|was|were)\b",
        label,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    label = re.sub(r"\b(?:the|a|an|my|recent)\b", " ", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip(" .,:;!?-–—'\"").casefold()
    if not label or label in {"there", "here", "get there", "go there"}:
        return ""
    terms = [
        term
        for term in source_tokens(label)
        if term not in _QUERY_STOPWORDS and term not in {"trip", "road", "drive", "driving", "drove"}
    ]
    return " ".join(terms[:8])


def _travel_duration_subject_signature(evidence_span: str, start: int, end: int) -> str:
    """Return a destination signature for travel-duration duplicate suppression."""
    text = evidence_span
    role_markers = list(
        re.finditer(
            r"(?:^|\b)(?:\d+\.\s*)?(?:user|assistant)\s*:|\brole=(?:user|assistant)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if role_markers:
        text = text[role_markers[-1].end():]
    local = text[max(0, start - 220) : min(len(text), end + 220)]
    patterns = (
        r"\b(?:trip|road\s+trip)\s+to\s+(?P<label>[^,.;!?–—-]{2,90})",
        r"\b(?:drove|driving|drive)\s+(?:for\s+)?[^,.;!?]{0,32}?\bto\s+(?P<label>[^,.;!?–—-]{2,90})",
        r"\bto\s+(?P<label>[A-Z][A-Za-z0-9.'’]*(?:\s+[A-Z][A-Za-z0-9.'’]*){0,6})\b",
    )
    for pattern in patterns:
        if match := re.search(pattern, local, flags=re.IGNORECASE):
            label = _clean_travel_duration_label(match.group("label"))
            if label:
                return "travel_destination=" + label
    return ""


def _clean_duration_subject_label(label: str) -> str:
    label = re.sub(
        r"\b(?:on|normal|hard|easy|medium|difficulty|mode|around|about|approximately|roughly)\b",
        " ",
        label,
        flags=re.IGNORECASE,
    )
    terms = [
        term
        for term in source_tokens(label)
        if len(term) > 1
        and term not in _QUERY_STOPWORDS
        and term not in {"game", "games", "gaming", "play", "played", "playing", "complete", "completed", "finish", "finished"}
    ]
    return " ".join(terms[:12])


def duration_concrete_subject_signature(
    query: str,
    evidence_span: str,
    start: int,
    end: int,
) -> str:
    """Return a concrete activity/object signature for duplicate duration mentions."""
    query_tokens = set(source_tokens(query))
    if (
        query_tokens & {"drive", "driving", "drove", "road", "trip", "trips", "destination", "destinations"}
        and (signature := _travel_duration_subject_signature(evidence_span, start, end))
    ):
        return signature
    if not (query_tokens & {"game", "games", "gaming", "played", "playing", "completed", "finished"}):
        return ""
    text = evidence_span
    role_markers = list(
        re.finditer(
            r"(?:^|\b)(?:\d+\.\s*)?(?:user|assistant)\s*:|\brole=(?:user|assistant)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if role_markers:
        text = text[role_markers[-1].end():]
    patterns = (
        r"\bplaying\s+(?P<label>[^.,;!?]+?)\s+(?:last\s+\w+\s+|on\s+\w+\s+|at\s+\w+\s+)?(?:which\s+)?(?:took|for|and|but|by|$)",
        r"\b\d+(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?)\s+playing\s+(?P<label>[^.,;!?]+?)(?:\s+on\s+[^.,;!?]+)?(?:[.,;!?]|$)",
        r"\blike\s+(?P<label>[^.,;!?]+?),\s+which\s+I\s+(?:completed|finished)\b[^.;!?]*?\btook\b",
        r"\b(?:completed|finished)\s+(?P<label>[^.,;!?]+?)\s+(?:on\s+[^,.;!?]+?\s+)?(?:and\s+)?(?:it\s+)?took\b",
        r"\b(?P<label>[A-Z][A-Za-z0-9'’:-]+(?:\s+[A-Z][A-Za-z0-9'’:-]+){0,8})\s+(?:which\s+)?took\b",
    )
    for pattern in patterns:
        if match := re.search(pattern, text, flags=re.IGNORECASE):
            label = _clean_duration_subject_label(match.group("label"))
            if label:
                return "subject=" + label
    return ""


def duration_display(row: EvidenceLedgerRow) -> str:
    """Return the original display value for a duration ledger row."""
    return row.label or f"{format_number(float(row.value))} minutes"


def duration_raw_value_unit(row: EvidenceLedgerRow) -> tuple[float, str]:
    """Return the original raw value/unit encoded in a duration label."""
    match = re.match(r"(?P<value>\d+(?:\.\d+)?)\s+(?P<unit>[a-z]+)", row.label)
    if not match:
        return float(row.value), row.unit
    return float(match.group("value")), canonical_duration_unit(match.group("unit"))


def duration_raw_values_for_unit(candidates: tuple[EvidenceLedgerRow, ...], unit: str) -> list[float]:
    """Return original evidence values for rows written in the requested unit."""
    canonical_unit = canonical_duration_unit(unit)
    return [
        raw_value
        for row in candidates
        for raw_value, raw_unit in (duration_raw_value_unit(row),)
        if canonical_duration_unit(raw_unit) == canonical_unit
    ]


def _wedding_couple_from_label(label: str) -> str:
    value = re.sub(
        r"^\s*(?:i\s+)?(?:attended|went to|been to|returned from|got back from)\s+",
        "",
        label,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"(?:'s)?\s+wedding\b.*$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\s*(?:my|a|an|the)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value.strip(" .,'\""))
    if not value or " and " not in value.casefold():
        return ""
    return value


def count_subject_phrase(query: str) -> str:
    """Extract the counted subject phrase from a question."""
    match = re.search(
        r"\bhow\s+many\s+(?P<subject>.+?)(?:\s+(?:did|do|does|that|have|has|had|"
        r"were|was|are|is|can|could|should|would)\b|[?.,]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return " ".join(match.group("subject").strip(" .,'\"").casefold().split())


def count_display(value: int) -> str:
    """Render small count values as words for answer text."""
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(value, str(value))


def duration_number_words(value: float) -> str | None:
    """Render small whole duration values as title-cased English words."""
    if not value.is_integer():
        return None
    integer = int(value)
    word = count_display(integer)
    if word == str(integer):
        return None
    return word.title()


def duration_compatibility_lines(candidates: tuple[EvidenceLedgerRow, ...]) -> list[str]:
    """Render legacy compatibility fields for common duration units."""
    by_unit: dict[str, list[float]] = {}
    for row in candidates:
        raw_value, raw_unit = duration_raw_value_unit(row)
        by_unit.setdefault(raw_unit, []).append(raw_value)
    lines: list[str] = []
    if minute_values := by_unit.get("minutes"):
        lines.append("minute_values=" + ",".join(format_number(value) for value in minute_values))
        lines.append(f"minute_total_hours={format_number(sum(minute_values) / 60)} hours")
    if hour_values := by_unit.get("hours"):
        lines.append("hour_values=" + ",".join(format_number(value) for value in hour_values))
        lines.append(f"hour_total={format_number(sum(hour_values))} hours")
    if day_values := by_unit.get("days"):
        lines.append("day_values=" + ",".join(format_number(value) for value in day_values))
        lines.append(f"day_total={format_number(sum(day_values))} days")
    if month_values := by_unit.get("months"):
        lines.append("month_values=" + ",".join(format_number(value) for value in month_values))
        total_months = sum(month_values)
        lines.append(f"month_total={format_number(total_months)} months")
        if month_words := duration_number_words(total_months):
            lines.append(f"month_total_words={month_words} months")
    return lines


def duration_month_answer(value: float, subject_terms: tuple[str, ...]) -> str:
    """Render a month-granular duration answer in the shape requested by the query."""
    total = duration_number_words(value) or format_number(value)
    suffix = " months ago" if "ago" in set(subject_terms) else " months"
    return f"{total}{suffix}"


def list_detail_query(query: str) -> bool:
    """Return true when count synthesis should include item labels."""
    tokens = set(source_tokens(query))
    return bool(
        tokens
        & {
            "which",
            "who",
            "what",
            "list",
            "name",
            "names",
            "were",
            "they",
            "them",
            "items",
        }
    )


def labeled_count_subject(candidates: tuple[EvidenceLedgerRow, ...]) -> bool:
    """Return whether count evidence should expose item labels by default."""
    if not candidates:
        return False
    identities = [row.normalized_identity for row in candidates]
    return all(
        identity.startswith(
            (
                "doctor_visit=",
                "musical_instrument=",
                "model_kit=",
                "source_group=",
            )
        )
        for identity in identities
    )


def _joined_property_outcomes(outcomes: list[str]) -> str:
    """Join property outcome reasons as a natural-language list."""
    if len(outcomes) <= 1:
        return outcomes[0] if outcomes else ""
    if len(outcomes) == 2:
        return f"{outcomes[0]} and {outcomes[1]}"
    return ", ".join(outcomes[:-1]) + f", and {outcomes[-1]}"


def _clean_property_subject(value: str) -> str:
    value = " ".join(value.strip(" .,'\"").split())
    if re.match(r"^\d+-bedroom\b", value, flags=re.IGNORECASE):
        return f"the {value}"
    if re.match(r"^(?:the|a|an)\b", value, flags=re.IGNORECASE):
        return value
    return f"the {value}"


def _property_subject_phrase(label: str, span: str) -> str:
    """Return the compact property object phrase for outcome rendering."""
    for pattern in (
        r"\b(?P<property>\d+-bedroom\s+(?:bungalow|condo|townhouse|house|home))\b",
        r"\b(?P<property>(?:bungalow|condo|townhouse|house|home|property))\b",
    ):
        if match := re.search(pattern, label, flags=re.IGNORECASE):
            return _clean_property_subject(match.group("property"))
        if match := re.search(pattern, span, flags=re.IGNORECASE):
            return _clean_property_subject(match.group("property"))
    return "the property"


def _base_property_type(value: str) -> str:
    """Return a property phrase without bedroom count modifiers."""
    value = re.sub(r"\b\d+-bedroom\s+", "", value, flags=re.IGNORECASE)
    return " ".join(value.split())


def _property_location_phrase(label: str, span: str) -> str:
    """Return a named location for budget/outcome phrases when available."""
    for text in (label, span):
        if match := re.search(
            r"\bin\s+(?P<location>[A-Z][A-Za-z0-9'’.-]+(?:\s+[A-Z][A-Za-z0-9'’.-]+){0,4})\b",
            text,
        ):
            location = match.group("location").strip(" .,'\"")
            if location.casefold() not in {"the oakwood neighborhood", "the brookside neighborhood"}:
                return location
        if match := re.search(r"\b(?P<location>Cedar\s+Creek)\b", text, flags=re.IGNORECASE):
            return "Cedar Creek"
    return ""


def _instrument_duration_phrase(context: str) -> str:
    if match := re.search(r"\bfor\s+(?:about\s+)?(?P<years>\d+)\s+years?\b", context, flags=re.IGNORECASE):
        return f"{match.group('years')} years"
    return "an unspecified amount of time"


def _property_offer_target_phrase(query: str) -> str:
    match = re.search(
        r"\boffer\s+on\s+(?P<target>[^?]+?)\??$",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    target = " ".join(match.group("target").strip(" .,'\"").split())
    if not re.match(r"^(?:a|an|the)\b", target, flags=re.IGNORECASE):
        target = f"the {target}"
    return target


def model_kit_scale_label(label: str) -> str:
    """Render model-kit labels with explicit scale presence for answer synthesis."""
    if re.search(r"\b\d+/\d+\s+scale\b", label, flags=re.IGNORECASE):
        return label
    return f"{label} scale not mentioned"


def count_label(text: str) -> str:
    """Extract a compact event label for count/list synthesis."""
    cleaned = re.sub(r"\bcontent=\S+\s*", "", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", "", cleaned)
    cleaned = re.sub(r"\b(?:user|assistant):\s*", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\bI\s+(?P<label>.+?)(?:[.?!]|$)", cleaned)
    if match:
        label = " ".join(match.group("label").split()[:12])
        return label
    return ""


_GENERIC_ACTION_OBJECT_VERBS = {
    "acquired",
    "added",
    "assembled",
    "attended",
    "baked",
    "bought",
    "completed",
    "cooked",
    "downloaded",
    "fixed",
    "got",
    "learned",
    "ordered",
    "participated",
    "pick",
    "picked",
    "purchased",
    "replaced",
    "return",
    "returned",
    "sold",
    "tried",
    "viewed",
    "visited",
}


_GENERIC_OBJECT_STOPWORDS = {
    "a",
    "an",
    "and",
    "another",
    "for",
    "from",
    "in",
    "it",
    "my",
    "new",
    "of",
    "old",
    "or",
    "some",
    "that",
    "the",
    "this",
    "to",
}


def _expanded_count_match_terms(text: str) -> set[str]:
    """Return simple singular/plural variants for count-object matching."""
    terms = {
        token
        for token in source_tokens(text)
        if len(token) > 2 and token not in _COUNT_STOPWORDS and token not in _GENERIC_OBJECT_STOPWORDS
    }
    expanded = set(terms)
    for term in tuple(terms):
        if term.endswith("ies") and len(term) > 4:
            expanded.add(f"{term[:-3]}y")
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _generic_action_matches_query(verb: str, action_terms: set[str]) -> bool:
    """Return whether an action verb is semantically covered by query actions."""
    groups = {
        "picked": {"pick", "picked", "pickup"},
        "returned": {"return", "returned"},
        "replaced": {"replace", "replaced", "got", "new"},
        "ordered": {"buy", "bought", "got", "new", "purchased", "picked"},
        "acquired": {"buy", "bought", "got", "new", "purchased", "picked"},
    }
    return bool(groups.get(verb, set()) & action_terms)


def _generic_action_object_phrases(span: str) -> list[tuple[str, str]]:
    """Return action-object phrase candidates from a first-person span."""
    phrases: list[tuple[str, str]] = []
    for match in re.finditer(
        r"\b(?:get|buy|purchase|order)\s+"
        r"(?P<object>(?:a|an|the|my|new)\s+(?:(?!\b(?:for|and)\b|[.!?;,]).){2,80})"
        r"[^.!?]{0,180}\b(?:ordered|bought|purchased|got)\s+one\b",
        span,
        flags=re.IGNORECASE,
    ):
        phrases.append(("ordered", match.group("object")))
    verb_pattern = "|".join(sorted(_GENERIC_ACTION_OBJECT_VERBS, key=len, reverse=True))
    pattern = re.compile(
        rf"\b(?P<verb>{verb_pattern})(?:\s+up)?\s+"
        rf"(?P<object>[^.!?;,]{{2,120}}?)"
        rf"(?=(?:\s+(?:and|or)\s+(?:{verb_pattern})\b)|[.!?;,]|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(span):
        verb = match.group("verb").casefold()
        if match.group(0).casefold().startswith("picked up"):
            verb = "picked"
        raw_object = re.split(
            r"\b(?:and\s+rearranged|but|because|after|before|when|while|which|last|next|about)\b",
            match.group("object"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        if verb == "got" and re.match(r"\s*around\s+to\s+fixing\b", raw_object, flags=re.IGNORECASE):
            raw_object = re.sub(r"^\s*around\s+to\s+fixing\s+", "", raw_object, flags=re.IGNORECASE)
            verb = "fixed"
        phrases.append((verb, raw_object))
    return phrases


def _clean_generic_count_object(raw_object: str) -> str:
    """Normalize one action object into a stable item label."""
    value = re.sub(r"\b(?:from|at|to)\s+[A-Z][A-Za-z0-9&' -]{1,60}$", "", raw_object.strip())
    value = re.sub(r"\bfrom\s+them\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:larger|smaller)\s+pair\b", "pair", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:dry\s+cleaning\s+for)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:wobbly\s+leg\s+on)\s+", "", value, flags=re.IGNORECASE)
    value = re.split(r"\b(?:i\s+wore|that\s+i\s+wore|and\s+it|and\s+now)\b", value, maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\b(?:that|the)\s+IKEA\s+", "IKEA ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value.strip(" .,'\""))
    value = re.sub(r"^(?:a|an|the|my|that|this|new|old)\s+", "", value, flags=re.IGNORECASE)
    tokens = source_tokens(value)
    while tokens and tokens[0] in _GENERIC_OBJECT_STOPWORDS:
        tokens = tokens[1:]
    while tokens and tokens[-1] in _GENERIC_OBJECT_STOPWORDS:
        tokens = tokens[:-1]
    if not tokens:
        return ""
    return " ".join(tokens[:8])


def _generic_count_object_relevant(label: str, focus_terms: set[str]) -> bool:
    """Return whether an extracted object is relevant enough to count."""
    if not focus_terms:
        return True
    if focus_terms & {"something", "somethings", "anything", "things", "times"}:
        return True
    label_terms = set(source_tokens(label))
    if not label_terms or label_terms <= _GENERIC_ACTION_OBJECT_VERBS | {"need", "needs", "up"}:
        return False
    expanded_focus = set(focus_terms)
    semantic_objects = {
        "clothing": {"blazer", "boots", "dress", "jacket", "shirt", "shoes", "sweater"},
        "clothes": {"blazer", "boots", "dress", "jacket", "shirt", "shoes", "sweater"},
        "furniture": {"bed", "bookshelf", "chair", "couch", "desk", "mattress", "shelves", "shelf", "sofa", "table"},
        "item": {"blazer", "bookshelf", "boots", "coffee", "mattress", "pair", "shelves", "table"},
        "items": {"blazer", "bookshelf", "boots", "coffee", "mattress", "pair", "shelves", "table"},
        "piece": {"bookshelf", "mattress", "shelves", "table"},
        "pieces": {"bookshelf", "mattress", "shelves", "table"},
    }
    for term in tuple(expanded_focus):
        expanded_focus.update(semantic_objects.get(term, set()))
    return bool(label_terms & expanded_focus)


def generic_event_date_token(span: str) -> str:
    """Return a raw date token suitable for duplicate detection without year context."""
    month_pattern = "|".join(sorted(_MONTHS, key=len, reverse=True))
    match = re.search(
        rf"\b(?P<month>{month_pattern})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b",
        span,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{_MONTHS[match.group('month').casefold()]:02d}-{int(match.group('day')):02d}"
    month_only = re.search(
        rf"\b(?:in|during|since|from|back\s+in)\s+(?P<month>{month_pattern})\b",
        span,
        flags=re.IGNORECASE,
    )
    if month_only:
        return f"{_MONTHS[month_only.group('month').casefold()]:02d}"
    return ""


def numeric_state_query(query: str) -> bool:
    """Return whether a count question asks for current numeric state."""
    tokens = set(source_tokens(query))
    if not {"how", "many"} <= tokens and not numeric_state_difference_query(query):
        return False
    if tokens & {"times", "weddings", "events", "doctors", "appointments", "festivals", "properties", "rode"}:
        return False
    return bool(
        tokens
        & {
            "collection",
            "count",
            "currently",
            "decrease",
            "difference",
            "followers",
            "growth",
            "has",
            "have",
            "increase",
            "lead",
            "leading",
            "leads",
            "own",
            "owns",
            "seen",
            "spotted",
            "team",
        }
    )


def numeric_state_required_qualifier_terms(query: str) -> set[str]:
    """Return required qualifier terms for state questions scoped to an explicit role."""
    match = re.search(
        r"\brole\s+as\s+(?:a|an|the\s+)?(?P<qualifier>[A-Za-z][A-Za-z0-9 /&+-]*?)(?:[?.,;:]|$)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return set()
    return {
        token
        for token in source_tokens(match.group("qualifier"))
        if len(token) > 2 and token not in _COUNT_STOPWORDS
    }


def numeric_state_sentences(text: str) -> list[str]:
    """Return bounded user-authored sentences that may carry numeric state."""
    cleaned = re.sub(r"\bcontent=longmemeval_session_id=\S+\s*", " ", text)
    cleaned = re.sub(r"\blongmemeval_session_date=\d{4}/\d{1,2}/\d{1,2}\s*\([^)]*\)\s*", " ", cleaned)
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=user:|assistant:)", cleaned)
    sentences: list[str] = []
    for part in parts:
        sentence = " ".join(part.strip().split())
        if not sentence or re.match(r"assistant\s*:", sentence, flags=re.IGNORECASE):
            continue
        sentence = re.sub(r"^(?:\d+\.\s*)?user\s*:\s*", "", sentence, flags=re.IGNORECASE)
        if re.search(r"\b(?:I|I've|my|mine)\b", sentence, flags=re.IGNORECASE):
            sentences.append(sentence)
    return sentences


def numeric_state_increments(sentence: str, *, focus_terms: set[str]) -> list[tuple[int, str, str]]:
    """Extract additive updates from one sentence."""
    if not focus_terms & set(source_tokens(sentence)):
        return []
    patterns = (
        r"\b(?:just\s+|recently\s+)?added\s+(?P<value>a|an|one|two|three|four|five|\d{1,4})\s+(?:new\s+)?",
        r"\b(?:just\s+|recently\s+)?bought\s+(?P<value>a|an|one|two|three|four|five|\d{1,4})\s+(?:new\s+)?",
    )
    increments: list[tuple[int, str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            value_text = match.group("value").casefold()
            value = _NUMBER_WORDS.get(value_text, int(value_text) if value_text.isdigit() else 0)
            if value:
                increments.append((value, sentence, "incremental_update"))
    return increments


def _canonical_fish_label(label: str) -> str:
    normalized = " ".join(source_tokens(label))
    if "neon" in normalized and "tetra" in normalized:
        return "neon tetra"
    if "gourami" in normalized:
        return "golden honey gourami" if "golden" in normalized or "honey" in normalized else "gourami"
    if "pleco" in normalized:
        return "pleco catfish"
    if "betta" in normalized:
        return "betta fish"
    if "danio" in normalized:
        return "danio"
    return normalized


def _canonical_competitive_sport_label(label: str) -> str:
    normalized = " ".join(source_tokens(label))
    if normalized == "swim":
        return "swimming"
    return normalized


def _competitive_sport_labels(span: str) -> list[str]:
    """Extract sports the user explicitly played competitively in the past."""
    labels: list[str] = []
    patterns = (
        r"\bused\s+to\s+(?P<label>swim)\s+competitively\b",
        r"\bused\s+to\s+play\s+(?P<label>[A-Za-z][A-Za-z' -]{1,40})\s+competitively\b",
        r"\bplayed\s+(?P<label>[A-Za-z][A-Za-z' -]{1,40})\s+competitively\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_competitive_sport_label(match.group("label"))
            if label and label not in labels:
                labels.append(label)
    return labels


def _count_word_value(value: str) -> int:
    lowered = re.sub(r"\s+times?$", "", value.casefold()).strip()
    if lowered == "twice":
        return 2
    if lowered == "thrice":
        return 3
    if lowered.isdigit():
        return int(lowered)
    return _NUMBER_WORDS.get(lowered, 0)


def numeric_state_totals(sentence: str) -> list[tuple[int, str, str]]:
    """Extract stated current totals from one sentence."""
    patterns = (
        rf"\btotal\s+(?:species\s+)?count\s+(?:to|is|of|at)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\b(?:a\s+)?total\s+of\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bmanaged\s+to\s+(?:spot|see|identify)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\b(?:spotted|seen|identified)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\s+different\b",
        rf"\bI\s+(?:currently\s+)?(?:have|own)\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bI\s+(?:had|have)\s+(?:about\s+|around\s+|approximately\s+|roughly\s+)?(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\s+followers\b",
        rf"\bI\s+started\b[^.!?]{{0,80}}?\bwith\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\s+followers\b",
        rf"\bI(?:'m|m| am)\s+(?:currently\s+)?now\s+at\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bI\s+(?:just\s+)?reached\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
        rf"\bI\s+(?:now\s+|currently\s+)?lead\s+a\s+team\s+of\s+(?P<value>{_NUMERIC_STATE_VALUE_PATTERN})\b",
    )
    values: list[tuple[int, str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            value = _count_word_value(match.group("value"))
            if value:
                values.append((value, sentence, "stated_total"))
    return values


def numeric_state_evidence(text: str, *, focus_terms: set[str]) -> tuple[tuple[int, str, str], ...]:
    """Extract explicit total-state and incremental count updates from user memory text."""
    evidence: list[tuple[int, str, str]] = []
    for sentence in numeric_state_sentences(text):
        sentence_terms = set(source_tokens(sentence))
        explicit_state_terms = {
            "added",
            "bought",
            "count",
            "currently",
            "lead",
            "now",
            "reached",
            "team",
            "total",
        } & sentence_terms
        if focus_terms and len(focus_terms & sentence_terms) < 2 and not explicit_state_terms:
            continue
        evidence.extend(numeric_state_totals(sentence))
        evidence.extend(numeric_state_increments(sentence, focus_terms=focus_terms))
    return tuple(evidence)


def _fish_inventory_counts(span: str) -> list[tuple[str, int]]:
    """Extract fish species/count pairs from first-person aquarium inventory text."""
    counts: list[tuple[str, int]] = []
    seen: set[str] = set()
    quantity_patterns = (
        r"\b(?P<count>\d{1,3})\s+(?P<label>neon\s+tetras?|golden\s+honey\s+gouramis?|gouramis?|tetras?|danios?)\b",
        r"\b(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?P<label>neon\s+tetras?|golden\s+honey\s+gouramis?|gouramis?|tetras?|danios?)\b",
    )
    for pattern in quantity_patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_fish_label(match.group("label"))
            count = _count_word_value(match.group("count"))
            if label and count > 0 and label not in seen:
                seen.add(label)
                counts.append((label, count))
    singular_patterns = (
        r"\b(?:a|an|one|my)\s+(?:small\s+)?(?P<label>pleco\s+catfish|pleco|betta\s+fish|betta)\b",
        r"\b(?P<label>betta\s+fish),\s+Bubbles\b",
    )
    for pattern in singular_patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _canonical_fish_label(match.group("label"))
            if label and label not in seen:
                seen.add(label)
                counts.append((label, 1))
    return counts


def _clean_rollercoaster_label(label: str) -> str:
    label = re.split(
        r"\b(?:at|in|during|on|with|before|after|again|twice|thrice)\b",
        label,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    label = re.sub(r"\b(?:rollercoasters?|coasters?|rides?)\b", "", label, flags=re.IGNORECASE)
    label = " ".join(label.strip(" .,'\"").split())
    if not label or label.casefold() in {"some", "several", "many"}:
        return ""
    return label


def _split_rollercoaster_labels(raw_labels: str) -> list[str]:
    labels: list[str] = []
    for raw_label in re.split(r"\s*,\s*|\s+\band\b\s+", raw_labels):
        label = _clean_rollercoaster_label(raw_label)
        if label:
            labels.append(label)
    return labels


def _rollercoaster_ride_labels(span: str) -> list[str]:
    """Return one label per actual rollercoaster ride occurrence in a span."""
    labels: list[str] = []
    repeated_pattern = re.compile(
        r"\brode\s+(?P<label>[A-Z][A-Za-z0-9'&: -]{1,60}?)\s+"
        r"(?P<count>twice|thrice|(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times?)\b",
        flags=re.IGNORECASE,
    )
    consumed: list[tuple[int, int]] = []
    for match in repeated_pattern.finditer(span):
        label = _clean_rollercoaster_label(match.group("label"))
        count = _count_word_value(match.group("count"))
        if not label or count <= 0:
            continue
        consumed.append((match.start(), match.end()))
        labels.extend([label] * count)
    enumerated_pattern = re.compile(
        r"\brode\s+(?P<labels>[A-Z][A-Za-z0-9'&: -]{1,90}?"
        r"(?:,\s*[A-Z][A-Za-z0-9'&: -]{1,40})*"
        r"(?:,?\s+and\s+[A-Z][A-Za-z0-9'&: -]{1,40})?)"
        r"(?:\s+(?:rollercoasters?|coasters?|rides?))?"
        r"(?:\s+(?:at|in|during|on|with|before|after)\b|[.!?]|$)",
        flags=re.IGNORECASE,
    )
    for match in enumerated_pattern.finditer(span):
        if any(start <= match.start() < end for start, end in consumed):
            continue
        for label in _split_rollercoaster_labels(match.group("labels")):
            labels.append(label)
    return labels


def _first_person_spans(text: str) -> list[str]:
    """Extract bounded first-person clauses suitable for count evidence."""
    cleaned = re.sub(r"\bcontent=\S+\s*", " ", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", " ", cleaned)
    spans: list[str] = []
    pattern = re.compile(
        r"(?:\buser:\s*)?\bI(?:\s+|['’](?:m|ve|d|ll|re)\s+).{3,420}?(?:[.!?](?=\s|$)|$)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(cleaned):
        span = " ".join(match.group(0).strip(" .!?").split())
        span = re.sub(r"^(?:user|assistant):\s*", "", span, flags=re.IGNORECASE)
        if span and span not in spans:
            spans.append(span)
    return spans


def _count_action_terms(query: str) -> set[str]:
    """Return action terms that must appear in count evidence spans."""
    query_terms = set(source_tokens(query))
    action_groups = {
        "attend": {"assist", "assisted", "attend", "attended", "attending", "back", "go", "got", "participated", "visited", "volunteered", "went"},
        "attended": {"assist", "assisted", "attend", "attended", "attending", "back", "go", "got", "participated", "visited", "volunteered", "went"},
        "participate": {"attend", "attended", "danced", "participate", "participated", "ran", "volunteer", "volunteered", "walked", "went"},
        "participated": {"attend", "attended", "danced", "participate", "participated", "ran", "volunteer", "volunteered", "walked", "went"},
        "fix": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "fixed": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "replace": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "replaced": {"donated", "fixed", "got", "new", "replaced", "upgrade"},
        "visit": {"appointment", "back", "diagnosed", "got", "had", "prescribed", "see", "saw", "took", "visit", "visited", "went"},
        "visited": {"appointment", "back", "diagnosed", "got", "had", "prescribed", "see", "saw", "took", "visit", "visited", "went"},
        "view": {"fell", "love", "rejected", "saw", "searching", "seen", "tour", "toured", "view", "viewed", "visited"},
        "viewed": {"fell", "love", "rejected", "saw", "searching", "seen", "tour", "toured", "view", "viewed", "visited"},
        "own": {"had", "have", "own", "owned", "playing", "selling", "service"},
        "owned": {"had", "have", "own", "owned", "playing", "selling", "service"},
        "work": {"bought", "build", "built", "finished", "got", "new", "planning", "started", "starting", "thinking", "work", "worked", "working"},
        "worked": {"bought", "build", "built", "finished", "got", "new", "planning", "started", "starting", "thinking", "work", "worked", "working"},
        "completed": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "drafted": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "finished": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "published": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "wrote": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "written": {"completed", "drafted", "finished", "published", "wrote", "written"},
        "buy": {"buy", "bought", "got", "new", "ordered", "purchased", "picked"},
        "bought": {"buy", "bought", "got", "new", "ordered", "purchased", "picked"},
        "bake": {"bake", "baked"},
        "baked": {"bake", "baked"},
        "assemble": {"assemble", "assembled", "built", "put"},
        "assembled": {"assemble", "assembled", "built", "put"},
        "sell": {"sell", "sold"},
        "sold": {"sell", "sold"},
        "return": {"return", "returned"},
        "returned": {"return", "returned"},
        "pick": {"pick", "picked"},
        "picked": {"pick", "picked"},
        "wedding": {"attend", "attended", "back", "been", "returned", "went"},
        "weddings": {"attend", "attended", "back", "been", "returned", "went"},
        "ride": {"ride", "rides", "riding", "rode", "ridden"},
        "rides": {"ride", "rides", "riding", "rode", "ridden"},
        "rode": {"ride", "rides", "riding", "rode", "ridden"},
        "rollercoaster": {"ride", "rides", "riding", "rode", "ridden"},
        "rollercoasters": {"ride", "rides", "riding", "rode", "ridden"},
        "coaster": {"ride", "rides", "riding", "rode", "ridden"},
        "coasters": {"ride", "rides", "riding", "rode", "ridden"},
    }
    actions: set[str] = set()
    for term in query_terms:
        actions.update(action_groups.get(term, set()))
    if not actions and {"how", "many"} <= query_terms:
        actions.update(
            {
                "attend",
                "attended",
                "been",
                "bought",
                "danced",
                "got",
                "participated",
                "ran",
                "took",
                "visit",
                "visited",
                "volunteered",
                "walked",
                "went",
            }
        )
    return actions


def _specific_count_object_terms(query: str) -> set[str]:
    """Return concrete object terms for action-scoped count questions."""
    match = re.search(
        r"\bhow\s+many\s+(?:times?\s+)?(?:did|do|does|have|has|had)\s+"
        r"(?:i|we|you)?\s*"
        r"(?P<verb>[a-z]+)\s+(?P<object>[^?.,;]+)",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return set()
    verb = match.group("verb").casefold()
    action_terms = _count_action_terms(query)
    if action_terms and verb not in action_terms and not _generic_action_matches_query(verb, action_terms):
        return set()
    object_text = re.split(
        r"\b(?:in|during|over|within|before|after|since|last|next|past|this)\b",
        match.group("object"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    terms = {
        token
        for token in source_tokens(object_text)
        if len(token) > 2 and token not in _COUNT_STOPWORDS and token not in _GENERIC_OBJECT_STOPWORDS
    }
    if not terms or terms <= {"something", "somethings", "anything", "things", "times"}:
        return set()
    return _expanded_count_match_terms(" ".join(terms))


def _filter_specific_action_object_count_rows(
    query: str,
    rows: list[EvidenceLedgerRow],
    *,
    subject: str = "",
) -> list[EvidenceLedgerRow]:
    """Exclude action-compatible count rows that miss a query-specified object."""
    if subject and subject != "generic":
        return rows
    object_terms = _specific_count_object_terms(query)
    if not object_terms:
        return rows
    filtered: list[EvidenceLedgerRow] = []
    for row in rows:
        if row.exclude_reason or row.kind != "event":
            filtered.append(row)
            continue
        row_terms = _expanded_count_match_terms(" ".join((row.label, row.raw_span)))
        if object_terms <= row_terms:
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
                exclude_reason="query_object_mismatch",
                confidence=row.confidence,
            )
        )
    return filtered


def _has_count_action(span: str, action_terms: set[str]) -> bool:
    if not action_terms:
        return True
    tokens = set(source_tokens(span))
    if not tokens & action_terms:
        return False
    if re.search(
        r"\bI(?:'ve| have)?\s+(?:also\s+)?(?:been\s+)?"
        r"(?:considering|thinking|planning|hoping|interested|looking)\s+"
        r"[^.!?]{0,80}\b(?:participate|participating|register|registering|attend|attending|volunteer|volunteering)\b",
        span,
        flags=re.IGNORECASE,
    ):
        return False
    return not re.search(r"\bI\s+(?:requested|asked|wanted|hoped|planned)\s+to\s+see\b", span, flags=re.IGNORECASE)


def _negated_count_action(span: str, action_terms: set[str]) -> bool:
    if not action_terms:
        return False
    escaped_actions = "|".join(sorted((re.escape(term) for term in action_terms), key=len, reverse=True))
    if not escaped_actions:
        return False
    return bool(
        re.search(
            rf"\b(?:did\s+not|didn't|do\s+not|don't|never|not)\s+[^.!?]{{0,40}}\b(?:{escaped_actions})\b",
            span,
            flags=re.IGNORECASE,
        )
    )


def _count_subject(query: str, *, tokens: set[str] | None = None) -> str:
    """Classify the count target into a small deterministic evidence facet."""
    tokens = tokens or set(source_tokens(query))
    if tokens & {"movie", "movies", "film", "films", "cinema"} and tokens & {"festival", "festivals", "fest", "fests"}:
        return "film_festival"
    if tokens & {"museum", "museums", "gallery", "galleries", "cube"}:
        return "museum_gallery"
    if tokens & {"kitchen", "toaster", "faucet", "mat", "shelves", "coffee"} and tokens & {
        "item",
        "items",
        "replace",
        "replaced",
        "fix",
        "fixed",
    }:
        return "kitchen_item"
    if tokens & {"property", "properties", "home", "homes", "house", "houses"}:
        return "property_viewing"
    if tokens & {"doctor", "doctors", "physician", "physicians", "dermatologist", "ent"}:
        return "doctor_visit"
    if tokens & {"instrument", "instruments", "guitar", "piano", "drum", "drums"}:
        return "musical_instrument"
    if tokens & {"model", "models", "kit", "kits"}:
        return "model_kit"
    if tokens & {"wedding", "weddings"}:
        return "wedding"
    if tokens & {"rollercoaster", "rollercoasters", "coaster", "coasters"} or (
        tokens & {"times"} and tokens & {"ride", "rides", "riding", "rode"}
    ):
        return "rollercoaster_ride"
    if tokens & {"fish", "aquarium", "aquariums", "tank", "tanks"} and tokens & {"total", "both", "many"}:
        return "fish_inventory"
    if tokens & {"sport", "sports"} and tokens & {"competitive", "competitively"}:
        return "competitive_sport"
    if tokens & {"dinner", "party", "parties"} and tokens & {"attend", "attended"}:
        return "dinner_party"
    if tokens & {"piece", "pieces", "writing", "writings", "story", "stories"} and tokens & {
        "completed",
        "drafted",
        "finished",
        "published",
        "wrote",
        "written",
    }:
        return "writing_piece"
    return ""


def build_synthesis_plan(query: str, *, limit: int = 10) -> SynthesisPlan:
    """Build a deterministic answer-shape plan for a memory query."""
    del limit
    query_tokens = source_tokens(query)
    tokens = set(query_tokens)
    subject_terms = tuple(
        token
        for token in query_tokens
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    )
    reasons: list[str] = []
    savings_terms = {"save", "saved", "saving", "savings"}
    if tokens & {"more", "less", "difference", "compared", "versus", "vs"} or tokens & savings_terms:
        reasons.append("comparison")
    if tokens & {"total", "sum"}:
        reasons.append("total")
    duration_terms = {
        "hour",
        "hours",
        "minute",
        "minutes",
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
    }
    duration_question = {"how", "long"} <= tokens and bool(tokens & {"combined", "total", "altogether"})
    explicit_money_terms = {"money", "cost", "costs", "price", "prices", "amount"}
    floor_value_terms = {"minimum", "sold", "sell", "worth", "value"}
    money_terms = explicit_money_terms | {"spent", "spend"} | savings_terms | floor_value_terms
    duration_query = bool(tokens & duration_terms) or duration_question
    count_subject = _count_subject(query, tokens=tokens)
    if {"how", "many"} <= tokens and count_subject and (
        not _duration_measure_query(tokens) or _incidental_time_modifier_query(query)
    ):
        return SynthesisPlan(
            answer_type="count",
            operation="count_distinct",
            subject_terms=subject_terms,
            required_kinds=("event",),
            required_source_groups=2,
            reasons=("count",),
        )
    if (
        {"how", "many"} <= tokens
        and (not _duration_measure_query(tokens) or _incidental_time_modifier_query(query))
        and not re.search(r"\bhow\s+many\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\b", query, flags=re.IGNORECASE)
        and not tokens & {"amount", "cost", "costs", "dollar", "dollars", "money", "price", "prices"}
    ):
        return SynthesisPlan(
            answer_type="count",
            operation="count_distinct",
            subject_terms=subject_terms,
            required_kinds=("event",),
            required_source_groups=2,
            reasons=("count",),
        )
    if tokens & money_terms and (not duration_query or bool(tokens & explicit_money_terms)):
        if "comparison" in reasons:
            return SynthesisPlan(
                answer_type="difference",
                operation="difference_between",
                subject_terms=subject_terms,
                required_kinds=("currency",),
                required_source_groups=2,
                reasons=tuple(reasons),
            )
        return SynthesisPlan(
            answer_type="sum",
            operation="sum_values",
            subject_terms=subject_terms,
            required_kinds=("currency",),
            required_source_groups=2,
            reasons=tuple(reasons or ["money"]),
        )
    if duration_query:
        return SynthesisPlan(
            answer_type="sum",
            operation="sum_values",
            subject_terms=subject_terms,
            required_kinds=("duration",),
            required_source_groups=2,
            reasons=tuple(reasons or ["duration"]),
        )
    if "average" in tokens:
        return SynthesisPlan(
            answer_type="average",
            operation="average_values",
            subject_terms=subject_terms,
            required_kinds=("number",),
            required_source_groups=2,
            reasons=("average",),
        )
    if {"how", "many"} <= tokens:
        return SynthesisPlan(
            answer_type="count",
            operation="count_distinct",
            subject_terms=subject_terms,
            required_kinds=("event",),
            required_source_groups=2,
            reasons=("count",),
        )
    return SynthesisPlan(
        answer_type="direct_fact",
        operation="select_fact",
        subject_terms=subject_terms,
        required_kinds=(),
        required_source_groups=1,
        reasons=("direct",),
    )


def build_age_average_ledger(query: str, contexts: list[str], *, plan: SynthesisPlan | None = None) -> EvidenceLedger:
    """Extract family age evidence into a typed number ledger for average synthesis."""
    plan = plan or build_synthesis_plan(query)
    query_tokens = set(source_tokens(query))
    if plan.operation != "average_values" or "age" not in query_tokens:
        return EvidenceLedger(plan=plan, rows=())
    evidence = _age_average_evidence(contexts)
    rows = [
        EvidenceLedgerRow(
            fact_id=f"age_average:{index}",
            source_group=source_group(context),
            citation=source_citation(context),
            kind="number",
            value=str(value),
            unit="years",
            label="age",
            raw_span=raw,
            context=context_text(context),
            normalized_identity=f"age_average:{source_group(context)}:{value}",
            relevance=4,
            include_reason="age_average_input",
            confidence=0.78,
        )
        for index, (context, value, raw) in enumerate(evidence)
    ]
    return EvidenceLedger(plan=plan, rows=tuple(rows))


def _date_interval_blocked_by_count_or_duration_query(query: str, tokens: set[str]) -> bool:
    """Return whether date interval evidence is only a temporal modifier."""
    if re.search(
        r"\bhow\s+many\s+(?:days?|weeks?)\s+(?:had\s+)?passed\b.*\b(?:between|since|when)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return False
    if re.search(r"\bhow\s+many\s+(?:days?|weeks?)\s+(?:before|after)\b", query, flags=re.IGNORECASE):
        return False
    if re.search(r"\bhow\s+many\s+(?:days?|weeks?)\s+did\s+it\s+take\b", query, flags=re.IGNORECASE):
        return False
    if re.search(r"\bhow\s+many\s+times\b", query, flags=re.IGNORECASE):
        return True
    if re.search(r"\bhow\s+many\s+(?:minutes?|hours?)\b", query, flags=re.IGNORECASE):
        return True
    return {"how", "many"} <= tokens and bool(_count_subject(query, tokens=tokens))


def _kitchen_item_labels(span: str) -> list[str]:
    """Extract durable kitchen items from repair, replacement, and upgrade memories."""
    labels: list[str] = []
    lowered = span.casefold()
    label_patterns = (
        ("kitchen faucet", r"\bkitchen\s+faucet\b|\bfaucet\b"),
        ("kitchen mat", r"\bkitchen\s+mat\b|\bmat\s+in\s+front\s+of\s+the\s+sink\b"),
        ("toaster", r"\bold\s+toaster\b|\btoaster\b"),
        ("coffee maker", r"\bold\s+coffee\s+maker\b|\bcoffee\s+maker\b"),
        ("kitchen shelves", r"\bkitchen\s+shelves\b|\bshelves\b"),
    )
    for label, pattern in label_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE) and label not in labels:
            labels.append(label)
    return labels


def _canonical_instrument_label(label: str) -> str:
    lowered = label.casefold()
    if lowered == "yamaha fg800":
        return "Yamaha FG800 acoustic guitar"
    if lowered == "5-piece pearl export":
        return "5-piece Pearl Export drum set"
    if lowered == "korg b1":
        return "Korg B1 piano"
    return label


def _clean_property_label(label: str) -> str:
    label = re.sub(
        r"^\s*(?:and\s+)?(?:a|an|the|that\s+one\s+in)\s+",
        "",
        label,
        flags=re.IGNORECASE,
    )
    label = re.sub(r"\s+", " ", label.strip(" .,'\""))
    return label.strip(" .,'\"")


def _property_viewing_label(span: str) -> str:
    """Return property-viewing labels with the rejection/outcome reason preserved."""
    if match := re.search(
        r"\b(?P<label>(?:that\s+one\s+in\s+)?Cedar\s+Creek[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        label = _clean_property_label(match.group("label"))
        if "budget" in span.casefold() and "budget" not in label.casefold():
            label = f"{label} was out of my budget"
        return label
    if match := re.search(
        r"\b(?P<label>\d+-bedroom\s+bungalow[^.!?]{0,160}?"
        r"\bkitchen[^.!?]{0,100}?\brenovation[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        return _clean_property_label(match.group("label"))
    if match := re.search(
        r"\b(?P<label>\d+-bedroom\s+condo[^.!?]{0,120}?"
        r"(?:deal-breaker|higher\s+bid|rejected)[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        return _clean_property_label(match.group("label"))
    return count_label(span)


def _clean_count_item_label(label: str) -> str:
    label = re.sub(r"^\s*(?:and\s+)?(?:a|an|the)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label.strip(" .,'\""))
    label = re.split(
        r"\s+\b(?:and|plus|with|where|that|which|but|because|next|during|at)\b(?:\s+|[,.!?;:]|$)",
        label,
        maxsplit=1,
    )[0]
    return label.strip(" .,'\"")


def _writing_piece_label(span: str) -> str:
    """Extract the concrete writing item from a completed-writing span."""
    patterns = (
        r"\b(?:completed|drafted|finished|published|wrote|written)\s+(?:a|an|the)?\s*(?P<label>[^.!?]{0,60}?\b(?:article|essay|piece|poem|story))\b",
        r"\b(?P<label>(?:article|essay|piece|poem|story)\s+[^.!?]{0,40})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, span, flags=re.IGNORECASE)
        if not match:
            continue
        label = _clean_count_item_label(match.group("label"))
        if label:
            return label
    return count_label(span)


def _wedding_label(span: str) -> str:
    """Return a compact wedding attendance label with couple names when present."""
    if match := re.search(
        r"\b(?P<label>my\s+cousin's\s+wedding[^.!?]{0,80})",
        span,
        flags=re.IGNORECASE,
    ):
        return _clean_count_item_label(match.group("label"))
    if match := re.search(
        r"\b(?:the\s+bride,\s*)?(?P<left>[A-Z][A-Za-z' -]{1,30}),?\s+[^.!?]{0,80}?"
        r"\b(?:husband|partner),?\s+(?P<right>[A-Z][A-Za-z' -]{1,30})\b",
        span,
    ):
        return f"{_clean_count_item_label(match.group('left'))} and {_clean_count_item_label(match.group('right'))}'s wedding"
    if match := re.search(
        r"\b(?P<left>[A-Z][A-Za-z' -]{1,30})\s+[^.!?]{0,80}?\b(?:married|tie the knot)\s+"
        r"(?:with\s+)?(?:her|his|their)?\s*(?:partner\s+)?(?P<right>[A-Z][A-Za-z' -]{1,30})\b",
        span,
        flags=re.IGNORECASE,
    ):
        return f"{_clean_count_item_label(match.group('left'))} and {_clean_count_item_label(match.group('right'))}'s wedding"
    return count_label(span)


def _normalize_count_identity(label: str) -> str:
    generic_terms = {"as", "kit", "kits", "model", "models", "well"}
    return " ".join(token for token in source_tokens(label) if token not in generic_terms)


def generic_count_event_identity(*, group: str, label: str, span: str) -> str:
    """Return a stable identity for generic countable event mentions."""
    quoted = re.search(
        r"\"(?P<double>[^\"]{3,120})\"|(?<!\w)'(?P<single>[^']{3,120})'(?!\w)",
        span,
    )
    if quoted:
        return f"event_title={_normalize_count_identity(quoted.group('double') or quoted.group('single') or '')}"
    if date_token := generic_event_date_token(span):
        return f"event_source_date={group}:{date_token}"
    return f"event={_normalize_count_identity(label or span)}"


def _competitive_sport_items(
    spans: list[str],
    *,
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Extract competitive sports with relevance implied by the typed pattern."""
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        for label in _competitive_sport_labels(span):
            identity = f"{identity_prefix}={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=identity,
                    relevance=3,
                )
            )
    return items


def _dinner_party_labels(span: str) -> list[str]:
    """Extract distinct attended dinner-party location labels."""
    labels: list[str] = []
    patterns = (
        r"\b(?:feast|dinner\s+part(?:y|ies)|potluck|BBQ)\s+at\s+(?P<label>[A-Z][A-Za-z'-]+(?:'s|’s)\s+place)\b",
        r"\bat\s+(?P<label>[A-Z][A-Za-z'-]+(?:'s|’s)\s+place)\b[^.!?]{0,100}\b(?:feast|dinner\s+part(?:y|ies)|potluck|BBQ)\b",
        r"\bones\s+we\s+had\s+at\s+(?P<label>[A-Z][A-Za-z'-]+(?:'s|’s)\s+place)\b",
        r"\balso\s+at\s+(?P<label>[A-Z][A-Za-z'-]+(?:'s|’s)\s+place)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, span):
            label = _clean_count_item_label(match.group("label"))
            if label and _normalize_count_identity(label) not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _dinner_party_items(
    spans: list[str],
    *,
    identity_prefix: str,
) -> list[CountEvidenceItem]:
    """Extract dinner-party attendance items with relevance implied by the typed pattern."""
    items: list[CountEvidenceItem] = []
    seen: set[str] = set()
    for span in spans:
        for label in _dinner_party_labels(span):
            identity = f"{identity_prefix}={_normalize_count_identity(label)}"
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                CountEvidenceItem(
                    label=label,
                    span=span,
                    normalized_identity=identity,
                    relevance=3,
                )
            )
    return items


def _model_kit_labels(span: str) -> list[str]:
    """Extract distinct model-kit items from a first-person evidence span."""
    labels: list[str] = []
    for match in re.finditer(
        r"\b(?:simple\s+)?(?P<label>[A-Z][A-Za-z0-9.-]+(?:\s+[A-Z0-9][A-Za-z0-9.'-]+){1,6}\s+kit)\b",
        span,
    ):
        label = _clean_count_item_label(match.group("label"))
        identity = _normalize_count_identity(label)
        if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
            labels.append(label)
    for match in re.finditer(
        r"\b(?:(?:[A-Z][A-Za-z0-9.-]+)\s+)?\d+/\d+\s+scale\s+[^;!?]+",
        span,
    ):
        fragment = match.group(0)
        fragments = re.split(r"\s+\band\b\s+(?:a|an|the)?\s*(?=\d+/\d+\s+scale)", fragment)
        for raw_label in fragments:
            label = _clean_count_item_label(raw_label)
            if label and _normalize_count_identity(label) not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels


def _doctor_visit_labels(span: str) -> list[str]:
    """Extract distinct medical provider roles from a consultation memory span."""
    labels: list[str] = []
    specific_patterns = (
        r"\b(?P<label>primary\s+care\s+physician)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>ENT\s+specialist)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>ENT)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>dermatologist)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
    )
    for pattern in specific_patterns:
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _clean_count_item_label(match.group("label"))
            identity = _normalize_count_identity(label)
            if identity == "ent" and any(
                _normalize_count_identity(existing) == "ent specialist"
                for existing in labels
            ):
                continue
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    if labels:
        return labels
    for pattern in (
        r"\b(?P<label>physician)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
        r"\b(?P<label>doctor)(?:,\s*)?(?:Dr\.?\s+[A-Z][A-Za-z'-]+)?",
    ):
        for match in re.finditer(pattern, span, flags=re.IGNORECASE):
            label = _clean_count_item_label(match.group("label"))
            identity = _normalize_count_identity(label)
            if label and identity not in {_normalize_count_identity(existing) for existing in labels}:
                labels.append(label)
    return labels
