"""Split from retrieval_plan.py (mechanical decomposition)."""


from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass

from zaxy.retrieval_plan.foundations import (
    _FIRST_PERSON_CONTEXT_RE,
    _QUERY_SOURCE_STOPWORDS,
    _quoted_query_title,
    source_context_group,
    source_context_snippet,
    source_tokens,
)
from zaxy.retrieval_plan.scalars import (
    _boolean_evidence_sentences,
    _query_bound_direct_answer_query,
    _query_bound_scalar_spec,
    _query_overlap_score,
    _query_specific_terms,
    _QueryBoundScalarSpec,
)


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


def _clean_direct_fact_value(value: str) -> str:
    value = re.split(r"\b(?:because|but|although|while|whereas)\b", value, maxsplit=1)[0]
    return " ".join(value.strip(" .,'\"()").split())


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


def _clean_scalar_state_value(value: str) -> str:
    value = re.split(
        r"\b(?:because|but|although|while|whereas|after|when|since)\b",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(value.strip(" .,'\"()").split())


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


def _clean_state_answer(value: str) -> str:
    return " ".join(value.strip(" .,'\"").split())


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


def _routine_time_slots(query: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return requested routine-duration slots from a query."""
    tokens = set(source_tokens(query))
    slots: list[tuple[str, tuple[str, ...]]] = []
    if tokens & {"ready", "breakfast", "morning", "routine"}:
        slots.append(("ready", ("ready", "morning", "routine", "breakfast", "meditation", "workout")))
    if tokens & {"commute", "commuting"}:
        slots.append(("commute", ("commute", "commuting", "work")))
    return tuple(slots)


def _routine_time_total_query(query: str) -> bool:
    """Return whether the query asks for total time across routine activity slots."""
    tokens = set(source_tokens(query))
    return bool(
        tokens & {"total", "combined"}
        and tokens & {"time"}
        and tokens & {"ready", "commute", "commuting", "routine"}
        and len(_routine_time_slots(query)) >= 2
    )


def _assistant_role_fragment(fragment: str) -> bool:
    user_index = fragment.casefold().find("user:")
    assistant_index = fragment.casefold().find("assistant:")
    return assistant_index >= 0 and (user_index < 0 or assistant_index < user_index)


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


def _routine_fragment_is_personal(fragment: str) -> bool:
    """Return whether a fragment records the user's own routine, not advice."""
    lowered = fragment.casefold()
    if re.search(r"\b(?:try|suggest|recommend|could|should|would|tips?)\b", lowered):
        return False
    if re.search(r"\b(?:your|you)\s+(?:morning\s+)?commute\b", lowered) and not _FIRST_PERSON_CONTEXT_RE.search(fragment):
        return False
    return bool(_FIRST_PERSON_CONTEXT_RE.search(fragment))


def _duration_match_is_range_fragment(fragment: str, start: int) -> bool:
    before = fragment[max(0, start - 3):start]
    return bool(re.search(r"\d\s*[-–]\s*$", before))


def _aggregate_total_answer_query(query: str) -> bool:
    """Return whether the query asks for a combined aggregate answer surface."""
    query_text = " ".join(query.casefold().split())
    return bool(
        re.search(
            r"\b(?:total|combined|altogether|sum)\b|\bin\s+total\b|\bhow\s+many\s+.*\btotal\b",
            query_text,
        )
    )


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


@dataclass(frozen=True)
class _QueryBoundScalarTotalSpec:
    kind: str
    answer_unit: str


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


def _semantic_number_value(sentence: str, pattern: str) -> float | None:
    match = re.search(pattern, sentence, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.groupdict().get("value") or match.groupdict().get("value_alt")
    if not value:
        return None
    return float(value.replace(",", ""))


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


def _format_grouped_number(value: float) -> str:
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:g}"


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


def _count_percentage_number_is_percentage_or_date(fragment: str, raw_value: str) -> bool:
    """Return whether a numeric span is not a plain count operand."""
    escaped = re.escape(raw_value)
    return bool(
        re.search(rf"\b{escaped}\s*%", fragment)
        or re.search(rf"\b{escaped}\s*(?:am|pm)\b", fragment, flags=re.IGNORECASE)
        or re.search(rf"\b(?:19|20){escaped}\b", fragment)
    )


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


def _local_percent_context(text: str, start: int, end: int) -> str:
    """Return a bounded clause around a percentage mention."""
    left = max(text.rfind(".", 0, start), text.rfind("|", 0, start), text.rfind("\n", 0, start))
    right_candidates = [
        index for index in (text.find(".", end), text.find("|", end), text.find("\n", end)) if index >= 0
    ]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right][:500]


def _text_mentions_title(text: str, title: str) -> bool:
    return title.casefold() in text.casefold()


def _arithmetic_context_text(context: str) -> str:
    """Return enough normalized source text to find split arithmetic operands."""
    return source_context_snippet(context, max_chars=12_000)


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


def _current_value_phrase_score(text: str) -> int:
    score = 0
    lowered = text.casefold()
    if re.search(r"\b(?:now|currently|so far|to date|at this point)\b", lowered):
        score += 20
    if re.search(r"\b(?:already|just|recently)\b", lowered):
        score += 5
    return score


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


def _session_recency_score(text: str) -> int:
    if match := re.search(r"\blongmemeval_session_date=(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})", text):
        return int(match.group("year")) * 400 + int(match.group("month")) * 31 + int(match.group("day"))
    return 0


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


def _numeric_context_text(context: str) -> str:
    """Return source text once, excluding Eventloom JSON payload echoes."""
    text = source_context_snippet(context)
    return text.split(' {"content":', 1)[0]


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


def _elapsed_duration_at_event_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    if not {"how", "long", "when"} <= query_tokens:
        return False
    if not query_tokens & {"had", "been"}:
        return False
    return bool(query_tokens & {"bought", "buy", "got", "purchased", "started", "joined"})


def _road_trip_drive_query(query: str) -> bool:
    query_tokens = set(source_tokens(query))
    return bool(query_tokens & {"road", "trip", "destinations"}) and bool(
        query_tokens & {"driving", "drove", "drive"}
    )
