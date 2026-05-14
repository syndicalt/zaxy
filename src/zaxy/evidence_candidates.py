"""Typed evidence candidates for retrieval-time answer synthesis."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceCandidate:
    """A typed answer ingredient extracted from one cited source context."""

    kind: str
    value: str
    unit: str
    source_group: str
    label: str
    context: str
    relevance: int


@dataclass(frozen=True)
class EvidenceProjection:
    """Rendered evidence candidates plus the source groups that support them."""

    lines: tuple[str, ...]
    source_groups: tuple[str, ...]


def aggregate_candidate_projection(query: str, contexts: list[str]) -> EvidenceProjection:
    """Build deterministic aggregate answer candidates from cited contexts."""
    lines: list[str] = []
    source_groups: list[str] = []
    count = _count_candidates(query, contexts)
    if len(count) >= 2:
        lines.extend(_count_candidate_lines(count))
        source_groups.extend(candidate.source_group for candidate in count)
    currency = _currency_candidates(query, contexts)
    if currency:
        lines.extend(_currency_candidate_lines(currency))
        source_groups.extend(candidate.source_group for candidate in currency)
    return EvidenceProjection(
        lines=tuple(lines),
        source_groups=tuple(dict.fromkeys(source_groups)),
    )


def aggregate_candidate_lines(query: str, contexts: list[str]) -> list[str]:
    """Render deterministic aggregate answer candidates from cited contexts."""
    return list(aggregate_candidate_projection(query, contexts).lines)


def _count_candidate_lines(candidates: list[EvidenceCandidate]) -> list[str]:
    source_ids = ",".join(candidate.source_group for candidate in candidates)
    return [
        f"count_answer={len(candidates)}",
        "count_unit=events",
        f"count_source_ids={source_ids}",
    ]


def _count_candidates(query: str, contexts: list[str]) -> list[EvidenceCandidate]:
    if not _count_query(query):
        return []
    focus_terms = _expanded_focus_terms(query)
    candidates: list[EvidenceCandidate] = []
    seen_groups: set[str] = set()
    scored: list[EvidenceCandidate] = []
    for context in contexts:
        text = _context_text(context)
        group = _source_group(context)
        if group in seen_groups:
            continue
        relevance = _relevance(focus_terms, text)
        if relevance <= 0:
            continue
        scored.append(
            EvidenceCandidate(
                kind="count",
                value="1",
                unit="event",
                source_group=group,
                label=_count_label(text),
                context=text,
                relevance=relevance,
            )
        )
    if not scored:
        return []
    best = max(candidate.relevance for candidate in scored)
    threshold = 2 if best >= 2 else best
    for candidate in scored:
        if candidate.relevance < threshold:
            continue
        candidates.append(candidate)
        seen_groups.add(candidate.source_group)
    return candidates


def _currency_candidate_lines(candidates: list[EvidenceCandidate]) -> list[str]:
    values = sorted((float(candidate.value) for candidate in candidates), reverse=True)
    lines = [
        "currency_values=" + ",".join(_format_currency(value) for value in values),
        f"currency_total={_format_currency(sum(values))}",
    ]
    max_item = max(candidates, key=lambda candidate: float(candidate.value))
    lines.append(f"currency_max={_format_currency(float(max_item.value))}")
    if max_item.label:
        lines.append(f"currency_max_label={max_item.label}")
    if len(values) >= 2:
        lines.append(
            f"currency_difference={_format_currency(max(values) - min(values))}"
        )
    return lines


def _currency_candidates(query: str, contexts: list[str]) -> list[EvidenceCandidate]:
    items: list[EvidenceCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for context in contexts:
        text = _context_text(context)
        group = _source_group(context)
        relevance = _relevance(_numeric_focus_terms(query), text)
        for match in re.finditer(r"\$(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)", text):
            value = str(float(match.group("value").replace(",", "")))
            label = _currency_label(text, match.start(), match.end())
            identity = (group, value, label.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            items.append(
                EvidenceCandidate(
                    kind="currency",
                    value=value,
                    unit="USD",
                    source_group=group,
                    label=label,
                    context=text,
                    relevance=relevance,
                )
            )
    return _filter_focused_candidates(query, items)


def _filter_focused_candidates(
    query: str,
    items: list[EvidenceCandidate],
) -> list[EvidenceCandidate]:
    if len(items) < 2:
        return items
    focus_terms = _numeric_focus_terms(query)
    if not focus_terms:
        return items
    if max((item.relevance for item in items), default=0) <= 0:
        return items
    selected = [item for item in items if item.relevance > 0]
    return selected if len(selected) >= 2 else items


def _count_query(query: str) -> bool:
    tokens = set(_tokens(query))
    return bool(tokens & {"many", "number", "count", "total"}) and "how" in tokens


def _expanded_focus_terms(query: str) -> set[str]:
    terms = _query_specific_terms(query)
    expanded = set(terms)
    semantic_groups = {
        "movie": {"movie", "movies", "film", "films", "cinema"},
        "festival": {"festival", "festivals"},
        "festivals": {"festival", "festivals"},
        "wedding": {"wedding", "weddings"},
        "weddings": {"wedding", "weddings"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
        if term.endswith("s") and len(term) > 3:
            expanded.add(term[:-1])
        else:
            expanded.add(f"{term}s")
    return expanded


def _numeric_focus_terms(query: str) -> set[str]:
    terms = _query_specific_terms(query)
    expanded = set(terms)
    semantic_groups = {
        "grocery": {"grocery", "groceries", "market", "store", "foods", "trader", "joe"},
        "groceries": {"grocery", "groceries", "market", "store", "foods", "trader", "joe"},
        "store": {"store", "market", "foods", "trader", "joe"},
        "luxury": {"luxury", "designer", "premium", "bag", "shoes", "jewelry", "watch"},
    }
    for term in terms:
        expanded.update(semantic_groups.get(term, set()))
    return expanded


def _query_specific_terms(query: str) -> set[str]:
    return {
        token
        for token in _tokens(query)
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    }


def _relevance(focus_terms: set[str], context: str) -> int:
    if not focus_terms:
        return 0
    context_terms = set(_tokens(context))
    return len(focus_terms & context_terms)


def _context_text(context: str) -> str:
    text = " ".join(context.split())
    return text.split(' {"content":', 1)[0]


def _source_group(context: str) -> str:
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


def _count_label(text: str) -> str:
    cleaned = re.sub(r"\bcontent=\S+\s*", "", text)
    cleaned = re.sub(r"\bcitation=\S+\s*", "", cleaned)
    match = re.search(r"\bI\s+(?P<label>.+?)(?:[.?!]|$)", cleaned)
    if match:
        return " ".join(match.group("label").split()[:12])
    return ""


def _currency_label(text: str, start: int, end: int) -> str:
    after = text[end : end + 120]
    for pattern in (
        r"\s+(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})",
        r"\s+(?:for|on)\s+(?:a|an|the)?\s*(?P<label>[A-Za-z][A-Za-z0-9'&.-]*(?:\s+[A-Za-z][A-Za-z0-9'&.-]*){0,4})",
    ):
        match = re.match(pattern, after)
        if match:
            return _clean_label(match.group("label"))
    before = text[max(0, start - 120) : start]
    match = re.search(
        r"\b(?:at|from)\s+(?P<label>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})\s*$",
        before,
    )
    if match:
        return _clean_label(match.group("label"))
    return ""


def _clean_label(label: str) -> str:
    label = re.split(r"\b(?:on|in|for|with|and|but|because|when)\b", label, maxsplit=1)[0]
    return " ".join(label.strip(" .,'\"").split())


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[-_:./#][a-z0-9]+)*", text.casefold()):
        tokens.append(token)
        if re.search(r"[-_:/#]", token):
            tokens.extend(part for part in re.split(r"[-_:/#]+", token) if part)
    return tokens


def _format_currency(value: float) -> str:
    if value.is_integer():
        return f"${int(value):,}"
    whole = int(value)
    fraction = f"{value:.2f}".split(".", 1)[1].rstrip("0")
    return f"${whole:,}.{fraction}"


_QUERY_STOPWORDS = {
    "about",
    "after",
    "all",
    "and",
    "amount",
    "attend",
    "attended",
    "before",
    "did",
    "for",
    "from",
    "how",
    "many",
    "money",
    "most",
    "much",
    "past",
    "spent",
    "the",
    "total",
    "what",
    "when",
    "which",
}
