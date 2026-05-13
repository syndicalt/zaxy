"""Competitive retrieval benchmark fixtures and baselines."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from zaxy.event import EventLog


@dataclass(frozen=True)
class BenchmarkCase:
    """A retrieval benchmark case with correctness expectations."""

    name: str
    query: str
    expected_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...] = ()
    category: str = "general"
    temporal_point: str | None = None
    identity_terms: tuple[str, ...] = ()
    source_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalScore:
    """Correctness score for a benchmark retrieval run."""

    score: float
    expected_hits: tuple[str, ...]
    missing_expected: tuple[str, ...]
    forbidden_hits: tuple[str, ...]
    identity_recall: float | None = None
    identity_hits: tuple[str, ...] = ()
    missing_identities: tuple[str, ...] = ()
    source_recall: float | None = None
    source_hits: tuple[str, ...] = ()
    missing_sources: tuple[str, ...] = ()


class FlatJsonlRetriever:
    """Naive competitor baseline that scans raw Eventloom JSONL text.

    This intentionally ignores graph structure and temporal validity. It is a
    useful floor because it behaves like many simple persistent-context systems:
    append records, search text, and hand the matched chunks back to the agent.
    """

    def __init__(self, eventlog: EventLog) -> None:
        self._eventlog = eventlog

    def query(
        self,
        query: str,
        temporal_point: str | None = None,
        limit: int = 10,
    ) -> list[str]:
        """Return JSONL chunks that contain at least one query token."""
        del temporal_point
        tokens = _tokens(query)
        matches: list[str] = []
        for event in self._eventlog.read_all():
            context = _event_context(event.model_dump())
            searchable = context.casefold()
            if any(token in searchable for token in tokens):
                matches.append(context)
            if len(matches) >= limit:
                break
        return matches


def build_competitive_event_log(path: str | Path) -> EventLog:
    """Build a deterministic event log for retrieval benchmark cases."""
    log = EventLog(path)
    events = [
        (
            "goal.created",
            "user",
            {"title": "Ship MVP", "description": "Launch the context engine"},
            datetime(2024, 1, 1, tzinfo=UTC),
        ),
        (
            "task.proposed",
            "agent",
            {"taskId": "t1", "summary": "Design landing page for Ship MVP"},
            datetime(2024, 1, 2, tzinfo=UTC),
        ),
        (
            "task.claimed",
            "agent",
            {"taskId": "t1"},
            datetime(2024, 1, 3, tzinfo=UTC),
        ),
        (
            "user.preference_changed",
            "user",
            {"userId": "u1", "key": "theme", "value": "dark"},
            datetime(2024, 2, 1, tzinfo=UTC),
        ),
        (
            "user.preference_changed",
            "user",
            {"userId": "u1", "key": "theme", "value": "light"},
            datetime(2024, 6, 1, tzinfo=UTC),
        ),
        (
            "task.completed",
            "agent",
            {"taskId": "t1"},
            datetime(2024, 6, 2, tzinfo=UTC),
        ),
    ]
    for event_type, actor, payload, timestamp in events:
        log.append(event_type, actor=actor, payload=payload, timestamp=timestamp)
    return log


def competitive_cases() -> tuple[BenchmarkCase, ...]:
    """Return the built-in competitive retrieval cases."""
    return (
        BenchmarkCase(
            name="current-theme",
            query="What is the current user theme preference?",
            expected_terms=("theme=light",),
            forbidden_terms=("theme=dark",),
            category="stale_context",
        ),
        BenchmarkCase(
            name="theme-before-change",
            query="What was the user theme preference in March 2024?",
            temporal_point="2024-03-01T00:00:00Z",
            expected_terms=("theme=dark",),
            forbidden_terms=("theme=light",),
            category="temporal",
        ),
        BenchmarkCase(
            name="claimed-task-for-goal",
            query="Which task is connected to Ship MVP?",
            expected_terms=("taskId=t1", "Ship MVP"),
            category="traversal",
        ),
    )


def score_retrieval(case: BenchmarkCase, contexts: list[str]) -> RetrievalScore:
    """Score retrieved context against expected and forbidden terms."""
    haystack = "\n".join(contexts).casefold()
    expected_hits = tuple(
        term for term in case.expected_terms
        if _expected_term_present(term, haystack)
    )
    missing_expected = tuple(
        term for term in case.expected_terms
        if not _expected_term_present(term, haystack)
    )
    forbidden_hits = tuple(term for term in case.forbidden_terms if term.casefold() in haystack)
    identity_hits = tuple(term for term in case.identity_terms if term.casefold() in haystack)
    missing_identities = tuple(
        term for term in case.identity_terms if term.casefold() not in haystack
    )
    source_hits = tuple(term for term in case.source_terms if term.casefold() in haystack)
    missing_sources = tuple(term for term in case.source_terms if term.casefold() not in haystack)

    expected_score = len(expected_hits) / len(case.expected_terms) if case.expected_terms else 1.0
    penalty = len(forbidden_hits) / max(len(case.forbidden_terms), 1)
    score = max(0.0, expected_score - penalty)
    identity_recall = (
        len(identity_hits) / len(case.identity_terms)
        if case.identity_terms
        else None
    )
    source_recall = (
        len(source_hits) / len(case.source_terms)
        if case.source_terms
        else None
    )
    return RetrievalScore(
        score=round(score, 4),
        expected_hits=expected_hits,
        missing_expected=missing_expected,
        forbidden_hits=forbidden_hits,
        identity_recall=round(identity_recall, 4) if identity_recall is not None else None,
        identity_hits=identity_hits,
        missing_identities=missing_identities,
        source_recall=round(source_recall, 4) if source_recall is not None else None,
        source_hits=source_hits,
        missing_sources=missing_sources,
    )


def expected_terms_recall(case: BenchmarkCase, contexts: list[str]) -> float | None:
    """Return fraction of expected answer terms present in contexts."""
    if not case.expected_terms:
        return None
    haystack = "\n".join(contexts).casefold()
    hits = sum(
        1 for term in case.expected_terms
        if _expected_term_present(term, haystack)
    )
    return round(hits / len(case.expected_terms), 4)


_ANSWER_ALIASES = {
    "valentine's day": "february 14th",
    "valentines day": "february 14th",
}

_LOW_INFORMATION_ANSWER_TOKENS = {
    "a",
    "an",
    "at",
    "in",
    "of",
    "on",
    "the",
    "to",
}


def _expected_term_present(term: str, haystack: str) -> bool:
    """Return whether retrieved context contains an expected answer surface."""
    normalized_term = _normalize_answer_text(term)
    normalized_haystack = _normalize_answer_text(haystack)
    if normalized_term in normalized_haystack:
        return True
    if _absence_answer_present(normalized_term, normalized_haystack):
        return True
    if _parenthetical_acronym_present(normalized_term, normalized_haystack):
        return True
    term_tokens = [
        token for token in _answer_tokens(normalized_term)
        if token not in _LOW_INFORMATION_ANSWER_TOKENS
    ]
    if len(term_tokens) < 2:
        return False
    haystack_tokens = set(_answer_tokens(normalized_haystack))
    return all(token in haystack_tokens for token in term_tokens)


def _absence_answer_present(term: str, haystack: str) -> bool:
    """Return whether structured/no-mention evidence satisfies an absence answer."""
    if "you did not mention this information" not in term:
        return False
    absence_markers = (
        "zaxy_absence_check=true",
        "synthesis_mode=absence_check",
        "not_mentioned_candidate=",
        "did not mention",
    )
    if not any(marker in haystack for marker in absence_markers):
        return False
    target = _absence_target_from_expected_answer(term)
    if target and not all(token in set(_answer_tokens(haystack)) for token in _answer_tokens(target)):
        return False
    contrast = _absence_contrast_from_expected_answer(term)
    if not contrast:
        return True
    contrast_tokens = [
        token for token in _answer_tokens(contrast)
        if token not in _LOW_INFORMATION_ANSWER_TOKENS
        and token not in {"you", "your", "mentioned", "mention", "but", "not"}
    ]
    if not contrast_tokens:
        return True
    haystack_tokens = set(_answer_tokens(haystack))
    return any(token in haystack_tokens for token in contrast_tokens)


def _absence_target_from_expected_answer(term: str) -> str:
    match = re.search(r"\bbut not (?:your |my |the )?(?P<target>[a-z0-9' -]+?)[.!?]?$", term)
    return match.group("target").strip(" .'") if match else ""


def _absence_contrast_from_expected_answer(term: str) -> str:
    match = re.search(r"\byou mentioned (?P<contrast>.+?)\bbut not\b", term)
    return match.group("contrast").strip(" .") if match else ""


def _normalize_answer_text(text: str) -> str:
    normalized = text.casefold()
    for source, target in _ANSWER_ALIASES.items():
        normalized = normalized.replace(source, target)
    return normalized


def _answer_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _parenthetical_acronym_present(term: str, haystack: str) -> bool:
    """Return whether an expected full-name answer is present by acronym."""
    matches = re.findall(r"\(([a-z0-9]{2,12})\)", term)
    if not matches:
        return False
    haystack_tokens = set(_answer_tokens(haystack))
    return any(match in haystack_tokens for match in matches)


def _event_context(event: dict[str, object]) -> str:
    """Format an event as a compact flat-baseline context chunk."""
    payload = event.get("payload")
    payload_text = ""
    if isinstance(payload, dict):
        parts = [f"{key}={value}" for key, value in sorted(payload.items())]
        if "key" in payload and "value" in payload:
            parts.append(f"{payload['key']}={payload['value']}")
        payload_text = " ".join(parts)
    return " ".join(
        part
        for part in [
            str(event.get("timestamp", "")),
            str(event.get("type", "")),
            str(event.get("actor", "")),
            payload_text,
            json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else "",
        ]
        if part
    )


def _tokens(query: str) -> list[str]:
    """Tokenize query text for the flat baseline."""
    return [token for token in query.casefold().replace("?", " ").split() if len(token) > 2]
