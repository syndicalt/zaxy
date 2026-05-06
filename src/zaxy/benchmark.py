"""Competitive retrieval benchmark fixtures and baselines."""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class RetrievalScore:
    """Correctness score for a benchmark retrieval run."""

    score: float
    expected_hits: tuple[str, ...]
    missing_expected: tuple[str, ...]
    forbidden_hits: tuple[str, ...]


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
    expected_hits = tuple(term for term in case.expected_terms if term.casefold() in haystack)
    missing_expected = tuple(
        term for term in case.expected_terms if term.casefold() not in haystack
    )
    forbidden_hits = tuple(term for term in case.forbidden_terms if term.casefold() in haystack)

    expected_score = len(expected_hits) / len(case.expected_terms) if case.expected_terms else 1.0
    penalty = len(forbidden_hits) / max(len(case.forbidden_terms), 1)
    score = max(0.0, expected_score - penalty)
    return RetrievalScore(
        score=round(score, 4),
        expected_hits=expected_hits,
        missing_expected=missing_expected,
        forbidden_hits=forbidden_hits,
    )


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
