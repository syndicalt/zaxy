"""Agent recall hardening for long-running Zaxy sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from zaxy.event import Event, EventLog
from zaxy.security import validate_session_id

MemoryActivity = Literal["bootstrap", "checkout", "feedback"]

MEMORY_ACTIVITY_EVENT_TYPES = {
    "bootstrap": "memory.bootstrap.shown",
    "checkout": "memory.checkout.completed",
    "feedback": "memory.feedback.recorded",
}
MEMORY_USE_EVENT_TYPES = {
    *MEMORY_ACTIVITY_EVENT_TYPES.values(),
    "memory.feedback",
    "memory.reinforced",
}
BOUNDARY_TRIGGERS = {
    "session-start",
    "start",
    "resume",
    "session-resumed",
    "precompact",
    "compaction",
    "postcompact",
    "long-tool-run",
}
WHERE_ARE_WE_TERMS = (
    "where are we",
    "what is left",
    "what's left",
    "continue",
    "roadmap",
    "current goal",
    "status",
)
DEFAULT_STALE_EVENT_THRESHOLD = 8
DEFAULT_LONG_TURN_THRESHOLD = 8


def record_memory_activity(
    eventloom_path: str | Path,
    *,
    session_id: str,
    activity: MemoryActivity,
    source: str,
    query: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Append a lightweight memory activity marker."""
    sid = validate_session_id(session_id)
    payload: dict[str, Any] = {
        "activity": activity,
        "source": source,
    }
    if query:
        payload["query"] = query
    if metadata:
        payload.update(metadata)
    return _eventlog(eventloom_path, sid).append(
        MEMORY_ACTIVITY_EVENT_TYPES[activity],
        actor="zaxy-memory",
        payload=payload,
        thread=sid,
    )


def inspect_memory_persistence(
    eventloom_path: str | Path,
    *,
    session_id: str,
    stale_event_threshold: int = DEFAULT_STALE_EVENT_THRESHOLD,
) -> dict[str, Any]:
    """Inspect recent bootstrap, checkout, feedback, and reminder state."""
    sid = validate_session_id(session_id)
    events = _eventlog(eventloom_path, sid).read_all()
    latest_seq = events[-1].seq if events else 0
    last_bootstrap = _last_seq(events, {"memory.bootstrap.shown"})
    last_checkout = _last_seq(events, {"memory.checkout.completed"})
    last_feedback = _last_seq(events, {"memory.feedback.recorded", "memory.feedback", "memory.reinforced"})
    last_reminder = _last_seq(events, {"memory.reminder.suggested"})
    last_memory_use = _last_seq(events, MEMORY_USE_EVENT_TYPES)
    events_since_memory_use = latest_seq - last_memory_use if last_memory_use else latest_seq
    stale = events_since_memory_use >= stale_event_threshold if latest_seq else True
    return {
        "session_id": sid,
        "latest_seq": latest_seq,
        "last_bootstrap_seq": last_bootstrap or None,
        "last_checkout_seq": last_checkout or None,
        "last_feedback_seq": last_feedback or None,
        "last_reminder_seq": last_reminder or None,
        "last_memory_use_seq": last_memory_use or None,
        "events_since_memory_use": events_since_memory_use,
        "stale_event_threshold": stale_event_threshold,
        "stale": stale,
        "warning": "memory checkout is stale or absent" if stale else None,
    }


def suggest_memory_reminder(
    eventloom_path: str | Path,
    *,
    session_id: str,
    trigger: str,
    source: str = "zaxy-policy",
    reason: str | None = None,
    turn_count: int | None = None,
    current_task: str | None = None,
    stale_event_threshold: int = DEFAULT_STALE_EVENT_THRESHOLD,
    long_turn_threshold: int = DEFAULT_LONG_TURN_THRESHOLD,
) -> dict[str, Any] | None:
    """Return a memory.reminder.suggested event input when policy says to reintroduce Zaxy."""
    sid = validate_session_id(session_id)
    normalized_trigger = trigger.casefold().strip().replace("_", "-")
    status = inspect_memory_persistence(
        eventloom_path,
        session_id=sid,
        stale_event_threshold=stale_event_threshold,
    )
    reasons: list[str] = []
    if normalized_trigger in BOUNDARY_TRIGGERS:
        reasons.append("context_boundary")
    if status["stale"]:
        reasons.append("stale_memory_activity")
    if turn_count is not None and turn_count >= long_turn_threshold:
        reasons.append("long_session")
    if _is_where_are_we_query(current_task):
        reasons.append("where_are_we_query")
    if not reasons:
        return None
    query = current_task or "current task, project direction, and recent decisions"
    payload = {
        "trigger": normalized_trigger,
        "source": source,
        "reason": reason,
        "turn_count": turn_count,
        "query": query,
        "recommended_tool": "memory_checkout",
        "recommended_arguments": {"query": query, "session_id": sid},
        "reasons": reasons,
        "last_bootstrap_seq": status["last_bootstrap_seq"],
        "last_checkout_seq": status["last_checkout_seq"],
        "last_feedback_seq": status["last_feedback_seq"],
        "events_since_memory_use": status["events_since_memory_use"],
        "prompt": _reminder_prompt(query=query),
    }
    return {
        "event_type": "memory.reminder.suggested",
        "actor": "zaxy-memory",
        "payload": payload,
    }


def build_memory_reminder(payload: dict[str, Any]) -> str:
    """Format a reminder payload as short prompt-ready guidance."""
    query = str(payload.get("query") or "current task, project direction, and recent decisions")
    return _reminder_prompt(query=query)


def append_memory_reminder_if_needed(
    eventloom_path: str | Path,
    *,
    session_id: str,
    trigger: str,
    source: str,
    reason: str | None = None,
    turn_count: int | None = None,
    current_task: str | None = None,
) -> Event | None:
    """Append memory.reminder.suggested when the reminder policy triggers."""
    event_input = suggest_memory_reminder(
        eventloom_path,
        session_id=session_id,
        trigger=trigger,
        source=source,
        reason=reason,
        turn_count=turn_count,
        current_task=current_task,
    )
    if event_input is None:
        return None
    sid = validate_session_id(session_id)
    return _eventlog(eventloom_path, sid).append(
        event_input["event_type"],
        actor=event_input["actor"],
        payload=event_input["payload"],
        thread=sid,
    )


def _eventlog(eventloom_path: str | Path, session_id: str) -> EventLog:
    base = Path(eventloom_path)
    path = base if base.suffix == ".jsonl" else base / f"{session_id}.jsonl"
    return EventLog(path)


def _last_seq(events: list[Event], event_types: set[str]) -> int:
    for event in reversed(events):
        if event.type in event_types:
            return event.seq
    return 0


def _is_where_are_we_query(query: str | None) -> bool:
    if not query:
        return False
    text = query.casefold()
    return any(term in text for term in WHERE_ARE_WE_TERMS)


def _reminder_prompt(*, query: str) -> str:
    return (
        "Zaxy memory reminder: Call memory_bootstrap if session awareness is unclear, "
        f"then Call memory_checkout(query={query!r}) before answering. Trust only cited "
        "current checkout facts, and call memory_feedback when cited context is used."
    )
