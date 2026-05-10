"""Deterministic memory working-set projection for prompt assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from zaxy.context import Context


@dataclass(frozen=True)
class WorkingSetItem:
    """One compact semantic memory item for model-facing context."""

    category: str
    summary: str
    citation: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class MemoryWorkingSet:
    """Bounded semantic projection of recent replay and retrieved context."""

    items: list[WorkingSetItem]
    truncated: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


def build_working_set(
    events: list[Any],
    contexts: list[Context],
    *,
    max_items: int = 8,
) -> MemoryWorkingSet:
    """Build a bounded semantic working set without LLM summarization."""
    candidates: list[WorkingSetItem] = []
    for event in events:
        item = _item_from_event(event)
        if item is not None:
            candidates.append(item)
    for context in contexts:
        item = _source_anchor_from_context(context)
        if item is not None:
            candidates.append(item)
    deduped = _dedupe_items(candidates)
    safe_limit = max(0, max_items)
    return MemoryWorkingSet(
        items=deduped[:safe_limit],
        truncated=len(deduped) > safe_limit,
    )


def format_working_set(working_set: MemoryWorkingSet) -> str:
    """Render the active memory working set for prompt injection."""
    lines = ["# Active Memory Working Set"]
    if not working_set.items:
        lines.append("- none")
        return "\n".join(lines)
    for item in working_set.items:
        citation = f" ({item.citation})" if item.citation else ""
        lines.append(f"- {item.category}: {item.summary}{citation}")
    if working_set.truncated:
        lines.append("- working_set_truncated: true")
    return "\n".join(lines)


def _item_from_event(event: Any) -> WorkingSetItem | None:
    event_type = str(getattr(event, "type", ""))
    payload = getattr(event, "payload", {})
    if not isinstance(payload, dict):
        return None
    category = _event_category(event_type)
    if category is None:
        return None
    summary = _summary(payload)
    if not summary:
        return None
    return WorkingSetItem(
        category=category,
        summary=summary,
        citation=_event_citation(event),
        source="replay",
    )


def _source_anchor_from_context(context: Context) -> WorkingSetItem | None:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    if not isinstance(citation, str) or not citation:
        return None
    summary = " ".join(context.content.split())
    if not summary:
        return None
    if len(summary) > 160:
        summary = f"{summary[:157]}..."
    return WorkingSetItem(
        category="source_anchor",
        summary=summary,
        citation=citation,
        source=context.source,
    )


def _event_category(event_type: str) -> str | None:
    if event_type.startswith("goal."):
        return "goal"
    if event_type.startswith("decision."):
        return "decision"
    if event_type.startswith("task."):
        return "task"
    if event_type == "command.completed":
        return "action"
    if event_type in {"document.indexed", "code.file.indexed", "file.edit.applied"}:
        return "artifact"
    if event_type == "llm.packet.projected":
        return "memory"
    if event_type in {"issue.diagnosed", "blocker.created"}:
        return "blocker"
    return None


def _summary(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("command"), str):
        outcome = payload.get("outcome")
        prefix = f"{outcome} " if isinstance(outcome, str) and outcome else ""
        return f"{prefix}{payload['command']}".strip()
    if isinstance(payload.get("path"), str):
        operation = payload.get("operation")
        prefix = f"{operation} " if isinstance(operation, str) and operation else ""
        summary = payload.get("summary")
        suffix = f": {summary}" if isinstance(summary, str) and summary else ""
        return f"{prefix}{payload['path']}{suffix}".strip()
    for key in ("title", "decision", "summary", "task", "content", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _event_citation(event: Any) -> str | None:
    thread = getattr(event, "thread", None)
    seq = getattr(event, "seq", None)
    event_hash = getattr(event, "hash", None)
    if not isinstance(thread, str) or not isinstance(seq, int) or not isinstance(event_hash, str):
        return None
    return f"eventloom://{thread}/events/{seq}#{event_hash[:12]}"


def _dedupe_items(items: list[WorkingSetItem]) -> list[WorkingSetItem]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[WorkingSetItem] = []
    for item in items:
        key = (item.category, item.summary.casefold(), item.citation)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
