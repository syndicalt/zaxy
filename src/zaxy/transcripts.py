"""Transcript ingestion helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from zaxy.security import secure_payload

DEFAULT_TRANSCRIPT_SOURCE = "transcript"


def collect_transcript_events(
    turns: Iterable[Mapping[str, Any]],
    *,
    source: str = DEFAULT_TRANSCRIPT_SOURCE,
) -> list[dict[str, Any]]:
    """Collect chat/session transcript turns as sanitized event inputs."""
    events: list[dict[str, Any]] = []
    for index, turn in enumerate(turns, start=1):
        role = str(turn.get("role") or "unknown").strip() or "unknown"
        content = str(turn.get("content") or "").strip()
        if not content:
            continue

        secured = secure_payload({"content": content})
        events.append(
            {
                "event_type": "transcript.turn",
                "actor": role,
                "payload": {
                    "source": source,
                    "turn_index": index,
                    "role": role,
                    "content": secured.payload["content"],
                    "redacted_paths": secured.redacted_paths,
                },
            }
        )
    return events
