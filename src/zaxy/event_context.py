"""Runtime helpers for rendering Eventloom events as compact text."""

from __future__ import annotations

import json


def event_context(event: dict[str, object]) -> str:
    """Format an event as a compact context chunk."""
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
