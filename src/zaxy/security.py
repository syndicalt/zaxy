"""Security validation helpers shared by public entrypoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_QUERY_LENGTH = 4096
MAX_QUERY_LIMIT = 100
MAX_REPLAY_EVENTS = 1000
MAX_TRAVERSAL_DEPTH = 5


def validate_session_id(session_id: str) -> str:
    """Return a safe session ID or raise ValueError."""
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError(
            "Invalid session_id: use 1-128 ASCII letters, digits, '.', '_', or '-'"
        )
    return session_id


def eventlog_path(base: Path, session_id: str) -> Path:
    """Build a log path and ensure it stays under the configured base."""
    safe_id = validate_session_id(session_id)
    base_resolved = base.resolve()
    path = (base_resolved / f"{safe_id}.jsonl").resolve()
    if path.parent != base_resolved:
        raise ValueError("Invalid session_id: resolved path escapes Eventloom base")
    return path


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate payload type, JSON serializability, and serialized size."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    try:
        encoded = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be JSON serializable") from exc
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    return payload


def validate_query(query: str) -> str:
    """Validate query text size."""
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query exceeds {MAX_QUERY_LENGTH} characters")
    return query


def validate_limit(limit: int | None, default: int = 10) -> int:
    """Validate result limits for database-backed operations."""
    value = default if limit is None else limit
    if not isinstance(value, int) or value < 1 or value > MAX_QUERY_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_QUERY_LIMIT}")
    return value


def validate_from_seq(from_seq: int | None) -> int:
    """Validate replay starting sequence."""
    value = 1 if from_seq is None else from_seq
    if not isinstance(value, int) or value < 1:
        raise ValueError("from_seq must be a positive integer")
    return value


def validate_traversal_depth(depth: int) -> int:
    """Validate traversal depth before it is interpolated into Cypher."""
    if not isinstance(depth, int) or depth < 1 or depth > MAX_TRAVERSAL_DEPTH:
        raise ValueError(f"depth must be between 1 and {MAX_TRAVERSAL_DEPTH}")
    return depth


def query_hash(query: str) -> str:
    """Return a stable hash for trace correlation without leaking query text."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()
