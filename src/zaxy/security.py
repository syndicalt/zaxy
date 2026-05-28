"""Security validation helpers shared by public entrypoints."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_QUERY_LENGTH = 4096
MAX_QUERY_LIMIT = 100
MAX_REPLAY_EVENTS = 1000
MAX_TRAVERSAL_DEPTH = 5
REDACTED_VALUE = "[REDACTED]"
SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|bearer|cookie|credential|password|private[_-]?key|"
    r"secret|token)",
    re.IGNORECASE,
)
NON_SECRET_TOKEN_METRIC_KEYS = {
    "answer_at_5_per_1k_injected_tokens",
    "answer_at_5_per_1k_returned_tokens",
    "completion_tokens",
    "facts_per_1k_prompt_tokens",
    "mean_injected_tokens",
    "mean_returned_tokens",
    "prompt_tokens",
    "quality_per_1k_injected_tokens",
    "quality_per_1k_returned_tokens",
    "token_efficiency",
    "total_tokens",
}
SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bpypi-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
)


@dataclass(frozen=True)
class SecuredPayload:
    """Payload plus durable security classification metadata."""

    payload: dict[str, Any]
    sensitivity: str
    redacted_paths: list[str]


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


def secure_payload(payload: dict[str, Any]) -> SecuredPayload:
    """Return a redacted copy of a payload and its sensitivity classification."""
    safe_payload, redacted_paths = _redact_value(payload, path="")
    if not isinstance(safe_payload, dict):
        safe_payload = {}
    return SecuredPayload(
        payload=safe_payload,
        sensitivity="restricted" if redacted_paths else "public",
        redacted_paths=redacted_paths,
    )


def _redact_value(value: Any, path: str) -> tuple[Any, list[str]]:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        paths: list[str] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if _is_secret_key(str(key)):
                redacted[key] = REDACTED_VALUE
                paths.append(child_path)
                continue
            safe_child, child_paths = _redact_value(child, child_path)
            redacted[key] = safe_child
            paths.extend(child_paths)
        return redacted, paths

    if isinstance(value, list):
        redacted_items: list[Any] = []
        paths = []
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            safe_child, child_paths = _redact_value(child, child_path)
            redacted_items.append(safe_child)
            paths.extend(child_paths)
        return redacted_items, paths

    if isinstance(value, str) and _looks_like_secret(value):
        return REDACTED_VALUE, [path or "$"]

    return value, []


def _is_secret_key(key: str) -> bool:
    if key.casefold() in NON_SECRET_TOKEN_METRIC_KEYS:
        return False
    return SECRET_KEY_PATTERN.search(key) is not None


def _looks_like_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS)


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


def vector_has_signal(values: list[float]) -> bool:
    """Return whether a vector can produce a meaningful similarity score."""
    for value in values:  # noqa: SIM110 - avoid generator allocation on retrieval hot paths.
        if value != 0.0:
            return True
    return False


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
