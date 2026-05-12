"""Opaque continuation cursors for ranked retrieval."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

from zaxy.security import validate_query, validate_session_id

CURSOR_VERSION = 1


@dataclass(frozen=True)
class QueryCursor:
    """Decoded state for continuing a ranked query result set."""

    query: str
    session_id: str
    temporal_point: str | None
    offset: int


def encode_query_cursor(
    *,
    query: str,
    session_id: str,
    temporal_point: str | None,
    offset: int,
) -> str:
    """Encode query continuation state into an opaque URL-safe token."""
    validated_query = validate_query(query)
    validated_session = validate_session_id(session_id)
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("cursor offset must be a non-negative integer")
    payload = {
        "v": CURSOR_VERSION,
        "query": validated_query,
        "session_id": validated_session,
        "temporal_point": temporal_point,
        "offset": offset,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_query_cursor(cursor: str) -> QueryCursor:
    """Decode an opaque query cursor."""
    if not isinstance(cursor, str) or not cursor:
        raise ValueError("invalid query cursor")
    try:
        padded = cursor + ("=" * (-len(cursor) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeEncodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid query cursor") from exc
    if not isinstance(payload, dict) or payload.get("v") != CURSOR_VERSION:
        raise ValueError("invalid query cursor")
    return _cursor_from_payload(payload)


def validate_query_cursor(
    cursor: str,
    *,
    query: str,
    session_id: str,
    temporal_point: str | None,
) -> QueryCursor:
    """Decode a cursor and ensure it belongs to the requested query scope."""
    decoded = decode_query_cursor(cursor)
    if decoded.query != validate_query(query):
        raise ValueError("cursor does not match query")
    if decoded.session_id != validate_session_id(session_id):
        raise ValueError("cursor does not match session_id")
    if decoded.temporal_point != temporal_point:
        raise ValueError("cursor does not match temporal_filter")
    return decoded


def _cursor_from_payload(payload: dict[str, Any]) -> QueryCursor:
    raw_query = payload.get("query")
    raw_session_id = payload.get("session_id")
    if not isinstance(raw_query, str) or not isinstance(raw_session_id, str):
        raise ValueError("invalid query cursor")
    query = validate_query(raw_query)
    session_id = validate_session_id(raw_session_id)
    temporal_point = payload.get("temporal_point")
    if temporal_point is not None and not isinstance(temporal_point, str):
        raise ValueError("invalid query cursor")
    offset = payload.get("offset")
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("invalid query cursor")
    return QueryCursor(
        query=query,
        session_id=session_id,
        temporal_point=temporal_point,
        offset=offset,
    )
