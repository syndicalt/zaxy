"""Tests for opaque query pagination cursors."""

from __future__ import annotations

import pytest

from zaxy.pagination import (
    QueryCursor,
    decode_query_cursor,
    encode_query_cursor,
    validate_query_cursor,
)


def test_query_cursor_round_trips_bound_query_state() -> None:
    """Cursors should preserve offset and the query scope they continue."""
    cursor = encode_query_cursor(
        query="What changed?",
        session_id="agent-1",
        temporal_point="2026-05-11T00:00:00Z",
        offset=10,
    )

    decoded = decode_query_cursor(cursor)

    assert decoded == QueryCursor(
        query="What changed?",
        session_id="agent-1",
        temporal_point="2026-05-11T00:00:00Z",
        offset=10,
    )


def test_validate_query_cursor_rejects_different_query_scope() -> None:
    """A cursor from one query must not be reused for a different query."""
    cursor = encode_query_cursor(
        query="What changed?",
        session_id="agent-1",
        temporal_point=None,
        offset=5,
    )

    with pytest.raises(ValueError, match="cursor does not match query"):
        validate_query_cursor(
            cursor,
            query="What else changed?",
            session_id="agent-1",
            temporal_point=None,
        )


def test_decode_query_cursor_rejects_malformed_payload() -> None:
    """Malformed cursors should fail closed instead of silently restarting."""
    with pytest.raises(ValueError, match="invalid query cursor"):
        decode_query_cursor("not-base64-json")
