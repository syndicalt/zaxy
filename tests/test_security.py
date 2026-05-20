"""Tests for shared security validation and payload sanitization."""

from __future__ import annotations

import pytest

from zaxy.security import (
    REDACTED_VALUE,
    eventlog_path,
    secure_payload,
    validate_limit,
    validate_payload,
    validate_query,
    validate_session_id,
    validate_traversal_depth,
)


class TestPayloadSecurity:
    """Tests for durable payload classification and redaction."""

    def test_public_payload_is_preserved(self) -> None:
        secured = secure_payload({"title": "Ship it", "tags": ["release", "docs"]})

        assert secured.payload == {"title": "Ship it", "tags": ["release", "docs"]}
        assert secured.sensitivity == "public"
        assert secured.redacted_paths == []

    def test_secret_keys_are_redacted_recursively(self) -> None:
        secured = secure_payload(
            {
                "password": "super-secret",
                "nested": {
                    "apiKey": "sk-live-value",
                    "items": [{"token": "abc123"}],
                },
            }
        )

        assert secured.payload == {
            "password": REDACTED_VALUE,
            "nested": {
                "apiKey": REDACTED_VALUE,
                "items": [{"token": REDACTED_VALUE}],
            },
        }
        assert secured.sensitivity == "restricted"
        assert secured.redacted_paths == [
            "password",
            "nested.apiKey",
            "nested.items[0].token",
        ]

    def test_secret_values_are_redacted_without_secret_keys(self) -> None:
        secured = secure_payload({"note": "temporary key sk-abcdefghijklmnop"})

        assert secured.payload == {"note": REDACTED_VALUE}
        assert secured.sensitivity == "restricted"
        assert secured.redacted_paths == ["note"]

    def test_version_like_values_are_not_treated_as_jwts(self) -> None:
        secured = secure_payload({"version": "1.2.3"})

        assert secured.payload == {"version": "1.2.3"}
        assert secured.sensitivity == "public"

    def test_token_count_metrics_are_not_treated_as_secrets(self) -> None:
        secured = secure_payload(
            {
                "token_efficiency": {
                    "prompt_tokens": 200,
                    "completion_tokens": 12,
                    "facts_per_1k_prompt_tokens": 15.0,
                },
                "token": "secret",
            }
        )

        assert secured.payload == {
            "token_efficiency": {
                "prompt_tokens": 200,
                "completion_tokens": 12,
                "facts_per_1k_prompt_tokens": 15.0,
            },
            "token": REDACTED_VALUE,
        }
        assert secured.sensitivity == "restricted"
        assert secured.redacted_paths == ["token"]


class TestValidation:
    """Smoke tests for shared public validation helpers."""

    def test_rejects_unsafe_session_id(self) -> None:
        with pytest.raises(ValueError, match="Invalid session_id"):
            validate_session_id("../escape")

    def test_eventlog_path_stays_under_base(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        assert eventlog_path(tmp_path, "agent-1") == tmp_path / "agent-1.jsonl"

    def test_payload_must_be_json_object(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            validate_payload([])  # type: ignore[arg-type]

    def test_query_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_query("")

    def test_limit_bounds(self) -> None:
        assert validate_limit(None, default=7) == 7
        with pytest.raises(ValueError, match="between"):
            validate_limit(0)

    def test_traversal_depth_bounds(self) -> None:
        assert validate_traversal_depth(3) == 3
        with pytest.raises(ValueError, match="between"):
            validate_traversal_depth(6)
