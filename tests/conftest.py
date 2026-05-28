"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from zaxy.event import EventLog


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep unit tests independent from a developer's local `.env` file."""
    from zaxy.config import get_settings

    monkeypatch.setenv("ZAXY_ENV", "test")
    monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")
    for env_name in (
        "NEO4J_PASSWORD_FILE",
        "MCP_ADMIN_TOKEN_FILE",
        "MCP_REMOTE_AUTH_TOKEN_FILE",
        "OPENAI_API_KEY_FILE",
        "RERANKER_API_KEY_FILE",
        "COORDINATION_SEMANTIC_CONFLICT_API_KEY_FILE",
        "PATHLIGHT_ACCESS_TOKEN_FILE",
    ):
        monkeypatch.setenv(env_name, "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def tmp_eventlog() -> EventLog:
    """Return an EventLog backed by a temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        path = fh.name
    log = EventLog(path)
    yield log
    Path(path).unlink(missing_ok=True)
