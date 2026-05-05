"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from zaxy.event import EventLog


@pytest.fixture
def tmp_eventlog() -> EventLog:
    """Return an EventLog backed by a temporary file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
        path = fh.name
    log = EventLog(path)
    yield log
    Path(path).unlink(missing_ok=True)
