"""Smoke tests for v0.5 public examples."""

from __future__ import annotations

import json
import subprocess
import sys


def test_single_agent_memory_example_runs() -> None:
    """Single-agent example should run without sidecars and print JSON evidence."""
    result = subprocess.run(
        [sys.executable, "examples/single_agent_memory.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "single-agent-demo"
    assert payload["bootstrap"]["session_id"] == "single-agent-demo"
    assert payload["checkout"]["session_id"] == "single-agent-demo"
    assert payload["event_count"] >= 2
