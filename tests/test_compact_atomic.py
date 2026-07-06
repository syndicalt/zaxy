"""Tests for atomic in-place rewrites of `zaxy compact` (fix A3).

`compact` rewrites the canonical append-only Eventloom log. The fix replaced a
plain `open(out_path, "w")` (which truncates the target immediately) with a
write-to-sibling-temp-file + fsync + os.replace() sequence, so a crash or I/O
error mid-write must never leave the canonical log truncated or corrupted.
"""

from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from zaxy.cli.runtime import app
from zaxy.event import EventLog

runner = CliRunner()


def _seed_log(log_path: Path) -> list:
    log = EventLog(str(log_path))
    log.append("goal.created", actor="planner", payload={"note": "first"})
    log.append("goal.updated", actor="planner", payload={"note": "second"})
    log.append("goal.completed", actor="planner", payload={"note": "third"})
    return log.read_all()


def test_compact_in_place_preserves_events(tmp_path: Path) -> None:
    """Happy path: compacting in place keeps every original event's content and order."""
    log_path = tmp_path / "session.jsonl"
    original_events = _seed_log(log_path)

    result = runner.invoke(app, ["compact", str(log_path)])

    assert result.exit_code == 0, result.output
    compacted_events = EventLog(str(log_path)).read_all()
    # compact() appends one lifecycle "compaction completed" event after the rewrite.
    assert len(compacted_events) == len(original_events) + 1
    for original, compacted in zip(original_events, compacted_events, strict=False):
        assert compacted.seq == original.seq
        assert compacted.type == original.type
        assert compacted.actor == original.actor
        assert compacted.payload == original.payload


def test_compact_in_place_failure_leaves_original_log_byte_identical(
    tmp_path: Path, monkeypatch
) -> None:
    """A failure during the atomic swap must never truncate the canonical log.

    This is the core regression test for A3: the old implementation opened the
    target path with `open(out_path, "w")`, which truncates it immediately, so
    any failure afterwards (or even before any bytes were written) left a
    zero-byte or partially written log. The fix always builds the full
    rewrite in a sibling temp file first and only swaps it in via
    `os.replace()`; if that swap fails, the original log must be completely
    untouched and the temp file must be cleaned up.
    """
    log_path = tmp_path / "session.jsonl"
    _seed_log(log_path)
    original_bytes = log_path.read_bytes()
    assert original_bytes  # sanity: fixture actually wrote something

    def _raise_on_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated os.replace failure")

    monkeypatch.setattr(os, "replace", _raise_on_replace)

    result = runner.invoke(app, ["compact", str(log_path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, OSError)
    assert log_path.read_bytes() == original_bytes

    leftover_tmp_files = list(tmp_path.glob("*.compact.tmp"))
    assert leftover_tmp_files == []
