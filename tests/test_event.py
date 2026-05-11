"""Tests for zaxy.event — Eventloom JSONL I/O and hash-chain integrity.

Karpathy rule: every function gets a test. We test the happy path, the
error paths, and the edge cases (empty files, races, broken chains)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from freezegun import freeze_time
from pydantic import ValidationError

from zaxy.event import Event, EventLog, ReplayResult

# ------------------------------------------------------------------
# Event model tests
# ------------------------------------------------------------------

class TestEvent:
    """Unit tests for the Event Pydantic model."""

    def test_minimal_event_valid(self) -> None:
        """A correctly formed event should validate."""
        ev = Event(
            seq=1,
            timestamp="2024-01-01T00:00:00Z",
            type="goal.created",
            actor="user",
            hash="a" * 64,
        )
        assert ev.seq == 1
        assert ev.thread == "default"
        assert ev.payload == {}

    def test_event_with_payload(self) -> None:
        """Payload should be preserved exactly."""
        ev = Event(
            seq=2,
            timestamp="2024-01-01T00:00:00Z",
            type="task.proposed",
            actor="codex",
            thread="t1",
            payload={"taskId": "abc", "title": "Do it"},
            prev_hash="b" * 64,
            hash="c" * 64,
        )
        assert ev.payload["taskId"] == "abc"
        assert ev.prev_hash == "b" * 64

    def test_seq_must_be_positive(self) -> None:
        """Sequence numbers must be >= 1."""
        with pytest.raises(ValidationError):
            Event(seq=0, timestamp="2024-01-01T00:00:00Z", type="x", actor="y", hash="a" * 64)

    def test_timestamp_must_be_iso(self) -> None:
        """Timestamps must be valid ISO-8601."""
        with pytest.raises(ValidationError):
            Event(seq=1, timestamp="not-a-date", type="x", actor="y", hash="a" * 64)

    def test_canonical_deterministic(self) -> None:
        """Canonical JSON must be deterministic (sorted keys, no spaces)."""
        ev = Event(
            seq=1,
            timestamp="2024-01-01T00:00:00Z",
            type="a",
            actor="b",
            payload={"z": 1, "a": 2},
            hash="c" * 64,
        )
        canonical = ev.canonical()
        assert b'"a":2' in canonical
        assert b'"z":1' in canonical
        # Keys should be sorted
        z_pos = canonical.index(b'"z"')
        a_pos = canonical.index(b'"a"')
        assert a_pos < z_pos

    def test_verify_correct_hash(self) -> None:
        """An event with a correctly computed hash should verify."""
        # Build with dummy hash first to get canonical form
        ev_tmp = Event(
            seq=1,
            timestamp="2024-01-01T00:00:00Z",
            type="goal.created",
            actor="user",
            hash="0" * 64,
        )
        import hashlib

        correct = hashlib.sha256(ev_tmp.canonical()).hexdigest()
        ev = ev_tmp.model_copy(update={"hash": correct})
        assert ev.verify() is True

    def test_legacy_event_without_security_metadata_still_verifies(self) -> None:
        """Older logs without security metadata should remain replayable."""
        ev_tmp = Event(
            seq=1,
            timestamp="2024-01-01T00:00:00Z",
            type="goal.created",
            actor="user",
            payload={"title": "Legacy"},
            hash="0" * 64,
        )
        import hashlib

        legacy_hash = hashlib.sha256(ev_tmp.canonical()).hexdigest()
        ev = ev_tmp.model_copy(update={"hash": legacy_hash})
        assert ev.security is None
        assert ev.verify() is True

    def test_verify_incorrect_hash(self) -> None:
        """An event with a wrong hash should fail verification."""
        ev = Event(
            seq=1,
            timestamp="2024-01-01T00:00:00Z",
            type="goal.created",
            actor="user",
            hash="0" * 64,
        )
        assert ev.verify() is False


# ------------------------------------------------------------------
# EventLog I/O tests
# ------------------------------------------------------------------

class TestEventLogIO:
    """Tests for reading and writing event logs."""

    def test_read_empty_file(self, tmp_eventlog: EventLog) -> None:
        """Reading a non-existent or empty log should return []."""
        assert tmp_eventlog.read_all() == []

    def test_append_creates_event(self, tmp_eventlog: EventLog) -> None:
        """Appending should return an Event with seq=1 and a valid hash."""
        ev = tmp_eventlog.append("goal.created", actor="user", payload={"title": "T"})
        assert ev.seq == 1
        assert ev.type == "goal.created"
        assert ev.verify() is True
        assert ev.prev_hash is None

    def test_append_redacts_secret_payload_values_before_sealing(
        self, tmp_eventlog: EventLog
    ) -> None:
        """Secrets should never be persisted into the immutable Eventloom log."""
        ev = tmp_eventlog.append(
            "credential.observed",
            actor="user",
            payload={
                "title": "Configure provider",
                "api_key": "sk-secret-value",
                "nested": {"authorization": "Bearer live-token"},
            },
        )

        assert ev.payload == {
            "title": "Configure provider",
            "api_key": "[REDACTED]",
            "nested": {"authorization": "[REDACTED]"},
        }
        assert ev.security.sensitivity == "restricted"
        assert ev.security.redacted_paths == ["api_key", "nested.authorization"]
        assert ev.verify() is True

        loaded = tmp_eventlog.read_all()[0]
        assert loaded.payload["api_key"] == "[REDACTED]"
        assert loaded.security.sensitivity == "restricted"
        assert loaded.verify() is True

    def test_append_increments_seq(self, tmp_eventlog: EventLog) -> None:
        """Multiple appends should produce monotonic seq numbers."""
        e1 = tmp_eventlog.append("a", actor="x")
        e2 = tmp_eventlog.append("b", actor="y")
        e3 = tmp_eventlog.append("c", actor="z")
        assert e1.seq == 1
        assert e2.seq == 2
        assert e3.seq == 3
        assert e2.prev_hash == e1.hash
        assert e3.prev_hash == e2.hash

    def test_append_many_writes_hash_linked_batch(self, tmp_eventlog: EventLog) -> None:
        """Batch append should preserve monotonic seq and hash-chain integrity."""
        first = tmp_eventlog.append("a", actor="x")

        events = tmp_eventlog.append_many(
            [
                {"event_type": "b", "actor": "y", "payload": {"n": 2}, "thread": "session-1"},
                {"event_type": "c", "actor": "z", "payload": {"n": 3}, "thread": "session-1"},
            ]
        )

        assert [event.seq for event in events] == [2, 3]
        assert events[0].prev_hash == first.hash
        assert events[1].prev_hash == events[0].hash
        assert [event.type for event in tmp_eventlog.read_all()] == ["a", "b", "c"]
        assert tmp_eventlog.verify().ok is True

    def test_append_many_rebases_when_another_writer_appends_before_lock(
        self, tmp_path
    ) -> None:
        """A writer that loses the append race should rebase on the locked tail."""

        class RacingEventLog(EventLog):
            injected = False

            def _lock(self, fd: int, *, exclusive: bool = False) -> None:
                if exclusive and not self.injected:
                    self.injected = True
                    EventLog(self.path).append("concurrent", actor="watcher")
                super()._lock(fd, exclusive=exclusive)

        log_path = tmp_path / "events.jsonl"
        EventLog(log_path).append("first", actor="setup")
        racing = RacingEventLog(log_path)

        events = racing.append_many(
            [
                {"event_type": "manual.one", "actor": "codex"},
                {"event_type": "manual.two", "actor": "codex"},
            ]
        )

        written = EventLog(log_path).read_all()
        assert [event.type for event in written] == [
            "first",
            "concurrent",
            "manual.one",
            "manual.two",
        ]
        assert [event.seq for event in events] == [3, 4]
        assert events[0].prev_hash == written[1].hash
        assert written[-1].hash == events[-1].hash
        assert EventLog(log_path).verify().ok is True

    def test_read_roundtrip(self, tmp_eventlog: EventLog) -> None:
        """Events written should be identical when read back."""
        original = tmp_eventlog.append("goal.created", actor="u", payload={"x": 1})
        loaded = tmp_eventlog.read_all()
        assert len(loaded) == 1
        assert loaded[0].seq == original.seq
        assert loaded[0].hash == original.hash
        assert loaded[0].payload == original.payload

    def test_custom_timestamp(self, tmp_eventlog: EventLog) -> None:
        """Users can supply an explicit timestamp."""
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        ev = tmp_eventlog.append("x", actor="y", timestamp=dt)
        assert ev.timestamp == "2024-06-15T12:00:00Z"

    @freeze_time("2024-01-01T00:00:00Z")
    def test_default_timestamp_is_now(self, tmp_eventlog: EventLog) -> None:
        """If no timestamp is given, use UTC now."""
        ev = tmp_eventlog.append("x", actor="y")
        assert ev.timestamp == "2024-01-01T00:00:00Z"

    def test_thread_default(self, tmp_eventlog: EventLog) -> None:
        """Default thread should be 'default'."""
        ev = tmp_eventlog.append("x", actor="y")
        assert ev.thread == "default"

    def test_custom_thread(self, tmp_eventlog: EventLog) -> None:
        """Thread can be overridden."""
        ev = tmp_eventlog.append("x", actor="y", thread="session-42")
        assert ev.thread == "session-42"

    def test_empty_payload_default(self, tmp_eventlog: EventLog) -> None:
        """Payload defaults to empty dict."""
        ev = tmp_eventlog.append("x", actor="y")
        assert ev.payload == {}


# ------------------------------------------------------------------
# Integrity tests
# ------------------------------------------------------------------

class TestIntegrity:
    """Tests for hash-chain verification."""

    def test_empty_log_is_valid(self, tmp_eventlog: EventLog) -> None:
        """An empty log passes verification."""
        report = tmp_eventlog.verify()
        assert report.ok is True
        assert report.total_events == 0

    def test_single_event_valid(self, tmp_eventlog: EventLog) -> None:
        """A single event with no prev_hash should verify."""
        tmp_eventlog.append("x", actor="y")
        report = tmp_eventlog.verify()
        assert report.ok is True
        assert report.total_events == 1
        assert report.broken_at_seq is None

    def test_chain_valid(self, tmp_eventlog: EventLog) -> None:
        """A multi-event chain with correct links should verify."""
        for i in range(5):
            tmp_eventlog.append(f"event.{i}", actor="a")
        report = tmp_eventlog.verify()
        assert report.ok is True
        assert report.total_events == 5

    def test_broken_hash(self, tmp_eventlog: EventLog) -> None:
        """Tampering with an event hash should break verification."""
        tmp_eventlog.append("x", actor="y")
        # Tamper: overwrite file with bad hash
        path = tmp_eventlog.path
        with open(path, "r+") as fh:
            data = json.loads(fh.readline())
            data["hash"] = "0" * 64
            fh.seek(0)
            fh.write(json.dumps(data) + "\n")
            fh.truncate()

        report = tmp_eventlog.verify()
        assert report.ok is False
        assert report.broken_at_seq == 1
        assert "hash mismatch" in (report.broken_reason or "")

    def test_broken_prev_hash_link(self, tmp_eventlog: EventLog) -> None:
        """Breaking the prev_hash chain should be detected."""
        tmp_eventlog.append("first", actor="a")
        tmp_eventlog.append("second", actor="b")

        # Tamper second event's prev_hash, but recompute its hash so
        # only the chain link is broken, not the event seal itself.
        path = tmp_eventlog.path
        lines = path.read_text().strip().split("\n")
        second = json.loads(lines[1])
        second["prev_hash"] = "deadbeef" * 8
        # Recompute hash with the tampered prev_hash
        import hashlib

        from zaxy.event import Event
        tmp = Event.model_validate({**second, "hash": "0" * 64})
        second["hash"] = hashlib.sha256(tmp.canonical()).hexdigest()
        lines[1] = json.dumps(second)
        path.write_text("\n".join(lines) + "\n")

        report = tmp_eventlog.verify()
        assert report.ok is False
        assert report.broken_at_seq == 2
        assert "does not link" in (report.broken_reason or "")

    def test_first_event_with_prev_hash(self, tmp_eventlog: EventLog) -> None:
        """The first event must not have a prev_hash."""
        import hashlib

        from zaxy.event import Event

        # Build a first event with prev_hash set, but with a valid hash
        bad = {
            "seq": 1,
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "x",
            "actor": "y",
            "payload": {},
            "prev_hash": "a" * 64,
            "hash": "0" * 64,
        }
        tmp = Event.model_validate(bad)
        bad["hash"] = hashlib.sha256(tmp.canonical()).hexdigest()

        path = tmp_eventlog.path
        path.write_text(json.dumps(bad) + "\n")

        report = tmp_eventlog.verify()
        assert report.ok is False
        assert "First event has prev_hash" in (report.broken_reason or "")


# ------------------------------------------------------------------
# Replay tests
# ------------------------------------------------------------------

class TestReplay:
    """Tests for deterministic replay."""

    def test_replay_all(self, tmp_eventlog: EventLog) -> None:
        """Replay without from_seq should return all events."""
        tmp_eventlog.append("a", actor="x")
        tmp_eventlog.append("b", actor="y")
        result = tmp_eventlog.replay()
        assert isinstance(result, ReplayResult)
        assert len(result.events) == 2
        assert result.integrity.ok is True

    def test_replay_from_seq(self, tmp_eventlog: EventLog) -> None:
        """Replay from_seq should filter events."""
        tmp_eventlog.append("a", actor="x")
        tmp_eventlog.append("b", actor="y")
        tmp_eventlog.append("c", actor="z")
        result = tmp_eventlog.replay(from_seq=2)
        assert len(result.events) == 2
        assert result.events[0].type == "b"
        assert result.events[1].type == "c"

    def test_replay_from_seq_greater_than_total(self, tmp_eventlog: EventLog) -> None:
        """from_seq > total should return empty list but valid integrity."""
        tmp_eventlog.append("a", actor="x")
        result = tmp_eventlog.replay(from_seq=99)
        assert result.events == []
        assert result.integrity.ok is True


# ------------------------------------------------------------------
# Handoff summary tests
# ------------------------------------------------------------------

class TestHandoff:
    """Tests for handoff summary generation."""

    def test_empty_summary(self, tmp_eventlog: EventLog) -> None:
        """Empty log should produce empty summary."""
        summary = tmp_eventlog.handoff_summary()
        assert summary["event_count"] == 0
        assert summary["goals"] == []
        assert summary["open_tasks"] == []
        assert summary["last_actor"] is None

    def test_summary_with_goals(self, tmp_eventlog: EventLog) -> None:
        """Goals should be extracted from the log."""
        tmp_eventlog.append("goal.created", actor="user", payload={"title": "Ship it"})
        tmp_eventlog.append("goal.created", actor="user", payload={"title": "Fix bug"})
        summary = tmp_eventlog.handoff_summary()
        assert summary["event_count"] == 2
        assert summary["goals"] == ["Ship it", "Fix bug"]

    def test_open_vs_completed_tasks(self, tmp_eventlog: EventLog) -> None:
        """Only tasks without completions should be open."""
        tmp_eventlog.append(
            "task.proposed", actor="codex", payload={"taskId": "t1", "title": "A"}
        )
        tmp_eventlog.append(
            "task.proposed", actor="codex", payload={"taskId": "t2", "title": "B"}
        )
        tmp_eventlog.append("task.completed", actor="codex", payload={"taskId": "t1"})
        summary = tmp_eventlog.handoff_summary()
        assert len(summary["open_tasks"]) == 1
        assert summary["open_tasks"][0]["payload"]["taskId"] == "t2"

    def test_last_actor_and_timestamp(self, tmp_eventlog: EventLog) -> None:
        """Summary should capture last actor and timestamp."""
        tmp_eventlog.append("a", actor="alice")
        tmp_eventlog.append("b", actor="bob")
        summary = tmp_eventlog.handoff_summary()
        assert summary["last_actor"] == "bob"
        assert summary["last_timestamp"] is not None


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and defensive checks."""

    def test_corrupted_json_line_ignored_or_fails(self, tmp_eventlog: EventLog) -> None:
        """A corrupted JSON line should raise on read (fail fast)."""
        with open(tmp_eventlog.path, "w") as fh:
            fh.write("this is not json\n")
        with pytest.raises((json.JSONDecodeError, ValidationError)):
            tmp_eventlog.read_all()

    def test_reappend_after_read(self, tmp_eventlog: EventLog) -> None:
        """Appending after reading should maintain correct chain."""
        e1 = tmp_eventlog.append("a", actor="x")
        _ = tmp_eventlog.read_all()
        e2 = tmp_eventlog.append("b", actor="y")
        assert e2.prev_hash == e1.hash
        assert tmp_eventlog.verify().ok is True

    def test_many_events(self, tmp_eventlog: EventLog) -> None:
        """The log should handle many events efficiently."""
        for i in range(100):
            tmp_eventlog.append(f"event.{i}", actor="bot")
        events = tmp_eventlog.read_all()
        assert len(events) == 100
        assert events[-1].seq == 100
        assert tmp_eventlog.verify().ok is True
