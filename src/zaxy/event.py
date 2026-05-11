"""Eventloom-compatible event log I/O with hash-chain integrity.

This module provides a Python-native interface to Eventloom's append-only
JSONL format. It supports:

- Reading and writing typed events with SHA-256 hash chains.
- Cross-process append locking (fcntl on Unix, portalocker fallback).
- Integrity verification and deterministic replay.
- Handoff summary generation for agent session resumption.

Example::

    log = EventLog(".eventloom/agent.jsonl")
    log.append("goal.created", actor="user", payload={"title": "Ship it"})
    replay = log.replay()
    assert replay.integrity.ok
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from zaxy.security import secure_payload


class EventSecurity(BaseModel):
    """Security classification for a durable Eventloom payload."""

    sensitivity: str = Field(default="public", description="Payload sensitivity tier.")
    redacted_paths: list[str] = Field(
        default_factory=list,
        description="Payload paths redacted before the event was sealed.",
    )


class Event(BaseModel):
    """A single Eventloom event.

    Events are immutable once written. The hash field seals the event
    and links it to the previous event via prev_hash, forming a chain.
    """

    seq: int = Field(..., ge=1, description="Monotonic sequence number (1-indexed).")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp.")
    type: str = Field(..., min_length=1, description="Event type, e.g. 'goal.created'.")
    actor: str = Field(..., min_length=1, description="Actor that emitted the event.")
    thread: str = Field(default="default", description="Logical thread / session ID.")
    payload: dict[str, Any] = Field(default_factory=dict, description="Structured payload.")
    security: EventSecurity | None = Field(
        default=None,
        description="Security classification metadata for the payload.",
    )
    prev_hash: str | None = Field(default=None, description="Hash of previous event.")
    hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 of this event.")

    @field_validator("timestamp")
    @classmethod
    def _validate_timestamp(cls, v: str) -> str:
        """Ensure timestamp is valid ISO-8601."""
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    def canonical(self) -> bytes:
        """Return the canonical byte representation used for hashing.

        The hash covers all fields *except* the hash field itself, in a
        deterministic JSON serialization with sorted keys and no whitespace.
        """
        obj = {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "type": self.type,
            "actor": self.actor,
            "thread": self.thread,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
        }
        if self.security is not None:
            obj["security"] = self.security.model_dump()
        return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def verify(self) -> bool:
        """Verify that the stored hash matches the canonical content."""
        expected = hashlib.sha256(self.canonical()).hexdigest()
        return self.hash == expected


class IntegrityReport(BaseModel):
    """Result of verifying an event log."""

    ok: bool
    total_events: int
    broken_at_seq: int | None = None
    broken_reason: str | None = None


class ReplayResult(BaseModel):
    """Result of replaying an event log."""

    events: list[Event]
    integrity: IntegrityReport
    projection: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    """Append-only event log backed by a JSONL file.

    Thread-safe and cross-process-safe on Unix via fcntl advisory locking.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def _lock(self, fd: int, exclusive: bool = True) -> None:
        """Advisory lock on the file descriptor."""
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, op)

    def _unlock(self, fd: int) -> None:
        """Release advisory lock."""
        fcntl.flock(fd, fcntl.LOCK_UN)

    def read_all(self) -> list[Event]:
        """Read every event from the log."""
        if not self.path.exists():
            return []

        events: list[Event] = []
        with open(self.path, encoding="utf-8") as fh:
            self._lock(fh.fileno(), exclusive=False)
            try:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    events.append(Event.model_validate_json(line))
            finally:
                self._unlock(fh.fileno())
        return events

    def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        thread: str = "default",
        timestamp: datetime | None = None,
    ) -> Event:
        """Append a new event to the log.

        The sequence number and hash chain are computed automatically.
        """
        return self.append_many(
            [
                {
                    "event_type": event_type,
                    "actor": actor,
                    "payload": payload,
                    "thread": thread,
                    "timestamp": timestamp,
                }
            ]
        )[0]

    def append_many(self, items: list[dict[str, Any]]) -> list[Event]:
        """Append multiple events while holding the append lock.

        The hash chain must be based on the locked file tail. Building the
        batch before the exclusive lock leaves a race window where another
        writer can append and make the precomputed ``prev_hash`` stale.
        """
        if not items:
            return []

        with open(self.path, "a+", encoding="utf-8") as fh:
            self._lock(fh.fileno(), exclusive=True)
            try:
                fh.seek(0)
                lines = fh.readlines()
                seq = 1
                prev_hash: str | None = None
                if lines:
                    last = Event.model_validate_json(lines[-1])
                    seq = last.seq + 1
                    prev_hash = last.hash

                batch: list[Event] = []
                for item in items:
                    event = self._build_event(
                        seq=seq,
                        prev_hash=prev_hash,
                        event_type=str(item["event_type"]),
                        actor=str(item["actor"]),
                        payload=item.get("payload") if isinstance(item.get("payload"), dict) else None,
                        thread=str(item.get("thread", "default")),
                        timestamp=item.get("timestamp") if isinstance(item.get("timestamp"), datetime) else None,
                    )
                    batch.append(event)
                    seq = event.seq + 1
                    prev_hash = event.hash

                fh.seek(0, os.SEEK_END)
                fh.writelines(event.model_dump_json() + "\n" for event in batch)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                self._unlock(fh.fileno())

        return batch

    def _build_event(
        self,
        *,
        seq: int,
        prev_hash: str | None,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None,
        thread: str,
        timestamp: datetime | None,
    ) -> Event:
        ts = (timestamp or datetime.now(UTC)).isoformat().replace("+00:00", "Z")
        secured = secure_payload(payload or {})
        raw = {
            "seq": seq,
            "timestamp": ts,
            "type": event_type,
            "actor": actor,
            "thread": thread,
            "payload": secured.payload,
            "security": {
                "sensitivity": secured.sensitivity,
                "redacted_paths": secured.redacted_paths,
            },
            "prev_hash": prev_hash,
            "hash": "0" * 64,
        }
        tmp = Event.model_validate(raw)
        raw["hash"] = hashlib.sha256(tmp.canonical()).hexdigest()
        return Event.model_validate(raw)

    # ------------------------------------------------------------------
    # Integrity & Replay
    # ------------------------------------------------------------------

    def verify(self) -> IntegrityReport:
        """Verify the entire log: hash chain + individual event seals."""
        events = self.read_all()
        total = len(events)

        if total == 0:
            return IntegrityReport(ok=True, total_events=0)

        prev_hash: str | None = None
        for i, ev in enumerate(events, start=1):
            if not ev.verify():
                return IntegrityReport(
                    ok=False,
                    total_events=total,
                    broken_at_seq=ev.seq,
                    broken_reason=f"Event {ev.seq} hash mismatch",
                )
            if i == 1:
                if ev.prev_hash is not None:
                    return IntegrityReport(
                        ok=False,
                        total_events=total,
                        broken_at_seq=ev.seq,
                        broken_reason="First event has prev_hash set",
                    )
            else:
                if ev.prev_hash != prev_hash:
                    return IntegrityReport(
                        ok=False,
                        total_events=total,
                        broken_at_seq=ev.seq,
                        broken_reason=f"Event {ev.seq} prev_hash does not link to previous",
                    )
            prev_hash = ev.hash

        return IntegrityReport(ok=True, total_events=total)

    def replay(self, from_seq: int = 1) -> ReplayResult:
        """Replay events from a given sequence number."""
        events = self.read_all()
        filtered = [e for e in events if e.seq >= from_seq]
        integrity = self.verify()
        return ReplayResult(events=filtered, integrity=integrity)

    # ------------------------------------------------------------------
    # Handoff & Summaries
    # ------------------------------------------------------------------

    def handoff_summary(self) -> dict[str, Any]:
        """Generate a concise handoff summary from the log.

        Returns task state, telemetry, and next actions suitable for
        resuming an agent session.
        """
        events = self.read_all()
        goals = [e for e in events if e.type.startswith("goal.")]
        tasks = [e for e in events if e.type.startswith("task.")]
        completions = [e for e in events if e.type.startswith("task.completed")]

        open_tasks = []
        for t in tasks:
            tid = t.payload.get("taskId")
            if tid and not any(c.payload.get("taskId") == tid for c in completions):
                open_tasks.append(t.model_dump(include={"type", "actor", "payload"}))

        return {
            "event_count": len(events),
            "goals": [g.payload.get("title", "untitled") for g in goals],
            "open_tasks": open_tasks,
            "last_actor": events[-1].actor if events else None,
            "last_timestamp": events[-1].timestamp if events else None,
        }
