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
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, Field, field_validator

from zaxy.security import secure_payload, validate_event_text, validate_payload

_EVENTLOOM_V1_TYPE_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9]*)+$")
_EVENTLOOM_V1_HASH_RE = re.compile(r"^sha256:[a-f0-9]{64}$")
_ZAXY_SECURITY_PAYLOAD_KEY = "__zaxy_security"


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
    id: str | None = Field(default=None, description="Eventloom v1 event id, when present.")
    parent_event_id: str | None = Field(
        default=None,
        description="Eventloom v1 parentEventId, when present.",
    )
    caused_by: list[str] = Field(
        default_factory=list,
        description="Eventloom v1 causedBy ids, when present.",
    )
    envelope_version: str = Field(
        default="zaxy.legacy",
        description="Normalized source envelope version.",
    )

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
        if self.envelope_version == "eventloom.v1":
            expected = _strip_sha256_prefix(
                _eventloom_v1_hash(self.to_eventloom_v1_unsigned(), _with_sha256_prefix(self.prev_hash))
            )
        else:
            expected = hashlib.sha256(self.canonical()).hexdigest()
        return self.hash == expected

    def to_eventloom_v1_unsigned(self) -> dict[str, Any]:
        """Return this event as an unsigned Eventloom v1 envelope."""
        payload = dict(self.payload)
        if self.security is not None:
            payload[_ZAXY_SECURITY_PAYLOAD_KEY] = self.security.model_dump()
        return {
            "id": self.id or _eventloom_v1_event_id(self.seq, self.type, self.actor, self.timestamp),
            "type": self.type,
            "actorId": self.actor,
            "threadId": self.thread,
            "parentEventId": self.parent_event_id,
            "causedBy": list(self.caused_by),
            "timestamp": self.timestamp,
            "payload": payload,
        }


class IntegrityReport(BaseModel):
    """Result of verifying an event log."""

    ok: bool
    total_events: int
    broken_at_seq: int | None = None
    broken_reason: str | None = None


class ReplayResult(BaseModel):
    """Result of replaying an event log."""

    events: list[Event]
    integrity: IntegrityReport | None
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
                    events.append(_event_from_json_line(line, seq_hint=len(events) + 1))
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
                last_line = _read_last_line(fh)
                seq = 1
                prev_hash: str | None = None
                if last_line:
                    last = _event_from_json_line(last_line)
                    seq = last.seq + 1
                    prev_hash = last.hash
                write_v1 = _should_write_eventloom_v1_from_tail(last_line, items)

                batch: list[Event] = []
                for item in items:
                    raw_payload = item.get("payload")
                    event = self._build_event(
                        seq=seq,
                        prev_hash=prev_hash,
                        event_type=validate_event_text(item["event_type"], "event_type"),
                        actor=validate_event_text(item["actor"], "actor"),
                        payload=validate_payload(raw_payload) if raw_payload is not None else None,
                        thread=validate_event_text(item.get("thread", "default"), "thread"),
                        timestamp=item.get("timestamp") if isinstance(item.get("timestamp"), datetime) else None,
                        envelope_version="eventloom.v1" if write_v1 else "zaxy.legacy",
                    )
                    batch.append(event)
                    seq = event.seq + 1
                    prev_hash = event.hash

                fh.seek(0, os.SEEK_END)
                if write_v1:
                    fh.writelines(_eventloom_v1_json(event) + "\n" for event in batch)
                else:
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
        envelope_version: str = "zaxy.legacy",
    ) -> Event:
        ts = (timestamp or datetime.now(UTC)).isoformat().replace("+00:00", "Z")
        secured = secure_payload(payload or {})
        security = {
            "sensitivity": secured.sensitivity,
            "redacted_paths": secured.redacted_paths,
        }
        raw = {
            "seq": seq,
            "timestamp": ts,
            "type": event_type,
            "actor": actor,
            "thread": thread,
            "payload": secured.payload,
            "security": security,
            "prev_hash": prev_hash,
            "hash": "0" * 64,
            "envelope_version": envelope_version,
        }
        tmp = Event.model_validate(raw)
        if envelope_version == "eventloom.v1":
            event_id = _eventloom_v1_event_id(seq, event_type, actor, ts)
            raw["id"] = event_id
            tmp = Event.model_validate(raw)
            raw["hash"] = _strip_sha256_prefix(
                _eventloom_v1_hash(tmp.to_eventloom_v1_unsigned(), _with_sha256_prefix(prev_hash))
            )
        else:
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
            if ev.seq != i:
                return IntegrityReport(
                    ok=False,
                    total_events=total,
                    broken_at_seq=ev.seq,
                    broken_reason=f"Event sequence expected {i} but found {ev.seq}",
                )
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

    def replay(
        self,
        from_seq: int = 1,
        to_seq: int | None = None,
        *,
        verify_integrity: bool = True,
    ) -> ReplayResult:
        """Replay events from an inclusive sequence window."""
        if from_seq < 1:
            raise ValueError("from_seq must be >= 1")
        if to_seq is not None and to_seq < 1:
            raise ValueError("to_seq must be >= 1")
        if to_seq is not None and from_seq > to_seq:
            raise ValueError("from_seq must be <= to_seq")
        events = self.read_all()
        filtered = [
            event
            for event in events
            if event.seq >= from_seq and (to_seq is None or event.seq <= to_seq)
        ]
        integrity = self.verify() if verify_integrity else None
        return ReplayResult(events=filtered, integrity=integrity)

    def last_event(self) -> Event | None:
        """Return the current tail event without parsing the full log."""
        if not self.path.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as fh:
            self._lock(fh.fileno(), exclusive=False)
            try:
                last_line = _read_last_line(fh)
                return _event_from_json_line(last_line) if last_line else None
            finally:
                self._unlock(fh.fileno())

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


def _read_last_line(fh: TextIO) -> str | None:
    """Read the last non-empty JSONL line from a locked file handle."""
    fd = fh.fileno()
    size = os.fstat(fd).st_size
    if size == 0:
        return None

    end = size
    while end > 0:
        char = os.pread(fd, 1, end - 1)
        if char not in {b"\n", b"\r"}:
            break
        end -= 1
    if end == 0:
        return None

    chunk_size = 8192
    chunks: list[bytes] = []
    position = end
    while position > 0:
        read_size = min(chunk_size, position)
        position -= read_size
        chunk = os.pread(fd, read_size, position)
        newline_at = chunk.rfind(b"\n")
        if newline_at != -1:
            chunks.insert(0, chunk[newline_at + 1 :])
            break
        chunks.insert(0, chunk)

    line = b"".join(chunks)
    return line.decode("utf-8") if line else None


def _event_from_json_line(line: str, *, seq_hint: int | None = None) -> Event:
    record = json.loads(line)
    if not isinstance(record, dict):
        return Event.model_validate(record)
    if _is_eventloom_v1_record(record):
        return _event_from_eventloom_v1(record, seq_hint=seq_hint)
    return Event.model_validate(record)


def _is_eventloom_v1_record(record: dict[str, Any]) -> bool:
    return {
        "id",
        "type",
        "actorId",
        "threadId",
        "timestamp",
        "payload",
        "integrity",
    }.issubset(record)


def _event_from_eventloom_v1(record: dict[str, Any], *, seq_hint: int | None) -> Event:
    integrity = record.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("Eventloom v1 event is missing integrity metadata")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Eventloom v1 event payload must be an object")

    payload_copy = dict(payload)
    security: dict[str, Any] | None = None
    raw_security = payload_copy.pop(_ZAXY_SECURITY_PAYLOAD_KEY, None)
    if isinstance(raw_security, dict):
        security = raw_security
    seq = seq_hint or _eventloom_v1_seq_from_id(record.get("id"))
    if seq is None:
        raise ValueError("Eventloom v1 event requires a sequence hint or Zaxy event id")

    event = Event.model_validate(
        {
            "seq": seq,
            "timestamp": record["timestamp"],
            "type": record["type"],
            "actor": record["actorId"],
            "thread": record["threadId"],
            "payload": payload_copy,
            "security": security,
            "prev_hash": _strip_sha256_prefix(integrity.get("previousHash")),
            "hash": _strip_sha256_prefix(integrity.get("hash")),
            "id": record["id"],
            "parent_event_id": record.get("parentEventId"),
            "caused_by": record.get("causedBy") or [],
            "envelope_version": "eventloom.v1",
        }
    )
    return event


def _eventloom_v1_seq_from_id(event_id: Any) -> int | None:
    if not isinstance(event_id, str):
        return None
    match = re.fullmatch(r"evt_zaxy_(\d{12})_[a-f0-9]{16}", event_id)
    if match is None:
        return None
    return int(match.group(1))


def _should_write_eventloom_v1_from_tail(
    last_line: str | None,
    items: list[dict[str, Any]],
) -> bool:
    if last_line:
        try:
            last = json.loads(last_line)
        except json.JSONDecodeError:
            return False
        if not isinstance(last, dict) or not _is_eventloom_v1_record(last):
            return False
    return all(_EVENTLOOM_V1_TYPE_RE.fullmatch(str(item.get("event_type", ""))) for item in items)


def _eventloom_v1_json(event: Event) -> str:
    envelope = event.to_eventloom_v1_unsigned()
    envelope["integrity"] = {
        "hash": _with_sha256_prefix(event.hash),
        "previousHash": _with_sha256_prefix(event.prev_hash),
    }
    return json.dumps(envelope, separators=(",", ":"))


def _eventloom_v1_event_id(seq: int, event_type: str, actor: str, timestamp: str) -> str:
    seed = json.dumps(
        {"actor": actor, "seq": seq, "timestamp": timestamp, "type": event_type},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"evt_zaxy_{seq:012d}_{digest}"


def _eventloom_v1_hash(unsigned_event: dict[str, Any], previous_hash: str | None) -> str:
    canonical = json.dumps(
        {"event": unsigned_event, "previousHash": previous_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _strip_sha256_prefix(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("hash value must be a string or None")
    if value.startswith("sha256:"):
        return value.removeprefix("sha256:")
    return value


def _with_sha256_prefix(value: str | None) -> str | None:
    if value is None:
        return None
    if _EVENTLOOM_V1_HASH_RE.fullmatch(value):
        return value
    return f"sha256:{value}"
