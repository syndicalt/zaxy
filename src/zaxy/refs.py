"""Git-style memory refs backed by Eventloom events."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from zaxy.event import Event, EventLog
from zaxy.security import validate_session_id

REF_NAME_PATTERN = re.compile(r"^(HEAD|refs/[A-Za-z0-9._/-]{1,240})$")
REF_LOG_SESSION = "__refs__"


@dataclass(frozen=True)
class MemoryRef:
    """A named pointer to an Eventloom event identity."""

    name: str
    session_id: str
    target_seq: int
    target_hash: str
    ref_type: str
    updated_at: str

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable payload."""
        return asdict(self)


class MemoryRefStore:
    """Durable memory ref store using an append-only Eventloom ref log."""

    def __init__(self, eventloom_path: str | Path) -> None:
        self._base = Path(eventloom_path)
        self._log = EventLog(self._base / f"{REF_LOG_SESSION}.jsonl")

    def update_ref(
        self,
        name: str,
        *,
        session_id: str,
        target_seq: int,
        target_hash: str,
        ref_type: str = "ref",
        actor: str = "zaxy",
    ) -> Event:
        """Append a ref update event and return its Eventloom event."""
        safe_name = validate_ref_name(name)
        safe_session = validate_session_id(session_id)
        if not isinstance(target_seq, int) or target_seq < 1:
            raise ValueError("target_seq must be a positive integer")
        if not isinstance(target_hash, str) or not re.fullmatch(r"[a-fA-F0-9]{12,64}", target_hash):
            raise ValueError("target_hash must be a 12-64 character hex hash")
        payload = {
            "name": safe_name,
            "session_id": safe_session,
            "target_seq": target_seq,
            "target_hash": target_hash,
            "ref_type": _validate_ref_type(ref_type),
        }
        return self._log.append(
            "memory.ref.updated",
            actor=actor,
            payload=payload,
            thread=REF_LOG_SESSION,
        )

    def resolve(self, name: str) -> MemoryRef | None:
        """Return the latest event identity for a ref name."""
        safe_name = validate_ref_name(name)
        for event in reversed(self._log.read_all()):
            if event.type != "memory.ref.updated":
                continue
            payload = event.payload
            if payload.get("name") != safe_name:
                continue
            return MemoryRef(
                name=safe_name,
                session_id=str(payload["session_id"]),
                target_seq=int(payload["target_seq"]),
                target_hash=str(payload["target_hash"]),
                ref_type=str(payload["ref_type"]),
                updated_at=event.timestamp,
            )
        return None

    def list_refs(self) -> list[MemoryRef]:
        """Return latest values for all refs."""
        latest: dict[str, MemoryRef] = {}
        for event in self._log.read_all():
            if event.type != "memory.ref.updated":
                continue
            payload = event.payload
            name = str(payload.get("name", ""))
            try:
                latest[name] = MemoryRef(
                    name=validate_ref_name(name),
                    session_id=validate_session_id(str(payload["session_id"])),
                    target_seq=int(payload["target_seq"]),
                    target_hash=str(payload["target_hash"]),
                    ref_type=str(payload["ref_type"]),
                    updated_at=event.timestamp,
                )
            except (KeyError, TypeError, ValueError):
                continue
        return sorted(latest.values(), key=lambda ref: ref.name)


def validate_ref_name(name: str) -> str:
    """Validate a Git-style memory ref name."""
    if not isinstance(name, str) or not REF_NAME_PATTERN.fullmatch(name):
        raise ValueError("Invalid memory ref: use HEAD or refs/<namespace>/<name>")
    if any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("Invalid memory ref: empty, '.', and '..' path parts are not allowed")
    return name


def _validate_ref_type(ref_type: str) -> str:
    if not isinstance(ref_type, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", ref_type):
        raise ValueError("ref_type must be 1-64 ASCII letters, digits, '.', '_', or '-'")
    return ref_type
