"""Transparency & controlled editability contracts (Zaxy 3 / I5a).

Two append-only, gated, **non-authoritative** evolution events let a human (or
agent) correct or reverse memory without ever mutating the sealed log:

* ``memory.corrected`` — the re-ingest of a human edit. A memory is exported to a
  human-readable, editable block (:func:`render_editable`); the edited block is
  parsed back (:func:`parse_editable`) and re-ingested as a new event that
  *cites* the original ({seq, hash}), carries the corrected content + reason, and
  retains the original. Retrieval surfaces the correction; the original is never
  deleted.
* ``memory.rolled_back`` — the explicit reversal of a prior evolution (e.g. a
  consolidation acceptance, a generated preventive rule). It cites the evolution
  event being reversed and, on replay/projection, undoes its effect (the
  consolidation candidate returns to its pre-acceptance status). Nothing is
  destroyed; the reversal is itself a cited, replayable event.

Both are additive and reversible: they are NEW events sealed into the hash chain,
so :meth:`EventLog.verify` stays green. The effect is purely a replay/projection
function, mirroring :mod:`zaxy.fleet`'s reversible ``fleet.promotion.rolled_back``
and :mod:`zaxy.outcome_learning`'s cited builder style. Both route through the I4
``update`` evolution gate so the decision is auditable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

_AUTHORITY_STATUS = "non_authoritative"

#: Event type for the re-ingest of a human edit (a cited correction).
MEMORY_CORRECTED_EVENT_TYPE = "memory.corrected"
#: Event type for the explicit reversal of a prior evolution.
MEMORY_ROLLBACK_EVENT_TYPE = "memory.rolled_back"

#: The evolution events a rollback may reverse.
#:
#: Reversal is only as real as the replay that honours it, and today that
#: differs per type:
#:
#: - ``consolidation.candidate.reviewed`` -- fully reversed; ``_reverts_descriptor``
#:   restores the candidate's prior review status.
#: - ``memory.rule.generated`` / ``memory.rule.proposed`` -- reversed at checkout
#:   assembly: a rolled-back rule is excluded from every retrieval lane, the same
#:   exclusion applied to a gate-withheld rule.
#: - ``evolution.gate.evaluated``, ``fleet.promotion.reviewed``,
#:   ``MEMORY_CORRECTED_EVENT_TYPE`` -- **audit markers only**. A rollback is
#:   accepted and recorded, but nothing downstream reverses it yet. Verified for
#:   corrections: after rolling one back, the corrected content still reaches the
#:   prompt. Kept accepted rather than rejected so existing callers and the fleet
#:   path keep working; making these effective is open work, and until then the
#:   success response must not be read as "the change is undone".
ROLLBACKABLE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "consolidation.candidate.reviewed",
        "memory.rule.generated",
        "memory.rule.proposed",
        "evolution.gate.evaluated",
        "fleet.promotion.reviewed",
        MEMORY_CORRECTED_EVENT_TYPE,
    }
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

#: Editable-block format (front-matter + body). Versioned so a consumer can pin.
EDITABLE_FORMAT_VERSION = "v1"
_OPEN_DELIM = f"--- zaxy:editable {EDITABLE_FORMAT_VERSION} ---"
_CLOSE_DELIM = "---"
_EDITABLE_HEADER_KEYS: tuple[str, ...] = ("seq", "hash", "session_id", "entity_name")


def build_memory_correction_event(
    *,
    actor: str,
    session_id: str,
    target: Mapping[str, Any],
    new_content: str,
    reason: str,
    original_content: str | None = None,
    entity_name: str | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative ``memory.corrected`` event spec (re-ingest of an edit).

    Cites the corrected memory via ``target`` ({seq, 64-hex hash}); carries the
    edited content + the reason and a deterministic ``correction_id``. The
    original memory event is never touched; this is a new, additive event.
    """
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    _validate_non_empty_string(new_content, field_name="new_content")
    _validate_non_empty_string(reason, field_name="reason")
    snapshot = _snapshot_event_ref(target)
    payload: dict[str, Any] = {
        "correction_id": _correction_id(snapshot, new_content, reason),
        "target": snapshot,
        "content": new_content,
        "reason": reason,
        "authority_status": _AUTHORITY_STATUS,
    }
    if original_content is not None:
        _validate_non_empty_string(original_content, field_name="original_content")
        payload["original_content"] = original_content
    if entity_name is not None:
        _validate_non_empty_string(entity_name, field_name="entity_name")
        payload["entity_name"] = entity_name
    return {
        "event_type": MEMORY_CORRECTED_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }


def build_memory_rollback_event(
    *,
    actor: str,
    session_id: str,
    target: Mapping[str, Any],
    reason: str,
    reverts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative ``memory.rolled_back`` event spec (reverse an evolution).

    Cites the evolution event being reversed via ``target`` ({seq, 64-hex hash}),
    the reason, a deterministic ``rollback_id``, and an optional ``reverts``
    descriptor ({event_type, candidate_id?, to_status?}) that lets replay/projection
    restore the pre-evolution state. The cited event is never mutated.
    """
    _validate_non_empty_string(actor, field_name="actor")
    _validate_non_empty_string(session_id, field_name="session_id")
    _validate_non_empty_string(reason, field_name="reason")
    snapshot = _snapshot_event_ref(target)
    payload: dict[str, Any] = {
        "rollback_id": _rollback_id(snapshot, reason),
        "target": snapshot,
        "reason": reason,
        "authority_status": _AUTHORITY_STATUS,
    }
    if reverts is not None:
        payload["reverts"] = _snapshot_reverts(reverts)
    return {
        "event_type": MEMORY_ROLLBACK_EVENT_TYPE,
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }


def render_editable(memory: Mapping[str, Any]) -> str:
    """Render a memory into a human-readable, editable block (the export half of the round-trip).

    ``memory`` is a mapping carrying the target identity (``seq``, ``hash``) and
    a content field (``content``, falling back to ``summary`` then ``text``),
    plus optional ``session_id`` / ``entity_name``. The returned block is a small
    front-matter header followed by the editable content body; feed the edited
    block back through :func:`parse_editable`.
    """
    if not isinstance(memory, Mapping):
        raise ValueError("memory must be a mapping with seq, hash, and content")
    seq = memory.get("seq")
    event_hash = memory.get("hash")
    _validate_seq(seq)
    _validate_hash(event_hash)
    content = memory.get("content")
    if content is None:
        content = memory.get("summary")
    if content is None:
        content = memory.get("text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("memory must carry a non-empty content/summary/text field")

    header_lines = [_OPEN_DELIM, f"seq: {seq}", f"hash: {event_hash}"]
    session_id = memory.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        header_lines.append(f"session_id: {session_id.strip()}")
    entity_name = memory.get("entity_name")
    if isinstance(entity_name, str) and entity_name.strip():
        header_lines.append(f"entity_name: {entity_name.strip()}")
    header_lines.append(_CLOSE_DELIM)
    return "\n".join(header_lines) + "\n\n" + content.strip() + "\n"


def parse_editable(text: str) -> dict[str, Any]:
    """Parse an edited editable block back into correction fields (the re-ingest half).

    Validates the format and returns ``{"target": {"seq", "hash"}, "content": ...}``
    plus any ``session_id`` / ``entity_name`` carried in the header. The result
    feeds :func:`build_memory_correction_event` (``content`` is the new content).
    Raises ``ValueError`` on a malformed block, a missing/invalid target, or an
    empty body.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("editable text must be a non-empty string")
    lines = text.splitlines()

    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or lines[index].strip() != _OPEN_DELIM:
        raise ValueError(f"editable text must begin with {_OPEN_DELIM!r}")
    index += 1

    header: dict[str, str] = {}
    closed = False
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.strip() == _CLOSE_DELIM:
            closed = True
            break
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"editable header line must be 'key: value': {line!r}")
        key, _, value = line.partition(":")
        key = key.strip()
        if key not in _EDITABLE_HEADER_KEYS:
            raise ValueError(f"unknown editable header key: {key!r}")
        header[key] = value.strip()
    if not closed:
        raise ValueError("editable text missing closing '---' header delimiter")

    body = "\n".join(lines[index:]).strip()
    if not body:
        raise ValueError("editable content body must not be empty")

    seq = _parse_seq(header.get("seq"))
    event_hash = header.get("hash")
    _validate_hash(event_hash)
    result: dict[str, Any] = {
        "target": {"seq": seq, "hash": event_hash},
        "content": body,
    }
    if header.get("session_id"):
        result["session_id"] = header["session_id"]
    if header.get("entity_name"):
        result["entity_name"] = header["entity_name"]
    return result


def _correction_id(target: Mapping[str, Any], content: str, reason: str) -> str:
    identity = {"target": dict(target), "content": content, "reason": reason}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"correction:{digest}"


def _rollback_id(target: Mapping[str, Any], reason: str) -> str:
    identity = {"target": dict(target), "reason": reason}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"rollback:{digest}"


def _snapshot_event_ref(ref: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(ref, Mapping):
        raise ValueError("event ref must be a mapping with seq and hash")
    seq = ref.get("seq")
    event_hash = ref.get("hash")
    _validate_seq(seq)
    _validate_hash(event_hash)
    return {"seq": seq, "hash": event_hash}


def _snapshot_reverts(reverts: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(reverts, Mapping):
        raise ValueError("reverts must be a mapping with an event_type")
    event_type = reverts.get("event_type")
    _validate_non_empty_string(event_type, field_name="reverts.event_type")
    snapshot: dict[str, Any] = {"event_type": event_type}
    candidate_id = reverts.get("candidate_id")
    if candidate_id is not None:
        _validate_non_empty_string(candidate_id, field_name="reverts.candidate_id")
        snapshot["candidate_id"] = candidate_id
    to_status = reverts.get("to_status")
    if to_status is not None:
        _validate_non_empty_string(to_status, field_name="reverts.to_status")
        snapshot["to_status"] = to_status
    return snapshot


def _validate_seq(seq: object) -> None:
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError("event ref seq must be a positive integer")


def _parse_seq(raw: object) -> int:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("editable header 'seq' is required")
    try:
        seq = int(raw.strip())
    except ValueError as exc:
        raise ValueError("editable header 'seq' must be an integer") from exc
    _validate_seq(seq)
    return seq


def _validate_hash(event_hash: object) -> None:
    if not isinstance(event_hash, str) or not _HASH_RE.fullmatch(event_hash):
        raise ValueError("event ref hash must be a 64-character hex digest")


def _validate_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
