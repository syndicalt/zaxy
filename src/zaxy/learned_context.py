"""Amortized learned-context artifacts (Zaxy 3 / I2).

The crystallization pass already builds a real compaction projection
(:func:`zaxy.compaction.build_compaction_projection`) and, before this module,
dropped it on the floor. This module is the persistence and staleness discipline
that turns that throwaway into an *amortized* precompute the read path can use:

- **The log stays truth.** A build appends a non-authoritative
  ``crystallization.projection.built`` event carrying the projection identity,
  the audit numbers, and the ``{seq, hash}`` head the projection covers. That
  event is the replayable record.
- **The artifact is a cache.** The JSON under
  ``<eventloom>/projections/learned-context/<session_id>.json`` is convenience.
  Deleting it loses nothing: the next pass rebuilds it from the log.
- **An artifact with no event is UNTRUSTED.** The asymmetry is deliberate — the
  event is evidence, the file is not. A file whose ``projection_id`` no event
  vouches for is ignored outright.
- **Staleness fails closed.** The covered head is re-verified against the log on
  every load. Any mismatch, missing event, or unreadable file yields *no*
  projection and a visible ``stale`` diagnostic, never a partially trusted one.
  A stale projection surfacing old context as current is the failure this whole
  module exists to prevent, so there is no degraded middle path.

Consumption lives in :func:`zaxy.long_horizon.build_long_horizon_plan`, which
treats a loaded projection as a *second* source for the consolidated remote tier.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zaxy.compaction import (
    CompactionProjection,
    CompactionProjectionRecord,
    projection_from_payload,
)

#: Event type recording that a learned-context projection was built.
LEARNED_CONTEXT_EVENT_TYPE = "crystallization.projection.built"

#: Directory (under the eventloom projections root) holding the artifacts.
LEARNED_CONTEXT_DIRNAME = "learned-context"

#: Artifact schema version, so a future envelope change can be detected on load.
LEARNED_CONTEXT_SCHEMA = 1

# ---------------------------------------------------------------------------
# Size bounds. This repo has been bitten by unbounded projection growth (the
# 397MB incident), so the artifact is bounded by construction rather than by
# hoping the inputs stay small. With these caps one artifact holds at most
# 64 records x (4 KiB text + 2 x 16 x 256 B identity/citation strings) plus a
# capped audit block, i.e. **under ~1 MiB per session**. There is exactly one
# artifact per session and it is rewritten in place, never accumulated, so total
# on-disk cost is O(sessions) x 1 MiB rather than O(passes).
# ---------------------------------------------------------------------------

#: Maximum projection records persisted in one artifact.
MAX_ARTIFACT_RECORDS = 64

#: Maximum characters of record text persisted per record.
MAX_RECORD_TEXT_CHARS = 4096

#: Maximum identity/citation strings persisted per record.
MAX_RECORD_STRINGS = 16

#: Maximum characters per persisted identity/citation string.
MAX_STRING_CHARS = 256

#: Maximum identity strings persisted in the artifact's audit block.
MAX_AUDIT_IDENTITIES = 256


@dataclass(frozen=True)
class LearnedContextLoad:
    """Outcome of attempting to load a session's learned-context artifact.

    ``projection`` is non-``None`` only when the artifact was readable, vouched
    for by a build event in the log, and its covered head still verifies.
    """

    projection: CompactionProjection | None = None
    stale: bool = False
    reason: str | None = None
    covered_seq: int | None = None
    projection_id: str | None = None

    def to_diagnostics(self) -> dict[str, Any]:
        """Return the ``learned_context`` stanza threaded into checkout diagnostics."""
        return {
            "available": self.projection is not None,
            "stale": self.stale,
            "reason": self.reason,
            "covered_seq": self.covered_seq,
            "projection_id": self.projection_id,
            "record_count": 0 if self.projection is None else len(self.projection.records),
        }


def learned_context_path(eventloom_dir: str | Path, session_id: str) -> Path:
    """Return the artifact path for one session's learned-context projection."""
    return Path(eventloom_dir) / "projections" / LEARNED_CONTEXT_DIRNAME / f"{session_id}.json"


def covered_head(events: list[Any]) -> tuple[int, str] | None:
    """Return the ``(seq, hash)`` of the newest event a projection would cover."""
    for event in reversed(events):
        seq = getattr(event, "seq", None)
        event_hash = getattr(event, "hash", None)
        if (
            isinstance(seq, int)
            and not isinstance(seq, bool)
            and seq > 0
            and isinstance(event_hash, str)
            and event_hash
        ):
            return seq, event_hash
    return None


def bound_projection(projection: CompactionProjection) -> CompactionProjection:
    """Return ``projection`` truncated to the persisted size bounds.

    Truncation is lossy on purpose: the artifact is a cache, and the log still
    holds every source event. Records are kept in build order, so the medoid /
    highest-priority exemplars (emitted first) survive the cut.
    """
    records = tuple(
        CompactionProjectionRecord(
            kind=record.kind,
            event_seq=record.event_seq,
            event_ref=record.event_ref,
            text=record.text[:MAX_RECORD_TEXT_CHARS],
            identities=_bound_strings(record.identities),
            citations=_bound_strings(record.citations),
            authority_scope=record.authority_scope,
            purpose_reasons=_bound_strings(record.purpose_reasons),
        )
        for record in projection.records[:MAX_ARTIFACT_RECORDS]
    )
    audit = projection.audit
    bounded_audit = type(audit)(
        **{
            **asdict(audit),
            "identities": _bound_strings(audit.identities, limit=MAX_AUDIT_IDENTITIES),
            "identity_hits": _bound_strings(audit.identity_hits, limit=MAX_AUDIT_IDENTITIES),
            "missing_identities": _bound_strings(
                audit.missing_identities, limit=MAX_AUDIT_IDENTITIES
            ),
        }
    )
    return CompactionProjection(
        projection_id=projection.projection_id,
        strategy=projection.strategy,
        source_event_count=projection.source_event_count,
        source_identities=_bound_strings(
            projection.source_identities, limit=MAX_AUDIT_IDENTITIES
        ),
        records=records,
        audit=bounded_audit,
        purpose=dict(projection.purpose),
        consolidation_policy=dict(projection.consolidation_policy),
    )


def write_learned_context(
    projection: CompactionProjection,
    path: str | Path,
    *,
    session_id: str,
    covered_seq: int,
    covered_hash: str,
) -> Path:
    """Atomically write a bounded learned-context artifact.

    The crystallization pass runs under cron in a *different* process from the
    MCP server that reads the artifact, so the write goes to a temp file in the
    same directory and is then ``os.replace``d — a reader either sees the whole
    previous artifact or the whole new one, never a torn file.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "schema": LEARNED_CONTEXT_SCHEMA,
        "session_id": session_id,
        "covered_seq": covered_seq,
        "covered_hash": covered_hash,
        "projection": asdict(bound_projection(projection)),
    }
    temp = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, output)
    return output


def build_projection_built_payload(
    projection: CompactionProjection,
    *,
    session_id: str,
    covered_seq: int,
    covered_hash: str,
    artifact_path: str,
) -> dict[str, Any]:
    """Return the non-authoritative payload for a ``crystallization.projection.built`` event."""
    bounded = bound_projection(projection)
    return {
        "authority_status": "non_authoritative",
        "non_authoritative": True,
        "session_id": session_id,
        "projection_id": bounded.projection_id,
        "strategy": bounded.strategy,
        "source_event_count": bounded.source_event_count,
        "record_count": len(bounded.records),
        "identity_recall": bounded.audit.identity_recall,
        "citation_coverage": bounded.audit.citation_coverage,
        "covered_head": {"seq": covered_seq, "hash": covered_hash},
        "artifact_path": artifact_path,
    }


def load_learned_context(path: str | Path, session_events: list[Any]) -> LearnedContextLoad:
    """Load a session's learned-context artifact, failing closed on any doubt.

    Returns a load whose ``projection`` is ``None`` unless every check passes:
    the artifact parses, a ``crystallization.projection.built`` event in the log
    vouches for its ``projection_id``, and the event at the artifact's covered
    ``seq`` still carries the covered ``hash``.
    """
    artifact = Path(path)
    if not artifact.exists():
        # Never built (or deliberately swept aside). Absence is not staleness:
        # nothing is claiming to be current, so there is nothing to distrust.
        return LearnedContextLoad(reason="missing")
    try:
        envelope = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return LearnedContextLoad(stale=True, reason="unreadable")
    if not isinstance(envelope, dict) or envelope.get("schema") != LEARNED_CONTEXT_SCHEMA:
        return LearnedContextLoad(stale=True, reason="unreadable")

    covered_seq = envelope.get("covered_seq")
    covered_hash_value = envelope.get("covered_hash")
    if (
        not isinstance(covered_seq, int)
        or isinstance(covered_seq, bool)
        or covered_seq <= 0
        or not isinstance(covered_hash_value, str)
        or not covered_hash_value
    ):
        return LearnedContextLoad(stale=True, reason="unreadable")

    try:
        projection = projection_from_payload(envelope["projection"])
    except (KeyError, TypeError, ValueError):
        return LearnedContextLoad(stale=True, reason="unreadable", covered_seq=covered_seq)

    if not _has_build_event(session_events, projection.projection_id):
        # The file claims a projection the log never recorded building. The event
        # is the evidence; an unvouched file is not trusted at all.
        return LearnedContextLoad(
            stale=True,
            reason="untrusted_no_build_event",
            covered_seq=covered_seq,
            projection_id=projection.projection_id,
        )

    if not _covered_head_verifies(session_events, covered_seq, covered_hash_value):
        return LearnedContextLoad(
            stale=True,
            reason="covered_head_mismatch",
            covered_seq=covered_seq,
            projection_id=projection.projection_id,
        )

    return LearnedContextLoad(
        projection=projection,
        stale=False,
        reason=None,
        covered_seq=covered_seq,
        projection_id=projection.projection_id,
    )


def _has_build_event(session_events: list[Any], projection_id: str) -> bool:
    for event in session_events:
        if getattr(event, "type", None) != LEARNED_CONTEXT_EVENT_TYPE:
            continue
        payload = getattr(event, "payload", None) or {}
        if payload.get("projection_id") == projection_id:
            return True
    return False


def _covered_head_verifies(session_events: list[Any], seq: int, event_hash: str) -> bool:
    for event in session_events:
        if getattr(event, "seq", None) != seq:
            continue
        return bool(getattr(event, "hash", None) == event_hash)
    return False


def _bound_strings(values: tuple[str, ...], *, limit: int = MAX_RECORD_STRINGS) -> tuple[str, ...]:
    return tuple(value[:MAX_STRING_CHARS] for value in values[:limit])
