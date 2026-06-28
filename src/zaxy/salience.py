"""Non-authoritative salience ledger event contracts and replay scoring.

These helpers build Eventloom append specs for memory reinforcement
(checkout surfacing, confirmed feedback, coordination promotion, explicit
invalidation) and replay those events into per-memory salience state.
Salience is observable projection policy only: it never deletes events,
never changes what is citable, and is fully rebuildable by replaying the
unchanged immutable log.

Scoring model::

    salience = base * reinforcement_factor * recency_factor

- ``base`` is :data:`SALIENCE_BASE` (1.0) — the implicit score of a memory
  that has never been reinforced.
- ``reinforcement_factor`` starts at ``base`` and multiplies in one entry of
  :data:`SALIENCE_REINFORCEMENT_MULTIPLIERS` (or the event's explicit
  ``weight`` override) per reinforcement event targeting the memory.
- ``recency_factor`` is exponential decay anchored at the most recent
  reinforcement: ``0.5 ** (age_days / half_life_days)``. Ages are clamped
  to zero so clock skew (``now`` earlier than the last reinforcement) can
  never inflate a score.

Scores and reinforcement factors are clamped to
[:data:`SALIENCE_MIN`, :data:`SALIENCE_MAX`] = [0.01, 10.0]. The floor keeps
attenuated memories nonzero — attenuation stays reversible and explicitly
queryable — and the ceiling bounds multiplicative runaway from repeated
reinforcement.

Replay is a pure function of the log: the same events and the same ``now``
always produce the same state map. ``now`` is an explicit parameter; this
module never reads the wall clock.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, NamedTuple, TypeGuard

REINFORCEMENT_EVENT_TYPE = "memory.reinforcement"

#: The single tuning table for reinforcement strength, ordered weak-to-strong
#: signal (surfacing alone is weaker reinforcement than confirmed use;
#: invalidation attenuates). Tuned by the forgetting lane, not by assertion.
SALIENCE_REINFORCEMENT_MULTIPLIERS: Mapping[str, float] = MappingProxyType(
    {
        "surfaced": 1.05,
        "confirmed": 1.5,
        "promoted": 2.0,
        "invalidated": 0.2,
    }
)

REINFORCEMENT_KINDS: frozenset[str] = frozenset(SALIENCE_REINFORCEMENT_MULTIPLIERS)

#: Recency decay half-life. A module constant (overridable per
#: :class:`SalienceLedger`) rather than a ``Settings`` field in this
#: increment; promotion to configuration happens with the forgetting lane.
SALIENCE_HALF_LIFE_DAYS = 30.0

#: Implicit salience of a memory that has never been reinforced.
SALIENCE_BASE = 1.0

#: Clamp floor: attenuated memories never reach zero, so they stay rankable
#: by explicit query and every attenuation remains reversible by replay.
SALIENCE_MIN = 0.01

#: Clamp ceiling: bounds multiplicative runaway from repeated reinforcement.
SALIENCE_MAX = 10.0

#: Upper bound for an explicit per-event ``weight`` override.
MAX_REINFORCEMENT_WEIGHT = SALIENCE_MAX

#: Smallest representable per-event ``weight``. Surprise-scaled attenuation
#: (e.g. an ``invalidated`` reinforcement at maximal prediction error) clamps
#: UP to this floor rather than crossing zero, keeping every weight a valid,
#: replayable, reversible multiplier within ``(0.0, MAX_REINFORCEMENT_WEIGHT]``.
MIN_REINFORCEMENT_WEIGHT = 0.01

#: Cue record convention (encoding specificity): event payloads may carry a
#: ``cues`` mapping with these optional string fields. Cues are plain payload
#: data — no log schema change — and only affect ranking under the cognitive
#: retrieval profile.
CUE_FIELDS: tuple[str, ...] = ("mission", "workspace", "tool", "phase")

#: Bounded weight of the cue-overlap bonus blended into cognitive ranking:
#: ``bonus = CUE_MATCH_WEIGHT * jaccard(query_cues, stored_cues)`` so a
#: perfect cue match adds at most ``CUE_MATCH_WEIGHT`` to a relevance score
#: normalized around [0, 1].
CUE_MATCH_WEIGHT = 0.25

#: Write-time encoding gate classifications, ordered novel-to-redundant.
ENCODING_CLASSIFICATIONS: tuple[str, ...] = ("novel", "reinforcing", "redundant")

#: Content-overlap (token Jaccard vs. the closest existing verbatim chunk) at
#: or above which an append duplicates existing memory.
ENCODING_REDUNDANT_MIN_OVERLAP = 0.9

#: Content-overlap at or above which an append confirms existing memory.
ENCODING_REINFORCING_MIN_OVERLAP = 0.6

#: Lower content-overlap bound at which full entity-name overlap corroborates
#: a reinforcing classification.
ENCODING_ENTITY_CORROBORATION_MIN_OVERLAP = 0.4

_AUTHORITY_STATUS = "non_authoritative"
_EVENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SECONDS_PER_DAY = 86_400.0
_KIND_ORDER: tuple[str, ...] = ("surfaced", "confirmed", "promoted", "invalidated")
_EVENTLOOM_CITATION_RE = re.compile(
    r"^eventloom://[^/\s]+/events/(?P<seq>[1-9][0-9]*)#(?P<fragment>[0-9a-f]{12}|[0-9a-f]{64})$"
)


class EventRef(NamedTuple):
    """A stable citation to one sealed Eventloom event (seq + content hash)."""

    seq: int
    hash: str


@dataclass(frozen=True, slots=True)
class SalienceState:
    """Replayed salience for one target memory, with its composition.

    The breakdown fields satisfy
    ``score == clamp(base * reinforcement_factor * recency_factor)`` so
    checkout diagnostics can show *why* a memory carries its score.
    """

    score: float
    last_reinforced_at: datetime
    reinforcement_counts: Mapping[str, int]
    base: float
    reinforcement_factor: float
    recency_factor: float

    def composition(self) -> dict[str, Any]:
        """Return the score breakdown as plain diagnostics-ready data."""
        return {
            "score": self.score,
            "base": self.base,
            "reinforcement_factor": self.reinforcement_factor,
            "recency_factor": self.recency_factor,
            "last_reinforced_at": self.last_reinforced_at.isoformat().replace("+00:00", "Z"),
            "reinforcement_counts": dict(self.reinforcement_counts),
        }


@dataclass(frozen=True, slots=True)
class _Reinforcement:
    """A validated, replay-ready view of one reinforcement event."""

    kind: str
    targets: tuple[EventRef, ...]
    multiplier: float
    at: datetime


def build_reinforcement_event(
    *,
    actor: str,
    session_id: str,
    kind: str,
    targets: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    weight: float | None = None,
) -> dict[str, Any]:
    """Build an append-ready memory reinforcement event spec.

    One event may reinforce many targets: checkout surfacing appends one
    batched event per checkout listing every surfaced ref, keeping log
    volume O(checkouts) rather than O(surfaced memories).

    Args:
        actor: Actor emitting the reinforcement.
        session_id: Session thread the event belongs to.
        kind: One of ``surfaced``, ``confirmed``, ``promoted``,
            ``invalidated``.
        targets: Event refs (``seq`` + 64-hex ``hash``) of the reinforced
            memories; non-empty and duplicate-free.
        source: Small mapping identifying the origin, e.g.
            ``{"checkout_id": ...}`` or ``{"feedback_id": ...}``.
        weight: Optional multiplier override replacing the
            :data:`SALIENCE_REINFORCEMENT_MULTIPLIERS` entry for this event;
            must be finite and in ``(0.0, MAX_REINFORCEMENT_WEIGHT]``.

    Returns:
        An append spec (``event_type`` / ``actor`` / ``thread`` /
        ``payload``) ready for the fabric's event-spec append path.
    """
    actor = _validate_text(actor, field_name="actor")
    session_id = _validate_text(session_id, field_name="session_id")
    kind = _validate_kind(kind)
    snapped_targets = _snapshot_targets(targets)
    snapped_source = _snapshot_source(source)
    validated_weight = _validate_weight(weight)

    payload: dict[str, Any] = {
        "kind": kind,
        "targets": snapped_targets,
        "source": snapped_source,
        "authority_status": _AUTHORITY_STATUS,
    }
    if validated_weight is not None:
        payload["weight"] = validated_weight

    return {
        "event_type": REINFORCEMENT_EVENT_TYPE,
        "actor": actor,
        "thread": session_id,
        "payload": payload,
    }


def build_surfaced_reinforcement_event(
    *,
    actor: str,
    session_id: str,
    checkout_id: str,
    targets: Sequence[Mapping[str, Any]],
    weight: float | None = None,
) -> dict[str, Any]:
    """Build the one batched weak-reinforcement event a checkout appends."""
    return build_reinforcement_event(
        actor=actor,
        session_id=session_id,
        kind="surfaced",
        targets=targets,
        source={"checkout_id": checkout_id},
        weight=weight,
    )


def build_confirmed_reinforcement_event(
    *,
    actor: str,
    session_id: str,
    feedback_id: str,
    targets: Sequence[Mapping[str, Any]],
    weight: float | None = None,
) -> dict[str, Any]:
    """Build a strong-reinforcement event for positive memory feedback."""
    return build_reinforcement_event(
        actor=actor,
        session_id=session_id,
        kind="confirmed",
        targets=targets,
        source={"feedback_id": feedback_id},
        weight=weight,
    )


def build_promoted_reinforcement_event(
    *,
    actor: str,
    session_id: str,
    promotion_id: str,
    targets: Sequence[Mapping[str, Any]],
    weight: float | None = None,
) -> dict[str, Any]:
    """Build a strong-reinforcement event for a coordination promotion."""
    return build_reinforcement_event(
        actor=actor,
        session_id=session_id,
        kind="promoted",
        targets=targets,
        source={"promotion_id": promotion_id},
        weight=weight,
    )


def build_invalidated_reinforcement_event(
    *,
    actor: str,
    session_id: str,
    invalidation_id: str,
    targets: Sequence[Mapping[str, Any]],
    weight: float | None = None,
) -> dict[str, Any]:
    """Build a negative-reinforcement event for an explicit invalidation."""
    return build_reinforcement_event(
        actor=actor,
        session_id=session_id,
        kind="invalidated",
        targets=targets,
        source={"invalidation_id": invalidation_id},
        weight=weight,
    )


def prediction_error_weight(kind: str, prediction_error: float) -> float:
    """Scale a reinforcement multiplier by prediction error (surprise).

    Surprise-weighted reinforcement (a Rescorla-Wagner intuition): an outcome
    that merely confirms what the agent already expected (low prediction
    error) barely moves salience, while a surprising outcome (high error)
    drives the multiplier to — and past — the fixed-table strength for
    ``kind``. The mapping is continuous with
    :data:`SALIENCE_REINFORCEMENT_MULTIPLIERS`:

    - ``pe == 0`` -> ``1.0`` (no surprise, no net salience change);
    - ``pe == 0.5`` -> ``SALIENCE_REINFORCEMENT_MULTIPLIERS[kind]`` exactly
      (continuity with the fixed table the forgetting lane tunes);
    - ``pe == 1`` -> maximal surprise, extrapolated linearly from those two.

    The result is clamped to ``[MIN_REINFORCEMENT_WEIGHT,
    MAX_REINFORCEMENT_WEIGHT]`` so an ``invalidated`` kind at high surprise
    floors (rather than crossing zero) and a positive kind never crosses the
    ceiling — every return value is a valid per-event ``weight`` override.

    Args:
        kind: A reinforcement kind in
            :data:`SALIENCE_REINFORCEMENT_MULTIPLIERS`.
        prediction_error: Surprise in ``[0.0, 1.0]`` (finite, non-bool).

    Returns:
        The surprise-scaled multiplier, ready to pass as ``weight``.
    """
    kind = _validate_kind(kind)
    pe = _validate_prediction_error(prediction_error)
    base = SALIENCE_REINFORCEMENT_MULTIPLIERS[kind]
    weight = base + (base - 1.0) * (2.0 * pe - 1.0)
    return min(max(weight, MIN_REINFORCEMENT_WEIGHT), MAX_REINFORCEMENT_WEIGHT)


def event_ref_index(events: Iterable[object]) -> dict[int, tuple[str, str]]:
    """Map sealed event seq -> (full content hash, event type) for one log slice.

    Accepts replayed :class:`zaxy.event.Event` objects or mappings; entries
    without a positive integer ``seq``, a 64-hex ``hash``, and a non-empty
    string type are skipped. The index lets emitters canonicalize checkout
    citations (whose hash fragments may be 12-char prefixes) into full-hash
    reinforcement target refs without re-reading the log.
    """
    index: dict[int, tuple[str, str]] = {}
    for event in events:
        if isinstance(event, Mapping):
            seq = event.get("seq")
            event_hash = event.get("hash")
            event_type = event.get("type") or event.get("event_type")
        else:
            seq = getattr(event, "seq", None)
            event_hash = getattr(event, "hash", None)
            event_type = getattr(event, "type", None)
        if (
            _is_valid_seq(seq)
            and _is_valid_hash(event_hash)
            and isinstance(event_type, str)
            and event_type
        ):
            index[seq] = (event_hash, event_type)
    return index


def resolve_citation_target(
    citation: object,
    *,
    event_index: Mapping[int, tuple[str, str]],
) -> EventRef | None:
    """Resolve one eventloom citation into a reinforcement-target event ref.

    Returns ``None`` for anything that may not be reinforced: non-eventloom
    citations, refs absent from ``event_index``, hash fragments that do not
    match the sealed hash (e.g. cross-session citations sharing a seq), and
    refs to ``memory.reinforcement`` events themselves — a reinforcement
    event is never a reinforcement target, so emitters cannot recurse.
    """
    if not isinstance(citation, str):
        return None
    match = _EVENTLOOM_CITATION_RE.fullmatch(citation.strip())
    if match is None:
        return None
    seq = int(match.group("seq"))
    entry = event_index.get(seq)
    if entry is None:
        return None
    event_hash, event_type = entry
    if event_type == REINFORCEMENT_EVENT_TYPE:
        return None
    if not event_hash.startswith(match.group("fragment")):
        return None
    return EventRef(seq=seq, hash=event_hash)


def reinforcement_targets_from_citations(
    citations: Iterable[object],
    *,
    event_index: Mapping[int, tuple[str, str]],
) -> list[dict[str, Any]]:
    """Canonicalize citations into a duplicate-free, builder-ready target list.

    Order-preserving over the first occurrence of each resolved ref; anything
    :func:`resolve_citation_target` rejects is dropped.
    """
    targets: list[dict[str, Any]] = []
    seen: set[EventRef] = set()
    for citation in citations:
        ref = resolve_citation_target(citation, event_index=event_index)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        targets.append({"seq": ref.seq, "hash": ref.hash})
    return targets


def target_ref(seq: object, event_hash: object) -> dict[str, Any] | None:
    """Return a builder-ready target for a (seq, full hash) pair, or None.

    Used by emitters whose source data already carries full event hashes
    (e.g. coordination finding provenance, projected entity provenance).
    """
    if _is_valid_seq(seq) and _is_valid_hash(event_hash):
        return {"seq": seq, "hash": event_hash}
    return None


class SalienceLedger:
    """Pure replay of reinforcement events into per-memory salience.

    The ledger holds only scoring configuration; all state lives in the
    event log. ``replay`` is exactly a left fold of ``apply`` over the
    events, which is what makes later incremental projection wiring safe:
    replaying a full log and applying events one at a time as they land
    produce identical state maps.
    """

    def __init__(self, *, half_life_days: float = SALIENCE_HALF_LIFE_DAYS) -> None:
        """Create a ledger with the given recency-decay half-life in days."""
        if (
            isinstance(half_life_days, bool)
            or not isinstance(half_life_days, int | float)
            or not math.isfinite(half_life_days)
            or half_life_days <= 0.0
        ):
            raise ValueError("half_life_days must be a finite number greater than 0")
        self._half_life_seconds = float(half_life_days) * _SECONDS_PER_DAY

    def replay(
        self,
        events: Iterable[object],
        *,
        now: datetime,
    ) -> dict[EventRef, SalienceState]:
        """Replay events into a target-ref -> salience state map.

        Pure function of the inputs: the same event sequence and the same
        ``now`` always produce the same map. Events that are not well-formed
        ``memory.reinforcement`` events are ignored, so any log slice may be
        passed verbatim.

        Args:
            events: Replayed log events — :class:`zaxy.event.Event` objects
                or mappings carrying ``type``/``event_type``, ``payload``,
                and ``timestamp``.
            now: Timezone-aware instant the scores are computed at.

        Returns:
            Mapping from reinforced :class:`EventRef` to its
            :class:`SalienceState` as of ``now``.
        """
        _validate_now(now)
        state: dict[EventRef, SalienceState] = {}
        for event in events:
            self._apply_into(state, event, now=now)
        return state

    def apply(
        self,
        state: Mapping[EventRef, SalienceState],
        reinforcement_event: object,
        *,
        now: datetime,
    ) -> dict[EventRef, SalienceState]:
        """Return ``state`` advanced by one event, without mutating the input.

        Folding ``apply`` over an event sequence from an empty map is
        equivalent to :meth:`replay` over that sequence — including the
        defensive handling of events that are not well-formed reinforcement
        events, which leave the state unchanged.
        """
        _validate_now(now)
        next_state = dict(state)
        self._apply_into(next_state, reinforcement_event, now=now)
        return next_state

    def _apply_into(
        self,
        state: dict[EventRef, SalienceState],
        event: object,
        *,
        now: datetime,
    ) -> None:
        """Apply one event to a mutable state map shared by replay and apply."""
        reinforcement = _parse_reinforcement(event)
        if reinforcement is None:
            return
        for target in reinforcement.targets:
            state[target] = self._reinforced_state(
                state.get(target),
                reinforcement=reinforcement,
                now=now,
            )

    def _reinforced_state(
        self,
        previous: SalienceState | None,
        *,
        reinforcement: _Reinforcement,
        now: datetime,
    ) -> SalienceState:
        """Fold one reinforcement of one target into a fresh state value."""
        if previous is None:
            counts = dict.fromkeys(_KIND_ORDER, 0)
            factor = SALIENCE_BASE
        else:
            counts = dict(previous.reinforcement_counts)
            factor = previous.reinforcement_factor
        counts[reinforcement.kind] = counts.get(reinforcement.kind, 0) + 1
        factor = _clamp(factor * reinforcement.multiplier)
        recency = self._recency_factor(last_reinforced_at=reinforcement.at, now=now)
        return SalienceState(
            score=_clamp(SALIENCE_BASE * factor * recency),
            last_reinforced_at=reinforcement.at,
            reinforcement_counts=counts,
            base=SALIENCE_BASE,
            reinforcement_factor=factor,
            recency_factor=recency,
        )

    def _recency_factor(self, *, last_reinforced_at: datetime, now: datetime) -> float:
        """Exponential decay since the last reinforcement, never above 1.0."""
        age_seconds = max(0.0, (now - last_reinforced_at).total_seconds())
        return math.pow(0.5, age_seconds / self._half_life_seconds)


def _clamp(value: float) -> float:
    return min(max(value, SALIENCE_MIN), SALIENCE_MAX)


def _validate_now(now: datetime) -> None:
    if not isinstance(now, datetime) or now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("now must be a timezone-aware datetime")


def _parse_reinforcement(event: object) -> _Reinforcement | None:
    """Parse one replayed event, returning None for anything non-applicable.

    Replay and incremental apply must skip identically — a malformed
    hand-appended payload may never make rebuild-from-scratch diverge from
    the incrementally maintained projection — so skipping is all-or-nothing
    per event and lives in this single parser.
    """
    fields = _event_fields(event)
    if fields is None:
        return None
    event_type, payload, timestamp = fields
    if event_type != REINFORCEMENT_EVENT_TYPE:
        return None
    at = _parse_timestamp(timestamp)
    if at is None:
        return None
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in REINFORCEMENT_KINDS:
        return None
    targets = _parse_targets(payload.get("targets"))
    if targets is None:
        return None
    multiplier = _parse_multiplier(payload.get("weight"), kind=kind)
    if multiplier is None:
        return None
    return _Reinforcement(kind=kind, targets=targets, multiplier=multiplier, at=at)


def _event_fields(event: object) -> tuple[str, Mapping[str, Any], str] | None:
    if isinstance(event, Mapping):
        event_type = event.get("type") or event.get("event_type")
        payload = event.get("payload")
        timestamp = event.get("timestamp")
    else:
        event_type = getattr(event, "type", None)
        payload = getattr(event, "payload", None)
        timestamp = getattr(event, "timestamp", None)
    if (
        not isinstance(event_type, str)
        or not isinstance(payload, Mapping)
        or not isinstance(timestamp, str)
    ):
        return None
    return event_type, payload, timestamp


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_targets(value: object) -> tuple[EventRef, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        return None
    refs: list[EventRef] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        seq = item.get("seq")
        if not _is_valid_seq(seq):
            return None
        event_hash = item.get("hash")
        if not _is_valid_hash(event_hash):
            return None
        refs.append(EventRef(seq=seq, hash=event_hash))
    if len(set(refs)) != len(refs):
        return None
    return tuple(refs)


def _parse_multiplier(weight: object, *, kind: str) -> float | None:
    if weight is None:
        return SALIENCE_REINFORCEMENT_MULTIPLIERS[kind]
    if (
        isinstance(weight, bool)
        or not isinstance(weight, int | float)
        or not math.isfinite(weight)
        or not 0.0 < float(weight) <= MAX_REINFORCEMENT_WEIGHT
    ):
        return None
    return float(weight)


def _is_valid_seq(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_valid_hash(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and _EVENT_HASH_RE.fullmatch(value) is not None


def _validate_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_kind(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("kind must be one of: " + ", ".join(sorted(REINFORCEMENT_KINDS)))
    kind = value.strip().casefold()
    if kind not in REINFORCEMENT_KINDS:
        raise ValueError("kind must be one of: " + ", ".join(sorted(REINFORCEMENT_KINDS)))
    return kind


def _validate_weight(value: float | None) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 < float(value) <= MAX_REINFORCEMENT_WEIGHT
    ):
        raise ValueError(
            f"weight must be a finite number greater than 0.0 and at most {MAX_REINFORCEMENT_WEIGHT}"
        )
    return float(value)


def _validate_prediction_error(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError("prediction_error must be a finite number in [0.0, 1.0]")
    return float(value)


def _snapshot_targets(targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(targets, Sequence) or isinstance(targets, str | bytes):
        raise ValueError("targets must be a non-empty sequence of event refs")
    if not targets:
        raise ValueError("targets must be non-empty")

    seen: set[EventRef] = set()
    snapshot: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        ref = _snapshot_target(target, index=index)
        key = EventRef(seq=ref["seq"], hash=ref["hash"])
        if key in seen:
            raise ValueError(f"targets[{index}] duplicates an earlier target ref")
        seen.add(key)
        snapshot.append(ref)
    return snapshot


def _snapshot_target(target: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(target, Mapping):
        raise ValueError(f"targets[{index}] must be an event ref mapping")

    seq = target.get("seq")
    if not _is_valid_seq(seq):
        raise ValueError(f"targets[{index}].seq must be a positive integer")

    event_hash = target.get("hash")
    if not _is_valid_hash(event_hash):
        raise ValueError(f"targets[{index}].hash must be exactly 64 lowercase hex characters")

    return {"seq": seq, "hash": event_hash}


def build_cue_record(
    *,
    mission: str | None = None,
    workspace: str | None = None,
    tool: str | None = None,
    phase: str | None = None,
) -> dict[str, str]:
    """Build a payload-ready cue record from whatever context is in hand.

    Only non-empty string fields are recorded; an empty mapping means the
    caller had no cue context, and callers should omit the ``cues`` key
    entirely in that case.
    """
    record: dict[str, str] = {}
    for field_name, value in (
        ("mission", mission),
        ("workspace", workspace),
        ("tool", tool),
        ("phase", phase),
    ):
        if isinstance(value, str) and value.strip():
            record[field_name] = value.strip()
    return record


def cue_pairs(cues: object) -> frozenset[str]:
    """Normalize a cue record into comparable ``field=value`` pairs.

    Accepts the payload's ``cues`` value verbatim; non-mapping values,
    unknown fields, and non-string/empty values are ignored so malformed
    hand-appended payloads can never break retrieval.
    """
    if not isinstance(cues, Mapping):
        return frozenset()
    pairs: set[str] = set()
    for field_name in CUE_FIELDS:
        value = cues.get(field_name)
        if isinstance(value, str) and value.strip():
            pairs.add(f"{field_name}={value.strip()}")
    return frozenset(pairs)


def cue_overlap(query_cues: frozenset[str], stored_cues: frozenset[str]) -> float:
    """Return the Jaccard overlap between two normalized cue-pair sets.

    Empty sets contribute zero — no cues means no encoding-specificity
    signal, never a penalty.
    """
    if not query_cues or not stored_cues:
        return 0.0
    intersection = len(query_cues & stored_cues)
    if intersection == 0:
        return 0.0
    return intersection / len(query_cues | stored_cues)


@dataclass(frozen=True, slots=True)
class EncodingDecision:
    """One write-time encoding gate decision with its observable signals.

    ``duplicate_of`` carries the Eventloom citation of the closest existing
    verbatim chunk when the classification is ``redundant``, so the gate can
    emit a weak reinforcement toward the duplicated memory.
    """

    classification: str
    content_overlap: float
    entity_overlap: float
    duplicate_of: str | None = None

    def tag_payload(self) -> dict[str, Any]:
        """Return the payload-ready ``encoding`` tag for this decision."""
        tag: dict[str, Any] = {
            "classification": self.classification,
            "content_overlap": round(self.content_overlap, 4),
            "entity_overlap": round(self.entity_overlap, 4),
        }
        if self.duplicate_of is not None:
            tag["duplicate_of"] = self.duplicate_of
        return tag


def classify_append(*, content_overlap: float, entity_overlap: float = 0.0) -> str:
    """Classify one append against existing memory from write-time signals.

    Pure threshold policy over two cheap signals computed at append time
    (no embedding calls):

    - ``content_overlap``: token Jaccard between the new event's content and
      the closest existing verbatim-index chunk (0 when nothing matches).
    - ``entity_overlap``: fraction of the event's entity names already
      projected for the session (0 when the event names no known entities).

    Classification:

    - ``redundant`` — near-verbatim duplicate
      (``content_overlap >= ENCODING_REDUNDANT_MIN_OVERLAP``);
    - ``reinforcing`` — substantially overlapping content
      (``>= ENCODING_REINFORCING_MIN_OVERLAP``), or moderate overlap
      (``>= ENCODING_ENTITY_CORROBORATION_MIN_OVERLAP``) corroborated by
      every named entity already being known;
    - ``novel`` — everything else, which deliberately includes appends that
      mention known entities with conflicting content (contradiction is a
      form of novelty, and feeds interference detection).
    """
    for name, value in (("content_overlap", content_overlap), ("entity_overlap", entity_overlap)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be a finite number between 0.0 and 1.0")
    if content_overlap >= ENCODING_REDUNDANT_MIN_OVERLAP:
        return "redundant"
    if content_overlap >= ENCODING_REINFORCING_MIN_OVERLAP:
        return "reinforcing"
    if content_overlap >= ENCODING_ENTITY_CORROBORATION_MIN_OVERLAP and entity_overlap >= 1.0:
        return "reinforcing"
    return "novel"


def _snapshot_source(source: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping of origin identifiers")
    if not source:
        raise ValueError("source must be non-empty")

    snapshot: dict[str, str] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("source keys must be non-empty strings")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source[{key.strip()!r}] must be a non-empty string identifier")
        snapshot[key.strip()] = value.strip()
    return snapshot
