"""Product-agnostic memory export contract (Phase 1: the pure projection).

This module turns a session's memory into a canonical, cited list of *entries*
that drop straight into :func:`zaxy.portable.build_export` (the signed-bundle wire
format). It is the missing piece between "Zaxy's internal memory" and "a bundle
any product can consume": a stable entry schema, a selection contract, and a
projector that derives entries from the verified event log.

Design constraints (Phase 1):

- **Product-agnostic.** No consumer-specific knowledge lives here. A consumer
  pulls by passing a :class:`ExportSelector`; Zaxy never learns who is asking.
- **Pure + backend-free.** Reads only through a :class:`SessionRetrievalCache`
  (verified replay + verbatim index). No graph/projection backend, no network,
  no signing — those are later phases.
- **Two grains, selectable.** ``event`` grain projects raw sealed events;
  ``semantic`` grain projects the deterministic :func:`zaxy.extract.extract`
  output (entities + edges) that also feeds the graph. Both carry the same sealed
  Eventloom citation, so every entry is provenance-bearing.
- **Canonical / byte-stable.** Re-projecting the same log with the same selector
  yields identical entries, so Merkle roots and signatures are reproducible.

Out of scope here (later phases): the ``memory_export`` MCP tool and CLI surface,
signing/bundle assembly, incremental delivery transport, and outbound sinks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from zaxy.extract import extract

if TYPE_CHECKING:
    from zaxy.event import Event
    from zaxy.retrieval_cache import SessionRetrievalCache

#: Versioned so consumers can pin the entry contract. Bump on any
#: backward-incompatible change to an entry's shape or semantics.
EXPORT_ENTRY_SCHEMA_VERSION = "zaxy.export.v1"

Grain = Literal["event", "semantic"]
_ALL_GRAINS: frozenset[str] = frozenset({"event", "semantic"})

# eventloom://<thread>/events/<seq>#<hash>
_CITATION_SEQ_RE = re.compile(r"^eventloom://[^/\s]+/events/([1-9][0-9]*)#")


def _citation(event: Event) -> str:
    """Sealed Eventloom provenance ref for an event (full-hash form)."""
    return f"eventloom://{event.thread}/events/{event.seq}#{event.hash}"


def _seq_from_citation(citation: str) -> int | None:
    match = _CITATION_SEQ_RE.match(citation)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class ExportSelector:
    """A product-agnostic description of *what* to export from a session.

    Every axis is optional; the default selects both grains of the whole log.
    Filters that gate the source event (``kinds``, the seq/time bounds,
    ``exclude_sensitivities``, ``query``, ``limit``) apply to both grains, so a
    semantic entry is present only when its source event is.
    """

    grains: frozenset[str] = _ALL_GRAINS
    #: Restrict to these event types (gates both grains). None = all types.
    kinds: frozenset[str] | None = None
    #: Exclusive delta cursor: include only events with ``seq`` strictly greater.
    since_seq: int | None = None
    #: Inclusive upper bound on ``seq``.
    max_seq: int | None = None
    #: Inclusive ISO-8601 time window (lexical compare; assumes UTC events).
    since_time: str | None = None
    until_time: str | None = None
    #: Lexical pre-filter: keep only events surfaced by this verbatim query.
    query: str | None = None
    query_limit: int = 50
    #: Redaction policy: drop entries whose event sensitivity tier is listed.
    exclude_sensitivities: frozenset[str] = field(default_factory=frozenset)
    #: Cap to the most recent N matching *events* (applied before projection).
    limit: int | None = None

    def __post_init__(self) -> None:
        grains = frozenset(self.grains)
        if not grains or not grains <= _ALL_GRAINS:
            raise ValueError(f"grains must be a non-empty subset of {sorted(_ALL_GRAINS)}")
        object.__setattr__(self, "grains", grains)
        if self.kinds is not None:
            object.__setattr__(self, "kinds", frozenset(self.kinds))
        object.__setattr__(self, "exclude_sensitivities", frozenset(self.exclude_sensitivities))
        if self.query_limit < 1:
            raise ValueError("query_limit must be >= 1")
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be >= 0")


def _event_sensitivity(event: Event) -> str:
    security = event.security
    return security.sensitivity if security is not None else "public"


def _event_passes(event: Event, selector: ExportSelector) -> bool:
    if selector.since_seq is not None and event.seq <= selector.since_seq:
        return False
    if selector.max_seq is not None and event.seq > selector.max_seq:
        return False
    if selector.kinds is not None and event.type not in selector.kinds:
        return False
    if selector.since_time is not None and event.timestamp < selector.since_time:
        return False
    if selector.until_time is not None and event.timestamp > selector.until_time:
        return False
    return _event_sensitivity(event) not in selector.exclude_sensitivities


def _entry(
    *,
    grain: Grain,
    kind: str,
    event: Event,
    valid_from: str | None,
    valid_to: str | None,
    source: str,
    content: dict[str, Any],
) -> dict[str, Any]:
    """Build one canonical, self-describing export entry.

    Entries are self-contained (they carry their own schema version and sealed
    citation) so a verifiably-disclosed subset stays meaningful in isolation.
    """
    return {
        "schema_version": EXPORT_ENTRY_SCHEMA_VERSION,
        "grain": grain,
        "kind": kind,
        "citation": _citation(event),
        "seq": event.seq,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "source": source,
        "content": content,
    }


def _event_entry(event: Event) -> dict[str, Any]:
    return _entry(
        grain="event",
        kind=event.type,
        event=event,
        valid_from=event.timestamp,
        valid_to=None,
        source="eventloom",
        content={
            "type": event.type,
            "actor": event.actor,
            "thread": event.thread,
            "payload": event.payload,
        },
    )


def _semantic_entries(event: Event) -> list[dict[str, Any]]:
    """Project an event's deterministic extraction into entity + edge entries."""
    result = extract(event)
    entries: list[dict[str, Any]] = []
    for entity in result.entities:
        entries.append(
            _entry(
                grain="semantic",
                kind=f"entity:{entity.entity_type}",
                event=event,
                valid_from=entity.observed_at,
                valid_to=None,
                source="extraction",
                content={
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                    "properties": entity.properties,
                },
            )
        )
    for edge in result.edges:
        entries.append(
            _entry(
                grain="semantic",
                kind=f"edge:{edge.relation_type}",
                event=event,
                valid_from=edge.valid_from,
                valid_to=edge.valid_to,
                source="extraction",
                content={
                    "source": edge.source,
                    "target": edge.target,
                    "relation_type": edge.relation_type,
                    "confidence": edge.confidence,
                    "inferred": edge.inferred,
                    "inference_method": edge.inference_method,
                    "evidence": edge.evidence,
                },
            )
        )
    return entries


def build_memory_export_view(
    session_id: str,
    selector: ExportSelector | None = None,
    *,
    retrieval_cache: SessionRetrievalCache,
) -> list[dict[str, Any]]:
    """Project a session's memory into canonical, cited export entries.

    Reads through ``retrieval_cache`` only: the verified replay supplies the
    events (so every entry is integrity-backed), and the verbatim index backs the
    optional ``query`` pre-filter. The returned list is deterministically ordered
    (ascending ``seq``; within an event: the event entry, then entities, then
    edges) and feeds :func:`zaxy.portable.build_export` unchanged.
    """
    selector = selector or ExportSelector()
    events = list(retrieval_cache.verified_replay(session_id).events)
    candidates = [event for event in events if _event_passes(event, selector)]

    if selector.query is not None:
        hits = retrieval_cache.verbatim_index(session_id).query(
            selector.query, limit=selector.query_limit
        )
        allowed = {seq for seq in (_seq_from_citation(h.citation) for h in hits) if seq is not None}
        candidates = [event for event in candidates if event.seq in allowed]

    if selector.limit is not None:
        candidates = candidates[-selector.limit :]

    entries: list[dict[str, Any]] = []
    for event in candidates:
        if "event" in selector.grains:
            entries.append(_event_entry(event))
        if "semantic" in selector.grains:
            entries.extend(_semantic_entries(event))
    return entries


def export_cursor(entries: list[dict[str, Any]]) -> int | None:
    """Return the max ``seq`` in ``entries`` for use as the next ``since_seq``.

    A consumer passes this back as ``ExportSelector(since_seq=...)`` to pull only
    entries derived from events appended since the last export (delta sync).
    """
    return max((entry["seq"] for entry in entries), default=None)
