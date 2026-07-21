"""Wire codec for out-of-process Zaxy plugins.

The host and the plugin worker exchange **newline-delimited JSON** (one JSON
object per line) over the worker's stdin/stdout pipes. The codec lives in its
own module because both sides must agree on it byte for byte, and because the
worker imports it *before* it imports any plugin code.

Protocol (``PROTOCOL_VERSION`` is sent in every handshake)::

    host -> worker   {"op": "describe"}
    worker -> host   {"ok": true, "protocol": 1, "name": ..., "version": ...,
                      "event_types": [...]}

    host -> worker   {"op": "extract", "event_type": ..., "event": {...}}
    worker -> host   {"ok": true, "result": {...}}

    host -> worker   {"op": "shutdown"}

Any failure is reported as ``{"ok": false, "error": "..."}`` rather than by
closing the pipe, so the host can distinguish a *handled* plugin error from a
crash (pipe closed) or a hang (no line within the deadline).

Encoding is total for the extraction data model: :class:`ExtractedEntity`
properties and :class:`ExtractedEdge` evidence are arbitrary JSON, which is how
citations survive the round trip intact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult

PROTOCOL_VERSION = 1


def encode_event(event: Event) -> dict[str, Any]:
    """Encode an :class:`~zaxy.event.Event` into a JSON-safe dict."""
    return dict(event.model_dump(mode="json"))


def decode_event(payload: Any) -> Event:
    """Decode a JSON-safe dict back into an :class:`~zaxy.event.Event`."""
    from zaxy.event import Event

    if not isinstance(payload, dict):
        raise ValueError("event payload must be a JSON object")
    return Event.model_validate(payload)


def encode_extraction_result(result: ExtractionResult) -> dict[str, Any]:
    """Encode an :class:`~zaxy.extract.ExtractionResult` into a JSON-safe dict."""
    return {
        "entities": [
            {
                "name": entity.name,
                "entity_type": entity.entity_type,
                "observed_at": entity.observed_at,
                "summary": entity.summary,
                "embedding": list(entity.embedding) if entity.embedding is not None else None,
                "properties": entity.properties,
            }
            for entity in result.entities
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "relation_type": edge.relation_type,
                "valid_from": edge.valid_from,
                "valid_to": edge.valid_to,
                "inferred": edge.inferred,
                "confidence": edge.confidence,
                "inference_method": edge.inference_method,
                "evidence": edge.evidence,
            }
            for edge in result.edges
        ],
        "source_event_seq": result.source_event_seq,
        "source_event_hash": result.source_event_hash,
        "source_event_prev_hash": result.source_event_prev_hash,
        "source_event_type": result.source_event_type,
        "source_thread": result.source_thread,
    }


def decode_extraction_result(payload: Any) -> ExtractionResult:
    """Decode a JSON-safe dict back into an :class:`~zaxy.extract.ExtractionResult`."""
    from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult

    if not isinstance(payload, dict):
        raise ValueError("extraction result must be a JSON object")

    entities = []
    for raw in payload.get("entities") or []:
        if not isinstance(raw, dict):
            raise ValueError("extraction entity must be a JSON object")
        embedding = raw.get("embedding")
        entities.append(
            ExtractedEntity(
                name=str(raw["name"]),
                entity_type=str(raw["entity_type"]),
                observed_at=str(raw["observed_at"]),
                summary=raw.get("summary"),
                embedding=[float(value) for value in embedding] if embedding is not None else None,
                properties=raw.get("properties"),
            )
        )

    edges = []
    for raw in payload.get("edges") or []:
        if not isinstance(raw, dict):
            raise ValueError("extraction edge must be a JSON object")
        edges.append(
            ExtractedEdge(
                source=str(raw["source"]),
                target=str(raw["target"]),
                relation_type=str(raw["relation_type"]),
                valid_from=str(raw["valid_from"]),
                valid_to=raw.get("valid_to"),
                inferred=bool(raw.get("inferred", False)),
                confidence=float(raw.get("confidence", 1.0)),
                inference_method=raw.get("inference_method"),
                evidence=dict(raw.get("evidence") or {}),
            )
        )

    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=int(payload["source_event_seq"]),
        source_event_hash=payload.get("source_event_hash"),
        source_event_prev_hash=payload.get("source_event_prev_hash"),
        source_event_type=payload.get("source_event_type"),
        source_thread=payload.get("source_thread"),
    )
