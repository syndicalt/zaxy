"""A well-behaved out-of-process plugin fixture emitting citation-bearing data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult


def extract_cited(event: Event) -> ExtractionResult:
    """Return an extraction carrying citations, nested properties, and an embedding."""
    from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult

    entity = ExtractedEntity(
        name=f"remote-note:{event.seq}",
        entity_type="remote_note",
        observed_at=event.timestamp,
        summary=str(event.payload.get("text") or ""),
        embedding=[0.25, -1.5, 3.0],
        properties={
            "citations": ["src/mod.py:10-14", "docs/spec.md:3"],
            "nested": {"depth": [1, 2, {"leaf": True}]},
            "score": 0.5,
        },
    )
    edge = ExtractedEdge(
        source=entity.name,
        target="remote-target",
        relation_type="cites",
        valid_from=event.timestamp,
        confidence=0.75,
        evidence={"citation": "src/mod.py:10-14", "lines": [10, 14]},
    )
    return ExtractionResult(
        entities=[entity],
        edges=[edge],
        source_event_seq=event.seq,
        source_event_type=event.type,
        source_thread=event.thread,
    )


class GoodPlugin:
    """Fixture plugin registering the ``remote.note`` extractor."""

    name = "remote-good"
    version = "9.9"

    def register(self, api: Any) -> None:
        """Install the citation-bearing extractor."""
        api.register_extractor("remote.note", extract_cited)


PLUGIN = GoodPlugin()
