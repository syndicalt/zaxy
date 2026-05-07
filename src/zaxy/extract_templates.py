"""Extractor authoring templates for new typed Eventloom events."""

from __future__ import annotations

import re
from dataclasses import dataclass

_EVENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ExtractorTemplateSpec:
    """Validated metadata for rendering a deterministic extractor starter."""

    event_type: str
    entity_type: str
    entity_name_payload_key: str
    summary_payload_key: str | None = None
    actor_relation_type: str | None = None

    def __post_init__(self) -> None:
        _validate("event_type", self.event_type, _EVENT_TYPE_RE)
        _validate("entity_type", self.entity_type, _IDENTIFIER_RE)
        _validate("entity_name_payload_key", self.entity_name_payload_key, _IDENTIFIER_RE)
        if self.summary_payload_key is not None:
            _validate("summary_payload_key", self.summary_payload_key, _IDENTIFIER_RE)
        if self.actor_relation_type is not None:
            _validate("actor_relation_type", self.actor_relation_type, _IDENTIFIER_RE)


def render_extractor_template(spec: ExtractorTemplateSpec) -> str:
    """Render a safe Python extractor starter for a typed event.

    The output is intentionally explicit so authors can copy it into
    ``extract.py`` and then tighten payload handling with event-specific tests.
    """
    function_name = f"extract_{spec.event_type.replace('.', '_')}"
    summary_expr = f'_payload_text(event.payload.get("{spec.summary_payload_key}"))' if spec.summary_payload_key else "None"
    imports = "from zaxy.extract import ExtractedEntity, ExtractionResult, register"
    edge_import = ""
    edge_block = "    edges=[]"
    if spec.actor_relation_type:
        imports = "from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult, register"
        edge_import = "\n"
        edge_block = f'''    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=event.actor,
        target=name,
        relation_type="{spec.actor_relation_type}",
        valid_from=event.timestamp,
    )
    entities.append(actor)
    edges = [edge]'''

    return f'''"""Extractor for {spec.event_type} events."""

from __future__ import annotations

from zaxy.event import Event
{imports}{edge_import}

@register("{spec.event_type}")
def {function_name}(event: Event) -> ExtractionResult:
    """Extract {spec.entity_type} memory from {spec.event_type}."""
    name = _payload_text(event.payload.get("{spec.entity_name_payload_key}")) or f"{spec.entity_type}:{{event.seq}}"
    entity = ExtractedEntity(
        name=name,
        entity_type="{spec.entity_type}",
        observed_at=event.timestamp,
        summary={summary_expr},
    )
    entities = [entity]
{edge_block}
    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=event.seq,
    )


def _payload_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
'''


def _validate(field: str, value: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise ValueError(f"{field} must match {pattern.pattern}")
