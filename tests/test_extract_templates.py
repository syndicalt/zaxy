"""Tests for extractor authoring templates."""

from __future__ import annotations

import pytest

from zaxy.extract_templates import ExtractorTemplateSpec, render_extractor_template


def test_render_extractor_template_uses_safe_event_metadata() -> None:
    spec = ExtractorTemplateSpec(
        event_type="decision.recorded",
        entity_type="decision",
        entity_name_payload_key="title",
        summary_payload_key="rationale",
        actor_relation_type="recorded_decision",
    )

    rendered = render_extractor_template(spec)

    assert '@register("decision.recorded")' in rendered
    assert "def extract_decision_recorded(event: Event) -> ExtractionResult:" in rendered
    assert 'event.payload.get("title"' in rendered
    assert 'event.payload.get("rationale")' in rendered
    assert 'relation_type="recorded_decision"' in rendered
    assert "source_event_seq=event.seq" in rendered


def test_render_extractor_template_omits_actor_edge_when_not_requested() -> None:
    spec = ExtractorTemplateSpec(
        event_type="artifact.saved",
        entity_type="artifact",
        entity_name_payload_key="path",
    )

    rendered = render_extractor_template(spec)

    assert "ExtractedEdge" not in rendered
    assert "edges=[]" in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_type", 'bad"); import os #'),
        ("entity_type", "bad-type"),
        ("entity_name_payload_key", "bad key"),
        ("summary_payload_key", "bad key"),
        ("actor_relation_type", "bad-relation"),
    ],
)
def test_extractor_template_rejects_unsafe_identifiers(field: str, value: str) -> None:
    kwargs = {
        "event_type": "decision.recorded",
        "entity_type": "decision",
        "entity_name_payload_key": "title",
        "summary_payload_key": None,
        "actor_relation_type": None,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        ExtractorTemplateSpec(**kwargs)
