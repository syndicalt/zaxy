"""Tests for neutral document/transcript substrate helpers."""

from __future__ import annotations

from zaxy.neutral import (
    audit_ingestion_purpose_labels,
    build_purpose_projection_record,
    neutral_document_record,
)


def test_neutral_document_record_preserves_source_without_purpose_label() -> None:
    record = neutral_document_record(
        actor="support-agent",
        timestamp="2026-06-02T12:00:00Z",
        path="customers/acme-email.txt",
        start_line=1,
        end_line=4,
        content="ACME asks whether the export clause applies to dashboard data.",
        source_event_ref="eventloom://default/events/5#abc",
    )

    assert record.to_properties()["permission_scope"] == "project-local"
    assert record.to_properties()["candidate_claim"] == (
        "ACME asks whether the export clause applies to dashboard data"
    )
    assert "legal_obligation" not in record.to_properties().values()


def test_purpose_projection_rebuilds_from_neutral_record_with_backpointer() -> None:
    record = neutral_document_record(
        actor="support-agent",
        timestamp="2026-06-02T12:00:00Z",
        path="customers/acme-email.txt",
        start_line=1,
        end_line=4,
        content="ACME asks whether the export clause applies to dashboard data.",
        source_event_ref="eventloom://default/events/5#abc",
    )

    legal = build_purpose_projection_record(
        record,
        purpose_profile="legal",
        purpose_label="legal_obligation",
    )
    product = build_purpose_projection_record(
        record.to_properties() | {"substrate_id": record.substrate_id},
        purpose_profile="product",
        purpose_label="roadmap_commitment",
    )

    assert legal.neutral_substrate_id == record.substrate_id
    assert legal.source_event_ref == "eventloom://default/events/5#abc"
    assert legal.source_backpointer == "customers/acme-email.txt:1-4"
    assert legal.purpose_label == "legal_obligation"
    assert product.neutral_substrate_id == record.substrate_id
    assert product.purpose_label == "roadmap_commitment"
    assert product.quote == legal.quote


def test_audit_ingestion_purpose_labels_flags_irreversible_labels() -> None:
    audit = audit_ingestion_purpose_labels(
        {
            "content": "ACME asks about export terms.",
            "labels": ["legal_obligation", "churn_risk"],
        },
        source_event_ref="eventloom://default/events/5#abc",
    )

    assert audit.safe is False
    assert audit.forbidden_labels == ("churn_risk", "legal_obligation")
    assert audit.action == "flag_ingestion_purpose_label"
