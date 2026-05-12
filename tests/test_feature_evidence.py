"""Tests for product evidence and pruning signals."""

from __future__ import annotations

from zaxy.feature_evidence import (
    FeatureEvidence,
    feature_pruning_candidates,
    render_feature_evidence_report,
)


def test_feature_pruning_candidates_orders_weakest_evidence_first() -> None:
    """Low-usage optional features should be visible pruning candidates."""
    features = [
        FeatureEvidence("memory_checkout", "core", usage_count=42, enabled_by_default=True),
        FeatureEvidence("experimental_adapter", "adapter", usage_count=0, enabled_by_default=False),
        FeatureEvidence("packet_capture", "capture", usage_count=2, enabled_by_default=False),
    ]

    candidates = feature_pruning_candidates(features, minimum_usage=3)

    assert [candidate.name for candidate in candidates] == [
        "experimental_adapter",
        "packet_capture",
    ]


def test_render_feature_evidence_report_marks_keep_and_review() -> None:
    """The report should separate proven core features from review candidates."""
    report = render_feature_evidence_report(
        [
            FeatureEvidence("memory_checkout", "core", usage_count=42, enabled_by_default=True),
            FeatureEvidence("experimental_adapter", "adapter", usage_count=0, enabled_by_default=False),
        ],
        minimum_usage=3,
    )

    assert "memory_checkout: keep" in report
    assert "experimental_adapter: review" in report
