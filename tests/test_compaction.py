"""Tests for compaction safety audits."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.compaction import audit_event_log
from zaxy.embedding import HashEmbeddingProvider
from zaxy.event import EventLog


def test_single_event_log_is_safe_to_represent_directly(tmp_path: Path) -> None:
    """A one-event cluster preserves its identity in its own representative text."""
    log = EventLog(tmp_path / "single.jsonl")
    event = log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/one.md",
            "start_line": 1,
            "end_line": 3,
            "content": "The single source carries identity-code-0001.",
        },
    )

    report = audit_event_log(log, provider=HashEmbeddingProvider(dimension=64))

    assert report.safe is True
    assert report.event_count == 1
    assert report.integrity_ok is True
    assert report.identity_count >= 3
    assert report.identity_recall == 1.0
    assert report.citation_coverage == 1.0
    assert report.mean_within_cluster_distance == 0.0
    assert not report.unsafe_reasons
    assert f"eventloom://default/events/{event.seq}#{event.hash[:12]}" in report.identities
    assert "docs/one.md:1-3" in report.identities


def test_audit_flags_centroid_like_identity_loss(tmp_path: Path) -> None:
    """Near-duplicate records with distinct IDs should fail identity preservation."""
    log = EventLog(tmp_path / "collapse.jsonl")
    for idx in range(5):
        log.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": f"docs/service-{idx:04d}.md",
                "start_line": 1,
                "end_line": 5,
                "content": (
                    "Migration rollback readiness note records "
                    f"identity-code-{idx:04d} for release coordination."
                ),
            },
        )

    report = audit_event_log(log, provider=HashEmbeddingProvider(dimension=64))

    assert report.safe is False
    assert report.event_count == 5
    assert report.identity_recall < 1.0
    assert "identity recall below 1.000" in report.unsafe_reasons
    assert report.missing_identities
    assert "docs/service-0004.md:1-5" in report.missing_identities


def test_audit_detects_missing_source_citations(tmp_path: Path) -> None:
    """Document events without path citations should be called out."""
    log = EventLog(tmp_path / "missing-citation.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "content": "A document chunk exists without a durable source path.",
        },
    )

    report = audit_event_log(log, provider=HashEmbeddingProvider(dimension=64))

    assert report.safe is False
    assert report.citation_coverage < 1.0
    assert "citation coverage below 1.000" in report.unsafe_reasons


def test_audit_blocks_broken_eventloom_integrity(tmp_path: Path) -> None:
    """Broken hash-chain integrity should make compaction unsafe."""
    log_path = tmp_path / "tampered.jsonl"
    log = EventLog(log_path)
    log.append("goal.created", actor="user", payload={"title": "Original"})
    raw = json.loads(log_path.read_text(encoding="utf-8"))
    raw["payload"]["title"] = "Tampered"
    log_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    report = audit_event_log(log, provider=HashEmbeddingProvider(dimension=64))

    assert report.safe is False
    assert report.integrity_ok is False
    assert report.integrity_reason == "Event 1 hash mismatch"
    assert "integrity check failed" in report.unsafe_reasons
