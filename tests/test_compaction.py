"""Tests for compaction safety audits."""

from __future__ import annotations

import json
from pathlib import Path

from zaxy.compaction import (
    audit_event_log,
    build_compaction_projection,
    compaction_remediation_plan,
    load_compaction_projection,
    search_compaction_projections,
    write_compaction_projection,
)
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

    plan = compaction_remediation_plan(report)
    assert plan[0]["code"] == "preserve_missing_identities"
    assert "identity-code-0004" in plan[0]["details"]["missing_identities"]
    assert plan[0]["action"] == "Use exemplar projection or increase max_records before compacting."


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

    plan = compaction_remediation_plan(report)
    assert any(step["code"] == "restore_source_citations" for step in plan)


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

    plan = compaction_remediation_plan(report)
    assert plan[0] == {
        "code": "repair_eventloom_integrity",
        "action": "Restore the Eventloom log from backup or remove the tampered candidate from compaction.",
        "details": {"reason": "Event 1 hash mismatch"},
    }


def test_builds_medoid_projection_with_source_backpointers(tmp_path: Path) -> None:
    """A medoid projection should keep a real source event and all source IDs."""
    log = EventLog(tmp_path / "projection.jsonl")
    first = log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/a.md",
            "start_line": 1,
            "end_line": 4,
            "content": "Cache migration note records identity-code-0001.",
        },
    )
    second = log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/b.md",
            "start_line": 5,
            "end_line": 8,
            "content": "Cache migration note records identity-code-0002.",
        },
    )

    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="medoid",
    )

    assert projection.strategy == "medoid"
    assert projection.source_event_count == 2
    assert len(projection.records) == 1
    assert projection.records[0].event_ref.startswith("eventloom://default/events/")
    assert projection.records[0].kind == "medoid"
    assert projection.records[0].event_seq in {first.seq, second.seq}
    assert "docs/a.md:1-4" in projection.source_identities
    assert "docs/b.md:5-8" in projection.source_identities
    assert projection.audit.identity_recall < 1.0


def test_builds_exemplar_projection_with_multiple_cited_records(tmp_path: Path) -> None:
    """Exemplar projections should store multiple real records with source citations."""
    log = EventLog(tmp_path / "exemplars.jsonl")
    for idx in range(4):
        log.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": f"docs/exemplar-{idx}.md",
                "start_line": idx + 1,
                "end_line": idx + 2,
                "content": f"Planner note records identity-code-{idx:04d}.",
            },
        )

    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="exemplar",
        max_records=3,
    )

    assert projection.strategy == "exemplar"
    assert len(projection.records) == 3
    assert {record.kind for record in projection.records} == {"exemplar"}
    assert all(record.citations for record in projection.records)
    assert len({record.event_seq for record in projection.records}) == 3
    assert "identity-code-0003" in projection.source_identities


def test_writes_projection_json_for_later_context_assembly(tmp_path: Path) -> None:
    """Projection storage should produce deterministic JSON with backpointers."""
    log = EventLog(tmp_path / "projection.jsonl")
    log.append(
        "goal.created",
        actor="user",
        payload={"title": "Goal 0001", "description": "Ship projection storage"},
    )
    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="medoid",
    )
    output = tmp_path / "projection.compaction.json"

    write_compaction_projection(projection, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["projection_id"] == projection.projection_id
    assert payload["strategy"] == "medoid"
    assert payload["source_identities"] == list(projection.source_identities)
    assert payload["records"][0]["event_ref"].startswith("eventloom://default/events/1#")


def test_loads_projection_json_roundtrip(tmp_path: Path) -> None:
    """Projection JSON should load back into typed records."""
    log = EventLog(tmp_path / "projection.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/cache.md",
            "start_line": 2,
            "end_line": 6,
            "content": "Cache routing note records identity-code-0001.",
        },
    )
    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="medoid",
    )
    output = write_compaction_projection(projection, tmp_path / "projection.json")

    loaded = load_compaction_projection(output)

    assert loaded == projection
    assert loaded.records[0].citations == projection.records[0].citations


def test_searches_projection_records_with_source_citations(tmp_path: Path) -> None:
    """Projection search should return cited routing candidates."""
    log = EventLog(tmp_path / "projection.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/cache.md",
            "start_line": 2,
            "end_line": 6,
            "content": "Cache routing note records identity-code-0001.",
        },
    )
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/auth.md",
            "start_line": 4,
            "end_line": 8,
            "content": "Auth routing note records identity-code-0002.",
        },
    )
    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="exemplar",
        max_records=2,
    )

    results = search_compaction_projections([projection], "cache identity-code-0001", limit=1)

    assert len(results) == 1
    assert results[0].projection_id == projection.projection_id
    assert results[0].record.text
    assert "docs/cache.md:2-6" in results[0].citations
    assert results[0].score > 0.0
