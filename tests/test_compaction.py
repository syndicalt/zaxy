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


def test_coordinate_compaction_keeps_only_authoritative_parent_rows_searchable(
    tmp_path: Path,
) -> None:
    """Coordinate compaction must not promote pending/rejected/stale worker rows."""
    log = EventLog(tmp_path / "coordinate.jsonl")
    pending = log.append(
        "coordination.finding.reported",
        actor="worker-api",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-api",
            "finding_id": "finding-pending",
            "claim_key": "release.package",
            "claim_value": "pending-wrong-claim",
            "coordination_status": "pending",
            "summary": "Pending worker-local claim should stay diagnostic.",
        },
    )
    accepted = log.append(
        "coordination.finding.reviewed",
        actor="coordinator",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-release",
            "finding_id": "finding-accepted",
            "claim_key": "release.package",
            "claim_value": "zaxy-memory-1.0.2-ready",
            "coordination_status": "accepted",
            "promotion_event_ref": "eventloom://release-1/events/7#aaaaaaaaaaaa",
            "review_event_ref": "eventloom://release-1/events/6#bbbbbbbbbbbb",
            "source_event_ref": "eventloom://worker-release/events/3#cccccccccccc",
            "summary": "Accepted parent state is authoritative.",
        },
    )
    rejected = log.append(
        "coordination.finding.reviewed",
        actor="coordinator",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-ui",
            "finding_id": "finding-rejected",
            "claim_key": "release.package",
            "claim_value": "rejected-claim",
            "coordination_status": "rejected",
            "summary": "Rejected row must not become authoritative memory.",
        },
    )
    stale = log.append(
        "coordination.finding.reported",
        actor="worker-docs",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-docs",
            "finding_id": "finding-stale",
            "claim_key": "release.docs",
            "claim_value": "old-doc-state",
            "coordination_status": "pending",
            "stale": True,
            "superseded_by": "finding-accepted",
            "summary": "Stale unpromoted row stays diagnostic.",
        },
    )

    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="medoid",
        max_records=1,
        purpose="coordinate",
    )

    assert projection.strategy == "coordinate_authoritative"
    assert projection.purpose["profile"] == "coordinate"
    assert [record.event_seq for record in projection.records] == [accepted.seq]
    assert projection.records[0].authority_scope == "authoritative"
    assert projection.records[0].purpose_reasons == ("accepted_parent_state",)
    assert projection.records[0].kind == "coordinate_authoritative"
    assert projection.consolidation_policy["requested_strategy"] == "medoid"
    assert projection.consolidation_policy["max_records_ignored"] is False
    assert projection.consolidation_policy["authoritative_event_seqs"] == [accepted.seq]
    assert projection.consolidation_policy["diagnostic_event_seqs"] == [
        pending.seq,
        rejected.seq,
        stale.seq,
    ]
    assert projection.consolidation_policy["suppressed_count"] == 3

    assert search_compaction_projections([projection], "zaxy-memory-1.0.2-ready")
    assert search_compaction_projections([projection], "pending-wrong-claim") == []
    assert search_compaction_projections([projection], "rejected-claim") == []
    assert search_compaction_projections([projection], "old-doc-state") == []


def test_security_compaction_preserves_all_source_records_for_risk_audit(
    tmp_path: Path,
) -> None:
    """Security purpose should not collapse distinct findings into one medoid."""
    log = EventLog(tmp_path / "security.jsonl")
    for idx, label in enumerate(("secret exposure", "auth bypass", "risk acceptance"), start=1):
        log.append(
            "document.indexed",
            actor="security-reviewer",
            payload={
                "path": f"security/finding-{idx}.md",
                "start_line": idx,
                "end_line": idx + 1,
                "content": f"Security review records {label} identity-code-000{idx}.",
            },
        )

    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="medoid",
        max_records=1,
        purpose="security",
    )

    assert projection.strategy == "purpose_preserve_all"
    assert projection.consolidation_policy["preserve_all"] is True
    assert projection.consolidation_policy["effective_max_records"] == 3
    assert [record.kind for record in projection.records] == [
        "security_retained",
        "security_retained",
        "security_retained",
    ]
    assert all("security_findings" in record.purpose_reasons for record in projection.records)
    assert search_compaction_projections([projection], "auth bypass")
    assert search_compaction_projections([projection], "risk acceptance")


def test_coding_compaction_uses_bounded_purpose_exemplars_with_record_floor(
    tmp_path: Path,
) -> None:
    """Coding purpose should preserve a broader exemplar set than generic medoid collapse."""
    log = EventLog(tmp_path / "coding.jsonl")
    for idx in range(10):
        log.append(
            "document.indexed",
            actor="developer",
            payload={
                "path": f"tests/regression-{idx}.md",
                "start_line": idx + 1,
                "end_line": idx + 2,
                "content": f"Regression invariant identity-code-{idx:04d} must survive.",
            },
        )

    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="medoid",
        max_records=2,
        purpose="coding",
    )

    assert projection.strategy == "purpose_exemplar"
    assert projection.consolidation_policy["preserve_all"] is False
    assert projection.consolidation_policy["effective_max_records"] == 8
    assert len(projection.records) == 8
    assert {record.kind for record in projection.records} == {"coding_exemplar"}
    assert all("test_results" in record.purpose_reasons for record in projection.records)


def test_broader_profile_compaction_rules_are_explicit(tmp_path: Path) -> None:
    expected = {
        "support": ("purpose_exemplar", "support_exemplar", False),
        "product": ("purpose_exemplar", "product_exemplar", False),
        "sales": ("purpose_preserve_all", "sales_retained", True),
        "legal": ("purpose_preserve_all", "legal_retained", True),
        "executive": ("purpose_preserve_all", "executive_retained", True),
    }

    for profile, (strategy, kind, preserve_all) in expected.items():
        log = EventLog(tmp_path / f"{profile}.jsonl")
        for idx in range(3):
            log.append(
                "document.indexed",
                actor=f"{profile}-operator",
                payload={
                    "path": f"{profile}/fixture-{idx}.md",
                    "start_line": idx + 1,
                    "end_line": idx + 2,
                    "content": f"{profile} fixture identity-code-{idx:04d} should survive.",
                },
            )

        projection = build_compaction_projection(
            log,
            provider=HashEmbeddingProvider(dimension=64),
            strategy="medoid",
            max_records=1,
            purpose=profile,
        )

        assert projection.purpose["profile"] == profile
        assert projection.strategy == strategy
        assert projection.consolidation_policy["preserve_all"] is preserve_all
        assert projection.consolidation_policy["retain"]
        assert projection.consolidation_policy["suppress"]
        assert {record.kind for record in projection.records} == {kind}
        assert all(record.purpose_reasons for record in projection.records)


def test_coordinate_compaction_blocks_status_erasure_from_authority(
    tmp_path: Path,
) -> None:
    """Accepted-looking rows without authority refs should be diagnostic-only."""
    log = EventLog(tmp_path / "coordinate-erased.jsonl")
    erased = log.append(
        "coordination.finding.reviewed",
        actor="coordinator",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-api",
            "finding_id": "finding-erased",
            "claim_key": "release.package",
            "claim_value": "accepted-without-proof",
            "coordination_status": "accepted",
            "summary": "Accepted text without promotion/review/source refs is not authority.",
        },
    )

    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        strategy="exemplar",
        purpose="coordinate",
    )

    assert projection.records == ()
    assert projection.consolidation_policy["authoritative_count"] == 0
    assert projection.consolidation_policy["diagnostic_event_seqs"] == [erased.seq]
    assert projection.consolidation_policy["suppressed_count"] == 1
    assert search_compaction_projections([projection], "accepted-without-proof") == []


def test_coordinate_compaction_roundtrip_preserves_authority_metadata(tmp_path: Path) -> None:
    """Projection JSON should preserve purpose, authority, and diagnostic metadata."""
    log = EventLog(tmp_path / "coordinate-roundtrip.jsonl")
    accepted = log.append(
        "coordination.finding.promoted",
        actor="coordinator",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-release",
            "finding_id": "finding-promoted",
            "claim_key": "release.package",
            "claim_value": "ready",
            "coordination_status": "promoted",
            "summary": "Promoted finding survives compaction.",
        },
    )
    pending = log.append(
        "coordination.finding.reported",
        actor="worker-release",
        payload={
            "mission_id": "release-1",
            "worker_id": "worker-release",
            "finding_id": "finding-pending",
            "claim_key": "release.package",
            "claim_value": "pending",
            "coordination_status": "pending",
            "summary": "Pending finding is diagnostic.",
        },
    )
    projection = build_compaction_projection(
        log,
        provider=HashEmbeddingProvider(dimension=64),
        purpose="coordinate",
    )
    output = write_compaction_projection(projection, tmp_path / "coordinate.compaction.json")

    loaded = load_compaction_projection(output)

    assert loaded.purpose["profile"] == "coordinate"
    assert loaded.strategy == "coordinate_authoritative"
    assert loaded.records[0].event_seq == accepted.seq
    assert loaded.records[0].kind == "coordinate_authoritative"
    assert loaded.records[0].authority_scope == "authoritative"
    assert loaded.records[0].purpose_reasons == ("accepted_parent_state",)
    assert loaded.consolidation_policy["diagnostic_event_seqs"] == [pending.seq]
    assert "finding-pending" in loaded.consolidation_policy["diagnostic_identities"]
