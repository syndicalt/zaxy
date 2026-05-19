"""Tests for incremental context refresh planning."""

from __future__ import annotations

from pathlib import Path

from zaxy.context_refresh import (
    ContextRefreshState,
    collect_source_snapshots,
    load_refresh_state,
    plan_context_refresh,
    save_refresh_state,
)


def test_plan_context_refresh_emits_discovered_and_index_events_for_documents(tmp_path: Path) -> None:
    """First refresh should discover sources and emit document chunks for indexing."""
    doc = tmp_path / "docs" / "guide.md"
    doc.parent.mkdir()
    doc.write_text("# Guide\n\nAlpha\nBeta\n", encoding="utf-8")

    plan = plan_context_refresh(tmp_path, kind="documents", previous=ContextRefreshState.empty())

    assert plan.summary == {
        "kind": "documents",
        "discovered": 1,
        "changed": 0,
        "unchanged": 0,
        "deleted": 0,
        "indexed": 1,
        "retired": 0,
        "transform_changed": 0,
    }
    assert [event["event_type"] for event in plan.events] == [
        "source.discovered",
        "document.indexed",
        "projection.updated",
    ]
    assert plan.events[0]["payload"]["path"] == "docs/guide.md"
    assert plan.events[0]["payload"]["source_kind"] == "documents"
    assert plan.events[1]["payload"]["path"] == "docs/guide.md"
    assert plan.events[2]["payload"]["source_event"] == "document.indexed"
    assert plan.events[1]["payload"]["transform_version"] == "documents-v1"
    assert plan.events[1]["payload"]["source_sha256"] == plan.events[0]["payload"]["sha256"]


def test_plan_context_refresh_skips_unchanged_sources_but_records_freshness(tmp_path: Path) -> None:
    """Refresh should avoid re-indexing unchanged sources while preserving an audit event."""
    doc = tmp_path / "guide.md"
    doc.write_text("Alpha\n", encoding="utf-8")
    snapshots = collect_source_snapshots(tmp_path, kind="documents")
    previous = ContextRefreshState.from_snapshots(kind="documents", snapshots=snapshots)

    plan = plan_context_refresh(tmp_path, kind="documents", previous=previous)

    assert plan.summary["unchanged"] == 1
    assert plan.summary["indexed"] == 0
    assert [event["event_type"] for event in plan.events] == ["source.unchanged"]
    assert plan.events[0]["payload"]["path"] == "guide.md"


def test_collect_codebase_snapshots_uses_indexer_exclusions(tmp_path: Path) -> None:
    """Refresh fingerprints should not track files the codebase indexer ignores."""
    tracked = tmp_path / "src" / "app.py"
    ignored = tmp_path / "node_modules" / "pkg" / "index.js"
    tracked.parent.mkdir()
    ignored.parent.mkdir(parents=True)
    tracked.write_text("def main():\n    return 1\n", encoding="utf-8")
    ignored.write_text("export function ignored() {}\n", encoding="utf-8")

    snapshots = collect_source_snapshots(tmp_path, kind="codebase")

    assert [snapshot.path for snapshot in snapshots] == ["src/app.py"]


def test_plan_context_refresh_detects_changed_and_deleted_code_sources(tmp_path: Path) -> None:
    """Codebase refresh should detect changed files and retire deleted file projections."""
    source = tmp_path / "src" / "app.py"
    deleted = tmp_path / "src" / "old.py"
    source.parent.mkdir()
    source.write_text("def main():\n    return 1\n", encoding="utf-8")
    deleted.write_text("def old():\n    return 0\n", encoding="utf-8")
    previous = ContextRefreshState.from_snapshots(
        kind="codebase",
        snapshots=collect_source_snapshots(tmp_path, kind="codebase"),
    )
    source.write_text("def main():\n    return 2\n", encoding="utf-8")
    deleted.unlink()

    plan = plan_context_refresh(tmp_path, kind="codebase", previous=previous)

    event_types = [event["event_type"] for event in plan.events]
    assert "source.changed" in event_types
    assert "source.deleted" in event_types
    assert "code.file.indexed" in event_types
    assert "projection.updated" in event_types
    assert event_types.count("projection.retired") == 2
    assert plan.summary["changed"] == 1
    assert plan.summary["deleted"] == 1
    assert plan.summary["retired"] == 2
    assert plan.summary["transform_changed"] == 0
    assert plan.summary["indexed"] >= 1
    retired = [event for event in plan.events if event["event_type"] == "projection.retired"]
    assert {event["payload"]["path"] for event in retired} == {"src/app.py", "src/old.py"}
    assert {event["payload"]["reason"] for event in retired} == {"source_changed", "source_deleted"}
    assert all(event["payload"]["source_kind"] == "codebase" for event in retired)


def test_plan_context_refresh_reprocesses_when_transform_version_changes(tmp_path: Path) -> None:
    """A transform-version bump should retire and re-index unchanged sources."""
    source = tmp_path / "README.md"
    source.write_text("Alpha\n", encoding="utf-8")
    previous = ContextRefreshState.from_snapshots(
        kind="documents",
        snapshots=collect_source_snapshots(tmp_path, kind="documents"),
    )
    previous = ContextRefreshState(
        kind=previous.kind,
        sources=previous.sources,
        transform_version="documents-v0",
    )

    plan = plan_context_refresh(tmp_path, kind="documents", previous=previous)

    assert plan.summary["changed"] == 1
    assert plan.summary["retired"] == 1
    assert plan.summary["transform_changed"] == 1
    assert [event["event_type"] for event in plan.events] == [
        "projection.retired",
        "source.changed",
        "document.indexed",
        "projection.updated",
    ]
    assert plan.events[0]["payload"]["reason"] == "transform_changed"
    assert plan.events[1]["payload"]["refresh_reason"] == "transform_changed"


def test_refresh_state_round_trips_under_eventloom_path(tmp_path: Path) -> None:
    """Refresh state should be durable and scoped by session and source kind."""
    eventloom = tmp_path / ".eventloom"
    source = tmp_path / "README.md"
    source.write_text("Alpha\n", encoding="utf-8")
    state = ContextRefreshState.from_snapshots(
        kind="documents",
        snapshots=collect_source_snapshots(tmp_path, kind="documents"),
    )

    save_refresh_state(eventloom, session_id="agent-1", state=state)
    loaded = load_refresh_state(eventloom, session_id="agent-1", kind="documents")

    assert loaded == state
    assert (eventloom / "context-refresh" / "agent-1.documents.json").exists()
