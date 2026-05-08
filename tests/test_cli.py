"""Tests for Zaxy CLI helper commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.event import EventLog


def test_ide_config_command_prints_copyable_mcp_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["ide-config", "claude-desktop", "--eventloom-path", ".eventloom"],
    )

    assert result.exit_code == 0
    assert '"mcpServers"' in result.output
    assert '"zaxy"' in result.output
    assert '"args": [' in result.output
    assert "token" not in result.output.casefold()


def test_schema_plan_command_prints_migration_plan() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema-plan"])

    assert result.exit_code == 0
    assert "Current schema version:" in result.output
    assert "entity_version_identity" in result.output


def test_extractor_template_command_prints_safe_starter() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "extractor-template",
            "decision.recorded",
            "--entity-type",
            "decision",
            "--name-key",
            "title",
            "--summary-key",
            "rationale",
            "--actor-relation",
            "recorded_decision",
        ],
    )

    assert result.exit_code == 0
    assert '@register("decision.recorded")' in result.output
    assert 'relation_type="recorded_decision"' in result.output


def test_compact_audit_reports_identity_safety_without_rewriting_log(tmp_path: Path) -> None:
    """compact --audit should report safety findings and leave the log untouched."""
    log_path = tmp_path / "work.jsonl"
    log = EventLog(log_path)
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/a.md",
            "start_line": 1,
            "end_line": 3,
            "content": "Runbook source records identity-code-0001.",
        },
    )
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/b.md",
            "start_line": 1,
            "end_line": 3,
            "content": "Runbook source records identity-code-0002.",
        },
    )
    before = Path(log_path).read_text(encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["compact", str(log_path), "--audit"])

    assert result.exit_code == 1
    assert "Compaction audit: UNSAFE" in result.output
    assert "Identity recall:" in result.output
    assert "Missing identities:" in result.output
    assert Path(log_path).read_text(encoding="utf-8") == before


def test_compact_audit_json_output(tmp_path: Path) -> None:
    """compact --audit --json should emit machine-readable audit results."""
    log_path = tmp_path / "work.jsonl"
    EventLog(log_path).append("goal.created", actor="user", payload={"title": "Ship"})
    runner = CliRunner()

    result = runner.invoke(app, ["compact", str(log_path), "--audit", "--json"])

    assert result.exit_code == 0
    assert '"safe": true' in result.output
    assert '"identity_recall": 1.0' in result.output


def test_compact_writes_projection_without_rewriting_log(tmp_path: Path) -> None:
    """compact --projection-output should store backpointer projections only."""
    log_path = tmp_path / "work.jsonl"
    projection_path = tmp_path / "work.compaction.json"
    EventLog(log_path).append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/context.md",
            "start_line": 10,
            "end_line": 12,
            "content": "Context note records identity-code-0001.",
        },
    )
    before = log_path.read_text(encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "compact",
            str(log_path),
            "--projection-output",
            str(projection_path),
            "--strategy",
            "medoid",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote compaction projection:" in result.output
    assert projection_path.exists()
    assert '"strategy": "medoid"' in projection_path.read_text(encoding="utf-8")
    assert log_path.read_text(encoding="utf-8") == before
