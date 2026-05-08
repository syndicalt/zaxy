"""Tests for Zaxy CLI helper commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.event import EventLog


def test_ide_config_command_prints_copyable_mcp_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["ide-config", "claude-desktop", "--eventloom-path", ".eventloom", "--domain", "zaxy"],
    )

    assert result.exit_code == 0
    assert '"mcpServers"' in result.output
    assert '"zaxy"' in result.output
    assert '"args": [' in result.output
    assert '"EVENTLOOM_THREAD": "zaxy-default"' in result.output
    assert '"ZAXY_DOMAIN": "zaxy"' in result.output
    assert '"ZAXY_ENV": "development"' in result.output
    assert '"NEO4J_URI": "bolt://localhost:7687"' in result.output
    assert '"NEO4J_AUTO_START": "true"' in result.output
    assert '"NEO4J_CA_CERT": ""' in result.output
    assert '"NEO4J_PASSWORD_FILE": ""' in result.output
    assert '"MCP_ADMIN_TOKEN_FILE": ""' in result.output
    assert '"MCP_REMOTE_AUTH_TOKEN_FILE": ""' in result.output
    assert '"OPENAI_API_KEY_FILE": ""' in result.output
    assert '"PATHLIGHT_ACCESS_TOKEN_FILE": ""' in result.output
    assert "testpassword" not in result.output.casefold()


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


def test_local_profile_command_prints_offline_env() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["local-profile"])

    assert result.exit_code == 0
    assert "EMBEDDING_PROVIDER=hash" in result.output
    assert "RERANKER_PROVIDER=lexical" in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_local_profile_command_writes_output_file(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / ".env.local"

    result = runner.invoke(app, ["local-profile", "--output", str(target)])

    assert result.exit_code == 0
    assert "Wrote local profile" in result.output
    assert "RERANKER_PROVIDER=lexical" in target.read_text(encoding="utf-8")


def test_local_profile_check_reports_success() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["local-profile", "--check"])

    assert result.exit_code == 0
    assert '"status": "ok"' in result.output
    assert '"reranker_provider": "lexical"' in result.output


@patch("zaxy.__main__.MemoryFabric")
def test_index_codebase_command_reports_indexed_count(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """index-codebase should append codebase file events through MemoryFabric."""
    fabric = AsyncMock()
    fabric.ingest_codebase.return_value = 3
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["index-codebase", str(tmp_path), "--session-id", "agent-1", "--max-bytes", "1024"],
    )

    assert result.exit_code == 0
    assert "Indexed 3 code files into session agent-1" in result.output
    fabric.ingest_codebase.assert_awaited_once_with(tmp_path, session_id="agent-1", max_bytes=1024)
    fabric.close.assert_awaited_once()


@patch("zaxy.__main__.GraphStore")
def test_reproject_command_replays_log_into_graph(mock_graph_store: MagicMock, tmp_path: Path) -> None:
    """reproject should rebuild graph projections from an Eventloom log."""
    log_path = tmp_path / "default.jsonl"
    log = EventLog(log_path)
    log.append(
        "decision.made",
        actor="assistant",
        payload={
            "decision": "Use structured Eventloom trace.",
            "rationale": ["Supports replayable memory."],
        },
        thread="default",
    )
    store = AsyncMock()
    mock_graph_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "reproject",
            str(log_path),
            "--session-id",
            "default",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Reprojected 1 events into session default" in result.output
    mock_graph_store.assert_called_once_with("bolt://test:7687", "neo4j", "testpassword")
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    extraction = store.upsert_extraction.await_args.args[0]
    assert extraction.entities[0].entity_type == "decision"
    assert store.upsert_extraction.await_args.kwargs == {"session_id": "default"}
    store.close.assert_awaited_once()


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
