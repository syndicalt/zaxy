"""Tests for Zaxy CLI helper commands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.main import get_command
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.event import EventLog


def test_version_option_reports_project_version() -> None:
    """The installed CLI should expose the packaged Zaxy version."""
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == "zaxy 0.2.1"


def test_memory_status_prints_eventloom_sessions(tmp_path: Path) -> None:
    """memory status should summarize Eventloom sessions without Neo4j."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    first = log.append("goal.created", actor="user", payload={"title": "Ship it"}, thread="agent-1")
    second = log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use source-aware assembly."},
        thread="agent-1",
    )
    assert first.seq == 1

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memory", "status", "--eventloom-path", str(tmp_path / ".eventloom")],
    )

    assert result.exit_code == 0
    assert "Eventloom: " in result.output
    assert "Sessions: 1" in result.output
    assert "Total events: 2" in result.output
    assert "agent-1" in result.output
    assert "events=2" in result.output
    assert "latest=2" in result.output
    assert second.hash[:12] in result.output
    assert "integrity=OK" in result.output


def test_memory_status_json_output(tmp_path: Path) -> None:
    """memory status --json should expose stable machine-readable fields."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "transcript.turn",
        actor="assistant",
        payload={"content": "Recorded source recall."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memory", "status", "--eventloom-path", str(tmp_path / ".eventloom"), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["eventloom_path"] == str((tmp_path / ".eventloom").resolve())
    assert payload["session_count"] == 1
    assert payload["total_events"] == 1
    assert payload["sessions"][0]["session_id"] == "agent"
    assert payload["sessions"][0]["latest_seq"] == event.seq
    assert payload["sessions"][0]["latest_hash"] == event.hash
    assert payload["sessions"][0]["integrity_ok"] is True


@patch("zaxy.__main__.GraphStore")
def test_memory_status_graph_json_reports_projection_health(
    mock_graph_store: MagicMock,
    tmp_path: Path,
) -> None:
    """memory status --graph should compare Eventloom and Neo4j projection state."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Projected graph chain."},
        thread="agent",
    )
    graph = AsyncMock()
    projection = MagicMock()
    projection.to_dict.return_value = {
        "session_id": "agent",
        "event_count": 1,
        "latest_seq": 1,
        "latest_hash": event.hash,
        "eventloom_latest_seq": 1,
        "eventloom_latest_hash": event.hash,
        "projection_lag": 0,
        "latest_hash_matches": True,
        "next_event_edges": 0,
        "previous_event_edges": 0,
        "missing_chain_links": 0,
        "integrity_ok": True,
    }
    graph.inspect_event_projection_status.return_value = projection
    mock_graph_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--graph",
            "--json",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["graph"]["sessions"][0]["session_id"] == "agent"
    assert payload["graph"]["sessions"][0]["integrity_ok"] is True
    mock_graph_store.assert_called_once_with("bolt://test:7687", "neo4j", "testpassword")
    graph.connect.assert_awaited_once()
    graph.inspect_event_projection_status.assert_awaited_once_with(
        "agent",
        eventloom_latest_seq=event.seq,
        eventloom_latest_hash=event.hash,
    )
    graph.close.assert_awaited_once()


def test_memory_status_handles_empty_eventloom_directory(tmp_path: Path) -> None:
    """memory status should be useful before any memory has been written."""
    eventloom_dir = tmp_path / ".eventloom"
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "status", "--eventloom-path", str(eventloom_dir)])

    assert result.exit_code == 0
    assert "Sessions: 0" in result.output
    assert "Total events: 0" in result.output


@patch("zaxy.__main__.GraphStore")
def test_memory_inferred_status_json_reports_graph_audit(
    mock_graph_store: MagicMock,
) -> None:
    """memory inferred-status --json should expose inferred-edge audit metadata."""
    graph = AsyncMock()
    status = MagicMock()
    status.to_dict.return_value = {
        "session_id": "agent",
        "total_edges": 3,
        "method_count": 1,
        "evidence_count": 2,
        "missing_evidence_count": 1,
        "missing_source_event_count": 0,
        "evidence_coverage": 0.666667,
        "methods": [
            {
                "method": "task_completed_decision_citation_v1",
                "edge_count": 3,
                "relation_types": ["likely_implemented_decision"],
                "average_confidence": 0.86,
                "minimum_confidence": 0.86,
                "evidence_count": 2,
                "missing_evidence_count": 1,
                "missing_source_event_count": 0,
            }
        ],
        "samples": [],
    }
    graph.inspect_inferred_edge_status.return_value = status
    mock_graph_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "inferred-status",
            "--session-id",
            "agent",
            "--limit",
            "7",
            "--json",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent"
    assert payload["total_edges"] == 3
    assert payload["methods"][0]["method"] == "task_completed_decision_citation_v1"
    mock_graph_store.assert_called_once_with("bolt://test:7687", "neo4j", "testpassword")
    graph.connect.assert_awaited_once()
    graph.inspect_inferred_edge_status.assert_awaited_once_with("agent", limit=7)
    graph.close.assert_awaited_once()


@patch("zaxy.__main__.GraphStore")
def test_memory_inferred_status_text_reports_evidence_gaps(
    mock_graph_store: MagicMock,
) -> None:
    """The human inferred-edge status should call out evidence coverage and gaps."""
    graph = AsyncMock()
    status = MagicMock()
    status.to_dict.return_value = {
        "session_id": "agent",
        "total_edges": 2,
        "method_count": 1,
        "evidence_count": 1,
        "missing_evidence_count": 1,
        "missing_source_event_count": 0,
        "evidence_coverage": 0.5,
        "methods": [
            {
                "method": "task_completed_decision_citation_v1",
                "edge_count": 2,
                "relation_types": ["likely_implemented_decision"],
                "average_confidence": 0.86,
                "minimum_confidence": 0.86,
                "evidence_count": 1,
                "missing_evidence_count": 1,
                "missing_source_event_count": 0,
            }
        ],
        "samples": [
            {
                "source": "task-7",
                "target": "decision:Use graph audit",
                "relation_type": "likely_implemented_decision",
                "confidence": 0.86,
                "method": "task_completed_decision_citation_v1",
                "source_event_seq": 12,
                "source_event_hash": "a" * 64,
                "evidence_keys": ["evidence_source_event_seq", "evidence_reason"],
            }
        ],
    }
    graph.inspect_inferred_edge_status.return_value = status
    mock_graph_store.return_value = graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "inferred-status",
            "--session-id",
            "agent",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Inferred edges: agent" in result.output
    assert "total=2" in result.output
    assert "evidence_coverage=50.0%" in result.output
    assert "task_completed_decision_citation_v1" in result.output
    assert "missing_evidence=1" in result.output
    assert "task-7 -[likely_implemented_decision]-> decision:Use graph audit" in result.output


def test_memory_capabilities_json_output(tmp_path: Path) -> None:
    """memory capabilities should expose a session-scoped model contract."""
    EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Capability manifest target."},
        thread="agent",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "capabilities",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "make zaxy invisible",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent"
    assert payload["current_task"] == "make zaxy invisible"
    assert payload["recommended_next_call"]["tool"] == "memory_checkout"
    assert payload["status"]["eventloom"]["latest_seq"] == 1


def test_memory_capabilities_text_output(tmp_path: Path) -> None:
    """The text form should be prompt-ready for model session bootstrap."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "capabilities",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
        ],
    )

    assert result.exit_code == 0
    assert "# Zaxy Memory Contract" in result.output
    assert "Session: agent" in result.output
    assert "memory_checkout" in result.output


def test_memory_bootstrap_json_output(tmp_path: Path) -> None:
    """memory bootstrap should expose a model-facing session-start handoff."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "bootstrap",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--current-task",
            "ship the next sprint",
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "session_start"
    assert payload["session_id"] == "agent"
    assert payload["startup_sequence"][1]["tool"] == "memory_checkout"
    assert payload["startup_sequence"][1]["arguments"]["query"] == "ship the next sprint"
    assert payload["capture"]["configured"] is False


def test_memory_bootstrap_text_output(tmp_path: Path) -> None:
    """The text form should be compact enough to inject into model startup context."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "bootstrap",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "# Zaxy Session Bootstrap" in result.output
    assert "1. memory_capabilities" in result.output
    assert "2. memory_checkout" in result.output


@patch("zaxy.__main__.MemoryFabric")
def test_memory_checkout_json_output(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """memory checkout --json should expose the Memory Checkout contract."""
    checkout = MagicMock()
    checkout.to_dict.return_value = {
        "session_id": "agent-1",
        "query": "current project direction",
        "prompt": "# Memory Checkout\nUse Memory Checkout.",
        "current_facts": [{"content": "Use Memory Checkout.", "citation": "eventloom://agent-1/events/1#abc"}],
        "evidence": [{"citation": "eventloom://agent-1/events/1#abc"}],
        "provenance": [{"event_seq": 1, "event_hash": "abc"}],
        "warnings": [],
    }
    fabric = AsyncMock()
    fabric.checkout_memory.return_value = checkout
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "checkout",
            "current project direction",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--ref",
            "refs/heads/main",
            "--neo4j-uri",
            "bolt://localhost:7687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "testpassword",
            "--neo4j-ca-cert",
            "",
            "--neo4j-trust-all",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent-1"
    assert payload["current_facts"][0]["citation"] == "eventloom://agent-1/events/1#abc"
    mock_fabric_cls.assert_called_once_with(
        eventloom_path=str(tmp_path / ".eventloom"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpassword",
        neo4j_ca_cert="",
        neo4j_trust_all=True,
    )
    fabric.connect.assert_awaited_once()
    fabric.checkout_memory.assert_awaited_once_with(
        "current project direction",
        session_id="agent-1",
        ref="refs/heads/main",
        replay_from_seq=1,
        limit=10,
        max_recent_events=20,
    )
    fabric.close.assert_awaited_once()


def test_packet_analyzer_cli_help_exposes_observe_only_gateway() -> None:
    """packet-analyzer should expose the low-latency observe-only gateway."""
    runner = CliRunner()

    result = runner.invoke(app, ["packet-analyzer", "--help"])
    command = get_command(app).commands["packet-analyzer"]
    option_names = {option for parameter in command.params for option in getattr(parameter, "opts", [])}

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "packet-analyzer" in result.output
    assert "--upstream-base-url" in option_names
    assert "--eventloom-path" in option_names
    assert "--session-id" in option_names


def test_packet_project_cli_projects_completed_packets(tmp_path: Path) -> None:
    """packet-project should run the cold-path packet projection worker."""
    eventloom_dir = tmp_path / ".eventloom"
    EventLog(eventloom_dir / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the codename is Atlas."}},
            "response": {"body": {"output_text": "I will remember Atlas."}},
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["packet-project", "--eventloom-path", str(eventloom_dir), "--session-id", "agent-1"],
    )

    assert result.exit_code == 0
    assert "Projected 1 packet event" in result.output
    events = EventLog(eventloom_dir / "agent-1.jsonl").read_all()
    assert events[-1].type == "llm.packet.projected"
    assert "Atlas" in events[-1].payload["summary"]


@patch("zaxy.__main__.GraphStore")
def test_packet_project_cli_can_project_new_packets_to_graph(
    mock_graph_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """packet-project --graph should upsert newly projected packets into Neo4j."""
    eventloom_dir = tmp_path / ".eventloom"
    EventLog(eventloom_dir / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the product owner is Nia."}},
            "response": {"body": {"output_text": "Product owner Nia recorded."}},
        },
    )
    mock_graph = AsyncMock()
    mock_graph_cls.return_value = mock_graph
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "packet-project",
            "--eventloom-path",
            str(eventloom_dir),
            "--session-id",
            "agent-1",
            "--graph",
            "--neo4j-uri",
            "bolt://localhost:7687",
            "--neo4j-user",
            "neo4j",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Projected 1 packet event" in result.output
    assert "graph_projected=1" in result.output
    assert "graph_failed=0" in result.output
    mock_graph.connect.assert_awaited_once()
    mock_graph.init_schema.assert_awaited_once()
    mock_graph.upsert_extraction.assert_awaited_once()
    mock_graph.close.assert_awaited_once()


def test_packet_project_cli_supports_bounded_watch_mode(tmp_path: Path) -> None:
    """packet-project watch mode should support bounded runs for supervisors/tests."""
    eventloom_dir = tmp_path / ".eventloom"
    EventLog(eventloom_dir / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the on-call is Dev."}},
            "response": {"body": {"output_text": "On-call Dev recorded."}},
        },
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "packet-project",
            "--eventloom-path",
            str(eventloom_dir),
            "--session-id",
            "agent-1",
            "--watch",
            "--watch-iterations",
            "2",
            "--interval-seconds",
            "0",
        ],
    )

    assert result.exit_code == 0
    assert "Watched 2 projection pass" in result.output
    assert "projected=1" in result.output


def test_memory_log_prints_recent_events(tmp_path: Path) -> None:
    """memory log should print recent events in compact git-style form."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Use memory log."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["memory", "log", "--eventloom-path", str(tmp_path / ".eventloom")],
    )

    assert result.exit_code == 0
    assert f"agent [{event.seq}] {event.hash[:12]}" in result.output
    assert "decision.recorded by assistant" in result.output
    assert "Use memory log." in result.output


def test_memory_log_json_filters_session_and_limit(tmp_path: Path) -> None:
    """memory log --json should expose stable event entries with filtering."""
    agent_log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    agent_log.append(
        "goal.created",
        actor="user",
        payload={"title": "Older"},
        thread="agent",
    )
    event = agent_log.append(
        "task.completed",
        actor="assistant",
        payload={"summary": "Newest"},
        thread="agent",
    )
    EventLog(tmp_path / ".eventloom" / "other.jsonl").append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Skip"},
        thread="other",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "memory",
            "log",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--limit",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["eventloom_path"] == str((tmp_path / ".eventloom").resolve())
    assert payload["limit"] == 1
    assert payload["session_id"] == "agent"
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["seq"] == event.seq
    assert payload["entries"][0]["hash"] == event.hash
    assert payload["entries"][0]["summary"] == "Newest"


def test_memory_diff_prints_event_range(tmp_path: Path) -> None:
    """memory diff should print added events in the requested sequence range."""
    log = EventLog(tmp_path / ".eventloom" / "agent.jsonl")
    log.append("goal.created", actor="user", payload={"title": "Older"}, thread="agent")
    event = log.append(
        "decision.recorded",
        actor="assistant",
        payload={"decision": "Add diff."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "memory",
            "diff",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--from-seq",
            "2",
            "--to-seq",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert f"agent +[{event.seq}] {event.hash[:12]} decision.recorded by assistant" in result.output
    assert "Add diff." in result.output


def test_memory_diff_json_output(tmp_path: Path) -> None:
    """memory diff --json should expose stable added event entries."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Added diff CLI."},
        thread="agent",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "memory",
            "diff",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--from-seq",
            "1",
            "--to-seq",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "agent"
    assert payload["from_seq"] == 1
    assert payload["to_seq"] == 1
    assert payload["integrity_ok"] is True
    assert payload["added"][0]["seq"] == event.seq
    assert payload["added"][0]["summary"] == "Added diff CLI."


def test_memory_ref_update_and_list(tmp_path: Path) -> None:
    """memory ref should create durable git-style refs and list latest pointers."""
    event = EventLog(tmp_path / ".eventloom" / "agent.jsonl").append(
        "task.completed",
        actor="codex",
        payload={"summary": "Ref target."},
        thread="agent",
    )
    runner = CliRunner()

    update = runner.invoke(
        app,
        [
            "memory",
            "ref",
            "refs/heads/main",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent",
            "--target-seq",
            str(event.seq),
            "--target-hash",
            event.hash,
            "--type",
            "branch",
        ],
    )
    listed = runner.invoke(
        app,
        ["memory", "refs", "--eventloom-path", str(tmp_path / ".eventloom"), "--json"],
    )

    assert update.exit_code == 0
    assert "refs/heads/main -> agent@1" in update.output
    assert listed.exit_code == 0
    payload = json.loads(listed.output)
    assert payload["refs"][0]["name"] == "refs/heads/main"
    assert payload["refs"][0]["target_hash"] == event.hash


def test_ide_config_command_prints_copyable_mcp_json() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "claude-desktop",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert '"mcpServers"' in result.output
    assert '"zaxy"' in result.output
    assert '"command": "/opt/zaxy/bin/zaxy"' in result.output
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


def test_ide_config_command_installs_project_cursor_config(tmp_path: Path) -> None:
    """ide-config --install should merge into the verified project-local target."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "cursor",
            "--install",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Installed cursor MCP config" in result.output
    config = json.loads((tmp_path / ".cursor" / "mcp.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["zaxy"]["command"] == "/opt/zaxy/bin/zaxy"


def test_ide_config_command_prints_codex_cli_install_command() -> None:
    """Codex install should keep workspace state out of global MCP config."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--install",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Run this Codex MCP install command:" in result.output
    assert "codex mcp add zaxy" in result.output
    assert "--env EVENTLOOM_THREAD" not in result.output
    assert "ZAXY_DOMAIN" not in result.output
    assert "--env NEO4J_URI=bolt://localhost:7687" in result.output
    assert "--env NEO4J_CA_CERT=" in result.output
    assert "--env NEO4J_PASSWORD_FILE=" in result.output
    assert "-- /opt/zaxy/bin/zaxy serve" in result.output
    assert "--eventloom-path" not in result.output


def test_ide_config_command_prints_codex_cli_command_without_install_flag() -> None:
    """Codex print mode should be useful and should not expose internal helper names."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "codex mcp add zaxy" in result.output
    assert "render_codex_mcp_add_command" not in result.output


def test_ide_config_command_prints_hermes_yaml_config() -> None:
    """Hermes print mode should emit the config.yaml shape without repo-local state."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "hermes",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "mcp_servers:" in result.output
    assert "zaxy:" in result.output
    assert "command: /opt/zaxy/bin/zaxy" in result.output
    assert "- serve" in result.output
    assert "memory_checkout" in result.output
    assert "EVENTLOOM_PATH" not in result.output
    assert "EVENTLOOM_THREAD" not in result.output
    assert "ZAXY_DOMAIN" not in result.output


def test_ide_config_command_writes_hermes_config(tmp_path: Path) -> None:
    """Hermes install should merge into an explicit config.yaml path."""
    runner = CliRunner()
    target = tmp_path / "config.yaml"
    target.write_text("model:\n  default: anthropic/claude-opus-4.6\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "ide-config",
            "hermes",
            "--install",
            "--hermes-config",
            str(target),
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert f"Wrote Hermes MCP config to {target}" in result.output
    config = target.read_text(encoding="utf-8")
    assert "mcp_servers:" in config
    assert "zaxy:" in config
    assert "command: /opt/zaxy/bin/zaxy" in config
    assert "EVENTLOOM_PATH" not in config


def test_ide_config_command_writes_trusted_project_codex_config(tmp_path: Path) -> None:
    """Codex direct config writes should require explicit project trust acknowledgement."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--install",
            "--codex-config-scope",
            "project",
            "--codex-trusted-project",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--zaxy-executable",
            "/opt/zaxy/bin/zaxy",
        ],
    )

    assert result.exit_code == 0
    assert "Wrote Codex MCP config" in result.output
    config = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "[mcp_servers.zaxy]" in config
    assert 'command = "/opt/zaxy/bin/zaxy"' in config
    assert 'args = ["serve"]' in config
    assert "EVENTLOOM_PATH" not in config
    assert 'NEO4J_URI = "bolt://localhost:7687"' in config
    assert 'NEO4J_CA_CERT = ""' in config
    assert 'NEO4J_PASSWORD_FILE = ""' in config


def test_ide_config_command_rejects_project_codex_config_without_trust(tmp_path: Path) -> None:
    """Project-scoped Codex writes should fail before touching disk without trust acknowledgement."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ide-config",
            "codex",
            "--install",
            "--codex-config-scope",
            "project",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code != 0
    assert "trusted" in result.output
    assert "project" in result.output
    assert not (tmp_path / ".codex" / "config.toml").exists()


@patch("zaxy.__main__.mcp_main", new_callable=AsyncMock)
@patch("zaxy.mcp_server.ZaxyMCPServer")
def test_serve_derives_workspace_defaults_when_not_overridden(
    mock_server_cls: MagicMock,
    mock_mcp_main: AsyncMock,
    monkeypatch,
) -> None:
    """A bare `zaxy serve` should scope memory to the process workspace."""
    monkeypatch.setattr("zaxy.mcp_server.server", None)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["serve"],
        catch_exceptions=False,
        obj=None,
        env={},
        color=False,
        prog_name="zaxy",
    )

    assert result.exit_code == 0
    mock_mcp_main.assert_awaited_once()
    mock_server_cls.assert_called_once()
    kwargs = mock_server_cls.call_args.kwargs
    assert kwargs["eventloom_path"] == str(Path.cwd() / ".eventloom")
    assert kwargs["workspace_root"] == Path.cwd()
    assert kwargs["default_session_id"] == "zaxy-default"


def test_integration_template_command_prints_framework_starter() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["integration-template", "langgraph", "--session-id", "zaxy-default"],
    )

    assert result.exit_code == 0
    assert "async def zaxy_langgraph_memory_node" in result.output
    assert "from zaxy import MemoryFabric" in result.output
    assert "session_id='zaxy-default'" in result.output
    assert "import langgraph" not in result.output.casefold()


def test_integration_template_command_can_print_install_hint() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["integration-template", "crewai", "--install-hint"],
    )

    assert result.exit_code == 0
    assert "python -m pip install 'zaxy-memory[crewai]'" in result.output
    assert "async def zaxy_crewai_memory_step" in result.output


def test_integrations_command_lists_framework_registry() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["integrations"])

    assert result.exit_code == 0
    assert "LangGraph" in result.output
    assert "zaxy-memory[langgraph]" in result.output
    assert "native-preview" in result.output
    assert "zaxy.adapters.langgraph" in result.output


def test_integrations_command_can_emit_json() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["integrations", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload[0]["framework"] == "langgraph"
    assert payload[0]["install"] == "python -m pip install 'zaxy-memory[langgraph]'"
    assert payload[0]["maturity"] == "native-preview"
    assert payload[0]["native_adapter"] == "zaxy.adapters.langgraph"


def test_hooks_command_prints_claude_code_settings(tmp_path: Path) -> None:
    """hooks should render copyable observer hook config."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hooks",
            "claude-code",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--domain",
            "zaxy",
        ],
    )

    assert result.exit_code == 0
    assert '"hooks"' in result.output
    assert '"Stop"' in result.output
    assert '"PreCompact"' in result.output
    assert "zaxy hook-event stop" in result.output
    assert "zaxy hook-event precompact" in result.output
    assert "--session-id zaxy-default" in result.output


def test_hooks_command_writes_output_file(tmp_path: Path) -> None:
    """hooks --output should write config instead of printing it."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"

    result = runner.invoke(
        app,
        [
            "hooks",
            "claude-code",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "Wrote hook config" in result.output
    assert output.is_file()
    assert '"PreCompact"' in output.read_text(encoding="utf-8")
    assert '"hooks"' not in result.output


def test_hooks_command_merges_claude_local_settings(tmp_path: Path) -> None:
    """Claude hook install should preserve unrelated local settings and hooks."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(pytest)"]},
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Edit",
                            "hooks": [{"type": "command", "command": "ruff check ."}],
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "hooks",
            "claude-code",
            "--eventloom-path",
            ".eventloom",
            "--domain",
            "zaxy",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    settings = json.loads(output.read_text(encoding="utf-8"))
    assert settings["permissions"]["allow"] == ["Bash(pytest)"]
    assert settings["hooks"]["PostToolUse"][0]["hooks"][0]["command"] == "ruff check ."
    assert "zaxy hook-event stop" in json.dumps(settings["hooks"]["Stop"])
    assert "zaxy hook-event precompact" in json.dumps(settings["hooks"]["PreCompact"])


def test_hooks_command_refuses_duplicate_claude_zaxy_hooks_without_force(tmp_path: Path) -> None:
    """Claude hook install should not duplicate existing Zaxy hook handlers."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"
    output.parent.mkdir()
    output.write_text(
        '{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "zaxy hook-event stop"}]}]}}\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["hooks", "claude-code", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "already contains Zaxy hook handlers" in result.output
    assert output.read_text(encoding="utf-8").count("zaxy hook-event") == 1


def test_hooks_command_force_replaces_claude_zaxy_hooks(tmp_path: Path) -> None:
    """Claude --force should replace Zaxy handlers while preserving unrelated settings."""
    runner = CliRunner()
    output = tmp_path / ".claude" / "settings.local.json"
    output.parent.mkdir()
    output.write_text(
        json.dumps(
            {
                "env": {"KEEP": "1"},
                "hooks": {
                    "Stop": [
                        {"hooks": [{"type": "command", "command": "zaxy hook-event stop --source old"}]},
                        {"hooks": [{"type": "command", "command": "echo keep"}]},
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["hooks", "claude-code", "--domain", "zaxy", "--output", str(output), "--force"],
    )

    assert result.exit_code == 0
    settings = json.loads(output.read_text(encoding="utf-8"))
    assert settings["env"] == {"KEEP": "1"}
    serialized = json.dumps(settings)
    assert "--source old" not in serialized
    assert "echo keep" in serialized
    assert "zaxy hook-event stop" in serialized


def test_hook_status_ignores_non_hook_text_in_claude_settings(tmp_path: Path) -> None:
    """Hook detection should inspect command handlers, not arbitrary JSON text."""
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir()
    settings_path.write_text('{"notes": "zaxy hook-event is mentioned here"}\n', encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["hook-status", "--eventloom-path", str(tmp_path / ".eventloom"), "--workspace-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "claude-code: not installed" in result.output


def test_hooks_command_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    """hooks --output should be non-destructive by default."""
    runner = CliRunner()
    output = tmp_path / "hooks.sh"
    output.write_text("existing\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["hooks", "generic", "--output", str(output)],
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == "existing\n"


def test_hooks_command_generic_output_documents_observation_sinks() -> None:
    """Generic hook output should advertise every first-class observation sink."""
    runner = CliRunner()

    result = runner.invoke(app, ["hooks", "generic", "--domain", "zaxy"])

    assert result.exit_code == 0
    assert "zaxy hook-event command" in result.output
    assert "zaxy hook-event file-edit" in result.output
    assert "zaxy hook-event tool-call" in result.output
    assert "zaxy hook-event transcript-turn" in result.output


def test_hooks_command_force_overwrites_output_file(tmp_path: Path) -> None:
    """hooks --force should replace an existing output file."""
    runner = CliRunner()
    output = tmp_path / "hooks.sh"
    output.write_text("existing\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["hooks", "generic", "--domain", "zaxy", "--output", str(output), "--force"],
    )

    assert result.exit_code == 0
    assert "Wrote hook config" in result.output
    assert "zaxy hook-event session-start" in output.read_text(encoding="utf-8")


def test_hook_event_command_appends_eventloom_event(tmp_path: Path) -> None:
    """hook-event should append lightweight lifecycle observations without Neo4j."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "precompact",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded hook precompact" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert len(events) == 1
    assert events[0].type == "hook.precompact"
    assert events[0].actor == "zaxy-hook"
    assert events[0].thread == "agent-1"
    assert events[0].payload["source"] == "codex"


def test_hook_event_checkpoint_carries_summary_and_reason(tmp_path: Path) -> None:
    """checkpoint hooks should carry retrieval-useful checkpoint metadata."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "checkpoint",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--summary",
            "Finished hook install mode.",
            "--reason",
            "manual",
            "--turn-count",
            "7",
        ],
    )

    assert result.exit_code == 0
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "hook.checkpoint"
    assert events[0].payload["summary"] == "Finished hook install mode."
    assert events[0].payload["reason"] == "manual"
    assert events[0].payload["turn_count"] == 7


def test_hook_event_heartbeat_appends_health_event(tmp_path: Path) -> None:
    """heartbeat hooks should prove the observer path can write Eventloom."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "heartbeat",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "claude-code",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded hook heartbeat" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "hook.heartbeat"
    assert events[0].payload["trigger"] == "heartbeat"
    assert events[0].payload["source"] == "claude-code"


def test_hook_event_command_observation_appends_normalized_event(tmp_path: Path) -> None:
    """hook-event command should write first-class command.completed observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "command",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--workspace",
            "/repo",
            "--command",
            "pytest --token secret",
            "--exit-code",
            "1",
            "--stdout",
            "ok",
            "--stderr",
            "failed",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation command.completed" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "command.completed"
    assert events[0].actor == "zaxy-observer"
    assert events[0].payload["command"] == "pytest [REDACTED]"
    assert events[0].payload["source"] == "codex"
    assert events[0].payload["workspace"] == "/repo"


def test_hook_event_file_edit_observation_appends_normalized_event(tmp_path: Path) -> None:
    """hook-event file-edit should write first-class file.edit.applied observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "file-edit",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--workspace",
            "/repo",
            "--path",
            "src/zaxy/core.py",
            "--operation",
            "modified",
            "--summary",
            "Updated context assembly.",
            "--line-count",
            "12",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation file.edit.applied" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "file.edit.applied"
    assert events[0].actor == "zaxy-observer"
    assert events[0].payload["path"] == "src/zaxy/core.py"
    assert events[0].payload["summary"] == "Updated context assembly."
    assert "content" not in events[0].payload


def test_hook_event_tool_call_observation_appends_redacted_event(tmp_path: Path) -> None:
    """hook-event tool-call should write first-class tool.call.completed observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "tool-call",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--workspace",
            "/repo",
            "--tool-name",
            "functions.exec_command",
            "--tool-status",
            "ok",
            "--call-id",
            "call-123",
            "--arguments-json",
            '{"cmd": "pytest", "token": "secret"}',
            "--result-summary",
            "3 passed",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation tool.call.completed" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "tool.call.completed"
    assert events[0].actor == "zaxy-observer"
    assert events[0].payload["tool_name"] == "functions.exec_command"
    assert events[0].payload["argument_keys"] == ["cmd", "token"]
    assert events[0].payload["arguments_redacted"] is True
    assert "arguments" not in events[0].payload
    assert events[0].payload["source"] == "codex"
    assert events[0].payload["workspace"] == "/repo"


def test_hook_event_transcript_turn_observation_appends_sanitized_event(tmp_path: Path) -> None:
    """hook-event transcript-turn should write sanitized transcript.turn observations."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-event",
            "transcript-turn",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--source",
            "codex",
            "--role",
            "assistant",
            "--content",
            "Use token sk-test-secret for the demo.",
            "--turn-index",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "Recorded observation transcript.turn" in result.output
    events = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").read_all()
    assert events[0].type == "transcript.turn"
    assert events[0].actor == "assistant"
    assert events[0].payload["source"] == "codex"
    assert events[0].payload["turn_index"] == 7
    assert events[0].payload["role"] == "assistant"
    assert "sk-test-secret" not in events[0].payload["content"]
    assert events[0].payload["redacted_paths"]


def test_hook_status_reports_observation_type_coverage(tmp_path: Path) -> None:
    """hook-status should show which automatic capture types are active."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"trigger": "heartbeat", "source": "codex"},
        thread="agent-1",
    )
    command = log.append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest", "exit_code": 0},
        thread="agent-1",
    )
    log.append(
        "file.edit.applied",
        actor="zaxy-observer",
        payload={"source": "codex", "path": "src/zaxy/core.py", "operation": "modified"},
        thread="agent-1",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["observation_coverage"]["command.completed"]["count"] == 1
    assert payload["observation_coverage"]["command.completed"]["latest"]["seq"] == command.seq
    assert payload["observation_coverage"]["file.edit.applied"]["count"] == 1
    assert payload["observation_coverage"]["transcript.turn"]["count"] == 0
    assert "transcript.turn" in payload["missing_observation_types"]
    assert payload["capture_readiness"] == {
        "status": "warning",
        "message": "2 of 4 high-value automatic capture lanes are active",
        "active_observation_types": ["command.completed", "file.edit.applied"],
        "missing_observation_types": ["tool.call.completed", "transcript.turn"],
        "actions": [
            "Wire hooks or adapter sinks for: tool.call.completed, transcript.turn.",
        ],
    }


def test_hook_status_reports_complete_observation_coverage(tmp_path: Path) -> None:
    """hook-status should clear missing coverage once every high-value type is captured."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    log.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("file.edit.applied", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("tool.call.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("transcript.turn", actor="assistant", payload={"source": "codex"}, thread="agent-1")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["observation_coverage"]["tool.call.completed"]["count"] == 1
    assert payload["observation_coverage"]["transcript.turn"]["count"] == 1
    assert payload["missing_observation_types"] == []
    assert payload["capture_readiness"] == {
        "status": "ok",
        "message": "4 of 4 high-value automatic capture lanes are active",
        "active_observation_types": [
            "command.completed",
            "file.edit.applied",
            "tool.call.completed",
            "transcript.turn",
        ],
        "missing_observation_types": [],
        "actions": [],
    }


@patch("zaxy.hooks._iter_process_cmdlines")
def test_hook_status_reports_codex_capture_watcher_runtime(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """hook-status should distinguish installed Codex capture config from a running watcher."""
    capture_config = tmp_path / ".codex" / "zaxy-capture.json"
    capture_config.parent.mkdir()
    capture_config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = [
        (
            123,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--session-id",
                "agent-1",
                "--watch",
            ],
        )
    ]
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["clients"]["codex"]["installed"] is True
    assert payload["clients"]["codex"]["runtime"] == {
        "running": True,
        "pids": [123],
        "message": "Codex capture watcher is running",
    }

    text = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
        ],
    )

    assert text.exit_code == 0
    assert "codex: installed (.codex/zaxy-capture.json)" in text.output
    assert "codex capture: running pid=123" in text.output


@patch("zaxy.hooks._iter_process_cmdlines")
def test_hook_status_warns_when_codex_capture_configured_but_not_running(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """hook-status should not treat stale Codex coverage as an active watcher."""
    capture_config = tmp_path / ".codex" / "zaxy-capture.json"
    capture_config.parent.mkdir()
    capture_config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
            }
        ),
        encoding="utf-8",
    )
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    log.append("hook.heartbeat", actor="zaxy-hook", payload={"source": "codex"}, thread="agent-1")
    log.append("command.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("file.edit.applied", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("tool.call.completed", actor="zaxy-observer", payload={"source": "codex"}, thread="agent-1")
    log.append("transcript.turn", actor="assistant", payload={"source": "codex"}, thread="agent-1")
    mock_processes.return_value = []
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "hook-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--workspace-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "warning"
    assert payload["clients"]["codex"]["runtime"]["running"] is False
    assert payload["capture_readiness"]["status"] == "warning"
    assert payload["capture_readiness"]["actions"] == [
        f"Start managed deterministic Codex capture: zaxy capture start --workspace {tmp_path}."
    ]


@patch("zaxy.hooks._iter_process_cmdlines")
def test_capture_status_reports_configured_codex_watcher_runtime(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """capture status should expose managed deterministic capture posture."""
    config = tmp_path / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "codex_home": str(tmp_path / ".codex-home"),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
                "source": "codex-local",
                "workspace": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = [
        (
            321,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--session-id",
                "agent-1",
                "--watch",
            ],
        )
    ]
    runner = CliRunner()

    result = runner.invoke(app, ["capture", "status", "--workspace", str(tmp_path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["client"] == "codex"
    assert payload["configured"] is True
    assert payload["running"] is True
    assert payload["pids"] == [321]
    assert payload["state_file"] == str(tmp_path / ".eventloom" / "runtime" / "codex-capture.json")


@patch.object(subprocess, "Popen")
def test_capture_start_launches_managed_codex_watcher(
    mock_popen: MagicMock,
    tmp_path: Path,
) -> None:
    """capture start should launch a watcher from repo-local Codex capture config."""
    config = tmp_path / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "codex_home": str(tmp_path / ".codex-home"),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "agent-1",
                "source": "codex-local",
                "workspace": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    process = MagicMock()
    process.pid = 321
    mock_popen.return_value = process
    runner = CliRunner()

    result = runner.invoke(app, ["capture", "start", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Started Codex capture watcher pid=321" in result.output
    command = mock_popen.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "zaxy"]
    assert "codex-capture" in command
    assert "--watch" in command
    assert mock_popen.call_args.kwargs["start_new_session"] is True
    state = json.loads((tmp_path / ".eventloom" / "runtime" / "codex-capture.json").read_text(encoding="utf-8"))
    assert state["pid"] == 321
    assert state["client"] == "codex"
    assert state["workspace"] == str(tmp_path)


@patch("os.kill")
@patch("zaxy.hooks._iter_process_cmdlines")
def test_capture_stop_only_stops_matching_managed_codex_watcher(
    mock_processes: MagicMock,
    mock_kill: MagicMock,
    tmp_path: Path,
) -> None:
    """capture stop should stop the managed watcher without targeting unrelated processes."""
    runtime = tmp_path / ".eventloom" / "runtime"
    runtime.mkdir(parents=True)
    state_file = runtime / "codex-capture.json"
    state_file.write_text(
        json.dumps(
            {
                "client": "codex",
                "pid": 321,
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "command": ["python", "-m", "zaxy", "codex-capture", "--watch"],
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = [
        (
            321,
            [
                "python",
                "-m",
                "zaxy",
                "codex-capture",
                "--workspace",
                str(tmp_path),
                "--eventloom-path",
                str(tmp_path / ".eventloom"),
                "--session-id",
                "agent-1",
                "--watch",
            ],
        )
    ]
    runner = CliRunner()

    result = runner.invoke(app, ["capture", "stop", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "Stopped Codex capture watcher pid=321" in result.output
    mock_kill.assert_called_once()
    assert mock_kill.call_args.args[0] == 321
    assert not state_file.exists()


def test_hooks_status_reports_installed_clients_and_recent_activity(tmp_path: Path) -> None:
    """hook-status should answer whether Zaxy is observing this workspace."""
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir()
    settings_path.write_text('{"hooks": {"Stop": [{"hooks": [{"command": "zaxy hook-event stop"}]}]}}', encoding="utf-8")
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"trigger": "heartbeat", "source": "claude-code"},
        thread="agent-1",
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["hook-status", "--eventloom-path", str(tmp_path / ".eventloom"), "--workspace-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Zaxy hooks: ok" in result.output
    assert "claude-code: installed" in result.output
    assert "codex: not installed" in result.output
    assert "last event: hook.heartbeat" in result.output
    assert "capture readiness: warning - 0 of 4 high-value automatic capture lanes are active" in result.output
    assert "command.completed: missing" in result.output
    assert "agent-1" in result.output


def test_schema_plan_command_prints_migration_plan() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["schema-plan"])

    assert result.exit_code == 0
    assert "Current schema version:" in result.output
    assert "entity_version_identity" in result.output


def test_schema_recovery_plan_command_prints_recovery_guidance() -> None:
    runner = CliRunner()

    with (
        patch("zaxy.__main__.GraphStore") as mock_store_cls,
        patch("zaxy.__main__.fetch_schema_migration_records", new_callable=AsyncMock) as mock_fetch,
    ):
        store = AsyncMock()
        mock_store_cls.return_value = store
        mock_fetch.return_value = {
            "001_entity_version_identity": {
                "checksum": "wrong",
                "statement_count": 4,
                "applied_at": "2026-05-11T00:00:00Z",
            }
        }
        result = runner.invoke(app, ["schema-recovery-plan"])

    assert result.exit_code == 0
    store.connect.assert_awaited_once()
    store.close.assert_awaited_once()
    assert "Schema recovery plan:" in result.output
    assert "001_entity_version_identity: checksum_mismatch" in result.output


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
    assert "NEO4J_URI=bolt://localhost:7687" in result.output
    assert "NEO4J_USER=neo4j" in result.output
    assert "NEO4J_PASSWORD=testpassword" in result.output
    assert "NEO4J_CA_CERT=" in result.output
    assert "NEO4J_PASSWORD_FILE=" in result.output
    assert "NEO4J_TRUST_ALL=false" in result.output
    assert "OPENAI_API_KEY" not in result.output


def test_local_profile_command_writes_output_file(tmp_path: Path) -> None:
    runner = CliRunner()
    target = tmp_path / ".env.local"

    result = runner.invoke(app, ["local-profile", "--output", str(target)])

    assert result.exit_code == 0
    assert "Wrote local profile" in result.output
    profile = target.read_text(encoding="utf-8")
    assert "RERANKER_PROVIDER=lexical" in profile
    assert "NEO4J_URI=bolt://localhost:7687" in profile
    assert "NEO4J_CA_CERT=" in profile
    assert "NEO4J_PASSWORD_FILE=" in profile


def test_local_profile_check_reports_success() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["local-profile", "--check"])

    assert result.exit_code == 0
    assert '"status": "ok"' in result.output
    assert '"reranker_provider": "lexical"' in result.output


def test_doctor_command_reports_text_summary(tmp_path: Path) -> None:
    runner = CliRunner()
    EventLog(tmp_path / ".eventloom" / "default.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "default"},
        thread="default",
    )

    result = runner.invoke(app, ["doctor", "--eventloom-path", str(tmp_path / ".eventloom")])

    assert result.exit_code == 0
    assert "Zaxy doctor:" in result.output
    assert "eventloom: ok" in result.output
    assert "viewer: ok" in result.output
    assert "captured=1 projected=0 unprojected=1 reinforced=0 eligible=0" in result.output


def test_doctor_command_reports_json(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["doctor", "--eventloom-path", str(tmp_path / ".eventloom"), "--json"],
    )

    assert result.exit_code == 0
    assert '"status": "ok"' in result.output
    assert '"name": "eventloom"' in result.output


def test_doctor_release_smoke_reports_packaging_readiness() -> None:
    """Release smoke mode should verify local release metadata without external services."""
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--release-smoke", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["package_version"]["status"] == "ok"
    assert checks["changelog"]["status"] == "ok"
    assert checks["trusted_publishing"]["status"] == "ok"
    assert checks["release_workflow"]["status"] == "ok"


def test_doctor_beta_readiness_reports_release_and_uat_gates() -> None:
    """Beta readiness should summarize the release, UAT, docs, and capture gates."""
    runner = CliRunner()

    result = runner.invoke(app, ["doctor", "--beta-readiness", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["release_smoke"]["status"] == "ok"
    assert checks["release_gate"]["status"] == "ok"
    assert checks["clean_repo_uat"]["status"] == "ok"
    assert checks["docs_happy_path"]["status"] == "ok"
    assert checks["capture_happy_path"]["status"] == "ok"
    assert "scripts/beta-uat.sh" in checks["clean_repo_uat"]["message"]


def test_doctor_beta_readiness_fails_nonzero_for_unready_project(tmp_path: Path) -> None:
    """Beta readiness should be shell-gatable when a project is missing beta gates."""
    runner = CliRunner()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["doctor", "--beta-readiness", "--project-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "error"
    checks = {check["name"]: check for check in payload["checks"]}
    assert checks["clean_repo_uat"]["status"] == "error"


def test_doctor_release_smoke_uses_explicit_project_root(tmp_path: Path) -> None:
    """Release smoke should support checking a repo root different from cwd."""
    runner = CliRunner()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.2.0 - 2026-05-11\n\n- Stable release.\n",
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "publish.yml").write_text(
        "on:\n"
        "  release:\n"
        "    types: [published]\n"
        "  workflow_dispatch:\n"
        "permissions:\n"
        "  id-token: write\n"
        "steps:\n"
        "  - run: python -m build --sdist --wheel\n"
        "  - run: python -m twine check dist/*\n"
        "  - uses: pypa/gh-action-pypi-publish@release/v1\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["doctor", "--release-smoke", "--project-root", str(tmp_path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"


def test_packet_status_command_reports_text_summary(tmp_path: Path) -> None:
    runner = CliRunner()
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    packet = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "agent-1"},
        thread="agent-1",
    )
    log.append(
        "llm.packet.projected",
        actor="zaxy-packet-projector",
        payload={"source_event_hash": packet.hash, "source_event_seq": packet.seq},
        thread="agent-1",
    )

    result = runner.invoke(
        app,
        ["packet-status", "--eventloom-path", str(tmp_path / ".eventloom"), "--session-id", "agent-1"],
    )

    assert result.exit_code == 0
    assert "Zaxy packet memory: ok" in result.output
    assert "captured=1 projected=1 unprojected=0 reinforced=0 eligible=1" in result.output


def test_packet_status_command_reports_activation_steps_when_inactive(tmp_path: Path) -> None:
    """packet-status should tell operators how to activate capture when none exists."""
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "packet-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--analyzer-port",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "Zaxy packet memory: warning" in result.output
    assert "analyzer: inactive (http://127.0.0.1:1/v1)" in result.output
    assert "Start packet analyzer: zaxy packet-analyzer" in result.output
    assert "Start packet projector: zaxy packet-project" in result.output
    assert "http://127.0.0.1:1/v1" in result.output


def test_packet_status_command_reports_json(tmp_path: Path) -> None:
    runner = CliRunner()
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "agent-1"},
        thread="agent-1",
    )

    result = runner.invoke(
        app,
        [
            "packet-status",
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "agent-1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"status": "warning"' in result.output
    assert '"unprojected": 1' in result.output


@patch("zaxy.__main__.capture_codex_sessions")
def test_codex_capture_command_imports_local_codex_records(mock_capture: MagicMock, tmp_path: Path) -> None:
    """codex-capture should expose deterministic local Codex observation import."""
    mock_capture.return_value.imported = 4
    mock_capture.return_value.scanned_files = 1
    mock_capture.return_value.skipped = 2
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "codex-capture",
            "--workspace",
            str(tmp_path),
            "--codex-home",
            str(tmp_path / "codex-home"),
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "repo-default",
        ],
    )

    assert result.exit_code == 0
    assert "Imported 4 Codex observations from 1 session log" in result.output
    mock_capture.assert_called_once_with(
        workspace=tmp_path,
        codex_home=tmp_path / "codex-home",
        eventloom_path=tmp_path / ".eventloom",
        session_id="repo-default",
        source="codex-local",
        max_records_per_file=1000,
    )


@patch("zaxy.__main__.time.sleep")
@patch("zaxy.__main__.capture_codex_sessions")
def test_codex_capture_watch_mode_supports_bounded_iterations(
    mock_capture: MagicMock,
    mock_sleep: MagicMock,
    tmp_path: Path,
) -> None:
    """codex-capture --watch should support bounded supervisor/test runs."""
    first = MagicMock(imported=2, scanned_files=1, skipped=0)
    second = MagicMock(imported=0, scanned_files=1, skipped=2)
    mock_capture.side_effect = [first, second]
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "codex-capture",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "repo-default",
            "--watch",
            "--watch-iterations",
            "2",
            "--interval-seconds",
            "0.25",
        ],
    )

    assert result.exit_code == 0
    assert "Watching Codex session logs" in result.output
    assert result.output.count("Imported ") == 2
    assert mock_capture.call_count == 2
    mock_sleep.assert_called_once_with(0.25)


@patch("zaxy.__main__.GraphStore")
@patch("zaxy.__main__.capture_codex_sessions")
def test_codex_capture_can_project_captured_events_to_graph(
    mock_capture: MagicMock,
    mock_graph_store: MagicMock,
    tmp_path: Path,
) -> None:
    """codex-capture --graph should project only events captured in that pass."""
    event = EventLog(tmp_path / ".eventloom" / "repo-default.jsonl").append(
        "transcript.turn",
        actor="assistant",
        payload={"content": "Remember bounded capture."},
        thread="repo-default",
    )
    mock_capture.return_value.imported = 1
    mock_capture.return_value.scanned_files = 1
    mock_capture.return_value.skipped = 0
    mock_capture.return_value.events = (event,)
    store = AsyncMock()
    mock_graph_store.return_value = store
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "codex-capture",
            "--workspace",
            str(tmp_path),
            "--eventloom-path",
            str(tmp_path / ".eventloom"),
            "--session-id",
            "repo-default",
            "--graph",
            "--neo4j-uri",
            "bolt://test:7687",
            "--neo4j-password",
            "testpassword",
        ],
    )

    assert result.exit_code == 0
    assert "Projected 1 captured observations into graph" in result.output
    mock_graph_store.assert_called_once_with("bolt://test:7687", "neo4j", "testpassword")
    store.connect.assert_awaited_once()
    store.init_schema.assert_awaited_once()
    store.upsert_extraction.assert_awaited_once()
    assert store.upsert_extraction.await_args.kwargs == {"session_id": "repo-default"}
    store.close.assert_awaited_once()


@patch("zaxy.__main__.MemoryFabric")
def test_index_codebase_command_reports_indexed_count(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """index-codebase should append codebase mapping events through MemoryFabric."""
    fabric = AsyncMock()
    fabric.ingest_codebase.return_value = 3
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["index-codebase", str(tmp_path), "--session-id", "agent-1", "--max-bytes", "1024"],
    )

    assert result.exit_code == 0
    assert "Indexed 3 codebase events into session agent-1" in result.output
    fabric.ingest_codebase.assert_awaited_once_with(tmp_path, session_id="agent-1", max_bytes=1024)
    fabric.close.assert_awaited_once()


@patch("zaxy.__main__.MemoryFabric")
def test_init_session_command_reports_workspace_profile(mock_fabric_cls: MagicMock, tmp_path: Path) -> None:
    """init-session should append a genesis event through MemoryFabric."""
    fabric = AsyncMock()
    fabric.initialize_session.return_value.workspace_type = "codebase"
    fabric.initialize_session.return_value.confidence = 0.8
    mock_fabric_cls.return_value = fabric
    runner = CliRunner()

    result = runner.invoke(app, ["init-session", str(tmp_path), "--session-id", "agent-1"])

    assert result.exit_code == 0
    assert "Initialized agent-1 as codebase workspace (confidence 0.8)" in result.output
    fabric.initialize_session.assert_awaited_once_with(tmp_path, session_id="agent-1")
    fabric.close.assert_awaited_once()


def test_init_command_runs_first_run_onboarding(tmp_path: Path) -> None:
    """init should expose the unified first-run onboarding orchestrator."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(workspace),
            "--domain",
            "demo",
            "--mcp-client",
            "claude-desktop",
            "--mcp-output",
            str(workspace / "mcp.json"),
            "--hook-client",
            "claude-code",
            "--hook-output",
            str(workspace / ".claude" / "settings.local.json"),
            "--local-profile-output",
            str(workspace / ".env.local"),
        ],
    )

    assert result.exit_code == 0
    assert "Zaxy init:" in result.output
    assert "session: demo-default" in result.output
    assert "mcp_config: ok" in result.output
    assert "hook_status:" in result.output
    assert (workspace / "mcp.json").is_file()
    assert (workspace / ".claude" / "settings.local.json").is_file()
    assert (workspace / ".eventloom" / "demo-default.jsonl").is_file()
    local_profile = (workspace / ".env.local").read_text(encoding="utf-8")
    assert "NEO4J_URI=bolt://localhost:7687" in local_profile
    assert "NEO4J_CA_CERT=" in local_profile
    assert "NEO4J_PASSWORD_FILE=" in local_profile
    assert "NEO4J_TRUST_ALL=false" in local_profile


def test_init_command_rejects_mcp_output_without_client(tmp_path: Path) -> None:
    """init should reject renderer output paths without the matching client option."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(workspace), "--mcp-output", str(workspace / "mcp.json")])

    assert result.exit_code != 0
    assert "mcp_client is required" in result.output


@patch("zaxy.__main__.run_onboarding")
def test_init_command_passes_infra_action(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --infra should pass explicit infra action into the orchestrator."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--infra", "check"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["infra"] == "check"


@patch("zaxy.__main__.run_onboarding")
def test_init_command_expands_local_claude_preset(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --preset local-claude should pass expanded explicit options."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-claude"])

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["mcp_client"] == "claude-desktop"
    assert kwargs["mcp_output"] == tmp_path / "zaxy-mcp.json"
    assert kwargs["hook_client"] == "claude-code"
    assert kwargs["hook_output"] == tmp_path / ".claude" / "settings.local.json"
    assert kwargs["local_profile_output"] == tmp_path / ".env.local"
    assert kwargs["infra"] == "check"
    assert kwargs["capture_mode"] == "deterministic"


@patch("zaxy.__main__.run_onboarding")
def test_init_command_expands_local_codex_preset(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --preset local-codex should install safe repo-local capture config."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-codex"])

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["mcp_client"] == "codex"
    assert kwargs["mcp_output"] is None
    assert kwargs["hook_client"] == "codex"
    assert kwargs["hook_output"] == tmp_path / ".codex" / "zaxy-capture.json"
    assert kwargs["local_profile_output"] == tmp_path / ".env.local"
    assert kwargs["infra"] == "check"
    assert kwargs["capture_mode"] == "deterministic"


@patch("zaxy.__main__.run_onboarding")
def test_init_command_passes_packet_capture_options(
    mock_run_onboarding: AsyncMock,
    tmp_path: Path,
) -> None:
    """init --packet-capture should pass packet activation settings to onboarding."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "init",
            str(tmp_path),
            "--packet-capture",
            "--packet-upstream-base-url",
            "https://api.openai.com/v1",
            "--packet-port",
            "8788",
        ],
    )

    assert result.exit_code == 0
    kwargs = mock_run_onboarding.await_args.kwargs
    assert kwargs["packet_capture"] is True
    assert kwargs["capture_mode"] == "hybrid"
    assert kwargs["packet_upstream_base_url"] == "https://api.openai.com/v1"
    assert kwargs["packet_port"] == 8788


@patch("zaxy.__main__.run_onboarding")
def test_init_command_accepts_capture_mode_packet(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --capture-mode packet should explicitly opt into packet-capture guidance."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--capture-mode", "packet"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["capture_mode"] == "packet"


@patch("zaxy.__main__.run_onboarding")
def test_init_command_accepts_capture_start_action(mock_run_onboarding: AsyncMock, tmp_path: Path) -> None:
    """init --capture start should ask onboarding to start deterministic capture."""
    result_obj = MagicMock()
    result_obj.status = "ok"
    mock_run_onboarding.return_value = result_obj
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(tmp_path), "--preset", "local-codex", "--capture", "start"])

    assert result.exit_code == 0
    assert mock_run_onboarding.await_args.kwargs["capture_action"] == "start"


def test_init_command_help_describes_full_onboarding_path() -> None:
    """init help should describe the full golden-path onboarding behavior."""
    runner = CliRunner()

    result = runner.invoke(app, ["init", "--help"])

    assert result.exit_code == 0
    assert "MCP config" in result.output
    assert "infra" in result.output
    assert "hook status" in result.output


def test_init_command_json_includes_next_steps_and_capture_summary(tmp_path: Path) -> None:
    """init --json should expose next_steps and capture state for client UIs and automation."""
    workspace = tmp_path / "repo"
    workspace.mkdir()
    runner = CliRunner()

    result = runner.invoke(app, ["init", str(workspace), "--domain", "demo", "--preset", "local-codex", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["session_id"] == "demo-default"
    assert any(step.startswith("Run zaxy hook-status") for step in payload["next_steps"])
    assert payload["capture"]["configured"] is True
    assert payload["capture"]["running"] is False
    assert payload["capture"]["pids"] == []
    assert payload["capture"]["doctor_status"] == "warning"


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


def test_compact_rewrite_appends_lifecycle_event(tmp_path: Path) -> None:
    """compact rewrite should record a compaction.completed lifecycle event."""
    log_path = tmp_path / "work.jsonl"
    EventLog(log_path).append("goal.created", actor="user", payload={"title": "Ship"})
    runner = CliRunner()

    result = runner.invoke(app, ["compact", str(log_path)])

    assert result.exit_code == 0
    events = EventLog(log_path).read_all()
    assert events[-1].type == "compaction.completed"
    assert events[-1].payload["mode"] == "rewrite"
    assert events[-1].payload["status"] == "succeeded"
    assert events[-1].payload["event_count"] == 1


def test_viewer_command_writes_static_html(tmp_path: Path) -> None:
    """viewer should write a standalone Eventloom inspection page."""
    log_path = tmp_path / "default.jsonl"
    output = tmp_path / "viewer.html"
    EventLog(log_path).append(
        "session.genesis",
        actor="zaxy",
        payload={"session_id": "default", "workspace_type": "codebase"},
        thread="default",
    )
    runner = CliRunner()

    result = runner.invoke(app, ["viewer", str(log_path), "--output", str(output)])

    assert result.exit_code == 0
    assert f"Wrote Eventloom viewer: {output}" in result.output
    assert output.exists()
    assert "Eventloom Session Viewer" in output.read_text(encoding="utf-8")


def test_dashboard_cli_help_exposes_localhost_default() -> None:
    """dashboard should expose the local read-only web app command."""
    runner = CliRunner()
    result = runner.invoke(app, ["dashboard", "--help"])

    assert result.exit_code == 0
    assert "127.0.0.1" in result.output
    assert "8765" in result.output
