"""Tests for local environment doctor checks."""

from __future__ import annotations

from pathlib import Path

from zaxy.config import Settings
from zaxy.doctor import run_doctor
from zaxy.event import EventLog


def test_run_doctor_reports_local_setup_ok(tmp_path: Path, monkeypatch) -> None:
    """Doctor should validate local-only setup with the embedded projection default."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        neo4j_uri="bolt://localhost:7687",
        zaxy_env="development",
        embedding_enabled=True,
        embedding_provider="hash",
        reranker_provider="lexical",
        mcp_lifecycle_capture_enabled=True,
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    assert report["status"] == "warning"
    assert {check["name"]: check["status"] for check in report["checks"]} == {
        "eventloom": "ok",
        "local_profile": "ok",
        "viewer": "ok",
        "cli_install": "ok",
        "mcp_defaults": "ok",
        "codex_mcp_scope": "ok",
        "hooks": "ok",
        "hook_installation": "warning",
        "hook_activity": "warning",
        "observation_coverage": "warning",
        "capture_health": "warning",
        "memory_activation": "warning",
        "packet_memory": "warning",
        "embedded_mcp_runtime": "ok",
        "embedded_graph": "ok",
        "production": "ok",
    }
    assert (tmp_path / ".eventloom").is_dir()


def test_run_doctor_reports_embedded_projection_without_neo4j(tmp_path: Path) -> None:
    """Doctor should not report Neo4j posture when the embedded backend is selected."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        projection_backend="embedded",
        embedded_graph_path=str(tmp_path / ".eventloom" / "projections" / "embedded.kuzu"),
        zaxy_env="development",
        embedding_enabled=True,
        embedding_provider="hash",
        reranker_provider="lexical",
        mcp_lifecycle_capture_enabled=True,
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert "neo4j" not in checks
    assert checks["embedded_graph"]["status"] == "ok"
    assert "Embedded graph projection" in checks["embedded_graph"]["message"]


def test_run_doctor_reports_resolved_cli_executable(tmp_path: Path) -> None:
    """Doctor should show the executable path MCP clients should call."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path, zaxy_executable="/opt/zaxy/bin/zaxy")

    check = next(check for check in report["checks"] if check["name"] == "cli_install")
    assert check["status"] == "ok"
    assert "/opt/zaxy/bin/zaxy" in check["message"]


def test_run_doctor_warns_on_generic_default_session(tmp_path: Path) -> None:
    """Doctor should flag default session bleed risk as actionable warning."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="default",
        zaxy_env="development",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "mcp_defaults")
    assert report["status"] == "warning"
    assert check["status"] == "warning"
    assert "EVENTLOOM_THREAD" in check["message"]


def test_run_doctor_warns_on_user_codex_config_with_repo_scoped_zaxy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Doctor should flag the global Codex MCP leak mode before sessions drift."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        "\n".join(
            [
                "[mcp_servers.zaxy]",
                'command = "zaxy"',
                'args = ["serve", "--eventloom-path", "/repos/zaxyhub/.eventloom"]',
                "",
                "[mcp_servers.zaxy.env]",
                'EVENTLOOM_PATH = "/repos/zaxyhub/.eventloom"',
                'EVENTLOOM_THREAD = "zaxyhub-default"',
                'ZAXY_DOMAIN = "zaxyhub"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path / "zaxy")

    check = next(check for check in report["checks"] if check["name"] == "codex_mcp_scope")
    assert check["status"] == "warning"
    assert "repo-specific Eventloom" in check["message"]
    assert "codex mcp add zaxy -- zaxy serve" in check["action"]


def test_run_doctor_reports_hook_adapter_guidance(tmp_path: Path) -> None:
    """Doctor should surface observer hook setup as an onboarding step."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "hooks")
    assert check["status"] == "ok"
    assert "observer hook adapters" in check["message"]
    assert "zaxy hooks claude-code" in check["action"]
    assert "--output .claude/settings.local.json" in check["action"]


def test_run_doctor_reports_hook_installation_ok_for_claude_settings(tmp_path: Path) -> None:
    """Doctor should detect installed Claude hook config in the workspace."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir()
    settings_path.write_text('{"hooks": {"Stop": [{"hooks": [{"command": "zaxy hook-event stop"}]}]}}', encoding="utf-8")

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "hook_installation")
    assert check["status"] == "ok"
    assert ".claude/settings.local.json" in check["message"]


def test_run_doctor_ignores_non_json_codex_hook_file(tmp_path: Path) -> None:
    """Doctor should not treat a shell snippet as installed Codex hooks.json."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    settings_path = tmp_path / ".codex" / "hooks.json"
    settings_path.parent.mkdir()
    settings_path.write_text(
        "# Zaxy observer hook commands\nzaxy hook-event stop --eventloom-path .eventloom\n",
        encoding="utf-8",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "hook_installation")
    assert check["status"] == "warning"
    assert "No installed observer hook config" in check["message"]


def test_run_doctor_warns_when_hook_config_not_installed(tmp_path: Path) -> None:
    """Doctor should make missing hook installation actionable."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "hook_installation")
    assert check["status"] == "warning"
    assert "No installed observer hook config" in check["message"]
    assert "zaxy hooks claude-code" in check["action"]


def test_run_doctor_warns_when_hooks_installed_but_no_activity(tmp_path: Path) -> None:
    """Doctor should surface installed-but-silent observer hooks."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    settings_path = tmp_path / ".claude" / "settings.local.json"
    settings_path.parent.mkdir()
    settings_path.write_text('{"hooks": {"Stop": [{"hooks": [{"command": "zaxy hook-event stop"}]}]}}', encoding="utf-8")

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "hook_activity")
    assert check["status"] == "warning"
    assert "No hook lifecycle events observed" in check["message"]
    assert "zaxy hook-event heartbeat" in check["action"]


def test_run_doctor_reports_recent_hook_activity(tmp_path: Path) -> None:
    """Doctor should show the latest observed hook lifecycle event."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"trigger": "heartbeat", "source": "claude-code"},
        thread="zaxy-default",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "hook_activity")
    assert check["status"] == "ok"
    assert "hook.heartbeat" in check["message"]
    assert "zaxy-default" in check["message"]


def test_run_doctor_warns_when_high_value_observation_types_are_missing(tmp_path: Path) -> None:
    """Doctor should surface missing automatic capture coverage."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl").append(
        "hook.heartbeat",
        actor="zaxy-hook",
        payload={"trigger": "heartbeat", "source": "codex"},
        thread="zaxy-default",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "observation_coverage")
    assert check["status"] == "warning"
    assert "command.completed" in check["message"]
    assert "file.edit.applied" in check["message"]
    assert check["details"]["active_observation_types"] == []
    assert check["details"]["missing_observation_types"] == [
        "command.completed",
        "file.edit.applied",
        "tool.call.completed",
        "transcript.turn",
    ]
    assert check["details"]["actions"] == [
        "Wire hooks or adapter sinks for: command.completed, file.edit.applied, tool.call.completed, transcript.turn.",
    ]


def test_run_doctor_reports_capture_health_when_all_lanes_are_active(tmp_path: Path) -> None:
    """Doctor should summarize whether automatic capture is actually producing memory events."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    log = EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl")
    for event_type in ("command.completed", "file.edit.applied", "tool.call.completed", "transcript.turn"):
        log.append(event_type, actor="zaxy-capture", payload={"source": "codex-local"}, thread="zaxy-default")

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "capture_health")
    assert check["status"] == "ok"
    assert check["message"] == "automatic capture is healthy: 4 of 4 high-value lanes are active"
    assert check["details"]["active_observation_types"] == [
        "command.completed",
        "file.edit.applied",
        "tool.call.completed",
        "transcript.turn",
    ]
    assert check["details"]["missing_observation_types"] == []


def test_run_doctor_reports_memory_activation_remediation(tmp_path: Path) -> None:
    """Doctor should make missing checkout use actionable, not only hook capture."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="agent-1",
        zaxy_env="development",
    )
    EventLog(tmp_path / ".eventloom" / "agent-1.jsonl").append(
        "command.completed",
        actor="zaxy-observer",
        payload={"source": "codex", "command": "pytest"},
        thread="agent-1",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "memory_activation")
    assert check["status"] == "warning"
    assert check["message"] == "No memory checkout events found"
    assert check["details"]["latest_capture"]["type"] == "command.completed"
    assert check["action"] == (
        "zaxy memory checkout 'current project memory and next useful action' "
        f"--eventloom-path {tmp_path / '.eventloom'} --session-id agent-1"
    )


def test_run_doctor_reports_capture_health_with_managed_codex_action(tmp_path: Path) -> None:
    """Doctor should tell Codex users to start the managed watcher when capture is configured but idle."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    config = tmp_path / ".codex" / "zaxy-capture.json"
    config.parent.mkdir()
    config.write_text(
        """{
  "capture": "local-session-jsonl",
  "client": "codex",
  "codex_home": ".codex-home",
  "eventloom_path": ".eventloom",
  "session_id": "zaxy-default",
  "source": "codex-local",
  "workspace": "."
}
""",
        encoding="utf-8",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "capture_health")
    assert check["status"] == "warning"
    assert check["message"] == "automatic capture is incomplete: 0 of 4 high-value lanes are active"
    assert f"zaxy capture start --workspace {tmp_path}" in check["action"]
    assert check["details"]["actions"] == [
        "Wire hooks or adapter sinks for: command.completed, file.edit.applied, tool.call.completed, transcript.turn.",
        f"Start managed deterministic Codex capture: zaxy capture start --workspace {tmp_path}.",
    ]


def test_run_doctor_warns_when_packets_are_not_projected(tmp_path: Path) -> None:
    """Doctor should surface captured LLM packets that are not memory-ready yet."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl").append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "zaxy-default", "provider_path": "/v1/responses"},
        thread="zaxy-default",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "packet_memory")
    assert check["status"] == "warning"
    assert "1 captured packet event has not been projected" in check["message"]
    assert check["details"] == {
        "captured": 1,
        "projected": 0,
        "unprojected": 1,
        "reinforced": 0,
        "eligible": 0,
    }
    assert "zaxy packet-project --watch" in check["action"]


def test_run_doctor_reports_packet_memory_ok_when_projected(tmp_path: Path) -> None:
    """Doctor should recognize packet capture that has reached memory projection."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    log = EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl")
    packet = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        payload={"session_id": "zaxy-default", "provider_path": "/v1/responses"},
        thread="zaxy-default",
    )
    log.append(
        "llm.packet.projected",
        actor="zaxy-packet-projector",
        payload={"source_event_hash": packet.hash, "source_event_seq": packet.seq},
        thread="zaxy-default",
    )
    log.append(
        "memory.reinforced",
        actor="assistant",
        payload={"entity_type": "packet_memory", "source_event_hash": packet.hash},
        thread="zaxy-default",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "packet_memory")
    assert check["status"] == "ok"
    assert "1 packet capture has projected memory" in check["message"]
    assert check["details"] == {
        "captured": 1,
        "projected": 1,
        "unprojected": 0,
        "reinforced": 1,
        "eligible": 1,
    }


def test_packet_status_reports_analyzer_probe_state(tmp_path: Path) -> None:
    """Packet status should expose whether the analyzer listener is reachable."""
    from zaxy.doctor import packet_memory_report

    report = packet_memory_report(
        eventloom_path=tmp_path / ".eventloom",
        session_id="zaxy-default",
        analyzer_host="127.0.0.1",
        analyzer_port=1,
    )

    assert report["capture"] == {
        "analyzer_host": "127.0.0.1",
        "analyzer_port": 1,
        "analyzer_listening": False,
        "client_base_url": "http://127.0.0.1:1/v1",
    }
