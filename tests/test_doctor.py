"""Tests for local environment doctor checks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from zaxy.config import Settings
from zaxy.doctor import run_doctor
from zaxy.event import EventLog


def test_run_doctor_reports_local_setup_ok(tmp_path: Path, monkeypatch) -> None:
    """Doctor should validate local-only setup with the embedded projection default."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    # Version drift is environment-dependent (installed vs declared); pin it so the
    # exact status map is deterministic regardless of the local install state.
    monkeypatch.setattr(
        "zaxy.release.check_version_consistency",
        lambda **_: {
            "name": "version_consistency",
            "status": "ok",
            "message": "imported zaxy matches the repository",
        },
    )
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        embedded_graph_path=str(tmp_path / ".eventloom" / "projections" / "embedded.kuzu"),
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
        "version_consistency": "ok",
        "eventloom": "ok",
        "event_chain": "ok",
        "local_profile": "ok",
        "embedding": "ok",
        "vector_cache": "ok",
        "embedding_versions": "ok",
        "viewer": "ok",
        "cli_install": "ok",
        "mcp_defaults": "ok",
        "codex_mcp_scope": "ok",
        "agent_instructions": "warning",
        "hooks": "ok",
        "hook_installation": "warning",
        "hook_activity": "warning",
        "observation_coverage": "warning",
        "capture_health": "warning",
        "memory_activation": "warning",
        "packet_memory": "warning",
        "embedded_mcp_runtime": "ok",
        "embedded_graph": "ok",
        "projection_freshness": "ok",
        "projection_backup_artifacts": "ok",
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
    assert "Review" in check["action"]
    assert str(codex_home / "config.toml") in check["action"]
    assert f"zaxy init {tmp_path / 'zaxy'} --codex-mcp-install user --force" in check["action"]
    assert "codex mcp add zaxy -- zaxy serve" not in check["action"]


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


def test_run_doctor_warns_when_agent_activation_instructions_missing(tmp_path: Path) -> None:
    """Doctor should detect when the model-visible activation fallback is absent."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    (tmp_path / "AGENTS.md").write_text("# Existing Rules\n\nUse tests.\n", encoding="utf-8")

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "agent_instructions")
    assert check["status"] == "warning"
    assert check["message"] == "AGENTS.md is present but missing the Zaxy Memory Activation block"
    assert "zaxy init" in check["action"]


def test_run_doctor_reports_agent_activation_instructions_ok(tmp_path: Path) -> None:
    """Doctor should pass the model-visible activation fallback installed by init."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    (tmp_path / "AGENTS.md").write_text(
        "\n".join(
            [
                "# Existing Rules",
                "",
                "<!-- zaxy-memory-activation:start -->",
                "## Zaxy Memory Activation",
                "<!-- zaxy-memory-activation:end -->",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "agent_instructions")
    assert check["status"] == "ok"
    assert check["message"] == "AGENTS.md contains marker-managed Zaxy Memory Activation instructions"


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


@patch("zaxy.hooks._iter_process_cmdlines")
def test_run_doctor_hard_warns_when_codex_capture_configured_but_not_running(
    mock_processes: MagicMock,
    tmp_path: Path,
) -> None:
    """Doctor should make configured-but-stopped Codex capture impossible to miss."""
    settings = Settings(
        _env_file=None,
        eventloom_path=str(tmp_path / ".eventloom"),
        eventloom_thread="zaxy-default",
        zaxy_env="development",
    )
    capture_config = tmp_path / ".codex" / "zaxy-capture.json"
    capture_config.parent.mkdir()
    capture_config.write_text(
        json.dumps(
            {
                "capture": "local-session-jsonl",
                "client": "codex",
                "workspace": str(tmp_path),
                "eventloom_path": str(tmp_path / ".eventloom"),
                "session_id": "zaxy-default",
            }
        ),
        encoding="utf-8",
    )
    mock_processes.return_value = []

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "capture_health")
    assert check["status"] == "warning"
    assert check["message"] == "Codex capture is configured, but the managed watcher is not running"
    assert f"zaxy capture start --workspace {tmp_path}" in check["action"]


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
    assert check["message"] == "Codex capture is configured, but the managed watcher is not running"
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


def _local_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "eventloom_path": str(tmp_path / ".eventloom"),
        "eventloom_thread": "zaxy-default",
        "zaxy_env": "development",
        "embedding_enabled": True,
        "embedding_provider": "hash",
        "reranker_provider": "lexical",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_run_doctor_verifies_event_chain_over_active_log(tmp_path: Path) -> None:
    """Doctor should fully verify the hash chain of a small active log."""
    settings = _local_settings(tmp_path)
    log = EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl")
    log.append("task.created", actor="user", payload={"taskId": "task-1", "title": "Ship it"}, thread="zaxy-default")
    log.append("task.completed", actor="agent", payload={"taskId": "task-1"}, thread="zaxy-default")

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "event_chain")
    assert check["status"] == "ok"
    assert "hash chain verified over 2 events (full)" in check["message"]


def test_run_doctor_reports_broken_event_chain_with_remediation(tmp_path: Path) -> None:
    """Doctor should flag in-place log tampering and print a one-line remediation."""
    settings = _local_settings(tmp_path)
    log_path = tmp_path / ".eventloom" / "zaxy-default.jsonl"
    log = EventLog(log_path)
    log.append("task.created", actor="user", payload={"taskId": "task-1", "title": "Ship it"}, thread="zaxy-default")
    log.append("task.completed", actor="agent", payload={"taskId": "task-1"}, thread="zaxy-default")
    log_path.write_text(log_path.read_text(encoding="utf-8").replace("Ship it", "Sink it"), encoding="utf-8")

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "event_chain")
    assert report["status"] == "error"
    assert check["status"] == "error"
    assert "hash chain broken" in check["message"]
    assert check["action"]


def test_run_doctor_reports_embedding_provider_unavailable(tmp_path: Path) -> None:
    """Doctor should turn a misconfigured embedding provider into an actionable error."""
    settings = _local_settings(tmp_path, embedding_provider="openai", openai_api_key=None)

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "embedding")
    assert check["status"] == "error"
    assert "embedding provider openai is unavailable" in check["message"]
    assert check["action"]


@patch("zaxy.doctor.build_embedding_provider")
def test_run_doctor_reports_embedding_dimension_disagreement(
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    """Doctor should refuse silent dimension drift between settings and provider."""
    provider = MagicMock()
    provider.dimension = 64
    mock_build.return_value = provider
    settings = _local_settings(tmp_path)

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "embedding")
    assert check["status"] == "error"
    assert "dimension 64" in check["message"]
    assert "EMBEDDING_DIMENSION=1536" in check["message"]
    assert check["action"]


def test_run_doctor_reports_embedding_provider_ok(tmp_path: Path) -> None:
    """Doctor should confirm the offline hash provider and its probed dimension."""
    settings = _local_settings(tmp_path)

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "embedding")
    assert check["status"] == "ok"
    assert "hash embeddings available at dimension 1536" in check["message"]


def test_run_doctor_warns_on_vector_cache_budget_headroom(tmp_path: Path) -> None:
    """Doctor should flag embedding dimensions that exhaust the vector cache budget."""
    settings = _local_settings(tmp_path, embedding_dimension=8_000_000)

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "vector_cache")
    assert check["status"] == "warning"
    assert "fits only" in check["message"]
    assert check["details"]["budget_vector_capacity"] < 1024
    assert check["action"]


def test_run_doctor_reports_vector_cache_budget_ok(tmp_path: Path) -> None:
    """Doctor should report healthy vector cache headroom at the default dimension."""
    settings = _local_settings(tmp_path)

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "vector_cache")
    assert check["status"] == "ok"
    assert "vector index cache budget holds about" in check["message"]
    assert check["details"]["budget_vector_capacity"] >= 1024


def test_run_doctor_warns_when_projection_is_missing_for_active_log(tmp_path: Path) -> None:
    """Doctor should flag an event log with no embedded projection state."""
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(tmp_path / ".eventloom" / "projections" / "embedded.kuzu"),
    )
    EventLog(tmp_path / ".eventloom" / "zaxy-default.jsonl").append(
        "task.created",
        actor="user",
        payload={"taskId": "task-1", "title": "Ship it"},
        thread="zaxy-default",
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "projection_freshness")
    assert check["status"] == "warning"
    assert "has no state for the active event log" in check["message"]
    assert "zaxy memory checkout" in check["action"]


def test_run_doctor_warns_when_projection_is_older_than_log(tmp_path: Path) -> None:
    """Doctor should flag projection state that predates the latest log writes."""
    import os

    projection_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    projection_path.mkdir(parents=True)
    (projection_path / "data.kz").write_text("projection", encoding="utf-8")
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(projection_path),
    )
    log_path = tmp_path / ".eventloom" / "zaxy-default.jsonl"
    EventLog(log_path).append(
        "task.created",
        actor="user",
        payload={"taskId": "task-1", "title": "Ship it"},
        thread="zaxy-default",
    )
    stale = log_path.stat().st_mtime - 60
    for path in (projection_path, projection_path / "data.kz"):
        os.utime(path, (stale, stale))

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "projection_freshness")
    assert check["status"] == "warning"
    assert "older than the active event log" in check["message"]
    assert "zaxy memory checkout" in check["action"]


def test_run_doctor_reports_projection_freshness_ok(tmp_path: Path) -> None:
    """Doctor should pass projection state that is at least as new as the log."""
    import os

    projection_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    projection_path.mkdir(parents=True)
    (projection_path / "data.kz").write_text("projection", encoding="utf-8")
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(projection_path),
    )
    log_path = tmp_path / ".eventloom" / "zaxy-default.jsonl"
    EventLog(log_path).append(
        "task.created",
        actor="user",
        payload={"taskId": "task-1", "title": "Ship it"},
        thread="zaxy-default",
    )
    fresh = log_path.stat().st_mtime + 60
    os.utime(projection_path / "data.kz", (fresh, fresh))

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "projection_freshness")
    assert check["status"] == "ok"
    assert "at least as new as the active event log" in check["message"]


def test_run_doctor_reports_leftover_pre_ladybug_backup(tmp_path: Path) -> None:
    """Doctor should flag a leftover pre-LadybugDB projection backup with remediation."""
    projection_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    projection_path.parent.mkdir(parents=True)
    backup = projection_path.with_name(projection_path.name + ".pre-ladybug.bak")
    backup.write_bytes(b"pre-fork projection bytes")
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(projection_path),
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "projection_backup_artifacts")
    assert check["status"] == "warning"
    assert str(backup) in check["message"]
    assert "safe" not in check["message"]  # remediation lives in the action line
    assert "delete the .pre-ladybug.bak" in check["action"]
    assert "zaxy reproject" in check["action"]


def _seed_embedded_projection_vectors(
    projection_path: Path,
    *,
    session_id: str,
    embedding_versions: list[str | None],
) -> None:
    """Project one embedded entity per requested embedding version tag."""
    import asyncio

    from zaxy.embedded_graph_store import EmbeddedGraphStore
    from zaxy.extract import ExtractedEntity, ExtractionResult

    async def _seed() -> None:
        store = EmbeddedGraphStore(projection_path)
        await store.connect()
        await store.init_schema()
        for position, version in enumerate(embedding_versions):
            properties = {"embedding_version": version} if version else None
            await store.upsert_extraction(
                ExtractionResult(
                    entities=[
                        ExtractedEntity(
                            name=f"Vector Entity {position}",
                            entity_type="goal",
                            observed_at="2026-06-10T00:00:00Z",
                            embedding=[1.0, 0.0],
                            properties=properties,
                        )
                    ],
                    edges=[],
                    source_event_seq=position + 1,
                    source_event_hash=f"hash-{position + 1}",
                ),
                session_id=session_id,
            )
        await store.close()

    asyncio.run(_seed())


def test_run_doctor_warns_on_mixed_embedding_versions(tmp_path: Path) -> None:
    """Doctor should flag stale-version vectors and point at the re-embed command."""
    from zaxy.embedding import hash_embedding_version_tag

    projection_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    active_tag = hash_embedding_version_tag(1536)
    _seed_embedded_projection_vectors(
        projection_path,
        session_id="zaxy-default",
        embedding_versions=[active_tag, None, "openai:text-embedding-3-small@1.0.0-dim2"],
    )
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(projection_path),
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "embedding_versions")
    assert check["status"] == "warning"
    assert active_tag in check["message"]
    assert "zaxy memory re-embed --session-id zaxy-default" in check["action"]
    assert check["details"]["versions"]["zaxy-default"]["legacy"] == 1
    assert check["details"]["versions"]["zaxy-default"][active_tag] == 1


def test_run_doctor_reports_single_embedding_version_ok(tmp_path: Path) -> None:
    """Doctor should pass corpora where every vector carries the active version."""
    from zaxy.embedding import hash_embedding_version_tag

    projection_path = tmp_path / ".eventloom" / "projections" / "embedded.kuzu"
    active_tag = hash_embedding_version_tag(1536)
    _seed_embedded_projection_vectors(
        projection_path,
        session_id="zaxy-default",
        embedding_versions=[active_tag, active_tag],
    )
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(projection_path),
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "embedding_versions")
    assert check["status"] == "ok"
    assert f"active embedding version {active_tag}" in check["message"]


def test_run_doctor_skips_embedding_versions_without_projection(tmp_path: Path) -> None:
    """Doctor should not warn when no embedded projection exists yet."""
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(tmp_path / ".eventloom" / "projections" / "embedded.kuzu"),
    )

    report = run_doctor(settings=settings, workspace_root=tmp_path)

    check = next(check for check in report["checks"] if check["name"] == "embedding_versions")
    assert check["status"] == "ok"
    assert "no embedded projection yet" in check["message"]


def test_doctor_version_check_uses_workspace_root(tmp_path: Path, monkeypatch) -> None:
    """The version-consistency check must inspect the doctor's workspace root.

    Regression: run_doctor previously called check_version_consistency() with no
    argument, so the repo walk started from the process cwd instead of the tree
    the operator asked about — reporting against the wrong directory.
    """
    captured: dict[str, object] = {}

    def _capture(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"name": "version_consistency", "status": "ok", "message": "pinned"}

    monkeypatch.setattr("zaxy.release.check_version_consistency", _capture)
    settings = _local_settings(
        tmp_path,
        projection_backend="embedded",
        embedded_graph_path=str(tmp_path / ".eventloom" / "projections" / "embedded.kuzu"),
    )

    run_doctor(settings=settings, workspace_root=tmp_path)

    assert captured.get("project_root") == tmp_path


def test_projection_store_size_check_warns_on_bloat(tmp_path: Path) -> None:
    """Doctor flags a store orders of magnitude larger than its source logs."""
    from zaxy.doctor import _check_projection_store_size

    eventloom = tmp_path / ".eventloom"
    (eventloom / "projections").mkdir(parents=True)
    (eventloom / "zaxy-default.jsonl").write_bytes(b"x" * 100)
    store = eventloom / "projections" / "embedded.kuzu"
    store.write_bytes(b"y" * 200_000)
    settings = Settings(
        _env_file=None,
        eventloom_path=str(eventloom),
        embedded_graph_path=str(store),
        embedded_store_bloat_min_bytes=1024,
        embedded_store_bloat_log_multiplier=10.0,
    )

    check = _check_projection_store_size(settings)

    assert check["status"] == "warning"
    assert "bloated" in check["message"] or "over" in check["message"]
    assert check["details"]["store_bytes"] == 200_000
    assert check["details"]["log_bytes"] == 100
    assert "self-heal" in check["action"]


def test_projection_store_size_check_ok_when_healthy_or_missing(tmp_path: Path) -> None:
    """A healthy-ratio store and a missing store both report ok."""
    from zaxy.doctor import _check_projection_store_size

    eventloom = tmp_path / ".eventloom"
    (eventloom / "projections").mkdir(parents=True)
    (eventloom / "zaxy-default.jsonl").write_bytes(b"x" * 100_000)
    store = eventloom / "projections" / "embedded.kuzu"
    store.write_bytes(b"y" * 50_000)  # smaller than its logs
    settings = Settings(
        _env_file=None,
        eventloom_path=str(eventloom),
        embedded_graph_path=str(store),
        embedded_store_bloat_min_bytes=1024,
        embedded_store_bloat_log_multiplier=10.0,
    )
    assert _check_projection_store_size(settings)["status"] == "ok"

    missing = Settings(
        _env_file=None,
        eventloom_path=str(eventloom),
        embedded_graph_path=str(eventloom / "projections" / "absent.kuzu"),
    )
    check = _check_projection_store_size(missing)
    assert check["status"] == "ok"
    assert "no embedded projection store" in check["message"]
