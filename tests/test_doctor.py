"""Tests for local environment doctor checks."""

from __future__ import annotations

from pathlib import Path

from zaxy.config import Settings
from zaxy.doctor import run_doctor


def test_run_doctor_reports_local_setup_ok(tmp_path: Path) -> None:
    """Doctor should validate local-only setup without requiring live Neo4j."""
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

    assert report["status"] == "ok"
    assert {check["name"]: check["status"] for check in report["checks"]} == {
        "eventloom": "ok",
        "local_profile": "ok",
        "viewer": "ok",
        "mcp_defaults": "ok",
        "hooks": "ok",
        "neo4j": "ok",
        "production": "ok",
    }
    assert (tmp_path / ".eventloom").is_dir()


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
