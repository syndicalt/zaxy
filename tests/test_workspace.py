"""Tests for workspace genesis and profile discovery."""

from __future__ import annotations

from pathlib import Path

from zaxy.workspace import (
    build_session_genesis_event,
    build_workspace_instruction_event,
    discover_workspace_profile,
    existing_workspace_instructions_signature,
)


def test_discover_workspace_profile_detects_codebase(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()

    profile = discover_workspace_profile(tmp_path)

    assert profile.workspace_type == "codebase"
    assert profile.instructions_profile == "codebase"
    assert profile.confidence >= 0.8
    assert profile.signals == ["pyproject.toml", "src/", "tests/"]


def test_discover_workspace_profile_falls_back_to_generic(tmp_path: Path) -> None:
    (tmp_path / "notes.bin").write_bytes(b"\x00\x01")

    profile = discover_workspace_profile(tmp_path)

    assert profile.workspace_type == "generic_workspace"
    assert profile.instructions_profile == "generic"
    assert profile.confidence == 0.2
    assert profile.signals == []


def test_build_session_genesis_event_contains_profile_and_write_instructions(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest"}}\n', encoding="utf-8")
    (tmp_path / "src").mkdir()

    event = build_session_genesis_event(tmp_path, session_id="demo-default")

    assert event == {
        "event_type": "session.genesis",
        "actor": "zaxy",
        "payload": {
            "root": str(tmp_path.resolve()),
            "workspace_type": "codebase",
            "confidence": 0.7,
            "signals": ["package.json", "src/"],
            "instructions_profile": "codebase",
            "session_id": "demo-default",
            "write_instructions": {
                "preferred_events": [
                    "decision.made",
                    "task.completed",
                    "code.file.indexed",
                    "code.symbol.indexed",
                    "code.import.indexed",
                    "code.dependency.indexed",
                    "code.call.indexed",
                    "code.coverage.indexed",
                ],
                "avoid_writing": ["raw_secrets", "full_source_bodies", "transient_chatter"],
                "indexing_strategy": "metadata_only_codebase_map",
            },
        },
    }


def test_build_workspace_instruction_event_summarizes_instruction_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(
        "# Project Rules\n\nUse tests first.\nDo not write secrets.\n",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text(
        "# Claude Notes\n\nPrefer concise answers.\n",
        encoding="utf-8",
    )

    event = build_workspace_instruction_event(tmp_path, session_id="demo-default")

    assert event is not None
    assert event["event_type"] == "workspace.instructions.discovered"
    payload = event["payload"]
    assert payload["root"] == str(tmp_path.resolve())
    assert payload["session_id"] == "demo-default"
    assert payload["summary"] == "Project Rules: Use tests first. Claude Notes: Prefer concise answers."
    assert payload["signature"]
    assert [item["path"] for item in payload["files"]] == ["AGENTS.md", "CLAUDE.md"]
    assert payload["files"][0]["kind"] == "agents"
    assert payload["files"][0]["citation"] == f"{tmp_path / 'AGENTS.md'}:1-4"
    assert "content" not in payload["files"][0]


def test_build_workspace_instruction_event_returns_none_without_instruction_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Notes\n", encoding="utf-8")

    assert build_workspace_instruction_event(tmp_path, session_id="demo-default") is None


def test_existing_workspace_instructions_signature_detects_matching_event(tmp_path: Path) -> None:
    event = build_workspace_instruction_event(tmp_path, session_id="demo-default")
    assert event is None
    (tmp_path / "SOUL.md").write_text("# Voice\n\nBe direct.\n", encoding="utf-8")
    event = build_workspace_instruction_event(tmp_path, session_id="demo-default")
    assert event is not None
    existing = type(
        "Event",
        (),
        {"type": "workspace.instructions.discovered", "payload": event["payload"]},
    )()

    assert existing_workspace_instructions_signature(
        [existing],
        root=tmp_path,
        session_id="demo-default",
    ) == event["payload"]["signature"]


def test_existing_workspace_instructions_signature_uses_latest_update(tmp_path: Path) -> None:
    discovered = type(
        "Event",
        (),
        {
            "type": "workspace.instructions.discovered",
            "payload": {
                "root": str(tmp_path.resolve()),
                "session_id": "demo-default",
                "signature": "old",
            },
        },
    )()
    updated = type(
        "Event",
        (),
        {
            "type": "workspace.instructions.updated",
            "payload": {
                "root": str(tmp_path.resolve()),
                "session_id": "demo-default",
                "signature": "new",
            },
        },
    )()

    assert existing_workspace_instructions_signature(
        [discovered, updated],
        root=tmp_path,
        session_id="demo-default",
    ) == "new"
