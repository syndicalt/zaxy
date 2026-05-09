"""Tests for workspace genesis and profile discovery."""

from __future__ import annotations

from pathlib import Path

from zaxy.workspace import build_session_genesis_event, discover_workspace_profile


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
