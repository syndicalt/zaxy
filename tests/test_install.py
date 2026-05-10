"""Tests for install-path helpers."""

from __future__ import annotations

from pathlib import Path

from zaxy.install import resolve_zaxy_executable


def test_resolve_zaxy_executable_uses_explicit_path() -> None:
    assert resolve_zaxy_executable("/opt/zaxy/bin/zaxy") == "/opt/zaxy/bin/zaxy"


def test_resolve_zaxy_executable_prefers_console_script(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("zaxy.install.shutil.which", lambda command: "/home/user/.local/bin/zaxy" if command == "zaxy" else None)

    assert resolve_zaxy_executable() == "/home/user/.local/bin/zaxy"


def test_resolve_zaxy_executable_falls_back_to_argv(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    script = tmp_path / "python"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr("zaxy.install.shutil.which", lambda _command: None)
    monkeypatch.setattr("zaxy.install.sys.argv", [str(script)])

    assert resolve_zaxy_executable() == str(script.resolve())
