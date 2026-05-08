"""Tests for project domain/session helpers."""

from __future__ import annotations

from pathlib import Path

from zaxy.domain import derive_domain, domain_default_session, slug_domain


def test_slug_domain_produces_safe_session_component() -> None:
    assert slug_domain("Cheap Seats Econ / Zaxy!") == "cheap-seats-econ-zaxy"


def test_derive_domain_uses_project_directory_name(tmp_path: Path) -> None:
    project = tmp_path / "My Project"
    project.mkdir()

    assert derive_domain(project) == "my-project"


def test_domain_default_session_uses_domain_prefix() -> None:
    assert domain_default_session("zaxy") == "zaxy-default"
