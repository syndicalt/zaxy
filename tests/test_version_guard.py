"""Tests for the version-consistency guard (install drift detection).

The guard catches the exact failure that once sent a diagnosis after the wrong
PID: the env had a stale ``zaxy-memory`` in site-packages while the repo had
moved on, so tests imported the wrong code. It compares the version declared in
the repo's ``pyproject.toml`` (read from disk, not the import) against the
imported package's recorded version and import path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import zaxy.release as release
from zaxy.release import check_version_consistency


def _write_repo(tmp_path: Path, version: str) -> Path:
    repo = tmp_path / "zaxy-repo"
    (repo / "src" / "zaxy").mkdir(parents=True)
    (repo / "src" / "zaxy" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "zaxy-memory"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    return repo


def test_guard_ok_when_installed_matches_declared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path, "3.0.1")
    monkeypatch.setattr(release, "installed_package_version", lambda: "3.0.1")
    monkeypatch.setattr(release, "zaxy_import_path", lambda: str(repo / "src" / "zaxy" / "__init__.py"))
    result = check_version_consistency(project_root=repo)
    assert result["status"] == "ok"
    assert result["details"]["declared"] == "3.0.1"
    assert result["details"]["installed"] == "3.0.1"


def test_guard_warns_on_version_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path, "3.0.1")
    monkeypatch.setattr(release, "installed_package_version", lambda: "2.6.3")  # stale
    monkeypatch.setattr(
        release, "zaxy_import_path", lambda: str(repo / "src" / "zaxy" / "__init__.py")
    )
    result = check_version_consistency(project_root=repo)
    assert result["status"] == "warning"
    assert "2.6.3" in result["message"]
    assert "3.0.1" in result["message"]
    assert "pip install -e ." in result["action"]


def test_guard_warns_on_import_path_drift_even_when_versions_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing a different copy than the repo on disk is drift even if the
    version strings happen to coincide (e.g. a stale same-version copy)."""
    repo = _write_repo(tmp_path, "3.0.1")
    monkeypatch.setattr(release, "installed_package_version", lambda: "3.0.1")
    monkeypatch.setattr(
        release,
        "zaxy_import_path",
        lambda: "/home/user/miniconda3/lib/python3.13/site-packages/zaxy/__init__.py",
    )
    result = check_version_consistency(project_root=repo)
    assert result["status"] == "warning"
    assert "outside the repo" in result["message"]


def test_guard_reports_unresolved_installed_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _write_repo(tmp_path, "3.0.1")
    monkeypatch.setattr(release, "installed_package_version", lambda: None)
    monkeypatch.setattr(release, "zaxy_import_path", lambda: str(repo / "src" / "zaxy" / "__init__.py"))
    result = check_version_consistency(project_root=repo)
    assert result["status"] == "warning"
    assert "could not be resolved" in result["message"]


def test_guard_without_repo_reports_posture_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "installed_package_version", lambda: "3.0.1")
    monkeypatch.setattr(release, "zaxy_import_path", lambda: "/some/site-packages/zaxy/__init__.py")
    # project_root is a dir with no zaxy pyproject.
    result = check_version_consistency(project_root=tmp_path)
    assert result["status"] == "ok"
    assert "no zaxy repo found to compare" in result["message"]


def test_installed_package_version_returns_live_metadata() -> None:
    """Sanity: the live resolver returns a parseable version string (or None)."""
    value = release.installed_package_version()
    assert value is None or isinstance(value, str)
