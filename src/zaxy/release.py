"""Release metadata and local release-readiness checks."""

from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGE_NAME = "zaxy-memory"


def package_version(*, project_root: Path | None = None) -> str:
    """Return the installed package version, falling back to local pyproject metadata."""
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        if project_root is not None:
            return pyproject_version(project_root)
        root = _source_project_root()
        if root is None:
            return "0+unknown"
        return pyproject_version(root)


def pyproject_version(project_root: Path) -> str:
    """Read the Zaxy package version from pyproject.toml."""
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml project.version must be a non-empty string")
    return version


def _source_project_root() -> Path | None:
    """Find the package source-tree root without depending on the caller cwd."""
    for candidate in Path(__file__).resolve().parents:
        pyproject_path = candidate / "pyproject.toml"
        if not pyproject_path.is_file():
            continue
        try:
            pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        project = pyproject.get("project")
        if isinstance(project, dict) and project.get("name") == PACKAGE_NAME:
            return candidate
    return None


def run_release_smoke(*, project_root: str | Path | None = None) -> dict[str, Any]:
    """Check local release metadata without network calls or external services."""
    root = Path(project_root or Path.cwd())
    checks = [
        _check_package_version(root),
        _check_changelog(root),
        _check_trusted_publishing(root),
        _check_release_workflow(root),
    ]
    return {
        "status": _overall_status(checks),
        "checks": checks,
    }


def _check_package_version(root: Path) -> dict[str, str]:
    try:
        version = pyproject_version(root)
    except Exception as exc:
        return {
            "name": "package_version",
            "status": "error",
            "message": f"could not read pyproject package version: {exc}",
            "action": "Set [project].version in pyproject.toml before publishing.",
        }
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", version):
        return {
            "name": "package_version",
            "status": "error",
            "message": f"package version {version!r} is not a publishable PEP 440 release version",
            "action": "Use a version such as 0.2.0 or 0.2.0b1.",
        }
    return {
        "name": "package_version",
        "status": "ok",
        "message": f"{PACKAGE_NAME} version is {version}",
    }


def _check_changelog(root: Path) -> dict[str, str]:
    try:
        version = pyproject_version(root)
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "changelog",
            "status": "error",
            "message": f"CHANGELOG.md is missing or unreadable: {exc}",
            "action": "Add a CHANGELOG.md entry for the release version.",
        }
    heading = f"## {version} - "
    if heading not in changelog:
        return {
            "name": "changelog",
            "status": "error",
            "message": f"CHANGELOG.md has no entry for {version}",
            "action": f"Add a '## {version} - YYYY-MM-DD' section before publishing.",
        }
    return {
        "name": "changelog",
        "status": "ok",
        "message": f"CHANGELOG.md includes {version}",
    }


def _check_trusted_publishing(root: Path) -> dict[str, str]:
    try:
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "trusted_publishing",
            "status": "error",
            "message": f"publish workflow is missing or unreadable: {exc}",
            "action": "Create .github/workflows/publish.yml.",
        }
    has_oidc = "id-token: write" in workflow
    has_token_secret = "PYPI_API_TOKEN" in workflow or "password:" in workflow
    if not has_oidc or has_token_secret:
        return {
            "name": "trusted_publishing",
            "status": "error",
            "message": "publish workflow is not configured for tokenless PyPI Trusted Publishing",
            "action": "Grant id-token: write and remove PyPI token/password inputs from the publish action.",
        }
    return {
        "name": "trusted_publishing",
        "status": "ok",
        "message": "publish workflow uses GitHub OIDC for PyPI Trusted Publishing",
    }


def _check_release_workflow(root: Path) -> dict[str, str]:
    try:
        workflow = (root / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "release_workflow",
            "status": "error",
            "message": f"publish workflow is missing or unreadable: {exc}",
            "action": "Create .github/workflows/publish.yml.",
        }
    required = {
        "release trigger": "release:" in workflow and "types: [published]" in workflow,
        "manual trigger": "workflow_dispatch:" in workflow,
        "artifact build": "python -m build --sdist --wheel" in workflow,
        "artifact check": "python -m twine check dist/*" in workflow,
        "PyPI publish action": "pypa/gh-action-pypi-publish@release/v1" in workflow,
    }
    missing = [label for label, ok in required.items() if not ok]
    if missing:
        return {
            "name": "release_workflow",
            "status": "error",
            "message": "publish workflow is missing: " + ", ".join(missing),
            "action": "Restore the release trigger, build, twine check, and PyPI publish steps.",
        }
    return {
        "name": "release_workflow",
        "status": "ok",
        "message": "publish workflow builds, checks, and publishes release artifacts",
    }


def _overall_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"
