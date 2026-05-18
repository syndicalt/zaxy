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
    root = project_root or _source_project_root()
    if root is not None:
        return pyproject_version(root)
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return "0+unknown"


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


def run_beta_readiness(*, project_root: str | Path | None = None) -> dict[str, Any]:
    """Check whether the repository exposes the beta release hardening gates."""
    root = Path(project_root or Path.cwd())
    checks = [
        _check_release_smoke_gate(root),
        _check_release_gate_script(root),
        _check_clean_repo_uat(root),
        _check_docs_happy_path(root),
        _check_capture_happy_path(root),
        _check_beta_roadmap(root),
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


def _check_release_smoke_gate(root: Path) -> dict[str, str]:
    report = run_release_smoke(project_root=root)
    if report["status"] != "ok":
        failing = [check["name"] for check in report["checks"] if check["status"] != "ok"]
        return {
            "name": "release_smoke",
            "status": "error",
            "message": "release smoke checks are not clean: " + ", ".join(failing),
            "action": "Run zaxy doctor --release-smoke and fix the failing checks.",
        }
    return {
        "name": "release_smoke",
        "status": "ok",
        "message": "release metadata, changelog, and Trusted Publishing checks are clean",
    }


def _check_release_gate_script(root: Path) -> dict[str, str]:
    path = root / "scripts" / "release-check.sh"
    try:
        script = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "release_gate",
            "status": "error",
            "message": f"scripts/release-check.sh is missing or unreadable: {exc}",
            "action": "Restore scripts/release-check.sh before beta.",
        }
    required = [
        'RUFF_CMD="ruff"',
        'MYPY_CMD="mypy"',
        "pytest",
        "scripts/check-coverage.py",
        "tests/test_packet_memory_e2e.py",
        "scripts/build-dist.sh",
        "scripts/validate-docs.sh",
        "scripts/validate-deployment.sh",
    ]
    missing = [item for item in required if item not in script]
    if missing:
        return {
            "name": "release_gate",
            "status": "error",
            "message": "release gate is missing checks: " + ", ".join(missing),
            "action": "Update scripts/release-check.sh so beta uses one authoritative gate.",
        }
    return {
        "name": "release_gate",
        "status": "ok",
        "message": "scripts/release-check.sh covers static, test, coverage, packet, package, docs, and deployment gates",
    }


def _check_clean_repo_uat(root: Path) -> dict[str, str]:
    path = root / "scripts" / "beta-uat.sh"
    try:
        script = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "clean_repo_uat",
            "status": "error",
            "message": f"scripts/beta-uat.sh is missing or unreadable: {exc}",
            "action": "Add a clean-repo UAT script for install, init, bootstrap, capture, and checkout.",
        }
    required = [
        "mktemp -d",
        "python -m pip install",
        "zaxy init",
        "local-codex",
        "local-claude",
        "zaxy memory bootstrap",
        "zaxy memory checkout",
        "zaxy doctor",
        "zaxy hook-status",
        "zaxy capture status",
        "zaxy capture-soak",
        "zaxy memory status",
    ]
    missing = [item for item in required if item not in script]
    if missing:
        return {
            "name": "clean_repo_uat",
            "status": "error",
            "message": "scripts/beta-uat.sh is missing happy-path steps: " + ", ".join(missing),
            "action": "Update scripts/beta-uat.sh to exercise the complete first-run beta path.",
        }
    return {
        "name": "clean_repo_uat",
        "status": "ok",
        "message": "scripts/beta-uat.sh covers clean install, init, bootstrap, capture, doctor, and checkout",
    }


def _check_docs_happy_path(root: Path) -> dict[str, str]:
    docs = _read_many(
        root,
        [
            "README.md",
            "docs/getting-started.md",
            "docs/testing.md",
        ],
    )
    required = [
        "pipx install zaxy-memory",
        "zaxy init",
        "zaxy memory bootstrap",
        "zaxy memory checkout",
        "scripts/beta-uat.sh",
        "zaxy doctor --beta-readiness",
    ]
    missing = [item for item in required if item not in docs]
    if missing:
        return {
            "name": "docs_happy_path",
            "status": "error",
            "message": "docs are missing beta happy-path references: " + ", ".join(missing),
            "action": "Update README.md, docs/getting-started.md, or docs/testing.md with the beta path.",
        }
    return {
        "name": "docs_happy_path",
        "status": "ok",
        "message": "docs describe install, init, bootstrap, checkout, beta UAT, and beta readiness",
    }


def _check_capture_happy_path(root: Path) -> dict[str, str]:
    docs = _read_many(
        root,
        [
            "docs/hooks.md",
            "docs/getting-started.md",
            "docs/mcp.md",
        ],
    )
    required = [
        "deterministic",
        "zaxy capture start",
        "zaxy capture status",
        "zaxy capture-soak",
        "zaxy hook-status",
        "observation coverage",
    ]
    missing = [item for item in required if item not in docs]
    if missing:
        return {
            "name": "capture_happy_path",
            "status": "error",
            "message": "capture docs are missing beta happy-path references: " + ", ".join(missing),
            "action": "Document deterministic capture startup, status, hook status, and coverage signals.",
        }
    return {
        "name": "capture_happy_path",
        "status": "ok",
        "message": "capture docs cover deterministic startup, runtime status, hook status, and observation coverage",
    }


def _check_beta_roadmap(root: Path) -> dict[str, str]:
    try:
        roadmap = (root / "BETA.md").read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "beta_roadmap",
            "status": "error",
            "message": f"BETA.md is missing or unreadable: {exc}",
            "action": "Add BETA.md with beta goals, remaining work, gates, and exit criteria.",
        }
    required = [
        "Git for LLM memory",
        "MemPalace-comparable",
        "temporal recall",
        "source recall",
        "graph traversal",
        "context-collapse",
        "CrewAI",
        "capture soak",
        "release criteria",
    ]
    missing = [item for item in required if item not in roadmap]
    if missing:
        return {
            "name": "beta_roadmap",
            "status": "error",
            "message": "BETA.md is missing roadmap items: " + ", ".join(missing),
            "action": "Update BETA.md so beta work tracks product-grade memory behavior and release criteria.",
        }
    return {
        "name": "beta_roadmap",
        "status": "ok",
        "message": "BETA.md tracks beta goals, remaining product work, gates, and release criteria",
    }


def _read_many(root: Path, paths: list[str]) -> str:
    content: list[str] = []
    for relative in paths:
        try:
            content.append((root / relative).read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(content)


def _overall_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "error" in statuses:
        return "error"
    if "warning" in statuses:
        return "warning"
    return "ok"
