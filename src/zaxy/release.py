"""Release metadata and local release-readiness checks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from zaxy.external_validation import validate_external_validation_report

PACKAGE_NAME = "zaxy-memory"
ACTIVATION_FIXTURE_NOW = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
ACTIVATION_FIXTURE_MAX_CHECKOUT_AGE_MINUTES = 120
ACTIVATION_FIXTURE_MAX_PROMPT_TOKENS = 5000
ACTIVATION_FIXTURE_MIN_FACTS_PER_1K_PROMPT_TOKENS = 0.1
HIGH_CONTEXT_EVENT_TYPES = {
    "command.completed",
    "file.edit.applied",
    "tool.call.completed",
    "transcript.turn",
}


def package_version(*, project_root: Path | None = None) -> str:
    """Return the installed package version, falling back to local pyproject metadata."""
    root = project_root or _source_project_root()
    if root is not None:
        return pyproject_version(root)
    metadata = _metadata()
    try:
        return cast(str, metadata.version(PACKAGE_NAME))
    except metadata.PackageNotFoundError:
        return "0+unknown"


def pyproject_version(project_root: Path) -> str:
    """Read the Zaxy package version from pyproject.toml."""
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject.toml project.version must be a non-empty string")
    return version


def installed_package_version() -> str | None:
    """Return the version of the imported ``zaxy-memory`` distribution, or None.

    This is the version recorded at install time (``importlib.metadata``), which
    stays stale when the on-disk ``pyproject.toml`` is bumped without reinstalling
    — exactly the drift :func:`check_version_consistency` detects.
    """
    try:
        return cast(str, _metadata().version(PACKAGE_NAME))
    except Exception:  # noqa: BLE001 - metadata unavailable in some source-tree runs
        return None


def zaxy_import_path() -> str | None:
    """Return the resolved location of the imported ``zaxy`` package, or None."""
    try:
        import zaxy

        return getattr(zaxy, "__file__", None)
    except Exception:  # noqa: BLE001
        return None


def _resolve_repo_root(project_root: Path | None) -> Path | None:
    """Locate a zaxy repo root from an explicit path or the cwd, on disk only.

    Walks parents of the explicit ``project_root`` (or cwd) looking for a
    ``pyproject.toml`` whose project name is ``zaxy-memory``. Deliberately does
    NOT follow the import, so a stale site-packages install cannot masquerade as
    the repo.
    """
    start = Path(project_root) if project_root is not None else Path.cwd()
    for candidate in (start, *start.parents):
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


def check_version_consistency(*, project_root: Path | None = None) -> dict[str, Any]:
    """Doctor check: warn if the imported zaxy drifts from the repo on disk.

    Compares the version declared in the repo's ``pyproject.toml`` (read from
    disk) against the imported package's recorded version and resolved import
    path. Two independent drift signals:

    1. Version mismatch — the installed dist version != the repo's declared
       version (a bumped ``pyproject.toml`` without ``pip install -e .``).
    2. Import-path drift — the imported ``zaxy`` resolves outside the repo,
       i.e. a foreign/stale copy is shadowing the source tree.

    Either yields a ``warning`` with the remediation. This is the exact drift
    that once made the test suite import the wrong code and sent a diagnosis
    after the wrong PID. Returns a doctor-compatible check dict.
    """
    root = _resolve_repo_root(project_root)
    installed = installed_package_version()
    import_path = zaxy_import_path()
    details: dict[str, Any] = {
        "installed": installed or "unknown",
        "import_path": import_path or "unknown",
        "repo_root": str(root) if root is not None else "unknown",
    }
    if root is None:
        # Not run from a zaxy repo: cannot detect drift, just report posture.
        return {
            "name": "version_consistency",
            "status": "ok",
            "message": (
                f"imported zaxy {installed or 'unknown'} (no zaxy repo found to "
                "compare; run from the repository root to detect drift)"
            ),
            "details": details,
        }
    try:
        declared = pyproject_version(root)
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "version_consistency",
            "status": "error",
            "message": f"could not read declared version from {root}: {exc}",
            "details": details,
        }
    details["declared"] = declared

    if installed is None:
        return {
            "name": "version_consistency",
            "status": "warning",
            "message": (
                f"imported zaxy version could not be resolved; declared {declared}"
            ),
            "details": details,
            "action": "ensure zaxy is installed (`pip install -e .`) so its version is resolvable",
        }

    version_drift = installed != declared
    path_drift = (
        import_path is not None
        and not Path(import_path).resolve().is_relative_to(root.resolve())
    )

    if not version_drift and not path_drift:
        return {
            "name": "version_consistency",
            "status": "ok",
            "message": f"imported zaxy {declared} matches the repository",
            "details": details,
        }

    if version_drift and path_drift:
        message = (
            f"imported zaxy {installed} is stale vs declared {declared} and "
            f"resolves from outside the repo ({import_path})"
        )
    elif version_drift:
        message = f"imported zaxy {installed} is stale vs declared {declared}"
    else:
        message = (
            f"zaxy imports from {import_path}, outside the repo at {root} "
            f"(versions match: {declared})"
        )
    return {
        "name": "version_consistency",
        "status": "warning",
        "message": message,
        "details": details,
        "action": (
            "run `pip install -e .` to point the imported package at this repository "
            "(or `pip install -U zaxy-memory==" + declared + "` for a clean install)"
        ),
    }


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


def _metadata() -> Any:
    """Import package metadata only when source-tree lookup cannot answer."""
    return import_module("importlib.metadata")


def __getattr__(name: str) -> Any:
    """Preserve release.metadata compatibility without importing it eagerly."""
    if name == "metadata":
        module = _metadata()
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_release_smoke(*, project_root: str | Path | None = None) -> dict[str, Any]:
    """Check local release metadata without network calls or external services."""
    root = Path(project_root or Path.cwd())
    checks = [
        _check_package_version(root),
        _check_changelog(root),
        _check_trusted_publishing(root),
        _check_release_workflow(root),
        _check_langgraph_example(root),
        _check_openai_compatible_example(root),
        _check_claude_compatible_example(root),
    ]
    return {
        "status": _overall_status(checks),
        "checks": checks,
    }


def run_beta_readiness(
    *,
    project_root: str | Path | None = None,
    external_validation_report: str | Path | None = None,
    require_external_validation: bool = False,
) -> dict[str, Any]:
    """Check whether the repository exposes the beta release hardening gates."""
    root = Path(project_root or Path.cwd())
    checks = [
        _check_release_smoke_gate(root),
        _check_release_gate_script(root),
        _check_backend_report_inputs(root),
        _check_benchmark_no_regression(root),
        _check_coordination_competitor_claim_posture(root),
        _check_purpose_benchmark_gate(root),
        _check_purpose_evidence_policy_fixture(root),
        _check_release_gate_surface_coverage(root),
        _check_external_validation_evidence(
            root,
            external_validation_report=external_validation_report,
            required=require_external_validation,
        ),
        _check_activation_release_fixture(root),
        _check_clean_repo_uat(root),
        _check_first_run_timing(root),
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


def _check_langgraph_example(root: Path) -> dict[str, str]:
    """Run the dependency-light LangGraph example as a release smoke check."""
    return _check_json_example(
        root,
        name="langgraph_example",
        relative_path="examples/langgraph_memory.py",
        expected_session_id="langgraph-demo",
        expected_kind={"memory_checkout", "context_assembly"},
        success_message="examples/langgraph_memory.py runs the dependency-light LangGraph checkout path",
    )


def _check_openai_compatible_example(root: Path) -> dict[str, str]:
    """Run the OpenAI-compatible direct model-call example."""
    return _check_json_example(
        root,
        name="openai_compatible_example",
        relative_path="examples/openai_compatible_memory.py",
        expected_session_id="openai-compatible-demo",
        expected_kind={"memory_checkout"},
        success_message="examples/openai_compatible_memory.py runs the outside-MCP OpenAI-compatible model-call path",
    )


def _check_claude_compatible_example(root: Path) -> dict[str, str]:
    """Run the Claude-compatible direct model-call example."""
    return _check_json_example(
        root,
        name="claude_compatible_example",
        relative_path="examples/claude_compatible_memory.py",
        expected_session_id="claude-compatible-demo",
        expected_kind={"memory_checkout"},
        success_message="examples/claude_compatible_memory.py runs the outside-MCP Claude-compatible model-call path",
    )


def _check_json_example(
    root: Path,
    *,
    name: str,
    relative_path: str,
    expected_session_id: str,
    expected_kind: set[str],
    success_message: str,
) -> dict[str, str]:
    """Run an example that prints stable smoke-test JSON."""
    example = root / relative_path
    if not example.is_file():
        return {
            "name": name,
            "status": "error",
            "message": f"{relative_path} is missing",
            "action": f"Restore {relative_path} before release validation.",
        }
    env = dict(os.environ)
    source_path = str(root / "src")
    env["PYTHONPATH"] = source_path if not env.get("PYTHONPATH") else f"{source_path}{os.pathsep}{env['PYTHONPATH']}"
    try:
        result = subprocess.run(
            [sys.executable, str(example)],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return {
            "name": name,
            "status": "error",
            "message": f"{relative_path} could not run: {exc}",
            "action": f"Run python {relative_path} and fix the example.",
        }
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        return {
            "name": name,
            "status": "error",
            "message": f"{relative_path} failed with exit {result.returncode}: {detail}",
            "action": f"Run python {relative_path} and fix the example.",
        }
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {
            "name": name,
            "status": "error",
            "message": f"{relative_path} did not print JSON: {exc}",
            "action": f"Make {relative_path} print the stable smoke-test JSON payload.",
        }
    if (
        payload.get("session_id") != expected_session_id
        or payload.get("has_zaxy_context") is not True
        or payload.get("kind") not in expected_kind
    ):
        return {
            "name": name,
            "status": "error",
            "message": f"{relative_path} printed an unexpected smoke payload",
            "action": "Restore session_id, has_zaxy_context, and kind in the example payload.",
        }
    return {
        "name": name,
        "status": "ok",
        "message": success_message,
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
        "PYTHONPATH=src python -m zaxy hook-status",
        "--eventloom-path reports/activation-release",
        "--now 2026-05-20T12:00:00+00:00",
        "--min-activation-rate 1.0",
        "--max-checkout-prompt-tokens 5000",
        "--min-checkout-facts-per-1k-tokens 0.1",
        "scripts/check-backend-shootout.py",
        "--forbid-backends neo4j,pggraph,latticedb",
        "--require-dashboard-source embedded=embedded",
        "--require-backends embedded,bm25",
        "--require-report-metadata",
        "--require-markdown-report",
        "--require-query-results",
        "--require-git-tracked-inputs",
        "--verify-report-fingerprints",
        "backend-shootout.json",
        "--min-quality-per-1k-injected-tokens embedded=1.0",
        "longmemeval-40-backend-shootout.json",
        "--min-quality-per-1k-returned-tokens",
        "--min-answer-at-5-per-1k-returned-tokens",
        "--min-quality-per-1k-injected-tokens",
        "--min-answer-at-5-per-1k-injected-tokens",
        "--max-cold-bootstrap-ms",
        "--max-first-checkout-ms",
        "--max-append-to-projection-p95-ms",
        "--max-resident-memory-delta-bytes",
        "--max-on-disk-footprint-bytes",
        "--max-dashboard-graph-load-ms",
        "longmemeval-100-backend-shootout.json",
        "--max-checkout-p95-ms embedded=200",
        "--max-checkout-p99-ms",
        "--max-exact-p99-ms",
        "--max-keyword-p99-ms",
        "--max-vector-p99-ms",
        "--max-traversal-p99-ms",
        "--max-keyword-p95-ms",
    ]
    missing = [item for item in required if item not in script]
    required_counts = {
        "--forbid-backends neo4j,pggraph,latticedb": 3,
        "--require-query-results": 3,
        "--require-git-tracked-inputs": 3,
    }
    missing.extend(
        f"{item} ({count} occurrences)"
        for item, count in required_counts.items()
        if script.count(item) < count
    )
    backend_command_flags = {
        "BACKEND_SHOOTOUT_CMD": [
            "--forbid-backends neo4j,pggraph,latticedb",
            "--require-dashboard-source embedded=embedded",
            "--require-backends embedded,bm25",
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-query-results",
            "--require-git-tracked-inputs",
            "--verify-report-fingerprints",
            "--require-labeled-metrics",
        ],
        "BACKEND_PERFORMANCE_CMD": [
            "--forbid-backends neo4j,pggraph,latticedb",
            "--require-dashboard-source embedded=embedded",
            "--require-backends embedded,bm25",
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-query-results",
            "--require-git-tracked-inputs",
            "--verify-report-fingerprints",
            "--require-labeled-metrics",
        ],
        "BACKEND_SCALE_CMD": [
            "--forbid-backends neo4j,pggraph,latticedb",
            "--require-dashboard-source embedded=embedded",
            "--require-backends embedded,bm25",
            "--require-report-metadata",
            "--require-markdown-report",
            "--require-query-results",
            "--require-git-tracked-inputs",
            "--verify-report-fingerprints",
            "--require-labeled-metrics",
            "--max-checkout-p95-ms embedded=200",
        ],
    }
    assignments = _shell_string_assignments(script)
    missing.extend(
        f"{name} must include {flag}"
        for name, flags in backend_command_flags.items()
        for flag in flags
        if flag not in assignments.get(name, "")
    )
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
        "message": (
            "scripts/release-check.sh covers static, test, coverage, packet, package, docs, deployment, "
            "activation, backend shootout, medium-scale performance, 100-query scale gates, "
            "and optional backend exclusion"
        ),
    }


def _check_backend_report_inputs(root: Path) -> dict[str, str]:
    reports = [
        root / "reports" / "backend-shootout" / "backend-shootout.json",
        root / "reports" / "backend-shootout" / "longmemeval-40-backend-shootout.json",
        root / "reports" / "backend-shootout" / "longmemeval-100-backend-shootout.json",
    ]
    missing: list[str] = []
    git_root = _git_root(root)
    for report in reports:
        if not report.exists():
            missing.append(f"{report.relative_to(root)} is missing")
            continue
        try:
            payload = _json_loads(report)
        except ValueError as exc:
            missing.append(f"{report.relative_to(root)} is unreadable: {exc}")
            continue
        query_results = payload.get("query_results")
        if not isinstance(query_results, dict) or not query_results:
            missing.append(f"{report.name} query_results are missing")
        else:
            for result_key, diagnostics in sorted(query_results.items()):
                if not isinstance(diagnostics, list):
                    missing.append(f"{report.name} query_results {result_key} must be a diagnostics list")
                elif not diagnostics:
                    missing.append(f"{report.name} query_results {result_key} has no diagnostics")
                else:
                    for index, diagnostic in enumerate(diagnostics):
                        if not isinstance(diagnostic, dict):
                            missing.append(
                                f"{report.name} query_results {result_key}[{index}] must be a diagnostic object"
                            )
        for key in ("eventloom_path", "queries_file"):
            value = payload.get(key)
            if not isinstance(value, str) or not value:
                missing.append(f"{report.relative_to(root)} {key} is missing")
                continue
            path = Path(value)
            absolute = path if path.is_absolute() else root / path
            if not absolute.exists():
                missing.append(f"{report.relative_to(root)} {key} {value} is missing")
                continue
            if git_root is not None and not _git_path_is_tracked(git_root, absolute):
                missing.append(f"{report.name} {key} {value} is not tracked by git")
    if missing:
        return {
            "name": "backend_report_inputs",
            "status": "error",
            "message": "backend report inputs are not reproducible: " + ", ".join(missing),
            "action": "Track benchmark Eventloom/query inputs or regenerate reports against tracked inputs.",
        }
    return {
        "name": "backend_report_inputs",
        "status": "ok",
        "message": "Existing backend report inputs are present and tracked when git metadata is available.",
    }


def _check_benchmark_no_regression(root: Path) -> dict[str, str]:
    path = root / "scripts" / "release-check.sh"
    try:
        script = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "benchmark_no_regression",
            "status": "error",
            "message": f"scripts/release-check.sh is missing or unreadable: {exc}",
            "action": "Restore scripts/release-check.sh benchmark guardrails before beta.",
        }
    assignments = _shell_string_assignments(script)
    required_by_command = {
        "BACKEND_SHOOTOUT_CMD": [
            "--min-answer-at-5",
            "--min-recall-at-5",
            "--min-citation-coverage 1.0",
            "--min-quality-per-1k-injected-tokens",
            "--min-answer-at-5-per-1k-injected-tokens",
            "--max-checkout-p99-ms",
        ],
        "BACKEND_PERFORMANCE_CMD": [
            "--min-citation-coverage 1.0",
            "--min-quality-per-1k-returned-tokens",
            "--min-answer-at-5-per-1k-returned-tokens",
            "--min-quality-per-1k-injected-tokens",
            "--min-answer-at-5-per-1k-injected-tokens",
            "--max-checkout-p95-ms",
            "--max-checkout-p99-ms",
        ],
        "BACKEND_SCALE_CMD": [
            "--min-recall-at-5",
            "--min-citation-coverage 1.0",
            "--min-quality-per-1k-returned-tokens",
            "--min-answer-at-5-per-1k-returned-tokens",
            "--min-quality-per-1k-injected-tokens",
            "--min-answer-at-5-per-1k-injected-tokens",
            "--max-checkout-p95-ms",
            "--max-checkout-p99-ms",
        ],
    }
    missing = [
        f"{name} must include {flag}"
        for name, flags in required_by_command.items()
        for flag in flags
        if flag not in assignments.get(name, "")
    ]
    if missing:
        return {
            "name": "benchmark_no_regression",
            "status": "error",
            "message": "benchmark no-regression gate is missing checks: " + ", ".join(missing),
            "action": "Restore checkout quality, citation coverage, and p95/p99 latency guardrails.",
        }
    return {
        "name": "benchmark_no_regression",
        "status": "ok",
        "message": (
            "Release benchmark checks gate checkout quality, citation coverage, "
            "and p95/p99 latency budgets across smoke, performance, and scale reports."
        ),
    }


def _check_coordination_competitor_claim_posture(root: Path) -> dict[str, str]:
    """Verify public CoordinationBench docs do not overclaim Quarq/Hybi results."""
    report_path = root / "reports" / "benchmarks" / "coordination-real-v1" / "coordination-benchmark.json"
    markdown_path = root / "reports" / "benchmarks" / "coordination-real-v1" / "coordination-benchmark.md"
    benchmarks_path = root / "docs" / "benchmarks.md"
    roadmap_path = root / "docs" / "coordinate-roadmap.md"
    manifest_dir = root / "reports" / "benchmarks" / "coordination-real-v1" / "competitor-runner-manifests"
    missing_files = [
        str(path.relative_to(root))
        for path in (report_path, markdown_path, benchmarks_path, roadmap_path)
        if not path.is_file()
    ]
    for name in ("quarq", "hybi"):
        manifest_path = manifest_dir / f"{name}.runner-manifest.template.json"
        if not manifest_path.is_file():
            missing_files.append(str(manifest_path.relative_to(root)))
    if missing_files:
        return {
            "name": "coordination_competitor_claims",
            "status": "error",
            "message": "CoordinationBench competitor claim artifacts are missing: " + ", ".join(missing_files),
            "action": "Restore the archived CoordinationBench report, docs, and Quarq/Hybi manifest templates.",
        }
    try:
        report = _json_loads(report_path)
    except ValueError as exc:
        return {
            "name": "coordination_competitor_claims",
            "status": "error",
            "message": f"CoordinationBench report is not valid JSON: {exc}",
            "action": "Regenerate or repair reports/benchmarks/coordination-real-v1/coordination-benchmark.json.",
        }
    docs = _read_many(root, ["docs/benchmarks.md", "docs/coordinate-roadmap.md"])
    markdown = markdown_path.read_text(encoding="utf-8")
    required_doc_fragments = [
        "competitor_claim_gate",
        "--require-competitor-claim quarq",
        "--require-competitor-claim hybi",
        "disclosure-only",
        "public-claim gate",
    ]
    missing_docs = [fragment for fragment in required_doc_fragments if fragment not in docs]
    if "## Competitor Claim Gate" not in markdown:
        missing_docs.append("archived report markdown must include Competitor Claim Gate")
    if missing_docs:
        return {
            "name": "coordination_competitor_claims",
            "status": "error",
            "message": "CoordinationBench competitor claim docs are incomplete: " + ", ".join(missing_docs),
            "action": "Document the Quarq/Hybi claim gate and disclosure-only posture before release.",
        }
    adapters = report.get("competitor_adapters")
    if not isinstance(adapters, dict):
        return {
            "name": "coordination_competitor_claims",
            "status": "error",
            "message": "CoordinationBench report is missing competitor_adapters.",
            "action": "Regenerate the CoordinationBench report with competitor disclosures.",
        }
    gate = report.get("competitor_claim_gate")
    if not isinstance(gate, dict):
        return {
            "name": "coordination_competitor_claims",
            "status": "error",
            "message": "CoordinationBench report is missing competitor_claim_gate.",
            "action": "Regenerate the CoordinationBench report with the public claim gate.",
        }
    blockers: list[str] = []
    for name in ("quarq", "hybi"):
        adapter = adapters.get(name)
        if not isinstance(adapter, dict):
            blockers.append(f"{name} disclosure row is missing")
            continue
        status = str(adapter.get("status") or "")
        claim_status = str(adapter.get("claim_status") or "")
        if status == "not_run" and claim_status == "disclosure_only":
            continue
        if status != "completed" or claim_status != "same_harness":
            blockers.append(f"{name} has unsupported status {status}/{claim_status}")
            continue
        if not isinstance(adapter.get("metrics"), dict):
            blockers.append(f"{name} completed row is missing locally scored metrics")
        audit = adapter.get("result_audit")
        if not isinstance(audit, dict):
            blockers.append(f"{name} completed row is missing result audit")
            continue
        if not str(audit.get("result_fingerprint") or ""):
            blockers.append(f"{name} result audit is missing result_fingerprint")
        manifest = audit.get("manifest")
        if not isinstance(manifest, dict):
            blockers.append(f"{name} result audit manifest is missing")
            continue
        missing_manifest_fields = [
            field
            for field in (
                "name",
                "display_name",
                "adapter_contract",
                "adapter_version",
                "install_command",
                "run_command",
                "source_url",
                "source_ref",
            )
            if not str(manifest.get(field) or "").strip()
        ]
        if missing_manifest_fields:
            blockers.append(f"{name} result audit manifest missing {', '.join(missing_manifest_fields)}")
    gate_status = str(gate.get("status") or "")
    blocked_adapters = gate.get("blocked_adapters")
    if gate_status == "blocked":
        blocked_names = set(blocked_adapters) if isinstance(blocked_adapters, dict) else set()
        if not {"quarq", "hybi"} <= blocked_names:
            blockers.append("blocked claim gate must name quarq and hybi")
    elif gate_status == "passed":
        completed = {str(item) for item in gate.get("completed_adapters", []) if isinstance(item, str)}
        if not {"quarq", "hybi"} <= completed:
            blockers.append("passed claim gate must include completed quarq and hybi")
    else:
        blockers.append(f"competitor_claim_gate has unsupported status {gate_status!r}")
    if blockers:
        return {
            "name": "coordination_competitor_claims",
            "status": "error",
            "message": "CoordinationBench competitor claim posture is unsafe: " + "; ".join(blockers),
            "action": "Keep Quarq/Hybi disclosure-only or attach completed same-harness result audits before release.",
        }
    return {
        "name": "coordination_competitor_claims",
        "status": "ok",
        "message": (
            "CoordinationBench Quarq/Hybi posture is guarded: disclosure-only rows stay blocked "
            "unless completed same-harness result audits exist."
        ),
    }


def _check_purpose_benchmark_gate(root: Path) -> dict[str, str]:
    """Verify purpose-conditioned memory claims have a passing archived gate."""
    report_path = root / "reports" / "benchmarks" / "purpose-v1" / "purpose-benchmark.json"
    markdown_path = root / "reports" / "benchmarks" / "purpose-v1" / "purpose-benchmark.md"
    docs_path = root / "docs" / "benchmarks.md"
    holdout_pack_path = root / "reports" / "benchmarks" / "purpose-v1" / "holdouts" / "public-derived-purpose-v1" / "holdout-pack.json"
    missing_files = [
        str(path.relative_to(root))
        for path in (report_path, markdown_path, docs_path, holdout_pack_path)
        if not path.is_file()
    ]
    if missing_files:
        return {
            "name": "purpose_benchmark_gate",
            "status": "error",
            "message": "Purpose benchmark artifacts are missing: " + ", ".join(missing_files),
            "action": "Run python -m zaxy purpose-benchmark --output-dir reports/benchmarks/purpose-v1.",
        }
    try:
        report = _json_loads(report_path)
    except ValueError as exc:
        return {
            "name": "purpose_benchmark_gate",
            "status": "error",
            "message": f"Purpose benchmark report is not valid JSON: {exc}",
            "action": "Regenerate reports/benchmarks/purpose-v1/purpose-benchmark.json.",
        }
    required_lanes = {
        "Purpose Recall",
        "Ontology Shift",
        "Consequence Retention",
        "Governed Forgetting",
        "Action Outcome Loop",
        "Evidence Policy Discipline",
        "Broader Profile Fixtures",
        "Neutral Substrate Projection",
        "Cross-Role Citation",
        "Accepted-State Discipline",
    }
    lanes = report.get("lanes")
    if not isinstance(lanes, list):
        return {
            "name": "purpose_benchmark_gate",
            "status": "error",
            "message": "Purpose benchmark report is missing lanes.",
            "action": "Regenerate the purpose-v1 benchmark report.",
        }
    lane_statuses = {
        str(lane.get("name") or ""): str(lane.get("status") or "")
        for lane in lanes
        if isinstance(lane, dict)
    }
    missing_lanes = sorted(required_lanes - set(lane_statuses))
    failing_lanes = sorted(name for name in required_lanes if lane_statuses.get(name) != "passed")
    blockers: list[str] = []
    if str(report.get("version") or "") != "purpose-v1":
        blockers.append("version must be purpose-v1")
    if str(report.get("status") or "") != "passed":
        blockers.append("report status must be passed")
    holdout_reports = report.get("holdout_reports")
    if not isinstance(holdout_reports, dict) or "public-derived-purpose-v1" not in holdout_reports:
        blockers.append("purpose benchmark report must include public-derived-purpose-v1 holdout diagnostics")
    else:
        holdout = holdout_reports["public-derived-purpose-v1"]
        if not isinstance(holdout, dict) or holdout.get("gate_status") != "diagnostic":
            blockers.append("public-derived-purpose-v1 holdout must be diagnostic, not a release lane")
        if holdout.get("claim_status") != "public_derived_holdout":
            blockers.append("public-derived-purpose-v1 holdout claim_status must be public_derived_holdout")
        metrics = holdout.get("metrics") if isinstance(holdout, dict) else None
        if not isinstance(metrics, dict) or metrics.get("case_count") != 5:
            blockers.append("public-derived-purpose-v1 holdout must report five representative cases")
    if missing_lanes:
        blockers.append("missing lanes: " + ", ".join(missing_lanes))
    if failing_lanes:
        blockers.append("failing lanes: " + ", ".join(failing_lanes))
    evidence_policy_lane = next(
        (
            lane
            for lane in lanes
            if isinstance(lane, dict) and str(lane.get("name") or "") == "Evidence Policy Discipline"
        ),
        None,
    )
    if not isinstance(evidence_policy_lane, dict):
        blockers.append("Evidence Policy Discipline lane is missing")
    else:
        evidence = evidence_policy_lane.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            blockers.append("Evidence Policy Discipline lane evidence is missing")
        else:
            for profile in (
                "security",
                "release",
                "coordinate",
                "support",
                "product",
                "sales",
                "legal",
                "executive",
            ):
                profile_evidence = evidence.get(profile)
                if not isinstance(profile_evidence, dict):
                    blockers.append(f"Evidence Policy Discipline missing {profile} fixture evidence")
                    continue
                unsupported = profile_evidence.get("unsupported")
                supported = profile_evidence.get("supported")
                if not isinstance(unsupported, dict) or not isinstance(supported, dict):
                    blockers.append(f"Evidence Policy Discipline {profile} fixture evidence is incomplete")
                    continue
                if unsupported.get("satisfied") is not False:
                    blockers.append(f"Evidence Policy Discipline {profile} unsupported fixture must fail")
                if supported.get("satisfied") is not True:
                    blockers.append(f"Evidence Policy Discipline {profile} supported fixture must pass")
    broader_profile_lane = next(
        (
            lane
            for lane in lanes
            if isinstance(lane, dict) and str(lane.get("name") or "") == "Broader Profile Fixtures"
        ),
        None,
    )
    if not isinstance(broader_profile_lane, dict):
        blockers.append("Broader Profile Fixtures lane is missing")
    else:
        evidence = broader_profile_lane.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            blockers.append("Broader Profile Fixtures lane evidence is missing")
        else:
            passed_profiles = evidence.get("passed_profiles")
            required_profiles = {"support", "product", "sales", "legal", "executive"}
            if not isinstance(passed_profiles, list) or required_profiles - set(map(str, passed_profiles)):
                blockers.append("Broader Profile Fixtures must pass support, product, sales, legal, and executive")
            if evidence.get("local_project_memory_positioning") is not True:
                blockers.append("Broader Profile Fixtures must preserve local/project-memory positioning")
    neutral_lane = next(
        (
            lane
            for lane in lanes
            if isinstance(lane, dict) and str(lane.get("name") or "") == "Neutral Substrate Projection"
        ),
        None,
    )
    if not isinstance(neutral_lane, dict):
        blockers.append("Neutral Substrate Projection lane is missing")
    else:
        evidence = neutral_lane.get("evidence")
        if not isinstance(evidence, dict) or not evidence:
            blockers.append("Neutral Substrate Projection lane evidence is missing")
        else:
            projections = evidence.get("purpose_projections")
            audit = evidence.get("ingestion_audit")
            if not isinstance(projections, dict) or set(projections) != {"support", "product", "legal", "executive"}:
                blockers.append("Neutral Substrate Projection must include support, product, legal, and executive projections")
            if not isinstance(audit, dict) or audit.get("safe") is not True:
                blockers.append("Neutral Substrate Projection ingestion audit must be safe")
    competitor_status = str(report.get("competitor_claim_status") or "")
    if competitor_status != "blocked":
        blockers.append("competitor_claim_status must remain blocked without same-harness adapters")
    docs = _read_many(root, ["docs/benchmarks.md"])
    markdown = markdown_path.read_text(encoding="utf-8")
    for fragment in ("purpose-v1", "Semantic Reach", "Quarq", "same-harness"):
        if fragment not in docs:
            blockers.append(f"docs/benchmarks.md missing {fragment}")
    if "Purpose Recall" not in markdown or "Accepted-State Discipline" not in markdown:
        blockers.append("purpose benchmark markdown is missing required lane names")
    if "Public-Derived Holdouts" not in markdown:
        blockers.append("purpose benchmark markdown is missing public-derived holdout section")
    if blockers:
        return {
            "name": "purpose_benchmark_gate",
            "status": "error",
            "message": "Purpose benchmark gate is unsafe: " + "; ".join(blockers),
            "action": "Regenerate the purpose-v1 report and document the blocked competitor-claim posture.",
        }
    return {
        "name": "purpose_benchmark_gate",
        "status": "ok",
        "message": "purpose-v1 benchmark passes all purpose-memory lanes with competitor claims blocked.",
    }


def _check_purpose_evidence_policy_fixture(root: Path) -> dict[str, str]:
    """Verify high-risk purpose evidence policies are executable release fixtures."""
    del root
    try:
        from zaxy.evidence import evaluate_evidence_policy
    except Exception as exc:
        return {
            "name": "purpose_evidence_policy",
            "status": "error",
            "message": f"could not import evidence policy evaluator: {exc}",
            "action": "Restore zaxy.evidence.evaluate_evidence_policy.",
        }
    fixtures = {
        "security": {
            "query": "review credential exposure",
            "content": "Credential exposure found in auth config.",
            "missing": {"mitigation_or_risk_owner"},
        },
        "release": {
            "query": "ship release",
            "content": "Release gate is green for the current candidate.",
            "missing": {"verification_refs"},
        },
        "coordinate": {
            "query": "handoff accepted auth state",
            "content": "Worker-local finding says auth cache is stale.",
            "missing": {"promotion_or_review_ref"},
        },
        "support": {
            "query": "triage customer escalation",
            "content": "Customer case says the dashboard is broken.",
            "missing": {"workaround_or_resolution_ref"},
        },
        "product": {
            "query": "prioritize roadmap signal",
            "content": "Roadmap should prioritize dashboard export.",
            "missing": {"tradeoff_ref"},
        },
        "sales": {
            "query": "prepare account follow-up",
            "content": "The account wants a follow-up.",
            "missing": {"commitment_ref"},
        },
        "legal": {
            "query": "review contract obligation",
            "content": "The contract allows redistribution.",
            "missing": {"exact_quote_ref"},
        },
        "executive": {
            "query": "summarize strategic exception",
            "content": "There is a strategic exception.",
            "missing": {"risk_or_metric_ref"},
        },
    }
    blockers: list[str] = []
    for profile, fixture in fixtures.items():
        fact = {
            "content": str(fixture["content"]),
            "source": "graph",
            "citation": f"eventloom://fixture/events/{profile}#abcdefabcdef",
        }
        result = evaluate_evidence_policy(
            profile=profile,
            query=str(fixture["query"]),
            current_facts=[fact],
            evidence=[fact],
        )
        if result is None:
            blockers.append(f"{profile} policy did not run")
            continue
        if result.satisfied:
            blockers.append(f"{profile} unsupported fixture unexpectedly satisfied policy")
        missing = set(result.missing_requirements)
        expected_missing = set(fixture["missing"])
        if not expected_missing <= missing:
            blockers.append(
                f"{profile} missing requirements {sorted(missing)} did not include {sorted(expected_missing)}"
            )
        if result.mode not in {"block_checkout", "require_refresh", "warn"}:
            blockers.append(f"{profile} policy mode {result.mode!r} is not actionable")
        if not result.suggested_queries:
            blockers.append(f"{profile} policy did not emit suggested refresh queries")
    if blockers:
        return {
            "name": "purpose_evidence_policy",
            "status": "error",
            "message": "Purpose evidence policy fixtures failed: " + "; ".join(blockers),
            "action": "Restore purpose evidence-policy fixtures before release.",
        }
    return {
        "name": "purpose_evidence_policy",
        "status": "ok",
        "message": "security, release, Coordinate, support, product, sales, legal, and executive evidence-policy fixtures enforce actionable behavior.",
    }


def _check_release_gate_surface_coverage(root: Path) -> dict[str, str]:
    path = root / "scripts" / "release-check.sh"
    try:
        script = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {
            "name": "release_gate_surface_coverage",
            "status": "error",
            "message": f"scripts/release-check.sh is missing or unreadable: {exc}",
            "action": "Restore scripts/release-check.sh before beta.",
        }
    assignments = _shell_string_assignments(script)
    required_surfaces = {
        "EXAMPLES_SMOKE_CMD": ("public examples", "tests/test_examples_v05.py"),
        "MCP_SMOKE_CMD": ("MCP smoke", "scripts/mcp_smoke_test.py"),
        "LANGGRAPH_SMOKE_CMD": ("LangGraph smoke", "test_langgraph_example_runs_without_langgraph_dependency"),
        "COORDINATE_SMOKE_CMD": ("Coordinate mission smoke", "test_coordinate_three_worker_example_runs"),
        "BACKEND_SHOOTOUT_CMD": ("benchmark comparison", "scripts/check-backend-shootout.py"),
        "DOCS_CMD": ("docs validation", "scripts/validate-docs.sh"),
        "BETA_UAT_CMD": ("beta UAT", "scripts/beta-uat.sh"),
        "EXTERNAL_VALIDATION_CMD": ("external validation", "scripts/check-external-validation.py"),
    }
    missing: list[str] = []
    labels: list[str] = []
    for name, (label, required_fragment) in required_surfaces.items():
        command = assignments.get(name, "").strip()
        labels.append(label)
        if not command:
            missing.append(f"{name} must run or use SKIP:<reason>")
            continue
        if command.startswith("SKIP:"):
            if not command.removeprefix("SKIP:").strip():
                missing.append(f"{name} SKIP must include a reason")
            continue
        if required_fragment not in command:
            missing.append(f"{name} must include {required_fragment}")
    if "run_gate" not in script or "SKIP:" not in script:
        missing.append("scripts/release-check.sh must print explicit SKIP:<reason> messages")
    if missing:
        return {
            "name": "release_gate_surface_coverage",
            "status": "error",
            "message": "release gate surface coverage is incomplete: " + ", ".join(missing),
            "action": "Update scripts/release-check.sh so every public smoke runs or prints SKIP:<reason>.",
        }
    return {
        "name": "release_gate_surface_coverage",
        "status": "ok",
        "message": "Release gate runs or explicitly skips with reasons: " + ", ".join(labels) + ".",
    }


def _check_external_validation_evidence(
    root: Path,
    *,
    external_validation_report: str | Path | None = None,
    required: bool = False,
) -> dict[str, str]:
    default_report = Path("reports") / "external-validation" / "external-validation-report.json"
    report_path = Path(external_validation_report) if external_validation_report is not None else root / default_report
    if not report_path.is_absolute():
        report_path = root / report_path
    display_report = str(Path(external_validation_report)) if external_validation_report is not None else str(default_report)
    if not report_path.exists():
        if not required and external_validation_report is None:
            return {
                "name": "external_validation_evidence",
                "status": "ok",
                "message": f"external validation is optional for v1.0 release; {display_report} is not present",
                "action": (
                    "Collect post-release outside-user evidence with docs/external-validation.md when available, "
                    f"write {display_report}, then run scripts/check-external-validation.py."
                ),
            }
        prefix = "external validation is required" if required else "external validation report was requested"
        return {
            "name": "external_validation_evidence",
            "status": "error",
            "message": f"{prefix}; {display_report} is not present",
            "action": (
                "Collect outside-user evidence with docs/external-validation.md, write "
                f"{display_report}, then run scripts/check-external-validation.py."
            ),
        }
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "name": "external_validation_evidence",
            "status": "error",
            "message": f"{display_report} is unreadable or invalid JSON: {exc}",
            "action": "Fix the external validation report JSON before release.",
        }

    errors = validate_external_validation_report(payload)
    if errors:
        return {
            "name": "external_validation_evidence",
            "status": "error",
            "message": "external validation report is invalid: " + ", ".join(errors),
            "action": f"Run scripts/check-external-validation.py {display_report} and fix the report.",
        }
    return {
        "name": "external_validation_evidence",
        "status": "ok",
        "message": f"{display_report} contains validated outside-user release evidence",
    }


def _json_loads(path: Path) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("top-level JSON value must be an object")
    return payload


def _git_root(root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _git_path_is_tracked(git_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(git_root)
    except ValueError:
        return False
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=git_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _shell_string_assignments(script: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for match in re.finditer(r"^([A-Z][A-Z0-9_]*)=(['\"])(.*?)\2$", script, flags=re.MULTILINE):
        assignments[match.group(1)] = match.group(3)
    return assignments


def _check_activation_release_fixture(root: Path) -> dict[str, str]:
    from zaxy.event import EventLog

    fixture = root / "reports" / "activation-release"
    paths = sorted(fixture.glob("*.jsonl")) if fixture.is_dir() else []
    if not paths:
        return {
            "name": "activation_release_fixture",
            "status": "error",
            "message": "reports/activation-release has no checked Eventloom JSONL activation fixture",
            "action": "Restore the activation fixture used by scripts/release-check.sh.",
        }
    checkout_seen = False
    high_context_after_checkout = False
    token_efficiency_errors: list[str] = []
    for path in paths:
        try:
            log = EventLog(path)
            integrity = log.verify()
            events = log.read_all()
        except Exception as exc:
            return {
                "name": "activation_release_fixture",
                "status": "error",
                "message": f"{path.relative_to(root)} is not readable Eventloom JSONL: {exc}",
                "action": "Restore the activation fixture used by scripts/release-check.sh.",
            }
        if not integrity.ok:
            return {
                "name": "activation_release_fixture",
                "status": "error",
                "message": (
                    f"{path.relative_to(root)} failed Eventloom integrity: "
                    f"{integrity.broken_reason or 'unknown integrity error'}"
                ),
                "action": "Restore the activation fixture used by scripts/release-check.sh.",
            }
        checkout_threads: set[str] = set()
        for event in events:
            if event.type == "memory.checkout.completed" and _has_token_efficiency(event.payload):
                checkout_seen = True
                token_efficiency_errors.extend(_activation_token_efficiency_errors(event.payload))
                token_efficiency_errors.extend(_activation_checkout_freshness_errors(event.timestamp))
                checkout_threads.add(event.thread)
            elif event.type in HIGH_CONTEXT_EVENT_TYPES and event.thread in checkout_threads:
                high_context_after_checkout = True
    if not checkout_seen:
        return {
            "name": "activation_release_fixture",
            "status": "error",
            "message": "reports/activation-release has no memory.checkout.completed event with token_efficiency",
            "action": "Restore the activation fixture used by scripts/release-check.sh.",
        }
    if not high_context_after_checkout:
        return {
            "name": "activation_release_fixture",
            "status": "error",
            "message": "reports/activation-release has no high-context event after checkout",
            "action": "Restore the activation fixture used by scripts/release-check.sh.",
        }
    if token_efficiency_errors:
        return {
            "name": "activation_release_fixture",
            "status": "error",
            "message": "reports/activation-release token-efficiency guardrail failed: "
            + ", ".join(token_efficiency_errors),
            "action": "Restore the activation fixture used by scripts/release-check.sh.",
        }
    return {
        "name": "activation_release_fixture",
        "status": "ok",
        "message": "reports/activation-release proves fresh checkout with token-efficiency metadata before work",
    }


def _has_token_efficiency(payload: dict[str, Any]) -> bool:
    value = payload.get("token_efficiency")
    if not isinstance(value, dict):
        return False
    prompt_tokens = value.get("prompt_tokens")
    facts_per_1k = value.get("facts_per_1k_prompt_tokens")
    if isinstance(prompt_tokens, bool) or isinstance(facts_per_1k, bool):
        return False
    return isinstance(prompt_tokens, int | float) and isinstance(facts_per_1k, int | float)


def _activation_token_efficiency_errors(payload: dict[str, Any]) -> list[str]:
    value = payload["token_efficiency"]
    prompt_tokens = value["prompt_tokens"]
    facts_per_1k = value["facts_per_1k_prompt_tokens"]
    errors: list[str] = []
    if prompt_tokens > ACTIVATION_FIXTURE_MAX_PROMPT_TOKENS:
        errors.append(f"prompt_tokens={int(prompt_tokens)} exceeds {ACTIVATION_FIXTURE_MAX_PROMPT_TOKENS}")
    if facts_per_1k < ACTIVATION_FIXTURE_MIN_FACTS_PER_1K_PROMPT_TOKENS:
        errors.append(
            f"facts_per_1k_prompt_tokens={float(facts_per_1k):g} "
            f"is below {ACTIVATION_FIXTURE_MIN_FACTS_PER_1K_PROMPT_TOKENS:g}"
        )
    return errors


def _activation_checkout_freshness_errors(timestamp: str) -> list[str]:
    checkout_time = _parse_event_timestamp(timestamp)
    age_minutes = (ACTIVATION_FIXTURE_NOW - checkout_time).total_seconds() / 60
    if age_minutes > ACTIVATION_FIXTURE_MAX_CHECKOUT_AGE_MINUTES:
        return [
            f"checkout age {age_minutes:.1f} minutes exceeds {ACTIVATION_FIXTURE_MAX_CHECKOUT_AGE_MINUTES} minutes"
        ]
    return []


def _parse_event_timestamp(timestamp: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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
        'run_workspace "embedded" "" "start"',
        'if [[ -n "${preset}" ]]',
        "PROJECTION_BACKEND=embedded",
        "NEO4J_AUTO_START=false",
        "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu",
        "zaxy memory bootstrap",
        "zaxy memory checkout",
        "zaxy doctor",
        "zaxy hook-status",
        "--min-activation-rate 1.0",
        "--max-checkout-prompt-tokens 5000",
        "--min-checkout-facts-per-1k-tokens 0.1",
        "zaxy capture status",
        "zaxy capture-soak",
        "zaxy memory status",
        "zaxy memory status --eventloom-path .eventloom --graph",
        "Graph projection (backend=embedded):",
        "zaxy memory inferred-status --session-id",
        '"backend": "embedded"',
        "zaxy reproject",
        "using embedded",
    ]
    missing = [item for item in required if item not in script]
    if missing:
        return {
            "name": "clean_repo_uat",
            "status": "error",
            "message": "scripts/beta-uat.sh is missing happy-path steps, including bare embedded init: "
            + ", ".join(missing),
            "action": "Update scripts/beta-uat.sh to exercise the complete first-run beta path.",
        }
    return {
        "name": "clean_repo_uat",
        "status": "ok",
        "message": "scripts/beta-uat.sh covers clean install, init, bootstrap, capture, doctor, and checkout",
    }


def _check_first_run_timing(root: Path) -> dict[str, str]:
    path = root / "docs" / "examples" / "first-run-timing-report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {
            "name": "first_run_timing",
            "status": "error",
            "message": f"docs/examples/first-run-timing-report.json is missing or unreadable: {exc}",
            "action": "Add a clean first-run timing report with doctor and example timings.",
        }
    except json.JSONDecodeError as exc:
        return {
            "name": "first_run_timing",
            "status": "error",
            "message": f"first-run timing report is invalid JSON: {exc}",
            "action": "Update docs/examples/first-run-timing-report.json with valid JSON.",
        }
    threshold = _number_field(payload, "threshold_seconds")
    doctor_seconds = _number_field(payload, "time_to_successful_doctor_seconds")
    example_seconds = _number_field(payload, "time_to_first_successful_example_seconds")
    requires_sidecar = payload.get("requires_sidecar")
    errors: list[str] = []
    if threshold != 300:
        errors.append(f"threshold_seconds={_format_number(threshold)} must be 300")
    if doctor_seconds is None:
        errors.append("time_to_successful_doctor_seconds is missing")
    elif doctor_seconds > 300:
        errors.append(f"time_to_successful_doctor_seconds={_format_number(doctor_seconds)} exceeds 300")
    if example_seconds is None:
        errors.append("time_to_first_successful_example_seconds is missing")
    elif example_seconds > 300:
        errors.append(f"time_to_first_successful_example_seconds={_format_number(example_seconds)} exceeds 300")
    if requires_sidecar is not False:
        errors.append("requires_sidecar must be false")
    if errors:
        return {
            "name": "first_run_timing",
            "status": "error",
            "message": "first-run timing report does not satisfy the v0.6 budget: " + "; ".join(errors),
            "action": "Update docs/examples/first-run-timing-report.json with a passing clean first-run timing report.",
        }
    return {
        "name": "first_run_timing",
        "status": "ok",
        "message": (
            "clean first-run timing stays under 300 seconds "
            f"(doctor={_format_number(doctor_seconds)}s, example={_format_number(example_seconds)}s)"
        ),
    }


def _number_field(payload: dict[str, Any], field: str) -> float | None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _format_number(value: float | None) -> str:
    if value is None:
        return "missing"
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


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
