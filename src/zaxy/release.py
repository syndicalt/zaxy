"""Release metadata and local release-readiness checks."""

from __future__ import annotations

import re
import subprocess
import tomllib
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

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
        _check_backend_report_inputs(root),
        _check_activation_release_fixture(root),
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
        'run_workspace "embedded" "" "status"',
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
