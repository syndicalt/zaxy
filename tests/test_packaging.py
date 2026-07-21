"""Tests for release packaging metadata and artifact gates."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import zaxy.external_validation as external_validation
from zaxy.event import EventLog
from zaxy.external_validation import validate_external_validation_report
from zaxy.release import (
    ACTIVATION_FIXTURE_NOW,
    BETA_BENCHMARK_LANES,
    _activation_checkout_freshness_errors,
    _activation_token_efficiency_errors,
    _beta_cli_claims,
    _check_activation_release_fixture,
    _check_backend_report_inputs,
    _check_benchmark_no_regression,
    _check_beta_roadmap_claims,
    _check_capture_happy_path,
    _check_changelog,
    _check_coordination_competitor_claim_posture,
    _check_docs_happy_path,
    _check_external_validation_evidence,
    _check_first_run_timing,
    _check_json_example,
    _check_package_version,
    _check_purpose_evidence_policy_fixture,
    _check_release_gate_surface_coverage,
    _check_release_smoke_gate,
    _check_release_workflow,
    _check_trusted_publishing,
    _has_token_efficiency,
    _overall_status,
    package_version,
    pyproject_version,
    run_beta_readiness,
)


def test_pyproject_declares_typed_package_and_release_tools() -> None:
    """The wheel should advertise typing and include build/check tooling."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "zaxy-memory"
    assert pyproject["project"]["urls"] == {
        "Homepage": "https://syndicalt.github.io/zaxy/",
        "Documentation": "https://syndicalt.github.io/zaxy/docs/getting-started.html",
        "Repository": "https://github.com/syndicalt/zaxy",
        "Issues": "https://github.com/syndicalt/zaxy/issues",
    }
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
    assert "build>=1.2.0" in dev_deps
    assert "twine>=5.0.0" in dev_deps
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/zaxy"]
    assert pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["artifacts"] == [
        "src/zaxy/py.typed"
    ]
    assert "site" in pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "zaxy_benchmarks" in pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]


def test_heavy_eval_modules_are_outside_runtime_package() -> None:
    """Benchmark/eval implementations should not ship in the runtime wheel package."""
    runtime_dir = Path("src/zaxy")
    eval_dir = Path("zaxy_benchmarks")
    moved_modules = {
        "benchmark.py",
        "causal_benchmark.py",
        "consolidation_benchmark.py",
        "coordination_benchmark.py",
        "harvey_lab_benchmark.py",
        "live_benchmark.py",
        "longmembench.py",
        "purpose_benchmark.py",
        "rc_benchmark_freeze.py",
        "reasoning_benchmark.py",
    }

    assert eval_dir.is_dir()
    assert (eval_dir / "__init__.py").exists()
    for module in moved_modules:
        assert not (runtime_dir / module).exists()
        assert (eval_dir / module).exists()


def test_dockerfile_defaults_to_production_environment() -> None:
    """Bare docker runs should use production safety checks unless explicitly overridden."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ENV ZAXY_ENV=production" in dockerfile


def test_core_install_includes_embedded_default_backend() -> None:
    """The embedded backend should be part of the core install, not duplicated in an extra."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    assert "ladybug==0.17.1" in dependencies  # exact pin: bus-factor-1 fork, see pyproject comment
    assert extras["embedded"] == []
    assert "neo4j>=5.20.0" not in dependencies
    assert extras["neo4j"] == ["neo4j>=5.20.0"]


def test_package_keywords_center_embedded_local_memory() -> None:
    """Package discovery metadata should match the embedded-first runtime."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    keywords = pyproject["project"]["keywords"]
    assert "embedded-memory" in keywords
    assert "ladybugdb" in keywords
    assert "local-first" in keywords
    assert "neo4j" not in keywords


def test_package_metadata_centers_coordinator_memory() -> None:
    """Package discovery metadata should match the v0.5 public positioning."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["description"] == (
        "Coordinator memory for auditable multi-agent projects"
    )
    keywords = pyproject["project"]["keywords"]
    assert "coordinator-memory" in keywords
    assert "multi-agent" in keywords
    assert "auditable-memory" in keywords


def test_core_install_excludes_unused_graphiti_abstraction() -> None:
    """Plain installs should not ship the unused Graphiti abstraction dependency."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    assert not any(dependency.startswith("graphiti-core") for dependency in dependencies)


def test_release_metadata_checks_report_actionable_file_errors(tmp_path: Path) -> None:
    """Release checks should fail with concrete remediation when required files are absent or invalid."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = ""\n',
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "publish.yml").write_text(
        "on:\n  push:\npermissions: {}\nsteps:\n  - run: echo missing release gate\n",
        encoding="utf-8",
    )

    version = _check_package_version(tmp_path)
    trusted = _check_trusted_publishing(tmp_path)
    workflow = _check_release_workflow(tmp_path)

    assert version["status"] == "error"
    assert "project.version" in version["message"]
    assert trusted["status"] == "error"
    assert "Trusted Publishing" in trusted["message"]
    assert workflow["status"] == "error"
    assert "release trigger" in workflow["message"]

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "2.0.0rc1"\n',
        encoding="utf-8",
    )
    changelog = _check_changelog(tmp_path)
    assert changelog["status"] == "error"
    assert "CHANGELOG.md is missing" in changelog["message"]


def test_release_json_example_check_reports_missing_failure_bad_json_and_bad_payload(
    tmp_path: Path,
) -> None:
    """Example smoke checks should distinguish missing, failed, non-JSON, and malformed payloads."""
    (tmp_path / "examples").mkdir()

    missing = _check_json_example(
        tmp_path,
        name="demo",
        relative_path="examples/missing.py",
        expected_session_id="demo-session",
        expected_kind={"memory_checkout"},
        success_message="ok",
    )
    assert missing["status"] == "error"
    assert "missing" in missing["message"]

    failed_path = tmp_path / "examples" / "failed.py"
    failed_path.write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n", encoding="utf-8")
    failed = _check_json_example(
        tmp_path,
        name="demo",
        relative_path="examples/failed.py",
        expected_session_id="demo-session",
        expected_kind={"memory_checkout"},
        success_message="ok",
    )
    assert failed["status"] == "error"
    assert "failed with exit 3" in failed["message"]

    bad_json_path = tmp_path / "examples" / "bad_json.py"
    bad_json_path.write_text("print('not-json')\n", encoding="utf-8")
    bad_json = _check_json_example(
        tmp_path,
        name="demo",
        relative_path="examples/bad_json.py",
        expected_session_id="demo-session",
        expected_kind={"memory_checkout"},
        success_message="ok",
    )
    assert bad_json["status"] == "error"
    assert "did not print JSON" in bad_json["message"]

    bad_payload_path = tmp_path / "examples" / "bad_payload.py"
    bad_payload_path.write_text(
        "import json\nprint(json.dumps({'session_id':'other','has_zaxy_context':False,'kind':'raw'}))\n",
        encoding="utf-8",
    )
    bad_payload = _check_json_example(
        tmp_path,
        name="demo",
        relative_path="examples/bad_payload.py",
        expected_session_id="demo-session",
        expected_kind={"memory_checkout"},
        success_message="ok",
    )
    assert bad_payload["status"] == "error"
    assert "unexpected smoke payload" in bad_payload["message"]


def test_release_smoke_gate_lists_failing_child_checks(tmp_path: Path) -> None:
    """Beta readiness should surface the specific release-smoke checks blocking publication."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "2.0.0rc1"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")

    result = _check_release_smoke_gate(tmp_path)

    assert result["status"] == "error"
    assert "changelog" in result["message"]
    assert "trusted_publishing" in result["message"]
    assert "release_workflow" in result["message"]
    assert "langgraph_example" in result["message"]


def test_backend_report_inputs_reports_unreadable_and_malformed_artifacts(
    tmp_path: Path,
) -> None:
    """Backend benchmark release gates should diagnose unreproducible archived inputs."""
    report_dir = tmp_path / "reports" / "backend-shootout"
    report_dir.mkdir(parents=True)
    for name in (
        "backend-shootout.json",
        "longmemeval-40-backend-shootout.json",
        "longmemeval-100-backend-shootout.json",
    ):
        (report_dir / name).write_text("{not-json", encoding="utf-8")

    unreadable = _check_backend_report_inputs(tmp_path)

    assert unreadable["status"] == "error"
    assert "backend-shootout.json is unreadable" in unreadable["message"]

    (report_dir / "backend-shootout.json").write_text(
        json.dumps(
            {
                "query_results": {"neo4j": "not-a-list", "pggraph": []},
                "eventloom_path": "",
                "queries_file": "missing-queries.json",
            }
        ),
        encoding="utf-8",
    )

    malformed = _check_backend_report_inputs(tmp_path)

    assert malformed["status"] == "error"
    assert "query_results neo4j must be a diagnostics list" in malformed["message"]
    assert "query_results pggraph has no diagnostics" in malformed["message"]
    assert "eventloom_path is missing" in malformed["message"]
    assert "queries_file missing-queries.json is missing" in malformed["message"]


def test_benchmark_no_regression_requires_release_script_guardrails(
    tmp_path: Path,
) -> None:
    """Release scripts must keep benchmark floors for quality, citations, tokens, and latency."""
    missing = _check_benchmark_no_regression(tmp_path)
    assert missing["status"] == "error"
    assert "release-check.sh is missing" in missing["message"]

    script_path = tmp_path / "scripts" / "release-check.sh"
    script_path.parent.mkdir()
    script_path.write_text(
        "BACKEND_SHOOTOUT_CMD='python -m zaxy backend-shootout'\n"
        "BACKEND_PERFORMANCE_CMD='python -m zaxy backend-performance --min-citation-coverage 1.0'\n"
        "BACKEND_SCALE_CMD='python -m zaxy backend-scale --min-recall-at-5'\n",
        encoding="utf-8",
    )

    incomplete = _check_benchmark_no_regression(tmp_path)

    assert incomplete["status"] == "error"
    assert "BACKEND_SHOOTOUT_CMD must include --min-answer-at-5" in incomplete["message"]
    assert "BACKEND_PERFORMANCE_CMD must include --max-checkout-p99-ms" in incomplete["message"]
    assert "BACKEND_SCALE_CMD must include --min-quality-per-1k-returned-tokens" in incomplete["message"]


def test_purpose_evidence_policy_fixture_reports_non_actionable_policy_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Release gating should fail if purpose evidence policies stop blocking unsupported claims."""

    def non_actionable_policy(**_kwargs):
        return SimpleNamespace(
            satisfied=True,
            missing_requirements=[],
            mode="noop",
            suggested_queries=[],
        )

    monkeypatch.setattr(
        "zaxy.evidence.evaluate_evidence_policy",
        non_actionable_policy,
    )

    result = _check_purpose_evidence_policy_fixture(tmp_path)

    assert result["status"] == "error"
    assert "security unsupported fixture unexpectedly satisfied policy" in result["message"]
    assert "security missing requirements [] did not include ['mitigation_or_risk_owner']" in result["message"]
    assert "security policy mode 'noop' is not actionable" in result["message"]
    assert "security policy did not emit suggested refresh queries" in result["message"]


def test_release_gate_surface_coverage_requires_all_public_gate_commands(
    tmp_path: Path,
) -> None:
    """Release scripts should run each public gate surface or skip it with an explicit reason."""
    missing = _check_release_gate_surface_coverage(tmp_path)
    assert missing["status"] == "error"
    assert "release-check.sh is missing" in missing["message"]

    script_path = tmp_path / "scripts" / "release-check.sh"
    script_path.parent.mkdir()
    script_path.write_text(
        "run_gate() { :; }\n"
        "EXAMPLES_SMOKE_CMD='SKIP:'\n"
        "MCP_SMOKE_CMD='pytest tests/test_mcp.py'\n"
        "LANGGRAPH_SMOKE_CMD='pytest test_langgraph_example_runs_without_langgraph_dependency'\n"
        "COORDINATE_SMOKE_CMD='pytest test_coordinate_three_worker_example_runs'\n"
        "BACKEND_SHOOTOUT_CMD='python scripts/check-backend-shootout.py'\n"
        "DOCS_CMD='scripts/validate-docs.sh'\n"
        "BETA_UAT_CMD='scripts/beta-uat.sh'\n"
        "EXTERNAL_VALIDATION_CMD='SKIP: outside validation collected after release'\n",
        encoding="utf-8",
    )

    incomplete = _check_release_gate_surface_coverage(tmp_path)

    assert incomplete["status"] == "error"
    assert "EXAMPLES_SMOKE_CMD SKIP must include a reason" in incomplete["message"]
    assert "MCP_SMOKE_CMD must include scripts/mcp_smoke_test.py" in incomplete["message"]


def test_external_validation_evidence_distinguishes_optional_required_and_invalid_reports(
    tmp_path: Path,
) -> None:
    """External validation gates should avoid overclaiming absent or malformed outside evidence."""
    optional = _check_external_validation_evidence(tmp_path)
    assert optional["status"] == "ok"
    assert "optional" in optional["message"]

    required = _check_external_validation_evidence(tmp_path, required=True)
    assert required["status"] == "error"
    assert "external validation is required" in required["message"]

    requested = _check_external_validation_evidence(
        tmp_path,
        external_validation_report="reports/external-validation/custom.json",
    )
    assert requested["status"] == "error"
    assert "external validation report was requested" in requested["message"]

    report_path = tmp_path / "reports" / "external-validation" / "custom.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("not-json", encoding="utf-8")
    bad_json = _check_external_validation_evidence(
        tmp_path,
        external_validation_report=report_path,
    )
    assert bad_json["status"] == "error"
    assert "unreadable or invalid JSON" in bad_json["message"]

    report_path.write_text(json.dumps({"report": "missing required fields"}), encoding="utf-8")
    invalid = _check_external_validation_evidence(
        tmp_path,
        external_validation_report=report_path,
    )
    assert invalid["status"] == "error"
    assert "external validation report is invalid" in invalid["message"]


def test_activation_release_fixture_requires_checkout_then_high_context_event(tmp_path: Path) -> None:
    """Release activation fixtures should prove memory checkout before high-context work."""
    missing = _check_activation_release_fixture(tmp_path)
    assert missing["status"] == "error"
    assert "no checked Eventloom JSONL" in missing["message"]

    fixture = tmp_path / "reports" / "activation-release"
    fixture.mkdir(parents=True)
    log = EventLog(fixture / "agent.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy",
        thread="agent",
        payload={
            "token_efficiency": {
                "prompt_tokens": 1200,
                "facts_per_1k_prompt_tokens": 0.5,
            }
        },
        timestamp=ACTIVATION_FIXTURE_NOW.isoformat(),
    )

    no_work = _check_activation_release_fixture(tmp_path)
    assert no_work["status"] == "error"
    assert "no high-context event after checkout" in no_work["message"]

    log.append(
        "command.completed",
        actor="codex",
        thread="agent",
        payload={"cmd": "pytest tests/test_packaging.py"},
        timestamp=ACTIVATION_FIXTURE_NOW.isoformat(),
    )

    ok = _check_activation_release_fixture(tmp_path)
    assert ok["status"] == "ok"
    assert "fresh checkout" in ok["message"]


def test_activation_release_fixture_reports_token_efficiency_and_freshness_errors(
    tmp_path: Path,
) -> None:
    """Activation fixtures should enforce token budget, density, and freshness guardrails."""
    assert _has_token_efficiency({"token_efficiency": {"prompt_tokens": 1, "facts_per_1k_prompt_tokens": 0.2}})
    assert not _has_token_efficiency({"token_efficiency": {"prompt_tokens": True, "facts_per_1k_prompt_tokens": 0.2}})
    assert _activation_token_efficiency_errors(
        {"token_efficiency": {"prompt_tokens": 6000, "facts_per_1k_prompt_tokens": 0.01}}
    ) == [
        "prompt_tokens=6000 exceeds 5000",
        "facts_per_1k_prompt_tokens=0.01 is below 0.1",
    ]
    assert _activation_checkout_freshness_errors("2026-05-20T11:30:00+00:00") == []
    stale_errors = _activation_checkout_freshness_errors("2026-05-20T08:00:00+00:00")
    assert stale_errors == ["checkout age 240.0 minutes exceeds 120 minutes"]

    fixture = tmp_path / "reports" / "activation-release"
    fixture.mkdir(parents=True)
    log = EventLog(fixture / "agent.jsonl")
    log.append(
        "memory.checkout.completed",
        actor="zaxy",
        thread="agent",
        payload={
            "token_efficiency": {
                "prompt_tokens": 6000,
                "facts_per_1k_prompt_tokens": 0.01,
            }
        },
        timestamp="2026-05-20T08:00:00+00:00",
    )
    log.append(
        "file.edit.applied",
        actor="codex",
        thread="agent",
        payload={"path": "src/zaxy/release.py"},
        timestamp=ACTIVATION_FIXTURE_NOW.isoformat(),
    )

    result = _check_activation_release_fixture(tmp_path)

    assert result["status"] == "error"
    assert "prompt_tokens=6000 exceeds 5000" in result["message"]
    assert "facts_per_1k_prompt_tokens=0.01 is below 0.1" in result["message"]


def test_gitignore_keeps_backend_diagnostics_scratch_out_of_release_inputs() -> None:
    """Generated diagnostics are scratch, but target query inputs must stay visible."""
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "reports/backend-shootout/*-diagnostics.json" in gitignore
    assert "reports/backend-shootout/*-diagnostics.md" in gitignore
    assert "reports/backend-shootout/*-target-diagnostics.json" in gitignore
    assert "reports/backend-shootout/*-target-diagnostics.md" in gitignore
    assert "reports/backend-shootout/*-queries-with-targets.json" not in gitignore


def test_pathlight_observability_is_opt_in_extra() -> None:
    """Plain installs should not require optional tracing infrastructure."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    assert "pathlight>=0.1.0" not in dependencies
    assert extras["pathlight"] == ["pathlight>=0.1.0"]


def test_neo4j_driver_imports_are_lazy_for_embedded_default() -> None:
    """Importing default runtime modules should not require the optional Neo4j driver."""
    graph_source = Path("src/zaxy/graph.py").read_text(encoding="utf-8")
    dashboard_source = Path("src/zaxy/dashboard.py").read_text(encoding="utf-8")
    live_benchmark_source = Path("zaxy_benchmarks/live_benchmark.py").read_text(encoding="utf-8")

    assert "from neo4j import" not in graph_source
    assert "from neo4j import" not in dashboard_source
    assert "from neo4j import" not in live_benchmark_source


def test_projection_backend_factory_does_not_import_neo4j_store_until_selected() -> None:
    """Importing the backend factory should not load the optional Neo4j store path."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import zaxy.projection_backends; print('zaxy.graph' in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_projection_backend_factory_avoids_event_model_import_until_needed() -> None:
    """Lightweight backend imports should not load Eventloom/Pydantic release gates."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zaxy.projection_backends; "
                "print('zaxy.event' in sys.modules); "
                "print('pydantic' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["False", "False"]


def test_projection_backend_factory_avoids_release_metadata_until_needed() -> None:
    """Submodule imports should not read package metadata for lazy __version__."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zaxy.projection_backends; "
                "print('zaxy.release' in sys.modules); "
                "from zaxy import __version__; "
                "print(bool(__version__)); "
                "print('zaxy.release' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["False", "True", "True"]


def test_package_lazy_exports_preserve_public_import_compatibility() -> None:
    """Lazy package exports should still support established public imports."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import zaxy; "
                "from zaxy import MemoryFabric, render_mcp_client_config; "
                "print(MemoryFabric.__name__); "
                "print(callable(render_mcp_client_config)); "
                "print('MemoryFabric' in dir(zaxy))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["MemoryFabric", "True", "True"]


def test_cli_version_exits_before_loading_command_graph() -> None:
    """The version path should avoid importing optional command subsystems."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys\n"
                "sys.argv = ['python -m zaxy', '--version']\n"
                "try:\n"
                "    runpy.run_module('zaxy.__main__', run_name='__main__')\n"
                "except SystemExit as exc:\n"
                "    print(f'exit={exc.code}')\n"
                "print('zaxy.mcp_server' in sys.modules)\n"
                "print('zaxy.graph' in sys.modules)\n"
                "print('typer' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [f"zaxy {package_version()}", "exit=0", "False", "False", "False"]


def test_cli_help_avoids_mcp_server_stack_until_serve_runs() -> None:
    """Rendering CLI help should not load command-only Zaxy subsystems."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import contextlib, io, runpy, sys\n"
                "sys.argv = ['python -m zaxy', '--help']\n"
                "with contextlib.redirect_stdout(io.StringIO()):\n"
                "    try:\n"
                "        runpy.run_module('zaxy.__main__', run_name='__main__')\n"
                "    except SystemExit:\n"
                "        pass\n"
                "loaded = sorted(\n"
                "    name\n"
                "    for name in sys.modules\n"
                "    if name.startswith('zaxy.')\n"
                "    and name != 'zaxy.__main__'\n"
                "    and not name.startswith('zaxy.cli')\n"
                ")\n"
                "print(loaded)\n"
                "print('mcp' in sys.modules)\n"
                "print('uvicorn' in sys.modules)\n"
                "print('sse_starlette' in sys.modules)\n"
                "print('httpx' in sys.modules)\n"
                "print('pydantic_settings' in sys.modules)\n"
                "print('tomllib' in sys.modules)\n"
                "print('tomlkit' in sys.modules)\n"
                "print('yaml' in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["[]", "False", "False", "False", "False", "False", "False", "False", "False"]


def test_mypy_config_does_not_keep_stale_optional_driver_overrides() -> None:
    """Lazy-loaded optional drivers should not leave unused mypy overrides behind."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    override_modules = {
        module
        for override in pyproject["tool"]["mypy"].get("overrides", [])
        for module in override["module"]
    }
    assert "neo4j.*" not in override_modules
    assert "mcp.*" not in override_modules
    assert "pathlight.*" not in override_modules
    assert "psycopg.*" not in override_modules


def test_package_version_source_fallback_is_independent_of_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Source-tree version fallback should not read pyproject.toml from the caller cwd."""
    from zaxy import release

    def missing_distribution(_name: str) -> str:
        raise release.metadata.PackageNotFoundError("zaxy-memory")

    monkeypatch.setattr(release.metadata, "version", missing_distribution)
    monkeypatch.chdir(tmp_path)

    assert package_version() == pyproject_version(Path(__file__).resolve().parents[1])


def test_package_version_prefers_source_tree_version_in_editable_checkout(monkeypatch) -> None:
    """Editable installs should not report stale metadata after a release version bump."""
    from zaxy import release

    monkeypatch.setattr(release.metadata, "version", lambda _name: "0.1.0")

    assert package_version() == pyproject_version(Path(__file__).resolve().parents[1])


def test_changelog_records_initial_pypi_release() -> None:
    """Public releases should have a user-facing changelog entry."""
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "# Changelog" in changelog
    assert "## 0.4.0 - 2026-05-28" in changelog
    assert "Zaxy Coordinate" in changelog
    assert "CoordinationBench" in changelog
    assert "## 0.3.1 - 2026-05-19" in changelog
    assert "## 0.3.0 - 2026-05-19" in changelog
    assert "## 0.2.3 - 2026-05-18" in changelog
    assert "## 0.2.2 - 2026-05-18" in changelog
    assert "## 0.2.0 - 2026-05-15" in changelog
    assert "default `pip install zaxy-memory`" in changelog
    assert "## 0.2.0b1 - 2026-05-15" in changelog
    assert "Answer@5 0.950" in changelog
    assert "## 0.1.0 - 2026-05-11" in changelog
    assert "PyPI" in changelog
    assert "Trusted Publishing" in changelog


def test_changelog_covers_release_candidate_path_from_04_to_10() -> None:
    """The v1.0 roadmap should have a comprehensive changelog path from 0.4 to 1.0."""
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    for heading in (
        "## 1.1.0 - 2026-06-05",
        "## 1.0.4 - 2026-06-05",
        "## 1.0.3 - 2026-06-04",
        "## 1.0.2 - 2026-06-02",
        "## 1.0.1 - 2026-05-31",
        "## 1.0.0 - 2026-05-31",
        "## 0.9.0 - Release Candidate",
        "## 0.8.0 - Unreleased",
        "## 0.7.0 - Unreleased",
        "## 0.6.0 - Unreleased",
        "## 0.5.0 - Unreleased",
        "## 0.4.0 - 2026-05-28",
    ):
        assert heading in changelog
    for required in (
        "stability commitment",
        "schema-freeze",
        "release validation checklist",
        "external validation",
        "API inventory",
        "Migration guide",
        "OpenAI-compatible",
        "Coordinate mission",
        "StateRecoveryBench",
    ):
        assert required in changelog


def test_pyproject_declares_optional_framework_extras() -> None:
    """Framework dependencies should be opt-in extras rather than core requirements."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["langgraph"] == ["langgraph>=0.6"]
    assert extras["crewai"] == ["crewai>=0.100"]
    assert extras["autogen"] == ["autogen-agentchat>=0.7"]
    assert extras["frameworks"] == [
        "langgraph>=0.6",
        "crewai>=0.100",
        "autogen-agentchat>=0.7",
    ]
    for dependency in extras["frameworks"]:
        assert dependency not in pyproject["project"]["dependencies"]


def test_pytest_default_options_exclude_docker_integration_tests() -> None:
    """Plain pytest should not require local Neo4j Docker services."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    addopts = pyproject["tool"]["pytest"]["ini_options"]["addopts"]

    assert "not integration" in addopts


def test_build_dist_runs_build_and_twine_check_in_order(tmp_path: Path) -> None:
    """Distribution builds should create both artifacts and validate metadata."""
    log_path = tmp_path / "commands.log"
    root = tmp_path / "project"
    dist = tmp_path / "dist"
    root.mkdir()
    dist.mkdir()
    stale_artifact = dist / "zaxy_memory-0.1.0-py3-none-any.whl"
    stale_artifact.write_text("stale", encoding="utf-8")
    build_stub = tmp_path / "build.sh"
    twine_stub = tmp_path / "twine.sh"
    build_stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"build $*\" >> \"$PACKAGE_CHECK_LOG\"\n",
        encoding="utf-8",
    )
    twine_stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"twine $*\" >> \"$PACKAGE_CHECK_LOG\"\n",
        encoding="utf-8",
    )
    build_stub.chmod(0o700)
    twine_stub.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            "scripts/build-dist.sh",
            "--root",
            str(root),
            "--dist-dir",
            str(dist),
            "--build-cmd",
            str(build_stub),
            "--twine-cmd",
            str(twine_stub),
        ],
        cwd=Path.cwd(),
        check=True,
        text=True,
        capture_output=True,
        env={"PACKAGE_CHECK_LOG": str(log_path)},
    )

    assert "Package artifacts passed" in result.stdout
    assert not stale_artifact.exists()
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        f"build --sdist --wheel --outdir {dist} {root}",
        f"twine check {dist}/*",
    ]


def test_beta_uat_script_exercises_clean_repo_happy_path() -> None:
    """The beta UAT script should cover install, init, bootstrap, capture, and checkout."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "mktemp -d" in script
    assert "python -m pip install" in script
    assert "zaxy init" in script
    assert 'run_workspace "codex" "local-codex" "start"' in script
    assert 'run_workspace "claude-code" "local-claude" "none"' in script
    assert "zaxy memory bootstrap" in script
    assert "zaxy memory checkout" in script
    assert "zaxy doctor" in script
    assert "zaxy hook-status" in script
    assert "zaxy capture status" in script
    assert "zaxy capture soak" in script
    assert "zaxy memory status" in script


def test_beta_uat_script_exercises_bare_embedded_init_path() -> None:
    """UAT should protect bare init as the no-sidecar embedded default."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert 'run_workspace "embedded" "" "start"' in script
    assert '"${preset}"' in script
    assert "if [[ -n \"${preset}\" ]]" in script
    assert "grep -q \"PROJECTION_BACKEND=embedded\" .env.local" in script
    assert "grep -q \"NEO4J_AUTO_START=false\" .env.local" in script
    assert "grep -q \"EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu\" .env.local" in script
    assert "zaxy memory status --eventloom-path .eventloom --graph" in script
    assert "grep -q \"Graph projection (backend=embedded):\"" in script
    assert "zaxy memory inferred-status --session-id" in script
    assert "grep -q '\"backend\": \"embedded\"'" in script
    assert "zaxy reproject" in script
    assert "grep -q \"using embedded\"" in script


def test_beta_uat_script_uses_unique_default_domain_per_run() -> None:
    """Repeated UAT runs should not reuse the same Eventloom session in Neo4j."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert 'basename "${WORKDIR}" | tr' in script
    assert 'BASE_DOMAIN="${ZAXY_BETA_DOMAIN:-zaxy-beta-uat-${RUN_ID}}"' in script
    assert 'local domain="${BASE_DOMAIN}-${label}"' in script
    assert 'local session_id="${domain}-default"' in script


def test_beta_uat_script_fails_when_checkout_has_no_memory() -> None:
    """UAT should not pass if checkout cannot retrieve cited first-run memory."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "CHECKOUT_OUTPUT=\"$(zaxy memory checkout" in script
    assert "grep -q \"Answerability: answer_from_memory\"" in script
    assert "grep -Eq \"Citations: [1-9]\"" in script


def test_beta_uat_script_verifies_model_facing_memory_guidance() -> None:
    """UAT should prove bootstrap and checkout teach models when to use memory."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "BOOTSTRAP_OUTPUT=\"$(zaxy memory bootstrap" in script
    assert "grep -q \"Call memory_checkout before answering roadmap or implementation questions.\"" in script
    assert "grep -q \"Call memory_feedback when cited checkout context was used.\"" in script
    assert "grep -q \"Feedback: call memory_feedback\" <<<\"${CHECKOUT_OUTPUT}\"" in script
    assert "grep -q \"Suggested next call: memory_checkout\" <<<\"${CHECKOUT_OUTPUT}\"" in script


def test_beta_uat_script_exercises_memory_persistence_boundaries() -> None:
    """UAT should simulate long, resumed, compacted, and roadmap-question memory reminders."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "zaxy hook-event session-start" in script
    assert "zaxy hook-event checkpoint" in script
    assert "--turn-count 12" in script
    assert "--reason resume" in script
    assert "zaxy hook-event precompact" in script
    assert "what is left on the roadmap" in script
    assert "grep -q \"memory.reminder.suggested\"" in script


def test_beta_uat_script_exercises_observation_sinks_for_capture_soak() -> None:
    """UAT should prove the hook protocol lanes that make model memory observable."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "zaxy hook-event command" in script
    assert "zaxy hook-event file-edit" in script
    assert "zaxy hook-event tool-call" in script
    assert "zaxy hook-event transcript-turn" in script
    assert "zaxy capture soak --eventloom-path .eventloom --workspace-root . --session-id" in script


def test_beta_uat_script_enforces_activation_efficiency_guardrail() -> None:
    """UAT should fail if clean first-run sessions are captured without fresh checkout."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "zaxy hook-status --eventloom-path .eventloom --min-activation-rate 1.0" in script
    assert "--max-checkout-prompt-tokens 5000" in script
    assert "--min-checkout-facts-per-1k-tokens 0.1" in script
    assert script.index("--min-activation-rate 1.0") < script.index("zaxy capture soak")


def test_beta_uat_script_stops_managed_capture_before_cleanup() -> None:
    """UAT should not leave its managed capture watcher running after success."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "zaxy capture stop --workspace \"${PROJECT}\" >/dev/null 2>&1 || true" in script
    assert script.index("zaxy capture stop --workspace \"${PROJECT}\"") < script.index("rm -rf \"${WORKDIR}\"")
    assert "zaxy capture stop --workspace ." in script
    assert script.index("zaxy capture stop --workspace .") < script.index("popd >/dev/null")


def test_beta_readiness_requires_maintained_beta_roadmap(tmp_path: Path) -> None:
    """Beta readiness should fail when the repo has no explicit beta roadmap artifact."""
    _write_minimal_beta_ready_project(tmp_path)

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["beta_roadmap_claims"]["status"] == "error"
    assert checks["beta_roadmap_claims"]["action"] == "Add BETA.md with beta goals, remaining work, gates, and exit criteria."


def _write_roadmap(root: Path, body: str) -> Path:
    """Write a BETA.md fixture whose claims the roadmap gate can resolve."""
    path = root / "BETA.md"
    path.write_text(body, encoding="utf-8")
    return path


_ROADMAP_LANES = "\n".join(f"- `--workload {lane}` lane" for lane in BETA_BENCHMARK_LANES)
_VALID_ROADMAP = (
    f"{_ROADMAP_LANES}\n"
    "- `zaxy benchmark-inventory` inventories the lanes.\n"
    "- `zaxy capture soak` reports capture coverage.\n"
    "- `zaxy memory inferred-status` reports inferred edges.\n"
    "- CrewAI and LangGraph adapters are maintained.\n"
)


def test_beta_roadmap_claims_pass_against_the_live_repository() -> None:
    """The shipped BETA.md's lane, command, and adapter claims resolve against the live code."""
    result = _check_beta_roadmap_claims(Path("."))

    assert result["status"] == "ok", result["message"]


def test_beta_roadmap_claims_accept_a_roadmap_whose_claims_all_resolve(tmp_path: Path) -> None:
    """A roadmap naming only real lanes, canonical commands, and real adapters passes."""
    _write_roadmap(tmp_path, _VALID_ROADMAP)

    assert _check_beta_roadmap_claims(tmp_path)["status"] == "ok"


def test_beta_roadmap_claims_reject_an_unregistered_workload_lane(tmp_path: Path) -> None:
    """A lane BETA.md claims but the benchmark CLI does not accept turns the gate red."""
    _write_roadmap(tmp_path, _VALID_ROADMAP.replace("--workload source-recall", "--workload sources"))

    result = _check_beta_roadmap_claims(tmp_path)

    assert result["status"] == "error"
    assert "`--workload source-recall`" in result["message"]
    assert "BETA_BENCHMARK_LANES" not in result["message"]


def test_beta_roadmap_claims_reject_a_dropped_lane_claim(tmp_path: Path) -> None:
    """Deleting a lane claim from BETA.md cannot make the gate vacuously pass."""
    _write_roadmap(tmp_path, _VALID_ROADMAP.replace("- `--workload graph-traversal` lane\n", ""))

    result = _check_beta_roadmap_claims(tmp_path)

    assert result["status"] == "error"
    assert "no longer claims the `--workload graph-traversal` lane" in result["message"]


def test_beta_roadmap_claims_reject_a_command_that_does_not_exist(tmp_path: Path) -> None:
    """A `zaxy ...` command BETA.md cites that the Typer tree cannot resolve turns the gate red."""
    _write_roadmap(tmp_path, _VALID_ROADMAP.replace("zaxy benchmark-inventory", "zaxy benchmark-manifest"))

    result = _check_beta_roadmap_claims(tmp_path)

    assert result["status"] == "error"
    assert "cites `zaxy benchmark-manifest` but no such command exists" in result["message"]


def test_beta_roadmap_claims_reject_a_deprecated_command_alias(tmp_path: Path) -> None:
    """Citing a deprecated flat alias instead of its canonical grouped form turns the gate red."""
    _write_roadmap(tmp_path, _VALID_ROADMAP.replace("zaxy capture soak", "zaxy capture-soak"))

    result = _check_beta_roadmap_claims(tmp_path)

    assert result["status"] == "error"
    assert "deprecated alias `zaxy capture-soak`" in result["message"]
    assert "use `zaxy capture soak`" in result["message"]


def test_beta_roadmap_claims_reject_an_adapter_that_cannot_import(tmp_path: Path) -> None:
    """An adapter BETA.md claims that fails to import turns the gate red naming the module."""
    _write_roadmap(tmp_path, _VALID_ROADMAP)

    import zaxy.release as release_module

    original = dict(release_module.BETA_ADAPTER_MODULES)
    release_module.BETA_ADAPTER_MODULES["CrewAI"] = "zaxy.adapters.not_an_adapter"
    try:
        result = _check_beta_roadmap_claims(tmp_path)
    finally:
        release_module.BETA_ADAPTER_MODULES.clear()
        release_module.BETA_ADAPTER_MODULES.update(original)

    assert result["status"] == "error"
    assert "zaxy.adapters.not_an_adapter" in result["message"]


def test_beta_cli_claims_extracts_grouped_and_flat_command_paths() -> None:
    """CLI claim extraction yields at most two words per backticked `zaxy ...` reference."""
    claims = _beta_cli_claims("`zaxy capture soak` and `zaxy doctor --beta-readiness`")

    assert claims == ["capture soak", "doctor"]


def test_beta_readiness_reports_missing_clean_repo_uat(tmp_path: Path) -> None:
    """Beta readiness should fail clearly when the clean-repo UAT harness is absent."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "examples").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.2.0 - 2026-05-11\n\n- Stable release.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "examples").mkdir()
    (tmp_path / "docs" / "examples" / "first-run-timing-report.json").write_text(
        json.dumps(
            {
                "threshold_seconds": 300,
                "time_to_successful_doctor_seconds": 240,
                "time_to_first_successful_example_seconds": 270,
                "requires_sidecar": False,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows" / "publish.yml").write_text(
        "on:\n"
        "  release:\n"
        "    types: [published]\n"
        "  workflow_dispatch:\n"
        "permissions:\n"
        "  id-token: write\n"
        "steps:\n"
        "  - run: python -m build --sdist --wheel\n"
        "  - run: python -m twine check dist/*\n"
        "  - uses: pypa/gh-action-pypi-publish@release/v1\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "langgraph_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'langgraph-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "openai_compatible_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'openai-compatible-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "claude_compatible_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'claude-compatible-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "release-check.sh").write_text(
        'RUFF_CMD="ruff"\n'
        'MYPY_CMD="mypy"\n'
        'EXAMPLES_SMOKE_CMD="pytest tests/test_examples_v05.py --no-cov -q"\n'
        'MCP_SMOKE_CMD="python scripts/mcp_smoke_test.py"\n'
        'LANGGRAPH_SMOKE_CMD="pytest tests/test_examples_v05.py::test_langgraph_example_runs_without_langgraph_dependency --no-cov -q"\n'
        'COORDINATE_SMOKE_CMD="pytest tests/test_examples_v05.py::test_coordinate_three_worker_example_runs --no-cov -q"\n'
        'DOCS_CMD="scripts/validate-docs.sh"\n'
        'BETA_UAT_CMD="scripts/beta-uat.sh"\n'
        'EXTERNAL_VALIDATION_CMD="SKIP:external validation is optional for v1.0 release"\n'
        "run_gate() { [[ \"$2\" == SKIP:* ]] && echo \"Skipping $1: ${2#SKIP:}\" || bash -c \"$2\"; }\n"
        "pytest\n"
        "scripts/check-coverage.py\n"
        "tests/test_packet_memory_e2e.py\n"
        "scripts/build-dist.sh\n"
        "scripts/validate-docs.sh\n"
        "scripts/validate-deployment.sh\n"
        "PYTHONPATH=src python -m zaxy hook-status\n"
        "--eventloom-path reports/activation-release\n"
        "--now 2026-05-20T12:00:00+00:00\n"
        "--min-activation-rate 1.0\n"
        "--max-checkout-prompt-tokens 5000\n"
        "--min-checkout-facts-per-1k-tokens 0.1\n"
        'BACKEND_SHOOTOUT_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/backend-shootout.json '
        "--require-report-metadata --require-markdown-report --require-query-results --require-git-tracked-inputs "
        "--verify-report-fingerprints --require-backends embedded,bm25 "
        '--require-labeled-metrics --require-dashboard-source embedded=embedded '
        "--min-answer-at-5 0.5 --min-recall-at-5 0.5 --min-citation-coverage 1.0 "
        "--min-quality-per-1k-injected-tokens embedded=1.0 "
        "--min-answer-at-5-per-1k-injected-tokens embedded=1.0 "
        "--max-checkout-p99-ms embedded=25 "
        '--forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_PERFORMANCE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-40-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        "--require-dashboard-source embedded=embedded --min-citation-coverage 1.0 "
        "--min-quality-per-1k-returned-tokens embedded=0.10 "
        "--min-answer-at-5-per-1k-returned-tokens embedded=0.10 "
        "--min-quality-per-1k-injected-tokens embedded=0.10 "
        "--min-answer-at-5-per-1k-injected-tokens embedded=0.10 "
        "--max-checkout-p95-ms embedded=100 --max-checkout-p99-ms embedded=85 "
        '--forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_SCALE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-100-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        '--require-dashboard-source embedded=embedded --forbid-backends neo4j,pggraph,latticedb '
        "--min-recall-at-5 0.90 --min-citation-coverage 1.0 "
        "--min-quality-per-1k-returned-tokens embedded=0.15 "
        "--min-answer-at-5-per-1k-returned-tokens embedded=0.15 "
        "--min-quality-per-1k-injected-tokens embedded=0.15 "
        "--min-answer-at-5-per-1k-injected-tokens embedded=0.15 "
        "--max-checkout-p99-ms embedded=250 "
        '--max-checkout-p95-ms embedded=200"\n'
        "--require-report-metadata\n"
        "--require-markdown-report\n"
        "--require-query-results\n"
        "--require-git-tracked-inputs\n"
        "--verify-report-fingerprints\n"
        "backend-shootout.json\n"
        "longmemeval-40-backend-shootout.json\n"
        "longmemeval-100-backend-shootout.json\n"
        "--min-quality-per-1k-injected-tokens embedded=1.0\n"
        "--min-citation-coverage 1.0\n"
        "--min-quality-per-1k-returned-tokens\n"
        "--min-answer-at-5-per-1k-returned-tokens\n"
        "--min-quality-per-1k-injected-tokens\n"
        "--min-answer-at-5-per-1k-injected-tokens\n"
        "--max-cold-bootstrap-ms\n"
        "--max-first-checkout-ms\n"
        "--max-append-to-projection-p95-ms\n"
        "--max-resident-memory-delta-bytes\n"
        "--max-on-disk-footprint-bytes\n"
        "--max-dashboard-graph-load-ms\n"
        "--max-checkout-p95-ms embedded=200\n"
        "--max-checkout-p99-ms\n"
        "--max-exact-p99-ms\n"
        "--max-keyword-p95-ms\n"
        "--max-keyword-p99-ms\n"
        "--max-vector-p99-ms\n"
        "--max-traversal-p99-ms\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_smoke"]["status"] == "ok"
    assert checks["first_run_timing"]["status"] == "ok"
    assert checks["release_gate"]["status"] == "ok"
    assert "backend shootout" in checks["release_gate"]["message"]
    assert "100-query scale" in checks["release_gate"]["message"]
    assert "optional backend exclusion" in checks["release_gate"]["message"]
    assert checks["clean_repo_uat"]["status"] == "error"
    assert checks["clean_repo_uat"]["action"] == (
        "Add a clean-repo UAT script for install, init, bootstrap, capture, and checkout."
    )


def test_beta_readiness_requires_activation_efficiency_guardrail(tmp_path: Path) -> None:
    """Beta readiness should reject UAT scripts that observe work without enforcing memory activation."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "beta-uat.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "beta-uat.sh").write_text(
        script.replace(" --min-activation-rate 1.0", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["clean_repo_uat"]["status"] == "error"
    assert "--min-activation-rate 1.0" in checks["clean_repo_uat"]["message"]
    assert checks["clean_repo_uat"]["action"] == (
        "Update scripts/beta-uat.sh to exercise the complete first-run beta path."
    )


def test_beta_readiness_requires_coordination_competitor_claim_gate(tmp_path: Path) -> None:
    """Beta readiness should reject stale CoordinationBench competitor claim artifacts."""
    _write_minimal_beta_ready_project(tmp_path)
    report_path = tmp_path / "reports" / "benchmarks" / "coordination-real-v1" / "coordination-benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("competitor_claim_gate")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    readiness = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "error"
    assert checks["coordination_competitor_claims"]["status"] == "error"
    assert "missing competitor_claim_gate" in checks["coordination_competitor_claims"]["message"]
    assert checks["coordination_competitor_claims"]["action"] == (
        "Regenerate the CoordinationBench report with the public claim gate."
    )


def test_beta_readiness_rejects_unscored_quarq_same_harness_claim(tmp_path: Path) -> None:
    """Same-harness competitor claims should require local metrics and result audit."""
    _write_minimal_beta_ready_project(tmp_path)
    report_path = tmp_path / "reports" / "benchmarks" / "coordination-real-v1" / "coordination-benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["competitor_claim_gate"] = {
        "status": "passed",
        "required_adapters": ["quarq", "hybi"],
        "completed_adapters": ["quarq", "hybi"],
        "blocked_adapters": {},
        "message": "claims passed",
    }
    report["competitor_adapters"]["quarq"] = {
        **report["competitor_adapters"]["quarq"],
        "status": "completed",
        "claim_status": "same_harness",
        "metrics": None,
        "result_audit": None,
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    readiness = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "error"
    assert checks["coordination_competitor_claims"]["status"] == "error"
    assert "quarq completed row is missing locally scored metrics" in checks["coordination_competitor_claims"]["message"]
    assert "quarq completed row is missing result audit" in checks["coordination_competitor_claims"]["message"]


def test_beta_readiness_requires_purpose_benchmark_gate(tmp_path: Path) -> None:
    """Purpose-memory claims should require an archived passing purpose-v1 report."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "reports" / "benchmarks" / "purpose-v1" / "purpose-benchmark.json").unlink()

    readiness = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in readiness["checks"]}
    assert readiness["status"] == "error"
    assert checks["purpose_benchmark_gate"]["status"] == "error"
    assert "Purpose benchmark artifacts are missing" in checks["purpose_benchmark_gate"]["message"]
    assert checks["purpose_benchmark_gate"]["action"] == (
        "Run python -m zaxy purpose-benchmark --output-dir reports/benchmarks/purpose-v1."
    )


def test_beta_readiness_rejects_failing_purpose_benchmark_lane(tmp_path: Path) -> None:
    """The purpose-v1 gate should fail release readiness when any lane fails."""
    _write_minimal_beta_ready_project(tmp_path)
    report_path = tmp_path / "reports" / "benchmarks" / "purpose-v1" / "purpose-benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report["lanes"][0]["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    checks = {check["name"]: check for check in run_beta_readiness(project_root=tmp_path)["checks"]}

    assert checks["purpose_benchmark_gate"]["status"] == "error"
    assert "Purpose benchmark gate is unsafe" in checks["purpose_benchmark_gate"]["message"]
    assert "failing lanes: Purpose Recall" in checks["purpose_benchmark_gate"]["message"]


def test_beta_readiness_rejects_empty_evidence_policy_fixture_evidence(tmp_path: Path) -> None:
    """Evidence Policy Discipline must include archived fixture evidence."""
    _write_minimal_beta_ready_project(tmp_path)
    report_path = tmp_path / "reports" / "benchmarks" / "purpose-v1" / "purpose-benchmark.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for lane in report["lanes"]:
        if lane["name"] == "Evidence Policy Discipline":
            lane["evidence"] = {}
            break
    report_path.write_text(json.dumps(report), encoding="utf-8")

    checks = {check["name"]: check for check in run_beta_readiness(project_root=tmp_path)["checks"]}

    assert checks["purpose_benchmark_gate"]["status"] == "error"
    assert "Evidence Policy Discipline lane evidence is missing" in checks[
        "purpose_benchmark_gate"
    ]["message"]


def test_beta_readiness_runs_purpose_evidence_policy_fixture(tmp_path: Path) -> None:
    """Beta readiness should exercise high-risk purpose evidence policy fixtures."""
    _write_minimal_beta_ready_project(tmp_path)

    checks = {check["name"]: check for check in run_beta_readiness(project_root=tmp_path)["checks"]}

    assert checks["purpose_evidence_policy"]["status"] == "ok"
    assert "support, product, sales, legal, and executive evidence-policy fixtures" in checks[
        "purpose_evidence_policy"
    ]["message"]


def test_beta_readiness_rejects_slow_first_run_timing_report(tmp_path: Path) -> None:
    """Beta readiness should enforce the five-minute first-run budget."""
    _write_minimal_beta_ready_project(tmp_path)
    report = tmp_path / "docs" / "examples" / "first-run-timing-report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["time_to_successful_doctor_seconds"] = 360
    report.write_text(json.dumps(payload), encoding="utf-8")

    checks = {check["name"]: check for check in run_beta_readiness(project_root=tmp_path)["checks"]}

    assert checks["first_run_timing"]["status"] == "error"
    assert "time_to_successful_doctor_seconds=360" in checks["first_run_timing"]["message"]
    assert checks["first_run_timing"]["action"] == (
        "Update docs/examples/first-run-timing-report.json with a passing clean first-run timing report."
    )


def test_first_run_timing_gate_reports_missing_and_invalid_json(tmp_path: Path) -> None:
    missing = _check_first_run_timing(tmp_path)
    assert missing["status"] == "error"
    assert "missing or unreadable" in missing["message"]

    report = tmp_path / "docs" / "examples" / "first-run-timing-report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{not-json", encoding="utf-8")

    invalid = _check_first_run_timing(tmp_path)
    assert invalid["status"] == "error"
    assert "invalid JSON" in invalid["message"]


def test_first_run_timing_gate_reports_missing_fields_and_formats_decimals(tmp_path: Path) -> None:
    report = tmp_path / "docs" / "examples" / "first-run-timing-report.json"
    report.parent.mkdir(parents=True)
    report.write_text(
        json.dumps(
            {
                "threshold_seconds": 299.5,
                "time_to_successful_doctor_seconds": True,
                "time_to_first_successful_example_seconds": 301.25,
                "requires_sidecar": True,
            }
        ),
        encoding="utf-8",
    )

    result = _check_first_run_timing(tmp_path)

    assert result["status"] == "error"
    assert "threshold_seconds=299.5 must be 300" in result["message"]
    assert "time_to_successful_doctor_seconds is missing" in result["message"]
    assert "time_to_first_successful_example_seconds=301.25 exceeds 300" in result["message"]
    assert "requires_sidecar must be false" in result["message"]


def test_release_doc_gates_report_missing_happy_path_references(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("zaxy init\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "getting-started.md").write_text("zaxy memory checkout\n", encoding="utf-8")
    (docs / "testing.md").write_text("scripts/beta-uat.sh\n", encoding="utf-8")

    docs_result = _check_docs_happy_path(tmp_path)
    capture_result = _check_capture_happy_path(tmp_path)
    roadmap_result = _check_beta_roadmap_claims(tmp_path)

    assert docs_result["status"] == "error"
    assert "pipx install zaxy-memory" in docs_result["message"]
    assert capture_result["status"] == "error"
    assert "deterministic" in capture_result["message"]
    assert roadmap_result["status"] == "error"
    assert "BETA.md is missing or unreadable" in roadmap_result["message"]


def test_release_doc_gates_accept_complete_happy_path_references(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text(
        "pipx install zaxy-memory\nzaxy init\nzaxy memory bootstrap\n",
        encoding="utf-8",
    )
    (docs / "getting-started.md").write_text(
        "zaxy memory checkout\nzaxy doctor --beta-readiness\n"
        "deterministic\nzaxy capture start\nzaxy capture status\n",
        encoding="utf-8",
    )
    (docs / "testing.md").write_text("scripts/beta-uat.sh\n", encoding="utf-8")
    (docs / "hooks.md").write_text("zaxy hook-status\nobservation coverage\n", encoding="utf-8")
    (docs / "mcp.md").write_text("zaxy capture soak\n", encoding="utf-8")
    _write_roadmap(tmp_path, _VALID_ROADMAP)

    assert _check_docs_happy_path(tmp_path)["status"] == "ok"
    assert _check_capture_happy_path(tmp_path)["status"] == "ok"
    assert _check_beta_roadmap_claims(tmp_path)["status"] == "ok"


def test_coordination_competitor_posture_reports_completed_adapter_audit_defects(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports" / "benchmarks" / "coordination-real-v1"
    manifest_dir = report_dir / "competitor-runner-manifests"
    docs_dir = tmp_path / "docs"
    manifest_dir.mkdir(parents=True)
    docs_dir.mkdir()
    docs_text = (
        "competitor_claim_gate\n--require-competitor-claim quarq\n"
        "--require-competitor-claim hybi\ndisclosure-only\npublic-claim gate\n"
    )
    (docs_dir / "benchmarks.md").write_text(docs_text, encoding="utf-8")
    (docs_dir / "coordinate-roadmap.md").write_text(docs_text, encoding="utf-8")
    (report_dir / "coordination-benchmark.md").write_text("## Competitor Claim Gate\n", encoding="utf-8")
    for name in ("quarq", "hybi"):
        (manifest_dir / f"{name}.runner-manifest.template.json").write_text("{}", encoding="utf-8")
    report = {
        "competitor_adapters": {
            "quarq": {
                "status": "completed",
                "claim_status": "same_harness",
                "metrics": {},
                "result_audit": {"manifest": {"name": "quarq"}},
            },
            "hybi": {
                "status": "completed",
                "claim_status": "same_harness",
                "metrics": {},
                "result_audit": {"result_fingerprint": "", "manifest": None},
            },
        },
        "competitor_claim_gate": {"status": "passed", "completed_adapters": ["quarq"]},
    }
    (report_dir / "coordination-benchmark.json").write_text(json.dumps(report), encoding="utf-8")

    result = _check_coordination_competitor_claim_posture(tmp_path)

    assert result["status"] == "error"
    assert "quarq result audit is missing result_fingerprint" in result["message"]
    assert "quarq result audit manifest missing" in result["message"]
    assert "hybi result audit manifest is missing" in result["message"]
    assert "passed claim gate must include completed quarq and hybi" in result["message"]


def test_release_overall_status_precedence() -> None:
    assert _overall_status([{"status": "ok"}, {"status": "warning"}]) == "warning"
    assert _overall_status([{"status": "warning"}, {"status": "error"}]) == "error"
    assert _overall_status([{"status": "ok"}]) == "ok"


def test_beta_readiness_requires_checkout_token_efficiency_guardrail(tmp_path: Path) -> None:
    """Beta readiness should reject UAT scripts that do not gate checkout token discipline."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script_path = tmp_path / "scripts" / "beta-uat.sh"
    script = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script.replace(" --max-checkout-prompt-tokens 5000", "")
        .replace(" --min-checkout-facts-per-1k-tokens 0.1", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["clean_repo_uat"]["status"] == "error"
    assert "--max-checkout-prompt-tokens 5000" in checks["clean_repo_uat"]["message"]
    assert "--min-checkout-facts-per-1k-tokens 0.1" in checks["clean_repo_uat"]["message"]
    assert checks["clean_repo_uat"]["action"] == (
        "Update scripts/beta-uat.sh to exercise the complete first-run beta path."
    )


def test_beta_readiness_requires_bare_embedded_uat_path(tmp_path: Path) -> None:
    """Beta readiness should fail if UAT does not protect the bare embedded default."""
    _write_minimal_beta_ready_project(tmp_path)
    script_path = tmp_path / "scripts" / "beta-uat.sh"
    script = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script.replace('run_workspace "embedded" "" "start"\n', "")
        .replace('grep -q "PROJECTION_BACKEND=embedded" .env.local\n', ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["clean_repo_uat"]["status"] == "error"
    assert "bare embedded init" in checks["clean_repo_uat"]["message"]


def test_beta_readiness_requires_bare_embedded_graph_operations(tmp_path: Path) -> None:
    """Beta readiness should fail if UAT skips embedded graph status and rebuild checks."""
    _write_minimal_beta_ready_project(tmp_path)
    script_path = tmp_path / "scripts" / "beta-uat.sh"
    script = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script.replace("zaxy memory status --eventloom-path .eventloom --graph\n", "")
        .replace("zaxy memory inferred-status --session-id\n", "")
        .replace("zaxy reproject\n", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["clean_repo_uat"]["status"] == "error"
    assert "zaxy memory status --eventloom-path .eventloom --graph" in checks["clean_repo_uat"]["message"]
    assert "zaxy memory inferred-status --session-id" in checks["clean_repo_uat"]["message"]
    assert "zaxy reproject" in checks["clean_repo_uat"]["message"]


def test_beta_readiness_requires_backend_shootout_release_gates(tmp_path: Path) -> None:
    """Beta readiness should reject release gates without backend shootout evidence checks."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("scripts/check-backend-shootout.py", "")
        .replace("backend-shootout.json", "")
        .replace("longmemeval-40-backend-shootout.json", "")
        .replace("longmemeval-100-backend-shootout.json", "")
        .replace("--forbid-backends neo4j,pggraph,latticedb", "")
        .replace("--require-query-results", "")
        .replace("--min-answer-at-5-per-1k-returned-tokens embedded=0.10", "")
        .replace("--min-answer-at-5-per-1k-returned-tokens embedded=0.15", "")
        .replace("--min-quality-per-1k-injected-tokens embedded=1.0", "")
        .replace("--min-answer-at-5-per-1k-injected-tokens embedded=1.0", "")
        .replace("--min-answer-at-5-per-1k-injected-tokens embedded=0.10", "")
        .replace("--min-answer-at-5-per-1k-injected-tokens embedded=0.15", "")
        .replace("--min-answer-at-5-per-1k-returned-tokens\n", "")
        .replace("--min-quality-per-1k-injected-tokens embedded=1.0\n", "")
        .replace("--max-cold-bootstrap-ms\n", "")
        .replace("--max-first-checkout-ms\n", "")
        .replace("--max-append-to-projection-p95-ms\n", "")
        .replace(" --max-checkout-p95-ms embedded=200", "")
        .replace("--max-checkout-p95-ms embedded=200\n", "")
        .replace("--min-answer-at-5-per-1k-injected-tokens\n", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "scripts/check-backend-shootout.py" in checks["release_gate"]["message"]
    assert "backend-shootout.json" in checks["release_gate"]["message"]
    assert "longmemeval-40-backend-shootout.json" in checks["release_gate"]["message"]
    assert "longmemeval-100-backend-shootout.json" in checks["release_gate"]["message"]
    assert "--forbid-backends neo4j,pggraph,latticedb" in checks["release_gate"]["message"]
    assert "--require-query-results" in checks["release_gate"]["message"]
    assert "--min-answer-at-5-per-1k-returned-tokens" in checks["release_gate"]["message"]
    assert "--min-quality-per-1k-injected-tokens embedded=1.0" in checks["release_gate"]["message"]
    assert "--max-cold-bootstrap-ms" in checks["release_gate"]["message"]
    assert "--max-first-checkout-ms" in checks["release_gate"]["message"]
    assert "--max-append-to-projection-p95-ms" in checks["release_gate"]["message"]
    assert "--max-checkout-p95-ms embedded=200" in checks["release_gate"]["message"]
    assert "--min-answer-at-5-per-1k-injected-tokens" in checks["release_gate"]["message"]


def test_beta_readiness_reports_release_gate_surface_coverage(tmp_path: Path) -> None:
    """Beta readiness should expose v0.9 run-or-skip coverage for every public release smoke."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["release_gate_surface_coverage"]["status"] == "ok"
    message = checks["release_gate_surface_coverage"]["message"]
    for surface in (
        "public examples",
        "MCP smoke",
        "LangGraph smoke",
        "Coordinate mission smoke",
        "benchmark comparison",
        "docs validation",
        "beta UAT",
        "external validation",
    ):
        assert surface in message


def test_beta_readiness_allows_release_without_external_validation_evidence(tmp_path: Path) -> None:
    """Beta readiness should not block v1.0 when outside validation is unavailable."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    _write_roadmap(tmp_path, "# Beta Roadmap\n\n" + _VALID_ROADMAP)

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "ok"
    assert checks["external_validation_evidence"]["status"] == "ok"
    assert "external validation is optional for v1.0 release" in checks["external_validation_evidence"]["message"]
    assert "post-release" in checks["external_validation_evidence"]["action"]


def test_beta_readiness_accepts_validated_external_validation_report(tmp_path: Path) -> None:
    """Beta readiness should turn the external-validation warning green for a validated report."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    report_path = tmp_path / "reports" / "external-validation" / "external-validation-report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "contract": "zaxy.v1.external-validation-report",
                "status": "validated",
                "validator": {
                    "name": "Independent Validation Project",
                    "external_to_implementation_session": True,
                },
                "date": "2026-05-31",
                "zaxy_version_or_commit": "v1.0.0-rc",
                "environment": {
                    "operating_system": "Linux",
                    "shell": "bash",
                    "python_version": "3.13",
                    "install_source": "pipx install zaxy-memory",
                },
                "validation_path": "first_run_local",
                "commands": [
                    "zaxy init",
                    "zaxy memory bootstrap --eventloom-path .eventloom",
                    "zaxy memory checkout current project memory --eventloom-path .eventloom",
                    "zaxy doctor --beta-readiness",
                ],
                "time_to_first_useful_checkout_seconds": 180,
                "unexpected_sidecar_or_credential_required": False,
                "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
                "friction_or_failure": "No blocking friction.",
                "release_decision": "pass",
                "supports_positioning": True,
            }
        ),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["external_validation_evidence"]["status"] == "ok"
    assert "reports/external-validation/external-validation-report.json" in checks[
        "external_validation_evidence"
    ]["message"]


def test_beta_readiness_accepts_explicit_external_validation_report_path(tmp_path: Path) -> None:
    """Beta readiness should accept release evidence from an explicit report path."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    report_path = tmp_path / "external-validation-report.json"
    report_path.write_text(
        json.dumps(
            {
                "contract": "zaxy.v1.external-validation-report",
                "status": "validated",
                "validator": {
                    "name": "Independent Validation Project",
                    "external_to_implementation_session": True,
                },
                "date": "2026-05-31",
                "zaxy_version_or_commit": "v1.0.0-rc",
                "environment": {
                    "operating_system": "Linux",
                    "shell": "bash",
                    "python_version": "3.13",
                    "install_source": "pipx install zaxy-memory",
                },
                "validation_path": "coordinate_workflow",
                "commands": ["python examples/coordinate_three_worker_project.py"],
                "time_to_first_useful_checkout_seconds": 90,
                "unexpected_sidecar_or_credential_required": False,
                "evidence_links": ["https://github.com/syndicalt/zaxy/discussions/1"],
                "friction_or_failure": "No blocking friction.",
                "release_decision": "pass",
                "supports_positioning": True,
            }
        ),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path, external_validation_report=report_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["external_validation_evidence"]["status"] == "ok"
    assert str(report_path) in checks["external_validation_evidence"]["message"]


def test_beta_readiness_rejects_missing_explicit_external_validation_report(tmp_path: Path) -> None:
    """A requested external-validation report path should still be enforced."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    report_path = tmp_path / "missing-external-validation-report.json"

    report = run_beta_readiness(project_root=tmp_path, external_validation_report=report_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["external_validation_evidence"]["status"] == "error"
    assert "external validation report was requested" in checks["external_validation_evidence"]["message"]
    assert str(report_path) in checks["external_validation_evidence"]["message"]


def test_beta_readiness_requires_external_validation_when_strict_mode_is_enabled(tmp_path: Path) -> None:
    """Strict release mode should preserve the external-validation evidence gate."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)

    report = run_beta_readiness(project_root=tmp_path, require_external_validation=True)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["external_validation_evidence"]["status"] == "error"
    assert "external validation is required" in checks["external_validation_evidence"]["message"]
    assert "reports/external-validation/external-validation-report.json" in checks[
        "external_validation_evidence"
    ]["message"]


def test_beta_readiness_rejects_unreadable_external_validation_report(tmp_path: Path) -> None:
    """External-validation reports should remain machine-checkable when present."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    report_path = tmp_path / "reports" / "external-validation" / "external-validation-report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{not-json", encoding="utf-8")

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["external_validation_evidence"]["status"] == "error"
    assert "unreadable or invalid JSON" in checks["external_validation_evidence"]["message"]


def test_beta_readiness_requires_explicit_release_gate_skip_reasons(tmp_path: Path) -> None:
    """Release gates may skip expensive public smokes only with an explicit reason."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    script_path = tmp_path / "scripts" / "release-check.sh"
    script = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script.replace(
            'MCP_SMOKE_CMD="python scripts/mcp_smoke_test.py"',
            'MCP_SMOKE_CMD=""',
        ),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["release_gate_surface_coverage"]["status"] == "error"
    assert "MCP_SMOKE_CMD must run or use SKIP:<reason>" in checks[
        "release_gate_surface_coverage"
    ]["message"]


def test_release_check_renders_external_validation_report_path_option() -> None:
    """The release gate should let operators pass an external-validation report path directly."""
    script = Path("scripts/release-check.sh").read_text(encoding="utf-8")

    assert "--external-validation-report" in script
    assert "EXTERNAL_VALIDATION_REPORT" in script
    assert 'EXTERNAL_VALIDATION_CMD="python scripts/check-external-validation.py ${EXTERNAL_VALIDATION_REPORT}"' in script

    help_result = subprocess.run(
        ["bash", "scripts/release-check.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert help_result.returncode == 0
    assert "--external-validation-report PATH" in help_result.stdout


def test_release_check_can_require_external_validation_without_running_full_gate() -> None:
    """Strict release mode should fail when external validation is still the default skip."""
    command = [
        "bash",
        "scripts/release-check.sh",
        "--require-external-validation",
        "--ruff-cmd",
        "true",
        "--mypy-cmd",
        "true",
        "--pytest-cmd",
        "true",
        "--coverage-cmd",
        "true",
        "--packet-smoke-cmd",
        "true",
        "--examples-smoke-cmd",
        "true",
        "--mcp-smoke-cmd",
        "true",
        "--langgraph-smoke-cmd",
        "true",
        "--coordinate-smoke-cmd",
        "true",
        "--package-cmd",
        "true",
        "--docs-cmd",
        "true",
        "--validate-cmd",
        "true",
        "--hook-status-cmd",
        "true",
        "--backend-shootout-cmd",
        "true",
        "--backend-performance-cmd",
        "true",
        "--backend-scale-cmd",
        "true",
        "--beta-uat-cmd",
        "true",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 2
    assert "External validation is required" in result.stderr


def test_release_check_requires_machine_checkable_external_validation_command() -> None:
    """Strict release mode should reject no-op external validation commands."""
    command = [
        "bash",
        "scripts/release-check.sh",
        "--require-external-validation",
        "--ruff-cmd",
        "true",
        "--mypy-cmd",
        "true",
        "--pytest-cmd",
        "true",
        "--coverage-cmd",
        "true",
        "--packet-smoke-cmd",
        "true",
        "--examples-smoke-cmd",
        "true",
        "--mcp-smoke-cmd",
        "true",
        "--langgraph-smoke-cmd",
        "true",
        "--coordinate-smoke-cmd",
        "true",
        "--package-cmd",
        "true",
        "--docs-cmd",
        "true",
        "--validate-cmd",
        "true",
        "--hook-status-cmd",
        "true",
        "--backend-shootout-cmd",
        "true",
        "--backend-performance-cmd",
        "true",
        "--backend-scale-cmd",
        "true",
        "--beta-uat-cmd",
        "true",
        "--external-validation-cmd",
        "true",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 2
    assert "External validation must run scripts/check-external-validation.py" in result.stderr


def test_release_check_rejects_spoofed_external_validation_checker_command() -> None:
    """Strict release mode should run the checker instead of matching its name in output."""
    command = [
        "bash",
        "scripts/release-check.sh",
        "--require-external-validation",
        "--ruff-cmd",
        "true",
        "--mypy-cmd",
        "true",
        "--pytest-cmd",
        "true",
        "--coverage-cmd",
        "true",
        "--packet-smoke-cmd",
        "true",
        "--examples-smoke-cmd",
        "true",
        "--mcp-smoke-cmd",
        "true",
        "--langgraph-smoke-cmd",
        "true",
        "--coordinate-smoke-cmd",
        "true",
        "--package-cmd",
        "true",
        "--docs-cmd",
        "true",
        "--validate-cmd",
        "true",
        "--hook-status-cmd",
        "true",
        "--backend-shootout-cmd",
        "true",
        "--backend-performance-cmd",
        "true",
        "--backend-scale-cmd",
        "true",
        "--beta-uat-cmd",
        "true",
        "--external-validation-cmd",
        "echo scripts/check-external-validation.py",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 2
    assert "External validation must run scripts/check-external-validation.py" in result.stderr


def test_release_check_rejects_shell_suffixed_external_validation_command() -> None:
    """Strict release mode should not allow shell suffixes that can mask checker failure."""
    command = [
        "bash",
        "scripts/release-check.sh",
        "--require-external-validation",
        "--ruff-cmd",
        "true",
        "--mypy-cmd",
        "true",
        "--pytest-cmd",
        "true",
        "--coverage-cmd",
        "true",
        "--packet-smoke-cmd",
        "true",
        "--examples-smoke-cmd",
        "true",
        "--mcp-smoke-cmd",
        "true",
        "--langgraph-smoke-cmd",
        "true",
        "--coordinate-smoke-cmd",
        "true",
        "--package-cmd",
        "true",
        "--docs-cmd",
        "true",
        "--validate-cmd",
        "true",
        "--hook-status-cmd",
        "true",
        "--backend-shootout-cmd",
        "true",
        "--backend-performance-cmd",
        "true",
        "--backend-scale-cmd",
        "true",
        "--beta-uat-cmd",
        "true",
        "--external-validation-cmd",
        "python scripts/check-external-validation.py missing-report.json; true",
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 2
    assert "External validation must run scripts/check-external-validation.py" in result.stderr


def test_beta_readiness_exposes_benchmark_no_regression_gate(tmp_path: Path) -> None:
    """Beta readiness should report the v0.8 benchmark no-regression evidence explicitly."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["benchmark_no_regression"]["status"] == "ok"
    assert "checkout quality" in checks["benchmark_no_regression"]["message"]
    assert "citation coverage" in checks["benchmark_no_regression"]["message"]
    assert "p95/p99 latency" in checks["benchmark_no_regression"]["message"]


def test_beta_readiness_requires_benchmark_no_regression_guardrails(tmp_path: Path) -> None:
    """Beta readiness should fail when release checks stop gating quality, citation, or latency budgets."""
    _write_minimal_beta_ready_project(tmp_path)
    _write_backend_report_inputs(tmp_path)
    script_path = tmp_path / "scripts" / "release-check.sh"
    script = script_path.read_text(encoding="utf-8")
    script_path.write_text(
        script.replace("--min-citation-coverage 1.0", "")
        .replace("--min-quality-per-1k-returned-tokens", "")
        .replace("--max-checkout-p99-ms", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["benchmark_no_regression"]["status"] == "error"
    assert "--min-citation-coverage 1.0" in checks["benchmark_no_regression"]["message"]
    assert "--min-quality-per-1k-returned-tokens" in checks["benchmark_no_regression"]["message"]
    assert "--max-checkout-p99-ms" in checks["benchmark_no_regression"]["message"]


def test_api_inventory_documents_v09_freeze_surfaces() -> None:
    """The v0.9 API inventory should classify every roadmap surface."""
    inventory = Path("docs/api-inventory.md").read_text(encoding="utf-8")

    for heading in (
        "## Stability Labels",
        "## MCP Tool Contracts",
        "## Python SDK Public Exports",
        "## Stable CLI Commands",
        "## Durable Eventloom Events",
        "## Projection Backend Contract",
        "## Benchmark Artifact Schemas",
        "## Migration and Freeze Policy",
    ):
        assert heading in inventory
    for label in ("Stable", "Beta", "Experimental", "Internal"):
        assert f"`{label}`" in inventory
    for tool in ("memory_bootstrap", "memory_checkout", "context_assemble", "coordination_start"):
        assert f"`{tool}`" in inventory
    for public_export in ("MemoryFabric", "MemoryCheckout", "CoordinationManager", "ProjectionStore"):
        assert f"`{public_export}`" in inventory
    for event_type in ("memory.checkout.completed", "coordination.finding.reported", "transcript.turn"):
        assert f"`{event_type}`" in inventory


def test_migration_guide_covers_04_through_09() -> None:
    """The v0.9 migration guide should cover every public release band."""
    guide = Path("docs/migration.md").read_text(encoding="utf-8")

    for heading in (
        "## Upgrade Checklist",
        "## From 0.4 to 0.5",
        "## From 0.5 to 0.6",
        "## From 0.6 to 0.7",
        "## From 0.7 to 0.8",
        "## From 0.8 to 0.9",
        "## Compatibility Tests",
        "## Rollback Policy",
    ):
        assert heading in guide
    for command in (
        "zaxy doctor --beta-readiness",
        "zaxy memory status --graph",
        "zaxy coordinate inspect",
        "zaxy trace export",
    ):
        assert f"`{command}`" in guide
    for contract in ("`zaxy.native.v0.6`", "`docs/api-inventory.md`", "`memory_checkout`"):
        assert contract in guide


def test_v1_schema_freeze_manifest_tracks_candidate_contracts() -> None:
    """The v0.9 freeze candidate should bind public schema contracts to migration policy."""
    manifest = json.loads(Path("docs/examples/v1-schema-freeze.json").read_text(encoding="utf-8"))

    assert manifest["contract"] == "zaxy.v1.schema-freeze"
    assert manifest["status"] == "freeze-candidate"
    assert manifest["change_policy"]["stable_or_beta"] == "migration_event_required"
    assert manifest["migration_event_type"] == "schema.migration.proposed"
    surfaces = {surface["name"]: surface for surface in manifest["surfaces"]}
    for name in (
        "mcp_tool_contract",
        "mcp_response_snapshots",
        "native_integration_contract",
        "memory_checkout_contract",
        "api_inventory",
        "eventloom_event_taxonomy",
        "benchmark_artifact_schemas",
    ):
        assert surfaces[name]["status"] in {"Stable", "Beta"}
        assert Path(surfaces[name]["path"]).exists()

    agent_events = Path("docs/agent-events.md").read_text(encoding="utf-8")
    assert "schema.migration.proposed" in agent_events
    assert "schema.migration.applied" in agent_events


def test_v09_gate_audit_records_current_gate_evidence() -> None:
    """The v0.9 gate audit should separate proven gates from optional external feedback."""
    audit = Path("docs/archive/v09-gate-audit.md").read_text(encoding="utf-8")

    for required in (
        "## Proven Gates",
        "## Optional Evidence",
        "Python 3.11, 3.12, and 3.13",
        "pytest -m \"not integration\" --benchmark-disable --cov --cov-report=xml",
        "scripts/check-coverage.py --root . --coverage-xml coverage.xml",
        "zaxy doctor --beta-readiness",
        "release_gate_surface_coverage",
        "backend_report_inputs",
        "docs/api-inventory.md",
        "docs/migration.md",
        "External User Feedback",
        "optional for v1.0 release",
    ):
        assert required in audit
    assert "Status: pending" not in audit


def test_v10_announcement_and_release_checklist_cover_launch_requirements() -> None:
    """v1.0 release artifacts should cover positioning, evidence, limitations, and gates."""
    announcement = Path("docs/announcements/zaxy-v1.0.md").read_text(encoding="utf-8")
    checklist = Path("docs/archive/release-validation-checklist.md").read_text(encoding="utf-8")

    for required in (
        "Coordinator Memory for Agent Teams",
        "Memory Bootstrap",
        "Memory Checkout",
        "Zaxy Coordinate",
        "LongMemEval",
        "CoordinationBench",
        "backend shootout",
        "Limitations",
        "Roadmap beyond 1.0",
        "External validation",
    ):
        assert required in announcement
    for required in (
        "Clean-repo UAT",
        "MCP smoke",
        "LangGraph smoke",
        "direct model integration smoke",
        "Coordinate mission smoke",
        "Benchmark guardrails",
        "Docs validation",
        "Release smoke",
        "Coverage remains at or above 92%",
        "Public surfaces are tagged",
        "External validation",
    ):
        assert required in checklist
    assert "scripts/release-check.sh --root ." in checklist
    assert "zaxy doctor --beta-readiness" in checklist


def test_v10_gate_audit_records_release_gate_evidence() -> None:
    """The v1.0 gate audit should map release gates to proof and optional evidence."""
    audit = Path("docs/archive/v10-gate-audit.md").read_text(encoding="utf-8")

    for required in (
        "## Proven Local Gates",
        "## Optional Post-Release Evidence",
        "Clean-repo UAT",
        "scripts/beta-uat.sh",
        "MCP smoke",
        "python scripts/mcp_smoke_test.py",
        "LangGraph smoke",
        "test_langgraph_example_runs_without_langgraph_dependency",
        "direct model integration smoke",
        "tests/test_openai_compatible_adapter.py",
        "Coordinate mission smoke",
        "test_coordinate_three_worker_example_runs",
        "Benchmark guardrails",
        "scripts/benchmark-guardrails.sh",
        "Docs validation",
        "scripts/build-site-docs.py --check",
        "Release smoke",
        "zaxy doctor --release-smoke",
        "Coverage remains at or above 92%",
        "scripts/check-coverage.py --root . --coverage-xml coverage.xml",
        "Public surfaces are tagged",
        "docs/api-inventory.md",
        "docs/examples/v1-schema-freeze.json",
        "External validation",
        "Status: optional for v1.0 release",
    ):
        assert required in audit


def test_v10_external_validation_packet_captures_optional_evidence() -> None:
    """External validation should have a concrete packet and issue template for v1.0 evidence."""
    packet = Path("docs/external-validation.md").read_text(encoding="utf-8")
    issue = Path(".github/ISSUE_TEMPLATE/external_validation.md").read_text(encoding="utf-8")

    for required in (
        "## Who Should Run This",
        "## Validation Paths",
        "## Required Evidence",
        "## Report Template",
        "## Acceptance Criteria",
        "zaxy init",
        "zaxy memory bootstrap",
        "zaxy memory checkout",
        "zaxy doctor --beta-readiness",
        "python examples/coordinate_three_worker_project.py",
        "scripts/beta-uat.sh",
        "archive/v10-gate-audit.md",
        "archive/release-validation-checklist.md",
        "Status: optional for the v1.0 release",
    ):
        assert required in packet
    for required in (
        "External validation",
        "Zaxy version",
        "Validation path",
        "Time to first useful checkout",
        "Evidence link",
        "absolute `http` or `https` URL",
        "docs/examples/external-validation-report.example.json",
        "Friction or failure",
        "Release decision",
    ):
        assert required in issue


def test_external_validation_checker_accepts_complete_report_and_rejects_pending(tmp_path: Path) -> None:
    """The release gate should validate external evidence instead of accepting prose."""
    complete_report = tmp_path / "external-validation-pass.json"
    complete_report.write_text(
        json.dumps(
            {
                "contract": "zaxy.v1.external-validation-report",
                "status": "validated",
                "validator": {
                    "name": "Independent Validation Project",
                    "external_to_implementation_session": True,
                },
                "date": "2026-05-31",
                "zaxy_version_or_commit": "v1.0.0-rc",
                "environment": {
                    "operating_system": "Linux",
                    "shell": "bash",
                    "python_version": "3.13",
                    "install_source": "pipx install zaxy-memory",
                },
                "validation_path": "first_run_local",
                "commands": [
                    "zaxy init",
                    "zaxy memory bootstrap --eventloom-path .eventloom",
                    "zaxy memory checkout current project memory --eventloom-path .eventloom",
                    "zaxy doctor --beta-readiness",
                ],
                "time_to_first_useful_checkout_seconds": 180,
                "unexpected_sidecar_or_credential_required": False,
                "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
                "friction_or_failure": "No blocking friction.",
                "release_decision": "pass_with_follow_up",
                "supports_positioning": True,
            }
        ),
        encoding="utf-8",
    )
    pending_report = tmp_path / "external-validation-pending.json"
    pending_report.write_text(
        json.dumps(
            {
                "contract": "zaxy.v1.external-validation-report",
                "status": "pending",
                "validator": {
                    "name": "Implementation Session",
                    "external_to_implementation_session": False,
                },
                "release_decision": "fail",
            }
        ),
        encoding="utf-8",
    )

    passed = subprocess.run(
        [sys.executable, "scripts/check-external-validation.py", str(complete_report)],
        check=False,
        capture_output=True,
        text=True,
    )
    failed = subprocess.run(
        [sys.executable, "scripts/check-external-validation.py", str(pending_report)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert passed.returncode == 0, passed.stderr
    assert "External validation passed" in passed.stdout
    assert failed.returncode == 1
    assert "status must be validated" in failed.stderr
    assert "validator.external_to_implementation_session must be true" in failed.stderr


def test_external_validation_uses_shared_validator() -> None:
    """The CLI checker and beta readiness should share one report validator."""
    script = Path("scripts/check-external-validation.py").read_text(encoding="utf-8")
    release = Path("src/zaxy/release.py").read_text(encoding="utf-8")
    payload = json.loads(Path("docs/examples/external-validation-report.example.json").read_text(encoding="utf-8"))

    payload["status"] = "validated"
    errors = validate_external_validation_report(payload)

    assert "validator.name must not be a placeholder" in errors
    assert "evidence_links must not contain placeholder values" in errors
    assert "friction_or_failure must not be a placeholder" in errors
    assert "from zaxy.external_validation import" in script
    assert "from zaxy.external_validation import" in release


def test_external_validation_rejects_non_object_report() -> None:
    """External validation reports must be structured JSON objects."""
    assert validate_external_validation_report(["not", "an", "object"]) == ["report must be a JSON object"]


def test_external_validation_reports_all_structural_field_errors() -> None:
    """Invalid report field shapes should produce actionable errors in one pass."""
    payload = {
        "contract": "wrong",
        "status": "pending",
        "validator": {},
        "environment": {},
        "date": 20260531,
        "zaxy_version_or_commit": "",
        "validation_path": "unknown",
        "commands": [],
        "time_to_first_useful_checkout_seconds": 0,
        "unexpected_sidecar_or_credential_required": True,
        "evidence_links": [],
        "friction_or_failure": "",
        "release_decision": "fail",
        "supports_positioning": False,
    }

    errors = validate_external_validation_report(payload)

    for expected in (
        "contract must be zaxy.v1.external-validation-report",
        "status must be validated",
        "validator.name must be a non-empty string",
        "validator.external_to_implementation_session must be true",
        "environment.operating_system must be a non-empty string",
        "environment.shell must be a non-empty string",
        "environment.python_version must be a non-empty string",
        "environment.install_source must be a non-empty string",
        "date must be an ISO date string",
        "zaxy_version_or_commit must be a non-empty string",
        "validation_path must be one of:",
        "commands must be a non-empty list of command strings",
        "time_to_first_useful_checkout_seconds must be a positive number",
        "unexpected_sidecar_or_credential_required must be false",
        "evidence_links must include at least one report, issue, discussion, or case-study link",
        "friction_or_failure must be a non-empty string",
        "release_decision must be pass or pass_with_follow_up",
        "supports_positioning must be true",
    ):
        assert any(expected in error for error in errors)


def test_external_validation_accepts_wrapped_validation_commands() -> None:
    """The validator should accept common command wrappers for supported validation paths."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "Python 3.13.1",
            "install_source": "pipx install zaxy-memory==1.0.0",
        },
        "validation_path": "first_run_local",
        "commands": [
            "uv run zaxy init --preset local-codex",
            "python -m zaxy memory bootstrap --eventloom-path .eventloom",
            "poetry run zaxy memory checkout current project memory",
            "pipx run zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass",
        "supports_positioning": True,
    }

    assert validate_external_validation_report(payload) == []


def test_external_validation_accepts_script_wrappers_for_script_paths() -> None:
    """Script-based validation paths should accept Python and shell wrapper commands."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "zsh",
            "python_version": "3.13.1",
            "install_source": "uv tool install zaxy-memory==1.0.0",
        },
        "validation_path": "coordinate_workflow",
        "commands": ["python examples/coordinate_three_worker_project.py"],
        "time_to_first_useful_checkout_seconds": 90,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/discussions/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass",
        "supports_positioning": True,
    }
    clean_repo_payload = dict(payload)
    clean_repo_payload["validation_path"] = "clean_repo_uat"
    clean_repo_payload["commands"] = ["bash scripts/beta-uat.sh"]

    assert validate_external_validation_report(payload) == []
    assert validate_external_validation_report(clean_repo_payload) == []


def test_external_validation_helper_predicates_ignore_non_string_inputs() -> None:
    """Validator helpers should be defensive when report fields have invalid types."""
    value = object()

    assert external_validation._is_placeholder(value) is False
    assert external_validation._is_implementation_session_name(value) is False
    assert external_validation._is_absolute_web_url(value) is False
    assert external_validation._has_url_credentials(value) is False
    assert external_validation._has_bare_origin_url(value) is False
    assert external_validation._has_single_label_hostname(value) is False
    assert external_validation._is_local_only_url(value) is False
    assert external_validation._is_private_network_url(value) is False
    assert external_validation._is_internal_only_domain_url(value) is False
    assert external_validation._is_example_domain_url(value) is False
    assert external_validation._is_invalid_github_artifact_url(value) is False
    assert external_validation._is_vague_version_reference(value) is False
    assert external_validation._is_vague_install_source(value) is False
    assert external_validation._is_vague_shell(value) is False
    assert external_validation._has_shell_control_operator(value) is False
    assert external_validation._command_starts_with_marker(value, "zaxy init") is False
    assert external_validation._is_substantive_validation_command(value) is False
    assert external_validation._parse_date(value) is None
    assert external_validation._parse_date("not-a-date") is None


def test_external_validation_requires_reviewable_evidence_links() -> None:
    """Validated reports should link to independently reviewable external evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["notes/external-validation.md"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must contain absolute http or https URLs" in errors


def test_external_validation_rejects_example_validator_name() -> None:
    """Validated reports should not use sample validator names."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Example External User",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "validator.name must not be a placeholder" in errors


def test_external_validation_rejects_sample_validator_name() -> None:
    """Validated reports should not use sample validator names from instructions or templates."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Sample External User",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "validator.name must not be a placeholder" in errors


def test_external_validation_rejects_implementation_session_validator_name() -> None:
    """Validated reports should not name the current implementation session as validator."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Implementation Session",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "validator.name must identify an external validator" in errors


def test_external_validation_rejects_agent_validator_name() -> None:
    """Validated reports should not name the implementing agent as validator."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Codex Agent",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "validator.name must identify an external validator" in errors


def test_external_validation_rejects_local_only_evidence_links() -> None:
    """Validated reports should not use local-only URLs as external evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["http://localhost:8000/external-validation"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must not use local-only URLs" in errors


def test_external_validation_rejects_private_network_evidence_links() -> None:
    """Validated reports should not rely on private-network evidence URLs."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["http://192.168.1.20/external-validation"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must not use private-network URLs" in errors


def test_external_validation_rejects_internal_only_evidence_domains() -> None:
    """Validated reports should not rely on internal-only DNS names as evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://validation.internal/external-validation"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must not use internal-only domains" in errors


def test_external_validation_rejects_evidence_links_with_credentials() -> None:
    """Validated reports should not embed credentials in evidence URLs."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://token:secret@github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must not include credentials" in errors


def test_external_validation_rejects_bare_origin_evidence_links() -> None:
    """Validated reports should link to a concrete reviewable evidence artifact."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must include a concrete artifact path" in errors


def test_external_validation_rejects_single_label_evidence_hosts() -> None:
    """Validated reports should not rely on single-label internal hostnames."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://validation/external-report"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must use fully qualified public hostnames" in errors


def test_external_validation_rejects_repository_home_evidence_links() -> None:
    """Validated reports should link to evidence artifacts, not repository home pages."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_collection_evidence_links() -> None:
    """Validated reports should link to concrete GitHub artifacts, not collection pages."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_invalid_github_artifact_ids() -> None:
    """Validated reports should link to numbered GitHub issue, discussion, or pull artifacts."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/not-a-number"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_query_decorated_github_artifact_links() -> None:
    """Validated reports should use canonical GitHub artifact URLs without query strings."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1?plain=1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_fragment_decorated_github_artifact_links() -> None:
    """Validated reports should use canonical GitHub artifact URLs without fragments."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1#issuecomment-1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_trailing_slash_github_artifact_links() -> None:
    """Validated reports should use exact GitHub artifact paths without trailing slashes."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1/"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_mixed_case_github_artifact_paths() -> None:
    """Validated reports should use canonical lowercase GitHub artifact path segments."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/Issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_duplicate_slash_github_artifact_paths() -> None:
    """Validated reports should not accept empty path segments in GitHub artifact URLs."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues//1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_numbered_github_artifact_links_with_extra_paths() -> None:
    """Validated reports should link to exact numbered GitHub issue artifacts."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1/extra"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_zero_github_artifact_ids() -> None:
    """Validated reports should link to positive-numbered GitHub artifacts."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/0"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_leading_zero_github_artifact_ids() -> None:
    """Validated reports should link to canonical positive-numbered GitHub artifacts."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/01"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_plural_pulls_artifact_links() -> None:
    """Validated reports should use GitHub's concrete pull-request artifact path."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/pulls/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_invalid_github_actions_run_ids() -> None:
    """Validated reports should link to concrete GitHub Actions run evidence artifacts."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/actions/runs/not-a-number"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_actions_run_collection_links() -> None:
    """Validated reports should link to a specific GitHub Actions run, not the runs collection."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/actions/runs"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_actions_run_links_with_extra_paths() -> None:
    """Validated reports should link to the exact GitHub Actions run artifact."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/actions/runs/123/extra"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_release_tag_collection_links() -> None:
    """Validated reports should link to a specific GitHub release tag, not the tag collection."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/releases/tag"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_unsupported_github_repo_paths() -> None:
    """Validated reports should not accept arbitrary GitHub repo pages as evidence artifacts."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/pulse"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_latest_release_links() -> None:
    """Validated reports should link to a specific GitHub release tag, not latest-release redirects."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/releases/latest"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_vague_github_release_tags() -> None:
    """Validated reports should not accept vague tag-shaped GitHub release links."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/releases/tag/latest"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_release_tag_links_with_extra_paths() -> None:
    """Validated reports should link to the exact GitHub release tag artifact."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/releases/tag/v1.0.0/extra"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_branch_like_github_commit_links() -> None:
    """Validated reports should link to concrete GitHub commit SHAs, not moving branch names."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/commit/main"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_commit_links_with_extra_paths() -> None:
    """Validated reports should link to the exact GitHub commit artifact."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/commit/abcdef1/extra"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_short_github_commit_sha_links() -> None:
    """Validated reports should link to full GitHub commit SHAs, not abbreviations."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/commit/abcdef1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_blob_links_with_moving_refs() -> None:
    """Validated reports should link to GitHub file evidence at commit SHA refs."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/blob/main/reports/external-validation.md"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_blob_links_without_file_paths() -> None:
    """Validated reports should link to a concrete GitHub file, not only a blob ref."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/blob/abcdef1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_github_raw_links_with_moving_refs() -> None:
    """Validated reports should not use moving refs in GitHub raw file evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/raw/main/reports/external-validation.json"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_raw_githubusercontent_links_with_moving_refs() -> None:
    """Validated reports should not use moving refs in raw.githubusercontent.com evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://raw.githubusercontent.com/syndicalt/zaxy/main/reports/external-validation.json"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_raw_githubusercontent_links_without_refs() -> None:
    """Validated reports should not use raw.githubusercontent.com URLs without file refs."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://raw.githubusercontent.com/syndicalt/zaxy"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must point to a reviewable evidence artifact" in errors


def test_external_validation_rejects_example_domain_evidence_links() -> None:
    """Validated reports should not use documentation example domains as evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://example.com/external-validation"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "evidence_links must not use example domains" in errors


def test_external_validation_rejects_future_dated_reports() -> None:
    """Validated reports should not be dated after the release gate run."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2999-01-01",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "date must not be in the future" in errors


def test_external_validation_rejects_zero_checkout_timing() -> None:
    """Validated reports should record a measured, positive time to useful checkout."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 0,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "time_to_first_useful_checkout_seconds must be a positive number" in errors


def test_external_validation_rejects_placeholder_commands() -> None:
    """Validated reports should include actual commands, not template text."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": ["Replace with exact commands used"],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must not contain placeholder values" in errors


def test_external_validation_rejects_na_friction_narrative() -> None:
    """Validated reports should include a real friction narrative, not N/A."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "N/A",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "friction_or_failure must not be a placeholder" in errors


def test_external_validation_requires_commands_for_selected_path() -> None:
    """Validated reports should prove the documented validation path they claim."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": ["python examples/coordinate_three_worker_project.py"],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must include zaxy init for first_run_local validation" in errors


def test_external_validation_requires_complete_first_run_command_path() -> None:
    """First-run validation should prove init, bootstrap, checkout, and readiness."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": ["zaxy init"],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must include zaxy memory bootstrap for first_run_local validation" in errors
    assert "commands must include zaxy memory checkout for first_run_local validation" in errors
    assert "commands must include zaxy doctor --beta-readiness for first_run_local validation" in errors


def test_external_validation_rejects_echoed_commands_for_selected_path() -> None:
    """Validated reports should prove commands were run, not echoed as text."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "echo zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must record executed commands, not echoed command text" in errors


def test_external_validation_rejects_echo_command_entries() -> None:
    """Validated reports should reject command text that only prints a command."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "echo 'zaxy init'",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must record executed commands, not echoed command text" in errors


def test_external_validation_rejects_help_commands_for_selected_path() -> None:
    """Validated reports should prove workflow commands ran, not help probes."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init --help",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must include zaxy init for first_run_local validation" in errors


def test_external_validation_rejects_compound_shell_commands() -> None:
    """Validated reports should record direct workflow commands, not shell compounds."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init && echo done",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must not contain shell control operators" in errors


def test_external_validation_rejects_background_shell_commands() -> None:
    """Validated reports should not hide backgrounded shell compounds behind a command marker."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init & echo done",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must not contain shell control operators" in errors


def test_external_validation_rejects_shell_comment_commands() -> None:
    """Validated reports should record executed commands without shell comments."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init # then review generated config",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must not contain shell control operators" in errors


def test_external_validation_rejects_parenthesized_shell_commands() -> None:
    """Validated reports should not accept parenthesized shell group evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init (echo done)",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must not contain shell control operators" in errors


def test_external_validation_rejects_multiline_command_entries() -> None:
    """Validated reports should keep each command entry to one executed command line."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init\nzaxy doctor --beta-readiness",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must be single-line strings" in errors


def test_external_validation_rejects_non_substantive_other_documented_commands() -> None:
    """Other documented validation still needs substantive Zaxy command evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "other_documented",
        "commands": ["pwd"],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must include at least one substantive Zaxy validation command" in errors


def test_external_validation_rejects_unknown_zaxy_other_documented_commands() -> None:
    """Other documented validation should not accept arbitrary zaxy command text."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "other_documented",
        "commands": ["zaxy frobnicate"],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "commands must include at least one substantive Zaxy validation command" in errors


def test_external_validation_rejects_placeholder_version_and_environment() -> None:
    """Validated reports should replace version and environment template text."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "REPLACE with version or commit",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "TBD install source",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "zaxy_version_or_commit must not be a placeholder" in errors
    assert "environment.install_source must not be a placeholder" in errors


def test_external_validation_rejects_vague_python_version() -> None:
    """Validated reports should record a concrete Python major/minor version."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.x",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "environment.python_version must be a concrete Python version" in errors


def test_external_validation_rejects_vague_install_source() -> None:
    """Validated reports should record the concrete Zaxy install source used."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "package manager",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "environment.install_source must be concrete" in errors


def test_external_validation_rejects_vague_install_source_phrases() -> None:
    """Validated reports should reject moving install-source descriptions."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "latest from PyPI",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "environment.install_source must be concrete" in errors


def test_external_validation_rejects_branch_like_install_source_phrases() -> None:
    """Validated reports should not use moving branch names as install-source evidence."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "from GitHub main",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "environment.install_source must be concrete" in errors


def test_external_validation_rejects_vague_shell() -> None:
    """Validated reports should record a concrete shell name."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "v1.0.0-rc",
        "environment": {
            "operating_system": "Linux",
            "shell": "terminal",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "environment.shell must be concrete" in errors


def test_external_validation_rejects_vague_version_reference() -> None:
    """Validated reports should identify a concrete Zaxy version or commit."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "latest",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "zaxy_version_or_commit must be a concrete version or commit" in errors


def test_external_validation_rejects_vague_version_reference_phrases() -> None:
    """Validated reports should reject phrase-form moving refs for the tested Zaxy version."""
    payload = {
        "contract": "zaxy.v1.external-validation-report",
        "status": "validated",
        "validator": {
            "name": "Independent Validation Project",
            "external_to_implementation_session": True,
        },
        "date": "2026-05-31",
        "zaxy_version_or_commit": "latest release",
        "environment": {
            "operating_system": "Linux",
            "shell": "bash",
            "python_version": "3.13",
            "install_source": "pipx install zaxy-memory",
        },
        "validation_path": "first_run_local",
        "commands": [
            "zaxy init",
            "zaxy memory bootstrap --eventloom-path .eventloom",
            "zaxy memory checkout current project memory --eventloom-path .eventloom",
            "zaxy doctor --beta-readiness",
        ],
        "time_to_first_useful_checkout_seconds": 180,
        "unexpected_sidecar_or_credential_required": False,
        "evidence_links": ["https://github.com/syndicalt/zaxy/issues/1"],
        "friction_or_failure": "No blocking friction.",
        "release_decision": "pass_with_follow_up",
        "supports_positioning": True,
    }

    errors = validate_external_validation_report(payload)

    assert "zaxy_version_or_commit must be a concrete version or commit" in errors


def test_external_validation_docs_reference_machine_checkable_report() -> None:
    """The v1.0 release docs should point validators at the machine-checkable report contract."""
    packet = Path("docs/external-validation.md").read_text(encoding="utf-8")
    checklist = Path("docs/archive/release-validation-checklist.md").read_text(encoding="utf-8")
    audit = Path("docs/archive/v10-gate-audit.md").read_text(encoding="utf-8")
    issue = Path(".github/ISSUE_TEMPLATE/external_validation.md").read_text(encoding="utf-8")
    example = Path("docs/examples/external-validation-report.example.json").read_text(encoding="utf-8")

    for required in (
        "scripts/check-external-validation.py",
        "docs/examples/external-validation-report.example.json",
        "zaxy.v1.external-validation-report",
        "Zaxy version or commit must be concrete, not `latest`, `current`, `main`, `master`, `head`, or `stable`",
        "validator name must not be a placeholder, sample name, or implementation-session name",
        "validator name must not identify the implementing agent",
        "friction or failure narrative must not be `none`, `n/a`, or placeholder text",
        "shell must be concrete enough to identify the shell used",
        "Python version must be concrete major/minor evidence",
        "install source must be concrete enough to reproduce the install path",
        "absolute `http` or `https` URL",
        "includes a concrete artifact path",
        "points to a reviewable evidence artifact instead of a repository homepage or collection page",
        "GitHub evidence links must use a supported artifact path",
        "GitHub evidence links must not include query strings",
        "GitHub evidence links must not include URL fragments",
        "GitHub evidence links must not include trailing slashes",
        "GitHub evidence links must not include empty path segments",
        "GitHub artifact path keywords must be lowercase",
        "GitHub issue, discussion, and pull-request links must use exact canonical positive-numbered artifact paths",
        "GitHub pull-request links must use `/pull/<number>`",
        "GitHub Actions run links must use exact `/actions/runs/<id>` paths with a concrete canonical positive numeric run ID",
        "GitHub release links must use exact `/releases/tag/<tag>` paths with a concrete non-vague release tag",
        "GitHub commit links must use `/commit/<sha>` with a full 40-character commit SHA",
        "GitHub file links (`blob`, `raw`, or `tree`) must use a full 40-character commit SHA ref and file path instead of a branch or tag",
        "raw.githubusercontent.com links must use a full 40-character commit SHA ref and file path instead of a branch or tag",
        "uses a fully qualified public hostname",
        "does not include credentials",
        "not `localhost`, loopback, link-local, unspecified, or private-network URL",
        "not an internal-only domain such as `.internal`, `.local`, `.lan`, `.test`, or `.invalid`",
        "not a reserved example domain",
        "`commands` match the selected `validation_path`",
        "command entries must be single-line strings",
        "command entries must record executed commands, not echoed command text",
        "not `echo` or `printf` command text",
        "not compound shell commands",
        "not backgrounded shell commands",
        "not parenthesized shell groups",
        "not shell comments",
        "not help or version probes",
        "`first_run_local` reports must include `zaxy init`, `zaxy memory bootstrap`, `zaxy memory checkout`, and `zaxy doctor --beta-readiness`",
        "`other_documented` reports must include at least one substantive Zaxy validation command",
        "arbitrary or unknown `zaxy` command text does not count as validation evidence",
    ):
        assert required in packet
        assert required in checklist
        assert required in audit
    assert "not `localhost`, loopback, link-local, unspecified, or private-network URL" in issue
    assert "include a concrete artifact path" in issue
    assert "point to a reviewable evidence artifact instead of a repository homepage or collection page" in issue
    assert "GitHub evidence links must use a supported artifact path" in issue
    assert "GitHub evidence links must not include query strings" in issue
    assert "GitHub evidence links must not include URL fragments" in issue
    assert "GitHub evidence links must not include trailing slashes" in issue
    assert "GitHub evidence links must not include empty path segments" in issue
    assert "GitHub artifact path keywords must be lowercase" in issue
    assert "GitHub issue, discussion, and pull-request links must use exact canonical positive-numbered artifact paths" in issue
    assert "GitHub pull-request links must use `/pull/<number>`" in issue
    assert "GitHub Actions run links must use exact `/actions/runs/<id>` paths with a concrete canonical positive numeric run ID" in issue
    assert "GitHub release links must use exact `/releases/tag/<tag>` paths with a concrete non-vague release tag" in issue
    assert "GitHub commit links must use `/commit/<sha>` with a full 40-character commit SHA" in issue
    assert (
        "GitHub file links (`blob`, `raw`, or `tree`) must use a full 40-character commit SHA ref and file path instead of a branch or tag"
        in issue
    )
    assert (
        "raw.githubusercontent.com links must use a full 40-character commit SHA ref and file path instead of a branch or tag"
        in issue
    )
    assert "use a fully qualified public hostname" in issue
    assert "does not include credentials" in issue
    assert "not an internal-only domain such as `.internal`, `.local`, `.lan`, `.test`, or `.invalid`" in issue
    assert "not a reserved example domain" in issue
    assert "validator name must not be a placeholder, sample name," in issue
    assert "or implementation-session name" in issue
    assert "validator name must not identify the implementing agent" in issue
    assert "Friction or failure narrative must not be `none`, `n/a`, or placeholder text" in issue
    assert "Shell must be concrete enough to identify the shell used" in issue
    assert "concrete Python major/minor version" in issue
    assert "Install source must be concrete enough to reproduce the install path" in issue
    assert "command entries must be single-line strings" in issue
    assert "command entries must record executed commands, not echoed command text" in issue
    assert "not `echo` or `printf` command text" in issue
    assert "not compound shell commands" in issue
    assert "not backgrounded shell commands" in issue
    assert "not parenthesized shell groups" in issue
    assert "not shell comments" in issue
    assert "not help or version probes" in issue
    assert (
        "`first_run_local` reports must include `zaxy init`, `zaxy memory bootstrap`, `zaxy memory checkout`, and `zaxy doctor --beta-readiness`"
        in issue
    )
    assert "`other_documented` reports must include at least one substantive Zaxy validation command" in issue
    assert "Arbitrary or unknown `zaxy` command text does not count as validation evidence" in issue
    assert (
        "Zaxy version or commit must be concrete, not `latest`, `current`, `main`, `master`, `head`, or `stable`"
        in issue
    )
    payload = json.loads(example)
    assert payload["contract"] == "zaxy.v1.external-validation-report"
    assert payload["status"] == "example"
    assert payload["validator"]["external_to_implementation_session"] is True


def test_v10_stability_commitment_covers_public_surfaces_and_data_model() -> None:
    """v1.0 should have an explicit API and data model stability commitment."""
    commitment = Path("docs/stability-commitment.md").read_text(encoding="utf-8")

    for required in (
        "## Stability Commitment",
        "## Public API Surfaces",
        "## Data Model Commitment",
        "## Compatibility Policy",
        "## Migration Events",
        "## Non-Commitments",
        "docs/api-inventory.md",
        "docs/examples/v1-schema-freeze.json",
        "Eventloom",
        "MCP",
        "CLI",
        "ProjectionStore",
        "schema.migration.proposed",
        "schema.migration.applied",
    ):
        assert required in commitment


def test_contributor_docs_and_issue_templates_cover_v09_release_inputs() -> None:
    """Contributor docs should cover code, issue, and benchmark contribution paths."""
    guide = Path("CONTRIBUTING.md").read_text(encoding="utf-8")
    benchmark = Path("docs/archive/benchmark-contributions.md").read_text(encoding="utf-8")
    bug = Path(".github/ISSUE_TEMPLATE/bug_report.md").read_text(encoding="utf-8")
    feature = Path(".github/ISSUE_TEMPLATE/feature_request.md").read_text(encoding="utf-8")
    benchmark_issue = Path(".github/ISSUE_TEMPLATE/benchmark_contribution.md").read_text(encoding="utf-8")

    for required in (
        "NO HACKS, ONLY DEVELOP PRODUCTION CODE",
        "test-first",
        "zaxy doctor --beta-readiness",
        "scripts/release-check.sh --root .",
        "docs/api-inventory.md",
        "docs/migration.md",
    ):
        assert required in guide
    for required in (
        "tracked Eventloom",
        "query_results",
        "citation coverage",
        "scripts/check-backend-shootout.py",
        "reports/backend-shootout/",
    ):
        assert required in benchmark
    for template in (bug, feature, benchmark_issue):
        assert "Zaxy version" in template
        assert "Reproduction" in template
    assert "Benchmark artifact path" in benchmark_issue
    assert "API stability surface" in feature


def test_v09_roadmap_records_failure_injection_evidence() -> None:
    """The v0.9 roadmap should cite the failure-injection coverage that gates hardening."""
    roadmap = Path("docs/archive/v1-roadmap.md").read_text(encoding="utf-8")

    assert "failure-injection" in roadmap
    assert "projection rebuild" in roadmap
    assert "corrupted projection artifacts" in roadmap
    assert "missing hooks" in roadmap
    assert "stale checkout" in roadmap
    assert "degraded backends" in roadmap
    assert "test_reproject_command_closes_pggraph_backend_after_projection_failure" in roadmap
    assert "hook-status" in roadmap
    assert "zaxy_degraded_operations_total" in roadmap


def test_beta_readiness_rejects_backend_reports_with_untracked_inputs(tmp_path: Path) -> None:
    """Beta readiness should reject release evidence that cannot be reproduced from tracked inputs."""
    _write_minimal_beta_ready_project(tmp_path)
    reports = tmp_path / "reports" / "backend-shootout"
    reports.mkdir(parents=True)
    eventloom = reports / "tracked.eventloom.jsonl"
    eventloom.write_text('{"seq":1,"type":"decision.recorded","payload":{}}\n', encoding="utf-8")
    queries = reports / "untracked-queries.json"
    queries.write_text('[{"query":"tracked input evidence"}]\n', encoding="utf-8")
    (reports / "backend-shootout.json").write_text(
        '{"eventloom_path":"reports/backend-shootout/tracked.eventloom.jsonl",'
        '"queries_file":"reports/backend-shootout/untracked-queries.json"}\n',
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "add", "."],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "rm", "--cached", "reports/backend-shootout/untracked-queries.json"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["backend_report_inputs"]["status"] == "error"
    assert "backend-shootout.json queries_file reports/backend-shootout/untracked-queries.json" in checks[
        "backend_report_inputs"
    ]["message"]


def test_beta_readiness_requires_all_backend_report_artifacts(tmp_path: Path) -> None:
    """Beta readiness should fail if release-gate benchmark reports are absent."""
    _write_minimal_beta_ready_project(tmp_path)

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["backend_report_inputs"]["status"] == "error"
    assert "reports/backend-shootout/backend-shootout.json is missing" in checks["backend_report_inputs"]["message"]
    assert "reports/backend-shootout/longmemeval-40-backend-shootout.json is missing" in checks[
        "backend_report_inputs"
    ]["message"]
    assert "reports/backend-shootout/longmemeval-100-backend-shootout.json is missing" in checks[
        "backend_report_inputs"
    ]["message"]


def test_beta_readiness_requires_query_results_in_backend_report_artifacts(tmp_path: Path) -> None:
    """Beta readiness should reject reproducible-looking reports without per-query evidence."""
    _write_minimal_beta_ready_project(tmp_path)
    reports = tmp_path / "reports" / "backend-shootout"
    reports.mkdir(parents=True)
    eventloom = reports / "sample.eventloom"
    eventloom.mkdir()
    (eventloom / "agent-1.jsonl").write_text(
        '{"seq":1,"type":"decision.recorded","payload":{},"thread":"agent-1"}\n',
        encoding="utf-8",
    )
    queries = reports / "queries.json"
    queries.write_text('[{"query":"embedded benchmark evidence"}]\n', encoding="utf-8")
    payload = (
        '{"eventloom_path":"reports/backend-shootout/sample.eventloom",'
        '"queries_file":"reports/backend-shootout/queries.json"}\n'
    )
    for filename in (
        "backend-shootout.json",
        "longmemeval-40-backend-shootout.json",
        "longmemeval-100-backend-shootout.json",
    ):
        (reports / filename).write_text(payload, encoding="utf-8")

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["backend_report_inputs"]["status"] == "error"
    assert "backend-shootout.json query_results are missing" in checks["backend_report_inputs"]["message"]
    assert "longmemeval-40-backend-shootout.json query_results are missing" in checks[
        "backend_report_inputs"
    ]["message"]
    assert "longmemeval-100-backend-shootout.json query_results are missing" in checks[
        "backend_report_inputs"
    ]["message"]


def test_beta_readiness_requires_non_empty_query_result_diagnostics(tmp_path: Path) -> None:
    """Beta readiness should reject query-results containers without diagnostics."""
    _write_minimal_beta_ready_project(tmp_path)
    reports = tmp_path / "reports" / "backend-shootout"
    reports.mkdir(parents=True)
    eventloom = reports / "sample.eventloom"
    eventloom.mkdir()
    (eventloom / "agent-1.jsonl").write_text(
        '{"seq":1,"type":"decision.recorded","payload":{},"thread":"agent-1"}\n',
        encoding="utf-8",
    )
    queries = reports / "queries.json"
    queries.write_text('[{"query":"embedded benchmark evidence"}]\n', encoding="utf-8")
    payload = (
        '{"eventloom_path":"reports/backend-shootout/sample.eventloom",'
        '"queries_file":"reports/backend-shootout/queries.json",'
        '"query_results":{"embedded:retrieve":[]}}\n'
    )
    for filename in (
        "backend-shootout.json",
        "longmemeval-40-backend-shootout.json",
        "longmemeval-100-backend-shootout.json",
    ):
        (reports / filename).write_text(payload, encoding="utf-8")

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["backend_report_inputs"]["status"] == "error"
    assert "backend-shootout.json query_results embedded:retrieve has no diagnostics" in checks[
        "backend_report_inputs"
    ]["message"]
    assert "longmemeval-40-backend-shootout.json query_results embedded:retrieve has no diagnostics" in checks[
        "backend_report_inputs"
    ]["message"]
    assert "longmemeval-100-backend-shootout.json query_results embedded:retrieve has no diagnostics" in checks[
        "backend_report_inputs"
    ]["message"]


def test_beta_readiness_requires_query_result_diagnostic_objects(tmp_path: Path) -> None:
    """Beta readiness should reject placeholder diagnostic rows."""
    _write_minimal_beta_ready_project(tmp_path)
    reports = tmp_path / "reports" / "backend-shootout"
    reports.mkdir(parents=True)
    eventloom = reports / "sample.eventloom"
    eventloom.mkdir()
    (eventloom / "agent-1.jsonl").write_text(
        '{"seq":1,"type":"decision.recorded","payload":{},"thread":"agent-1"}\n',
        encoding="utf-8",
    )
    queries = reports / "queries.json"
    queries.write_text('[{"query":"embedded benchmark evidence"}]\n', encoding="utf-8")
    payload = (
        '{"eventloom_path":"reports/backend-shootout/sample.eventloom",'
        '"queries_file":"reports/backend-shootout/queries.json",'
        '"query_results":{"embedded:retrieve":["placeholder"]}}\n'
    )
    for filename in (
        "backend-shootout.json",
        "longmemeval-40-backend-shootout.json",
        "longmemeval-100-backend-shootout.json",
    ):
        (reports / filename).write_text(payload, encoding="utf-8")

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["backend_report_inputs"]["status"] == "error"
    assert "backend-shootout.json query_results embedded:retrieve[0] must be a diagnostic object" in checks[
        "backend_report_inputs"
    ]["message"]
    assert "longmemeval-40-backend-shootout.json query_results embedded:retrieve[0] must be a diagnostic object" in checks[
        "backend_report_inputs"
    ]["message"]
    assert "longmemeval-100-backend-shootout.json query_results embedded:retrieve[0] must be a diagnostic object" in checks[
        "backend_report_inputs"
    ]["message"]


def test_beta_readiness_requires_forbidden_candidate_guardrail_on_all_backend_gates(tmp_path: Path) -> None:
    """Beta readiness should reject release gates that forbid LatticeDB in only one backend report."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("--forbid-backends neo4j,pggraph,latticedb", "", 2),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "--forbid-backends neo4j,pggraph,latticedb (3 occurrences)" in checks["release_gate"]["message"]


def test_beta_readiness_requires_forbidden_candidate_guardrail_inside_each_backend_command(tmp_path: Path) -> None:
    """Beta readiness should reject parked-candidate flags that are not attached to backend commands."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    detached_flags = (
        "--forbid-backends neo4j,pggraph,latticedb\n"
        "--forbid-backends neo4j,pggraph,latticedb\n"
        "--forbid-backends neo4j,pggraph,latticedb\n"
    )
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("--forbid-backends neo4j,pggraph,latticedb", "") + detached_flags,
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert (
        "BACKEND_SHOOTOUT_CMD must include --forbid-backends neo4j,pggraph,latticedb"
        in checks["release_gate"]["message"]
    )
    assert (
        "BACKEND_PERFORMANCE_CMD must include --forbid-backends neo4j,pggraph,latticedb"
        in checks["release_gate"]["message"]
    )
    assert (
        "BACKEND_SCALE_CMD must include --forbid-backends neo4j,pggraph,latticedb"
        in checks["release_gate"]["message"]
    )


def test_beta_readiness_requires_strict_backend_flags_inside_each_backend_command(tmp_path: Path) -> None:
    """Beta readiness should reject strict backend flags that are detached from backend commands."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    strict_flags = [
        "--require-report-metadata",
        "--require-markdown-report",
        "--require-query-results",
        "--require-git-tracked-inputs",
        "--verify-report-fingerprints",
        "--require-labeled-metrics",
    ]
    for flag in strict_flags:
        script = script.replace(f" {flag}", "")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script + "\n".join(strict_flags) + "\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "BACKEND_SHOOTOUT_CMD must include --require-report-metadata" in checks["release_gate"]["message"]
    assert "BACKEND_PERFORMANCE_CMD must include --require-markdown-report" in checks["release_gate"]["message"]
    assert "BACKEND_SCALE_CMD must include --require-query-results" in checks["release_gate"]["message"]
    assert "BACKEND_SCALE_CMD must include --require-git-tracked-inputs" in checks["release_gate"]["message"]
    assert "BACKEND_SCALE_CMD must include --verify-report-fingerprints" in checks["release_gate"]["message"]
    assert "BACKEND_SCALE_CMD must include --require-labeled-metrics" in checks["release_gate"]["message"]


def test_beta_readiness_requires_query_results_inside_each_backend_command(tmp_path: Path) -> None:
    """Beta readiness should reject query-result evidence flags that are detached from backend commands."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace(" --require-query-results", "") + "--require-query-results\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "BACKEND_SHOOTOUT_CMD must include --require-query-results" in checks["release_gate"]["message"]
    assert "BACKEND_PERFORMANCE_CMD must include --require-query-results" in checks["release_gate"]["message"]
    assert "BACKEND_SCALE_CMD must include --require-query-results" in checks["release_gate"]["message"]


def test_beta_readiness_requires_required_backends_inside_each_backend_command(tmp_path: Path) -> None:
    """Beta readiness should reject required-backend flags that are detached from backend commands."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace(" --require-backends embedded,bm25", "") + "--require-backends embedded,bm25\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "BACKEND_SHOOTOUT_CMD must include --require-backends embedded,bm25" in checks["release_gate"]["message"]
    assert "BACKEND_PERFORMANCE_CMD must include --require-backends embedded,bm25" in checks["release_gate"]["message"]
    assert "BACKEND_SCALE_CMD must include --require-backends embedded,bm25" in checks["release_gate"]["message"]


def test_beta_readiness_requires_dashboard_source_inside_each_backend_command(tmp_path: Path) -> None:
    """Beta readiness should reject dashboard-source flags that are detached from backend commands."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace(" --require-dashboard-source embedded=embedded", "")
        + "--require-dashboard-source embedded=embedded\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "BACKEND_SHOOTOUT_CMD must include --require-dashboard-source embedded=embedded" in checks[
        "release_gate"
    ]["message"]
    assert "BACKEND_PERFORMANCE_CMD must include --require-dashboard-source embedded=embedded" in checks[
        "release_gate"
    ]["message"]
    assert "BACKEND_SCALE_CMD must include --require-dashboard-source embedded=embedded" in checks["release_gate"][
        "message"
    ]


def test_beta_readiness_requires_scale_threshold_inside_scale_backend_command(tmp_path: Path) -> None:
    """Beta readiness should reject scale thresholds that are detached from the scale backend command."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace(" --max-checkout-p95-ms embedded=200", "") + "--max-checkout-p95-ms embedded=200\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "BACKEND_SCALE_CMD must include --max-checkout-p95-ms embedded=200" in checks["release_gate"]["message"]


def test_beta_readiness_requires_activation_release_gate(tmp_path: Path) -> None:
    """Beta readiness should reject release gates without activation and checkout-token guardrails."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("zaxy hook-status\n", "")
        .replace("--min-activation-rate 1.0\n", "")
        .replace("--max-checkout-prompt-tokens 5000\n", "")
        .replace("--min-checkout-facts-per-1k-tokens 0.1\n", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "zaxy hook-status" in checks["release_gate"]["message"]
    assert "--min-activation-rate 1.0" in checks["release_gate"]["message"]
    assert "--max-checkout-prompt-tokens 5000" in checks["release_gate"]["message"]
    assert "--min-checkout-facts-per-1k-tokens 0.1" in checks["release_gate"]["message"]


def test_beta_readiness_requires_source_tree_activation_release_gate_invocation(tmp_path: Path) -> None:
    """Beta readiness should reject activation gates that depend on an installed zaxy executable."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("PYTHONPATH=src python -m zaxy hook-status\n", "python -m zaxy hook-status\n"),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "PYTHONPATH=src python -m zaxy hook-status" in checks["release_gate"]["message"]


def test_beta_readiness_requires_deterministic_activation_release_gate_time(tmp_path: Path) -> None:
    """Beta readiness should reject activation gates that depend on wall-clock time."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("--now 2026-05-20T12:00:00+00:00\n", ""),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "--now 2026-05-20T12:00:00+00:00" in checks["release_gate"]["message"]


def test_beta_readiness_requires_release_gate_to_use_checked_activation_fixture(tmp_path: Path) -> None:
    """Beta readiness should reject activation gates pointed at ad hoc Eventloom paths."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    script = (tmp_path / "scripts" / "release-check.sh").read_text(encoding="utf-8")
    (tmp_path / "scripts" / "release-check.sh").write_text(
        script.replace("--eventloom-path reports/activation-release\n", "--eventloom-path .eventloom\n")
        + "reports/activation-release\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_gate"]["status"] == "error"
    assert "--eventloom-path reports/activation-release" in checks["release_gate"]["message"]


def test_beta_readiness_requires_activation_release_fixture(tmp_path: Path) -> None:
    """Beta readiness should reject release gates without checked activation evidence."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    for path in (tmp_path / "reports" / "activation-release").glob("*.jsonl"):
        path.unlink()

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["activation_release_fixture"]["status"] == "error"
    assert "reports/activation-release" in checks["activation_release_fixture"]["message"]
    assert checks["activation_release_fixture"]["action"] == (
        "Restore the activation fixture used by scripts/release-check.sh."
    )


def test_beta_readiness_rejects_tampered_activation_release_fixture(tmp_path: Path) -> None:
    """Beta readiness should reject activation evidence with a broken Eventloom hash chain."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "reports" / "activation-release" / "agent-1.jsonl"
    fixture.write_text(
        fixture.read_text(encoding="utf-8").replace("release-gate", "tampered-release-gate", 1),
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["activation_release_fixture"]["status"] == "error"
    assert "integrity" in checks["activation_release_fixture"]["message"]
    assert checks["activation_release_fixture"]["action"] == (
        "Restore the activation fixture used by scripts/release-check.sh."
    )


def test_beta_readiness_rejects_inefficient_activation_release_fixture(tmp_path: Path) -> None:
    """Beta readiness should reject activation fixtures that cannot meet release token guardrails."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "reports" / "activation-release" / "agent-1.jsonl"
    fixture.unlink()
    log = EventLog(fixture)
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={
            "source": "release-gate",
            "token_efficiency": {
                "prompt_tokens": 6000,
                "current_fact_count": 1,
                "evidence_count": 1,
                "facts_per_1k_prompt_tokens": 0.05,
            },
        },
        thread="agent-1",
        timestamp=now - timedelta(minutes=5),
    )
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "release-gate", "role": "assistant"},
        thread="agent-1",
        timestamp=now - timedelta(minutes=1),
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["activation_release_fixture"]["status"] == "error"
    assert "prompt_tokens=6000 exceeds 5000" in checks["activation_release_fixture"]["message"]
    assert "facts_per_1k_prompt_tokens=0.05 is below 0.1" in checks["activation_release_fixture"]["message"]


def test_beta_readiness_rejects_stale_activation_release_fixture(tmp_path: Path) -> None:
    """Beta readiness should reject activation fixtures whose checkout is stale at release time."""
    _write_minimal_beta_ready_project(tmp_path)
    (tmp_path / "BETA.md").write_text(
        "# Beta Roadmap\n\n- remaining work\n- release criteria\n",
        encoding="utf-8",
    )
    fixture = tmp_path / "reports" / "activation-release" / "agent-1.jsonl"
    fixture.unlink()
    log = EventLog(fixture)
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={
            "source": "release-gate",
            "token_efficiency": {
                "prompt_tokens": 400,
                "current_fact_count": 2,
                "evidence_count": 2,
                "facts_per_1k_prompt_tokens": 5.0,
            },
        },
        thread="agent-1",
        timestamp=now - timedelta(hours=3),
    )
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "release-gate", "role": "assistant"},
        thread="agent-1",
        timestamp=now - timedelta(minutes=1),
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["activation_release_fixture"]["status"] == "error"
    assert "checkout age 180.0 minutes exceeds 120 minutes" in checks["activation_release_fixture"]["message"]


def _write_minimal_beta_ready_project(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "docs" / "examples").mkdir()
    (root / "examples").mkdir()
    report_dir = root / "reports" / "benchmarks" / "coordination-real-v1"
    purpose_report_dir = root / "reports" / "benchmarks" / "purpose-v1"
    manifest_dir = report_dir / "competitor-runner-manifests"
    manifest_dir.mkdir(parents=True)
    purpose_report_dir.mkdir(parents=True)
    activation_log = EventLog(root / "reports" / "activation-release" / "agent-1.jsonl")
    now = datetime(2026, 5, 20, 12, 0, tzinfo=UTC)
    activation_log.append(
        "memory.checkout.completed",
        actor="zaxy-memory",
        payload={
            "source": "release-gate",
            "token_efficiency": {
                "prompt_tokens": 400,
                "current_fact_count": 2,
                "evidence_count": 2,
                "facts_per_1k_prompt_tokens": 5.0,
            },
        },
        thread="agent-1",
        timestamp=now - timedelta(minutes=5),
    )
    activation_log.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "release-gate", "role": "assistant"},
        thread="agent-1",
        timestamp=now - timedelta(minutes=1),
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.2.0 - 2026-05-11\n\n- Stable release.\n",
        encoding="utf-8",
    )
    (root / "docs" / "benchmarks.md").write_text(
        "CoordinationBench competitor_claim_gate blocks public same-harness claims. "
        "Run `zaxy coordinate benchmark --require-competitor-claim quarq "
        "--require-competitor-claim hybi ...`; otherwise Quarq/Hybi remain "
        "disclosure-only. The purpose-v1 benchmark blocks Semantic Reach and "
        "Quarq comparative claims until same-harness adapters are pinned. "
        "Public-derived purpose holdouts are diagnostic and separate from lanes.\n",
        encoding="utf-8",
    )
    (root / "docs" / "coordinate-roadmap.md").write_text(
        "The CoordinationBench public-claim gate is now implemented for Quarq and Semantic Reach/Hybi.\n",
        encoding="utf-8",
    )
    coordination_report = {
        "version": "coordination-real-v1",
        "workload_fingerprint": "fixture-fingerprint",
        "metrics": {},
        "baselines": {},
        "cases": [],
        "competitor_claim_gate": {
            "status": "blocked",
            "required_adapters": ["quarq", "hybi"],
            "completed_adapters": [],
            "blocked_adapters": {
                "quarq": "adapter status is not_run/disclosure_only",
                "hybi": "adapter status is not_run/disclosure_only",
            },
            "message": "Public same-harness competitor claims are blocked.",
        },
        "competitor_adapters": {
            "quarq": {
                "name": "quarq",
                "display_name": "Quarq",
                "adapter_contract": "coordinationbench-v1",
                "status": "not_run",
                "claim_status": "disclosure_only",
                "blockers": ["No pinned runner."],
                "metrics": None,
                "result_audit": None,
            },
            "hybi": {
                "name": "hybi",
                "display_name": "Semantic Reach / HyperBinder / Hybi",
                "adapter_contract": "coordinationbench-v1",
                "status": "not_run",
                "claim_status": "disclosure_only",
                "blockers": ["No pinned runner."],
                "metrics": None,
                "result_audit": None,
            },
        },
    }
    (report_dir / "coordination-benchmark.json").write_text(
        json.dumps(coordination_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / "coordination-benchmark.md").write_text(
        "## Competitor Claim Gate\n\nQuarq and Semantic Reach / HyperBinder / Hybi are disclosure-only.\n",
        encoding="utf-8",
    )
    for name in ("quarq", "hybi"):
        (manifest_dir / f"{name}.runner-manifest.template.json").write_text(
            json.dumps({"name": name, "template": True, "workload_fingerprint": "fixture-fingerprint"}),
            encoding="utf-8",
        )
    purpose_lanes = [
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
    ]
    purpose_report = {
        "version": "purpose-v1",
        "status": "passed",
        "lane_count": len(purpose_lanes),
        "passed_lanes": len(purpose_lanes),
        "lanes": [
            {
                "name": name,
                "score": 1.0,
                "threshold": 1.0,
                "status": "passed",
                "measurement": "fixture",
                "evidence": (
                    {
                        "security": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "release": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "coordinate": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "support": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "product": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "sales": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "legal": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                        "executive": {
                            "unsupported": {"satisfied": False},
                            "supported": {"satisfied": True},
                        },
                    }
                    if name == "Evidence Policy Discipline"
                    else {
                        "passed_profiles": ["support", "product", "sales", "legal", "executive"],
                        "local_project_memory_positioning": True,
                    }
                    if name == "Broader Profile Fixtures"
                    else {
                        "ingestion_audit": {"safe": True},
                        "purpose_projections": {
                            "support": {},
                            "product": {},
                            "legal": {},
                            "executive": {},
                        },
                    }
                    if name == "Neutral Substrate Projection"
                    else {}
                ),
            }
            for name in purpose_lanes
        ],
        "competitor_claim_status": "blocked",
        "holdout_reports": {
            "public-derived-purpose-v1": {
                "pack_id": "public-derived-purpose-v1",
                "claim_status": "public_derived_holdout",
                "gate_status": "diagnostic",
                "pack_fingerprint": "fixture-holdout-fingerprint",
                "metrics": {"case_count": 5},
            }
        },
        "competitor_claim_blockers": [
            "Semantic Reach and Quarq require pinned same-harness adapters.",
        ],
        "generated_at": "2026-06-02T00:00:00+00:00",
        "elapsed_ms": 1.0,
    }
    (purpose_report_dir / "purpose-benchmark.json").write_text(
        json.dumps(purpose_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (purpose_report_dir / "purpose-benchmark.md").write_text(
        "# Purpose Benchmark\n\nPurpose Recall\nAccepted-State Discipline\nPublic-Derived Holdouts\n",
        encoding="utf-8",
    )
    holdout_dir = purpose_report_dir / "holdouts" / "public-derived-purpose-v1"
    holdout_dir.mkdir(parents=True)
    (holdout_dir / "holdout-pack.json").write_text(
        json.dumps(
            {
                "schema_version": "purpose-holdout-pack-v1",
                "pack_id": "public-derived-purpose-v1",
                "claim_status": "public_derived_holdout",
                "fingerprint": "fixture-holdout-fingerprint",
            }
        ),
        encoding="utf-8",
    )
    (root / "docs" / "examples" / "first-run-timing-report.json").write_text(
        json.dumps(
            {
                "threshold_seconds": 300,
                "time_to_successful_doctor_seconds": 240,
                "time_to_first_successful_example_seconds": 270,
                "requires_sidecar": False,
            }
        ),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "publish.yml").write_text(
        "on:\n"
        "  release:\n"
        "    types: [published]\n"
        "  workflow_dispatch:\n"
        "permissions:\n"
        "  id-token: write\n"
        "steps:\n"
        "  - run: python -m build --sdist --wheel\n"
        "  - run: python -m twine check dist/*\n"
        "  - uses: pypa/gh-action-pypi-publish@release/v1\n",
        encoding="utf-8",
    )
    (root / "examples" / "langgraph_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'langgraph-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (root / "examples" / "openai_compatible_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'openai-compatible-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (root / "examples" / "claude_compatible_memory.py").write_text(
        "import json\n"
        "print(json.dumps({'session_id': 'claude-compatible-demo', 'has_zaxy_context': True, 'kind': 'memory_checkout'}))\n",
        encoding="utf-8",
    )
    (root / "scripts" / "release-check.sh").write_text(
        'RUFF_CMD="ruff"\n'
        'MYPY_CMD="mypy"\n'
        'EXAMPLES_SMOKE_CMD="pytest tests/test_examples_v05.py --no-cov -q"\n'
        'MCP_SMOKE_CMD="python scripts/mcp_smoke_test.py"\n'
        'LANGGRAPH_SMOKE_CMD="pytest tests/test_examples_v05.py::test_langgraph_example_runs_without_langgraph_dependency --no-cov -q"\n'
        'COORDINATE_SMOKE_CMD="pytest tests/test_examples_v05.py::test_coordinate_three_worker_example_runs --no-cov -q"\n'
        'DOCS_CMD="scripts/validate-docs.sh"\n'
        'BETA_UAT_CMD="scripts/beta-uat.sh"\n'
        'EXTERNAL_VALIDATION_CMD="SKIP:external validation is optional for v1.0 release"\n'
        "run_gate() { [[ \"$2\" == SKIP:* ]] && echo \"Skipping $1: ${2#SKIP:}\" || bash -c \"$2\"; }\n"
        "pytest\n"
        "scripts/check-coverage.py\n"
        "tests/test_packet_memory_e2e.py\n"
        "scripts/build-dist.sh\n"
        "scripts/validate-docs.sh\n"
        "scripts/validate-deployment.sh\n"
        "PYTHONPATH=src python -m zaxy hook-status\n"
        "--eventloom-path reports/activation-release\n"
        "--now 2026-05-20T12:00:00+00:00\n"
        "--min-activation-rate 1.0\n"
        "--max-checkout-prompt-tokens 5000\n"
        "--min-checkout-facts-per-1k-tokens 0.1\n"
        'BACKEND_SHOOTOUT_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/backend-shootout.json '
        "--require-report-metadata --require-markdown-report --require-query-results --require-git-tracked-inputs "
        "--verify-report-fingerprints --require-backends embedded,bm25 "
        '--require-labeled-metrics --require-dashboard-source embedded=embedded '
        "--min-answer-at-5 0.5 --min-recall-at-5 0.5 --min-citation-coverage 1.0 "
        "--min-quality-per-1k-injected-tokens embedded=1.0 "
        "--min-answer-at-5-per-1k-injected-tokens embedded=1.0 "
        "--max-checkout-p99-ms embedded=25 "
        '--forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_PERFORMANCE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-40-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        "--require-dashboard-source embedded=embedded --min-citation-coverage 1.0 "
        "--min-quality-per-1k-returned-tokens embedded=0.10 "
        "--min-answer-at-5-per-1k-returned-tokens embedded=0.10 "
        "--min-quality-per-1k-injected-tokens embedded=0.10 "
        "--min-answer-at-5-per-1k-injected-tokens embedded=0.10 "
        "--max-checkout-p95-ms embedded=100 --max-checkout-p99-ms embedded=85 "
        '--forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_SCALE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-100-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        '--require-dashboard-source embedded=embedded --forbid-backends neo4j,pggraph,latticedb '
        "--min-recall-at-5 0.90 --min-citation-coverage 1.0 "
        "--min-quality-per-1k-returned-tokens embedded=0.15 "
        "--min-answer-at-5-per-1k-returned-tokens embedded=0.15 "
        "--min-quality-per-1k-injected-tokens embedded=0.15 "
        "--min-answer-at-5-per-1k-injected-tokens embedded=0.15 "
        "--max-checkout-p99-ms embedded=250 "
        '--max-checkout-p95-ms embedded=200"\n'
        "--require-report-metadata\n"
        "--require-markdown-report\n"
        "--require-query-results\n"
        "--require-git-tracked-inputs\n"
        "--verify-report-fingerprints\n"
        "backend-shootout.json\n"
        "longmemeval-40-backend-shootout.json\n"
        "longmemeval-100-backend-shootout.json\n"
        "--min-quality-per-1k-injected-tokens embedded=1.0\n"
        "--min-citation-coverage 1.0\n"
        "--max-checkout-p95-ms embedded=200\n"
        "--min-quality-per-1k-returned-tokens\n"
        "--min-answer-at-5-per-1k-returned-tokens\n"
        "--min-quality-per-1k-injected-tokens\n"
        "--min-answer-at-5-per-1k-injected-tokens\n"
        "--max-cold-bootstrap-ms\n"
        "--max-first-checkout-ms\n"
        "--max-append-to-projection-p95-ms\n"
        "--max-resident-memory-delta-bytes\n"
        "--max-on-disk-footprint-bytes\n"
        "--max-dashboard-graph-load-ms\n"
        "--max-checkout-p99-ms\n"
        "--max-exact-p99-ms\n"
        "--max-keyword-p95-ms\n"
        "--max-keyword-p99-ms\n"
        "--max-vector-p99-ms\n"
        "--max-traversal-p99-ms\n",
        encoding="utf-8",
    )
    (root / "scripts" / "beta-uat.sh").write_text(
        "mktemp -d\n"
        "python -m pip install\n"
        "zaxy init local-codex local-claude\n"
        'run_workspace "embedded" "" "start"\n'
        "if [[ -n \"${preset}\" ]]\n"
        '"${preset}"\n'
        'grep -q "PROJECTION_BACKEND=embedded" .env.local\n'
        'grep -q "NEO4J_AUTO_START=false" .env.local\n'
        'grep -q "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" .env.local\n'
        "zaxy memory bootstrap\n"
        "zaxy memory checkout\n"
        "zaxy doctor\n"
        "zaxy hook-status --min-activation-rate 1.0 --max-checkout-prompt-tokens 5000 "
        "--min-checkout-facts-per-1k-tokens 0.1\n"
        "zaxy capture status\n"
        "zaxy capture soak\n"
        "zaxy memory status\n"
        "zaxy memory status --eventloom-path .eventloom --graph\n"
        "Graph projection (backend=embedded):\n"
        "zaxy memory inferred-status --session-id\n"
        '"backend": "embedded"\n'
        "zaxy reproject\n"
        "using embedded\n",
        encoding="utf-8",
    )
    docs = (
        "pipx install zaxy-memory\n"
        "zaxy init\n"
        "zaxy memory bootstrap\n"
        "zaxy memory checkout\n"
        "scripts/beta-uat.sh\n"
        "zaxy doctor --beta-readiness\n"
        "deterministic\n"
        "zaxy capture start\n"
        "zaxy capture status\n"
        "zaxy capture soak\n"
        "zaxy hook-status\n"
        "observation coverage\n"
    )
    (root / "README.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "getting-started.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "testing.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "hooks.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "mcp.md").write_text(docs, encoding="utf-8")


def _write_backend_report_inputs(root: Path) -> None:
    reports = root / "reports" / "backend-shootout"
    reports.mkdir(parents=True, exist_ok=True)
    eventloom = reports / "sample.eventloom"
    eventloom.mkdir()
    (eventloom / "agent-1.jsonl").write_text(
        '{"seq":1,"type":"decision.recorded","payload":{},"thread":"agent-1"}\n',
        encoding="utf-8",
    )
    queries = reports / "queries.json"
    queries.write_text('[{"query":"embedded benchmark evidence"}]\n', encoding="utf-8")
    payload = {
        "eventloom_path": "reports/backend-shootout/sample.eventloom",
        "queries_file": "reports/backend-shootout/queries.json",
        "query_results": {
            "embedded:retrieve": [
                {
                    "query": "embedded benchmark evidence",
                    "citation_coverage": 1.0,
                    "checkout_latency_ms": 18.0,
                }
            ]
        },
    }
    for filename in (
        "backend-shootout.json",
        "longmemeval-40-backend-shootout.json",
        "longmemeval-100-backend-shootout.json",
    ):
        (reports / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_build_dist_fails_fast_when_build_fails(tmp_path: Path) -> None:
    """Metadata checks should not run if artifact creation failed."""
    log_path = tmp_path / "commands.log"
    root = tmp_path / "project"
    dist = tmp_path / "dist"
    root.mkdir()
    fail = tmp_path / "fail.sh"
    twine = tmp_path / "twine.sh"
    fail.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"build $*\" >> \"$PACKAGE_CHECK_LOG\"\n"
        "exit 9\n",
        encoding="utf-8",
    )
    twine.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"twine $*\" >> \"$PACKAGE_CHECK_LOG\"\n",
        encoding="utf-8",
    )
    fail.chmod(0o700)
    twine.chmod(0o700)

    result = subprocess.run(
        [
            "bash",
            "scripts/build-dist.sh",
            "--root",
            str(root),
            "--dist-dir",
            str(dist),
            "--build-cmd",
            str(fail),
            "--twine-cmd",
            str(twine),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
        env={"PACKAGE_CHECK_LOG": str(log_path)},
    )

    assert result.returncode == 9
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        f"build --sdist --wheel --outdir {dist} {root}",
    ]


def test_ci_runs_distribution_artifact_gate() -> None:
    """CI should build and validate release artifacts on every PR."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "package:" in workflow
    assert "scripts/build-dist.sh --root ." in workflow


def test_github_workflows_opt_into_node24_action_runtime() -> None:
    """JavaScript actions should opt into Node 24 before GitHub removes Node 20."""
    for workflow_path in Path(".github/workflows").glob("*.yml"):
        workflow = workflow_path.read_text(encoding="utf-8")
        if "uses:" not in workflow:
            continue
        assert "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true" in workflow, workflow_path


def test_ci_disables_benchmark_timing_for_correctness_matrix() -> None:
    """Performance benchmarks should not make the Python matrix flaky."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'pytest -m "not integration" --benchmark-disable --cov --cov-report=xml' in workflow


def test_publish_workflow_publishes_release_artifacts_to_pypi() -> None:
    """Published releases should build artifacts and upload them via trusted publishing."""
    workflow = Path(".github/workflows/publish.yml").read_text(encoding="utf-8")

    assert "release:" in workflow
    assert "types: [published]" in workflow
    assert "workflow_dispatch:" in workflow
    assert "id-token: write" in workflow
    assert "python -m build --sdist --wheel" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "PYPI_API_TOKEN" not in workflow
    assert "password:" not in workflow
    assert "https://pypi.org/project/zaxy-memory/" in workflow


def test_readme_documents_trusted_publishing_release_path() -> None:
    """Public release docs should not instruct maintainers to rely on PyPI tokens."""
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "PyPI Trusted Publishing" in readme
    assert "zaxy doctor --release-smoke" in readme
    assert "dependency-light LangGraph example" in readme
    assert "PYPI_API_TOKEN" not in readme
