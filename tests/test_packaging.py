"""Tests for release packaging metadata and artifact gates."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from zaxy.event import EventLog
from zaxy.release import package_version, run_beta_readiness


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


def test_core_install_includes_embedded_default_backend() -> None:
    """The embedded backend should be part of the core install, not duplicated in an extra."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    assert "kuzu>=0.11.0" in dependencies
    assert extras["embedded"] == []
    assert "neo4j>=5.20.0" not in dependencies
    assert extras["neo4j"] == ["neo4j>=5.20.0"]


def test_package_keywords_center_embedded_local_memory() -> None:
    """Package discovery metadata should match the embedded-first runtime."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    keywords = pyproject["project"]["keywords"]
    assert "embedded-memory" in keywords
    assert "kuzu" in keywords
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
    live_benchmark_source = Path("src/zaxy/live_benchmark.py").read_text(encoding="utf-8")

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
    assert result.stdout.splitlines() == ["zaxy 0.4.0", "exit=0", "False", "False", "False"]


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
                "loaded = sorted(name for name in sys.modules if name.startswith('zaxy.') and name != 'zaxy.__main__')\n"
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

    assert package_version() == "0.4.0"


def test_package_version_prefers_source_tree_version_in_editable_checkout(monkeypatch) -> None:
    """Editable installs should not report stale metadata after a release version bump."""
    from zaxy import release

    monkeypatch.setattr(release.metadata, "version", lambda _name: "0.1.0")

    assert package_version() == "0.4.0"


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
    assert 'run_workspace "claude-code" "local-claude" "status"' in script
    assert "zaxy memory bootstrap" in script
    assert "zaxy memory checkout" in script
    assert "zaxy doctor" in script
    assert "zaxy hook-status" in script
    assert "zaxy capture status" in script
    assert "zaxy capture-soak" in script
    assert "zaxy memory status" in script


def test_beta_uat_script_exercises_bare_embedded_init_path() -> None:
    """UAT should protect bare init as the no-sidecar embedded default."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert 'run_workspace "embedded" "" "status"' in script
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
    assert "zaxy capture-soak --eventloom-path .eventloom --workspace-root . --session-id" in script


def test_beta_uat_script_enforces_activation_efficiency_guardrail() -> None:
    """UAT should fail if clean first-run sessions are captured without fresh checkout."""
    script = Path("scripts/beta-uat.sh").read_text(encoding="utf-8")

    assert "zaxy hook-status --eventloom-path .eventloom --min-activation-rate 1.0" in script
    assert "--max-checkout-prompt-tokens 5000" in script
    assert "--min-checkout-facts-per-1k-tokens 0.1" in script
    assert script.index("--min-activation-rate 1.0") < script.index("zaxy capture-soak")


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
    assert checks["beta_roadmap"]["status"] == "error"
    assert checks["beta_roadmap"]["action"] == "Add BETA.md with beta goals, remaining work, gates, and exit criteria."


def test_beta_roadmap_tracks_post_uat_product_work() -> None:
    """The beta roadmap should point beyond gate plumbing into product-grade memory behavior."""
    roadmap = Path("BETA.md").read_text(encoding="utf-8")

    assert "Git for LLM memory" in roadmap
    assert "MemPalace-comparable" in roadmap
    assert "temporal recall" in roadmap
    assert "source recall" in roadmap
    assert "graph traversal" in roadmap
    assert "context-collapse" in roadmap
    assert "zaxy benchmark-inventory" in roadmap
    assert "CrewAI" in roadmap
    assert "capture soak" in roadmap
    assert "zaxy capture-soak" in roadmap
    assert "release criteria" in roadmap


def test_beta_readiness_reports_missing_clean_repo_uat(tmp_path: Path) -> None:
    """Beta readiness should fail clearly when the clean-repo UAT harness is absent."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "zaxy-memory"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## 0.2.0 - 2026-05-11\n\n- Stable release.\n",
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
    (tmp_path / "scripts" / "release-check.sh").write_text(
        'RUFF_CMD="ruff"\n'
        'MYPY_CMD="mypy"\n'
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
        '--forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_PERFORMANCE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-40-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        '--require-dashboard-source embedded=embedded --forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_SCALE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-100-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        '--require-dashboard-source embedded=embedded --forbid-backends neo4j,pggraph,latticedb '
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
        script.replace('run_workspace "embedded" "" "status"\n', "")
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
    (root / "scripts" / "release-check.sh").write_text(
        'RUFF_CMD="ruff"\n'
        'MYPY_CMD="mypy"\n'
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
        '--forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_PERFORMANCE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-40-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        '--require-dashboard-source embedded=embedded --forbid-backends neo4j,pggraph,latticedb"\n'
        'BACKEND_SCALE_CMD="python scripts/check-backend-shootout.py '
        "reports/backend-shootout/longmemeval-100-backend-shootout.json --require-report-metadata "
        "--require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints "
        "--require-backends embedded,bm25 --require-labeled-metrics "
        '--require-dashboard-source embedded=embedded --forbid-backends neo4j,pggraph,latticedb '
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
        'run_workspace "embedded" "" "status"\n'
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
        "zaxy capture-soak\n"
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
        "zaxy capture-soak\n"
        "zaxy hook-status\n"
        "observation coverage\n"
    )
    (root / "README.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "getting-started.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "testing.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "hooks.md").write_text(docs, encoding="utf-8")
    (root / "docs" / "mcp.md").write_text(docs, encoding="utf-8")


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
    assert "PYPI_API_TOKEN" not in readme
