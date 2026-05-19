"""Tests for release packaging metadata and artifact gates."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

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

    assert package_version() == "0.3.0"


def test_package_version_prefers_source_tree_version_in_editable_checkout(monkeypatch) -> None:
    """Editable installs should not report stale metadata after a release version bump."""
    from zaxy import release

    monkeypatch.setattr(release.metadata, "version", lambda _name: "0.1.0")

    assert package_version() == "0.3.0"


def test_changelog_records_initial_pypi_release() -> None:
    """Public releases should have a user-facing changelog entry."""
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "# Changelog" in changelog
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
        "scripts/validate-deployment.sh\n",
        encoding="utf-8",
    )

    report = run_beta_readiness(project_root=tmp_path)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "error"
    assert checks["release_smoke"]["status"] == "ok"
    assert checks["release_gate"]["status"] == "ok"
    assert checks["clean_repo_uat"]["status"] == "error"
    assert checks["clean_repo_uat"]["action"] == (
        "Add a clean-repo UAT script for install, init, bootstrap, capture, and checkout."
    )


def _write_minimal_beta_ready_project(root: Path) -> None:
    (root / "scripts").mkdir()
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "docs").mkdir()
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
        "scripts/validate-deployment.sh\n",
        encoding="utf-8",
    )
    (root / "scripts" / "beta-uat.sh").write_text(
        "mktemp -d\n"
        "python -m pip install\n"
        "zaxy init local-codex local-claude\n"
        "zaxy memory bootstrap\n"
        "zaxy memory checkout\n"
        "zaxy doctor\n"
        "zaxy hook-status\n"
        "zaxy capture status\n"
        "zaxy capture-soak\n"
        "zaxy memory status\n",
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
