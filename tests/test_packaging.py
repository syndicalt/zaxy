"""Tests for release packaging metadata and artifact gates."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path


def test_pyproject_declares_typed_package_and_release_tools() -> None:
    """The wheel should advertise typing and include build/check tooling."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "zaxy-memory"
    assert pyproject["project"]["urls"] == {
        "Homepage": "https://syndicalt.github.io/zaxy/",
        "Documentation": "https://syndicalt.github.io/zaxy/docs/getting-started.md",
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


def test_changelog_records_initial_pypi_release() -> None:
    """Public releases should have a user-facing changelog entry."""
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "# Changelog" in changelog
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
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        f"build --sdist --wheel --outdir {dist} {root}",
        f"twine check {dist}/*",
    ]


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
