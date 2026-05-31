"""Tests for the coverage ratchet gate."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path


def _coverage_xml(path: Path, line_rate: float) -> None:
    path.write_text(
        f"""<?xml version="1.0" ?>
<coverage version="7.1.0" line-rate="{line_rate}" branch-rate="0" lines-covered="0" lines-valid="0">
  <packages />
</coverage>
""",
        encoding="utf-8",
    )


def test_coverage_ratchet_fails_below_configured_floor(tmp_path: Path) -> None:
    """The ratchet should fail when XML coverage drops below the configured floor."""
    coverage_xml = tmp_path / "coverage.xml"
    _coverage_xml(coverage_xml, 0.9199)

    result = subprocess.run(
        [
            "python",
            "scripts/check-coverage.py",
            "--root",
            ".",
            "--coverage-xml",
            str(coverage_xml),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "91.99%" in result.stderr
    assert "92.00%" in result.stderr


def test_coverage_ratchet_passes_at_configured_floor(tmp_path: Path) -> None:
    """Coverage equal to the floor should pass so the ratchet is deterministic."""
    coverage_xml = tmp_path / "coverage.xml"
    _coverage_xml(coverage_xml, 0.92)

    result = subprocess.run(
        [
            "python",
            "scripts/check-coverage.py",
            "--root",
            ".",
            "--coverage-xml",
            str(coverage_xml),
        ],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Coverage ratchet passed" in result.stdout


def test_coverage_ratchet_floor_is_declared_in_pyproject() -> None:
    """The ratchet floor should be visible as project configuration."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["tool"]["coverage"]["report"]["fail_under"] == 92
    assert pyproject["tool"]["zaxy"]["coverage"]["min_total_percent"] == "92.00"


def test_ci_and_release_gate_run_coverage_ratchet() -> None:
    """CI and release checks should enforce the same coverage ratchet."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    release_check = Path("scripts/release-check.sh").read_text(encoding="utf-8")

    assert "scripts/check-coverage.py" in workflow
    assert "scripts/check-coverage.py" in release_check


def test_testing_docs_explain_coverage_ratchet() -> None:
    """Testing docs should explain both the broad gate and the ratchet floor."""
    docs = Path("docs/testing.md").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "coverage ratchet" in docs
    assert "92.00%" in docs
    assert "--cov-fail-under=92" in docs
    assert "Coverage gate: \u226592%" in readme
    assert "scripts/check-coverage.py" in docs
