"""Tests for v0.5 public positioning and release-gate docs."""

from __future__ import annotations

from pathlib import Path


def test_readme_leads_with_coordinator_memory_positioning() -> None:
    """README should lead with the v0.5 ownable product thesis."""
    text = Path("README.md").read_text(encoding="utf-8")

    assert text.startswith("# Zaxy\n\n**Coordinator Memory for Agent Teams.**")
    assert "auditable, replayable, and coordinated memory" in text
    assert "Eventloom append-only source of truth" in text
    assert "embedded Kuzu graph projection" in text
    intro = "\n".join(text.splitlines()[:10])
    assert "temporal knowledge graph fabric" not in intro


def test_why_zaxy_states_coordinator_memory_thesis() -> None:
    """Why Zaxy should explain the product thesis before architecture."""
    text = Path("docs/why-zaxy.md").read_text(encoding="utf-8")

    assert "Coordinator Memory for Agent Teams" in text
    assert "worker-local findings" in text
    assert "accepted parent mission state" in text
    assert "vector search alone cannot audit" in text


def test_getting_started_documents_five_minute_first_run() -> None:
    """Getting started should provide the measured v0.5 first-run path."""
    text = Path("docs/getting-started.md").read_text(encoding="utf-8")

    assert "Five-minute first run" in text
    assert "pipx install zaxy-memory" in text
    assert "zaxy init" in text
    assert "zaxy memory bootstrap --eventloom-path .eventloom" in text
    assert "zaxy memory checkout" in text
    assert "zaxy doctor --eventloom-path .eventloom" in text
    assert "No Neo4j, Postgres, Docker, or graph password is required" in text


def test_first_run_validation_template_exists() -> None:
    """External validation should have a concrete reporting template."""
    text = Path("docs/first-run-validation.md").read_text(encoding="utf-8")

    assert "# First-Run Validation" in text
    assert "Time to successful `zaxy doctor`" in text
    assert "Where did you get stuck?" in text
    assert "Operating system" in text
    assert "Python version" in text
    assert "docs/examples/first-run-timing-report.json" in text
    assert "300 seconds" in text


def test_mcp_quickstart_names_recommended_local_clients() -> None:
    """MCP quickstart should route users through one clear local path per client."""
    text = Path("docs/mcp-quickstart.md").read_text(encoding="utf-8")

    assert "# MCP Quickstart" in text
    assert "zaxy init" in text
    assert "codex mcp add" in text
    assert "Claude Desktop" in text
    assert "Cursor" in text
    assert "memory_bootstrap" in text
    assert "memory_checkout" in text


def test_coordinate_quickstart_has_full_review_workflow() -> None:
    """Coordinate quickstart should show the whole accepted-state workflow."""
    text = Path("docs/coordinate-quickstart.md").read_text(encoding="utf-8")

    for command in (
        "zaxy coordinate start",
        "zaxy coordinate worker create",
        "zaxy coordinate assign",
        "zaxy coordinate report",
        "zaxy coordinate brief",
        "zaxy coordinate decide",
        "zaxy coordinate promote",
        "zaxy coordinate checkout",
        "zaxy coordinate handoff",
    ):
        assert command in text
    assert "worker-local findings are not trusted parent state" in text


def test_changelog_has_unreleased_v05_section() -> None:
    """v0.5 planning work should have a changelog section before release."""
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "## 0.5.0 - Unreleased" in text
    assert "Coordinator Memory for Agent Teams" in text
    assert "first-run validation" in text
    assert "MCP Quickstart" in text
