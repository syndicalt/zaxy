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
    assert "temporal knowledge graph fabric" not in text.splitlines()[3]


def test_why_zaxy_states_coordinator_memory_thesis() -> None:
    """Why Zaxy should explain the product thesis before architecture."""
    text = Path("docs/why-zaxy.md").read_text(encoding="utf-8")

    assert "Coordinator Memory for Agent Teams" in text
    assert "worker-local findings" in text
    assert "accepted parent mission state" in text
    assert "vector search alone cannot audit" in text
