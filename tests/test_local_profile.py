"""Tests for offline local retrieval profile helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from zaxy.local_profile import check_local_profile, render_local_profile, write_local_profile


def test_render_local_profile_outputs_offline_env_without_secrets() -> None:
    text = render_local_profile()

    assert "PROJECTION_BACKEND=embedded" in text
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in text
    assert "EMBEDDING_ENABLED=true" in text
    assert "EMBEDDING_PROVIDER=hash" in text
    assert "RERANKER_PROVIDER=lexical" in text
    assert "NEO4J_AUTO_START=false" in text
    assert "PGGRAPH_AUTO_START=false" in text
    assert "OPENAI_API_KEY" not in text


def test_render_local_profile_can_target_embedded_projection_without_sidecar_autostart() -> None:
    text = render_local_profile(projection_backend="embedded")

    assert "PROJECTION_BACKEND=embedded" in text
    assert "EMBEDDED_GRAPH_PATH=.eventloom/projections/embedded.kuzu" in text
    assert "NEO4J_AUTO_START=false" in text
    assert "PGGRAPH_AUTO_START=false" in text
    assert "OPENAI_API_KEY" not in text


def test_write_local_profile_refuses_to_overwrite_existing_file(tmp_path: Path) -> None:
    target = tmp_path / ".env.local"
    target.write_text("existing=true\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_local_profile(target)


def test_write_local_profile_can_force_overwrite(tmp_path: Path) -> None:
    target = tmp_path / ".env.local"
    target.write_text("existing=true\n", encoding="utf-8")

    write_local_profile(target, force=True)

    assert "EMBEDDING_PROVIDER=hash" in target.read_text(encoding="utf-8")
    assert "PROJECTION_BACKEND=embedded" in target.read_text(encoding="utf-8")


def test_write_local_profile_can_write_embedded_projection_defaults(tmp_path: Path) -> None:
    target = tmp_path / ".env.local"

    write_local_profile(target, projection_backend="embedded")

    text = target.read_text(encoding="utf-8")
    assert "PROJECTION_BACKEND=embedded" in text
    assert "NEO4J_AUTO_START=false" in text


def test_check_local_profile_accepts_deterministic_defaults() -> None:
    report = check_local_profile()

    assert report["embedding_provider"] == "hash"
    assert report["reranker_provider"] == "lexical"
    assert report["status"] == "ok"
