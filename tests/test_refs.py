"""Tests for git-style memory refs."""

from __future__ import annotations

from pathlib import Path

import pytest

from zaxy.refs import MemoryRefStore


def test_ref_store_updates_and_resolves_latest_ref(tmp_path: Path) -> None:
    """A memory ref should durably point to a session event identity."""
    store = MemoryRefStore(tmp_path / ".eventloom")

    first = store.update_ref(
        "refs/heads/main",
        session_id="agent-1",
        target_seq=3,
        target_hash="a" * 64,
        ref_type="branch",
        actor="tester",
    )
    second = store.update_ref(
        "refs/heads/main",
        session_id="agent-1",
        target_seq=4,
        target_hash="b" * 64,
        ref_type="branch",
        actor="tester",
    )

    resolved = store.resolve("refs/heads/main")

    assert first.seq == 1
    assert second.prev_hash == first.hash
    assert resolved is not None
    assert resolved.name == "refs/heads/main"
    assert resolved.session_id == "agent-1"
    assert resolved.target_seq == 4
    assert resolved.target_hash == "b" * 64
    assert resolved.ref_type == "branch"


def test_ref_store_rejects_unsafe_ref_names(tmp_path: Path) -> None:
    """Ref names should not be able to escape the Eventloom ref namespace."""
    store = MemoryRefStore(tmp_path / ".eventloom")

    with pytest.raises(ValueError, match="Invalid memory ref"):
        store.update_ref("../main", session_id="agent-1", target_seq=1, target_hash="a" * 64)
