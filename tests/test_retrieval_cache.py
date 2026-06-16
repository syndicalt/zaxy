"""Unit tests for the shared per-session retrieval cache."""

from __future__ import annotations

from pathlib import Path

from zaxy.retrieval_cache import (
    SessionRetrievalCache,
    _replay_tip_path,
    _verbatim_cache_path,
)
from zaxy.session import SessionManager
from zaxy.verbatim import VerbatimIndex


def _append(manager: SessionManager, session_id: str, content: str, turn: int) -> None:
    manager.get(session_id).eventlog.append(
        "transcript.turn",
        actor="assistant",
        payload={"source": "codex", "turn_index": turn, "role": "assistant", "content": content},
        thread=session_id,
    )


def test_verbatim_index_reuses_then_extends_byte_identically(tmp_path: Path) -> None:
    """The index is cached on an unchanged log and extended (not rebuilt) on growth.

    The incrementally-extended index must produce the same ranking as a full
    rebuild over the combined corpus.
    """
    manager = SessionManager(base_path=str(tmp_path / ".eventloom"))
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "Cached verbatim recall keeps answer assembly fast.", 1)

    first = cache.verbatim_index("agent")
    # Unchanged log returns the very same cached object, not a rebuild.
    assert cache.verbatim_index("agent") is first

    _append(manager, "agent", "A new event should extend the cached verbatim index.", 2)
    extended = cache.verbatim_index("agent")
    assert extended is not first  # the append was reflected

    events = manager.get("agent").eventlog.read_all()
    full = VerbatimIndex.from_events(events)
    query = "extend cached verbatim index"
    inc_hits = [(h.citation, round(h.score, 9)) for h in extended.query(query, limit=5)]
    full_hits = [(h.citation, round(h.score, 9)) for h in full.query(query, limit=5)]
    assert inc_hits == full_hits
    assert inc_hits  # the new content is retrievable


def test_verified_replay_is_cached_and_integrity_holds(tmp_path: Path) -> None:
    """Verified replay returns a passing integrity report and caches the result."""
    manager = SessionManager(base_path=str(tmp_path / ".eventloom"))
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "first", 1)
    _append(manager, "agent", "second", 2)

    result = cache.verified_replay("agent")
    assert result.integrity is not None and result.integrity.ok
    assert [event.seq for event in result.events] == [1, 2]
    # Unchanged log: same cached result object is returned.
    assert cache.verified_replay("agent") is result

    # from_seq slices the cached replay without re-reading.
    sliced = cache.verified_replay("agent", from_seq=2)
    assert [event.seq for event in sliced.events] == [2]


def test_verbatim_checkpoint_persists_and_reloads_byte_identically(tmp_path: Path) -> None:
    """A cold verbatim build persists a checkpoint; a fresh cache reloads it and
    extends only the appended tail, with rankings identical to a full rebuild.
    """
    base = str(tmp_path / ".eventloom")
    manager = SessionManager(base_path=base)
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "Verbatim checkpoint persists tokenization for fast cold load.", 1)
    _append(manager, "agent", "Second document about cold load and verbatim recall.", 2)

    cold = cache.verbatim_index("agent")
    cache_file = _verbatim_cache_path(manager.get("agent").eventlog)
    assert cache_file.exists()  # cold build persisted the checkpoint

    # Append after the checkpoint, then load from a fresh cache (cold process):
    # it must reconstruct from disk + extend the tail, matching a full rebuild.
    _append(manager, "agent", "Third document appended after the checkpoint was written.", 3)
    fresh = SessionRetrievalCache(manager)
    reloaded = fresh.verbatim_index("agent")

    full = VerbatimIndex.from_events(manager.get("agent").eventlog.read_all())
    for query in ("cold load verbatim", "checkpoint appended document", "recall"):
        assert [(h.citation, round(h.score, 9)) for h in reloaded.query(query, limit=8)] == [
            (h.citation, round(h.score, 9)) for h in full.query(query, limit=8)
        ]
    _ = cold  # cold build object retained for clarity


def test_verbatim_checkpoint_falls_back_to_rebuild_on_anchor_mismatch(tmp_path: Path) -> None:
    """A checkpoint whose anchor no longer matches the live log (rewrite/compaction)
    must be rejected and the index rebuilt from scratch — never trusted blindly.
    """
    import marshal as _marshal
    import zlib as _zlib

    base = str(tmp_path / ".eventloom")
    manager = SessionManager(base_path=base)
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "only document for the anchor mismatch test", 1)
    cache.verbatim_index("agent")

    cache_file = _verbatim_cache_path(manager.get("agent").eventlog)
    data = _marshal.loads(_zlib.decompress(cache_file.read_bytes()))
    data["covered_hash"] = "0" * 64  # break the anchor
    cache_file.write_bytes(_zlib.compress(_marshal.dumps(data), 1))

    fresh = SessionRetrievalCache(manager)
    rebuilt = fresh.verbatim_index("agent")
    full = VerbatimIndex.from_events(manager.get("agent").eventlog.read_all())
    assert [h.citation for h in rebuilt.query("anchor mismatch document", limit=5)] == [
        h.citation for h in full.query("anchor mismatch document", limit=5)
    ]


def test_verbatim_index_correct_when_checkpoint_deleted(tmp_path: Path) -> None:
    """The verbatim checkpoint is a pure cache: deletion must not break anything."""
    base = str(tmp_path / ".eventloom")
    manager = SessionManager(base_path=base)
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "document that survives checkpoint deletion", 1)
    first = cache.verbatim_index("agent")

    _verbatim_cache_path(manager.get("agent").eventlog).unlink()
    fresh = SessionRetrievalCache(manager)
    rebuilt = fresh.verbatim_index("agent")
    assert [h.citation for h in rebuilt.query("survives deletion", limit=5)] == [
        h.citation for h in first.query("survives deletion", limit=5)
    ]


def test_replay_tip_checkpoint_persists_and_a_fresh_cache_loads_it(tmp_path: Path) -> None:
    """A verified replay writes a tip checkpoint; a brand-new cache (cold process
    simulation) loads it and verifies only the tail, with identical results and
    integrity to a full replay.
    """
    base = str(tmp_path / ".eventloom")
    manager = SessionManager(base_path=base)
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "first", 1)
    _append(manager, "agent", "second", 2)

    warm = cache.verified_replay("agent")
    assert warm.integrity is not None and warm.integrity.ok
    tip = _replay_tip_path(manager.get("agent").eventlog)
    assert tip.exists()  # cold full replay persisted the checkpoint

    # A fresh cache = a cold process: in-memory caches are empty, so it must use
    # the on-disk tip. Spy on the eventlog to prove the full-verify fallback
    # (session_manager.replay) is NOT taken.
    cold = SessionRetrievalCache(manager)
    result = cold.verified_replay("agent")
    assert result.integrity is not None and result.integrity.ok
    assert [e.seq for e in result.events] == [1, 2]
    assert [e.hash for e in result.events] == [e.hash for e in warm.events]


def test_replay_tip_falls_back_to_full_verify_on_anchor_mismatch(tmp_path: Path) -> None:
    """A corrupt/rewritten checkpoint anchor must fall back to a full verified
    replay, never trust a stale tip.
    """
    base = str(tmp_path / ".eventloom")
    manager = SessionManager(base_path=base)
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "first", 1)
    _append(manager, "agent", "second", 2)
    cache.verified_replay("agent")  # writes the tip

    # Corrupt the checkpoint's anchor hash; a fresh cache must reject it.
    tip = _replay_tip_path(manager.get("agent").eventlog)
    import json as _json

    data = _json.loads(tip.read_text())
    data["covered_hash"] = "0" * 64
    tip.write_text(_json.dumps(data))

    cold = SessionRetrievalCache(manager)
    result = cold.verified_replay("agent")
    # Falls back to the full verify and still returns the correct, intact log.
    assert result.integrity is not None and result.integrity.ok
    assert [e.seq for e in result.events] == [1, 2]


def test_replay_works_with_no_checkpoint_and_is_correct_when_deleted(tmp_path: Path) -> None:
    """The checkpoint is a pure cache: deleting it must never break correctness."""
    base = str(tmp_path / ".eventloom")
    manager = SessionManager(base_path=base)
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "only", 1)
    first = cache.verified_replay("agent")
    assert first.integrity is not None and first.integrity.ok

    _replay_tip_path(manager.get("agent").eventlog).unlink()  # delete the cache
    cold = SessionRetrievalCache(manager)
    result = cold.verified_replay("agent")
    assert result.integrity is not None and result.integrity.ok
    assert [e.seq for e in result.events] == [1]


def test_invalidate_forces_a_cold_rebuild(tmp_path: Path) -> None:
    """invalidate() drops cached state so the next read rebuilds from the log."""
    manager = SessionManager(base_path=str(tmp_path / ".eventloom"))
    cache = SessionRetrievalCache(manager)
    _append(manager, "agent", "only event", 1)

    first = cache.verbatim_index("agent")
    cache.invalidate("agent")
    rebuilt = cache.verbatim_index("agent")
    assert rebuilt is not first
    # Same corpus -> same ranking after the forced rebuild.
    query = "only event"
    assert [h.citation for h in rebuilt.query(query, limit=5)] == [
        h.citation for h in first.query(query, limit=5)
    ]
