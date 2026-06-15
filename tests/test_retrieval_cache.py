"""Unit tests for the shared per-session retrieval cache."""

from __future__ import annotations

from pathlib import Path

from zaxy.retrieval_cache import SessionRetrievalCache
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
