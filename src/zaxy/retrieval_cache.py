"""Per-session, incrementally-extended retrieval caches.

Both the verbatim BM25 index and the verified replay scale with the whole log
if rebuilt from scratch on every read. This module owns the cache state and the
incremental-extension logic so any long-lived holder (the :class:`MemoryFabric`
and the MCP server's checkout front door) can share a single, byte-identical
implementation instead of re-reading and re-hashing the entire log per call.

The two guarantees the callers depend on live here:

- ``verbatim_index`` extends a cached index with only the newly appended events
  (:meth:`VerbatimIndex.append_chunks`); the result is identical to a full
  rebuild over the combined corpus.
- ``verified_replay`` verifies only the appended tail against the cached,
  already-verified prefix. The tail check doubles as a consistency guard: any
  offset skew or log rewrite surfaces as a seq/hash mismatch and falls back to a
  full verified replay, so the fast path never silently misses, duplicates, or
  trusts a bad event. The hash-chain integrity guarantee is preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from zaxy.event import EventLog, IntegrityReport, ReplayResult, verify_event_chain
from zaxy.verbatim import VerbatimIndex, _chunks_from_events

if TYPE_CHECKING:
    from zaxy.session import SessionManager


def _eventlog_file_signature(eventlog: EventLog) -> tuple[int, int]:
    """Return a cheap invalidation signature for a local Eventloom log."""
    try:
        stat = eventlog.path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


class SessionRetrievalCache:
    """Cache verbatim indexes and verified replays per session, incrementally.

    Holds no projection/graph state: it depends only on a ``SessionManager`` and
    the append-only Eventloom logs, so it is cheap to instantiate anywhere that
    needs warm, cited retrieval without standing up a full :class:`MemoryFabric`.
    """

    def __init__(self, session_manager: SessionManager) -> None:
        self.session_manager = session_manager
        self._verbatim_index_cache: dict[str, tuple[tuple[int, int], VerbatimIndex, int]] = {}
        self._replay_cache: dict[str, tuple[tuple[int, int], ReplayResult, int]] = {}

    def invalidate(self, session_id: str | None = None) -> None:
        """Drop cached state for one session, or all sessions when ``None``."""
        if session_id is None:
            self._verbatim_index_cache = {}
            self._replay_cache = {}
            return
        self._verbatim_index_cache.pop(session_id, None)
        self._replay_cache.pop(session_id, None)

    def verbatim_index(self, session_id: str) -> VerbatimIndex:
        """Return a verbatim index for the current Eventloom file state.

        The index is cached per session and extended incrementally: when the
        append-only log has only grown, just the newly appended events are read
        and tokenized (:meth:`VerbatimIndex.append_chunks`) instead of
        rebuilding the BM25 index over the whole corpus on every change. The
        stored cursor is the exact byte offset that was indexed, so concurrent
        appends during a build never cause missed or duplicated events. A full
        rebuild only happens on a cold cache or if the log shrank / was
        rewritten (e.g. compaction).
        """
        eventlog = self.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._verbatim_index_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        if cached is not None and signature[1] > cached[2]:
            new_events, end_offset = eventlog.read_from_offset(cached[2])
            index = cached[1].append_chunks(_chunks_from_events(new_events))
            self._verbatim_index_cache[session_id] = (signature, index, end_offset)
            return index
        events, end_offset = eventlog.read_from_offset(0)
        index = VerbatimIndex.from_events(events)
        self._verbatim_index_cache[session_id] = (signature, index, end_offset)
        return index

    def verified_replay(self, session_id: str, from_seq: int = 1) -> ReplayResult:
        """Return the verified replay for a session, cached + incremental.

        Returns the full replay result including integrity verification, sliced
        to ``from_seq`` when requested. See module docstring for the integrity
        guard on the incremental tail path.
        """
        result = self._cached_full_replay(session_id)
        if from_seq <= 1:
            return result
        filtered = [event for event in result.events if event.seq >= from_seq]
        return ReplayResult.model_construct(events=filtered, integrity=result.integrity)

    def _cached_full_replay(self, session_id: str) -> ReplayResult:
        """Return the full verified replay for a session, cached + incremental.

        The cold/full path delegates to ``session_manager.replay`` (the
        authoritative read + full integrity verify). When the cached log has
        only grown, the appended tail is read and verified against the cached
        prefix instead. The tail verification doubles as a consistency guard:
        any offset skew (a concurrent append during the cold read, a rewrite)
        surfaces as a seq/hash mismatch and falls back to a full replay, so the
        fast path can never silently miss, duplicate, or trust a bad event.
        """
        eventlog = self.session_manager.get(session_id).eventlog
        signature = _eventlog_file_signature(eventlog)
        cached = self._replay_cache.get(session_id)
        if cached is not None and cached[0] == signature:
            return cached[1]
        if (
            cached is not None
            and isinstance(cached[2], int)
            and isinstance(signature[1], int)
            and signature[1] > cached[2]
            and cached[1].integrity is not None
            and cached[1].integrity.ok
        ):
            new_events, end_offset = eventlog.read_from_offset(cached[2])
            extended = self._extend_replay(cached[1], new_events)
            if extended is not None:
                self._replay_cache[session_id] = (signature, extended, end_offset)
                return extended
        result = cast(ReplayResult, self.session_manager.replay(session_id, from_seq=1))
        offset = _eventlog_file_signature(eventlog)[1]
        self._replay_cache[session_id] = (signature, result, offset)
        return result

    @staticmethod
    def _extend_replay(cached: ReplayResult, new_events: list[Any]) -> ReplayResult | None:
        """Extend a verified replay with appended events, or None to rebuild.

        Verifies only the new tail against the cached prefix's last event.
        Returns ``None`` (signalling a full re-verify) when the tail fails
        verification, so a tampered or reordered append never silently passes.
        """
        if not new_events:
            return cached
        last = cached.events[-1] if cached.events else None
        tail = verify_event_chain(
            new_events,
            first_seq=(last.seq + 1) if last is not None else 1,
            prev_hash=last.hash if last is not None else None,
        )
        if not tail.ok:
            return None
        combined = [*cached.events, *new_events]
        return ReplayResult(
            events=combined,
            integrity=IntegrityReport(ok=True, total_events=len(combined)),
        )
