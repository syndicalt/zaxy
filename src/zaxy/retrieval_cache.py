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

import contextlib
import json
import os
from typing import TYPE_CHECKING, Any, cast

from zaxy.event import EventLog, IntegrityReport, ReplayResult, verify_event_chain
from zaxy.verbatim import VerbatimIndex, _chunks_from_events

if TYPE_CHECKING:
    from zaxy.session import SessionManager

# On-disk verified-replay checkpoint format. A cold process re-reads the log
# (the events themselves are always needed) but, instead of re-hashing the whole
# chain, anchors on a previously-verified tip {covered_seq, covered_hash} and
# verifies only the appended tail. The checkpoint is a pure cache: any miss,
# corruption, version skew, or anchor mismatch falls back to a full verified
# replay, so the hash-chain integrity guarantee is never weakened.
_REPLAY_TIP_VERSION = 1


def _eventlog_file_signature(eventlog: EventLog) -> tuple[int, int]:
    """Return a cheap invalidation signature for a local Eventloom log."""
    try:
        stat = eventlog.path.stat()
    except FileNotFoundError:
        return (0, 0)
    return (stat.st_mtime_ns, stat.st_size)


def _replay_tip_path(eventlog: EventLog):  # type: ignore[no-untyped-def]
    """Return the verified-replay checkpoint path beside the projections."""
    return eventlog.path.parent / "projections" / f"{eventlog.path.stem}.replay-tip.json"


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
        with contextlib.suppress(Exception):
            _replay_tip_path(self.session_manager.get(session_id).eventlog).unlink(missing_ok=True)

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
        from_tip = self._try_cold_replay_from_tip(eventlog)
        if from_tip is not None:
            offset = _eventlog_file_signature(eventlog)[1]
            self._replay_cache[session_id] = (signature, from_tip, offset)
            return from_tip
        result = cast(ReplayResult, self.session_manager.replay(session_id, from_seq=1))
        offset = _eventlog_file_signature(eventlog)[1]
        self._replay_cache[session_id] = (signature, result, offset)
        if result.integrity is not None and result.integrity.ok:
            self._save_replay_tip(eventlog, result.events)
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

    def _try_cold_replay_from_tip(self, eventlog: EventLog) -> ReplayResult | None:
        """Cold replay anchored on a persisted verified tip, or None to full-verify.

        Reads the log (the events are needed regardless), then verifies only the
        tail appended after the checkpointed ``{covered_seq, covered_hash}``
        anchor instead of re-hashing the whole chain. Returns ``None`` — falling
        back to a full verified replay — on any miss: no/corrupt/old-version
        checkpoint, an anchor that does not match (log rewritten/compacted), a
        tail that fails verification, or a non-file eventlog (the unit-test
        path). The integrity guarantee is preserved: the prefix is trusted only
        because the checkpoint was written after a successful verify and the live
        anchor event still re-hashes to ``covered_hash``; the tail is fully
        verified forward from it.
        """
        try:
            tip = self._load_replay_tip(eventlog)
            if tip is None:
                return None
            covered_seq, covered_hash = tip
            events = eventlog.read_all()
            if covered_seq < 1 or covered_seq > len(events):
                return None
            anchor = events[covered_seq - 1]
            if anchor.seq != covered_seq or anchor.hash != covered_hash or not anchor.verify():
                return None
            tail = events[covered_seq:]
            report = verify_event_chain(tail, first_seq=covered_seq + 1, prev_hash=covered_hash)
            if not report.ok:
                return None
            self._save_replay_tip(eventlog, events)
            return ReplayResult(
                events=events,
                integrity=IntegrityReport(ok=True, total_events=len(events)),
            )
        except Exception:
            return None

    @staticmethod
    def _load_replay_tip(eventlog: EventLog) -> tuple[int, str] | None:
        """Load a validated ``(covered_seq, covered_hash)`` checkpoint, or None."""
        try:
            raw = _replay_tip_path(eventlog).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or data.get("version") != _REPLAY_TIP_VERSION:
            return None
        covered_seq = data.get("covered_seq")
        covered_hash = data.get("covered_hash")
        if not isinstance(covered_seq, int) or isinstance(covered_seq, bool) or covered_seq < 1:
            return None
        if not isinstance(covered_hash, str) or len(covered_hash) != 64:
            return None
        return (covered_seq, covered_hash)

    @staticmethod
    def _save_replay_tip(eventlog: EventLog, events: list[Any]) -> None:
        """Persist the verified tip atomically; best-effort, never raises."""
        try:
            if not events:
                return
            last = events[-1]
            seq = getattr(last, "seq", None)
            event_hash = getattr(last, "hash", None)
            if not isinstance(seq, int) or not isinstance(event_hash, str) or len(event_hash) != 64:
                return
            path = _replay_tip_path(eventlog)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"version": _REPLAY_TIP_VERSION, "covered_seq": seq, "covered_hash": event_hash}
            )
            tmp = path.with_suffix(f".json.tmp.{os.getpid()}")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        except Exception:
            return
