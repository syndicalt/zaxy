"""Tests for the I2 amortized learned-context artifact (persist + fail-closed load).

The artifact is a rebuildable cache and the log is the evidence, so these assert
both directions of that asymmetry: a verified artifact loads, and an artifact
that is unreadable, unvouched by a build event, or covering a head that no longer
matches the log is ignored ENTIRELY rather than partially trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from zaxy.compaction import (
    CompactionAuditReport,
    CompactionProjection,
    CompactionProjectionRecord,
    build_compaction_projection,
    projection_from_payload,
    text_tokens,
)
from zaxy.event import EventLog
from zaxy.learned_context import (
    LEARNED_CONTEXT_EVENT_TYPE,
    LEARNED_CONTEXT_SCHEMA,
    MAX_ARTIFACT_RECORDS,
    MAX_RECORD_STRINGS,
    MAX_RECORD_TEXT_CHARS,
    MAX_STRING_CHARS,
    LearnedContextLoad,
    bound_projection,
    build_projection_built_payload,
    covered_head,
    learned_context_path,
    load_learned_context,
    write_learned_context,
)

_HASH = "a" * 64


def _ev(seq: int, event_type: str = "tool.call.completed", payload: dict | None = None, event_hash: str = _HASH) -> SimpleNamespace:
    return SimpleNamespace(seq=seq, type=event_type, payload=payload or {}, hash=event_hash, thread="t")


def _record(seq: int, text: str = "medoid record text", identities: tuple[str, ...] = ()) -> CompactionProjectionRecord:
    ref = f"eventloom://t/events/{seq}#{_HASH[:12]}"
    return CompactionProjectionRecord(
        kind="medoid",
        event_seq=seq,
        event_ref=ref,
        text=text,
        identities=identities or (ref,),
        citations=(ref,),
    )


def _audit(**overrides) -> CompactionAuditReport:
    base = {
        "safe": True,
        "event_count": 3,
        "integrity_ok": True,
        "integrity_reason": None,
        "identity_count": 1,
        "identity_recall": 1.0,
        "citation_coverage": 1.0,
        "mean_within_cluster_distance": 0.0,
        "identities": ("identity-code-0001",),
        "identity_hits": ("identity-code-0001",),
        "missing_identities": (),
        "unsafe_reasons": (),
    }
    return CompactionAuditReport(**{**base, **overrides})


def _projection(records: tuple[CompactionProjectionRecord, ...] | None = None, projection_id: str = "p" * 64) -> CompactionProjection:
    return CompactionProjection(
        projection_id=projection_id,
        strategy="medoid",
        source_event_count=3,
        source_identities=("identity-code-0001",),
        records=records if records is not None else (_record(1),),
        audit=_audit(),
    )


def _build_event(projection_id: str, seq: int = 9) -> SimpleNamespace:
    return _ev(seq, LEARNED_CONTEXT_EVENT_TYPE, {"projection_id": projection_id})


# --------------------------------------------------------------------------
# Path, head, and bounding helpers.
# --------------------------------------------------------------------------


def test_learned_context_path_is_a_per_session_projection_artifact(tmp_path: Path) -> None:
    """The artifact lives under the eventloom projections root, one file per session."""
    path = learned_context_path(tmp_path / ".eventloom", "sess-1")
    assert path == tmp_path / ".eventloom" / "projections" / "learned-context" / "sess-1.json"


def test_covered_head_returns_the_newest_well_formed_event() -> None:
    """The covered head is the newest event carrying a usable (seq, hash) pair."""
    assert covered_head([_ev(1), _ev(2), _ev(3)]) == (3, _HASH)


def test_covered_head_skips_malformed_tail_events_and_empty_logs() -> None:
    """A tail event without a usable seq/hash is skipped; an empty log has no head."""
    assert covered_head([]) is None
    assert covered_head([_ev(1), _ev(2, event_hash="")]) == (1, _HASH)
    assert covered_head([_ev(0)]) is None


def test_bound_projection_truncates_records_text_and_identity_lists() -> None:
    """Persisted projections are bounded in record count, text length, and string lists."""
    long_text = "x" * (MAX_RECORD_TEXT_CHARS + 500)
    many_identities = tuple(f"id-{i}-" + "y" * (MAX_STRING_CHARS + 10) for i in range(MAX_RECORD_STRINGS + 20))
    records = tuple(_record(i, text=long_text, identities=many_identities) for i in range(1, MAX_ARTIFACT_RECORDS + 25))

    bounded = bound_projection(_projection(records))

    assert len(bounded.records) == MAX_ARTIFACT_RECORDS
    assert all(len(record.text) == MAX_RECORD_TEXT_CHARS for record in bounded.records)
    assert all(len(record.identities) == MAX_RECORD_STRINGS for record in bounded.records)
    assert all(len(value) == MAX_STRING_CHARS for value in bounded.records[0].identities)
    # Records are kept in build order, so the medoid/highest-priority exemplars survive.
    assert bounded.records[0].event_seq == 1


def test_bound_projection_preserves_identity_and_audit_fields() -> None:
    """Bounding is lossy only in size: identity, strategy, and audit numbers survive."""
    bounded = bound_projection(_projection())
    assert bounded.projection_id == "p" * 64
    assert bounded.strategy == "medoid"
    assert bounded.source_event_count == 3
    assert bounded.audit.identity_recall == 1.0
    assert bounded.audit.citation_coverage == 1.0


# --------------------------------------------------------------------------
# Write / payload.
# --------------------------------------------------------------------------


def test_write_learned_context_writes_a_versioned_envelope_and_leaves_no_temp_file(tmp_path: Path) -> None:
    """The artifact is a schema-versioned envelope written without leaving temp files."""
    path = learned_context_path(tmp_path, "s")
    written = write_learned_context(_projection(), path, session_id="s", covered_seq=7, covered_hash=_HASH)

    envelope = json.loads(written.read_text(encoding="utf-8"))
    assert envelope["schema"] == LEARNED_CONTEXT_SCHEMA
    assert envelope["session_id"] == "s"
    assert envelope["covered_seq"] == 7
    assert envelope["covered_hash"] == _HASH
    assert envelope["projection"]["projection_id"] == "p" * 64
    # Atomic temp+rename must not leave the intermediate behind.
    assert [p.name for p in path.parent.iterdir()] == ["s.json"]


def test_write_learned_context_replaces_a_previous_artifact_in_place(tmp_path: Path) -> None:
    """Re-persisting overwrites the session's single artifact rather than accumulating."""
    path = learned_context_path(tmp_path, "s")
    write_learned_context(_projection(), path, session_id="s", covered_seq=7, covered_hash=_HASH)
    write_learned_context(_projection(projection_id="q" * 64), path, session_id="s", covered_seq=9, covered_hash=_HASH)

    assert len(list(path.parent.iterdir())) == 1
    assert json.loads(path.read_text(encoding="utf-8"))["covered_seq"] == 9


def test_build_projection_built_payload_is_non_authoritative_and_carries_the_covered_head() -> None:
    """The build event records identity, audit numbers, covered head, and artifact path."""
    payload = build_projection_built_payload(
        _projection(),
        session_id="s",
        covered_seq=7,
        covered_hash=_HASH,
        artifact_path="/tmp/s.json",
    )
    assert payload["authority_status"] == "non_authoritative"
    assert payload["non_authoritative"] is True
    assert payload["projection_id"] == "p" * 64
    assert payload["strategy"] == "medoid"
    assert payload["source_event_count"] == 3
    assert payload["record_count"] == 1
    assert payload["identity_recall"] == 1.0
    assert payload["citation_coverage"] == 1.0
    assert payload["covered_head"] == {"seq": 7, "hash": _HASH}
    assert payload["artifact_path"] == "/tmp/s.json"


# --------------------------------------------------------------------------
# Load — the fail-closed contract.
# --------------------------------------------------------------------------


def test_load_learned_context_returns_the_projection_when_vouched_and_current(tmp_path: Path) -> None:
    """A vouched artifact whose covered head still verifies loads and is not stale."""
    path = learned_context_path(tmp_path, "s")
    write_learned_context(_projection(), path, session_id="s", covered_seq=3, covered_hash=_HASH)
    events = [_ev(1), _ev(2), _ev(3), _build_event("p" * 64)]

    load = load_learned_context(path, events)

    assert load.projection is not None
    assert load.stale is False
    assert load.reason is None
    assert load.covered_seq == 3
    assert load.projection_id == "p" * 64
    assert load.to_diagnostics() == {
        "available": True,
        "stale": False,
        "reason": None,
        "covered_seq": 3,
        "projection_id": "p" * 64,
        "record_count": 1,
    }


def test_load_learned_context_treats_a_missing_artifact_as_absence_not_staleness(tmp_path: Path) -> None:
    """A never-built artifact yields no projection and is reported as missing, not stale."""
    load = load_learned_context(learned_context_path(tmp_path, "s"), [])
    assert load.projection is None
    assert load.stale is False
    assert load.reason == "missing"


def test_load_learned_context_ignores_an_unparseable_artifact(tmp_path: Path) -> None:
    """A corrupt (torn or truncated) artifact is ignored entirely and marked stale."""
    path = learned_context_path(tmp_path, "s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    load = load_learned_context(path, [])

    assert load.projection is None
    assert load.stale is True
    assert load.reason == "unreadable"


def test_load_learned_context_ignores_an_unknown_schema_version(tmp_path: Path) -> None:
    """An artifact from a future/unknown envelope version is ignored rather than guessed at."""
    path = learned_context_path(tmp_path, "s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": LEARNED_CONTEXT_SCHEMA + 99}), encoding="utf-8")

    load = load_learned_context(path, [])
    assert load.projection is None
    assert load.stale is True
    assert load.reason == "unreadable"


def test_load_learned_context_ignores_a_malformed_covered_head(tmp_path: Path) -> None:
    """An artifact without a usable covered head cannot be verified, so it is ignored."""
    path = learned_context_path(tmp_path, "s")
    path.parent.mkdir(parents=True, exist_ok=True)
    for envelope in (
        {"schema": LEARNED_CONTEXT_SCHEMA, "covered_seq": 0, "covered_hash": _HASH},
        {"schema": LEARNED_CONTEXT_SCHEMA, "covered_seq": 3, "covered_hash": ""},
        {"schema": LEARNED_CONTEXT_SCHEMA, "covered_seq": True, "covered_hash": _HASH},
    ):
        path.write_text(json.dumps(envelope), encoding="utf-8")
        load = load_learned_context(path, [])
        assert load.projection is None
        assert load.stale is True
        assert load.reason == "unreadable"


def test_load_learned_context_ignores_an_undecodable_projection_body(tmp_path: Path) -> None:
    """A well-formed envelope wrapping a broken projection body is ignored, not half-read."""
    path = learned_context_path(tmp_path, "s")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema": LEARNED_CONTEXT_SCHEMA, "covered_seq": 3, "covered_hash": _HASH, "projection": {}}),
        encoding="utf-8",
    )

    load = load_learned_context(path, [])
    assert load.projection is None
    assert load.stale is True
    assert load.reason == "unreadable"
    assert load.covered_seq == 3


def test_load_learned_context_rejects_an_artifact_no_build_event_vouches_for(tmp_path: Path) -> None:
    """An artifact whose projection_id the log never recorded is UNTRUSTED and ignored.

    The event is the evidence and the file is only convenience, so a file that
    appeared without a corresponding build event gets no trust at all.
    """
    path = learned_context_path(tmp_path, "s")
    write_learned_context(_projection(), path, session_id="s", covered_seq=3, covered_hash=_HASH)
    # The log has a build event, but for a DIFFERENT projection id.
    events = [_ev(1), _ev(2), _ev(3), _build_event("z" * 64)]

    load = load_learned_context(path, events)

    assert load.projection is None
    assert load.stale is True
    assert load.reason == "untrusted_no_build_event"
    assert load.to_diagnostics()["available"] is False


def test_load_learned_context_fails_closed_when_the_covered_head_hash_changed(tmp_path: Path) -> None:
    """A covered-head hash mismatch means the log moved under the artifact: ignore it."""
    path = learned_context_path(tmp_path, "s")
    write_learned_context(_projection(), path, session_id="s", covered_seq=3, covered_hash=_HASH)
    # Same seq, different hash -> the covered event is not the one that was compacted.
    events = [_ev(1), _ev(2), _ev(3, event_hash="b" * 64), _build_event("p" * 64)]

    load = load_learned_context(path, events)

    assert load.projection is None
    assert load.stale is True
    assert load.reason == "covered_head_mismatch"


def test_load_learned_context_fails_closed_when_the_covered_event_is_absent(tmp_path: Path) -> None:
    """A covered seq that no longer exists in the replay is a mismatch, not a pass."""
    path = learned_context_path(tmp_path, "s")
    write_learned_context(_projection(), path, session_id="s", covered_seq=99, covered_hash=_HASH)
    events = [_ev(1), _build_event("p" * 64)]

    load = load_learned_context(path, events)

    assert load.projection is None
    assert load.stale is True
    assert load.reason == "covered_head_mismatch"


def test_learned_context_load_defaults_report_nothing_available() -> None:
    """A bare load reports no projection, no staleness, and a zero record count."""
    assert LearnedContextLoad().to_diagnostics() == {
        "available": False,
        "stale": False,
        "reason": None,
        "covered_seq": None,
        "projection_id": None,
        "record_count": 0,
    }


# --------------------------------------------------------------------------
# Round-trip against a REAL projection built from a real log.
# --------------------------------------------------------------------------


def test_real_projection_round_trips_through_the_artifact(tmp_path: Path) -> None:
    """A projection built from a real log survives write+load with its citations intact."""
    log = EventLog(tmp_path / "real.jsonl")
    for i in range(3):
        log.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": f"docs/{i}.md",
                "start_line": 1,
                "end_line": 3,
                "content": f"Source {i} carries identity-code-000{i}.",
            },
        )
    projection = build_compaction_projection(log)

    path = learned_context_path(tmp_path, "s")
    write_learned_context(projection, path, session_id="s", covered_seq=3, covered_hash=_HASH)
    events = [_ev(1), _ev(2), _ev(3), _build_event(projection.projection_id)]

    load = load_learned_context(path, events)

    assert load.projection is not None
    assert load.projection.projection_id == projection.projection_id
    assert load.projection.records
    for record in load.projection.records:
        assert record.citations
        assert any(citation.startswith("eventloom://") for citation in record.citations)


def test_projection_from_payload_rehydrates_a_serialized_projection(tmp_path: Path) -> None:
    """The public rehydration seam reconstructs a projection from its serialized mapping."""
    from dataclasses import asdict

    projection = _projection()
    assert projection_from_payload(json.loads(json.dumps(asdict(projection)))) == projection


def test_text_tokens_shares_the_projection_routing_tokenizer() -> None:
    """Long-horizon scoring and projection search tokenize identically."""
    assert text_tokens("Refactor the QueryEngine") == {"refactor", "the", "queryengine"}
