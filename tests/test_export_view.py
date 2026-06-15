"""Tests for the product-agnostic memory export contract (Phase 1)."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest

from zaxy.export_view import (
    EXPORT_ENTRY_SCHEMA_VERSION,
    UNSIGNED_BUNDLE_VERSION,
    ExportSelector,
    build_memory_export,
    build_memory_export_view,
    disclose_export_bundle,
    entry_matches,
    export_cursor,
    load_signing_key,
    verify_memory_export_subset,
)
from zaxy.retrieval_cache import SessionRetrievalCache
from zaxy.session import SessionManager


def _cache(tmp_path: Path) -> SessionRetrievalCache:
    return SessionRetrievalCache(SessionManager(base_path=str(tmp_path / ".eventloom")))


def _append(
    cache: SessionRetrievalCache,
    session_id: str,
    event_type: str,
    payload: dict,
    *,
    actor: str = "assistant",
    timestamp: datetime | None = None,
) -> None:
    cache.session_manager.get(session_id).eventlog.append(
        event_type, actor=actor, payload=payload, thread=session_id, timestamp=timestamp
    )


def test_event_grain_projects_cited_entries(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "Ship export", "description": "do it"})
    _append(cache, "s", "task.completed", {"title": "Ship export"})

    entries = build_memory_export_view("s", ExportSelector(grains=frozenset({"event"})), retrieval_cache=cache)

    assert [e["kind"] for e in entries] == ["goal.created", "task.completed"]
    first = entries[0]
    assert first["schema_version"] == EXPORT_ENTRY_SCHEMA_VERSION
    assert first["grain"] == "event"
    assert first["source"] == "eventloom"
    assert first["seq"] == 1
    assert first["valid_from"]  # event timestamp
    assert first["citation"].startswith("eventloom://s/events/1#")
    assert len(first["citation"].split("#")[1]) == 64  # full sealed hash
    assert first["content"] == {
        "type": "goal.created",
        "actor": "assistant",
        "thread": "s",
        "payload": {"title": "Ship export", "description": "do it"},
    }


def test_semantic_grain_projects_extraction(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "Ship export", "description": "the goal"})

    entries = build_memory_export_view(
        "s", ExportSelector(grains=frozenset({"semantic"})), retrieval_cache=cache
    )

    assert entries  # extraction yielded at least one entity
    entity = entries[0]
    assert entity["grain"] == "semantic"
    assert entity["source"] == "extraction"
    assert entity["kind"].startswith("entity:")
    assert entity["citation"].startswith("eventloom://s/events/1#")
    assert entity["content"]["name"] == "Ship export"


def test_both_grains_default_with_deterministic_ordering(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    _append(cache, "s", "decision.made", {"decision": "use export contract"})

    entries = build_memory_export_view("s", retrieval_cache=cache)  # default = both grains

    # Ascending seq; within each event the event entry precedes its semantic ones.
    assert [e["seq"] for e in entries] == sorted(e["seq"] for e in entries)
    by_seq_1 = [e for e in entries if e["seq"] == 1]
    assert by_seq_1[0]["grain"] == "event"
    assert all(e["grain"] == "semantic" for e in by_seq_1[1:])


def test_kinds_filter_gates_both_grains(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "keep"})
    _append(cache, "s", "task.completed", {"title": "drop"})

    entries = build_memory_export_view(
        "s", ExportSelector(kinds=frozenset({"goal.created"})), retrieval_cache=cache
    )

    assert entries
    assert {e["seq"] for e in entries} == {1}
    assert all(e["citation"].startswith("eventloom://s/events/1#") for e in entries)


def test_seq_range_and_since_cursor(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    for i in range(1, 5):
        _append(cache, "s", "goal.created", {"title": f"g{i}"})

    # Inclusive max_seq, exclusive since_seq.
    windowed = build_memory_export_view(
        "s",
        ExportSelector(grains=frozenset({"event"}), since_seq=1, max_seq=3),
        retrieval_cache=cache,
    )
    assert [e["seq"] for e in windowed] == [2, 3]


def test_time_window_filter(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "old"}, timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    _append(cache, "s", "goal.created", {"title": "new"}, timestamp=datetime(2026, 6, 1, tzinfo=UTC))

    entries = build_memory_export_view(
        "s",
        ExportSelector(grains=frozenset({"event"}), since_time="2026-03-01T00:00:00Z"),
        retrieval_cache=cache,
    )
    assert [e["content"]["payload"]["title"] for e in entries] == ["new"]


def test_query_prefilter_uses_verbatim_index(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "transcript.turn", {"content": "the quick brown fox", "role": "user", "turn_index": 1})
    _append(cache, "s", "transcript.turn", {"content": "lazy dog sleeps", "role": "user", "turn_index": 2})

    entries = build_memory_export_view(
        "s",
        ExportSelector(grains=frozenset({"event"}), query="fox"),
        retrieval_cache=cache,
    )
    assert [e["seq"] for e in entries] == [1]


def test_since_cursor_delta_correctness(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    _append(cache, "s", "goal.created", {"title": "g2"})

    first = build_memory_export_view("s", retrieval_cache=cache)
    cursor = export_cursor(first)
    assert cursor == 2

    _append(cache, "s", "goal.created", {"title": "g3"})
    delta = build_memory_export_view("s", ExportSelector(since_seq=cursor), retrieval_cache=cache)

    assert delta
    assert {e["seq"] for e in delta} == {3}


def test_redaction_excludes_sensitive_events(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "decision.made", {"decision": "public call"})
    # A secret-looking key forces the sealed event's sensitivity to "restricted".
    _append(cache, "s", "decision.made", {"decision": "leaky", "password": "hunter2"})

    without_policy = build_memory_export_view(
        "s", ExportSelector(grains=frozenset({"event"})), retrieval_cache=cache
    )
    assert {e["seq"] for e in without_policy} == {1, 2}

    redacted = build_memory_export_view(
        "s",
        ExportSelector(grains=frozenset({"event"}), exclude_sensitivities=frozenset({"restricted"})),
        retrieval_cache=cache,
    )
    assert {e["seq"] for e in redacted} == {1}


def test_schema_version_is_pinned_on_every_entry(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    entries = build_memory_export_view("s", retrieval_cache=cache)
    assert entries
    assert all(e["schema_version"] == EXPORT_ENTRY_SCHEMA_VERSION for e in entries)


def test_projection_is_canonical_byte_stable(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1", "description": "stable"})
    _append(cache, "s", "decision.made", {"decision": "be deterministic"})

    a = build_memory_export_view("s", retrieval_cache=cache)
    b = build_memory_export_view("s", retrieval_cache=cache)
    assert a == b

    def dump(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    assert dump(a) == dump(b)


def test_entries_feed_build_export_unchanged(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    _append(cache, "s", "decision.made", {"decision": "ship it"})
    entries = build_memory_export_view("s", retrieval_cache=cache)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from zaxy.portable import build_export, generate_keypair, verify_export

    keypair = generate_keypair()
    bundle = build_export(
        entries,
        keypair=keypair,
        session_id="s",
        created_at=datetime.now(UTC).isoformat(),
        nonce="0" * 32,
    )
    assert verify_export(bundle)["ok"] is True
    assert len(bundle["entries"]) == len(entries)


def test_selector_rejects_empty_or_unknown_grains() -> None:
    with pytest.raises(ValueError, match="grains"):
        ExportSelector(grains=frozenset())
    with pytest.raises(ValueError, match="grains"):
        ExportSelector(grains=frozenset({"event", "bogus"}))


def test_build_memory_export_unsigned_envelope(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    _append(cache, "s", "decision.made", {"decision": "ship"})

    bundle = build_memory_export("s", retrieval_cache=cache)
    view = build_memory_export_view("s", retrieval_cache=cache)

    assert bundle["version"] == UNSIGNED_BUNDLE_VERSION
    assert bundle["signed"] is False
    assert bundle["schema_version"] == EXPORT_ENTRY_SCHEMA_VERSION
    assert bundle["session_id"] == "s"
    assert bundle["entries"] == view  # unsigned carries the raw projector entries


def test_build_memory_export_signed_roundtrips(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    view = build_memory_export_view("s", retrieval_cache=cache)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from zaxy.portable import generate_keypair, verify_export

    keypair = generate_keypair()
    bundle = build_memory_export(
        "s",
        retrieval_cache=cache,
        signing_key=keypair,
        created_at="2026-06-15T00:00:00+00:00",
        nonce="0" * 32,
    )
    assert "signature" in bundle and "merkle_root" in bundle
    assert verify_export(bundle)["ok"] is True
    assert len(bundle["entries"]) == len(view)


def test_load_signing_key_reads_files(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from zaxy.portable import generate_keypair, verify_export

    keypair = generate_keypair()
    priv = tmp_path / "k.pem"
    pub = tmp_path / "k.pub"
    priv.write_bytes(keypair["private_pem"])
    pub.write_text(keypair["public_key"].hex(), encoding="utf-8")

    loaded = load_signing_key(
        private_key_path=priv, public_key_path=pub, algorithm=keypair["algorithm"]
    )
    assert loaded["algorithm"] == keypair["algorithm"]
    assert loaded["private_pem"] == keypair["private_pem"]
    assert loaded["public_key"] == keypair["public_key"]

    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    bundle = build_memory_export(
        "s", retrieval_cache=cache, signing_key=loaded, created_at="2026-06-15T00:00:00+00:00", nonce="0" * 32
    )
    assert verify_export(bundle)["ok"] is True


def test_entry_matches_supported_axes(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"}, timestamp=datetime(2026, 1, 1, tzinfo=UTC))
    _append(cache, "s", "decision.made", {"decision": "d2"}, timestamp=datetime(2026, 6, 1, tzinfo=UTC))
    entries = build_memory_export_view("s", retrieval_cache=cache)  # both grains

    only_event = [e for e in entries if entry_matches(e, ExportSelector(grains=frozenset({"event"})))]
    assert only_event and all(e["grain"] == "event" for e in only_event)

    goals = [e for e in entries if entry_matches(e, ExportSelector(kinds=frozenset({"goal.created"})))]
    assert goals and all(e["kind"] == "goal.created" for e in goals)

    after = [e for e in entries if entry_matches(e, ExportSelector(since_seq=1))]
    assert after and all(e["seq"] > 1 for e in after)
    upto = [e for e in entries if entry_matches(e, ExportSelector(max_seq=1))]
    assert upto and all(e["seq"] <= 1 for e in upto)

    recent = [
        e for e in entries if entry_matches(e, ExportSelector(since_time="2026-03-01T00:00:00Z"))
    ]
    assert recent and all(e["valid_from"] >= "2026-03-01T00:00:00Z" for e in recent)


def test_entry_matches_ignores_projection_only_axes(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    entries = build_memory_export_view("s", retrieval_cache=cache)
    # query and exclude_sensitivities are projection-time only -> ignored by entry_matches.
    selector = ExportSelector(query="nomatch", exclude_sensitivities=frozenset({"restricted"}))
    assert entries and all(entry_matches(e, selector) for e in entries)


def _signed_bundle(cache: SessionRetrievalCache, session_id: str = "s") -> tuple[dict, dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from zaxy.portable import generate_keypair

    keypair = generate_keypair()
    bundle = build_memory_export(
        session_id,
        ExportSelector(grains=frozenset({"event"})),
        retrieval_cache=cache,
        signing_key=keypair,
        created_at="2026-06-15T00:00:00+00:00",
        nonce="0" * 32,
    )
    return bundle, keypair


def test_disclose_export_bundle_reveals_only_matching(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "keep"})
    _append(cache, "s", "decision.made", {"decision": "hide"})
    bundle, keypair = _signed_bundle(cache)

    subset = disclose_export_bundle(
        bundle, ExportSelector(grains=frozenset({"event"}), kinds=frozenset({"goal.created"}))
    )
    assert verify_memory_export_subset(subset, expect_public_key=keypair["public_key"].hex())["ok"]
    assert {d["content"]["kind"] for d in subset["disclosed"]} == {"goal.created"}
    assert "entries" not in subset  # undisclosed entries never shipped
    payloads = [d["content"]["content"].get("payload", {}) for d in subset["disclosed"]]
    assert {"decision": "hide"} not in payloads


def test_disclose_subset_tamper_is_detected(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "keep"})
    bundle, _ = _signed_bundle(cache)
    subset = disclose_export_bundle(bundle, ExportSelector(grains=frozenset({"event"})))
    subset["disclosed"][0]["content"]["content"]["payload"]["title"] = "TAMPERED"
    assert verify_memory_export_subset(subset)["ok"] is False


def test_disclose_requires_signed_bundle(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    _append(cache, "s", "goal.created", {"title": "g1"})
    unsigned = build_memory_export("s", retrieval_cache=cache)
    with pytest.raises(ValueError, match="signed bundle"):
        disclose_export_bundle(unsigned, ExportSelector())
