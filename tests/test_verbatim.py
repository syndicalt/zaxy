"""Tests for first-class verbatim Eventloom retrieval."""

from __future__ import annotations

import zaxy.verbatim as verbatim
from zaxy.event import EventLog
from zaxy.verbatim import VerbatimIndex


def test_verbatim_index_retrieves_document_chunks_with_citations(tmp_path) -> None:
    """Document content should be retrievable with exact Eventloom provenance."""
    log = EventLog(tmp_path / "agent.jsonl")
    event = log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/runbook.md",
            "start_line": 10,
            "end_line": 14,
            "content": "Rollback owner is platform operations. Use marker RB-42.",
            "sha256": "abc123",
        },
        thread="agent",
    )

    index = VerbatimIndex.from_event_logs([log])
    hits = index.query("Who owns rollback marker RB-42?", limit=1)

    assert len(hits) == 1
    assert hits[0].content == "Rollback owner is platform operations. Use marker RB-42."
    assert hits[0].citation == f"eventloom://agent/events/{event.seq}#{event.hash}"
    assert hits[0].source_kind == "document"
    assert hits[0].metadata["source_path"] == "docs/runbook.md"
    assert hits[0].metadata["source_start_line"] == 10
    assert hits[0].metadata["source_end_line"] == 14
    assert index._term_counts[0]["rollback"] == 1
    assert index._document_lengths == (8,)


def test_verbatim_index_retrieves_transcript_turn_identity(tmp_path) -> None:
    """Transcript turns should preserve source and turn identity."""
    log = EventLog(tmp_path / "chat.jsonl")
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={
            "source": "codex",
            "turn_index": 7,
            "role": "assistant",
            "content": "We chose the Postgres adapter for audit replay.",
            "redacted_paths": [],
        },
        thread="chat",
    )

    hits = VerbatimIndex.from_event_logs([log]).query("Which adapter did we choose?", limit=1)

    assert hits[0].content == "assistant: We chose the Postgres adapter for audit replay."
    assert hits[0].source_kind == "transcript"
    assert hits[0].metadata["transcript_source"] == "codex"
    assert hits[0].metadata["transcript_turn_index"] == 7
    assert hits[0].metadata["transcript_role"] == "assistant"


def test_verbatim_index_retrieves_packet_projection_as_memory(tmp_path) -> None:
    """Projected LLM packets should be retrievable as clean memory, not raw JSON."""
    log = EventLog(tmp_path / "agent.jsonl")
    event = log.append(
        "llm.packet.projected",
        actor="zaxy-packet-projector",
        payload={
            "session_id": "agent",
            "source_event_seq": 3,
            "source_event_hash": "b" * 64,
            "provider_path": "/v1/responses",
            "status_code": 200,
            "model": "gpt-test",
            "summary": "LLM packet /v1/responses gpt-test status 200. User: Remember Mira owns dashboards.",
        },
        thread="agent",
    )

    hits = VerbatimIndex.from_event_logs([log]).query("Who owns dashboards?", limit=1)

    assert hits[0].content == (
        "LLM packet /v1/responses gpt-test status 200. User: Remember Mira owns dashboards."
    )
    assert hits[0].citation == f"eventloom://agent/events/{event.seq}#{event.hash}"
    assert hits[0].source_kind == "packet_projection"
    assert hits[0].metadata["source_event_seq"] == 3
    assert hits[0].metadata["source_event_hash"] == "b" * 64
    assert hits[0].metadata["provider_path"] == "/v1/responses"
    assert hits[0].metadata["model"] == "gpt-test"


def test_verbatim_index_preserves_generic_event_authority_metadata(tmp_path) -> None:
    """Generic Eventloom source recall should expose authority metadata to checkout policy."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append(
        "decision.accepted",
        actor="worker",
        payload={
            "summary": "Worker-local accepted-looking diagnosis should stay diagnostic.",
            "authority_scope": "worker",
            "status": "accepted",
            "promoted": False,
            "superseded_by": "agent:4",
        },
        thread="agent",
    )

    hits = VerbatimIndex.from_event_logs([log]).query("worker local diagnosis", limit=1)

    assert hits[0].metadata["authority_scope"] == "worker"
    assert hits[0].metadata["status"] == "accepted"
    assert hits[0].metadata["promoted"] is False
    assert hits[0].metadata["superseded_by"] == "agent:4"


def test_verbatim_index_prefers_exact_identity_terms(tmp_path) -> None:
    """Rare identifiers should outrank broad lexical overlap."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/a.md",
            "start_line": 1,
            "end_line": 1,
            "content": "Release planning mentions rollback deployment context.",
        },
        thread="agent",
    )
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/b.md",
            "start_line": 1,
            "end_line": 1,
            "content": "Release planning records source identity-code-0042.",
        },
        thread="agent",
    )

    hits = VerbatimIndex.from_event_logs([log]).query("release planning identity-code-0042", limit=1)

    assert hits[0].metadata["source_path"] == "docs/b.md"


def test_verbatim_index_matches_identifiers_next_to_punctuation(tmp_path) -> None:
    """Identifier lookup should not depend on sentence punctuation."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append(
        "transcript.turn",
        actor="assistant",
        payload={
            "role": "assistant",
            "content": "The audit trail uses identity-code-0042.",
        },
        thread="agent",
    )

    hits = VerbatimIndex.from_event_logs([log]).query("identity-code-0042", limit=1)

    assert hits[0].content == "assistant: The audit trail uses identity-code-0042."


def test_verbatim_index_scores_only_documents_matching_query_terms(tmp_path, monkeypatch) -> None:
    """BM25 queries should skip chunks that cannot match any query term."""
    log = EventLog(tmp_path / "agent.jsonl")
    for index in range(20):
        log.append(
            "document.indexed",
            actor="indexer",
            payload={
                "path": f"docs/{index}.md",
                "start_line": 1,
                "end_line": 1,
                "content": f"Background document {index} about unrelated planning.",
            },
            thread="agent",
        )
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/target.md",
            "start_line": 1,
            "end_line": 1,
            "content": "The migration checkpoint marker is precise-needle-42.",
        },
        thread="agent",
    )
    index = VerbatimIndex.from_event_logs([log])
    original_score = verbatim._bm25_score_from_precomputed
    score_calls = 0

    def tracking_score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(verbatim, "_bm25_score_from_precomputed", tracking_score)

    hits = index.query("precise-needle-42", limit=1)

    assert hits[0].metadata["source_path"] == "docs/target.md"
    assert score_calls == 1


def test_verbatim_index_query_uses_precomputed_bm25_statistics(tmp_path, monkeypatch) -> None:
    """Runtime queries should not recompute generic BM25 document statistics."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/target.md",
            "start_line": 1,
            "end_line": 1,
            "content": "The benchmark marker is reusable-needle-7.",
        },
        thread="agent",
    )
    index = VerbatimIndex.from_event_logs([log])

    def fail_generic_score(*args, **kwargs):
        raise AssertionError("query should use precomputed BM25 statistics")

    monkeypatch.setattr(verbatim, "_bm25_score_from_counts", fail_generic_score)

    hits = index.query("reusable-needle-7", limit=1)

    assert hits[0].metadata["source_path"] == "docs/target.md"


def test_verbatim_index_scores_only_terms_present_in_candidate(tmp_path, monkeypatch) -> None:
    """Candidate scoring should not scan query terms absent from that candidate."""
    log = EventLog(tmp_path / "agent.jsonl")
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/alpha.md",
            "start_line": 1,
            "end_line": 1,
            "content": "alpha topic only",
        },
        thread="agent",
    )
    log.append(
        "document.indexed",
        actor="indexer",
        payload={
            "path": "docs/beta.md",
            "start_line": 1,
            "end_line": 1,
            "content": "beta topic only",
        },
        thread="agent",
    )
    index = VerbatimIndex.from_event_logs([log])
    original_score = verbatim._bm25_score_from_precomputed
    scored_terms: list[tuple[str, ...]] = []

    def tracking_score(query_terms, *args, **kwargs):
        scored_terms.append(tuple(query_terms))
        return original_score(query_terms, *args, **kwargs)

    monkeypatch.setattr(verbatim, "_bm25_score_from_precomputed", tracking_score)

    hits = index.query("alpha beta missing", limit=2)

    assert {hit.metadata["source_path"] for hit in hits} == {"docs/alpha.md", "docs/beta.md"}
    assert sorted(scored_terms) == [("alpha",), ("beta",)]


def _append_turns(log: EventLog, contents: list[str]) -> None:
    for index, content in enumerate(contents):
        log.append(
            "transcript.turn",
            actor="assistant",
            payload={"source": "codex", "turn_index": index, "role": "assistant", "content": content},
            thread="agent",
        )


def test_append_chunks_matches_full_rebuild(tmp_path) -> None:
    """Incrementally extending the index must equal a full rebuild exactly."""
    log = EventLog(tmp_path / "agent.jsonl")
    _append_turns(
        log,
        [
            "Rollback owner is platform operations marker RB-42.",
            "We chose the Postgres adapter for audit replay.",
            "The dashboard owner is the analytics guild.",
            "Release planning uses identity-code-0042 for the cut.",
            "Worker local diagnosis runs before the remote sweep.",
            "Postgres adapter keeps verbatim recall fast for checkout.",
        ],
    )
    chunks = verbatim._chunks_from_events(log.read_all())

    full = VerbatimIndex(chunks)
    incremental = VerbatimIndex(chunks[:2]).append_chunks(chunks[2:4]).append_chunks(chunks[4:])

    # Every corpus statistic is identical to a from-scratch build.
    assert incremental._chunks == full._chunks
    assert incremental._document_lengths == full._document_lengths
    assert incremental._document_count == full._document_count
    assert incremental._term_document_ids == full._term_document_ids
    assert incremental._document_frequencies == full._document_frequencies
    assert incremental._term_idf == full._term_idf
    assert incremental._document_length_norms == full._document_length_norms

    for query in (
        "Postgres adapter",
        "rollback marker RB-42",
        "dashboard owner",
        "identity-code-0042",
        "verbatim recall checkout",
        "worker local diagnosis",
    ):
        expected = [(hit.score, hit.citation) for hit in full.query(query, limit=10)]
        actual = [(hit.score, hit.citation) for hit in incremental.query(query, limit=10)]
        assert actual == expected, query


def test_append_chunks_empty_returns_same_index(tmp_path) -> None:
    """Extending with no new chunks is a no-op that returns the same index."""
    log = EventLog(tmp_path / "a.jsonl")
    _append_turns(log, ["hello world from verbatim recall"])
    index = VerbatimIndex.from_event_logs([log])
    assert index.append_chunks(()) is index


def test_read_from_offset_reads_only_new_events(tmp_path) -> None:
    """read_from_offset returns just the events appended after the cursor."""
    log = EventLog(tmp_path / "a.jsonl")
    _append_turns(log, ["first event content"])

    all_events, offset = log.read_from_offset(0)
    assert [event.seq for event in all_events] == [1]

    empty, offset_unchanged = log.read_from_offset(offset)
    assert empty == []
    assert offset_unchanged == offset

    _append_turns(log, ["second event content"])
    new_events, grown_offset = log.read_from_offset(offset)
    assert [event.seq for event in new_events] == [2]
    assert grown_offset > offset

    full, _ = log.read_from_offset(0)
    assert [event.seq for event in full] == [1, 2]
