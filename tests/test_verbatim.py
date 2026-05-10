"""Tests for first-class verbatim Eventloom retrieval."""

from __future__ import annotations

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
