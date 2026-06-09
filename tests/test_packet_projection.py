"""Tests for cold-path LLM packet projection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from zaxy import packet_projection
from zaxy.event import EventLog
from zaxy.packet_projection import (
    build_packet_projection_payload,
    project_packet_events,
    project_packet_events_to_graph,
    watch_packet_events,
)


def test_project_packet_events_appends_memory_ready_summary(tmp_path: Path) -> None:
    """Completed packet events should project into compact searchable memory."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    packet = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/chat/completions",
            "status_code": 200,
            "model": "gpt-test",
            "usage_counts": {"prompt": 12, "completion": 5, "total": 17},
            "request": {
                "body": {
                    "messages": [
                        {"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "Remember that the launch code is quartz."},
                    ]
                }
            },
            "response": {
                "body": {
                    "choices": [
                        {"message": {"role": "assistant", "content": "I will remember quartz."}}
                    ]
                }
            },
        },
    )

    result = project_packet_events(eventloom_path=eventloom_path, session_id="agent-1")

    events = log.read_all()
    assert result.read == 1
    assert result.projected == 1
    assert result.skipped == 0
    assert [event.type for event in events] == ["llm.packet.completed", "llm.packet.projected"]
    projection = events[1]
    assert projection.actor == "zaxy-packet-projector"
    assert projection.payload["source_event_seq"] == packet.seq
    assert projection.payload["source_event_hash"] == packet.hash
    assert projection.payload["summary"] == (
        "LLM packet /v1/chat/completions gpt-test status 200. "
        "User: Remember that the launch code is quartz. "
        "Assistant: I will remember quartz."
    )
    assert projection.payload["request_summary"] == {
        "message_count": 2,
        "last_user_message": "Remember that the launch code is quartz.",
    }
    assert projection.payload["response_summary"] == {
        "assistant_message": "I will remember quartz.",
    }
    assert result.projected_events == (projection,)


def test_project_packet_events_is_idempotent_by_source_hash(tmp_path: Path) -> None:
    """Projection should not duplicate packets it has already summarized."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    packet = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Persist this preference."}},
            "response": {"body": {"output_text": "Preference persisted."}},
        },
    )
    log.append(
        "llm.packet.projected",
        actor="zaxy-packet-projector",
        thread="agent-1",
        payload={"source_event_hash": packet.hash, "source_event_seq": packet.seq},
    )

    result = project_packet_events(eventloom_path=eventloom_path, session_id="agent-1")

    assert result.read == 1
    assert result.projected == 0
    assert result.skipped == 1
    assert result.projected_events == ()
    assert [event.type for event in log.read_all()] == [
        "llm.packet.completed",
        "llm.packet.projected",
    ]


def test_project_packet_events_respects_from_seq_and_limit(tmp_path: Path) -> None:
    """Projection windows should support incremental packet projector runs."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    first = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={"provider_path": "/v1/ignored", "status_code": 200},
    )
    second = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember alpha."}},
            "response": {"body": {"output_text": "Alpha recorded."}},
        },
    )
    third = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember beta."}},
            "response": {"body": {"output_text": "Beta recorded."}},
        },
    )

    result = project_packet_events(
        eventloom_path=eventloom_path,
        session_id="agent-1",
        from_seq=second.seq,
        limit=1,
    )
    empty = project_packet_events(
        eventloom_path=eventloom_path,
        session_id="agent-1",
        from_seq=first.seq,
        limit=0,
    )

    assert result.read == 1
    assert result.projected == 1
    assert result.projected_events[0].payload["source_event_seq"] == second.seq
    assert third.hash not in {event.payload.get("source_event_hash") for event in log.read_all()}
    assert empty.read == 0
    assert empty.projected == 0


def test_build_packet_projection_payload_handles_non_chat_shapes(tmp_path: Path) -> None:
    """Projection summaries should cover Responses API text, delta chunks, raw bytes, and defaults."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    event = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "",
            "provider_path": "",
            "status_code": "not-int",
            "request": {
                "body": {
                    "input": [
                        {"type": "input_text", "text": "Summarize this deployment note."},
                        "Keep it cited.",
                    ]
                }
            },
            "response": {
                "body": {
                    "choices": [
                        {"delta": {"content": "Streaming summary."}},
                    ],
                    "bytes": 42,
                }
            },
        },
    )

    payload = build_packet_projection_payload(event)

    assert payload["session_id"] == "agent-1"
    assert payload["provider_path"] == "unknown-provider-path"
    assert payload["status_code"] == 0
    assert payload["request_summary"] == {
        "input": "Summarize this deployment note. Keep it cited."
    }
    assert payload["response_summary"] == {"assistant_message": "Streaming summary."}
    assert payload["summary"] == (
        "LLM packet unknown-provider-path status 0. "
        "Input: Summarize this deployment note. Keep it cited. "
        "Assistant: Streaming summary."
    )


def test_build_packet_projection_payload_handles_raw_bytes_and_truncated_text(
    tmp_path: Path,
) -> None:
    """Packet summaries should preserve non-chat diagnostics without oversized prompt blobs."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    long_user_message = " ".join(["retain"] * 100)
    event = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "provider_path": "/v1/chat/completions",
            "status_code": True,
            "request": {
                "body": {
                    "messages": [
                        {"role": "assistant", "content": "Prior answer."},
                        {"role": "user", "content": long_user_message},
                    ]
                }
            },
            "response": {"body": {"choices": [None], "bytes": "4096"}},
        },
    )

    payload = build_packet_projection_payload(event)

    assert payload["status_code"] == 0
    assert payload["request_summary"]["message_count"] == 2
    assert payload["request_summary"]["last_user_message"].endswith("...")
    assert len(payload["request_summary"]["last_user_message"]) <= packet_projection.MAX_PACKET_EXCERPT_CHARS
    assert payload["response_summary"] == {"raw_response_bytes": 4096}
    assert "Response body: 4096 bytes." in payload["summary"]


def test_build_packet_projection_payload_ignores_malformed_nested_bodies(
    tmp_path: Path,
) -> None:
    """Malformed packet envelopes should degrade to a minimal searchable summary."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    event = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "provider_path": "/v1/responses",
            "status_code": 502,
            "request": {"body": ["not", "a", "dict"]},
            "response": {"body": None},
        },
    )

    payload = build_packet_projection_payload(event)

    assert payload["request_summary"] == {}
    assert payload["response_summary"] == {}
    assert payload["summary"] == "LLM packet /v1/responses status 502."


def test_build_packet_projection_payload_handles_packets_without_user_or_assistant_text(
    tmp_path: Path,
) -> None:
    """System-only prompts and empty choices should still produce a valid packet projection."""
    log = EventLog(tmp_path / ".eventloom" / "agent-1.jsonl")
    event = log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "provider_path": "/v1/chat/completions",
            "status_code": 204,
            "request": {
                "body": {
                    "messages": [
                        {"role": "system", "content": "Trace this request."},
                        {"role": "assistant", "content": "No user turn yet."},
                    ]
                }
            },
            "response": {
                "body": {
                    "choices": [
                        {"message": {"role": "assistant", "content": ""}},
                        {"delta": {"content": ""}},
                    ]
                }
            },
        },
    )

    payload = build_packet_projection_payload(event)

    assert payload["request_summary"] == {
        "message_count": 2,
        "last_user_message": None,
    }
    assert payload["response_summary"] == {}
    assert payload["summary"] == "LLM packet /v1/chat/completions status 204."


async def test_project_packet_events_to_graph_upserts_projected_events(tmp_path: Path) -> None:
    """New packet projections should be ingestible into Neo4j without replay."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the reviewer is Sam."}},
            "response": {"body": {"output_text": "Reviewer Sam recorded."}},
        },
    )
    projection = project_packet_events(eventloom_path=eventloom_path, session_id="agent-1")
    graph = AsyncMock()

    result = await project_packet_events_to_graph(
        projection.projected_events,
        graph=graph,
        session_id="agent-1",
    )

    assert result.projected == 1
    assert result.failed == 0
    graph.upsert_extraction.assert_awaited_once()
    extraction = graph.upsert_extraction.await_args.args[0]
    assert {entity.entity_type for entity in extraction.entities} >= {
        "session",
        "llm_packet_projection",
    }
    assert graph.upsert_extraction.await_args.kwargs == {"session_id": "agent-1"}


async def test_project_packet_events_to_graph_degrades_when_upsert_fails(
    tmp_path: Path,
) -> None:
    """Graph failures should not undo Eventloom packet projection."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the lane is alpha."}},
            "response": {"body": {"output_text": "Lane alpha recorded."}},
        },
    )
    projection = project_packet_events(eventloom_path=eventloom_path, session_id="agent-1")
    graph = AsyncMock()
    graph.upsert_extraction.side_effect = RuntimeError("neo4j unavailable")

    result = await project_packet_events_to_graph(
        projection.projected_events,
        graph=graph,
        session_id="agent-1",
    )

    assert result.projected == 0
    assert result.failed == 1
    assert [event.type for event in log.read_all()] == [
        "llm.packet.completed",
        "llm.packet.projected",
    ]


def test_watch_packet_events_runs_bounded_projection_passes(tmp_path: Path) -> None:
    """Watch mode should be testable with bounded iterations."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the dashboard owner is Mira."}},
            "response": {"body": {"output_text": "Dashboard owner Mira recorded."}},
        },
    )

    result = watch_packet_events(
        eventloom_path=eventloom_path,
        session_id="agent-1",
        interval_seconds=0,
        max_iterations=2,
    )

    assert result.iterations == 2
    assert result.read == 2
    assert result.projected == 1
    assert result.skipped == 1
    assert [event.type for event in log.read_all()] == [
        "llm.packet.completed",
        "llm.packet.projected",
    ]


def test_watch_packet_events_invokes_callback_and_sleeps_between_iterations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Watch mode should expose projected batches and honor nonzero polling intervals."""
    eventloom_path = tmp_path / ".eventloom"
    log = EventLog(eventloom_path / "agent-1.jsonl")
    log.append(
        "llm.packet.completed",
        actor="zaxy-packet-analyzer",
        thread="agent-1",
        payload={
            "session_id": "agent-1",
            "provider_path": "/v1/responses",
            "status_code": 200,
            "request": {"body": {"input": "Remember the release gate owner is Noor."}},
            "response": {"body": {"output_text": "Release gate owner Noor recorded."}},
        },
    )
    callbacks = []
    sleeps = []
    monkeypatch.setattr(packet_projection.time, "sleep", sleeps.append)

    result = watch_packet_events(
        eventloom_path=eventloom_path,
        session_id="agent-1",
        interval_seconds=0.25,
        max_iterations=2,
        on_projected=callbacks.append,
    )

    assert result.iterations == 2
    assert result.projected == 1
    assert len(callbacks) == 1
    assert callbacks[0].projected == 1
    assert sleeps == [0.25]
