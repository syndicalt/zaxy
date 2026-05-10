"""Tests for cold-path LLM packet projection."""

from __future__ import annotations

from pathlib import Path

from zaxy.event import EventLog
from zaxy.packet_projection import project_packet_events


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
    assert [event.type for event in log.read_all()] == [
        "llm.packet.completed",
        "llm.packet.projected",
    ]

