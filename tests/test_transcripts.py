"""Tests for sanitized transcript ingestion."""

from __future__ import annotations

from zaxy.transcripts import collect_transcript_events


class TestCollectTranscriptEvents:
    """Tests for turning session transcripts into durable events."""

    def test_collects_turns_and_redacts_secret_text(self) -> None:
        turns = [
            {"role": "user", "content": "Use key sk-abcdefghijklmnop for tests"},
            {"role": "assistant", "content": "I will store only safe context."},
        ]

        events = collect_transcript_events(turns, source="codex")

        assert [event["event_type"] for event in events] == [
            "transcript.turn",
            "transcript.turn",
        ]
        assert events[0]["actor"] == "user"
        assert events[0]["payload"] == {
            "source": "codex",
            "turn_index": 1,
            "role": "user",
            "content": "[REDACTED]",
            "redacted_paths": ["content"],
        }
        assert events[1]["payload"]["content"] == "I will store only safe context."

    def test_skips_empty_turns(self) -> None:
        events = collect_transcript_events([
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "Done"},
        ])

        assert len(events) == 1
        assert events[0]["payload"]["turn_index"] == 2
