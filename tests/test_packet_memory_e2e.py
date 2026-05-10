"""End-to-end smoke tests for the packet memory workflow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from zaxy.__main__ import app
from zaxy.core import MemoryFabric
from zaxy.doctor import packet_memory_report
from zaxy.event import EventLog
from zaxy.packet_analyzer import LlmPacketAnalyzer, PacketAnalyzerConfig
from zaxy.packet_projection import project_packet_events


@pytest.mark.asyncio
async def test_packet_memory_capture_project_status_and_context_assembly(tmp_path: Path) -> None:
    """Packet capture should become status-visible and prompt-eligible memory."""
    eventloom_path = tmp_path / ".eventloom"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://upstream.example/v1/responses"
        return httpx.Response(
            200,
            json={
                "model": "gpt-test",
                "output_text": "Dashboard owner Mira recorded.",
                "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
            },
        )

    analyzer = LlmPacketAnalyzer(
        PacketAnalyzerConfig(
            eventloom_path=eventloom_path,
            session_id="agent-1",
            upstream_base_url="https://upstream.example/v1",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = analyzer.forward(
        "POST",
        "/v1/responses",
        headers={"content-type": "application/json"},
        body=json.dumps(
            {
                "model": "gpt-test",
                "input": "Remember the dashboard owner is Mira.",
            }
        ).encode(),
    )
    analyzer.close()

    assert response.status_code == 200
    projection = project_packet_events(eventloom_path=eventloom_path, session_id="agent-1")
    assert projection.projected == 1

    status = packet_memory_report(eventloom_path=eventloom_path, session_id="agent-1")
    assert status["status"] == "ok"
    assert status["details"] == {
        "captured": 1,
        "projected": 1,
        "unprojected": 0,
        "reinforced": 0,
        "eligible": 1,
    }

    cli_result = CliRunner().invoke(
        app,
        ["packet-status", "--eventloom-path", str(eventloom_path), "--session-id", "agent-1"],
    )
    assert cli_result.exit_code == 0
    assert "captured=1 projected=1 unprojected=0 reinforced=0 eligible=1" in cli_result.output

    fabric = MemoryFabric(eventloom_path=str(eventloom_path), tracer_disabled=True)
    with (
        patch.object(fabric, "connect", AsyncMock()),
        patch.object(fabric.query_router, "query", AsyncMock(return_value=[])),
    ):
        assembly = await fabric.assemble_context(
            "current operating context",
            session_id="agent-1",
            limit=1,
        )

    assert [context.source for context in assembly.contexts] == ["packet_memory"]
    assert "Mira" in assembly.contexts[0].content
    assert "eventloom://agent-1/events/2#" in assembly.prompt
    assert assembly.context_counts == {"graph": 0, "verbatim": 0, "packet_memory": 1, "replay": 2}

    events = EventLog(eventloom_path / "agent-1.jsonl").read_all()
    assert [event.type for event in events] == ["llm.packet.completed", "llm.packet.projected"]
