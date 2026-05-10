"""Operator guidance for activating the LLM packet-memory pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PacketCaptureGuidance:
    """Concrete commands for packet capture activation."""

    analyzer_command: str
    projector_command: str
    client_base_url: str

    def next_steps(self) -> list[str]:
        """Return human-readable activation steps."""
        return [
            f"Start packet analyzer: {self.analyzer_command}",
            f"Start packet projector: {self.projector_command}",
            f"Point OpenAI-compatible clients at {self.client_base_url}.",
        ]


def build_packet_capture_guidance(
    *,
    eventloom_path: str | Path,
    session_id: str,
    upstream_base_url: str = "https://api.openai.com/v1",
    upstream_api_key_env: str = "OPENAI_API_KEY",
    host: str = "127.0.0.1",
    port: int = 8787,
    graph: bool = True,
) -> PacketCaptureGuidance:
    """Build copy-pasteable commands for analyzer and projector processes."""
    eventloom = Path(eventloom_path)
    analyzer_command = (
        "zaxy packet-analyzer "
        f"--eventloom-path {eventloom} "
        f"--session-id {session_id} "
        f"--upstream-base-url {upstream_base_url} "
        f'--upstream-api-key "${upstream_api_key_env}" '
        f"--host {host} "
        f"--port {port}"
    )
    projector_command = (
        "zaxy packet-project "
        f"--eventloom-path {eventloom} "
        f"--session-id {session_id} "
        "--watch"
    )
    if graph:
        projector_command = f"{projector_command} --graph"
    return PacketCaptureGuidance(
        analyzer_command=analyzer_command,
        projector_command=projector_command,
        client_base_url=f"http://{host}:{port}/v1",
    )
