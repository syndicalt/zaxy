"""Opt-in, Eventloom-owned content-addressed store for full tool I/O.

Default capture stays lean: ``tool.call.completed`` keeps only ``argument_keys``,
and ``command.completed`` truncates output to ``OUTPUT_EXCERPT_CHARS``. That keeps
the log token-cheap and never persists secret argument values -- good defaults,
but it caps the "full, replayable provenance" thesis for tool I/O.

When ``ZAXY_OFFLOAD_TOOL_IO`` is enabled, the *full* text is additionally written
to a content-addressed blob under ``<eventloom>/refs/`` and the lean event carries
a ``full_io_ref`` pointer ``{ref, sha256, bytes}``. Properties:

- **Self-contained & tamper-evident.** The blob id *is* its sha256, and it lives
  inside the Eventloom directory -- unlike ``codex_source_ref``, which points at
  foreign Codex rollout files the hash chain can't attest to.
- **Lean by default.** Context / ``memory_checkout`` still see only the summary;
  the full text is fetched on demand (drill-down) by sha.
- **Privacy preserved.** Opt-in, and tool arguments are still secret-masked before
  offload (values under secret-looking keys are dropped).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

OFFLOAD_ENV = "ZAXY_OFFLOAD_TOOL_IO"
_TRUTHY = {"1", "true", "yes", "on"}

# Mask argument values whose KEY name looks secret-bearing.
_SECRET_KEY = re.compile(
    r"(?i)(?:api[-_]?key|authorization|bearer|cookie|credential|password|private[-_]?key|secret|token)"
)


def tool_io_offload_enabled() -> bool:
    """True when full tool-I/O retention is opted in via env."""
    return os.environ.get(OFFLOAD_ENV, "").strip().lower() in _TRUTHY


def redact_secret_args(arguments: dict[str, Any]) -> dict[str, Any]:
    """Return args with secret-looking values masked, others preserved verbatim."""
    return {
        key: ("<redacted>" if _SECRET_KEY.search(str(key)) else value)
        for key, value in arguments.items()
    }


def _refs_root(eventloom_path: str | Path) -> Path:
    base = Path(eventloom_path)
    if base.suffix == ".jsonl":
        base = base.parent
    return base / "refs"


def _blob_path(eventloom_path: str | Path, sha256: str) -> Path:
    return _refs_root(eventloom_path) / sha256[:2] / f"{sha256}.blob"


def write_offload_ref(eventloom_path: str | Path, content: str) -> dict[str, Any]:
    """Content-address ``content`` under ``<eventloom>/refs/`` and return a pointer.

    Idempotent: identical content yields an identical ref, written once.
    """
    data = content.encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    blob = _blob_path(eventloom_path, sha)
    if not blob.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
    return {"ref": f"refs/{sha[:2]}/{sha}.blob", "sha256": sha, "bytes": len(data)}


def read_offload_ref(eventloom_path: str | Path, sha256: str) -> str | None:
    """Read a blob back by sha, verifying integrity. Returns None if missing/tampered."""
    blob = _blob_path(eventloom_path, sha256)
    if not blob.exists():
        return None
    data = blob.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        return None  # tamper-evident: id must equal content hash
    return data.decode("utf-8")


def offload_command_output(
    eventloom_path: str | Path, *, stdout: str, stderr: str
) -> dict[str, Any] | None:
    """Offload full command output when it exceeds the inline excerpt. Else None."""
    from zaxy.lifecycle import OUTPUT_EXCERPT_CHARS

    if len(stdout) <= OUTPUT_EXCERPT_CHARS and len(stderr) <= OUTPUT_EXCERPT_CHARS:
        return None
    full = f"=== stdout ===\n{stdout}\n=== stderr ===\n{stderr}" if stderr else stdout
    return write_offload_ref(eventloom_path, full)


def offload_tool_arguments(
    eventloom_path: str | Path, arguments: dict[str, Any]
) -> dict[str, Any] | None:
    """Offload full tool arguments (secret-masked) for provenance. Else None."""
    if not arguments:
        return None
    masked = redact_secret_args(arguments)
    full = json.dumps(masked, ensure_ascii=False, sort_keys=True)
    return write_offload_ref(eventloom_path, full)
