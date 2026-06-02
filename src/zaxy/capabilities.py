"""Model-facing memory capability manifest for Zaxy sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zaxy.capture_manager import inspect_codex_capture
from zaxy.doctor import packet_memory_report
from zaxy.hooks import inspect_hook_status
from zaxy.memory_status import inspect_memory_status
from zaxy.security import validate_session_id


def build_memory_capabilities(
    *,
    eventloom_path: str | Path,
    session_id: str,
    workspace_root: str | Path | None = None,
    current_task: str | None = None,
) -> dict[str, Any]:
    """Return a compact manifest that teaches a model how to use Zaxy."""
    sid = validate_session_id(session_id)
    base = Path(eventloom_path)
    workspace = Path(workspace_root or Path.cwd()).resolve()
    memory_status = inspect_memory_status(base)
    session_status = next((session for session in memory_status.sessions if session.session_id == sid), None)
    hook_status = inspect_hook_status(eventloom_path=base, workspace_root=workspace)
    packet_status = packet_memory_report(eventloom_path=base, session_id=sid)
    manifest: dict[str, Any] = {
        "session_id": sid,
        "current_task": current_task,
        "purpose": (
            "Zaxy is the active persistent memory substrate for this session: "
            "checkout relevant memory before important work, capture meaningful work after it, "
            "and reinforce cited context that was useful."
        ),
        "recommended_next_call": {
            "tool": "memory_checkout",
            "arguments": {
                "query": current_task or "current task, project direction, and recent decisions",
                "session_id": sid,
            },
            "reason": "Session-start memory is only a bootstrap; checkout refreshes cited working state.",
        },
        "reminder_policy": _reminder_policy(),
        "ambient_loop": _ambient_loop(),
        "tools": _tool_guidance(),
        "status": {
            "eventloom": {
                "path": str(base.resolve()),
                "session_exists": session_status is not None,
                "latest_seq": session_status.latest_seq if session_status else None,
                "latest_hash": session_status.latest_hash if session_status else None,
                "integrity_ok": session_status.integrity_ok if session_status else None,
                "skipped_log_count": len(memory_status.skipped_logs),
                "skipped_logs": [skipped.path for skipped in memory_status.skipped_logs],
            },
            "hooks": {
                "status": hook_status["status"],
                "message": hook_status["message"],
            },
            "packet_memory": {
                "status": packet_status["status"],
                "message": packet_status["message"],
                "details": packet_status["details"],
            },
            "graph": {
                "status": "available_through_memory_checkout",
                "message": "Use memory_checkout or memory_query for Neo4j-backed temporal retrieval.",
            },
        },
    }
    manifest["prompt"] = format_memory_capabilities(manifest)
    return manifest


def format_memory_capabilities(manifest: dict[str, Any]) -> str:
    """Format the manifest as concise prompt-ready instructions."""
    next_call = manifest["recommended_next_call"]
    eventloom = manifest["status"]["eventloom"]
    packet = manifest["status"]["packet_memory"]
    lines = [
        "# Zaxy Memory Contract",
        f"Session: {manifest['session_id']}",
    ]
    if manifest.get("current_task"):
        lines.append(f"Current task: {manifest['current_task']}")
    lines.extend(
        [
            "",
            "Zaxy is active. Do not treat session-start memory as sufficient.",
            "Refresh memory at context boundaries, then capture meaningful work.",
            "",
            "Use:",
            "- session start: memory_capabilities, then memory_checkout",
            "- before major work or roadmap decisions: memory_checkout",
            "- after compaction/resume: memory_checkout",
            "- after meaningful work: context_after_turn or memory_append",
            "- when exact source is needed: memory_verbatim",
            "- when retrieved context was used: memory_feedback",
            "- if memory.reminder.suggested appears: follow its memory_checkout call before answering",
            "",
            (
                f"Recommended next call: {next_call['tool']}("
                f"query={next_call['arguments']['query']!r}, session_id={manifest['session_id']!r})"
            ),
            (
                "Status: "
                f"eventloom latest={eventloom['latest_seq'] or '-'} "
                f"integrity={_status_label(eventloom['integrity_ok'])}; "
                f"packet_memory={packet['status']}"
            ),
        ]
    )
    return "\n".join(lines)


def build_memory_bootstrap(
    *,
    eventloom_path: str | Path,
    session_id: str,
    workspace_root: str | Path | None = None,
    current_task: str | None = None,
) -> dict[str, Any]:
    """Return the compact session-start handoff packet for model-facing memory use."""
    sid = validate_session_id(session_id)
    workspace = Path(workspace_root or Path.cwd()).resolve()
    capabilities = build_memory_capabilities(
        eventloom_path=eventloom_path,
        session_id=sid,
        workspace_root=workspace,
        current_task=current_task,
    )
    checkout_query = current_task or "current task, project direction, and recent decisions"
    bootstrap: dict[str, Any] = {
        "mode": "session_start",
        "session_id": sid,
        "workspace": str(workspace),
        "current_task": current_task,
        "capabilities": capabilities,
        "startup_sequence": [
            {
                "tool": "memory_capabilities",
                "arguments": {"session_id": sid, "current_task": current_task},
                "reason": "Confirm active Zaxy memory posture and health.",
            },
            {
                "tool": "memory_checkout",
                "arguments": {"query": checkout_query, "session_id": sid},
                "reason": "Load cited current working state before relying on memory.",
            },
        ],
        "capture": _capture_summary(workspace),
        "trust_policy": {
            "prefer": "cited current facts from memory_checkout",
            "ignore": "uncited, superseded, or warning-bearing context until refreshed",
            "record": "meaningful decisions, completed work, corrections, and retrieval feedback",
        },
    }
    bootstrap["prompt"] = format_memory_bootstrap(bootstrap)
    return bootstrap


def format_memory_bootstrap(bootstrap: dict[str, Any]) -> str:
    """Format the bootstrap packet as short prompt-ready startup context."""
    capture = bootstrap["capture"]
    configured = "configured" if capture.get("configured") else "not configured"
    running = "running" if capture.get("running") else "not running"
    next_checkout = bootstrap["startup_sequence"][1]
    query = next_checkout["arguments"]["query"]
    lines = [
        "# Zaxy Session Bootstrap",
        f"Session: {bootstrap['session_id']}",
    ]
    if bootstrap.get("current_task"):
        lines.append(f"Current task: {bootstrap['current_task']}")
    lines.extend(
        [
            f"Capture: {configured}, {running}",
            "",
            "Startup sequence:",
            "1. memory_capabilities",
            f"2. memory_checkout(query={query!r})",
            "",
            "Trust policy:",
            f"- Prefer: {bootstrap['trust_policy']['prefer']}",
            f"- Ignore: {bootstrap['trust_policy']['ignore']}",
            f"- Record: {bootstrap['trust_policy']['record']}",
            "",
            "Call memory_checkout before answering roadmap or implementation questions.",
            "Call memory_feedback when cited checkout context was used.",
        ]
    )
    return "\n".join(lines)


def _ambient_loop() -> dict[str, dict[str, str]]:
    return {
        "session_start": {
            "tool": "memory_capabilities",
            "reason": "Learn the active memory contract and health before work begins.",
        },
        "before_major_work": {
            "tool": "memory_checkout",
            "reason": "Retrieve cited, current working state before decisions or implementation.",
        },
        "after_compaction_or_resume": {
            "tool": "memory_checkout",
            "reason": "Restore working memory after local context may have collapsed.",
        },
        "after_meaningful_work": {
            "tool": "context_after_turn",
            "reason": "Persist the completed turn and prepare compact context for the next one.",
        },
        "when_exact_source_is_needed": {
            "tool": "memory_verbatim",
            "reason": "Retrieve exact Eventloom source chunks with citations.",
        },
        "when_context_is_used": {
            "tool": "memory_feedback",
            "reason": "Reinforce useful cited memory so future checkout ranks it higher.",
        },
    }


def _reminder_policy() -> dict[str, Any]:
    return {
        "event_type": "memory.reminder.suggested",
        "triggers": [
            "session_start",
            "resume",
            "compaction",
            "long_session",
            "long_tool_run",
            "where_are_we_question",
        ],
        "instruction": (
            "When a reminder is suggested, call memory_bootstrap if awareness is unclear, "
            "then call memory_checkout and rely only on cited current facts."
        ),
    }


def _tool_guidance() -> list[dict[str, str]]:
    return [
        {"name": "memory_capabilities", "use_when": "At session start or when tool awareness is unclear."},
        {"name": "memory_checkout", "use_when": "Before major work, after compaction/resume, and for what-next questions."},
        {"name": "context_after_turn", "use_when": "After a meaningful assistant/user turn should become durable context."},
        {"name": "memory_append", "use_when": "When recording a structured task, decision, observation, or correction."},
        {"name": "memory_feedback", "use_when": "When cited context was used, helpful, or irrelevant."},
        {"name": "memory_verbatim", "use_when": "When exact source text, identifiers, or citations matter."},
        {"name": "memory_replay", "use_when": "For audit, handoff, or reconstructing session history."},
    ]


def _capture_summary(workspace: Path) -> dict[str, Any]:
    try:
        runtime = inspect_codex_capture(workspace=workspace)
    except (OSError, ValueError) as exc:
        return {
            "configured": False,
            "running": False,
            "pids": [],
            "latest_observation": None,
            "error": str(exc),
        }
    return {
        "configured": bool(runtime.get("configured", False)),
        "running": bool(runtime.get("running", False)),
        "pids": list(runtime.get("pids", [])),
        "latest_observation": runtime.get("latest_observation"),
    }


def _status_label(value: bool | None) -> str:
    if value is True:
        return "ok"
    if value is False:
        return "failed"
    return "unknown"
