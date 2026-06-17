"""First-run onboarding orchestration."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zaxy.capture_manager import inspect_codex_capture, start_codex_capture
from zaxy.codex_capture import write_codex_capture_config
from zaxy.config import Settings
from zaxy.core import MemoryFabric
from zaxy.doctor import run_doctor
from zaxy.domain import domain_default_session, slug_domain
from zaxy.hooks import (
    build_hook_payload,
    hook_event_type,
    inspect_hook_status,
    render_hook_config,
    write_hook_config,
)
from zaxy.install import resolve_zaxy_executable
from zaxy.integrations import (
    render_codex_mcp_add_command,
    render_mcp_client_config,
    write_codex_mcp_config,
)
from zaxy.local_profile import write_local_profile
from zaxy.mcp_runtime import EmbeddedMcpRuntimeCoordinator
from zaxy.packet_guidance import build_packet_capture_guidance
from zaxy.runtime import LocalEmbeddedGraphRuntime, LocalNeo4jRuntime, LocalPgGraphRuntime
from zaxy.session import SessionManager


@dataclass(frozen=True)
class OnboardingStep:
    """A single observable onboarding step."""

    name: str
    status: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class OnboardingResult:
    """Summary of a first-run onboarding execution."""

    status: str
    workspace: str
    domain: str
    session_id: str
    profile: dict[str, Any]
    steps: list[OnboardingStep] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    doctor: dict[str, Any] = field(default_factory=dict)
    hook_status: dict[str, Any] = field(default_factory=dict)
    capture: dict[str, Any] = field(default_factory=dict)


def apply_onboarding_preset(
    preset: str | None,
    *,
    workspace: str | Path,
    mcp_client: str | None,
    mcp_output: str | Path | None,
    hook_client: str | None,
    hook_output: str | Path | None,
    local_profile_output: str | Path | None,
    infra: str,
    capture_mode: str = "deterministic",
) -> dict[str, Any]:
    """Expand a named onboarding preset without overriding explicit options."""
    normalized_capture_mode = _normalize_capture_mode(capture_mode)
    if preset is None:
        return {
            "mcp_client": mcp_client,
            "mcp_output": mcp_output,
            "hook_client": hook_client,
            "hook_output": hook_output,
            "local_profile_output": local_profile_output,
            "infra": infra,
            "projection_backend": None,
            "capture_mode": normalized_capture_mode,
        }
    normalized = preset.casefold().strip().replace("_", "-")
    if normalized not in {"local-claude", "local-codex", "local-embedded-codex"}:
        raise ValueError("preset must be one of: local-claude, local-codex, local-embedded-codex")
    root = Path(workspace)
    if normalized in {"local-codex", "local-embedded-codex"}:
        return {
            "mcp_client": mcp_client or "codex",
            "mcp_output": Path(mcp_output) if mcp_output is not None else None,
            "hook_client": hook_client or "codex",
            "hook_output": (
                Path(hook_output)
                if hook_output is not None
                else root / ".codex" / "zaxy-capture.json"
            ),
            "local_profile_output": (
                Path(local_profile_output)
                if local_profile_output is not None
                else root / ".env.local"
            ),
            "infra": infra if infra != "none" else "check",
            "projection_backend": "embedded" if normalized == "local-embedded-codex" else None,
            "capture_mode": normalized_capture_mode,
        }
    return {
        "mcp_client": mcp_client or "claude-desktop",
        "mcp_output": Path(mcp_output) if mcp_output is not None else root / "zaxy-mcp.json",
        "hook_client": hook_client or "claude-code",
        "hook_output": Path(hook_output) if hook_output is not None else root / ".claude" / "settings.local.json",
        "local_profile_output": (
            Path(local_profile_output)
            if local_profile_output is not None
            else root / ".env.local"
        ),
        "infra": infra if infra != "none" else "check",
        "projection_backend": None,
        "capture_mode": normalized_capture_mode,
    }


async def run_onboarding(
    workspace: str | Path,
    *,
    eventloom_path: str | Path = ".eventloom",
    domain: str | None = None,
    session_id: str | None = None,
    mcp_client: str | None = None,
    mcp_output: str | Path | None = None,
    hook_client: str | None = None,
    hook_output: str | Path | None = None,
    local_profile_output: str | Path | None = None,
    infra: str = "none",
    projection_backend: str | None = None,
    pggraph_dsn: str | None = None,
    pggraph_repo: str | Path | None = None,
    capture_mode: str = "deterministic",
    packet_capture: bool = False,
    packet_upstream_base_url: str = "https://api.openai.com/v1",
    packet_port: int = 8787,
    capture_action: str = "none",
    codex_mcp_install: str = "command",
    codex_mcp_conflict_path: str | Path | None = None,
    codex_trusted_project: bool = False,
    codex_home: str | Path | None = None,
    agent_instructions: bool = True,
    zaxy_executable: str | Path | None = None,
    force: bool = False,
    fabric_factory: Callable[[str], MemoryFabric] = MemoryFabric,
    runtime_factory: Callable[[], Any] | None = None,
) -> OnboardingResult:
    """Run the idempotent first-run onboarding flow."""
    root = Path(workspace).resolve()
    raw_eventloom = Path(eventloom_path)
    eventloom = raw_eventloom if raw_eventloom.is_absolute() else root / raw_eventloom
    resolved_domain = slug_domain(domain) if domain else slug_domain(root.name)
    sid = session_id or domain_default_session(resolved_domain)
    infra_action = _normalize_infra(infra)
    codex_install_mode = _normalize_codex_mcp_install(codex_mcp_install)
    normalized_capture_mode = _normalize_capture_mode("hybrid" if packet_capture else capture_mode)
    executable = resolve_zaxy_executable(zaxy_executable)
    _validate_render_requests(
        mcp_client=mcp_client,
        mcp_output=mcp_output,
        hook_client=hook_client,
        hook_output=hook_output,
    )
    preflight_paths: list[str | Path] = []
    if mcp_output is not None:
        preflight_paths.append(mcp_output)
    if hook_output is not None and _normalize_hook_client_name(hook_client or "") != "codex":
        preflight_paths.append(hook_output)
    _preflight_outputs(force=force, paths=preflight_paths)

    steps: list[OnboardingStep] = []
    mcp_install_command: str | None = None
    mcp_installed_path: Path | None = None
    codex_conflict_path = Path(codex_mcp_conflict_path) if codex_mcp_conflict_path is not None else None
    eventloom.mkdir(parents=True, exist_ok=True)
    steps.append(OnboardingStep("eventloom", "ok", "Eventloom directory is ready", str(eventloom)))
    selected_projection_backend = projection_backend or Settings().projection_backend
    if selected_projection_backend.casefold().strip() == "embedded":
        embedded_runtime_report = EmbeddedMcpRuntimeCoordinator.from_embedded_graph_path(
            Path(eventloom) / "projections" / "embedded.kuzu"
        ).repair_stale_runtime()
        if embedded_runtime_report["repaired"] or embedded_runtime_report["status"] != "ok":
            steps.append(
                OnboardingStep(
                    "embedded_mcp_runtime",
                    str(embedded_runtime_report["status"]),
                    str(embedded_runtime_report["message"]),
                    str(embedded_runtime_report["owner_path"]),
                )
            )

    if local_profile_output is not None:
        written = write_local_profile(
            Path(local_profile_output),
            projection_backend=selected_projection_backend,
            force=force,
        )
        steps.append(OnboardingStep("local_profile", "ok", "Local retrieval profile written", str(written)))

    if mcp_client is not None:
        if _normalize_mcp_client_name(mcp_client) == "codex":
            if mcp_output is not None:
                raise ValueError("Codex onboarding renders a CLI install command; do not provide mcp_output")
            if codex_install_mode == "command":
                if codex_conflict_path is not None:
                    steps.append(
                        OnboardingStep(
                            "mcp_config",
                            "warning",
                            "Existing Codex zaxy MCP config needs review before replacement",
                            str(codex_conflict_path),
                        )
                    )
                else:
                    mcp_install_command = shlex.join(
                        render_codex_mcp_add_command(
                            eventloom_path=str(eventloom),
                            domain=resolved_domain,
                            zaxy_executable=executable,
                        )
                    )
                    steps.append(OnboardingStep("mcp_config", "preview", "codex MCP install command rendered"))
            else:
                mcp_installed_path = write_codex_mcp_config(
                    scope=codex_install_mode,
                    workspace=root,
                    eventloom_path=str(eventloom),
                    domain=resolved_domain,
                    zaxy_executable=executable,
                    force=force,
                    trusted_project=codex_trusted_project,
                    codex_home=codex_home,
                )
                steps.append(
                    OnboardingStep(
                        "mcp_config",
                        "ok",
                        "codex MCP config installed",
                        str(mcp_installed_path),
                    )
                )
        else:
            config = render_mcp_client_config(
                mcp_client,
                eventloom_path=str(eventloom),
                domain=resolved_domain,
                zaxy_executable=executable,
            )
            if mcp_output is not None:
                written = _write_json(Path(mcp_output), config, force=force)
                steps.append(OnboardingStep("mcp_config", "ok", f"{mcp_client} MCP config written", str(written)))
            else:
                steps.append(OnboardingStep("mcp_config", "preview", f"{mcp_client} MCP config rendered"))

    if hook_client is not None:
        if _normalize_hook_client_name(hook_client) == "codex" and hook_output is not None:
            written = write_codex_capture_config(
                workspace=root,
                eventloom_path=eventloom,
                session_id=sid,
                force=force,
            )
            steps.append(OnboardingStep("codex_capture", "ok", "Codex local capture config written", str(written)))
        else:
            hook_config = render_hook_config(
                hook_client,
                eventloom_path=str(eventloom),
                domain=resolved_domain,
            )
            if hook_output is not None:
                written = write_hook_config(Path(hook_output), hook_config, force=force)
                steps.append(OnboardingStep("hook_config", "ok", f"{hook_client} hook config written", str(written)))
            else:
                steps.append(OnboardingStep("hook_config", "preview", f"{hook_client} hook config rendered"))

    if agent_instructions:
        written = write_agent_activation_instructions(
            root,
            eventloom_path=eventloom,
            session_id=sid,
        )
        steps.append(
            OnboardingStep(
                "agent_instructions",
                "ok",
                "Model-visible Zaxy activation instructions installed",
                str(written),
            )
        )

    if infra_action != "none":
        settings = _onboarding_settings(
            eventloom=eventloom,
            session_id=sid,
            domain=resolved_domain,
            projection_backend=selected_projection_backend,
            pggraph_dsn=pggraph_dsn,
            pggraph_repo=pggraph_repo,
        )
        runtime = runtime_factory() if runtime_factory is not None else _build_runtime(settings)
        steps.append(_run_infra_action(runtime, infra_action))

    fabric = fabric_factory(str(eventloom))
    try:
        profile = await fabric.ensure_session_initialized(root, session_id=sid)
    finally:
        await fabric.close()
    profile_payload = {
        "workspace_type": profile.workspace_type,
        "confidence": profile.confidence,
        "signals": profile.signals,
        "instructions_profile": profile.instructions_profile,
    }
    steps.append(OnboardingStep("session_genesis", "ok", f"Session {sid} registered"))

    heartbeat = _append_heartbeat(eventloom, session_id=sid, source="zaxy-init", workspace=root)
    steps.append(OnboardingStep("heartbeat", "ok", f"Hook heartbeat recorded seq={heartbeat.seq}"))

    normalized_capture_action = _normalize_capture_action(capture_action)
    if normalized_capture_action == "start":
        if _normalize_hook_client_name(hook_client or "") != "codex":
            steps.append(OnboardingStep("capture_runtime", "warning", "Capture start requires Codex local capture config"))
        else:
            try:
                capture_result = start_codex_capture(workspace=root)
                steps.append(OnboardingStep("capture_runtime", "ok", str(capture_result["message"])))
            except (FileNotFoundError, ValueError) as exc:
                steps.append(OnboardingStep("capture_runtime", "error", str(exc)))

    settings = _onboarding_settings(
        eventloom=eventloom,
        session_id=sid,
        domain=resolved_domain,
        projection_backend=selected_projection_backend,
        pggraph_dsn=pggraph_dsn,
        pggraph_repo=pggraph_repo,
    )
    doctor = run_doctor(settings=settings, workspace_root=root, zaxy_executable=executable)
    capture = _build_capture_summary(workspace=root, doctor=doctor)
    steps.append(
        OnboardingStep(
            "doctor",
            _onboarding_doctor_status(
                doctor,
                hook_installation_required=hook_client is not None,
                agent_instructions_required=agent_instructions,
            ),
            _onboarding_doctor_message(
                doctor,
                hook_installation_required=hook_client is not None,
                agent_instructions_required=agent_instructions,
            ),
        )
    )
    hook_status = inspect_hook_status(eventloom_path=eventloom, workspace_root=root, session_id=sid)
    steps.append(
        OnboardingStep(
            "hook_status",
            _onboarding_hook_status(hook_status, hook_client=hook_client),
            hook_status["message"],
        )
    )
    return OnboardingResult(
        status=_overall_status(step.status for step in steps),
        workspace=str(root),
        domain=resolved_domain,
        session_id=sid,
        profile=profile_payload,
        steps=steps,
        next_steps=_build_next_steps(
            workspace=root,
            eventloom=eventloom,
            session_id=sid,
            mcp_client=mcp_client,
            mcp_output=mcp_output,
            mcp_install_command=mcp_install_command,
            mcp_installed_path=mcp_installed_path,
            codex_mcp_conflict_path=codex_conflict_path,
            infra_action=infra_action,
            projection_backend=selected_projection_backend,
            capture_mode=normalized_capture_mode,
            packet_capture=packet_capture,
            packet_upstream_base_url=packet_upstream_base_url,
            packet_port=packet_port,
            steps=steps,
        ),
        doctor=doctor,
        hook_status=hook_status,
        capture=capture,
    )


def format_onboarding_result(
    result: OnboardingResult,
    *,
    verbose: bool = False,
    verbose_command: str | None = None,
) -> str:
    """Format onboarding output for humans."""
    lines = [
        f"{_status_badge(result.status, bracketed=False)}  Zaxy init complete: {result.status}",
        f"Workspace: {result.workspace}",
        f"Domain: {result.domain}",
        f"Session: {result.session_id}",
        f"Profile: {result.profile['workspace_type']} ({result.profile['confidence']})",
        _format_readiness_summary(result),
    ]
    if result.steps:
        if verbose:
            lines.extend(["", "Setup:"])
            lines.extend(_format_step_row(step) for step in result.steps)
        else:
            lines.extend(["", _format_setup_summary(result.steps)])
            issues = [step for step in result.steps if step.status in {"warning", "error"}]
            if issues:
                lines.extend(["", "Setup issues:"])
                lines.extend(_format_step_row(step) for step in issues)
    if result.capture:
        lines.append("")
        lines.extend(_format_capture_summary(result.capture))
    if result.next_steps:
        lines.extend(_format_next_steps(result.next_steps, verbose=verbose, verbose_command=verbose_command))
    return "\n".join(lines)


def onboarding_result_payload(result: OnboardingResult) -> dict[str, Any]:
    """Return the machine-readable onboarding payload with readiness summary."""
    payload = {
        "status": result.status,
        "workspace": result.workspace,
        "domain": result.domain,
        "session_id": result.session_id,
        "profile": result.profile,
        "steps": [step.__dict__.copy() for step in result.steps],
        "next_steps": list(result.next_steps),
        "doctor": result.doctor,
        "hook_status": result.hook_status,
        "capture": result.capture,
    }
    payload["setup"] = _onboarding_setup_payload(result.steps)
    payload["readiness"] = _onboarding_readiness_payload(result)
    return payload


def _onboarding_setup_payload(steps: list[OnboardingStep]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    issues: list[dict[str, str | None]] = []
    pending: list[dict[str, str | None]] = []
    for step in steps:
        counts[step.status] = counts.get(step.status, 0) + 1
        if step.status in {"warning", "error"}:
            issues.append(step.__dict__.copy())
        elif step.status == "preview":
            pending.append(step.__dict__.copy())
    return {
        "status": _overall_status(step.status for step in steps),
        "summary": _format_setup_summary(steps),
        "counts": counts,
        "issues": issues,
        "pending": pending,
    }


def _onboarding_readiness_payload(result: OnboardingResult) -> dict[str, Any]:
    actions = _onboarding_readiness_actions(result.next_steps)
    reasons = _onboarding_readiness_reasons(result)
    status = "ready" if result.status == "ok" and not actions and not reasons else "needs_action"
    return {
        "status": status,
        "setup_status": result.status,
        "summary": _format_readiness_summary(result),
        "reasons": reasons,
        "reason_count": len(reasons),
        "actions": actions,
        "required_action_count": len(actions),
        "action_items": _onboarding_readiness_action_items(result.next_steps),
        "blocking_diagnostics": _onboarding_doctor_diagnostics(result, blocking=True),
        "non_blocking_diagnostics": _onboarding_doctor_diagnostics(result, blocking=False),
        "capture": result.capture,
    }


def _onboarding_readiness_actions(next_steps: list[str]) -> list[str]:
    actions: list[str] = []
    for step in next_steps:
        if _required_next_action(step) is not None:
            actions.append(step)
    return actions


def _onboarding_readiness_action_items(next_steps: list[str]) -> list[dict[str, Any]]:
    notes = [step for step in next_steps if _is_onboarding_note(step)]
    items: list[dict[str, Any]] = []
    for step in next_steps:
        action = _required_next_action(step)
        if action is not None:
            label, command = action
            hints: list[str] = []
            if command is not None:
                hints.extend(_command_hints(label, command))
            hints.extend(_required_action_hints(label, notes))
            items.append({"label": label, "command": command, "source": step, "hints": hints})
    return items


def _required_next_action(step: str) -> tuple[str, str | None] | None:
    """Return a machine-readable required action for a human next-step line."""
    if _split_deferred_command(step) is not None:
        return None
    action = _split_action_command(step)
    if action is not None:
        return action
    if _extract_leading_command(step) is not None:
        return None
    if step.startswith("If Zaxy MCP tools are absent"):
        return None
    if _is_onboarding_note(step):
        return None
    if step.startswith("Add ") or step.startswith("Restart "):
        return step, None
    if step.startswith("Review existing Codex MCP config before replacing zaxy at "):
        return "Review existing Codex MCP config before replacing zaxy:", None
    return step, None


def _onboarding_readiness_reasons(result: OnboardingResult) -> list[str]:
    return [
        f"{step.name} {step.status}: {step.message}"
        for step in result.steps
        if step.status in {"warning", "error"}
    ]


def _onboarding_doctor_diagnostics(result: OnboardingResult, *, blocking: bool) -> list[dict[str, Any]]:
    """Return doctor warnings split by whether onboarding readiness treats them as blocking."""
    ignored = _ignored_onboarding_doctor_checks(
        hook_installation_required=_onboarding_step_present(result.steps, {"hook_config", "codex_capture"}),
        agent_instructions_required=_onboarding_step_present(result.steps, {"agent_instructions"}),
    )
    diagnostics: list[dict[str, Any]] = []
    for check in result.doctor.get("checks", []):
        if check.get("status") == "ok":
            continue
        is_blocking = check.get("name") not in ignored
        if is_blocking == blocking:
            diagnostics.append(dict(check))
    return diagnostics


def _onboarding_step_present(steps: list[OnboardingStep], names: set[str]) -> bool:
    return any(step.name in names for step in steps)


def _format_step_row(step: OnboardingStep) -> str:
    suffix = f" ({step.path})" if step.path else ""
    return f"{_status_badge(step.status)} {step.name} - {step.message}{suffix}"


def _status_badge(status: str, *, bracketed: bool = True) -> str:
    normalized = status.casefold().strip()
    labels = {
        "ok": "OK",
        "warning": "WARN",
        "error": "ERR",
        "preview": "INFO",
    }
    label = labels.get(normalized, normalized.upper() or "INFO")
    return f"[{label}]" if bracketed else label


def _format_readiness_summary(result: OnboardingResult) -> str:
    actions = _onboarding_readiness_actions(result.next_steps)
    if actions:
        noun = "action" if len(actions) == 1 else "actions"
        return f"Readiness: needs action ({len(actions)} required {noun})"
    if result.status == "ok":
        return "Readiness: ready"
    return "Readiness: review setup issues"


def _format_setup_summary(steps: list[OnboardingStep]) -> str:
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.status] = counts.get(step.status, 0) + 1
    ordered = ["ok", "warning", "error", "preview"]
    parts = [f"{counts.pop(status)} {status}" for status in ordered if status in counts]
    parts.extend(f"{count} {status}" for status, count in sorted(counts.items()))
    return "Setup: " + ", ".join(parts)


def _format_next_steps(
    next_steps: list[str],
    *,
    verbose: bool = False,
    verbose_command: str | None = None,
) -> list[str]:
    required: list[tuple[str, str | None]] = []
    checks: list[str] = []
    fallbacks: list[str] = []
    later: list[tuple[str, str | None]] = []
    notes: list[str] = []

    for step in next_steps:
        deferred = _split_deferred_command(step)
        if deferred is not None:
            later.append(deferred)
            continue
        formatted = _split_action_command(step)
        if formatted is not None:
            required.append(formatted)
            continue
        command = _extract_leading_command(step)
        if command is not None:
            checks.append(command)
            continue
        if step.startswith("If Zaxy MCP tools are absent"):
            fallback = step.split(": ", 1)[1] if ": " in step else step
            fallbacks.append(fallback)
            continue
        if _is_onboarding_note(step):
            notes.append(step)
            continue
        required_action = _required_next_action(step)
        if required_action is not None:
            label, command = required_action
            if command is None and _is_codex_mcp_conflict_review_action(step):
                label = step
            required.append((label, command))

    lines: list[str] = []
    if required:
        lines.extend(["", "Required next actions:"])
        for index, (label, command) in enumerate(required, start=1):
            lines.append(f"{index}. {label}")
            if command:
                lines.append(f"   {command}")
                lines.extend(f"   {hint}" for hint in _command_hints(label, command))
            lines.extend(f"   {hint}" for hint in _required_action_hints(label, notes))
    if not verbose:
        if checks or fallbacks or later or notes:
            command = verbose_command or "zaxy init --verbose"
            lines.extend(
                [
                    "",
                    f"More: run {command} to show checks, fallbacks, later commands, and notes.",
                ]
            )
        return lines
    if checks:
        lines.extend(["", "Useful checks:"])
        lines.extend(f"- {check}" for check in checks)
    if fallbacks:
        lines.extend(["", "Fallbacks:"])
        lines.extend(f"- {fallback}" for fallback in fallbacks)
    if later:
        lines.extend(["", "Later:"])
        for label, command in later:
            lines.append(f"- {label}")
            if command:
                lines.append(f"  {command}")
    if notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {note}" for note in notes)
    return lines


def _split_action_command(step: str) -> tuple[str, str | None] | None:
    action_prefixes = {
        "Run this Codex MCP install command: ": "Install Codex MCP:",
        "Start or restart Codex through the activation launcher: ": "Start or restart Codex through the activation launcher:",
        "Start managed deterministic Codex capture: ": "Start managed deterministic Codex capture:",
    }
    for prefix, label in action_prefixes.items():
        if step.startswith(prefix):
            return label, step[len(prefix) :]
    return None


def _command_hints(label: str, command: str) -> list[str]:
    if label == "Install Codex MCP:":
        if " -- /" not in command or "zaxy" not in command:
            return []
        return [
            "Tip: this uses the resolved zaxy executable for MCP client reliability.",
            "Tip: use zaxy init --codex-mcp-install user when Codex has no conflicting zaxy entry.",
            "Tip: add --force only when you intentionally want to replace an existing zaxy MCP entry.",
        ]
    if label == "Start or restart Codex through the activation launcher:":
        hints: list[str] = []
        if "<task>" in command:
            hints.append("Tip: replace <task> with the work you are starting.")
        if "--eventloom-path" in command and "--workspace-root" in command:
            hints.append(
                "Tip: explicit --eventloom-path and --workspace-root values keep activation tied to this repo from any shell."
            )
        return hints
    return []


def _required_action_hints(label: str, notes: list[str]) -> list[str]:
    if not label.startswith("Review existing Codex MCP config before replacing zaxy"):
        return []
    hints: list[str] = []
    prefix = "If you intentionally want Zaxy to replace that entry, "
    for note in notes:
        if note.startswith(prefix):
            hints.append(f"Tip: {note[len(prefix) :]}")
    return hints


def _is_codex_mcp_conflict_review_action(step: str) -> bool:
    return step.startswith("Review existing Codex MCP config before replacing zaxy at ")


def _split_deferred_command(step: str) -> tuple[str, str | None] | None:
    deferred_prefixes = {
        "After Codex resume or update, emit the resume boundary: ": "After Codex resume or update, emit the resume boundary:",
    }
    for prefix, label in deferred_prefixes.items():
        if step.startswith(prefix):
            return label, step[len(prefix) :]
    return None


def _extract_leading_command(step: str) -> str | None:
    command_prefixes = ("Run ", "Smoke test recent memory: ", "Inspect model-facing memory bootstrap: ")
    for prefix in command_prefixes:
        if step.startswith(prefix):
            return step[len(prefix) :]
    return None


def _is_onboarding_note(step: str) -> bool:
    note_prefixes = (
        "Captured events can be replayed",
        "Data lives in ",
        "Default capture mode:",
        "Optional packet capture is disabled",
        "Point OpenAI-compatible clients at ",
        "Packet capture ",
        "Codex MCP config installed at ",
        "Managed deterministic Codex capture is already running.",
        "Managed deterministic Codex capture starts through the activation launcher.",
        "If you intentionally want Zaxy to replace that entry, ",
    )
    return step.startswith(note_prefixes)


def _preflight_outputs(*, force: bool, paths: list[str | Path]) -> None:
    if force:
        return
    for path in paths:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"{target} already exists; pass --force to overwrite")


def _validate_render_requests(
    *,
    mcp_client: str | None,
    mcp_output: str | Path | None,
    hook_client: str | None,
    hook_output: str | Path | None,
) -> None:
    if mcp_output is not None and mcp_client is None:
        raise ValueError("mcp_client is required when mcp_output is provided")
    if hook_output is not None and hook_client is None:
        raise ValueError("hook_client is required when hook_output is provided")


def _normalize_infra(infra: str) -> str:
    normalized = infra.casefold().strip()
    if normalized in {"none", "check", "start"}:
        return normalized
    raise ValueError("infra must be one of: none, check, start")


def _normalize_capture_mode(capture_mode: str) -> str:
    normalized = capture_mode.casefold().strip().replace("_", "-")
    if normalized in {"deterministic", "packet", "hybrid"}:
        return normalized
    raise ValueError("capture_mode must be one of: deterministic, packet, hybrid")


def _normalize_capture_action(capture_action: str) -> str:
    normalized = capture_action.casefold().strip().replace("_", "-")
    if normalized in {"none", "start"}:
        return normalized
    raise ValueError("capture action must be one of: none, start")


def _normalize_codex_mcp_install(mode: str) -> str:
    normalized = mode.casefold().strip().replace("_", "-")
    if normalized in {"command", "user", "project"}:
        return normalized
    raise ValueError("codex_mcp_install must be one of: command, user, project")


def _normalize_mcp_client_name(client: str) -> str:
    return client.casefold().strip().replace("_", "-")


def _normalize_hook_client_name(client: str) -> str:
    return client.casefold().strip().replace("_", "-")


def _build_runtime(settings: Settings) -> LocalNeo4jRuntime | LocalPgGraphRuntime | LocalEmbeddedGraphRuntime:
    backend = settings.projection_backend.casefold().strip()
    if backend == "pggraph":
        return LocalPgGraphRuntime(
            dsn=settings.pggraph_dsn,
            enabled=settings.pggraph_auto_start and settings.zaxy_env.lower() != "production",
            image=settings.pggraph_auto_start_image,
            container_name=settings.pggraph_auto_start_container,
            pggraph_repo=settings.pggraph_repo,
        )
    if backend == "embedded":
        return LocalEmbeddedGraphRuntime(path=settings.embedded_graph_path)
    return LocalNeo4jRuntime(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        enabled=settings.neo4j_auto_start and settings.zaxy_env.lower() != "production",
        image=settings.neo4j_auto_start_image,
        container_name=settings.neo4j_auto_start_container,
    )


def _run_infra_action(runtime: Any, infra: str) -> OnboardingStep:
    if infra == "check":
        check = runtime.check()
        return OnboardingStep("infra", str(_runtime_field(check, "status")), str(_runtime_field(check, "message")))
    runtime.ensure_available()
    display_name = getattr(runtime, "display_name", "selected")
    return OnboardingStep("infra", "ok", f"{display_name} local runtime is available")


def _runtime_field(check: Any, field: str) -> Any:
    if isinstance(check, dict):
        return check[field]
    return getattr(check, field)


def _build_next_steps(
    *,
    workspace: Path,
    eventloom: Path,
    session_id: str,
    mcp_client: str | None,
    mcp_output: str | Path | None,
    mcp_install_command: str | None,
    mcp_installed_path: Path | None,
    codex_mcp_conflict_path: Path | None,
    infra_action: str,
    projection_backend: str,
    capture_mode: str,
    packet_capture: bool,
    packet_upstream_base_url: str,
    packet_port: int,
    steps: list[OnboardingStep],
) -> list[str]:
    next_steps: list[str] = []
    if mcp_client is not None and mcp_output is not None:
        next_steps.append(f"Add {Path(mcp_output)} to your {mcp_client} MCP client config.")
        next_steps.append("Restart the MCP client so it loads the Zaxy server config.")
    if mcp_install_command is not None:
        next_steps.append(f"Run this Codex MCP install command: {mcp_install_command}")
    if mcp_installed_path is not None:
        next_steps.append(f"Codex MCP config installed at {mcp_installed_path}")
    if codex_mcp_conflict_path is not None:
        next_steps.append(f"Review existing Codex MCP config before replacing zaxy at {codex_mcp_conflict_path}.")
        next_steps.append(
            "If you intentionally want Zaxy to replace that entry, rerun with "
            "--codex-mcp-install user --force after reviewing it."
        )
    if mcp_install_command is not None or mcp_installed_path is not None:
        next_steps.append(
            "Start or restart Codex through the activation launcher: "
            + _activation_command(eventloom=eventloom, session_id=session_id, workspace=workspace)
        )
        next_steps.append(
            "After Codex resume or update, emit the resume boundary: "
            f"zaxy hook-event resume --eventloom-path {eventloom} --session-id {session_id} "
            '--source codex --summary "<task>"'
        )
        next_steps.append(
            "If Zaxy MCP tools are absent, use the CLI checkout fallback before substantial work: "
            f"zaxy memory checkout \"<task>\" --eventloom-path {eventloom} --session-id {session_id}"
        )
        capture_started = any(
            step.name == "capture_runtime" and step.status == "ok"
            for step in steps
        )
        if capture_started:
            next_steps.append("Managed deterministic Codex capture is already running.")
        else:
            next_steps.append("Managed deterministic Codex capture starts through the activation launcher.")
        backend = projection_backend.casefold().strip()
        if backend == "neo4j":
            next_steps.append(
                "For live Neo4j projections, start graph-enabled capture separately: "
                f"zaxy capture start --workspace {workspace} --graph"
            )
        elif backend == "embedded":
            next_steps.append("Captured events can be replayed into the repo-local embedded projection.")
    next_steps.append(f"Data lives in {eventloom}; each session is an append-only JSONL log.")
    next_steps.append(f"Run zaxy hook-status --eventloom-path {eventloom}")
    next_steps.append(
        "Smoke test recent memory: "
        f"zaxy memory log --eventloom-path {eventloom} --session-id {session_id} --limit 5"
    )
    next_steps.append(
        "Inspect model-facing memory bootstrap: "
        f"zaxy memory bootstrap --eventloom-path {eventloom} --session-id {session_id}"
    )
    if capture_mode == "deterministic":
        next_steps.append("Default capture mode: deterministic MCP lifecycle and observer hooks; no provider proxy required.")
        next_steps.append("Optional packet capture is disabled by default because it can consume provider quota.")
    if capture_mode in {"packet", "hybrid"}:
        next_steps.extend(
            build_packet_capture_guidance(
                eventloom_path=eventloom,
                session_id=session_id,
                upstream_base_url=packet_upstream_base_url,
                port=packet_port,
            ).next_steps()
        )
    infra_step = next((step for step in steps if step.name == "infra"), None)
    if infra_action == "check" and infra_step is not None and infra_step.status != "ok":
        backend = projection_backend.casefold().strip()
        if backend == "pggraph":
            next_steps.append(
                f"Run zaxy init {workspace} --projection-backend pggraph --infra start "
                "if you want Zaxy to start local pgGraph/PostgreSQL now."
            )
        else:
            next_steps.append(f"Run zaxy init {workspace} --infra start if you want Zaxy to start local Neo4j now.")
    return next_steps


AGENT_ACTIVATION_BEGIN = "<!-- zaxy-memory-activation:start -->"
AGENT_ACTIVATION_END = "<!-- zaxy-memory-activation:end -->"


def write_agent_activation_instructions(
    workspace: str | Path,
    *,
    eventloom_path: str | Path,
    session_id: str,
) -> Path:
    """Install a bounded model-visible activation block into AGENTS.md."""
    root = Path(workspace)
    target = root / "AGENTS.md"
    block = _render_agent_activation_block(
        eventloom_path=Path(eventloom_path),
        session_id=session_id,
        workspace=root,
    )
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    if AGENT_ACTIVATION_BEGIN in existing and AGENT_ACTIVATION_END in existing:
        before, rest = existing.split(AGENT_ACTIVATION_BEGIN, 1)
        _, after = rest.split(AGENT_ACTIVATION_END, 1)
        content = before.rstrip() + "\n\n" + block + after
    elif existing.strip():
        content = existing.rstrip() + "\n\n" + block + "\n"
    else:
        content = "# Agent Instructions\n\n" + block + "\n"
    target.write_text(content, encoding="utf-8")
    return target


def _render_agent_activation_block(*, eventloom_path: Path, session_id: str, workspace: Path) -> str:
    eventloom = str(eventloom_path)
    return "\n".join(
        [
            AGENT_ACTIVATION_BEGIN,
            "## Zaxy Memory Activation",
            "",
            "Before roadmap, implementation, release, review, resume, or high-context work:",
            "- Start Codex through Zaxy when opening a new session: "
            f"`{_activation_command(eventloom=eventloom_path, session_id=session_id, workspace=workspace)}`.",
            f"- After `/resume`, Codex update, or MCP/tool reload, record the boundary: `zaxy hook-event resume --eventloom-path {eventloom} --session-id {session_id} --source codex --summary \"<task>\"`.",
            f"- If Zaxy MCP tools are unavailable, run CLI checkout before substantial work: `zaxy memory checkout \"<task>\" --eventloom-path {eventloom} --session-id {session_id}`.",
            "- If no fresh activation packet or cited checkout is available, treat memory as degraded and pause substantial work until checkout succeeds.",
            "- Do not rely only on ordinary Codex summaries when Zaxy activation is missing.",
            AGENT_ACTIVATION_END,
        ]
    )


def _activation_command(*, eventloom: Path, session_id: str, workspace: Path) -> str:
    return shlex.join(
        [
            "zaxy",
            "activate",
            "codex",
            "--eventloom-path",
            str(eventloom),
            "--session-id",
            session_id,
            "--current-task",
            "<task>",
            "--workspace-root",
            str(workspace),
            "--launch",
        ]
    )


def _build_capture_summary(*, workspace: Path, doctor: dict[str, Any]) -> dict[str, Any]:
    try:
        runtime = inspect_codex_capture(workspace=workspace)
    except (OSError, ValueError) as exc:
        runtime = {
            "configured": False,
            "running": False,
            "pids": [],
            "latest_observation": None,
            "error": str(exc),
        }
    health = _doctor_check(doctor, "capture_health")
    summary = {
        "configured": bool(runtime.get("configured", False)),
        "running": bool(runtime.get("running", False)),
        "pids": list(runtime.get("pids", [])),
        "latest_observation": runtime.get("latest_observation"),
        "doctor_status": health.get("status", "unknown") if health else "unknown",
        "doctor_message": health.get("message", "") if health else "",
    }
    if runtime.get("error"):
        summary["error"] = runtime["error"]
    return summary


def _doctor_check(doctor: dict[str, Any], name: str) -> dict[str, Any] | None:
    for check in doctor.get("checks", []):
        if isinstance(check, dict) and check.get("name") == name:
            return check
    return None


def _format_capture_summary(capture: dict[str, Any]) -> list[str]:
    configured = "configured" if capture.get("configured") else "not configured"
    running = "running" if capture.get("running") else "not running"
    lines = [f"capture: {configured}, {running}"]
    pids = capture.get("pids") or []
    if pids:
        lines.append("capture pids: " + ", ".join(str(pid) for pid in pids))
    latest = capture.get("latest_observation")
    if latest:
        lines.append(
            f"latest capture: {latest['type']} seq={latest['seq']} "
            f"session={latest['thread']} source={latest['source']}"
        )
    if capture.get("doctor_status"):
        doctor_message = str(capture.get("doctor_message", ""))
        if (
            capture.get("configured")
            and not capture.get("running")
            and capture.get("doctor_status") == "warning"
            and doctor_message == "Codex capture is configured, but the managed watcher is not running"
        ):
            lines.append(
                "capture next: start Codex through the activation launcher when you want live local capture"
            )
        else:
            lines.append(f"capture health: {capture['doctor_status']} - {doctor_message}")
    if capture.get("error"):
        lines.append(f"capture error: {capture['error']}")
    return lines


def _onboarding_settings(
    *,
    eventloom: Path,
    session_id: str,
    domain: str,
    projection_backend: str | None = None,
    pggraph_dsn: str | None = None,
    pggraph_repo: str | Path | None = None,
) -> Settings:
    settings_values: dict[str, Any] = {
        "_env_file": None,
        "eventloom_path": str(eventloom),
        "eventloom_thread": session_id,
        "zaxy_domain": domain,
        "zaxy_env": "development",
        "mcp_lifecycle_capture_enabled": True,
    }
    if projection_backend is not None:
        settings_values["projection_backend"] = projection_backend
        if projection_backend.casefold().strip() == "embedded":
            settings_values["embedded_graph_path"] = str(eventloom / "projections" / "embedded.kuzu")
    if pggraph_dsn is not None:
        settings_values["pggraph_dsn"] = pggraph_dsn
    if pggraph_repo is not None:
        settings_values["pggraph_repo"] = str(pggraph_repo)
    return Settings(**settings_values)


def _write_json(path: Path, payload: dict[str, Any], *, force: bool = False) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _append_heartbeat(eventloom_path: Path, *, session_id: str, source: str, workspace: Path) -> Any:
    event_type = hook_event_type("heartbeat")
    payload = build_hook_payload(
        trigger="heartbeat",
        source=source,
        workspace=str(workspace),
    )
    eventlog = SessionManager(base_path=str(eventloom_path)).get(session_id).eventlog
    return eventlog.append(event_type, actor="zaxy-hook", payload=payload, thread=session_id)


def _onboarding_doctor_status(
    doctor: dict[str, Any],
    *,
    hook_installation_required: bool = True,
    agent_instructions_required: bool = True,
) -> str:
    ignored = _ignored_onboarding_doctor_checks(
        hook_installation_required=hook_installation_required,
        agent_instructions_required=agent_instructions_required,
    )
    actionable_statuses = [check["status"] for check in doctor["checks"] if check["name"] not in ignored]
    return _overall_status(actionable_statuses)


def _onboarding_doctor_message(
    doctor: dict[str, Any],
    *,
    hook_installation_required: bool = True,
    agent_instructions_required: bool = True,
) -> str:
    ignored = _ignored_onboarding_doctor_checks(
        hook_installation_required=hook_installation_required,
        agent_instructions_required=agent_instructions_required,
    )
    issues = [
        check
        for check in doctor.get("checks", [])
        if check.get("name") not in ignored and check.get("status") != "ok"
    ]
    if not issues:
        return "Doctor checks completed"
    rendered = [_format_doctor_issue(check) for check in issues[:3]]
    if len(issues) > 3:
        rendered.append(f"+{len(issues) - 3} more")
    return "; ".join(rendered)


def _format_doctor_issue(check: dict[str, Any]) -> str:
    name = str(check.get("name", "doctor"))
    status = str(check.get("status", "unknown"))
    message = str(check.get("message", "")).strip()
    summary = f"{name} {status}"
    if message:
        summary += f": {message}"
    action = str(check.get("action", "")).strip()
    if action:
        summary += f" (action: {action})"
    return summary


def _ignored_onboarding_doctor_checks(
    *,
    hook_installation_required: bool = True,
    agent_instructions_required: bool = True,
) -> set[str]:
    ignored = {
        "codex_mcp_scope",
        "observation_coverage",
        "capture_health",
        "memory_activation",
        "packet_memory",
        # A freshly initialized workspace has genesis events but no projection
        # state yet; lazy projection catches up on first memory use.
        "projection_freshness",
    }
    if not hook_installation_required:
        ignored.add("hook_installation")
    if not agent_instructions_required:
        ignored.add("agent_instructions")
    return ignored


def _onboarding_hook_status(report: dict[str, Any], *, hook_client: str | None) -> str:
    status = str(report["status"])
    if status == "ok":
        return "ok"
    if hook_client is None:
        return status
    normalized_client = _normalize_hook_client_name(hook_client)
    client = report.get("clients", {}).get(normalized_client, {})
    latest_event = report.get("latest_event") or {}
    if client.get("installed") and latest_event.get("type") == hook_event_type("heartbeat"):
        return "ok"
    return status


def _overall_status(statuses: Any) -> str:
    status_set = set(statuses)
    if "error" in status_set:
        return "error"
    if "warning" in status_set:
        return "warning"
    return "ok"
