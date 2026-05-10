"""First-run onboarding orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from zaxy.integrations import render_mcp_client_config
from zaxy.local_profile import write_local_profile
from zaxy.runtime import LocalNeo4jRuntime
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
) -> dict[str, Any]:
    """Expand a named onboarding preset without overriding explicit options."""
    if preset is None:
        return {
            "mcp_client": mcp_client,
            "mcp_output": mcp_output,
            "hook_client": hook_client,
            "hook_output": hook_output,
            "local_profile_output": local_profile_output,
            "infra": infra,
        }
    normalized = preset.casefold().strip().replace("_", "-")
    if normalized != "local-claude":
        raise ValueError("preset must be one of: local-claude")
    root = Path(workspace)
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
    executable = resolve_zaxy_executable(zaxy_executable)
    _validate_render_requests(
        mcp_client=mcp_client,
        mcp_output=mcp_output,
        hook_client=hook_client,
        hook_output=hook_output,
    )
    _preflight_outputs(
        force=force,
        paths=[path for path in (mcp_output, hook_output, local_profile_output) if path is not None],
    )

    steps: list[OnboardingStep] = []
    eventloom.mkdir(parents=True, exist_ok=True)
    steps.append(OnboardingStep("eventloom", "ok", "Eventloom directory is ready", str(eventloom)))

    if local_profile_output is not None:
        written = write_local_profile(Path(local_profile_output), force=force)
        steps.append(OnboardingStep("local_profile", "ok", "Local retrieval profile written", str(written)))

    if mcp_client is not None:
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

    if infra_action != "none":
        settings = _onboarding_settings(eventloom=eventloom, session_id=sid, domain=resolved_domain)
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

    settings = _onboarding_settings(eventloom=eventloom, session_id=sid, domain=resolved_domain)
    doctor = run_doctor(settings=settings, workspace_root=root, zaxy_executable=executable)
    steps.append(OnboardingStep("doctor", _onboarding_doctor_status(doctor), "Doctor checks completed"))
    hook_status = inspect_hook_status(eventloom_path=eventloom, workspace_root=root)
    steps.append(OnboardingStep("hook_status", hook_status["status"], hook_status["message"]))
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
            mcp_client=mcp_client,
            mcp_output=mcp_output,
            infra_action=infra_action,
            steps=steps,
        ),
        doctor=doctor,
        hook_status=hook_status,
    )


def format_onboarding_result(result: OnboardingResult) -> str:
    """Format onboarding output for humans."""
    lines = [
        f"Zaxy init: {result.status}",
        f"workspace: {result.workspace}",
        f"domain: {result.domain}",
        f"session: {result.session_id}",
        f"profile: {result.profile['workspace_type']} ({result.profile['confidence']})",
    ]
    for step in result.steps:
        suffix = f" - {step.path}" if step.path else ""
        lines.append(f"- {step.name}: {step.status} - {step.message}{suffix}")
    if result.next_steps:
        lines.append("")
        lines.append("Next:")
        lines.extend(f"- {step}" for step in result.next_steps)
    return "\n".join(lines)


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


def _build_runtime(settings: Settings) -> LocalNeo4jRuntime:
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
    return OnboardingStep("infra", "ok", "Neo4j local runtime is available")


def _runtime_field(check: Any, field: str) -> Any:
    if isinstance(check, dict):
        return check[field]
    return getattr(check, field)


def _build_next_steps(
    *,
    workspace: Path,
    eventloom: Path,
    mcp_client: str | None,
    mcp_output: str | Path | None,
    infra_action: str,
    steps: list[OnboardingStep],
) -> list[str]:
    next_steps: list[str] = []
    if mcp_client is not None and mcp_output is not None:
        next_steps.append(f"Add {Path(mcp_output)} to your {mcp_client} MCP client config.")
        next_steps.append("Restart the MCP client so it loads the Zaxy server config.")
    next_steps.append(f"Run zaxy hook-status --eventloom-path {eventloom}")
    infra_step = next((step for step in steps if step.name == "infra"), None)
    if infra_action == "check" and infra_step is not None and infra_step.status != "ok":
        next_steps.append(f"Run zaxy init {workspace} --infra start if you want Zaxy to start local Neo4j now.")
    return next_steps


def _onboarding_settings(*, eventloom: Path, session_id: str, domain: str) -> Settings:
    settings_values: dict[str, Any] = {
        "_env_file": None,
        "eventloom_path": str(eventloom),
        "eventloom_thread": session_id,
        "zaxy_domain": domain,
        "zaxy_env": "development",
        "mcp_lifecycle_capture_enabled": True,
    }
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


def _onboarding_doctor_status(doctor: dict[str, Any]) -> str:
    actionable_statuses = [
        check["status"]
        for check in doctor["checks"]
        if check["name"] != "observation_coverage"
    ]
    return _overall_status(actionable_statuses)


def _overall_status(statuses: Any) -> str:
    status_set = set(statuses)
    if "error" in status_set:
        return "error"
    if "warning" in status_set:
        return "warning"
    return "ok"
