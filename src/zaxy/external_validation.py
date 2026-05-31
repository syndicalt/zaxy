"""Validation helpers for v1.0 external-validation reports."""

from __future__ import annotations

import re
from datetime import date
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

EXTERNAL_VALIDATION_CONTRACT = "zaxy.v1.external-validation-report"
EXTERNAL_VALIDATION_PATHS = {"first_run_local", "coordinate_workflow", "clean_repo_uat", "other_documented"}
EXTERNAL_VALIDATION_RELEASE_DECISIONS = {"pass", "pass_with_follow_up"}
VALIDATION_PATH_COMMAND_MARKERS = {
    "first_run_local": (
        "zaxy init",
        "zaxy memory bootstrap",
        "zaxy memory checkout",
        "zaxy doctor --beta-readiness",
    ),
    "coordinate_workflow": ("examples/coordinate_three_worker_project.py",),
    "clean_repo_uat": ("scripts/beta-uat.sh",),
}


def validate_external_validation_report(payload: Any) -> list[str]:
    """Return validation errors for a v1.0 external-validation report."""
    if not isinstance(payload, dict):
        return ["report must be a JSON object"]

    errors: list[str] = []
    _require_equal(payload, "contract", EXTERNAL_VALIDATION_CONTRACT, errors)
    _require_equal(payload, "status", "validated", errors)

    validator = payload.get("validator")
    if not isinstance(validator, dict):
        errors.append("validator must be an object")
    else:
        validator_name = validator.get("name")
        if not _non_empty_string(validator_name):
            errors.append("validator.name must be a non-empty string")
        elif _is_placeholder(validator_name):
            errors.append("validator.name must not be a placeholder")
        elif _is_implementation_session_name(validator_name):
            errors.append("validator.name must identify an external validator")
        if validator.get("external_to_implementation_session") is not True:
            errors.append("validator.external_to_implementation_session must be true")

    environment = payload.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment must be an object")
    else:
        for field in ("operating_system", "shell", "python_version", "install_source"):
            value = environment.get(field)
            if not _non_empty_string(value):
                errors.append(f"environment.{field} must be a non-empty string")
            elif _is_placeholder(value):
                errors.append(f"environment.{field} must not be a placeholder")
        python_version = environment.get("python_version")
        if (
            _non_empty_string(python_version)
            and not _is_placeholder(python_version)
            and not _is_concrete_python_version(python_version)
        ):
            errors.append("environment.python_version must be a concrete Python version")
        install_source = environment.get("install_source")
        if (
            _non_empty_string(install_source)
            and not _is_placeholder(install_source)
            and _is_vague_install_source(install_source)
        ):
            errors.append("environment.install_source must be concrete")
        shell = environment.get("shell")
        if _non_empty_string(shell) and not _is_placeholder(shell) and _is_vague_shell(shell):
            errors.append("environment.shell must be concrete")

    report_date = _parse_date(payload.get("date"))
    if report_date is None:
        errors.append("date must be an ISO date string")
    elif report_date > date.today():
        errors.append("date must not be in the future")
    zaxy_version_or_commit = payload.get("zaxy_version_or_commit")
    if not _non_empty_string(zaxy_version_or_commit):
        errors.append("zaxy_version_or_commit must be a non-empty string")
    elif _is_placeholder(zaxy_version_or_commit):
        errors.append("zaxy_version_or_commit must not be a placeholder")
    elif _is_vague_version_reference(zaxy_version_or_commit):
        errors.append("zaxy_version_or_commit must be a concrete version or commit")
    if payload.get("validation_path") not in EXTERNAL_VALIDATION_PATHS:
        errors.append("validation_path must be one of: " + ", ".join(sorted(EXTERNAL_VALIDATION_PATHS)))
    commands = payload.get("commands")
    if not _non_empty_string_list(commands):
        errors.append("commands must be a non-empty list of command strings")
    elif isinstance(commands, list) and any(_is_placeholder(command) for command in commands):
        errors.append("commands must not contain placeholder values")
    elif isinstance(commands, list) and any(_is_multiline_command(command) for command in commands):
        errors.append("commands must be single-line strings")
    elif isinstance(commands, list) and any(_is_echoed_command_text(command) for command in commands):
        errors.append("commands must record executed commands, not echoed command text")
    elif isinstance(commands, list) and any(_has_shell_control_operator(command) for command in commands):
        errors.append("commands must not contain shell control operators")
    elif isinstance(commands, list):
        _validate_commands_for_path(payload.get("validation_path"), commands, errors)
    if not _positive_number(payload.get("time_to_first_useful_checkout_seconds")):
        errors.append("time_to_first_useful_checkout_seconds must be a positive number")
    if payload.get("unexpected_sidecar_or_credential_required") is not False:
        errors.append("unexpected_sidecar_or_credential_required must be false")
    evidence_links = payload.get("evidence_links")
    if not _non_empty_string_list(evidence_links):
        errors.append("evidence_links must include at least one report, issue, discussion, or case-study link")
    elif isinstance(evidence_links, list) and any(_is_placeholder(link) for link in evidence_links):
        errors.append("evidence_links must not contain placeholder values")
    elif isinstance(evidence_links, list) and not all(_is_absolute_web_url(link) for link in evidence_links):
        errors.append("evidence_links must contain absolute http or https URLs")
    elif isinstance(evidence_links, list) and any(_has_url_credentials(link) for link in evidence_links):
        errors.append("evidence_links must not include credentials")
    elif isinstance(evidence_links, list) and any(_has_bare_origin_url(link) for link in evidence_links):
        errors.append("evidence_links must include a concrete artifact path")
    elif isinstance(evidence_links, list) and any(_is_local_only_url(link) for link in evidence_links):
        errors.append("evidence_links must not use local-only URLs")
    elif isinstance(evidence_links, list) and any(_is_private_network_url(link) for link in evidence_links):
        errors.append("evidence_links must not use private-network URLs")
    elif isinstance(evidence_links, list) and any(_is_internal_only_domain_url(link) for link in evidence_links):
        errors.append("evidence_links must not use internal-only domains")
    elif isinstance(evidence_links, list) and any(_has_single_label_hostname(link) for link in evidence_links):
        errors.append("evidence_links must use fully qualified public hostnames")
    elif isinstance(evidence_links, list) and any(_is_invalid_github_artifact_url(link) for link in evidence_links):
        errors.append("evidence_links must point to a reviewable evidence artifact")
    elif isinstance(evidence_links, list) and any(_is_example_domain_url(link) for link in evidence_links):
        errors.append("evidence_links must not use example domains")
    friction_or_failure = payload.get("friction_or_failure")
    if not _non_empty_string(friction_or_failure):
        errors.append("friction_or_failure must be a non-empty string")
    elif _is_placeholder(friction_or_failure):
        errors.append("friction_or_failure must not be a placeholder")
    if payload.get("release_decision") not in EXTERNAL_VALIDATION_RELEASE_DECISIONS:
        errors.append("release_decision must be pass or pass_with_follow_up")
    if payload.get("supports_positioning") is not True:
        errors.append("supports_positioning must be true")

    return errors


def _require_equal(payload: dict[str, Any], field: str, expected: str, errors: list[str]) -> None:
    if payload.get(field) != expected:
        errors.append(f"{field} must be {expected}")


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return (
        "replace" in normalized
        or "placeholder" in normalized
        or normalized in {"todo", "to do", "tbd", "n/a", "na", "none", "not applicable"}
        or normalized.startswith("todo ")
        or normalized.startswith("to do ")
        or normalized.startswith("tbd ")
        or normalized.startswith("example ")
        or normalized.startswith("sample ")
        or normalized == "external validator name or project"
        or normalized.startswith("external validator name")
    )


def _is_implementation_session_name(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    implementation_names = (
        "codex",
        "codex agent",
        "implementation session",
        "current implementation session",
        "zaxy implementation session",
        "codex implementation session",
        "implementation agent",
        "current agent",
        "this agent",
        "this implementation session",
    )
    return normalized in implementation_names


def _is_absolute_web_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _has_url_credentials(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.username is not None or parsed.password is not None


def _has_bare_origin_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and parsed.path.strip("/") == ""


def _has_single_label_hostname(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    hostname = urlparse(value.strip()).hostname
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    try:
        ip_address(normalized)
    except ValueError:
        return "." not in normalized
    return False


def _is_local_only_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    hostname = urlparse(value.strip()).hostname
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized in {"localhost", "0.0.0.0"} or normalized.endswith(".localhost"):
        return True
    try:
        address = ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_link_local or address.is_unspecified


def _is_private_network_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    hostname = urlparse(value.strip()).hostname
    if hostname is None:
        return False
    try:
        address = ip_address(hostname)
    except ValueError:
        return False
    return address.is_private


def _is_internal_only_domain_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    hostname = urlparse(value.strip()).hostname
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized.endswith((".internal", ".local", ".lan", ".test", ".invalid"))


def _is_example_domain_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    hostname = urlparse(value.strip()).hostname
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    return normalized in {"example.com", "example.net", "example.org"} or normalized.endswith(".example")


def _is_invalid_github_artifact_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    hostname = parsed.hostname
    if hostname is None:
        return False
    normalized_hostname = hostname.rstrip(".").lower()
    if normalized_hostname in {"github.com", "raw.githubusercontent.com"} and (parsed.query or parsed.fragment):
        return True
    if normalized_hostname in {"github.com", "raw.githubusercontent.com"} and parsed.path.endswith("/"):
        return True
    if normalized_hostname in {"github.com", "raw.githubusercontent.com"} and "//" in parsed.path.lstrip("/"):
        return True
    if normalized_hostname == "raw.githubusercontent.com":
        path_parts = [part for part in parsed.path.split("/") if part]
        return len(path_parts) < 4 or not _is_git_sha(path_parts[2])
    if normalized_hostname != "github.com":
        return False
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) == 2:
        return True
    artifact_segment = path_parts[2] if len(path_parts) >= 3 else ""
    if artifact_segment != artifact_segment.lower():
        return True
    collection_paths = {"issues", "discussions", "releases", "pulls", "actions", "wiki"}
    if len(path_parts) == 3 and artifact_segment in collection_paths:
        return True
    numbered_artifact_paths = {"issues", "discussions", "pull"}
    if len(path_parts) >= 4 and artifact_segment in numbered_artifact_paths:
        return len(path_parts) != 4 or not _is_positive_int_string(path_parts[3])
    if len(path_parts) >= 4 and artifact_segment == "pulls":
        return True
    if len(path_parts) >= 4 and artifact_segment == "releases" and path_parts[3] != "tag":
        return True
    if len(path_parts) >= 4 and artifact_segment == "releases" and path_parts[3] == "tag":
        return len(path_parts) != 5 or _is_vague_version_reference(path_parts[4])
    if len(path_parts) >= 4 and artifact_segment == "actions" and path_parts[3] != "runs":
        return True
    if len(path_parts) >= 4 and artifact_segment == "actions" and path_parts[3] == "runs":
        return len(path_parts) != 5 or not _is_positive_int_string(path_parts[4])
    if len(path_parts) >= 4 and artifact_segment == "commit":
        return len(path_parts) != 4 or not _is_git_sha(path_parts[3])
    if len(path_parts) >= 4 and artifact_segment in {"blob", "raw", "tree"}:
        return len(path_parts) < 5 or not _is_git_sha(path_parts[3])
    return len(path_parts) >= 3


def _is_git_sha(value: str) -> bool:
    return re.fullmatch(r"[0-9a-fA-F]{40}", value) is not None


def _is_positive_int_string(value: str) -> bool:
    return value.isdigit() and value == str(int(value)) and int(value) > 0


def _is_vague_version_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    vague_references = {"latest", "current", "main", "master", "head", "stable"}
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    return bool(tokens & vague_references)


def _is_concrete_python_version(value: Any) -> bool:
    return isinstance(value, str) and re.search(r"\b\d+\.\d+(?:\.\d+)?\b", value) is not None


def _is_vague_install_source(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if normalized in {"package manager", "pip", "pipx", "uv", "pypi", "github", "source"}:
        return True
    moving_reference_tokens = {"latest", "current", "head", "stable", "main", "master"}
    tokens = {token for token in re.split(r"[^a-z0-9]+", normalized) if token}
    return bool(tokens & moving_reference_tokens)


def _is_vague_shell(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return normalized in {"terminal", "console", "command line", "cli", "shell"}


def _validate_commands_for_path(validation_path: Any, commands: list[Any], errors: list[str]) -> None:
    if not isinstance(validation_path, str):
        return
    if validation_path == "other_documented":
        if not any(_is_substantive_validation_command(command) for command in commands):
            errors.append("commands must include at least one substantive Zaxy validation command")
        return
    required_markers = VALIDATION_PATH_COMMAND_MARKERS.get(validation_path)
    if required_markers is None:
        return
    for marker in required_markers:
        if not any(_command_starts_with_marker(command, marker) for command in commands):
            errors.append(f"commands must include {marker} for {validation_path} validation")


def _has_shell_control_operator(command: Any) -> bool:
    if not isinstance(command, str):
        return False
    if any(operator in command for operator in ("&", "||", ";", "|", "`", "$(", ">", "<", "#")):
        return True
    return re.search(r"\s\(|\)\s*(?:$|[;&|])", command) is not None


def _is_multiline_command(command: Any) -> bool:
    return isinstance(command, str) and any(separator in command for separator in ("\n", "\r"))


def _is_echoed_command_text(command: Any) -> bool:
    return isinstance(command, str) and re.match(r"\s*(?:echo|printf)\b", command, flags=re.IGNORECASE) is not None


def _command_starts_with_marker(command: Any, marker: str) -> bool:
    if not isinstance(command, str):
        return False
    normalized = command.strip().lower()
    if _is_help_or_version_command(normalized):
        return False
    for prefix in ("python -m ", "python3 -m ", "uv run ", "poetry run ", "pipx run "):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix).lstrip()
            break
    allowed_starts = [marker]
    if marker.endswith(".py"):
        allowed_starts.extend((f"python {marker}", f"python3 {marker}"))
    elif marker.endswith(".sh"):
        allowed_starts.extend((f"bash {marker}", f"sh {marker}"))
    return any(normalized == start or normalized.startswith(start + " ") for start in allowed_starts)


def _is_help_or_version_command(command: str) -> bool:
    parts = command.split()
    return any(part in {"--help", "-h", "help", "--version", "-v", "version"} for part in parts)


def _is_substantive_validation_command(command: Any) -> bool:
    if not isinstance(command, str):
        return False
    normalized = command.strip().lower()
    if _is_help_or_version_command(normalized):
        return False
    substantive_prefixes = (
        "zaxy init",
        "zaxy memory ",
        "zaxy doctor ",
        "zaxy coordinate",
        "zaxy serve",
        "zaxy replay",
        "python examples/",
        "python3 examples/",
        "python scripts/",
        "python3 scripts/",
        "bash scripts/",
        "sh scripts/",
        "scripts/",
    )
    return any(normalized.startswith(prefix) for prefix in substantive_prefixes)


def _non_empty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_non_empty_string(item) for item in value)


def _positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
