"""Read-only git metadata capture for coordination evidence."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

MAX_CHANGED_FILES = 200
MAX_WORKTREES = 50
MAX_DIFF_SUMMARY_BYTES = 4000

#: Wall-clock ceiling for each read-only git invocation. Capture runs on the
#: interactive `coordinate report` path, so a wedged git (a stale index lock, a
#: network-backed filesystem) must fail fast rather than hang the CLI forever.
GIT_COMMAND_TIMEOUT_SECONDS = 10.0

#: Suffix appended to a diff summary that hit MAX_DIFF_SUMMARY_BYTES, so a
#: consumer can tell a bounded excerpt from a genuinely short diff.
DIFF_TRUNCATION_MARKER = "\n[truncated]"


class GitCaptureError(RuntimeError):
    """Raised when read-only git metadata capture cannot complete."""


def capture_git_metadata(
    workspace: str | Path,
    *,
    test_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return branch, worktree, changed files, and diff summary for a git workspace.

    Raises:
        GitCaptureError: If git is unavailable, times out, or the workspace is
            not a git repository.
    """
    root = _git(workspace, "rev-parse", "--show-toplevel")
    worktree = Path(root).resolve()
    raw_branch = _git(worktree, "rev-parse", "--abbrev-ref", "HEAD")
    detached = raw_branch == "HEAD"
    branch = raw_branch if raw_branch else "HEAD"
    head = _git(worktree, "rev-parse", "HEAD")
    changed_files = _changed_files(worktree)
    diff_summary, diff_truncated = _bounded_text(_git(worktree, "diff", "--stat", "--summary"))
    worktrees = _worktrees(worktree)
    return {
        "kind": "git",
        "reference": branch,
        "repo_root": str(worktree),
        "worktree": str(worktree),
        "branch": branch,
        "head": head,
        "detached": detached,
        "dirty": bool(changed_files),
        "changed_files": changed_files,
        "diff_summary": diff_summary,
        "diff_summary_truncated": diff_truncated,
        "worktrees": worktrees,
        "test_results": test_results or [],
    }


def build_test_result_evidence(
    command: str,
    *,
    status: str,
    summary: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Return structured test-result evidence for a coordination finding."""
    payload: dict[str, Any] = {
        "kind": "test_result",
        "reference": command,
        "command": command,
        "status": status,
    }
    if summary:
        payload["summary"] = summary
    if exit_code is not None:
        payload["exit_code"] = exit_code
    return payload


def _changed_files(workspace: str | Path) -> list[dict[str, Any]]:
    raw = _git(workspace, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    files: list[dict[str, str]] = []
    entries = [entry for entry in raw.split("\0") if entry]
    index = 0
    while index < len(entries):
        entry = entries[index]
        if len(entry) < 3:
            index += 1
            continue
        status = entry[:2].strip()
        path = entry[3:].strip()
        operation = _git_status_operation(status)
        if "R" in status or "C" in status:
            index += 1
            if index < len(entries):
                path = entries[index]
        files.append({"path": path, "status": status, "operation": operation})
        index += 1
    return files[:MAX_CHANGED_FILES]


def _worktrees(workspace: str | Path) -> list[dict[str, Any]]:
    raw = _git(workspace, "worktree", "list", "--porcelain")
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in raw.splitlines():
        if not line:
            if current:
                entries.append(_normalize_worktree(current))
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree" and current:
            entries.append(_normalize_worktree(current))
            current = {}
        current[key] = value
    if current:
        entries.append(_normalize_worktree(current))
    return entries[:MAX_WORKTREES]


def _normalize_worktree(value: dict[str, Any]) -> dict[str, Any]:
    branch = str(value.get("branch") or "")
    if branch.startswith("refs/heads/"):
        branch = branch.removeprefix("refs/heads/")
    detached = not branch
    return {
        "path": str(Path(str(value.get("worktree") or "")).resolve()),
        "branch": branch or "HEAD",
        "head": str(value.get("HEAD") or ""),
        "detached": detached,
    }


def _git_status_operation(status: str) -> str:
    if status == "??":
        return "untracked"
    if "R" in status:
        return "renamed"
    if "C" in status:
        return "copied"
    if "D" in status:
        return "deleted"
    if "A" in status:
        return "added"
    if "M" in status:
        return "modified"
    return "changed"


def _bounded_text(value: str) -> tuple[str, bool]:
    """Return the value bounded to MAX_DIFF_SUMMARY_BYTES plus whether it was cut."""
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_DIFF_SUMMARY_BYTES:
        return value, False
    # The marker is part of the budget, so a truncated summary still fits the
    # documented ceiling instead of overshooting it by the marker's length.
    budget = MAX_DIFF_SUMMARY_BYTES - len(DIFF_TRUNCATION_MARKER.encode("utf-8"))
    kept = encoded[:budget].decode("utf-8", errors="ignore")
    return kept + DIFF_TRUNCATION_MARKER, True


def _git(workspace: str | Path, *args: str) -> str:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    label = " ".join(args)
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", *args],
            cwd=Path(workspace),
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        # A missing cwd also surfaces as ENOENT here, so separate the two rather
        # than always blaming a missing git binary.
        if not Path(workspace).is_dir():
            raise GitCaptureError(f"git workspace {workspace} does not exist") from exc
        raise GitCaptureError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitCaptureError(
            f"git {label} timed out after {GIT_COMMAND_TIMEOUT_SECONDS:g}s in {workspace}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit status {exc.returncode}"
        raise GitCaptureError(f"git {label} failed in {workspace}: {reason}") from exc
    return completed.stdout.rstrip("\n")

