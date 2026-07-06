"""Machine-level MCP registration for supported agent harnesses.

Detects which agent harnesses are installed on this machine and registers the
Zaxy MCP server with each at **user scope** (global), so every project gets Zaxy
without per-project setup. Each write merges into the harness's own user config
and is idempotent (re-running replaces only Zaxy's own entry).

This is the engine behind ``zaxy install`` and the ``install.sh`` bootstrap.
Detection and every writer accept an explicit ``home`` so the behaviour is fully
unit-testable against a temporary directory.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from zaxy.install import resolve_zaxy_executable
from zaxy.integrations import (
    codex_mcp_config_path,
    hermes_mcp_config_path,
    write_codex_mcp_config,
    write_hermes_mcp_config,
)

SERVER_NAME = "zaxy"


@dataclass(frozen=True)
class HarnessSpec:
    """Detection metadata for one supported harness."""

    name: str
    display: str
    aliases: tuple[str, ...]
    binaries: tuple[str, ...]
    config_probes: tuple[str, ...]  # home-relative paths whose existence => installed
    note: str = ""


# Ordered by how common the harness is. Detection = a binary on PATH OR a
# home-relative config path exists.
HARNESSES: tuple[HarnessSpec, ...] = (
    HarnessSpec("claude-code", "Claude Code", ("claude-code",), ("claude",), (".claude.json", ".claude")),
    HarnessSpec("codex", "Codex", ("codex",), ("codex",), (".codex/config.toml", ".codex")),
    HarnessSpec("opencode", "opencode", ("opencode",), ("opencode",), (".config/opencode",)),
    HarnessSpec("openclaw", "OpenClaw", ("openclaw",), ("openclaw",), (".openclaw/openclaw.json", ".openclaw")),
    HarnessSpec("hermes", "Hermes", ("hermes",), ("hermes",), (".hermes/config.yaml", ".hermes")),
    HarnessSpec("zcode", "Z.ai ZCode", ("zcode", "zai"), ("zcode", "zai"), (".zcode",)),
    HarnessSpec(
        "pi",
        "Pi",
        ("pi",),
        ("pi",),
        (".pi",),
        note="native Pi has no MCP; configured only when the MCP adapter (~/.pi/agent) is present",
    ),
)

_BY_NAME: dict[str, HarnessSpec] = {}
for _spec in HARNESSES:
    _BY_NAME[_spec.name] = _spec
    for _alias in _spec.aliases:
        _BY_NAME[_alias] = _spec


def normalize_harness(name: str) -> str:
    spec = _BY_NAME.get(name.strip().casefold())
    if spec is None:
        raise ValueError(f"unknown harness: {name!r} (supported: {', '.join(s.name for s in HARNESSES)})")
    return spec.name


@dataclass(frozen=True)
class Registration:
    harness: str
    status: str  # "configured" | "skipped" | "error"
    method: str  # "file:<path>" | "reused:<writer>" | ""
    detail: str


def detect_harnesses(
    *,
    home: str | Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """Return the canonical names of harnesses installed on this machine."""
    root = Path(home).expanduser() if home is not None else Path.home()
    found: list[str] = []
    for spec in HARNESSES:
        on_path = any(which(binary) for binary in spec.binaries)
        has_config = any((root / probe).exists() for probe in spec.config_probes)
        if on_path or has_config:
            found.append(spec.name)
    return found


def _merge_json(path: Path, keys: tuple[str, ...], entry: dict[str, Any], *, dry_run: bool) -> None:
    """Merge ``{**keys: {zaxy: entry}}`` into a JSON config, preserving the rest."""
    document: dict[str, Any] = {}
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            document = json.loads(text)
    node = document
    for key in keys:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[SERVER_NAME] = entry
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _merge_codex_toml(path: Path, executable: str, *, dry_run: bool) -> None:
    document = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    servers = document.get("mcp_servers")
    if not isinstance(servers, dict):
        servers = tomlkit.table()
        document["mcp_servers"] = servers
    table = tomlkit.table()
    table["command"] = executable
    table["args"] = ["serve"]
    servers[SERVER_NAME] = table
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomlkit.dumps(document), encoding="utf-8")


# JSON harnesses: (home-relative path, nested keys, entry builder from executable).
_JSON_TARGETS: dict[str, tuple[str, tuple[str, ...], Callable[[str], dict[str, Any]]]] = {
    "claude-code": (".claude.json", ("mcpServers",), lambda exe: {"command": exe, "args": ["serve"]}),
    "opencode": (
        ".config/opencode/opencode.json",
        ("mcp",),
        lambda exe: {"type": "local", "command": [exe, "serve"], "enabled": True},
    ),
    "openclaw": (".openclaw/openclaw.json", ("mcp", "servers"), lambda exe: {"command": exe, "args": ["serve"]}),
    "zcode": (".agents/mcp.json", ("mcpServers",), lambda exe: {"command": exe, "args": ["serve"]}),
    "pi": (".pi/agent/mcp.json", ("mcpServers",), lambda exe: {"command": exe, "args": ["serve"]}),
}


def register_harness(
    name: str,
    *,
    executable: str | None = None,
    home: str | Path | None = None,
    dry_run: bool = False,
) -> Registration:
    """Register the Zaxy MCP server for one harness at user scope."""
    harness = normalize_harness(name)
    root = Path(home).expanduser() if home is not None else Path.home()
    exe = resolve_zaxy_executable(executable)

    try:
        # Native Pi has no MCP; only the third-party adapter (which owns
        # ~/.pi/agent) can load one. Configure only when that dir exists.
        if harness == "pi" and not (root / ".pi" / "agent").exists():
            return Registration(
                harness,
                "skipped",
                "",
                "native Pi has no MCP support; install pi-mcp-adapter, then re-run",
            )

        if harness == "codex":
            if dry_run:
                target = codex_mcp_config_path(scope="user", workspace=root, codex_home=root / ".codex")
            else:
                target = write_codex_mcp_config(
                    scope="user",
                    workspace=root,
                    zaxy_executable=exe,
                    force=True,
                    codex_home=root / ".codex",
                )
            return Registration(harness, "configured", f"file:{target}", "codex user config.toml")

        if harness == "hermes":
            if dry_run:
                target = hermes_mcp_config_path(hermes_home=root / ".hermes")
            else:
                target = write_hermes_mcp_config(hermes_home=root / ".hermes", zaxy_executable=exe, force=True)
            return Registration(harness, "configured", f"file:{target}", "hermes config.yaml")

        rel, keys, build_entry = _JSON_TARGETS[harness]
        target = root / rel
        _merge_json(target, keys, build_entry(exe), dry_run=dry_run)
        return Registration(harness, "configured", f"file:{target}", f"{'/'.join(keys)} entry")
    except Exception as exc:  # a harness-specific failure must not abort the batch
        return Registration(harness, "error", "", str(exc))


def install_for_detected(
    *,
    executable: str | None = None,
    home: str | Path | None = None,
    only: list[str] | None = None,
    dry_run: bool = False,
    which: Callable[[str], str | None] = shutil.which,
) -> list[Registration]:
    """Detect harnesses (or use ``only``) and register Zaxy with each."""
    names = [normalize_harness(n) for n in only] if only else detect_harnesses(home=home, which=which)
    return [register_harness(n, executable=executable, home=home, dry_run=dry_run) for n in names]
