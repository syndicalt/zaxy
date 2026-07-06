"""Tests for machine-level harness MCP registration (zaxy.harness_install)."""

from __future__ import annotations

import json
from pathlib import Path

import tomlkit
import yaml

from zaxy import harness_install as hi

EXE = "/opt/zaxy/bin/zaxy"


def test_detect_by_binary_and_by_config(tmp_path: Path) -> None:
    # Nothing installed.
    assert hi.detect_harnesses(home=tmp_path, which=lambda _b: None) == []

    # Detected by a binary on PATH.
    found = hi.detect_harnesses(home=tmp_path, which=lambda b: "/usr/bin/x" if b == "codex" else None)
    assert "codex" in found

    # Detected by an existing config path.
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "config.yaml").write_text("mcp_servers: {}\n")
    found = hi.detect_harnesses(home=tmp_path, which=lambda _b: None)
    assert "hermes" in found


def test_aliases_normalize() -> None:
    assert hi.normalize_harness("zai") == "zcode"
    assert hi.normalize_harness("Claude-Code") == "claude-code"


def test_claude_code_json_merge_preserves_others(tmp_path: Path) -> None:
    cfg = tmp_path / ".claude.json"
    cfg.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}))
    reg = hi.register_harness("claude-code", executable=EXE, home=tmp_path)
    assert reg.status == "configured"
    doc = json.loads(cfg.read_text())
    assert doc["theme"] == "dark"  # untouched
    assert doc["mcpServers"]["other"] == {"command": "x"}  # untouched
    assert doc["mcpServers"]["zaxy"] == {"command": EXE, "args": ["serve"]}


def test_opencode_uses_mcp_key_and_command_array(tmp_path: Path) -> None:
    reg = hi.register_harness("opencode", executable=EXE, home=tmp_path)
    assert reg.status == "configured"
    doc = json.loads((tmp_path / ".config/opencode/opencode.json").read_text())
    assert doc["mcp"]["zaxy"] == {"type": "local", "command": [EXE, "serve"], "enabled": True}


def test_openclaw_nested_mcp_servers(tmp_path: Path) -> None:
    reg = hi.register_harness("openclaw", executable=EXE, home=tmp_path)
    assert reg.status == "configured"
    doc = json.loads((tmp_path / ".openclaw/openclaw.json").read_text())
    assert doc["mcp"]["servers"]["zaxy"] == {"command": EXE, "args": ["serve"]}


def test_zcode_generic_agents_path(tmp_path: Path) -> None:
    reg = hi.register_harness("zai", executable=EXE, home=tmp_path)
    assert reg.status == "configured"
    doc = json.loads((tmp_path / ".agents/mcp.json").read_text())
    assert doc["mcpServers"]["zaxy"] == {"command": EXE, "args": ["serve"]}


def test_codex_toml(tmp_path: Path) -> None:
    reg = hi.register_harness("codex", executable=EXE, home=tmp_path)
    assert reg.status == "configured"
    doc = tomlkit.parse((tmp_path / ".codex/config.toml").read_text())
    assert doc["mcp_servers"]["zaxy"]["command"] == EXE
    assert list(doc["mcp_servers"]["zaxy"]["args"]) == ["serve"]


def test_hermes_yaml(tmp_path: Path) -> None:
    reg = hi.register_harness("hermes", executable=EXE, home=tmp_path)
    assert reg.status == "configured"
    doc = yaml.safe_load((tmp_path / ".hermes/config.yaml").read_text())
    assert doc["mcp_servers"]["zaxy"]["command"] == EXE


def test_pi_skipped_without_adapter_then_configured_with_it(tmp_path: Path) -> None:
    skip = hi.register_harness("pi", executable=EXE, home=tmp_path)
    assert skip.status == "skipped"
    assert "adapter" in skip.detail

    (tmp_path / ".pi" / "agent").mkdir(parents=True)
    ok = hi.register_harness("pi", executable=EXE, home=tmp_path)
    assert ok.status == "configured"
    doc = json.loads((tmp_path / ".pi/agent/mcp.json").read_text())
    assert doc["mcpServers"]["zaxy"] == {"command": EXE, "args": ["serve"]}


def test_register_is_idempotent(tmp_path: Path) -> None:
    hi.register_harness("claude-code", executable=EXE, home=tmp_path)
    hi.register_harness("claude-code", executable="/new/zaxy", home=tmp_path)
    doc = json.loads((tmp_path / ".claude.json").read_text())
    # exactly one zaxy entry, updated to the latest executable
    assert doc["mcpServers"]["zaxy"]["command"] == "/new/zaxy"


def test_install_for_detected_only_list(tmp_path: Path) -> None:
    regs = hi.install_for_detected(executable=EXE, home=tmp_path, only=["opencode", "codex"])
    assert {r.harness for r in regs} == {"opencode", "codex"}
    assert all(r.status == "configured" for r in regs)
