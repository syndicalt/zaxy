# MCP Install Targets

This page records the verified MCP client targets that `zaxy init` and future
write-and-merge helpers may support. The rule is conservative: Zaxy can print a
config fragment for any compatible client, but it should only write into a
client-owned file when the path, schema, and merge behavior are verified from
official client documentation.

The goal is not to hide client differences behind a fragile universal writer.
Each target has its own config shape, trust model, and scope semantics. A
professional onboarding path should preserve unrelated user settings, avoid
silent global writes, and make restart or trust prompts explicit.

## Write Policy

Installer helpers must follow these constraints:

- Default to dry-run or print-only unless the target is listed as safe to write.
- Merge one `zaxy` server entry without replacing unrelated server entries.
- Preserve formatting when practical, but prefer a valid config over cosmetic
  whitespace preservation.
- Refuse malformed existing config and show a repair-oriented error.
- Create parent directories only for documented project-local paths.
- Avoid hardcoding bearer tokens, API keys, or admin secrets.
- Prefer official client CLI commands when they exist and are less risky than
  editing user-level config directly.
- Make global writes explicit through a flag such as `--scope user` or
  `--global`.

## Target Matrix

| Client | Verified config target | Schema | Safe write posture |
|--------|------------------------|--------|--------------------|
| Claude Code MCP | Project `.mcp.json`; local and user scopes live in `~/.claude.json` | top-level `mcpServers` | Project `.mcp.json` is safe with explicit project scope. Local/user scope should prefer `claude mcp add` or `claude mcp add-json`. |
| Claude Code hooks | `.claude/settings.local.json`, `.claude/settings.json`, and `~/.claude/settings.json` | top-level `hooks` | Local project settings are safe for personal observer hooks. Shared project settings should require an explicit shared flag. |
| Cursor | Project `.cursor/mcp.json`; global `~/.cursor/mcp.json` | top-level `mcpServers` | Project-local `.cursor/mcp.json` is safe. Global config should require an explicit global flag. |
| VS Code | Workspace `.vscode/mcp.json`; user-profile `mcp.json` opened by command | top-level `servers` | Workspace `.vscode/mcp.json` is safe. User profile writes should prefer VS Code commands or an explicit global flag. |
| Codex | User `~/.codex/config.toml`; trusted project `.codex/config.toml` | TOML tables under `[mcp_servers.<name>]` | CLI-assisted through `codex mcp add` by default. Direct TOML merge is available only for explicit user or trusted project scope. |
| Hermes Agent | Global `~/.hermes/config.yaml` or `HERMES_HOME/config.yaml` | YAML mapping under `mcp_servers.<name>` | Global YAML merge is explicit through `zaxy ide-config hermes --install`; the generated server is workspace-neutral and does not pin repo-local Eventloom/session values. |

## Client Notes

Claude Code supports MCP scopes directly in its CLI. Its project scope writes
`.mcp.json` at the project root and uses the standard `mcpServers` JSON shape.
Local and user scoped MCP servers are stored in `~/.claude.json`, so Zaxy should
not silently edit that file. For those scopes, the safer generated next step is
an explicit `claude mcp add zaxy -- zaxy serve` or `claude mcp add-json zaxy ...`
command.

Claude Code hooks are separate from MCP server installation. They are still part
of Zaxy onboarding because they provide observer capture for tool calls, session
events, and compaction events. The documented hook settings locations match the
existing Zaxy default of `.claude/settings.local.json` for personal project
setup.

Cursor uses a JSON file named `mcp.json` with a top-level `mcpServers` object.
The project target is `.cursor/mcp.json`; the global target is
`~/.cursor/mcp.json`. Zaxy should default to project-local output because that
matches repository onboarding and avoids surprising cross-project memory wiring.

VS Code uses `mcp.json` too, but its workspace schema uses top-level `servers`,
not `mcpServers`. The project-local target is `.vscode/mcp.json`. This is why
Zaxy keeps a separate VS Code renderer instead of reusing the Claude or Cursor
JSON fragment.

Codex supports MCP in the CLI and IDE extension. The documented config file is
`~/.codex/config.toml`, with trusted projects allowed to use
`.codex/config.toml`. Codex also exposes `codex mcp add`, so Zaxy supports Codex
by rendering a workspace-neutral `codex mcp add zaxy -- zaxy serve` command
unless a direct config scope is requested. The rendered config may include
local runtime defaults, such as localhost Neo4j and empty TLS/password-file
overrides, so an old production `.env` does not break interactive MCP startup.
It does not write repo-specific Eventloom/session/domain environment into Codex
config; `zaxy serve` derives the active workspace at startup. Direct TOML
support requires either user scope or project scope with an explicit trusted
project acknowledgement. The TOML merge preserves unrelated server entries,
rejects malformed TOML, and refuses to replace an existing `zaxy` entry unless
`--force` is passed.

Hermes Agent documents MCP servers in the global YAML config at
`~/.hermes/config.yaml`, under the top-level `mcp_servers` mapping. Zaxy treats
that as a global integration point, so `zaxy ide-config hermes` renders YAML
with `zaxy serve`, local Neo4j startup defaults, and model-facing memory tools,
but without `EVENTLOOM_PATH`, `EVENTLOOM_THREAD`, or `ZAXY_DOMAIN`. At runtime,
`zaxy serve` resolves the active workspace and default session from the process
working directory, which prevents one repository from leaking into another.
`zaxy ide-config hermes --install` merges into the Hermes YAML file, preserves
unrelated settings and servers, rejects malformed YAML, and refuses to replace
an existing `zaxy` entry unless `--force` is passed.

## Implementation Order

1. Add Claude Code local hook detection/write coverage using the existing hook
   adapter path.

Completed: the shared JSON merge engine and project-local write helpers now
cover Cursor, VS Code, and Claude Code project MCP targets. Codex is now
supported through CLI-assisted `codex mcp add` command rendering and explicit
TOML merge support for user and trusted-project scopes. Hermes Agent is
supported through workspace-neutral YAML rendering and explicit global
`config.yaml` merge support.

## Sources

- [Claude Code MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
- [Claude Code hooks](https://docs.anthropic.com/en/docs/claude-code/hooks)
- [Claude Code settings](https://docs.anthropic.com/en/docs/claude-code/settings)
- [Cursor MCP](https://docs.cursor.com/advanced/model-context-protocol)
- [Cursor CLI MCP](https://docs.cursor.com/cli/mcp)
- [VS Code MCP servers](https://code.visualstudio.com/docs/copilot/customization/mcp-servers)
- [Codex MCP](https://developers.openai.com/codex/mcp)
- [Codex config reference](https://developers.openai.com/codex/config-reference)
- [Hermes Agent MCP](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/user-guide/features/mcp.md)

See [mcp.md](mcp.md), [hooks.md](hooks.md), and [README.md](../README.md) for
the current generated config and onboarding commands.
