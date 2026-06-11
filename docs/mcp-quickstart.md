# MCP Quickstart

MCP is Zaxy's primary framework-neutral interface. Start with the default local
path in the repository where you want memory captured:

```bash
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
```

`zaxy init` chooses the embedded local runtime by default. It writes local
profile and capture files, checks the repo-local graph projection posture, and
prints the exact client install command or config path. For Codex, the default
`zaxy init --codex-mcp-install auto` path writes or reuses user-level MCP config
when it can do so non-destructively, and falls back to the copyable
`codex mcp add` command when it cannot. No Neo4j, Postgres, Docker, Bolt URI,
graph password, or hosted service is required for the local MCP path. The
durable data source remains `.eventloom/`; the embedded graph is a rebuildable
projection of that log. Human output is compact by default; run
`zaxy init --verbose` to show every setup step, optional check, fallback command,
resume command, and note when debugging an install.

## Client Setup

Use one local route per client. These commands keep the default embedded graph
runtime and `.eventloom/` source log; no sidecar service is required.

| Client | Recommended local route | Why |
|--------|--------------------------|-----|
| Codex | `zaxy init` | Uses `zaxy init --codex-mcp-install auto` to merge user-level Codex MCP config when safe, or prints the exact command when not. |
| Claude Code | Project-local `.mcp.json` install | Keeps the server definition next to the trusted project without editing user-global Claude settings. |
| Claude Desktop | Copyable JSON config | Claude Desktop normally uses an app-level MCP config, so inspect and paste the generated JSON. |
| Cursor | Project-local `.cursor/mcp.json` install | Cursor supports a repo-local MCP file with a top-level `mcpServers` object. |
| Generic MCP | Direct stdio server command | Any MCP-capable client can launch `zaxy serve` over stdio. |

### Codex

For Codex, the fastest route is the default init path:

```bash
zaxy init
```

That expands to:

```bash
zaxy init --codex-mcp-install auto
```

Auto mode writes or reuses user-level Codex MCP config when that is safe. If an
existing `mcp_servers.zaxy` entry is different, it asks you to review that
config before replacing it. If no safe config target exists, it prints the
copyable `codex mcp add` command.

To force a direct user-level config merge, run:

```bash
zaxy init --codex-mcp-install user
```

This writes user-level Codex MCP config and keeps the MCP server
workspace-neutral while resolving the current workspace at runtime. Restart
Codex after init so it reloads the server list.

If you prefer not to let init write Codex config, force command mode and copy or
run the printed command. The command shape is:

```bash
zaxy init --codex-mcp-install command
codex mcp add zaxy -- zaxy serve
```

If you need Zaxy to print the exact command with resolved executable and
environment values:

```bash
zaxy ide-config codex --install --eventloom-path .eventloom
```

### Claude Code

For Claude Code, install the project-local MCP config:

```bash
zaxy ide-config claude-code --install --workspace . --eventloom-path .eventloom
```

This writes `.mcp.json` in the current workspace. Use this route for trusted
projects where the MCP server should travel with the repo.

### Claude Desktop

For Claude Desktop, print the app-level JSON fragment and paste it into the
Claude Desktop MCP config:

```bash
zaxy ide-config claude-desktop --eventloom-path .eventloom
```

### Cursor

For Cursor, install the project-local MCP config:

```bash
zaxy ide-config cursor --install --workspace . --eventloom-path .eventloom
```

This writes `.cursor/mcp.json` in the current workspace.

### Generic MCP

For a generic MCP client, configure a stdio server command:

```bash
zaxy serve --transport stdio
```

After installing the client config, start a fresh model session and ask the
client to list available MCP tools. You should see `memory_bootstrap`,
`memory_checkout`, `memory_feedback`, and the Coordinate tools. If the client
does not show them, run `zaxy doctor --eventloom-path .eventloom` and inspect
the generated MCP config before changing memory code.

## Model Call Rhythm

`memory_checkout` is the front door: call it first, with the current task,
before substantial work. At session start, call `memory_bootstrap` for
awareness, then checkout. After using projected context, call
`memory_feedback` so Zaxy can reinforce useful memory. Everything else is
plumbing or power use; `memory_capabilities` is the discovery surface for the
rest of the tool set. The experimental `memory_feeling_of_knowing` offers a
~1 ms pre-check of whether checkout would likely return anything for a query —
a cheap hint before checkout, never a memory answer.

For roadmap, release, review, implementation, resume, and high-context
questions, treat checkout as required before answering. Bootstrap is awareness;
checkout is the current cited prompt state. When checkout returns warnings,
unsupported context, stale state, or a required action, follow that guidance
instead of treating old model context as authoritative.

The server lists the core profile by default (since 2.1.0): only the
front-door verbs (`memory_checkout`, `memory_append`, `memory_query`,
`context_assemble`, `memory_feedback`, `memory_invalidate`,
`memory_capabilities`, `memory_feeling_of_knowing`) appear in the listing;
every other tool stays callable by name, and `memory_capabilities` reports
the available-but-unlisted set. To restore the full listing, serve with
`zaxy serve --profile full` or set `MCP_TOOL_PROFILE=full`.

For multi-agent missions, use the `coordination_*` tools rather than plain
append calls when recording worker findings. `coordination_report_finding`
captures worker-local claims and evidence. `coordination_review_finding`,
`coordination_promote`, and `coordination_checkout` keep accepted parent state
separate from pending or conflicted worker-local findings.

Related references: [README.md](../README.md), [MCP interface](mcp.md),
[MCP install targets](mcp-install-targets.md), and
[Coordinate Quickstart](coordinate-quickstart.md).
