# MCP Quickstart

MCP is Zaxy's primary framework-neutral interface. Start with the default local
path in the repository where you want memory captured:

```bash
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
```

`zaxy init` chooses the embedded local runtime by default. It writes local
profile and capture files, checks the repo-local graph projection posture, and
prints the exact client install command or config path. No Neo4j, Postgres,
Docker, Bolt URI, graph password, or hosted service is required for the local
MCP path. The durable data source remains `.eventloom/`; the embedded graph is a
rebuildable projection of that log.

## Client Setup

Use one local route per client. These commands keep the default embedded graph
runtime and `.eventloom/` source log; no sidecar service is required.

| Client | Recommended local route | Why |
|--------|--------------------------|-----|
| Codex | `codex mcp add` CLI command | Keeps user-level MCP config workspace-neutral while `zaxy serve` resolves the current repo at runtime. |
| Claude Code | Project-local `.mcp.json` install | Keeps the server definition next to the trusted project without editing user-global Claude settings. |
| Claude Desktop | Copyable JSON config | Claude Desktop normally uses an app-level MCP config, so inspect and paste the generated JSON. |
| Cursor | Project-local `.cursor/mcp.json` install | Cursor supports a repo-local MCP file with a top-level `mcpServers` object. |
| Generic MCP | Direct stdio server command | Any MCP-capable client can launch `zaxy serve` over stdio. |

### Codex

For Codex, copy or run the `codex mcp add` command printed by `zaxy init`. That
is the recommended path for this repo because it keeps the MCP server
workspace-neutral while resolving the current workspace at runtime. The command
shape is:

```bash
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

At session start, call `memory_bootstrap`. Before substantial work, call
`memory_checkout` with the current task. After using projected context, call
`memory_feedback` so Zaxy can reinforce useful memory.

For roadmap, release, review, implementation, resume, and high-context
questions, treat checkout as required before answering. Bootstrap is awareness;
checkout is the current cited prompt state. When checkout returns warnings,
unsupported context, stale state, or a required action, follow that guidance
instead of treating old model context as authoritative.

For multi-agent missions, use the `coordination_*` tools rather than plain
append calls when recording worker findings. `coordination_report_finding`
captures worker-local claims and evidence. `coordination_review_finding`,
`coordination_promote`, and `coordination_checkout` keep accepted parent state
separate from pending or conflicted worker-local findings.

Related references: [README.md](../README.md), [MCP interface](mcp.md),
[MCP install targets](mcp-install-targets.md), and
[Coordinate Quickstart](coordinate-quickstart.md).
