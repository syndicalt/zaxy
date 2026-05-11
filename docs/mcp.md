# MCP Interface

MCP is Zaxy's primary integration surface. The server exposes memory operations
as typed tools so LangGraph, Claude Desktop, custom agents, or any MCP-capable
client can use the same memory system without linking directly against Python
application code.

`memory_append(event_type, actor, payload, thread?)` appends a typed event to
the Eventloom log for the selected session, extracts graph entities and edges,
upserts the Neo4j projection, records metrics, and emits a Pathlight span when
tracing is enabled. Payload size is bounded and session IDs are validated before
they affect filesystem paths.

`memory_query(query, temporal_filter?, limit?)` returns ranked context chunks.
The query router validates the string and limit, optionally embeds the query,
runs exact/keyword/vector/traversal search, fuses scores, and returns compact
context suitable for an agent prompt. Temporal filters let a client ask what was
valid at a specific time. Remote SSE requests are constrained to the session
from the configured session header. Results include Eventloom citations when
available so clients can display or replay the source event. Results also
include `score_explanation` metadata for ranking diagnostics.

`memory_verbatim(query, session_id?, limit?)` returns exact source chunks from
the Eventloom log without requiring Neo4j. It is the source-recall lane for
questions that need raw transcript turns, document chunks, identifiers, quoted
phrases, or file/source citations. Results include the raw content, BM25 score,
`eventloom://...` citation, source kind, and source metadata such as document
path/line range or transcript source/turn index.

`memory_feedback(entity_name, entity_type, feedback, ...)` records whether a
retrieved graph entity was useful. Positive feedback values, `used` and
`helpful`, append a `memory.reinforced` event with optional `importance`,
`query`, `source`, `score`, `citation`, and `reason` fields. Negative feedback,
`irrelevant`, appends an audit-only `memory.feedback` event and does not delete
or decay existing memory. Fabric-level feedback for assembled `packet_memory`
context also preserves source packet projection metadata. The tool uses the same
session scoping rules as append and query.

`memory_replay(session_id, from_seq?)` rebuilds session history from the
Eventloom log. This is useful for handoffs, audits, and debugging. In remote SSE
mode, the authenticated session scope is enforced so a client cannot replay a
different session.

`memory_invalidate(entity_name, entity_type, invalid_at)` closes the validity
window for a graph fact without deleting history. This lets agents correct
memory while preserving provenance.

`memory_capabilities(session_id?, current_task?)` returns the model-facing
memory contract for the active session. It tells the model what Zaxy is, which
tools are available, when to refresh context, which capture paths appear
healthy, and which call should normally happen next. Models should call this at
session start or whenever tool awareness is unclear, then call
`memory_checkout` for the current task. The contract explicitly treats memory
as an ambient loop: session-start awareness is not enough, so models should
refresh memory before major work, after compaction/resume, and before roadmap or
architecture decisions.

`memory_checkout(query, session_id?, ref?, replay_from_seq?, limit?, max_recent_events?)`
returns the high-level contract an agent should condition on before a turn. It
wraps context assembly with a `# Memory Checkout` prompt, current facts that
exclude superseded context, cited evidence, provenance parsed from
`eventloom://...` citations, retention metadata, warnings, the active working
set, and Checkout diagnostics. Diagnostics include source lane counts, total
citation count, current-fact citation count, current fact count, excluded
superseded context count, warning count, and a `memory_feedback` recommendation
when cited context is returned. This is the preferred tool when a model needs a
bounded, auditable working state rather than a raw list of retrieval hits. The
response also includes `guidance` with
model-facing trust and ignore instructions, a recommended follow-up
`memory_checkout` call, and concrete `memory_feedback` payload templates for
cited facts that materially influence the next response. The `quality` block
adds an answerability decision (`answer_from_memory`, `refresh_recommended`, or
`ask_user`), a bounded confidence score, reasons, and any required next action.
Checkout only returns `answer_from_memory` when current facts have current
Eventloom citations and the checkout has no warnings; missing, superseded-only,
uncited, or compacted checkout states ask the model to refresh memory or ask the
user instead of answering from stale context.
When `ref` is supplied, checkout resolves a Git-style memory ref such as `HEAD`
or `refs/heads/main` and filters replay/context to the target event identity.
MCP clients discover this tool through the standard `tools/list` handshake, so
an already-running client must reconnect before the new tool appears in the
model-visible tool registry.

Example checkout response fragment:

The canonical docs/test fixture for this contract is
`docs/examples/memory-checkout-contract.json`; keep examples aligned with that
fixture when changing the tool response.

```json
{
  "current_facts": [
    {
      "content": "Memory Checkout is the model-facing context contract.",
      "citation": "eventloom://zaxy-default/events/1882#abc123def456",
      "source_lane": "graph",
      "entity_name": "memory checkout",
      "entity_type": "decision"
    }
  ],
  "diagnostics": {
    "source_lanes": {"graph": 1},
    "citation_count": 1,
    "current_citation_count": 1,
    "current_fact_count": 1,
    "superseded_contexts_excluded": 0,
    "warning_count": 0,
    "feedback_recommended": true,
    "feedback_tool": "memory_feedback"
  },
  "quality": {
    "answerability": "answer_from_memory",
    "confidence": 0.75,
    "reasons": ["Retrieved current facts with Eventloom citations."],
    "required_action": null
  },
  "guidance": {
    "feedback": {
      "tool": "memory_feedback",
      "payloads": [
        {
          "entity_name": "memory checkout",
          "entity_type": "decision",
          "feedback": "used",
          "actor": "assistant",
          "citation": "eventloom://zaxy-default/events/1882#abc123def456"
        }
      ]
    }
  }
}
```

Model consumption rule: answer from memory only when `quality.answerability` is
`answer_from_memory`. If it is `refresh_recommended`, call the
`quality.required_action` tool before relying on the checkout. If it is
`ask_user`, ask for missing context instead of inventing continuity. When cited
facts materially influence the response, call `memory_feedback` with one of the
listed payloads so Zaxy can reinforce useful context.

Degraded checkout response fragment:

```json
{
  "current_facts": [
    {
      "content": "Memory Checkout is current.",
      "citation": null,
      "source_lane": "graph"
    }
  ],
  "diagnostics": {
    "citation_count": 0,
    "current_citation_count": 0,
    "current_fact_count": 1,
    "warning_count": 1
  },
  "quality": {
    "answerability": "refresh_recommended",
    "confidence": 0.29,
    "reasons": [
      "Retrieved current facts, but they lack Eventloom citations.",
      "Checkout contains warnings that reduce confidence."
    ],
    "required_action": {
      "tool": "memory_checkout",
      "query": "current decisions, blockers, and next actions for: Memory Checkout is current.",
      "reason": "Refresh memory before major follow-up work, after compaction/resume, or when task scope changes."
    }
  }
}
```

`context_assemble(query, session_id?, replay_from_seq?, limit?, max_recent_events?)`
returns a prompt-ready bundle containing bounded recent replay plus ranked
retrieval. The assembled retrieval set includes a reserved verbatim Eventloom
source-recall lane by default, and each context item includes `metadata` with an
`assembly_lane` value such as `graph` or `verbatim`.
The response also includes `assembly_policy` and `context_counts` fields so
clients can inspect the active policy and the number of graph, verbatim, and
replay items returned. The prompt begins with an `# Active Memory Working Set`
section, and the JSON response includes `working_set` items for bounded goals,
decisions, tasks, artifacts, blockers, and source anchors.
`context_after_turn(role, content, ...)` first appends a `transcript.turn` event,
then assembles context for the next model call.
`subagent_cleanup(parent_session_id, subagent_session_id, summary, ...)` records
`subagent.cleaned` in the subagent session and returns a handoff bundle with
summary and integrity status. These lifecycle tools are session-scoped under
remote SSE auth just like query and append.

MCP dispatch also performs automatic lifecycle capture by default. After each
tool call, Zaxy appends a `tool.call.completed` event to the resolved session
with the tool name, status, argument keys, and a bounded result summary. Raw
argument values are not persisted in the lifecycle payload. Capture is
best-effort: failures while recording metadata do not fail the original MCP
tool call. Set `MCP_LIFECYCLE_CAPTURE_ENABLED=false` to disable this automatic
capture. Server shutdown also records a best-effort `session.ended` event for
the default session when lifecycle capture is enabled, and subagent cleanup
records `subagent.completed` alongside the existing `subagent.cleaned` event.

Run stdio locally:

```bash
zaxy serve
```

At session start, clients or agents should bootstrap model awareness:

```bash
zaxy memory capabilities --session-id zaxy-default
zaxy memory checkout "current task, project direction, and recent decisions" --session-id zaxy-default
```

During a long session, repeat checkout before important work and after
compaction or resume. Capture meaningful completions with `context_after_turn`
or typed `memory_append`, and reinforce cited context that was actually used
with `memory_feedback`.

Generate first-run MCP client config:

```bash
zaxy ide-config claude-desktop --eventloom-path .eventloom
zaxy ide-config claude-code --install --workspace .
zaxy ide-config cursor --eventloom-path .eventloom
zaxy ide-config cursor --install --workspace .
zaxy ide-config vscode --eventloom-path .eventloom
zaxy ide-config vscode --install --workspace .
zaxy ide-config codex --install --domain zaxy
zaxy ide-config codex --install --codex-config-scope project --codex-trusted-project --workspace .
zaxy ide-config claude-desktop --eventloom-path .eventloom --domain zaxy
```

By default, `ide-config` prints copyable config. The `--install` flag is
limited to verified project-local JSON targets: `.mcp.json` for Claude Code,
`.cursor/mcp.json` for Cursor, and `.vscode/mcp.json` for VS Code. Install mode
merges the `zaxy` server entry into the documented root object, preserves
unrelated servers, and refuses to replace an existing `zaxy` entry unless
`--force` is passed.

Codex is CLI-assisted by default: `zaxy ide-config codex --install` prints the
official `codex mcp add zaxy ... -- zaxy serve ...` command. Direct TOML writes
are opt-in through `--codex-config-scope project|user`. Project-scoped writes
target `.codex/config.toml` and require `--codex-trusted-project` because Codex
only loads project config from trusted projects. User-scoped writes target
`CODEX_HOME/config.toml` or `~/.codex/config.toml`.

`zaxy ide-config codex` without `--install` also prints the Codex CLI command,
because Codex does not use the JSON config shapes consumed by Claude, Cursor, or
VS Code.

Codex troubleshooting: prefer launching restored work with terminal-level
`codex resume ...` rather than calling `/resume` from inside an already running
Codex session. In-session resume can leave the old MCP child alive and start a
second identical `zaxy serve` process. If that happens, fully exit Codex and
start a fresh resume from the terminal, then verify with:

```bash
ps -ef | awk '/[z]axy serve/ {print}'
```

Install the `zaxy` CLI before generating MCP config:

```bash
pipx install zaxy-memory
# or: pip install zaxy-memory
```

MCP clients cannot launch a server command that does not exist yet. Generated
stdio config uses the resolved executable path by default, preferring the
installed `zaxy` console script and falling back to the current process path.
This is more reliable for GUI clients than assuming they inherit your shell
`PATH`.

Generate observer hook adapters:

```bash
zaxy hooks claude-code --eventloom-path .eventloom --domain zaxy
zaxy hooks codex --eventloom-path .eventloom --domain zaxy
```

Hook adapters do not proxy tool execution. Agents and tools continue to execute
normally while hooks append lightweight Eventloom observations through
`zaxy hook-event`. The sink is intentionally graph-independent so session stop
and pre-compaction hooks can record provenance even when Neo4j is unavailable.
Custom clients can implement the same contract by emitting normalized triggers
such as `session-start`, `stop`, `precompact`, `checkpoint`, `command`, and
`file-edit`. Command and file-edit triggers become first-class
`command.completed` and `file.edit.applied` events, which lets automatic capture
feed retrieval and working-set projection without storing raw file content.

This deterministic capture path is the default Zaxy onboarding posture. Packet
capture is optional and should be enabled only when raw provider request/response
audit is worth the provider quota and transport-compatibility cost.

These commands print copyable JSON fragments and do not include bearer tokens,
passwords, or admin secrets. Keep remote SSE credentials in the client secret
store or environment, not in committed config.

The verified write targets for future client-specific installers are tracked in
[mcp-install-targets.md](mcp-install-targets.md). That matrix is the guardrail
for deciding when `zaxy init` may write or merge config directly versus printing
copyable instructions.

Generated configs include `ZAXY_DOMAIN` and a domain-prefixed
`EVENTLOOM_THREAD`, such as `zaxy-default`. This prevents clients that omit
`session_id` from accidentally sharing the global `default` session across
different projects. For remote SSE configs, the same default is sent through the
session header.

For local stdio clients, the generated config is intentionally self-contained:
it forces development-mode localhost settings and enables `NEO4J_AUTO_START`.
On startup, Zaxy reuses `bolt://localhost:7687` when it is already available;
otherwise it starts a named Docker container, `zaxy-neo4j`, with the default
local credentials. Generated stdio configs set `startup_timeout_sec` to `90`
so MCP clients do not kill startup while Docker is creating or warming the
local Neo4j container. Set `NEO4J_AUTO_START=false` or provide a non-local
`NEO4J_URI` when you want to manage Neo4j yourself.

Run SSE daemon mode:

```bash
zaxy serve --transport sse --host 127.0.0.1 --port 8080
```

Production SSE requires either static bearer auth or OIDC. Static bearer auth
uses `MCP_REMOTE_AUTH_TOKEN` or `MCP_REMOTE_AUTH_TOKEN_FILE`; clients send
`Authorization: Bearer <token>` and a session header such as
`x-zaxy-session-id: agent-1`. OIDC uses `MCP_OIDC_ISSUER`,
`MCP_OIDC_AUDIENCE`, and `MCP_OIDC_JWKS_URL`; clients send an access token and
Zaxy scopes the request from `MCP_OIDC_SESSION_CLAIM`. Production also requires
`MCP_ADMIN_TOKEN` or `MCP_ADMIN_TOKEN_FILE` for replay and invalidation.

The MCP implementation lives in `src/zaxy/mcp_server.py`. Core orchestration
lives in `src/zaxy/core.py`. Security helpers live in `src/zaxy/security.py`.
See [api.md](api.md) for Python-level calls, [configuration.md](configuration.md)
for environment variables, and [security.md](security.md) for remote transport
hardening. The public overview is [site/index.html](../site/index.html), while
[README.md](../README.md) keeps the short command list.
