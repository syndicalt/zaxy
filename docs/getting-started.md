# Getting Started

Zaxy is a local-first memory fabric for AI agents. It keeps the immutable
work history in Eventloom JSONL files and projects structured facts into Neo4j
so agents can retrieve connected, temporal context through MCP tools. The
fastest path is to run Neo4j with Docker, start the MCP server, append a few
typed events, and query them back.

Start from the repository root:

```bash
pipx install zaxy-memory
# or: pip install zaxy-memory
zaxy status
```

MCP clients can only launch Zaxy after the `zaxy` CLI is installed on the
machine. Generated MCP config uses the resolved CLI executable path so GUI
clients do not have to inherit the same shell `PATH` as your terminal. Zaxy
prefers the installed `zaxy` console script and falls back to the current
process path when no console script is discoverable.

For local development from a checkout:

```bash
pip install -e ".[dev]"
./scripts/setup.sh
docker compose up -d
zaxy status
```

`./scripts/setup.sh` creates `.env`, `.eventloom`, and local runtime
directories. Development mode uses `neo4j/testpassword` and localhost-bound
ports through `bolt://localhost:7687` with no Neo4j TLS override. The Docker
Compose `zaxy` service still talks to its sibling Neo4j container internally;
local CLI and MCP onboarding should use localhost. Production mode is
different: `./scripts/setup.sh --production` writes secret files under
`./secrets`, configures `ZAXY_ENV=production`, and expects a TLS-enabled Neo4j
profile. See [deployment.md](deployment.md) before exposing remote SSE.

For an offline retrieval profile with no hosted services or API keys:

```bash
zaxy local-profile --output .env.local
zaxy local-profile --check
```

This configures deterministic hash embeddings, lexical reranking, and local
Neo4j auto-start at `bolt://localhost:7687`. It also clears local Neo4j TLS and
password-file overrides so stale container or production settings do not leak
into local CLI use. It is the recommended baseline for local development before
switching to hosted embeddings or model-backed rerankers.

Check local onboarding prerequisites before wiring an agent:

```bash
zaxy doctor
zaxy doctor --json
```

The doctor command verifies Eventloom writeability, local embedding/reranker
construction, static viewer generation, MCP default-session posture, observer
hook coverage, optional LLM packet memory projection, Neo4j configuration
posture, and production-mode warnings. It does not start Docker or require a
live Neo4j connection; use `zaxy status` when you want a live graph connectivity
test.

Before meaningful model work, inspect the memory contract and checkout current
state:

```bash
zaxy memory capabilities --session-id my-project-default
zaxy memory checkout "current task, project direction, and recent decisions" --session-id my-project-default
```

`memory capabilities` is the model-awareness surface. It explains when to use
Zaxy tools during a session: checkout at session start, before major work, after
compaction/resume, and when the user asks what is next; capture meaningful work
after completion; reinforce cited memories that were useful.

For a single first-run flow, use `zaxy init`. The happy path is deterministic
capture: local profile writing, MCP config rendering, observer hook config,
workspace genesis, hook heartbeat, doctor, and hook status. It does not proxy
model traffic and does not require a provider API key.

```bash
zaxy init . \
  --domain my-project \
  --preset local-claude
```

`--preset local-claude` expands to the explicit local setup:

```bash
zaxy init . \
  --domain my-project \
  --mcp-client claude-desktop \
  --mcp-output ./zaxy-mcp.json \
  --hook-client claude-code \
  --hook-output .claude/settings.local.json \
  --local-profile-output .env.local \
  --infra check
```

For Codex, use the deterministic Codex preset:

```bash
zaxy init . \
  --domain my-project \
  --preset local-codex
```

`local-codex` renders Codex MCP install guidance and writes the local profile.
It does not generate `.codex/hooks.json`: Codex parses that file as JSON, and
Zaxy does not assume a native Codex hook schema unless your Codex version
documents one. It does not enable packet capture.

Generated output files are non-destructive by default. Pass `--force` only when
you intentionally want to replace generated config. `--infra check` reports
local Neo4j and Docker posture without starting containers. Use
`--infra start` when you explicitly want onboarding to start the local Neo4j
runtime. The command prints a `Next:` section with the client install and
verification steps still required after files are generated.

Packet capture is optional diagnostic/high-fidelity mode because it can consume
provider quota and requires runtime/provider wire compatibility. Opt in with
`--capture-mode packet` or `--capture-mode hybrid` only when raw provider
request/response capture is worth that cost.

To start the default stdio MCP server:

```bash
zaxy serve
```

When stdio starts in local development mode, Zaxy checks
`bolt://localhost:7687`. If Neo4j is not reachable and Docker is available, it
starts a `zaxy-neo4j` container automatically and waits for Bolt before serving
MCP tools. This is the default in generated MCP client config. Set
`NEO4J_AUTO_START=false` to opt out.

To run the SSE transport for daemon-style clients:

```bash
zaxy serve --transport sse --port 8080
```

To let Zaxy observe client lifecycle checkpoints without proxying tool
execution, generate hook adapter config:

```bash
zaxy hooks claude-code \
  --eventloom-path .eventloom \
  --domain my-project \
  --output .claude/settings.local.json
```

The stable hook contract is documented in [hooks.md](hooks.md).
Claude Code is the preferred first install target. Codex currently gets a
generic shell snippet unless your local Codex version documents and enables a
working hook JSON path.

To verify that Zaxy can see lifecycle observations:

```bash
zaxy hook-status --eventloom-path .eventloom
zaxy hook-event heartbeat --eventloom-path .eventloom --session-id my-project-default --source manual
zaxy hook-status --eventloom-path .eventloom
```

The MCP tool names are stable: `memory_append`, `memory_query`,
`memory_feedback`, `memory_replay`, and `memory_invalidate`. A simple client can
append a typed `goal.created` or `task.proposed` event, query for the goal
title, record whether retrieved context was useful, and receive compact context
chunks from the graph. Zaxy also exposes a Python API through `MemoryFabric`;
see [api.md](api.md) for constructor and method details.

For day-to-day validation, run:

```bash
ruff check src tests
mypy src
pytest
zaxy doctor --release-smoke
scripts/release-check.sh --root .
```

The full pytest command enforces the 90 percent coverage gate configured in
`pyproject.toml`. Integration tests require Docker Neo4j services. The release
smoke check verifies the package version, changelog entry, release workflow,
and PyPI Trusted Publishing posture without contacting external services. The
release gate adds package artifact checks, documentation link validation, and
deployment preflight checks. The current public overview is
[site/index.html](../site/index.html), and the operational checklist remains in
[runbook.md](runbook.md). The [README.md](../README.md) is intentionally short;
these docs are the detailed operator and integrator reference.

The mental model is simple: append operational facts, extract graph facts,
retrieve connected context, and replay from the original event stream whenever
you need auditability. Do not treat Neo4j as the source of truth. Neo4j is the
reasoning layer. Eventloom is the durable history. MCP is the interface agents
call.
