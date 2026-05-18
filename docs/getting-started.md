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
hook coverage, capture health, optional LLM packet memory projection, Neo4j
configuration posture, and production-mode warnings. `capture_health` summarizes
whether the high-value automatic lanes are active and includes the managed
`zaxy capture start --workspace ...` action when Codex capture is configured but
idle. It does not start Docker or require a live Neo4j connection; use
`zaxy status` when you want a live graph connectivity test.

Before meaningful model work, inspect the session-start bootstrap packet:

```bash
zaxy memory bootstrap --session-id my-project-default
```

`memory bootstrap` is the compact model-awareness surface for session start. It
packages the active capability manifest, the recommended first
`memory_checkout` call, deterministic capture status, and the trust policy for
what to prefer, ignore, and record. `memory capabilities` remains available when
you want the fuller manifest, and `memory checkout` loads the cited working
state before real work begins.

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

The one-command local Codex happy path starts the managed watcher during init:

```bash
zaxy init . --domain my-project --preset local-codex --capture start
zaxy doctor
zaxy memory bootstrap --session-id my-project-default
```

`local-codex` renders Codex MCP install guidance, writes the local profile, and
writes `.codex/zaxy-capture.json` for deterministic local capture. It does not
generate `.codex/hooks.json`: Codex parses that file as JSON, and Zaxy does not
assume a native Codex hook schema unless your Codex version documents one.
Start `zaxy capture start --workspace .` from the printed next steps to run the
managed watcher that imports local Codex session JSONL into Eventloom. To start
that watcher during onboarding, pass `--capture start`. It does not enable
packet capture. After startup, `zaxy doctor` should show `capture_health: ok`
once command, file-edit, tool-call, and transcript observations have appeared.
`zaxy init` also prints a capture summary showing whether local capture is
configured, whether the watcher is running, the latest imported observation when
available, and the doctor capture-health result. The same block is available in
`zaxy init --json` as `capture`.

Generated output files are non-destructive by default. Pass `--force` only when
you intentionally want to replace generated config. `--infra check` reports the
selected local projection backend and Docker posture without starting
containers. Use `--infra start` when you explicitly want onboarding to start the
local runtime. Neo4j remains the default backend. For experimental pgGraph
bootstrap, install the optional extra and point Zaxy at a local pgGraph checkout
so it can run the extension installer instead of starting plain Postgres:

```bash
pip install "zaxy-memory[pggraph]"
zaxy init . \
  --projection-backend pggraph \
  --pggraph-dsn postgresql://postgres:postgres@localhost:5432/zaxy \
  --pggraph-repo /path/to/pggraph \
  --infra start
```

The command prints a `Next:` section with the client install and verification
steps still required after files are generated.

Packet capture is optional diagnostic/high-fidelity mode because it can consume
provider quota and requires runtime/provider wire compatibility. Opt in with
`--capture-mode packet` or `--capture-mode hybrid` only when raw provider
request/response capture is worth that cost.

To start the default stdio MCP server:

```bash
zaxy serve
```

When stdio starts in local development mode, Zaxy checks the selected projection
backend. With the default Neo4j backend, it checks `bolt://localhost:7687`. If
Neo4j is not reachable and Docker is available, it starts a `zaxy-neo4j`
container automatically and waits for Bolt before serving MCP tools. With
`PROJECTION_BACKEND=pggraph`, it checks `PGGRAPH_DSN`; local automatic startup
requires `PGGRAPH_REPO` to point at a pgGraph checkout containing
`scripts/quickstart.sh`, because Zaxy must install pgGraph into the local
PostgreSQL container before graph traversal is available. Set
`NEO4J_AUTO_START=false` or `PGGRAPH_AUTO_START=false` to opt out.

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
zaxy doctor --beta-readiness
zaxy capture-soak --eventloom-path .eventloom --session-id my-project-default
scripts/release-check.sh --root .
scripts/beta-uat.sh
```

The full pytest command enforces the 90 percent coverage gate configured in
`pyproject.toml`. Integration tests require Docker Neo4j services. The release
smoke check verifies the package version, changelog entry, release workflow,
and PyPI Trusted Publishing posture without contacting external services. The
beta readiness check verifies that the release smoke gate, release gate script,
clean-repo UAT script, docs happy path, and deterministic capture happy path are
present. The release gate adds package artifact checks, documentation link
validation, and deployment preflight checks. `scripts/beta-uat.sh` creates a
throwaway workspace, installs Zaxy into a fresh virtual environment, runs
`zaxy init`, starts deterministic capture, runs `zaxy memory bootstrap`,
performs a cited `zaxy memory checkout`, and checks doctor, hook, capture, and
memory status plus the `zaxy capture-soak` beta evidence report. The current public overview is
[site/index.html](../site/index.html), and the operational checklist remains in
[runbook.md](runbook.md). The [README.md](../README.md) is intentionally short;
these docs are the detailed operator and integrator reference.

The mental model is simple: append operational facts, extract graph facts,
retrieve connected context, and replay from the original event stream whenever
you need auditability. Do not treat Neo4j as the source of truth. Neo4j is the
reasoning layer. Eventloom is the durable history. MCP is the interface agents
call.
