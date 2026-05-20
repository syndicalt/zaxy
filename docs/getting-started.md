# Getting Started

Zaxy is a local-first memory fabric for AI agents. It keeps the immutable
work history in Eventloom JSONL files and projects structured facts into Neo4j
so agents can retrieve connected, temporal context through MCP tools. The
fastest path is to install the CLI, run one local onboarding command, and
verify that Eventloom plus the model-facing bootstrap are readable. For the
architecture tradeoffs behind this shape, see [why-zaxy.md](why-zaxy.md).

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

## Five-minute local smoke test

Use this path when you want to know whether Zaxy is viable on a workstation
without committing to a full graph deployment. It writes local config, records
session genesis and heartbeat events, checks infrastructure posture, and prints
copyable next steps. It does not require a hosted API key.

```bash
pipx install zaxy-memory
zaxy init
zaxy memory log --eventloom-path .eventloom --limit 5
zaxy activate codex --eventloom-path .eventloom --current-task "continue onboarding"
zaxy doctor --eventloom-path .eventloom
```

Expected local artifacts:

- `.eventloom/`: append-only Eventloom JSONL logs, one file per session.
- `.env.local`: local embedding, reranker, and embedded graph defaults.
- `.codex/zaxy-capture.json`: deterministic local Codex capture config.
- A printed `codex mcp add` command: MCP install guidance for Codex.

The first smoke-test command should show recent genesis or heartbeat events.
The bootstrap command should print the model-facing Memory Bootstrap packet for
`my-project-default`, including the recommended first checkout and capture
trust policy. `zaxy doctor --eventloom-path .eventloom` should make any missing
MCP config, hook, capture, or graph posture explicit before you start relying on
the memory layer. The same local data can be inspected later with
`zaxy memory status --eventloom-path .eventloom` and
`zaxy memory diff --eventloom-path .eventloom --session-id my-project-default --from-seq 1 --to-seq 10`.

For Claude Code, swap the init command:

```bash
zaxy init . --domain my-project --preset local-claude --infra check
```

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
zaxy local-profile --projection-backend embedded --output .env.local
zaxy local-profile --check
```

This configures deterministic hash embeddings, lexical reranking, and local
Neo4j auto-start at `bolt://localhost:7687`. It also clears local Neo4j TLS and
password-file overrides so stale container or production settings do not leak
into local CLI use. It is the recommended baseline for local development before
switching to hosted embeddings or model-backed rerankers.
Use `--projection-backend embedded` for the no-sidecar profile that writes
`PROJECTION_BACKEND=embedded`, `.eventloom/projections/embedded.kuzu`, and
disables Neo4j/pgGraph autostart.

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

Before meaningful Codex work, emit the session-start activation packet:

```bash
zaxy activate codex --session-id my-project-default --current-task "ship the next change"
```

`activate codex` prints a prompt-ready Memory Bootstrap packet and records that
the activation handoff was shown. The packet includes the active capability
manifest, the recommended first `memory_checkout` call, deterministic capture
status, and the trust policy for what to prefer, ignore, and record. `memory
bootstrap` remains available when you only want the raw bootstrap packet,
`memory capabilities` exposes the fuller manifest, and `memory checkout` loads
the cited working state before real work begins.

To start Codex with that activation packet as the initial prompt:

```bash
zaxy activate codex --session-id my-project-default --current-task "ship the next change" --launch
```

Use `--dry-run` to inspect the exact `codex --cd ... <prompt>` command without
starting Codex.

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
zaxy init
```

The bare local path expands to the no-sidecar embedded Codex onboarding preset:
it writes `.eventloom/`, `.env.local`, `.codex/zaxy-capture.json`, renders the
Codex MCP install command, checks the repo-local embedded projection posture,
and records the first workspace heartbeat. It installs deterministic capture
configuration safely; start the watcher when you want live local capture:

```bash
zaxy capture start --workspace .
zaxy doctor
zaxy memory bootstrap
```

To start managed deterministic capture during onboarding, pass `--capture
start`:

```bash
zaxy init --capture start
```

Bare `zaxy init` selects `--projection-backend embedded` for local onboarding
and checks the repo-local projection path instead of Neo4j or pgGraph. The
embedded projection is created lazily under `.eventloom/projections/`, so this
path needs no external graph service. `--preset local-embedded-codex` remains
available as an explicit spelling of the same local path.

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

## Keep source-derived context fresh

Use `zaxy refresh-context` when documents or code should stay synchronized with
the memory projection without replaying or re-embedding every unchanged file.
The command fingerprints supported files, records source lifecycle events,
indexes only discovered or changed sources, records the current transform
version on derived rows, and retires stale backend projection rows for changed
or deleted sources.

```bash
zaxy refresh-context docs --kind documents --session-id my-project-default
zaxy refresh-context . --kind codebase --session-id my-project-default
```

The refresh state is stored under `.eventloom/context-refresh/` and is scoped by
session and source kind. Eventloom remains the audit log: each run records
`source.discovered`, `source.changed`, `source.unchanged`, `source.deleted`,
`projection.updated`, and `projection.retired` events as applicable. A transform
version change is treated as a reprocessing trigger even when file bytes are
unchanged. Neo4j is the default projection backend, and the same refresh command
can target the experimental pgGraph adapter when it is explicitly selected:

```bash
zaxy refresh-context . \
  --kind codebase \
  --session-id my-project-default \
  --projection-backend pggraph \
  --pggraph-dsn postgresql://postgres:postgres@localhost:5432/zaxy
```

This is the local, deterministic refresh path for source-derived context. Use
`zaxy reproject` when an extractor or schema migration requires rebuilding a
projection from the immutable Eventloom history.

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
validation, deployment preflight checks, backend shootout evidence, injected-token
efficiency floors, and 100-query embedded scale validation. `scripts/beta-uat.sh`
creates a throwaway workspace, installs Zaxy into a fresh virtual environment,
runs `zaxy init`, starts deterministic capture, runs `zaxy memory bootstrap`,
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
