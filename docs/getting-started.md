# Getting Started

Zaxy is a local-first memory fabric for AI agents. It keeps the immutable
work history in Eventloom JSONL files and projects structured facts into an
embedded Kuzu graph so agents can retrieve connected, temporal context through MCP tools. The
fastest path is to install the CLI, run one local onboarding command, and
verify that Eventloom plus the model-facing bootstrap are readable. For the
architecture tradeoffs behind this shape, see [why-zaxy.md](why-zaxy.md).

## Five-minute first run

The default local path is embedded and repo-local. No Neo4j, Postgres, Docker, or graph password is required for the first run.

```bash
pipx install zaxy-memory
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
zaxy memory checkout "current project memory and next useful action" --eventloom-path .eventloom
zaxy doctor --eventloom-path .eventloom
```

Success means:

- `.eventloom/` exists;
- `zaxy memory bootstrap` returns model-facing guidance;
- `zaxy memory checkout` returns a prompt-ready checkout with diagnostics;
- `zaxy doctor` reports no blocking local setup errors.

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
- Either a user-level Codex MCP config merge, a printed `codex mcp add`
  command when no safe config target exists, or an explicit review prompt when
  an existing `zaxy` Codex entry would be replaced.

For Codex, `zaxy init --codex-mcp-install auto` is the default. It writes or
reuses `~/.codex/config.toml` when the merge is non-destructive, including when
an existing `mcp_servers.zaxy` entry already matches the workspace-neutral
server Zaxy would install. If the existing entry is different, auto mode falls
back to a review prompt instead of handing you a command that can silently
replace it. If no safe config target exists, auto mode prints the copyable
`codex mcp add` command. Use an explicit mode when you want to force the
decision after review:

```bash
zaxy init --codex-mcp-install user
# or: zaxy init --codex-mcp-install command
```

Both paths use the same workspace-neutral server definition. After init, start
or restart Codex through the printed `zaxy activate codex ... --launch` command
so the MCP server list and Zaxy activation packet load together. The printed
command includes explicit `--eventloom-path` and `--workspace-root` values, so
it still targets the initialized repo when copied from another shell.

The first smoke-test command should show recent genesis or heartbeat events.
The bootstrap command should print the model-facing Memory Bootstrap packet for
`my-project-default`, including the recommended first checkout and capture
trust policy. `zaxy doctor --eventloom-path .eventloom` should make any missing
MCP config, hook, capture, or graph posture explicit before you start relying on
the memory layer. The same local data can be inspected later with
`zaxy memory status --eventloom-path .eventloom` and
`zaxy memory diff --eventloom-path .eventloom --session-id my-project-default --from-seq 1 --to-seq 10`.

`zaxy init` prints compact human output by default so the required actions are
easy to scan. The compact header includes a `Readiness:` line, so a successful
setup that still needs MCP install, config review, or activation work does not
look ready by accident. Use `zaxy init --verbose` when you need full setup
diagnostics, optional checks, fallback commands, resume guidance, and notes for
support or troubleshooting.
For installer scripts and client UIs, `zaxy init --json` preserves the raw
onboarding payload and adds `setup.status`, `setup.issues`, `readiness.status`,
`setup.pending`, `readiness.reasons`, `readiness.actions`, and structured
`readiness.action_items`. It also includes `setup.summary`,
`readiness.summary`, `readiness.required_action_count`, and
`readiness.reason_count` so installers can render compact status without
parsing the human output. Treat `readiness.status = "ready"` as the point where
the initialized memory path has no remaining required actions; `setup.status`
can still describe whether file/config writes completed.
`readiness.action_items` carries split `label`, `command`, source text, and
`hints` so installer UIs can render copy buttons, non-command review tasks, and
compact-output tips without parsing human prose. Activation actions use those
hints to explain `<task>` replacement and why explicit Eventloom/workspace paths
make the launcher safe to run from any shell.
`readiness.blocking_diagnostics` and `readiness.non_blocking_diagnostics` split
doctor warnings that block first-run readiness from advisory checks that remain
visible for follow-up.
Configured-but-idle Codex capture is advisory when the activation launcher will
start it; inspect `readiness.capture` or `readiness.non_blocking_diagnostics`
for that state instead of treating it as a required readiness action.
Rendered-but-not-yet-applied config, such as a copyable MCP install command,
appears in `setup.pending` rather than `setup.issues`.

For Claude Code, swap the init command:

```bash
zaxy init . --domain my-project --preset local-claude --infra check
```

For local development from a checkout:

```bash
pip install -e ".[dev]"
./scripts/setup.sh
zaxy status
```

`./scripts/setup.sh` creates `.env`, `.eventloom`, and local runtime
directories. Development mode can still start optional Docker services for
integration tests, but local CLI and MCP onboarding use the embedded graph
profile by default. Production mode is different: `./scripts/setup.sh
--production` writes secret files under `./secrets`, configures
`ZAXY_ENV=production`, and expects any external sidecar backend to be configured
explicitly. See [deployment.md](deployment.md) before exposing remote SSE.

Start Docker sidecars only when you are running integration tests or explicitly
comparing external backends:

```bash
docker compose --profile integration up -d neo4j-test neo4j-tls
pytest -m integration
```

For an offline retrieval profile with no hosted services or API keys:

```bash
zaxy local-profile --output .env.local
zaxy local-profile --projection-backend embedded --output .env.local
zaxy local-profile --check
```

This configures deterministic hash embeddings, lexical reranking,
`PROJECTION_BACKEND=embedded`, `.eventloom/projections/embedded.kuzu`, and
disabled Neo4j/pgGraph autostart. It also clears local Neo4j TLS and
password-file overrides so stale container or production settings do not leak
into local CLI use. Use `--projection-backend neo4j` only when you explicitly
want the optional sidecar profile, and install `zaxy-memory[neo4j]` first.

Check local onboarding prerequisites before wiring an agent:

```bash
zaxy doctor
zaxy doctor --json
```

The doctor command verifies Eventloom writeability, local embedding/reranker
construction, static viewer generation, MCP default-session posture,
model-visible `AGENTS.md` activation instructions, observer hook coverage,
capture health, optional LLM packet memory projection, Neo4j configuration
posture, and production-mode warnings. `capture_health` summarizes whether the
high-value automatic lanes are active and includes the managed
`zaxy capture start --workspace ...` action when Codex capture is configured but
idle. `agent_instructions` warns when the marker-managed `Zaxy Memory Activation`
block is missing. It does not start Docker or require a live Neo4j connection;
use `zaxy status` when you want a live graph connectivity test.

Before meaningful Codex work, emit the session-start activation packet:

```bash
zaxy activate codex \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --current-task "ship the next change" \
  --workspace-root .
```

`activate codex` prints a prompt-ready Memory Bootstrap packet, records that
the activation handoff was shown, and starts the managed Codex capture watcher
when `.codex/zaxy-capture.json` is configured. If capture cannot be started, the
packet marks the session as degraded and prints the setup or `zaxy capture
start` action instead of implying that automatic capture is healthy. Use
`--no-ensure-capture` only when another supervisor is already managing capture.
The packet includes the active capability manifest, the recommended first
`memory_checkout` call, deterministic capture status, MCP-tool availability as
runtime-unverified, a CLI checkout fallback for resumed sessions where MCP tools
are missing, and the trust policy for what to prefer, ignore, and record.
For launcher or CI flows that must fail closed when Codex capture is configured
but stopped, run `zaxy hook-status --require-capture-running` before substantial
work. Resume-aware clients should also emit `zaxy hook-event resume`, which
records `hook.resumed` and appends a fresh checkout reminder even when memory was
used recently.
`zaxy init` persists the same activation contract into
`session.genesis.payload.write_instructions.memory_activation`: the activation
launcher command, resume hook command, CLI checkout fallback, runtime-unverified
MCP-tool status, and a fail-closed blocker when no fresh checkout is present.
`memory bootstrap` remains available when you only want the raw bootstrap
packet, `memory capabilities` exposes the fuller manifest, and `memory checkout`
loads the cited working state before real work begins.

To start Codex with that activation packet as the initial prompt:

```bash
zaxy activate codex \
  --eventloom-path .eventloom \
  --session-id my-project-default \
  --current-task "ship the next change" \
  --workspace-root . \
  --launch
```

Use `--dry-run` to inspect the exact `codex --cd ... <prompt>` command without
starting Codex or managed capture.

If a long-running `zaxy serve` process already owns the default embedded Kuzu
projection, CLI checkout retries with a per-process isolated projection under
`.eventloom/projections/checkout-<session>-<pid>.kuzu` and reports
`diagnostics.projection_fallback`. This keeps memory activation from silently
falling back to ordinary repo context after a resume or MCP process churn.

For a single first-run flow, use `zaxy init`. The happy path is deterministic
capture: local profile writing, MCP config rendering, observer hook config,
workspace genesis, hook heartbeat, doctor, and hook status. It does not proxy
model traffic and does not require a provider API key.
For Codex onboarding, the printed next steps include either the MCP install
command or the installed Codex config path, the path-stable
`zaxy activate codex ... --launch` session-start path, the `hook-event resume`
boundary command, the CLI checkout fallback for missing MCP tools, and optional
diagnostics. The activation launcher starts managed deterministic capture when
`.codex/zaxy-capture.json` is configured, so the normal first run has one
startup command instead of a separate capture command.
By default, `zaxy init` also installs a bounded `Zaxy Memory Activation` block in
`AGENTS.md`. The block is marker-managed, so rerunning init replaces only the
Zaxy section while preserving unrelated repo instructions. Pass
`--no-agent-instructions` when you need init to avoid writing model-visible repo
instructions.

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
it writes `.eventloom/`, `.env.local`, `.codex/zaxy-capture.json`, resolves
Codex MCP setup through `zaxy init --codex-mcp-install auto`, checks the
repo-local embedded projection posture, and records the first workspace
heartbeat. It installs deterministic capture configuration safely; start the
watcher when you want live local capture.

Use an explicit install mode only when you want to override the auto decision:

```bash
zaxy init --codex-mcp-install user
# or: zaxy init --codex-mcp-install command
```

`user` writes or verifies the user-level Codex MCP config. `command` always
prints the `codex mcp add` command and leaves Codex config untouched.

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

`local-codex` resolves Codex MCP setup through the same non-destructive auto
install policy used by bare `zaxy init`, writes the local profile, and writes
`.codex/zaxy-capture.json` for deterministic local capture. It does not generate
`.codex/hooks.json`: Codex parses that file as JSON, and Zaxy does not assume a
native Codex hook schema unless your Codex version documents one.
Start Codex through the printed `zaxy activate codex ... --launch` command to
load the bootstrap packet and ensure the managed watcher is running. The
explicit
`zaxy capture start --workspace .` command remains available for supervisors or
for starting capture during onboarding with `zaxy init --capture start`. It does
not enable packet capture. After startup, `zaxy doctor` should show
`capture_health: ok` once command, file-edit, tool-call, and transcript
observations have appeared.
It also reports `memory_activation`; if checkout is missing or stale, the
doctor action is a runnable `zaxy memory checkout ...` command for the affected
session.
`zaxy init` also prints a capture summary showing whether local capture is
configured, whether the watcher is running, the latest imported observation when
available, and the doctor capture-health result. The same block is available in
`zaxy init --json` as `capture`.

Generated output files are non-destructive by default. Pass `--force` only when
you intentionally want to replace generated config. `--infra check` reports the
selected local projection backend and Docker posture without starting
containers. Use `--infra start` when you explicitly want onboarding to prepare
the selected local runtime. The default embedded backend creates only the local
projection directory and lazy Kuzu database. For experimental pgGraph bootstrap,
install the optional extra and point Zaxy at a local pgGraph checkout so it can
run the extension installer instead of starting plain Postgres:

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
backend. With the default embedded backend, it opens the repo-local Kuzu
projection path and does not start Docker. With `PROJECTION_BACKEND=neo4j`, it
requires `pip install "zaxy-memory[neo4j]"` and checks
`bolt://localhost:7687`. Set `NEO4J_AUTO_START=true` only when you want Zaxy to
start a local `zaxy-neo4j` container automatically and wait for Bolt before
serving MCP tools. With `PROJECTION_BACKEND=pggraph`, it checks `PGGRAPH_DSN`;
local automatic startup requires `PGGRAPH_AUTO_START=true` and `PGGRAPH_REPO`
pointing at a pgGraph checkout containing `scripts/quickstart.sh`, because Zaxy
must install pgGraph into the local PostgreSQL container before graph traversal
is available.

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
unchanged. The embedded projection backend is the default, and the same refresh
command can target sidecar adapters when they are explicitly selected:

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

`zaxy status --eventloom-path .eventloom` and `zaxy hook-status --json` expose
the latest checkout, latest capture, activation efficiency, and structured
`memory_activation.remediations`. Text output prints the same remediation
command under "Memory next steps" so a stale or missing checkout is actionable
without reading Eventloom JSONL.

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

The full pytest command enforces the 92 percent coverage gate configured in
`pyproject.toml`. Integration tests require Docker Neo4j services. The release
smoke check verifies the package version, changelog entry, release workflow,
PyPI Trusted Publishing posture, and dependency-light LangGraph example without
contacting external services. The
beta readiness check verifies that the release smoke gate, release gate script,
clean-repo UAT script, first-run timing evidence, docs happy path, and
deterministic capture happy path are present. The release gate adds package artifact checks, documentation link
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
you need auditability. Do not treat any projection backend as the source of
truth. The graph projection is the reasoning layer. Eventloom is the durable
history. MCP is the interface agents call.
