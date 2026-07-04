# Zaxy

**Production memory for agent teams that need receipts.**

<!-- mcp-name: io.github.syndicalt/zaxy -->

Zaxy turns agent context into an auditable project memory fabric. It captures
parent missions, worker sessions, tool observations, cited findings, conflict
review, approval packets, and accepted merge-back into one durable history that
can be queried, replayed, and inspected.

Under the hood, Zaxy uses Eventloom append-only JSONL as the source of truth and
an embedded LadybugDB graph projection for local reasoning. It is built for agents
that need to remember what happened, cite where it came from, and avoid turning
project state into a pile of markdown files and vector chunks.

The embedded LadybugDB graph projection is the default local runtime.

The plain install uses embedded LadybugDB. Install `zaxy-memory[neo4j]` only for the
optional Neo4j sidecar, and `zaxy-memory[pathlight]` only for Pathlight tracing.

## Why It Matters

- **Auditable memory**: every accepted fact can point back to Eventloom history.
- **Agent-team coordination**: parent and worker sessions stay isolated until
  findings are reviewed and merged.
- **Local-first runtime**: the default path uses embedded LadybugDB, no Neo4j sidecar.
- **MCP-native integration**: Codex, Claude Code, Cursor, VS Code, Hermes Agent,
  LangGraph, CrewAI, and AutoGen can use the same memory interface.

> **Benchmark claims withdrawn (2026-07-03).** The prior LongMemEval numbers
> were withdrawn: they were produced in *oracle* mode (mean ~1.9 candidate
> sessions per question, so Recall@5/citation-coverage were ~1.0 by
> construction, not by retrieval) and the preference-question scores rested on a
> hardcoded answer table that has since been removed. Zaxy does not currently
> publish a LongMemEval score. A real, full-haystack LongMemEval run is planned;
> until it lands, treat the earlier `0.956`/`0.910`/`1.000` figures as retracted.
> The Harvey LAB claim is pending the same audit.

## Quick Start

### Install, init, verify

```bash
pipx install zaxy-memory
zaxy init
zaxy memory log --eventloom-path .eventloom --limit 5
zaxy memory bootstrap --eventloom-path .eventloom
zaxy doctor --eventloom-path .eventloom
```

The PyPI distribution is `zaxy-memory`; the import package and console command
are still `zaxy`. Bare `zaxy init` sets up the local embedded graph posture,
repo-local profile, deterministic capture config, genesis event, heartbeat, and
MCP guidance. For Codex, the printed activation launcher starts the managed
capture watcher when the local capture config is present; pass `--capture start`
only when you want init itself to start the watcher before opening Codex. The
default human output is compact and action-first; add `--verbose` when you need
the full setup diagnostics, optional checks, fallback commands, resume guidance,
and notes.
For automation, `zaxy init --json` keeps the raw onboarding fields and adds
`setup.status`, `setup.issues`, `setup.pending`, `readiness.status`,
`readiness.reasons`, `readiness.actions`, and structured
`readiness.action_items` for both commands and non-command review tasks. Each
structured action carries `label`, `command`, original `source`, and `hints`
for compact-output tips such as activation `<task>` replacement and path-stable
command guidance. Installers can render those tips without parsing prose. It also
includes `setup.summary`, `readiness.summary`,
`readiness.required_action_count`, and `readiness.reason_count`, so client UIs
can render compact status without parsing human output. It also
separates `readiness.blocking_diagnostics` from
`readiness.non_blocking_diagnostics` so scripts can distinguish setup
completion, required actions, and advisory doctor warnings before relying on
live memory.

For Codex, `zaxy init --codex-mcp-install auto` is the default. It writes or
reuses the user-level Codex MCP config when that can be done without replacing
an existing `zaxy` server entry. If no safe config target exists, it prints the
copyable `codex mcp add` command. If an existing `zaxy` entry differs, it asks
you to review that config before replacing it because Codex can silently replace
servers with the same name. Use an explicit mode when you need to force one side
of that decision after review:

```bash
zaxy init --codex-mcp-install user
# or: zaxy init --codex-mcp-install command
```

Both Codex paths keep the server workspace-neutral. After init, start or
restart Codex through the printed `zaxy activate codex ... --launch` command so
the MCP server list and Zaxy activation packet are loaded together. The printed
command includes explicit `--eventloom-path` and `--workspace-root` values, so
it still targets the initialized repo when copied from another shell.

Run the single-agent memory example:

```bash
python examples/single_agent_memory.py
```

Your local data lives under `.eventloom/` as one append-only JSONL file per
session.

For Claude Code instead of Codex:

```bash
zaxy init . --domain my-project --preset local-claude --infra check
```

For Hermes Agent:

```bash
zaxy ide-config hermes --install
```

For repository development, use `pip install -e ".[dev]"`, `./scripts/setup.sh`,
and `zaxy status`. Start Docker sidecars only for integration tests or explicit
backend comparisons. Production setup writes Docker secret files under
`./secrets/`; see [docs/deployment.md](docs/deployment.md).

## Architecture

```
Agent (LangGraph / Any MCP Client)
    |
    v
MCP Server — memory_append / memory_query / memory_feedback / memory_replay / memory_invalidate
    |
    v
Eventloom (immutable JSONL log)  →  Hybrid Extraction  →  Embedded LadybugDB graph
    |                                                               |
    +—————— Optional Pathlight traces ———————————————→  Query Router
                                                              |
                                                    Hybrid Retrieval
                                                    (exact + BM25 + vector + traversal)
```

Zaxy also includes an observe-only OpenAI-compatible packet analyzer for model
call provenance. It forwards packets to one configured upstream endpoint and
records `llm.packet.completed` events to Eventloom without acting as a router.
See [LLM Packet Analyzer](docs/packet-analyzer.md).

## Public Site and Documentation

- Public static site: `site/index.html`
- Why Zaxy: `docs/why-zaxy.md`
- Getting started: `docs/getting-started.md`
- MCP quickstart: `docs/mcp-quickstart.md`
- Architecture: `docs/architecture.md`
- Configuration: `docs/configuration.md`
- MCP interface: `docs/mcp.md`
- Memory export contract: `docs/export-contract.md`
- Eventloom contract: `docs/eventloom.md`
- Graph schema: `docs/graph-schema.md`
- Retrieval: `docs/retrieval.md`
- Benchmarks: `docs/benchmarks.md`
- LLM packet analyzer: `docs/packet-analyzer.md`
- Embeddings: `docs/embeddings.md`
- Security: `docs/security.md`
- Operations and deployment: `docs/operations.md`, `docs/deployment.md`, `docs/runbook.md`
- Python API: `docs/api.md`
- Stability commitment: `docs/stability-commitment.md`
- Migration guide: `docs/migration.md`
- Archived benchmark iteration notes, release drafts, and research notes live
  under `docs/archive/`, `docs/announcements/`, and `docs/research/`.
- Contributing: `CONTRIBUTING.md`

## Key Features

- **Immutable audit trail**: Eventloom append-only JSONL with SHA-256 hash chains.
- **Bi-temporal graph**: Facts have validity windows (`valid_from`, `valid_to`).
- **Hybrid extraction**: Rule-based for typed events (60–80% cost reduction), LLM fallback.
- **Hybrid retrieval**: Exact + keyword + vector + graph traversal with configurable fusion weights.
- **Session sharding**: One Eventloom log per agent/session, with a shared graph.
- **MCP-native**: Drop-in memory for any MCP-compatible agent framework over stdio or SSE.
- **Observable**: Optional Pathlight traces, breakpoints, and diff support via `zaxy-memory[pathlight]`.
- **Hardened local defaults**: bounded MCP inputs, safe session IDs, no-sidecar embedded graph projection, and optional admin token support for replay/invalidation.

## Project Structure

| File | Purpose |
|------|---------|
| `src/zaxy/event.py` | Eventloom JSONL I/O + hash chain integrity |
| `src/zaxy/extract.py` | Hybrid extraction engine + rule registry |
| `src/zaxy/embedded_graph_store.py` | Embedded LadybugDB projection store |
| `src/zaxy/graph.py` | Optional Neo4j bi-temporal wrapper via `zaxy-memory[neo4j]` |
| `src/zaxy/query.py` | Hybrid retrieval router |
| `src/zaxy/mcp_server.py` | MCP stdio/SSE server |
| `src/zaxy/trace.py` | Optional Pathlight observability hooks |
| `src/zaxy/core.py` | MemoryFabric orchestrator |
| `src/zaxy/session.py` | Per-session Eventloom log manager |
| `src/zaxy/security.py` | Shared validation and input bounds |
| `src/zaxy/__main__.py` | CLI (`zaxy serve`, `zaxy replay`, etc.) |

## Production Secrets

Zaxy supports Docker/Kubernetes-style secret files for sensitive settings:

| Variable | Secret-file variant |
|----------|---------------------|
| `NEO4J_PASSWORD` | `NEO4J_PASSWORD_FILE` |
| `MCP_ADMIN_TOKEN` | `MCP_ADMIN_TOKEN_FILE` |
| `PATHLIGHT_ACCESS_TOKEN` | `PATHLIGHT_ACCESS_TOKEN_FILE` |

Direct environment variables take precedence over their `*_FILE` variants.
Use `docker-compose.prod.yml` as the production compose baseline.

## Development

- **Tests first** (Karpathy rule). Every public function has a test.
- **Unit tests** mock external services. **Integration tests** use Docker for optional sidecar backends.
- **Coverage gate: ≥92%** enforced by CI.
- **Lint/format**: `ruff`. **Types**: `mypy`.

```bash
# Run full suite with coverage gate
pytest

# Run integration tests (requires Docker)
./scripts/generate-certs.sh .certs
docker compose --profile integration up -d neo4j-test neo4j-tls
pytest -m integration --no-cov

# Lint and type-check
ruff check src tests
mypy src

# LongMemEval benchmark numbers are WITHDRAWN (see the note at the top of this
# README). The `--dataset .cache/.../longmemeval_oracle.json` path only exercises
# the answer step over pre-selected gold sessions (oracle mode) and does NOT
# measure retrieval on the full LongMemEval haystack; do not publish figures from
# it as a LongMemEval score. A real full-haystack harness is TBD.

# Harvey LAB external memory-ablation comparison
# Consumes externally generated Harvey normalized-result artifacts for Zaxy;
# does not reuse LongMemEval statistics as legal-agent benchmark evidence.
# Current full external Harvey LAB evidence:
# reports/benchmarks/harvey-lab-memory-ablation/publishable-statistics.md
# reports/benchmarks/harvey-lab-memory-ablation/harvey-lab-benchmark.json
# 10/10 tasks, mean criterion pass rate 0.788, +0.184 vs regular/no-memory,
# +0.081 vs article-best task rows, 9/10 wins vs article-best rows.

# Production deployment preflight
scripts/validate-deployment.sh --root .

# Build and validate Python release artifacts
scripts/build-dist.sh --root .

# Verify local release metadata, PyPI Trusted Publishing, and LangGraph smoke
zaxy doctor --release-smoke

# Validate public site and documentation links
scripts/validate-docs.sh --root .

# Clean-repo beta UAT: install into a throwaway workspace and verify init,
# bootstrap, deterministic capture, doctor, and memory checkout.
scripts/beta-uat.sh

# Summarize beta readiness gates without external services.
zaxy doctor --beta-readiness

# Go-live release gate
scripts/release-check.sh --root .
```

The full suite must stay at or above 92% coverage before a sprint is complete.

## Release Publishing

The PyPI distribution name is `zaxy-memory` because `zaxy` is already occupied
on PyPI. Published releases build from GitHub Actions and upload to
<https://pypi.org/project/zaxy-memory/> using PyPI Trusted Publishing with
GitHub OIDC. The import package and console command remain `zaxy`.

Before publishing, run `zaxy doctor --release-smoke` to verify the package
version, changelog entry, release workflow, tokenless publishing posture, and
dependency-light LangGraph example.

## License

MIT
