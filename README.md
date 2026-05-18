# Zaxy

**Event-sourced temporal knowledge graph fabric for AI agent memory.**

Zaxy replaces markdown files + vector DBs with a structured, replayable,
bi-temporal memory system built on Eventloom and Neo4j, with optional
Pathlight tracing.

## Quick Start

### Five-minute local smoke test

```bash
# Install the Zaxy CLI before generating MCP config. The distribution is
# zaxy-memory; the import package and console command remain zaxy.
pipx install zaxy-memory
# or: pip install zaxy-memory

# Initialize local memory, MCP config guidance, profile, genesis, and heartbeat.
# This checks graph infrastructure posture without starting containers.
zaxy init . --domain my-project --preset local-codex --capture start --infra check

# Prove the local Eventloom log and model bootstrap are readable.
zaxy memory log --eventloom-path .eventloom --session-id my-project-default --limit 5
zaxy memory bootstrap --eventloom-path .eventloom --session-id my-project-default
zaxy doctor --eventloom-path .eventloom
```

Your local data lives under `.eventloom/` as one append-only JSONL file per
session. `zaxy init` prints the MCP config or install command for your selected
client, shows the selected graph backend posture, and ends with copyable local
verification commands.

For Claude Code instead of Codex:

```bash
zaxy init . --domain my-project --preset local-claude --infra check
```

For Hermes Agent:

```bash
zaxy ide-config hermes --install
```

For repository development, use `pip install -e ".[dev]"`, `./scripts/setup.sh`,
and `docker compose up -d`. Production setup writes Docker secret files under
`./secrets/`; see [docs/deployment.md](docs/deployment.md).

## Architecture

```
Agent (LangGraph / Any MCP Client)
    |
    v
MCP Server — memory_append / memory_query / memory_feedback / memory_replay / memory_invalidate
    |
    v
Eventloom (immutable JSONL log)  →  Hybrid Extraction  →  Neo4j (temporal KG)
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
- Architecture: `docs/architecture.md`
- Configuration: `docs/configuration.md`
- MCP interface: `docs/mcp.md`
- Eventloom contract: `docs/eventloom.md`
- Graph schema: `docs/graph-schema.md`
- Retrieval: `docs/retrieval.md`
- Benchmarks: `docs/benchmarks.md`
- LLM packet analyzer: `docs/packet-analyzer.md`
- Embeddings: `docs/embeddings.md`
- Security: `docs/security.md`
- Operations and deployment: `docs/operations.md`, `docs/deployment.md`, `docs/runbook.md`
- Python API: `docs/api.md`

## Key Features

- **Immutable audit trail**: Eventloom append-only JSONL with SHA-256 hash chains.
- **Bi-temporal graph**: Facts have validity windows (`valid_from`, `valid_to`).
- **Hybrid extraction**: Rule-based for typed events (60–80% cost reduction), LLM fallback.
- **Hybrid retrieval**: Exact + keyword + vector + graph traversal with configurable fusion weights.
- **Session sharding**: One Eventloom log per agent/session, with a shared graph.
- **MCP-native**: Drop-in memory for any MCP-compatible agent framework over stdio or SSE.
- **Observable**: Optional Pathlight traces, breakpoints, and diff support.
- **Hardened local defaults**: bounded MCP inputs, safe session IDs, localhost-bound Neo4j ports, and optional admin token support for replay/invalidation.

## Project Structure

| File | Purpose |
|------|---------|
| `src/zaxy/event.py` | Eventloom JSONL I/O + hash chain integrity |
| `src/zaxy/extract.py` | Hybrid extraction engine + rule registry |
| `src/zaxy/graph.py` | Neo4j bi-temporal wrapper |
| `src/zaxy/query.py` | Hybrid retrieval router |
| `src/zaxy/mcp_server.py` | MCP stdio/SSE server |
| `src/zaxy/trace.py` | Pathlight observability hooks |
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
- **Unit tests** mock Neo4j/Pathlight. **Integration tests** use Docker.
- **Coverage gate: ≥90%** enforced by CI.
- **Lint/format**: `ruff`. **Types**: `mypy`.

```bash
# Run full suite with coverage gate
pytest

# Run integration tests (requires Docker)
docker compose up -d neo4j-test
./scripts/generate-certs.sh .certs
docker compose up -d neo4j-tls
pytest -m integration --no-cov

# Lint and type-check
ruff check src tests
mypy src

# Competitive retrieval benchmark harness
pytest tests/test_competitive_benchmarks.py --benchmark-only --no-cov

# Frozen live benchmark: markdown vs BM25 vs vector vs markdown+vector vs Zaxy
scripts/live-benchmark.sh --embedding-provider openai --workload frozen --runs 1 --reset-graph

# Representative benchmark suite: temporal memory + docs + transcripts + mixed context
scripts/live-benchmark.sh --embedding-provider openai --workload suite --subjects 100 --documents 250 --sessions 50 --runs 1 --reset-graph

# LongMemEval-compatible memory benchmark and BM25 comparison
zaxy benchmark --embedding-provider hash --workload longmemeval \
  --dataset .cache/zaxy/benchmarks/longmemeval_oracle.json \
  --questions 100 --runs 1 --limit 10 --zaxy-backend checkout \
  --baseline-backends bm25 --embedding-cache .cache/zaxy/longmemeval-embeddings.json
zaxy benchmark --output-dir reports/benchmarks/longmemeval-100-comparison \
  --embedding-provider hash --workload longmemeval \
  --dataset .cache/zaxy/benchmarks/longmemeval_oracle.json \
  --questions 100 --runs 1 --limit 5 --baseline-backends bm25 \
  --zaxy-backend checkout --reuse-projection \
  --embedding-cache .cache/zaxy/longmemeval-embeddings.json

# Production deployment preflight
scripts/validate-deployment.sh --root .

# Build and validate Python release artifacts
scripts/build-dist.sh --root .

# Verify local release metadata and PyPI Trusted Publishing configuration
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

The full suite must stay at or above 90% coverage before a sprint is complete.

## Release Publishing

The PyPI distribution name is `zaxy-memory` because `zaxy` is already occupied
on PyPI. Published releases build from GitHub Actions and upload to
<https://pypi.org/project/zaxy-memory/> using PyPI Trusted Publishing with
GitHub OIDC. The import package and console command remain `zaxy`.

Before publishing, run `zaxy doctor --release-smoke` to verify the package
version, changelog entry, release workflow, and tokenless publishing posture.

## License

MIT
