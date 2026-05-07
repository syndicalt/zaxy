# Zaxy

**Event-sourced temporal knowledge graph fabric for AI agent memory.**

Zaxy replaces markdown files + vector DBs with a structured, replayable,
bi-temporal memory system built on Eventloom and Neo4j, with optional
Pathlight tracing.

## Quick Start

```bash
# Install the public package. The distribution is zaxy-memory; the import and
# CLI remain zaxy.
pip install zaxy-memory

# 1. Setup (generates .env and directories)
./scripts/setup.sh

# Production setup writes Docker secret files under ./secrets/.
./scripts/setup.sh --production
./scripts/generate-certs.sh .certs
docker compose -f docker-compose.prod.yml up -d

# 2. Start Neo4j + Zaxy MCP server
docker compose up -d

# 3. Verify
zaxy status
pytest

# Integration-only runs need --no-cov because the project-level coverage gate
# is intended for the full suite.
pytest -m integration --no-cov

# 4. Test drive (no agent needed)
python scripts/test_drive.py
```

## Architecture

```
Agent (LangGraph / Any MCP Client)
    |
    v
MCP Server — memory_append / memory_query / memory_replay / memory_invalidate
    |
    v
Eventloom (immutable JSONL log)  →  Hybrid Extraction  →  Neo4j (temporal KG)
    |                                                               |
    +—————— Optional Pathlight traces ———————————————→  Query Router
                                                              |
                                                    Hybrid Retrieval
                                                    (exact + BM25 + vector + traversal)
```

## Public Site and Documentation

- Public static site: `site/index.html`
- Getting started: `docs/getting-started.md`
- Architecture: `docs/architecture.md`
- Configuration: `docs/configuration.md`
- MCP interface: `docs/mcp.md`
- Eventloom contract: `docs/eventloom.md`
- Graph schema: `docs/graph-schema.md`
- Retrieval: `docs/retrieval.md`
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

# Live retrieval benchmark: markdown vs vector vs markdown+vector vs Zaxy
scripts/live-benchmark.sh --embedding-provider openai --workload statistical --subjects 100 --runs 1 --reset-graph

# Production deployment preflight
scripts/validate-deployment.sh --root .

# Build and validate Python release artifacts
scripts/build-dist.sh --root .

# Validate public site and documentation links
scripts/validate-docs.sh --root .

# Go-live release gate
scripts/release-check.sh --root .
```

The full suite must stay at or above 90% coverage before a sprint is complete.

## Release Publishing

The PyPI distribution name is `zaxy-memory` because `zaxy` is already occupied
on PyPI. Published releases build from GitHub Actions and upload to
<https://pypi.org/project/zaxy-memory/> using the `PYPI_API_TOKEN` repository
secret. The import package and console command remain `zaxy`.

## License

MIT
