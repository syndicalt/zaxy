# Zaxy

**Coordinator Memory for Agent Teams.**

Zaxy gives agent teams auditable, replayable, and coordinated memory: parent
missions, isolated worker sessions, cited findings, conflict review, approval
packets, and accepted merge-back into one durable project history. Under the
hood, it uses an Eventloom append-only source of truth and an embedded Kuzu graph
projection instead of flattening project memory into markdown files and vector
chunks.

The embedded Kuzu graph projection is the default local runtime.

The plain install uses embedded Kuzu. Install `zaxy-memory[neo4j]` only for the
optional Neo4j sidecar, and `zaxy-memory[pathlight]` only for Pathlight tracing.

## Quick Start

### Five-minute local smoke test

```bash
# Install the Zaxy CLI before generating MCP config. The distribution is
# zaxy-memory; the import package and console command remain zaxy.
pipx install zaxy-memory
# or: pip install zaxy-memory

# Initialize local memory, Codex MCP guidance, deterministic capture config,
# profile, genesis, heartbeat, and no-sidecar embedded graph posture.
zaxy init

# Prove the local Eventloom log and model bootstrap are readable.
zaxy memory log --eventloom-path .eventloom --limit 5
zaxy memory bootstrap --eventloom-path .eventloom
zaxy doctor --eventloom-path .eventloom
```

Run the single-agent memory example:

```bash
python examples/single_agent_memory.py
```

Your local data lives under `.eventloom/` as one append-only JSONL file per
session. Bare `zaxy init` now expands to the local embedded Codex path: it
prints the Codex MCP install command, writes `.codex/zaxy-capture.json`, writes
`.env.local`, checks the repo-local embedded graph posture, and ends with
copyable local verification commands. It does not start a background watcher
unless you pass `--capture start`.

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
Eventloom (immutable JSONL log)  →  Hybrid Extraction  →  Embedded Kuzu graph
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
- Zaxy 1.0.0 release: `docs/announcements/zaxy-v1.0.md`
- Zaxy 1.0.0 X article draft: `docs/announcements/zaxy-v1.0-x-article.md`
- Zaxy Coordinate/Collaborate demo media: `docs/media/zaxy-collaborate-demo.md`
- Zaxy Coordinate announcement: `docs/announcements/zaxy-coordinate.md`
- Zaxy 0.4.0 archive: `docs/announcements/zaxy-0.4.0.md`
- Coordinate roadmap: `docs/coordinate-roadmap.md`
- Why Zaxy: `docs/why-zaxy.md`
- Getting started: `docs/getting-started.md`
- First-run validation: `docs/first-run-validation.md`
- Public external verification request: `docs/external-validation.md`
- MCP quickstart: `docs/mcp-quickstart.md`
- Coordinate quickstart: `docs/coordinate-quickstart.md`
- Architecture: `docs/architecture.md`
- Configuration: `docs/configuration.md`
- MCP interface: `docs/mcp.md`
- Eventloom contract: `docs/eventloom.md`
- Graph schema: `docs/graph-schema.md`
- Retrieval: `docs/retrieval.md`
- Benchmarks: `docs/benchmarks.md`
- Benchmark contributions: `docs/benchmark-contributions.md`
- LLM packet analyzer: `docs/packet-analyzer.md`
- Embeddings: `docs/embeddings.md`
- Security: `docs/security.md`
- Operations and deployment: `docs/operations.md`, `docs/deployment.md`, `docs/runbook.md`
- Python API: `docs/api.md`
- API inventory: `docs/api-inventory.md`
- Stability commitment: `docs/stability-commitment.md`
- Migration guide: `docs/migration.md`
- v0.9 gate audit: `docs/v09-gate-audit.md`
- v1.0 gate audit: `docs/v10-gate-audit.md`
- Release validation checklist: `docs/release-validation-checklist.md`
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
| `src/zaxy/embedded_graph_store.py` | Embedded Kuzu projection store |
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

# Competitive retrieval benchmark harness
pytest tests/test_competitive_benchmarks.py --benchmark-only --no-cov

# Frozen live benchmark: markdown vs BM25 vs vector vs markdown+vector vs embedded Zaxy
# Uses deterministic hash embeddings and embedded projection by default.
scripts/live-benchmark.sh --workload frozen --runs 1 --reset-graph

# Representative benchmark suite: temporal memory + docs + transcripts + mixed context
scripts/live-benchmark.sh --workload suite --subjects 100 --documents 250 --sessions 50 --runs 1 --reset-graph

# LongMemEval-compatible memory benchmark and BM25 comparison
# Plain benchmark commands use the embedded projection backend by default.
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

# Current full-set LongMemEval-compatible checkout evidence:
# reports/benchmarks/longmemeval-500-current74-zaxyonly-gated-relative-temporal-anchor-embedded-reuse-20260604/
# Mean 0.940, Answer@5 0.906, citation coverage 1.000, R@1/R@5/R@10 0.906/1.000/1.000.

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
