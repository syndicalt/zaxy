# Zaxy: Event-Sourced Temporal Knowledge Graph Fabric

## Problem Statement

Markdown files + vector DBs are the dominant approach for agent persistent context, but they are fundamentally lossy and inefficient:

- **No relational reasoning**: Vector similarity can't do multi-hop traversal or follow causal chains.
- **No temporal awareness**: Can't answer "What was true then vs. now?" — facts overwrite each other silently.
- **Non-replayable**: Context is chunked and flattened; you can't reconstruct how the agent arrived at a decision.
- **Un-auditable**: No provenance chain for compliance or debugging.

## Architecture Decision Record

### ADR-1: Event-Sourced Foundation

**Decision**: Use Eventloom's append-only JSONL as the immutable source of truth.

**Rationale**:
- Hash-chain integrity (tamper-evident).
- Deterministic replay.
- Zero write overhead (local file append).
- Cross-process locking already solved.

**Trade-off**: Single-writer per file. For multi-agent distributed setups, shard by session or add a log aggregation layer later.

### ADR-2: Hybrid Extraction (Rule-Based + LLM Fallback)

**Decision**: Extract entities/relations from events using registered rule-based extractors first, LLM fallback only for unstructured events.

**Rationale**:
- Eventloom events are strongly typed (`goal.created`, `task.proposed`, etc.).
- Typed events map deterministically to graph schema.
- Reduces LLM extraction cost by 60–80%.
- Faster ingestion (<50ms vs 500ms–2s for LLM).

**Trade-off**: New event types require writing an extractor. This is intentional — it forces schema discipline.

### ADR-3: Direct Neo4j Cypher vs. Graphiti Abstraction

**Decision**: Use the official `neo4j` Python driver with custom Cypher rather than Graphiti's high-level `add_episode` API.

**Rationale**:
- Full control over bi-temporal schema (`valid_from`, `valid_to`).
- Our extraction engine already produces structured `ExtractedEntity`/`ExtractedEdge` objects.
- Graphiti's LLM-based extraction is redundant with our hybrid extractor.
- Graphiti's hybrid search (vector + BM25 + traversal) can be replicated with native Neo4j indexes.

**Trade-off**: We maintain more Cypher. Mitigated by keeping queries simple and tested.

### ADR-4: Hybrid Retrieval (Exact + Keyword + Traversal)

**Decision**: Query router fuses three strategies with configurable weights.

**Rationale**:
- **Exact**: Fast lookup when the query is an entity name.
- **Keyword/BM25**: Full-text for semantic similarity on names/summaries.
- **Traversal**: Multi-hop expansion from top keyword hits.
- Each covers blind spots of the others.

**Trade-off**: Fusion adds ~10–50ms latency. Acceptable for agent context quality.

### ADR-5: Pathlight for Observability (Not Storage)

**Decision**: Pathlight can trace every memory operation when enabled, but does not store context itself.

**Rationale**:
- Eventloom = durable history.
- Neo4j = structured reasoning layer.
- Pathlight = execution tracing + breakpoints + diff.
- Clean separation of concerns.

**Trade-off**: Extra network call per traced operation. Mitigated by async batching and optional disabling.

### ADR-6: MCP as Primary Interface

**Decision**: Expose memory via MCP tools (`memory_append`, `memory_query`, `memory_replay`, `memory_invalidate`).

**Rationale**:
- Framework-agnostic (LangGraph, CrewAI, AutoGen, Claude Desktop, etc.).
- Standardized schema discovery and type safety.
- One-click integration via `mcpServers` config.

**Trade-off**: Requires MCP client support. Major frameworks already have it (2025+).

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| Graph DB | Neo4j Community | 5.26+ |
| Graph Driver | neo4j (official) | 5.20+ |
| Validation | Pydantic | 2.7+ |
| MCP Server | mcp (official Python SDK) | 1.0+ |
| Observability | pathlight (Python SDK) | 0.1+ |
| CLI | typer | 0.12+ |
| Testing | pytest + pytest-asyncio + pytest-cov | 8.0+ |
| Lint/Format | ruff | 0.4+ |
| Types | mypy | 1.10+ |

## Directory Structure

```
zaxy/
├── pyproject.toml              # Project config, deps, tool settings
├── docker-compose.yml          # Neo4j + test services
├── AGENTS.md                   # This file
├── src/zaxy/
│   ├── __init__.py             # Public API exports
│   ├── __main__.py             # CLI entrypoint (`python -m zaxy`)
│   ├── core.py                 # MemoryFabric orchestrator
│   ├── event.py                # Eventloom JSONL I/O + hash chain
│   ├── extract.py              # Hybrid extraction engine + registry
│   ├── graph.py                # Neo4j bi-temporal wrapper
│   ├── query.py                # Hybrid retrieval router
│   ├── mcp_server.py           # MCP stdio/sse server
│   └── trace.py                # Pathlight observability hooks
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_event.py           # Event log I/O + integrity
│   ├── test_extract.py         # Rule-based extractors
│   ├── test_graph.py           # Neo4j operations (mock + integration)
│   ├── test_query.py           # Query routing + fusion
│   ├── test_mcp.py             # MCP protocol compliance
│   └── test_trace.py           # Pathlight span emission
├── examples/
│   └── langgraph_memory.py     # Full LangGraph integration demo
└── scripts/
    └── setup_neo4j_indexes.cypher  # Manual index setup
```

## Development Workflow

1. **Write the test first** (Karpathy rule). Every public function must have a test before the implementation is considered complete.
2. **Unit tests mock external deps** (Neo4j, Pathlight, filesystem).
3. **Integration tests use Docker** (marked `@pytest.mark.integration`).
4. **Coverage gate: ≥90%** (enforced by CI).
5. **Lint/format with ruff**, type-check with mypy.

## Testing Strategy

| Test Type | Scope | External Deps | Marker |
|-----------|-------|---------------|--------|
| Unit | Single function/class | Mocked | `unit` (default) |
| Integration | Cross-module + real DB | Neo4j Docker | `integration` |
| E2E | Full agent run | Full Docker stack | `e2e` (future) |

Run unit tests: `pytest`
Run integration tests: `pytest -m integration`
Run with coverage: `pytest --cov` (default in pyproject.toml)

## Integration Points

### Eventloom (Bottom Layer)
- **Input**: Zaxy reads `.eventloom/*.jsonl` files.
- **Output**: Zaxy appends projection metadata back to Eventloom (optional).
- **Contract**: `Event` Pydantic model matches Eventloom JSONL schema.

### Neo4j (Core Memory)
- **Bolt URI**: `bolt://localhost:7687`
- **Auth**: `neo4j/testpassword` (local dev)
- **Schema**: `Entity(name, entity_type, valid_from, valid_to, ...)` + `RELATES(relation_type, valid_from, valid_to)`
- **Indexes**: Vector index (embedding), Fulltext index (BM25)

### Pathlight (Top Layer)
- **Collector**: `http://localhost:4100`
- **Dashboard**: `http://localhost:3100`
- **Traced Operations**: `append`, `query`, `replay`, `invalidate`
- **Eventloom Panel**: Pathlight renders Eventloom exports natively.

### MCP (Interface Layer)
- **Transport**: stdio (default) or SSE
- **Tools**:
  - `memory_append(event_type, actor, payload, thread?)`
  - `memory_query(query, temporal_filter?, limit?)`
  - `memory_replay(session_id, from_seq?)`
  - `memory_invalidate(entity_name, entity_type, invalid_at)`

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Event append | <50ms | Local JSONL write + lock |
| Rule-based extraction | <10ms | Pure Python dict mapping |
| Neo4j upsert | <100ms | MERGE + index lookup |
| Hybrid query | <200ms | Parallel exact + keyword + traversal |
| Total context retrieval | <300ms | End-to-end |
| Token reduction vs. chunk RAG | 70–90% | Structured paths vs. raw text |

## Current Status

- [x] Project scaffold (pyproject.toml, docker-compose.yml)
- [x] Eventloom JSONL I/O with hash-chain integrity
- [x] Hybrid extraction engine with rule registry
- [x] Neo4j graph store (schema, upsert, retrieval, invalidation)
- [x] Hybrid query router (exact + keyword + traversal)
- [x] Pathlight tracing integration
- [x] MCP server implementation
- [x] `MemoryFabric` orchestrator wiring
- [x] CLI entrypoint (`zaxy serve`, `zaxy replay`, `zaxy compact`)
- [x] LangGraph adapter example
- [x] CI/CD (GitHub Actions)
- [x] Operational runbooks
- [x] Configuration management (pydantic-settings, env vars)
- [x] Docker containerization (Dockerfile + compose + SSE production command)
- [x] Structured logging (console + JSON)
- [x] Graceful shutdown (SIGTERM/SIGINT handling)
- [x] Production secrets management (Docker secrets + `*_FILE` config)
- [x] TLS for Neo4j (generated certs + TLS compose service + integration test)
- [x] Multi-agent session sharding (SessionManager + MemoryFabric/MCP wiring + graph session isolation)
- [x] Prometheus metrics
- [x] Vector index and vector similarity search in query router
- [x] SSE transport for MCP daemon mode
- [x] Embedding generation pipeline (deterministic local provider + entity/query vectors)
- [x] True temporal versioning for reasserted facts and multi-version entity state
- [x] Remote MCP security (SSE bearer auth + per-client session scopes)
- [x] Operational backup/restore/log-rotation scripts backed by tests
- [x] Competitive benchmark suite vs. flat JSONL context baseline
- [x] Hosted embedding provider adapter with secret-managed credentials
- [x] Remote deployment environment validation for MCP/SSE
- [x] Go-live readiness checklist and release gate
- [x] Release packaging and versioned distribution artifacts
- [x] Public static site and expanded documentation set
- [x] Eventloom provenance citations on graph-backed retrieval results
- [x] Append-time secret redaction and payload classification
- [x] MMR diversity and explainable score metadata in query router
- [x] Filesystem document ingestion with source path and line citations
- [x] Sanitized transcript ingestion and replay-to-context assembly API
- [x] Query expansion and temporal-aware retrieval scoring policies
- [x] Configurable scoring profiles and local lexical reranker provider
- [x] Hosted OpenAI-compatible and local HTTP reranker providers
- [x] Graceful degradation for graph, embedding, vector, and reranker outages
- [x] Context lifecycle hooks for after-turn assembly, handoff bundles, and subagent cleanup
- [x] OIDC/JWKS remote MCP authentication for public multi-tenant deployments
- [x] Degraded-mode Prometheus metrics and alerting guidance
- [x] Extractor authoring templates and auditable schema migration tooling
- [x] Frozen benchmark workload fingerprints and external comparison disclosures

## Metrics

| Metric | Value |
|--------|-------|
| Tests | 341 passed |
| Coverage | 92.46% |
| Lint | ruff clean |
| Types | mypy clean |
| Python versions | 3.11, 3.12, 3.13 |

## Next Steps

1. Run and publish OpenAI embedding results for the frozen benchmark workload.
2. Add lifecycle import/export adapters for common agent frameworks.
3. Add remote MCP rate limiting and audit event export.
4. Add local-first embedding/reranker setup helpers.
5. Add extractor schema-pack examples for common agent event taxonomies.
