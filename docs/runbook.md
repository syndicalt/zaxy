# Zaxy Operational Runbook

## Architecture Overview

Zaxy is an event-sourced temporal knowledge graph fabric for AI agent memory.
It consists of three layers:

1. **Eventloom** (bottom): Immutable append-only JSONL logs with SHA-256 hash chains.
2. **Embedded LadybugDB projection** (default): Repo-local bi-temporal graph projection with entity/relationship validity windows.
3. **Pathlight** (optional top layer): Observability, tracing, and debugging dashboard.

Neo4j and pgGraph are optional sidecar projection backends. Use them only when
you explicitly need same-harness control runs or existing infrastructure
interop. Install Neo4j support with `zaxy-memory[neo4j]`.
Install Pathlight tracing support with `zaxy-memory[pathlight]` before setting
`PATHLIGHT_ENABLED=true`.

For provider-neutral trace inspection, export replay-derived spans and edges
from Eventloom without enabling Pathlight:

```bash
zaxy trace export --eventloom-path .eventloom --json
zaxy trace export --eventloom-path .eventloom --format jsonl --output trace.jsonl
```

The `zaxy.trace.v0.8` payload includes per-session integrity metadata plus
spans and edges that correlate missions, memory checkout, model calls, tool
calls, findings, reviews, promotions, and handoffs. Use `--format jsonl` when a
local pipeline or external tracing adapter expects append-only ingestion
records.

## Quick Start

```bash
# Install Zaxy
pip install -e ".[dev]"

# Initialize local memory with the embedded graph profile
python -m zaxy init

# The generated local profile should select the no-sidecar backend
grep -q "PROJECTION_BACKEND=embedded" .env.local

# Verify local Eventloom, capture, and embedded graph posture
python -m zaxy doctor --eventloom-path .eventloom

# Run tests
pytest

# Start MCP server
python -m zaxy serve

# Check local onboarding prerequisites
python -m zaxy doctor

# Emit machine-readable setup diagnostics
python -m zaxy doctor --json
```

Beyond onboarding posture, `zaxy doctor` verifies the active session log's
hash chain (full verify on small logs, bounded tail verify on large ones),
compares embedded projection state against the event log signature, confirms
the embedding provider builds and agrees with `EMBEDDING_DIMENSION`, reports
mixed embedding-version corpora (remediated with `zaxy memory re-embed`), and
reports vector index cache budget headroom. Every failing check prints a
one-line remediation.

```bash

# Start MCP over SSE for daemon mode
python -m zaxy serve --transport sse --port 8080
```

## Daily Operations

### Health Checks

```bash
# Check all services
python -m zaxy status

# Or manually:
python -m zaxy memory status --eventloom-path .eventloom --graph

# Only when PATHLIGHT_ENABLED=true:
curl http://localhost:4100/health  # Pathlight collector
curl http://localhost:3100  # Pathlight dashboard
```

### Event Log Inspection

```bash
# Replay a session
python -m zaxy replay .eventloom/work.jsonl

# Replay from a specific point
python -m zaxy replay .eventloom/work.jsonl --from-seq 42

# Replay a bounded window in a long-running session
python -m zaxy replay .eventloom/work.jsonl --from-seq 42 --to-seq 80

# Export as JSON
python -m zaxy replay .eventloom/work.jsonl --json

# Write a standalone HTML viewer for one log or an Eventloom directory
python -m zaxy viewer .eventloom --output eventloom-viewer.html

# Rebuild the selected projection after extractor changes
python -m zaxy reproject .eventloom/default.jsonl --session-id default

# Audit identity and citation safety before compacting
python -m zaxy compact .eventloom/work.jsonl --audit

# Export a machine-readable audit report
python -m zaxy compact .eventloom/work.jsonl --audit --json

# Store a source-backed medoid projection without rewriting the log
python -m zaxy compact .eventloom/work.jsonl --projection-output .eventloom/work.compaction.json

# Store a bounded exemplar projection for high-spread clusters
python -m zaxy compact .eventloom/work.jsonl --projection-output .eventloom/work.compaction.json --strategy exemplar --max-records 5

# Rewrite compaction appends compaction.completed to the output log.
# Audit and projection-only modes leave the source log unchanged.

# Projections under the Eventloom directory are auto-discovered
python - <<'PY'
from zaxy import MemoryFabric

fabric = MemoryFabric(eventloom_path=".eventloom")
PY

# Explicit projection paths are still supported for artifacts stored elsewhere
python - <<'PY'
from zaxy import MemoryFabric

fabric = MemoryFabric(projection_paths=["/secure/projections/work.compaction.json"])
PY

# Compact old logs
python -m zaxy compact .eventloom/work.jsonl --snapshot-every 10000
```

### Memory Queries (via MCP)

When the MCP server is running, any MCP client can:

```json
{
  "tool": "memory_append",
  "arguments": {
    "event_type": "goal.created",
    "actor": "user",
    "payload": {"title": "Ship MVP"}
  }
}
```

```json
{
  "tool": "memory_query",
  "arguments": {
    "query": "What are our goals?",
    "temporal_filter": "2024-06-01T00:00:00Z",
    "limit": 5
  }
}
```

## Backup & Recovery

### Critical Data

| Data | Location | Backup Priority |
|------|----------|-----------------|
| Eventloom logs | `.eventloom/*.jsonl` | **Critical** — immutable source of truth |
| Embedded LadybugDB projection | `.eventloom/projections/embedded.kuzu` | Medium — can be rebuilt from Eventloom |
| Optional sidecar projection | Backend-specific volume or service | Medium — can be rebuilt from Eventloom |
| Pathlight traces | Pathlight deployment volume, if enabled | Medium — observability only |

### Backup Procedures

```bash
scripts/backup.sh \
  --root . \
  --output-dir /backups/zaxy \
  --name "zaxy-$(date -u +%Y%m%dT%H%M%SZ)"
```

This archives `.eventloom/` and non-secret operational docs/config templates,
excludes `secrets/` and `.certs/`, and writes a `.sha256` manifest next to the
archive. Graph projections can be rebuilt from Eventloom; back up optional
sidecar stores only when fast point-in-time restore matters more than
minimizing backup surface.

### Recovery Procedures

```bash
scripts/restore.sh \
  --archive /backups/zaxy/zaxy-20260506T120000Z.tar.gz \
  --manifest /backups/zaxy/zaxy-20260506T120000Z.sha256 \
  --target /srv/zaxy
```

Restore validates the checksum before extraction and refuses to overwrite an
existing target `.eventloom/` unless `--force` is provided.

## Monitoring & Alerting

### Key Metrics

| Metric | Target | Alert If |
|--------|--------|----------|
| Event append latency | <50ms | >100ms |
| Graph upsert latency | <100ms | >200ms |
| Hybrid query latency | <200ms | >500ms |
| Event log size | <10GB | >50GB |
| Projection disk usage | <80% | >90% |

### Graph Projection Monitoring

For the default embedded backend:

```bash
python -m zaxy memory status --eventloom-path .eventloom --graph --json
python -m zaxy memory inferred-status --session-id default --json
```

For the optional Neo4j sidecar:

```cypher
// Check database size
CALL dbms.database.state("neo4j") YIELD status;

// Check index status
SHOW INDEXES;

// Check constraint status
SHOW CONSTRAINTS;

// Entity count
MATCH (e:Entity) RETURN count(e) AS entities;

// Provenance backbone count
MATCH (:Session)-[:HAS_EVENT]->(:Event) RETURN count(*) AS projected_events;

// Relationship count
MATCH ()-[r:RELATES]->() RETURN count(r) AS relations;

// Typed relationship label sample
MATCH p=()-[:CALLS_SYMBOL|DEFINES_SYMBOL|PROJECTED_LLM_PACKET]->() RETURN p LIMIT 25;

// Temporal validity check — entities without valid_to
MATCH (e:Entity) WHERE e.valid_to IS NULL RETURN count(e) AS active_entities;
```

### Log Rotation

Eventloom logs grow indefinitely. Set up rotation:

```bash
scripts/rotate-logs.sh \
  --log .eventloom/default.jsonl \
  --archive-dir .eventloom/archive \
  --name "default-$(date -u +%Y%m%dT%H%M%SZ)"
```

Rotation copies the active JSONL file into an archive, writes a checksum
manifest, then truncates the active file only after archive creation succeeds.
Verify rotated logs with `zaxy replay .eventloom/archive/<name>.jsonl`.

## Troubleshooting

### "Agent is hallucinating / using stale context"

1. **Check Eventloom**: Verify the event was actually recorded.
   ```bash
   python -m zaxy replay .eventloom/work.jsonl --from-seq N
   ```

2. **Check graph temporal validity**:
   ```cypher
   MATCH (e:Entity {name: "X"})
   RETURN e.valid_from, e.valid_to, e.entity_type
   ```

3. **Pathlight trace** (if enabled): Inspect the query/result metadata and operation timing.

### "Hash chain verification failed"

1. Identify the broken event:
   ```python
   from zaxy.event import EventLog
   log = EventLog(".eventloom/work.jsonl")
   report = log.verify()
   print(f"Broken at seq: {report.broken_at_seq}")
   ```

2. If tampered: Restore from backup. Eventloom logs are append-only and should never be modified.

3. If corrupted disk: Check filesystem integrity (`fsck`, SMART tests).

### "Graph projection unavailable"

1. For the default embedded backend, check the repo-local projection path:
   ```bash
   python -m zaxy status --projection-backend embedded
   python -m zaxy memory status --eventloom-path .eventloom --graph --projection-backend embedded
   ```

2. Rebuild from Eventloom if the projection is stale or missing:
   ```bash
   python -m zaxy reproject .eventloom/default.jsonl --session-id default --projection-backend embedded
   ```

3. For the optional Neo4j sidecar, install the extra and check container status:

   ```bash
   pip install "zaxy-memory[neo4j]"
   docker compose ps neo4j
   docker compose logs neo4j
   ```

   Check memory and ports only when that sidecar is selected:

   ```bash
   docker compose exec neo4j neo4j-admin memrec
   netstat -tlnp | grep 7687
   ```

### Performance Degradation

1. **Query slow?** Check projection status and benchmark the selected backend:
   ```bash
   python -m zaxy memory status --eventloom-path .eventloom --graph --json
   ```

2. **Optional Neo4j query slow?** Check Neo4j query plan:
   ```cypher
   PROFILE MATCH (e:Entity {name: "X"}) RETURN e;
   ```

3. **Event append slow?** Check disk I/O:
   ```bash
   iostat -x 1
   ```

4. **Graph upsert slow?** Check for missing indexes in the selected backend:
   ```cypher
   SHOW INDEXES;
   ```

## Scaling Considerations

### Current

- One Eventloom file per session/agent
- Repo-local embedded LadybugDB projection by default
- Optional Pathlight tracing
- Embedded read-index warmup and hot caches for current entities, exact lookup,
  keyword, vector, traversal, temporal snapshots, and verbatim source lanes

### Future Scale-Out

- Optional Neo4j Aura, pgGraph/PostgreSQL, or future ContinuityDB-backed projection for teams that need external infrastructure
- Kafka/NATS for event log aggregation
- Keep embedded read-index warmup and hot caches benchmark-gated: add or retain
  cache layers only when release reports show lower checkout latency or better
  token-efficient retrieval without weakening citations, temporal semantics, or
  projection integrity diagnostics

## Security

### Encryption

- **At rest**: Eventloom and embedded projection files should live on encrypted volumes for production workstations or servers.
- **In transit**: Use `bolt+s` (TLS) only when using the optional Neo4j sidecar.

### Access Control

```bash
# Optional Neo4j sidecar: create read-only user for agents
CREATE USER agent_reader SET PASSWORD 'secure_password';
GRANT ROLE reader TO agent_reader;
```

### Secrets Management

Use Docker secrets or `*_FILE` environment variables for sensitive settings.
Direct environment variables take precedence over file-backed values.

```bash
# Local production scaffold
./scripts/setup.sh --production

# Starts embedded-backed Zaxy with Docker secrets from ./secrets/
docker compose -f docker-compose.prod.yml up -d

# Optional Neo4j sidecar profile
./scripts/generate-certs.sh .certs
docker compose -f docker-compose.prod.yml --profile neo4j up -d zaxy-neo4j
```

Production mode rejects `NEO4J_PASSWORD=testpassword` only when
`PROJECTION_BACKEND=neo4j`. When using the generated custom CA for the optional
Neo4j sidecar, set `NEO4J_URI=bolt://...` with `NEO4J_CA_CERT` so the Neo4j
driver enables encryption and trusts the mounted CA.

For external secret managers such as Vault or AWS Secrets Manager, write values
to mounted files and set:

```bash
NEO4J_PASSWORD_FILE=/run/secrets/neo4j_password
MCP_ADMIN_TOKEN_FILE=/run/secrets/mcp_admin_token
PATHLIGHT_ACCESS_TOKEN_FILE=/run/secrets/pathlight_access_token
```

## Maintenance Windows

### Weekly

- Review Pathlight traces for anomalies, if enabled
- Check Eventloom log sizes
- Verify backup integrity

### Monthly

- Compact Eventloom logs
- Review and update extraction rules
- Update optional sidecar backends to latest patch versions
- Run full integration test suite

### Quarterly

- Performance benchmark regression test:
  `pytest tests/test_competitive_benchmarks.py --benchmark-only --no-cov`
- Review the active benchmark hub:
  `docs/benchmarks.md`
- Re-run the benchmark report guard only against the current headline 500 or
  Harvey LAB artifacts listed in that hub.
- Capacity planning review
- Security audit (dependency updates, key rotation)

## Go-Live Gate

Before promoting a build, run:

```bash
zaxy doctor --release-smoke
scripts/release-check.sh --root .
```

The release smoke check verifies the package version, changelog entry, publish
workflow, PyPI Trusted Publishing posture, dependency-light LangGraph example,
and outside-MCP OpenAI-compatible and Claude-compatible model-call examples.
The release gate runs `ruff`,
`mypy`, the full coverage-gated pytest suite, Python artifact build/metadata
validation, public examples, MCP smoke, LangGraph smoke, Coordinate mission
smoke, public site/documentation validation, deployment validation, backend
shootout evidence, injected-token efficiency floors, explicit benchmark
no-regression guardrails for checkout quality, citation coverage, p95/p99
latency budgets, 100-query embedded scale validation, and beta UAT. A release is not ready
until all gates pass, the production `.env` selects the intended projection
backend, optional Neo4j deployments use TLS-enabled Bolt with
`zaxy-memory[neo4j]` installed, remote MCP/SSE bearer auth is configured, and
secret files are not world-readable.

If a release smoke must be intentionally skipped in an environment-specific
dry run, set its command to `SKIP:<reason>` so the log records the reason. Do
not remove the command variable or leave it blank.

## Prometheus Alerts

```yaml
groups:
  - name: zaxy-degraded-mode
    rules:
      - alert: ZaxyGraphFallbacks
        expr: increase(zaxy_degraded_operations_total{reason=~"graph_.*"}[10m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Zaxy graph degradation detected
      - alert: ZaxyEmbeddingFallbacks
        expr: increase(zaxy_degraded_operations_total{reason="embedding_provider_unavailable"}[10m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Zaxy embedding provider unavailable
      - alert: ZaxyRerankerFallbacks
        expr: increase(zaxy_degraded_operations_total{reason="reranker_unavailable"}[10m]) > 0
        for: 10m
        labels:
          severity: info
        annotations:
          summary: Zaxy reranker degraded to MMR
```

## Incident Response

### Severity Levels

| Level | Example | Response Time |
|-------|---------|---------------|
| P0 | Data loss, all agents down | Immediate |
| P1 | Query failures, single agent down | <1 hour |
| P2 | Performance degradation | <4 hours |
| P3 | Observability gaps | <24 hours |

### P0: Data Loss

1. Stop all writes immediately
2. Restore from most recent backup
3. Replay Eventloom from last known good state
4. Verify graph consistency
5. Post-mortem within 24 hours

### Escalation

- **Zaxy maintainers**: GitHub Issues
- **Neo4j support**: neo4j.com/support, when using the optional Neo4j sidecar
- **Pathlight issues**: syndicalt/pathlight GitHub

## Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROJECTION_BACKEND` | `embedded` | Projection backend: `embedded`, `neo4j`, `pggraph`, or `latticedb` |
| `EMBEDDED_GRAPH_PATH` | `.eventloom/projections/embedded.kuzu` | Repo-local embedded LadybugDB projection path |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `testpassword` | Neo4j password |
| `NEO4J_PASSWORD_FILE` | unset | File containing Neo4j password |
| `NEO4J_CA_CERT` | unset | CA certificate path for encrypted custom-CA Bolt connections |
| `NEO4J_TRUST_ALL` | `false` | Trust all Neo4j certs; development only |
| `NEO4J_AUTO_START` | `false` | Auto-start a local Docker Neo4j container only when explicitly enabled with `PROJECTION_BACKEND=neo4j` |
| `NEO4J_AUTO_START_IMAGE` | `neo4j:5.26-community` | Docker image used by local Neo4j auto-start |
| `NEO4J_AUTO_START_CONTAINER` | `zaxy-neo4j` | Container name used by local Neo4j auto-start |
| `PATHLIGHT_URL` | `http://localhost:4100` | Pathlight collector |
| `PATHLIGHT_ENABLED` | `false` | Enable Pathlight client and health check |
| `PATHLIGHT_ACCESS_TOKEN` | unset | Optional Pathlight token |
| `PATHLIGHT_ACCESS_TOKEN_FILE` | unset | File containing optional Pathlight token |
| `TRACE_RAW_QUERIES` | `false` | Include raw query text in traces |
| `EVENTLOOM_PATH` | `.eventloom` | Event log directory |
| `EVENTLOOM_THREAD` | `default` | Default session/log name |
| `ZAXY_DOMAIN` | unset | Stable project/domain label used by generated MCP configs |
| `ZAXY_ENV` | `development` | Runtime environment; production enables stricter config validation |
| `MCP_ADMIN_TOKEN` | unset | Optional token for replay/invalidate tools |
| `MCP_ADMIN_TOKEN_FILE` | unset | File containing optional admin token |
| `MCP_REMOTE_AUTH_TOKEN` | unset | Bearer token required for remote MCP/SSE requests when configured |
| `MCP_REMOTE_AUTH_TOKEN_FILE` | unset | File containing remote MCP/SSE bearer token |
| `MCP_REMOTE_SESSION_HEADER` | `x-zaxy-session-id` | HTTP header that scopes remote MCP/SSE requests to a session |
| `MCP_OIDC_ISSUER` | unset | OIDC issuer for remote MCP/SSE JWT validation |
| `MCP_OIDC_AUDIENCE` | unset | Expected JWT audience for remote MCP/SSE |
| `MCP_OIDC_JWKS_URL` | unset | JWKS URL for remote MCP/SSE JWT signatures |
| `MCP_OIDC_REQUIRED_SCOPE` | `zaxy:mcp` | Required OAuth scope for remote MCP/SSE |
| `MCP_OIDC_SESSION_CLAIM` | `zaxy_session` | JWT claim containing the Zaxy session/tenant ID |
| `MCP_OIDC_CLIENT_SECRET_FILE` | unset | Optional OIDC client secret file for future introspection flows |
| `MCP_RATE_LIMIT_ENABLED` | `true` | Enable session-scoped remote MCP/SSE request rate limiting |
| `MCP_RATE_LIMIT_REQUESTS` | `120` | Maximum remote MCP/SSE requests per window |
| `MCP_RATE_LIMIT_WINDOW_SECONDS` | `60` | Remote MCP/SSE rate-limit window |
| `MCP_AUDIT_ENABLED` | `false` | Export remote MCP/SSE request audit JSONL |
| `MCP_AUDIT_PATH` | `.eventloom/remote_audit.jsonl` | Remote MCP/SSE request audit JSONL path |
| `QUERY_DEFAULT_LIMIT` | `10` | Default query result limit |
| `CONTEXT_VERBATIM_ENABLED` | `true` | Include exact Eventloom source recall in assembled context |
| `CONTEXT_VERBATIM_SLOTS` | `1` | Assembled context slots reserved for verbatim source recall |
| `EMBEDDING_ENABLED` | `true` | Generate embeddings for vector search |
| `EMBEDDING_PROVIDER` | `hash` | Embedding provider: `hash`, `openai`, `local-http`, or `sentence-transformers` |
| `EMBEDDING_DIMENSION` | `1536` | Vector dimension; must match the selected projection backend vector index |
| `OPENAI_API_KEY` | unset | OpenAI API key for hosted embeddings |
| `OPENAI_API_KEY_FILE` | unset | File containing OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `EMBEDDING_HTTP_URL` | unset | Local HTTP embedding endpoint for `local-http` |
| `EMBEDDING_HTTP_MODEL` | unset | Optional local HTTP embedding model name |
| `EMBEDDING_HTTP_API_KEY` | unset | Optional bearer token for local HTTP embeddings |
| `EMBEDDING_SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local model used by `sentence-transformers`; install `zaxy-memory[local-embeddings]` |

### CLI Commands

```bash
zaxy serve          # Start MCP stdio server
zaxy serve --transport sse --port 8080  # Start MCP SSE server bound to localhost
zaxy ide-config claude-desktop --eventloom-path .eventloom  # Print first-run MCP config
zaxy local-profile --output .env.local  # Write offline retrieval profile
zaxy local-profile --projection-backend embedded --output .env.local  # Write no-sidecar embedded profile
zaxy local-profile --check  # Validate deterministic local retrieval providers
zaxy init-session . --session-id zaxy-default  # Append workspace genesis profile event
zaxy index-codebase . --session-id zaxy-default  # Append codebase file, symbol, import, dependency, call, and coverage events
zaxy memory status --eventloom-path .eventloom  # Inspect Eventloom sessions, latest hashes, and integrity
zaxy memory log --eventloom-path .eventloom --limit 20  # Show recent Eventloom events
zaxy memory diff --eventloom-path .eventloom --session-id zaxy-default --from-seq 10 --to-seq 20  # Show added events in a sequence range
zaxy replay PATH    # Replay Eventloom log
zaxy compact PATH --audit  # Audit compaction safety without rewriting the log
zaxy compact PATH   # Compact log + create snapshot
zaxy status         # Check service health
scripts/backup.sh --root . --output-dir backups
scripts/restore.sh --archive backups/zaxy.tar.gz --manifest backups/zaxy.sha256 --target restored
scripts/rotate-logs.sh --log .eventloom/default.jsonl
scripts/validate-deployment.sh --root .
scripts/build-dist.sh --root .
scripts/validate-docs.sh --root .
scripts/release-check.sh --root .
```

### MCP Tools

| Tool | Purpose |
|------|---------|
| `memory_append` | Write event to log + graph |
| `memory_query` | Hybrid retrieval from graph |
| `memory_replay` | Replay session events; requires `admin_token` if configured |
| `memory_invalidate` | Soft-delete (bi-temporal); requires `admin_token` if configured |

---

*Last updated: 2026-05-05*
