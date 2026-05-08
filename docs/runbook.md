# Zaxy Operational Runbook

## Architecture Overview

Zaxy is an event-sourced temporal knowledge graph fabric for AI agent memory.
It consists of three layers:

1. **Eventloom** (bottom): Immutable append-only JSONL logs with SHA-256 hash chains.
2. **Neo4j** (core): Bi-temporal knowledge graph with entity/relationship validity windows.
3. **Pathlight** (optional top layer): Observability, tracing, and debugging dashboard.

## Quick Start

```bash
# Start infrastructure
docker compose up -d neo4j

# Install Zaxy
pip install -e ".[dev]"

# Verify connectivity
python -m zaxy status

# Run tests
pytest

# Start MCP server
python -m zaxy serve

# Start MCP over SSE for daemon mode
python -m zaxy serve --transport sse --port 8080
```

## Daily Operations

### Health Checks

```bash
# Check all services
python -m zaxy status

# Or manually:
curl http://localhost:7474  # Neo4j HTTP

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

# Export as JSON
python -m zaxy replay .eventloom/work.jsonl --json

# Audit identity and citation safety before compacting
python -m zaxy compact .eventloom/work.jsonl --audit

# Export a machine-readable audit report
python -m zaxy compact .eventloom/work.jsonl --audit --json

# Store a source-backed medoid projection without rewriting the log
python -m zaxy compact .eventloom/work.jsonl --projection-output .eventloom/work.compaction.json

# Store a bounded exemplar projection for high-spread clusters
python -m zaxy compact .eventloom/work.jsonl --projection-output .eventloom/work.compaction.json --strategy exemplar --max-records 5

# Load projections in Python integrations
python - <<'PY'
from zaxy import MemoryFabric

fabric = MemoryFabric(projection_paths=[".eventloom/work.compaction.json"])
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
| Neo4j database | Docker volume `neo4j_data` | High — can be rebuilt from Eventloom |
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
archive. Neo4j can be rebuilt from Eventloom; take a separate Neo4j dump only
when fast point-in-time restore matters more than minimizing backup surface.

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
| Neo4j disk usage | <80% | >90% |

### Neo4j Monitoring

```cypher
// Check database size
CALL dbms.database.state("neo4j") YIELD status;

// Check index status
SHOW INDEXES;

// Check constraint status
SHOW CONSTRAINTS;

// Entity count
MATCH (e:Entity) RETURN count(e) AS entities;

// Relationship count
MATCH ()-[r:RELATES]->() RETURN count(r) AS relations;

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

### "Neo4j connection refused"

1. Check container status:
   ```bash
   docker compose ps neo4j
   docker compose logs neo4j
   ```

2. Check memory: Neo4j needs at least 2GB heap.
   ```bash
   docker compose exec neo4j neo4j-admin memrec
   ```

3. Check ports:
   ```bash
   netstat -tlnp | grep 7687
   ```

### Performance Degradation

1. **Query slow?** Check Neo4j query plan:
   ```cypher
   PROFILE MATCH (e:Entity {name: "X"}) RETURN e;
   ```

2. **Event append slow?** Check disk I/O:
   ```bash
   iostat -x 1
   ```

3. **Graph upsert slow?** Check for missing indexes:
   ```cypher
   SHOW INDEXES;
   ```

## Scaling Considerations

### Current

- One Eventloom file per session/agent
- Single Neo4j instance
- Optional Pathlight tracing

### Future Scale-Out

- Neo4j Aura or causal clustering
- Kafka/NATS for event log aggregation
- Add Redis hot cache between Eventloom and Neo4j

## Security

### Encryption

- **At rest**: Neo4j Enterprise supports native encryption. For Community, use encrypted volumes (LUKS, AWS EBS encryption).
- **In transit**: Use `bolt+s` (TLS) for Neo4j connections.

### Access Control

```bash
# Neo4j: Create read-only user for agents
CREATE USER agent_reader SET PASSWORD 'secure_password';
GRANT ROLE reader TO agent_reader;
```

### Secrets Management

Use Docker secrets or `*_FILE` environment variables for sensitive settings.
Direct environment variables take precedence over file-backed values.

```bash
# Local production scaffold
./scripts/setup.sh --production

# Starts Neo4j and Zaxy with Docker secrets from ./secrets/
./scripts/generate-certs.sh .certs
docker compose -f docker-compose.prod.yml up -d
```

Production mode rejects `NEO4J_PASSWORD=testpassword`. When using the generated
custom CA, set `NEO4J_URI=bolt://...` with `NEO4J_CA_CERT` so the Neo4j driver
enables encryption and trusts the mounted CA.

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
- Update Neo4j to latest patch version
- Run full integration test suite

### Quarterly

- Performance benchmark regression test:
  `pytest tests/test_competitive_benchmarks.py --benchmark-only --no-cov`
- Frozen live retrieval benchmark:
  `scripts/live-benchmark.sh --embedding-provider openai --workload frozen --runs 1 --reset-graph`
- Representative retrieval benchmark suite:
  `scripts/live-benchmark.sh --embedding-provider openai --workload suite --subjects 100 --documents 250 --sessions 50 --runs 1 --reset-graph`
- Capacity planning review
- Security audit (dependency updates, key rotation)

## Go-Live Gate

Before promoting a build, run:

```bash
scripts/release-check.sh --root .
```

The release gate runs `ruff`, `mypy`, the full coverage-gated pytest suite,
Python artifact build/metadata validation, public site/documentation validation,
and deployment validation. A release is not ready until all six gates pass, the
production `.env` points at
TLS-enabled Neo4j, remote MCP/SSE bearer auth is configured, and secret files
are not world-readable.

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
- **Neo4j support**: neo4j.com/support
- **Pathlight issues**: syndicalt/pathlight GitHub

## Reference

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `testpassword` | Neo4j password |
| `NEO4J_PASSWORD_FILE` | unset | File containing Neo4j password |
| `NEO4J_CA_CERT` | unset | CA certificate path for encrypted custom-CA Bolt connections |
| `NEO4J_TRUST_ALL` | `false` | Trust all Neo4j certs; development only |
| `PATHLIGHT_URL` | `http://localhost:4100` | Pathlight collector |
| `PATHLIGHT_ENABLED` | `false` | Enable Pathlight client and health check |
| `PATHLIGHT_ACCESS_TOKEN` | unset | Optional Pathlight token |
| `PATHLIGHT_ACCESS_TOKEN_FILE` | unset | File containing optional Pathlight token |
| `TRACE_RAW_QUERIES` | `false` | Include raw query text in traces |
| `EVENTLOOM_PATH` | `.eventloom` | Event log directory |
| `EVENTLOOM_THREAD` | `default` | Default session/log name |
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
| `QUERY_DEFAULT_LIMIT` | `10` | Default query result limit |
| `EMBEDDING_ENABLED` | `true` | Generate embeddings for vector search |
| `EMBEDDING_PROVIDER` | `hash` | Embedding provider: `hash` or `openai` |
| `EMBEDDING_DIMENSION` | `1536` | Vector dimension; must match the Neo4j vector index |
| `OPENAI_API_KEY` | unset | OpenAI API key for hosted embeddings |
| `OPENAI_API_KEY_FILE` | unset | File containing OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |

### CLI Commands

```bash
zaxy serve          # Start MCP stdio server
zaxy serve --transport sse --port 8080  # Start MCP SSE server bound to localhost
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
