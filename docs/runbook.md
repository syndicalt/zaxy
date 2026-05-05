# Zaxy Operational Runbook

## Architecture Overview

Zaxy is an event-sourced temporal knowledge graph fabric for AI agent memory.
It consists of three layers:

1. **Eventloom** (bottom): Immutable append-only JSONL logs with SHA-256 hash chains.
2. **Neo4j** (core): Bi-temporal knowledge graph with entity/relationship validity windows.
3. **Pathlight** (top): Observability, tracing, and debugging dashboard.

## Quick Start

```bash
# Start infrastructure
docker compose up -d neo4j

# Install Zaxy
pip install -e ".[dev]"

# Verify connectivity
python -m zaxy status

# Run tests
pytest -m "not integration"

# Start MCP server
python -m zaxy serve
```

## Daily Operations

### Health Checks

```bash
# Check all services
python -m zaxy status

# Or manually:
curl http://localhost:7474  # Neo4j HTTP
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
| Pathlight traces | Docker volume `pathlight_data` | Medium — observability only |

### Backup Procedures

```bash
#!/bin/bash
# backup.sh — Run daily via cron

DATE=$(date +%Y%m%d)
BACKUP_DIR=/backups/zaxy/$DATE
mkdir -p $BACKUP_DIR

# 1. Eventloom logs (most critical)
cp -r .eventloom $BACKUP_DIR/eventloom

# 2. Neo4j database
docker compose exec neo4j neo4j-admin database dump neo4j --to-path=/tmp/neo4j-backup
docker cp neo4j:/tmp/neo4j-backup $BACKUP_DIR/neo4j.dump

# 3. Pathlight SQLite (if self-hosted)
docker compose cp pathlight:/app/data $BACKUP_DIR/pathlight

echo "Backup complete: $BACKUP_DIR"
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh — Restore from backup

BACKUP_DIR=$1

# 1. Restore Eventloom logs
cp -r $BACKUP_DIR/eventloom .eventloom

# 2. Restore Neo4j
docker compose down neo4j
# Clear old data
docker volume rm zaxy_neo4j_data
docker compose up -d neo4j
sleep 30  # Wait for Neo4j to start
docker compose exec neo4j neo4j-admin database load neo4j --from-path=/tmp/restore
docker cp $BACKUP_DIR/neo4j.dump neo4j:/tmp/restore

# 3. Replay Eventloom to rebuild graph (if Neo4j backup is missing)
python << 'PY'
import asyncio
from zaxy.core import MemoryFabric

async def rebuild():
    fabric = MemoryFabric()
    await fabric.connect()
    
    # Replay all events and re-project
    replay = await fabric.replay()
    for event in replay.events:
        from zaxy.extract import extract
        from zaxy.graph import GraphStore
        extraction = extract(event)
        await fabric.graph.upsert_extraction(extraction)
    
    await fabric.close()

asyncio.run(rebuild())
PY
```

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
#!/bin/bash
# rotate-eventloom.sh

LOG_DIR=.eventloom
MAX_SIZE=100000000  # 100MB
MAX_AGE=30          # days

for log in $LOG_DIR/*.jsonl; do
    size=$(stat -f%z "$log" 2>/dev/null || stat -c%s "$log" 2>/dev/null)
    if [ $size -gt $MAX_SIZE ]; then
        # Compact and snapshot
        python -m zaxy compact "$log" --snapshot-every 10000
        mv "$log" "$log.$(date +%Y%m%d)"
    fi
done

# Remove old rotated logs
find $LOG_DIR -name "*.jsonl.*" -mtime +$MAX_AGE -delete
```

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

3. **Check Redis cache** (if enabled): Force refresh by restarting cache.

4. **Pathlight trace**: Inspect the exact context injected into the prompt.

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

### Single-Agent (Current)

- One Eventloom file per session
- Single Neo4j instance
- Pathlight local SQLite

### Multi-Agent (Future)

- Shard Eventloom by session ID (file per agent)
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

Use environment variables or a secrets manager (Vault, AWS Secrets Manager):

```bash
# .env (do not commit)
NEO4J_PASSWORD=$(vault read -field=password secret/zaxy/neo4j)
PATHLIGHT_ACCESS_TOKEN=$(vault read -field=token secret/zaxy/pathlight)
```

## Maintenance Windows

### Weekly

- Review Pathlight traces for anomalies
- Check Eventloom log sizes
- Verify backup integrity

### Monthly

- Compact Eventloom logs
- Review and update extraction rules
- Update Neo4j to latest patch version
- Run full integration test suite

### Quarterly

- Performance benchmark regression test
- Capacity planning review
- Security audit (dependency updates, key rotation)

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
| `PATHLIGHT_URL` | `http://localhost:4100` | Pathlight collector |
| `EVENTLOOM_PATH` | `.eventloom` | Event log directory |
| `ZAXY_TRACER_DISABLED` | `false` | Disable Pathlight tracing |

### CLI Commands

```bash
zaxy serve          # Start MCP server
zaxy replay PATH    # Replay Eventloom log
zaxy compact PATH   # Compact log + create snapshot
zaxy status         # Check service health
```

### MCP Tools

| Tool | Purpose |
|------|---------|
| `memory_append` | Write event to log + graph |
| `memory_query` | Hybrid retrieval from graph |
| `memory_replay` | Replay session events |
| `memory_invalidate` | Soft-delete (bi-temporal) |

---

*Last updated: 2024-01-01*
