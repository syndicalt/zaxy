# Operations

Zaxy operations center on four tasks: keep Eventloom logs durable, keep Neo4j
healthy, validate deployments before exposure, and preserve enough observability
to debug memory behavior. The full incident checklist remains in
[runbook.md](runbook.md); this page is the day-to-day operator summary.

Backups should include Eventloom logs, relevant configuration, and any Neo4j
data that is expensive to rebuild. Eventloom is the required source of truth.
Neo4j can be rebuilt by replay, but backing it up can reduce recovery time for
large deployments. Use `scripts/backup.sh` and `scripts/restore.sh` for tested
local archive flows.

Log rotation is available through `scripts/rotate-logs.sh`. Rotation should not
discard active history until backups are verified. After rotation, run replay on
the archived log to confirm hash-chain integrity. A corrupted archive is not a
backup.

Deployment validation is run with:

```bash
scripts/validate-deployment.sh --root .
```

This checks production mode, Neo4j TLS configuration, remote MCP auth, and
secret-file permissions. The broader release gate is:

```bash
scripts/release-check.sh --root .
```

That gate runs ruff, mypy, pytest with coverage, package artifact validation,
documentation validation, deployment validation, activation efficiency and
checkout token guardrails, backend shootout evidence, injected-token efficiency
floors, and 100-query embedded scale validation.

PyPI publishing is handled by the `Publish Python Package` GitHub Actions
workflow. Publish a GitHub release after the release gate passes; the workflow
builds artifacts, checks them with Twine, and uploads the `zaxy-memory`
distribution using the `PYPI_API_TOKEN` repository secret.

Metrics are exposed through the Prometheus collector when enabled. Track append
counts, query counts, query latency, graph upserts, and invalidations. Sudden
changes in query latency often mean index health, vector settings, or traversal
fanout changed.

Graceful degradation is tracked separately through
`zaxy_degraded_operations_total{operation,reason}`. Alert on sustained increases
for `graph_unavailable`, `graph_retrieval_unavailable`,
`graph_projection_unavailable`, `embedding_provider_unavailable`,
`vector_search_unavailable`, and `reranker_unavailable`. Fallbacks keep agents
working, but any nonzero production rate means an operator should verify graph
health, embedding credentials, vector indexes, and reranker endpoints.

Pathlight tracing is optional but recommended for production debugging. It gives
span-level visibility into append, query, replay, and invalidate operations.
Pathlight is not the storage layer; it is the inspection layer. If tracing is
down, memory operations should continue.

Related documents: [deployment.md](deployment.md), [security.md](security.md),
[configuration.md](configuration.md), [testing.md](testing.md), and
[README.md](../README.md). Public product positioning is in
[site/index.html](../site/index.html).
