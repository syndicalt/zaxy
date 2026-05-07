# Testing

Zaxy follows test-first development. Public behavior should have a test before
implementation, and the full suite must remain above the 90 percent coverage
gate. Unit tests mock external dependencies such as Neo4j and Pathlight.
Integration tests use Docker services and are marked with `integration`.

Common commands:

```bash
pytest
pytest -m integration --no-cov
ruff check src tests
mypy src
scripts/release-check.sh --root .
```

The default pytest command includes coverage reporting and `--cov-fail-under=90`
from `pyproject.toml`. Integration-only runs use `--no-cov` because the
project-level coverage gate is intended for the full suite. Before running
integration tests, start the Neo4j services:

```bash
./scripts/generate-certs.sh .certs
docker compose up -d neo4j-test neo4j-tls
```

Tests are organized by module: event log integrity, extraction, graph behavior,
query routing, MCP tools, tracing, configuration, embeddings, operations
scripts, packaging, and site/docs validation. New modules should get focused
tests rather than relying only on high-level workflows.

For graph changes, write both mock tests for Cypher behavior and integration
tests against Neo4j when the real database semantics matter. For security
changes, test both accepted and rejected inputs. For scripts, use temporary
fixtures and injectable command stubs so tests can assert ordering and
fail-fast behavior without running destructive commands.

Benchmark tests cover extraction latency, append latency, graph upsert latency,
query latency, and competitive retrieval harness behavior. Benchmarks are useful
for detecting large regressions, but correctness tests decide release readiness.
For live comparative statistics against markdown, vector, markdown+vector, and
Zaxy retrieval, run the statistically powered workload:

```bash
./scripts/generate-certs.sh .certs
docker compose up -d neo4j-test
scripts/live-benchmark.sh --embedding-provider openai --workload statistical --subjects 100 --runs 1 --reset-graph
```

OpenAI mode uses `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, and
`EMBEDDING_DIMENSION`. The default model is `text-embedding-3-small`.
The script writes `reports/benchmarks/live-benchmark.json` for automation and
`reports/benchmarks/live-benchmark.md` for human review. Use
`--embedding-provider hash` for deterministic offline smoke checks.

For publishable comparisons, use the frozen workload instead of a custom
subject count:

```bash
scripts/live-benchmark.sh --embedding-provider openai --workload frozen --runs 1 --reset-graph
```

Frozen reports include a workload version, event count, query count, and
SHA-256 fingerprint so later runs can prove they used the same corpus. External
systems such as QMD/OpenClaw, Graphiti/Zep, or Mem0 can be included only as
operator-supplied disclosure rows via the Python CLI's `--external-results`
JSON option; those rows are not treated as harness-verified results.

Interpret these results narrowly. The statistical corpus is a generated temporal
event workload, not a broad document-search benchmark or a claim against every
production-grade vector RAG implementation. Use it to measure Zaxy's target
problem: current versus historical facts, stale-context avoidance, graph
connections, latency, and returned context size on the same paired workload.

CI runs lint, mypy, the full test matrix, package artifact validation, and
integration tests. The local release gate mirrors the important pieces. See
[operations.md](operations.md), [deployment.md](deployment.md), and
[README.md](../README.md). The public docs entry is [site/index.html](../site/index.html).
