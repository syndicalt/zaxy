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

For production-scale representative evaluation, use the suite workload. It keeps
the same paired backends but expands the corpus to current facts, historical
facts, graph traversal, indexed documents, sanitized transcript turns, and mixed
cross-lane queries:

```bash
scripts/live-benchmark.sh --embedding-provider openai --workload suite --subjects 100 --documents 250 --sessions 50 --runs 1 --reset-graph
```

Suite reports disclose subject, document, session, lane, event, query, and
SHA-256 workload metadata. Increase `--subjects`, `--documents`, and
`--sessions` for capacity tests after the smoke run is stable.

For consolidation safety checks, use the identity-collapse workload. It creates
near-duplicate source records with distinct durable identifiers and adds an
identity-recall metric to the report. The `centroid` baseline intentionally
models semantic consolidation that keeps one representative text, so it can
look topically relevant while losing exact source identities:

```bash
scripts/live-benchmark.sh --embedding-provider openai --workload consolidation --documents 100 --runs 1 --reset-graph
```

Use this lane to detect whether a compaction strategy preserves exact event,
document, transcript, or entity identity under retrieval, not just broad topic
coverage.

Interpret the frozen temporal results narrowly. The suite workload is broader,
but still synthetic; use it to measure Zaxy's target problem before making broad
market claims: current versus historical facts, stale-context avoidance, graph
connections, cited document recall, transcript recall, mixed context assembly,
latency, and returned context size on the same paired workload.

CI runs lint, mypy, the full test matrix, package artifact validation, and
integration tests. The local release gate mirrors the important pieces. See
[operations.md](operations.md), [deployment.md](deployment.md), and
[README.md](../README.md). The public docs entry is [site/index.html](../site/index.html).
