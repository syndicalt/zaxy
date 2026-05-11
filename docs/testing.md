# Testing

Zaxy follows test-first development. Public behavior should have a test before
implementation. The full suite has a broad 90 percent pytest coverage gate plus
a coverage ratchet that currently requires at least 91.89% total line coverage
from `coverage.xml`. Unit tests mock external dependencies such as Neo4j and
Pathlight. Integration tests use Docker services and are marked with
`integration`.

Common commands:

```bash
pytest
pytest -m integration --no-cov
scripts/integration-check.sh --start
ruff check src tests
mypy src
pytest tests/test_packet_memory_e2e.py --no-cov -q
zaxy doctor --beta-readiness
scripts/beta-uat.sh
scripts/release-check.sh --root .
```

The default pytest command includes coverage reporting and `--cov-fail-under=90`
from `pyproject.toml`. CI and `scripts/release-check.sh` also run
`scripts/check-coverage.py` against the generated XML report. The ratchet floor
lives in `[tool.zaxy.coverage]`, is based on the canonical CI Python 3.13
measurement, and can be intentionally raised after coverage improvements.
Integration-only runs use `--no-cov` because the project-level coverage gate is
intended for the full suite. Before running integration tests, start the Neo4j
services:

```bash
./scripts/generate-certs.sh .certs
docker compose up -d neo4j-test neo4j-tls
```

For local full-suite checks, prefer the integration helper so the Neo4j
dependency is explicit:

```bash
scripts/integration-check.sh --start
scripts/integration-check.sh --require
scripts/integration-check.sh --skip-if-unavailable
```

Use `--start` when Docker is available and the helper should generate TLS
certs, boot `neo4j-test` and `neo4j-tls`, then run pytest. Use `--require`
when services should already be running and absence should fail fast. Use
`--skip-if-unavailable` for development loops where graph integration tests
should be omitted only after the helper verifies the Neo4j test ports are not
reachable.

Tests are organized by module: event log integrity, extraction, graph behavior,
query routing, MCP tools, tracing, configuration, embeddings, operations
scripts, packaging, and site/docs validation. New modules should get focused
tests rather than relying only on high-level workflows.

The packet-memory product path has an explicit smoke check:

```bash
pytest tests/test_packet_memory_e2e.py --no-cov -q
```

`scripts/release-check.sh` runs this packet smoke after the full pytest suite so
the analyzer-to-projection-to-context workflow remains a named release gate.

The beta hardening path has two additional checks. `zaxy doctor
--beta-readiness` is a fast local inventory of release metadata, release gate
coverage, clean-repo UAT coverage, documentation, and deterministic capture
posture. `scripts/beta-uat.sh` performs a clean first-run exercise in a
throwaway workspace: install, `zaxy init`, deterministic capture startup,
`zaxy memory bootstrap`, `zaxy memory checkout`, doctor, hook status, capture
status, and memory status.

For graph changes, write both mock tests for Cypher behavior and integration
tests against Neo4j when the real database semantics matter. For security
changes, test both accepted and rejected inputs. For scripts, use temporary
fixtures and injectable command stubs so tests can assert ordering and
fail-fast behavior without running destructive commands.

Benchmark tests cover extraction latency, append latency, graph upsert latency,
query latency, and competitive retrieval harness behavior. Benchmarks are useful
for detecting large regressions, but correctness tests decide release readiness.
For live comparative statistics against markdown, BM25, vector, markdown+vector, and
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

For MemPalace-comparable temporal recall beyond the original frozen statistical
lane, use the dedicated temporal workload. It creates three time-versioned
preference states per subject, queries each state with an explicit as-of point,
and reports citation coverage for otherwise successful retrievals:

```bash
scripts/live-benchmark.sh --embedding-provider openai --workload temporal-recall --subjects 100 --runs 1 --reset-graph
```

For MemPalace-comparable source recall, use the dedicated source workload. It
creates a target document and a near-miss distractor per case, then reports
whether retrieval returned the exact expected source path as a separate source
recall metric:

```bash
scripts/live-benchmark.sh --embedding-provider openai --workload source-recall --documents 100 --runs 1 --reset-graph
```

For public memory-benchmark comparisons against systems that report
LongMemEval recall, download the cleaned LongMemEval JSON and run the
`longmemeval` workload. This workload preserves answer session identifiers and
reports identity recall in addition to answer-term recall:

```bash
mkdir -p /tmp/longmemeval-data
curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

scripts/live-benchmark.sh \
  --embedding-provider openai \
  --embedding-cache .cache/zaxy/longmemeval-embeddings.json \
  --progress \
  --workload longmemeval \
  --dataset /tmp/longmemeval-data/longmemeval_s_cleaned.json \
  --runs 1 \
  --limit 5 \
  --reset-graph
```

Use `--questions 1` to validate credentials and service wiring, then
`--questions 20` for a larger smoke run before the full 500-question pass. Keep
`--embedding-cache` enabled for hosted embedding runs; LongMemEval contains many
haystack chunks and reusable corpus embeddings make interrupted or repeated runs
much cheaper. `--progress` prints backend/case counters to stderr so long runs
do not appear stalled. The headline comparison field for this workload is
identity recall at the requested limit, which corresponds to whether the
answer-bearing session was retrieved.

Frozen reports include a workload version, event count, query count, source
recall, citation coverage, and SHA-256 fingerprint so later runs can prove they
used the same corpus. External systems such as QMD/OpenClaw, Graphiti/Zep,
MemPalace, or Mem0 can be included only as operator-supplied disclosure rows via
the Python CLI's `--external-results` JSON option; those rows are not treated as
harness-verified results.

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
