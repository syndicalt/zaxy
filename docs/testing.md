# Testing

Zaxy follows test-first development. Public behavior should have a test before
implementation. The full suite has a broad 92 percent pytest coverage gate plus
a coverage ratchet that currently requires at least 92.00% total line coverage
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
zaxy benchmark-freeze --json
```

The default pytest command includes coverage reporting and `--cov-fail-under=92`
from `pyproject.toml`. CI and `scripts/release-check.sh` also run
`scripts/check-coverage.py` against the generated XML report. The ratchet floor
lives in `[tool.zaxy.coverage]`, is based on the canonical CI Python 3.13
measurement, and can be intentionally raised after coverage improvements.
Integration-only runs use `--no-cov` because the project-level coverage gate is
intended for the full suite. Before running integration tests, start the Neo4j
services:

```bash
./scripts/generate-certs.sh .certs
docker compose --profile integration up -d neo4j-test neo4j-tls
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

The contract-validation path (introduced with the v0.9 freeze) includes
fuzz-style parametrized checks for Eventloom payload validation, hash-chain
replay, and bounded MCP inputs. These
tests intentionally exercise malformed payload shapes, oversized JSON objects,
sequence-tampered but hash-valid Eventloom records, and invalid direct
`memory_append` handler inputs so contract validation does not depend only on
MCP client schema enforcement.

The packet-memory product path has an explicit smoke check:

```bash
pytest tests/test_packet_memory_e2e.py --no-cov -q
```

Keep packet-memory smoke coverage active so the analyzer-to-projection-to-context
workflow remains tested. Public benchmark gates are documented separately in
[benchmarks.md](benchmarks.md); do not add release checks that depend on
archived benchmark reports unless the benchmark hub is updated first.

The beta hardening path has two additional checks. `zaxy doctor
--beta-readiness` is a fast local inventory of release metadata, release gate
coverage, clean-repo UAT coverage, documentation, and deterministic capture
posture. `scripts/beta-uat.sh` performs a clean first-run exercise in a
throwaway workspace: install, `zaxy init`, deterministic capture startup,
`zaxy memory bootstrap`, `zaxy memory checkout`, doctor, hook status, capture
status, capture soak, and memory status. It also runs the bare embedded init
path and verifies the generated profile includes `PROJECTION_BACKEND=embedded`,
`NEO4J_AUTO_START=false`, and the repo-local embedded projection path, so the
zero-friction default cannot silently drift back to a sidecar requirement. The
bare embedded branch also runs `zaxy memory status --eventloom-path .eventloom --graph`,
`zaxy memory inferred-status`, and `zaxy reproject` so clean UAT
proves the repo-local embedded projection can be inspected, audited, rebuilt,
and rechecked without backend flags. The UAT path runs `zaxy hook-status
--min-activation-rate 1.0`, so it fails if its clean first-run captured sessions
did not all start substantive work after fresh checkout. `zaxy capture-soak` is
the beta evidence command for
deterministic capture: it checks transcript, tool-call, command, and file-edit
observation coverage, freshness, latest seq/hash, and remediation steps.
`zaxy doctor --beta-readiness` also reads
`docs/examples/first-run-timing-report.json`; keep
`time_to_successful_doctor_seconds` and
`time_to_first_successful_example_seconds` at or below 300 seconds, and keep
`requires_sidecar` false for the default local path.

For activation hardening, `zaxy hook-status --json` reports activation
efficiency under `memory_activation.activation_efficiency`. The metric counts
high-context sessions that have command, file-edit, tool-call, or transcript
activity, then reports what percentage had a fresh `memory.checkout.completed`
event before the first substantive captured event. Keep this as a product KPI:
capture without fresh checkout means Zaxy observed the work but did not become
the model's working context. Treat activation efficiency as a release-readiness
signal for launcher, hook, and dashboard work. When hooks emit
`memory.reminder.suggested`, `zaxy hook-status --json` also exposes
`memory_activation.latest_reminder` so the warning is tied back to an auditable
Eventloom event. Checkout activity markers now preserve numeric
`token_efficiency` diagnostics, including prompt-token estimates and current
facts per 1k prompt tokens, so `hook-status` and dashboard status can show
whether activation is both fresh and token-disciplined.

Use the same command as a guardrail when the evidence fixture should prove that
models are actually starting work with memory loaded:

```bash
zaxy hook-status --eventloom-path .eventloom --json --min-activation-rate 0.8 \
  --max-checkout-prompt-tokens 5000 \
  --min-checkout-facts-per-1k-tokens 0.1
```

The release gate names every public smoke path it expects. A command may be set
to `SKIP:<reason>` only when the skip is intentional and auditable; otherwise
`zaxy doctor --beta-readiness` reports the missing surface through
`release_gate_surface_coverage`.

The historical v0.9 gate evidence is archived in
[v09-gate-audit.md](archive/v09-gate-audit.md). That audit intentionally leaves
the external-user feedback gate pending until feedback from outside the current
implementation session exists.

The command exits non-zero when fewer than 80% of high-context sessions had
fresh checkout before substantive captured work, when the latest checkout
exceeds the prompt-token ceiling, or when checkout facts per 1k prompt tokens
falls below the required floor.

For graph changes, write both mock tests for Cypher behavior and integration
tests against Neo4j when the real database semantics matter. For security
changes, test both accepted and rejected inputs. For scripts, use temporary
fixtures and injectable command stubs so tests can assert ordering and
fail-fast behavior without running destructive commands.

Benchmark tests cover extraction latency, append latency, graph upsert latency,
query latency, and retrieval harness behavior. Benchmarks are useful for
detecting large regressions, but correctness tests decide release readiness.

The current public benchmark evidence is documented in
[benchmarks.md](benchmarks.md): a full-haystack, held-out LongMemEval-S result
(`0.898` gpt-5 full-500 / `0.777` gpt-4o held-out answer accuracy, Recall@5
`0.99`). The prior oracle-mode "headline 500" stays retracted; see
benchmarks.md. Active public claims should point only
to the Harvey LAB external legal-agent memory-ablation report.

Older benchmark suites, backend shootouts, StateRecoveryBench, PurposeBench,
LongMemBench adapter artifacts, debug runs, and partial LongMemEval iterations
are archived development history. Use those artifacts for engineering
investigation only, not as current public benchmark claims.

The `2.0.0-rc.1` StateRecoveryBench, CoordinationBench, and PurposeBench
artifacts under `reports/benchmarks/` are active RC release guardrails, but
remain project-defined internal evidence unless a future release publishes a
separate external validation boundary for them.

When a future full LongMemEval run is published, write it to a new report
directory under `reports/benchmarks/`, update [benchmarks.md](benchmarks.md)
with the run's own claim boundary, and keep any retracted report under
`reports/archive/`.

For the 2.0 RC.1 release candidate, run:

```bash
zaxy benchmark-freeze --json
```

This validates Harvey LAB external-anchor artifacts, StateRecoveryBench,
CoordinationBench, PurposeBench, and project-defined 2.0 internal lanes. It is
a release evidence and claim-boundary gate; it must not be used to tune answers
or mix internal guardrail lanes into public benchmark claims.

For consolidation safety checks, use the identity-collapse workload. It creates
near-duplicate source records with distinct durable identifiers and adds an
identity-recall metric to the report. The `centroid` baseline intentionally
models semantic consolidation that keeps one representative text, so it can
look topically relevant while losing exact source identities:

```bash
scripts/live-benchmark.sh --workload consolidation --documents 100 --runs 1 --reset-graph
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
