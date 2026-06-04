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

The v0.9 freeze-candidate path includes fuzz-style parametrized checks for
Eventloom payload validation, hash-chain replay, and bounded MCP inputs. These
tests intentionally exercise malformed payload shapes, oversized JSON objects,
sequence-tampered but hash-valid Eventloom records, and invalid direct
`memory_append` handler inputs so contract validation does not depend only on
MCP client schema enforcement.

The packet-memory product path has an explicit smoke check:

```bash
pytest tests/test_packet_memory_e2e.py --no-cov -q
```

`scripts/release-check.sh` runs this packet smoke after the full pytest suite so
the analyzer-to-projection-to-context workflow remains a named release gate.
It also runs `PYTHONPATH=src python -m zaxy hook-status` with
`--eventloom-path reports/activation-release`,
`--now 2026-05-20T12:00:00+00:00`, `--min-activation-rate 1.0`,
`--max-checkout-prompt-tokens 5000`, and
`--min-checkout-facts-per-1k-tokens 0.1`, so release evidence includes a checked
activation fixture where substantive work starts after fresh, token-disciplined
checkout without depending on the wall clock. `zaxy doctor --beta-readiness`
verifies that fixture exists, passes Eventloom hash-chain integrity, and
contains both a token-efficiency-bearing checkout event and a later high-context
event before accepting the release gate inventory. It also checks the fixture's
checkout freshness, prompt-token estimate, and facts-per-1k-prompt-token values
against the release thresholds, so the artifact cannot drift below the command's
advertised guardrails. The release gate inventory also requires the hook-status
command to point at `reports/activation-release` instead of an ad hoc Eventloom
path. It also runs the backend
shootout guardrail through
`scripts/check-backend-shootout.py` against the checked active-backend report,
so release evidence fails if labeled Answer@5/Recall@5, citation coverage, or
the embedded dashboard graph source are missing or below the documented floor.
The backend report checks use `--require-report-metadata`,
`--require-markdown-report`, `--require-query-results`, and
`--require-git-tracked-inputs`, so release evidence must include the report
schema version, generation timestamp, source fingerprints, normalized workload
fingerprints, a matching human-readable Markdown sidecar, per-query diagnostics
needed to reproduce or audit the run, and checked-in Eventloom/query inputs.
They also use `--verify-report-fingerprints`, which recomputes the Eventloom,
query-file, filtered-event, and normalized-query fingerprints before accepting
the report. Active release reports also use `--forbid-backends
neo4j,pggraph,latticedb`, so optional sidecar or candidate evidence cannot be
mistaken for the current sidecar-free active-backend gate.
That checked active-backend report must also preserve embedded injected-token
efficiency with `--min-quality-per-1k-injected-tokens embedded=1.0` and
`--min-answer-at-5-per-1k-injected-tokens embedded=1.0`.
The backend shootout release gate also checks medium-scale embedded runtime evidence against
`reports/backend-shootout/longmemeval-40-backend-shootout.json` with backend
scoped performance thresholds: `--min-projection-events-per-second
embedded=40`, `--max-cold-bootstrap-ms embedded=250`,
`--max-first-useful-init-ms embedded=15000`,
`--max-first-checkout-ms embedded=50`,
`--max-append-to-projection-p95-ms embedded=35`,
`--max-resident-memory-delta-bytes embedded=768000000`,
`--max-on-disk-footprint-bytes embedded=256000000`,
`--max-dashboard-graph-load-ms embedded=500`,
`--max-rebuild-recovery-ms embedded=15000`, `--max-checkout-p95-ms
embedded=100`, and `--max-checkout-p99-ms embedded=85`. The same command now guards token efficiency and lane latency
directly with `--min-quality-per-1k-returned-tokens embedded=0.10`,
`--min-answer-at-5-per-1k-returned-tokens embedded=0.10`,
`--min-quality-per-1k-injected-tokens embedded=0.10`,
`--min-answer-at-5-per-1k-injected-tokens embedded=0.10`,
`--max-exact-p95-ms embedded=15`, `--max-exact-p99-ms embedded=10`,
`--max-keyword-p95-ms embedded=75`, `--max-keyword-p99-ms embedded=40`,
`--max-vector-p95-ms embedded=25`, `--max-vector-p99-ms embedded=35`, `--max-traversal-p95-ms embedded=10`, and `--max-traversal-p99-ms embedded=10`.
Finally, the release gate checks 100-query embedded scale evidence at
`reports/backend-shootout/longmemeval-100-backend-shootout.json` with the same
token-efficiency dimensions and a looser scale latency ceiling, including
`--max-cold-bootstrap-ms embedded=600`,
`--max-first-useful-init-ms embedded=45000`,
`--max-first-checkout-ms embedded=150`,
`--max-append-to-projection-p95-ms embedded=40`,
`--max-resident-memory-delta-bytes embedded=1700000000`,
`--max-on-disk-footprint-bytes embedded=512000000`,
`--max-dashboard-graph-load-ms embedded=500`,
`--max-rebuild-recovery-ms embedded=45000`,
`--max-checkout-p95-ms embedded=200`,
`--max-checkout-p99-ms embedded=250`,
`--max-keyword-p95-ms embedded=20`,
`--max-keyword-p99-ms embedded=15`,
`--min-recall-at-5 0.90`,
`--min-quality-per-1k-injected-tokens embedded=0.15`, and
`--min-answer-at-5-per-1k-injected-tokens embedded=0.15`.

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

The v0.9 release gate also names every public smoke path it expects:
`EXAMPLES_SMOKE_CMD`, `MCP_SMOKE_CMD`, `LANGGRAPH_SMOKE_CMD`,
`COORDINATE_SMOKE_CMD`, docs validation, backend benchmark comparison, and
`BETA_UAT_CMD`. A command may be set to `SKIP:<reason>` only when the skip is
intentional and auditable; otherwise `zaxy doctor --beta-readiness` reports the
missing surface through `release_gate_surface_coverage`.

The current v0.9 gate evidence is summarized in
[v09-gate-audit.md](v09-gate-audit.md). That audit intentionally leaves the
external-user feedback gate pending until feedback from outside the current
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
query latency, and competitive retrieval harness behavior. Benchmarks are useful
for detecting large regressions, but correctness tests decide release readiness.
For live comparative statistics against markdown, BM25, vector,
markdown+vector, and Zaxy retrieval, run the statistically powered workload. The
wrapper uses deterministic hash embeddings and the embedded projection backend
by default, so this path is offline and reproducible:

```bash
scripts/live-benchmark.sh --workload statistical --subjects 100 --runs 1 --reset-graph
```

OpenAI mode is opt-in with `--embedding-provider openai` and uses
`OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`, and `EMBEDDING_DIMENSION`. The
default hosted model is `text-embedding-3-small`. The same wrapper also passes
through `--embedding-provider local-http` and `--embedding-provider
sentence-transformers` for local provider comparisons.
The script writes `reports/benchmarks/live-benchmark.json` for automation and
`reports/benchmarks/live-benchmark.md` for human review.
For sidecar comparisons, set `--projection-backend neo4j` with `--neo4j-uri`,
`--neo4j-user`, and `--neo4j-password` or the matching `NEO4J_*` environment
values. Set `--projection-backend pggraph --pggraph-dsn ...` to pass a
PostgreSQL/pgGraph DSN.
Use `--baseline-backends`, `--zaxy-backend`, `--external-results`, and
`--reuse-projection` on the wrapper when reproducing the published comparison
commands.
Add `--dry-run` to print the exact `zaxy benchmark` command without starting a
benchmark run.

For publishable comparisons, use the frozen workload instead of a custom
subject count:

```bash
scripts/live-benchmark.sh --workload frozen --runs 1 --reset-graph
```

For MemPalace-comparable temporal recall beyond the original frozen statistical
lane, use the dedicated temporal workload. It creates three time-versioned
preference states per subject, queries each state with an explicit as-of point,
and reports citation coverage for otherwise successful retrievals:

```bash
scripts/live-benchmark.sh --workload temporal-recall --subjects 100 --runs 1 --reset-graph
```

For MemPalace-comparable source recall, use the dedicated source workload. It
creates a target document and a near-miss distractor per case, then reports
whether retrieval returned the exact expected source path as a separate source
recall metric:

```bash
scripts/live-benchmark.sh --workload source-recall --documents 100 --runs 1 --reset-graph
```

For MemPalace-comparable graph traversal, use the dedicated graph workload. It
creates a goal, linked task, and completion actor per case, then asks for the
actor who completed the task connected to the goal:

```bash
scripts/live-benchmark.sh --workload graph-traversal --subjects 100 --runs 1 --reset-graph
```

For MemPalace-comparable context-collapse behavior, use the dedicated
context-collapse workload. It creates noisy same-session transcript turns before
a compact checkpoint, then asks for the preserved decision that should survive a
small context window:

```bash
scripts/live-benchmark.sh --workload context-collapse --sessions 100 --runs 1 --reset-graph
```

To inventory the comparable proof lanes without Neo4j or hosted embeddings,
generate the frozen workload metadata directly. This is useful for release notes
and CI checks that need versions, SHA-256 fingerprints, event/query counts,
product claims, and required metrics without running retrieval:

```bash
zaxy benchmark-inventory --json
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
scripts/live-benchmark.sh --workload suite --subjects 100 --documents 250 --sessions 50 --runs 1 --reset-graph
```

Suite reports disclose subject, document, session, lane, event, query, and
SHA-256 workload metadata. Increase `--subjects`, `--documents`, and
`--sessions` for capacity tests after the smoke run is stable.

Use `zaxy benchmark-compare` to enforce beta guardrails on a completed report.
The command exits non-zero when quality drops below the configured floors, p95
or p99 exceed latency budgets, or latency regresses too far from a baseline:

```bash
zaxy benchmark-compare reports/benchmarks/live-benchmark.json \
  --baseline reports/benchmarks/baseline-live-benchmark.json \
  --min-mean-score 0.95 \
  --min-answer-recall-at-5 0.95 \
  --min-recall-at-5 0.99 \
  --min-citation-coverage 0.95 \
  --max-p95-ms 500 \
  --max-p99-ms 750
```

For the current full 500-question LongMemEval-compatible headline, use the
current74 quality floors and set latency budgets from the archived report. The
current headline report is
`reports/benchmarks/longmemeval-500-current74-zaxyonly-gated-relative-temporal-anchor-embedded-reuse-20260604/live-benchmark.json`:
mean score 0.940, Answer@5 0.906, citation coverage 1.000,
R@1/R@5/R@10 0.906/1.000/1.000, p95 687.67 ms, and p99 969.10 ms.
The current same-harness BM25 comparison remains archived at
`reports/benchmarks/longmemeval-100-comparison/live-benchmark.json`: BM25 R@5
0.840 versus Zaxy checkout R@5 0.990 on the same 100-question slice. See
[benchmarks.md](benchmarks.md) for public copy rules and external disclosure
links for MemPalace, Mem0, and Agent Memory.

The legacy BM25-included `limit=10` full 500-question LongMemEval-compatible
archive is `reports/benchmarks/longmemeval-500-hash/live-benchmark.json`. Its
Zaxy checkout floor remains mean score 0.626, Answer@5 0.608, citation coverage
1.000, and R@5 0.956. Keep it as historical BM25-included evidence, not as the
current public headline.

The current same-harness `limit=5` backend-evaluation archive is
`reports/benchmarks/longmemeval-500-neo4j-current-checkout/live-benchmark.json`.
Its workload SHA-256 is
`0dc36a139bb9a4fdc7c6cd34400737a58a1eb7410517341f015e9fbfc76ed854`. Its Zaxy
checkout floor is mean score 0.714, Answer@5 0.626, citation coverage 1.000,
R@5 0.958, p95 1089.53 ms, and p99 2456.86 ms. Use the floor matching the
candidate harness; projection-backend work should use the current same-harness
control unless it also reruns the legacy `limit=10` harness.

Guard the current headline report with the current74 floors:

```bash
zaxy benchmark-compare reports/benchmarks/longmemeval-500-current74-zaxyonly-gated-relative-temporal-anchor-embedded-reuse-20260604/live-benchmark.json \
  --backend zaxy-checkout \
  --min-mean-score 0.940 \
  --min-answer-recall-at-5 0.906 \
  --min-recall-at-5 1.000 \
  --min-citation-coverage 1.0 \
  --max-p95-ms 800 \
  --max-p99-ms 1100
```

Guard projection-backend comparisons against the same-harness backend-control
archive:

```bash
zaxy benchmark-compare reports/benchmarks/longmemeval-500-neo4j-current-checkout/live-benchmark.json \
  --backend zaxy-checkout \
  --min-mean-score 0.714 \
  --min-answer-recall-at-5 0.626 \
  --min-recall-at-5 0.958 \
  --min-citation-coverage 1.0 \
  --max-p95-ms 1200 \
  --max-p99-ms 2500
```

To verify every archived public LongMemEval guardrail from a clean checkout, use
the script wrapper. It checks the cached LongMemEval dataset, cached embeddings,
archived JSON reports, benchmark inventory, and all published guardrail
thresholds:

```bash
scripts/benchmark-guardrails.sh
```

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
