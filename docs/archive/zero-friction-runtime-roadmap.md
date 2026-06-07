# Zero-Friction Runtime Roadmap

Zaxy's next product goal is to make durable graph memory feel as cheap to adopt
as markdown files while preserving the capabilities that markdown cannot
provide: temporal state, provenance, replay, graph traversal, citations,
automatic capture, and memory activation.

The bar is not "a simpler Zaxy." The bar is frontier-grade memory: Zaxy should
beat every credible memory, context, and retrieval product on the combination of
onboarding efficiency, runtime efficiency, token efficiency, retrieval quality,
auditability, and agent activation. If existing architectures cannot hit that
bar, Zaxy should invent new techniques instead of settling for a weaker
abstraction.

The current Neo4j and pgGraph paths prove the memory model, but they expose too
much infrastructure during first use. The next runtime must make graph storage,
capture, checkout, and dashboard visibility an implementation detail of `zaxy
init` rather than a separate operator concern.

## Product Bar

A new repo should become useful with one command:

```bash
zaxy init
```

The command should leave the workspace in this state:

- `.eventloom` exists and remains the canonical event source.
- A repo-local embedded graph projection exists or can be created lazily.
- MCP config, supported hooks, and capture defaults are installed safely.
- The first agent session receives bootstrap and checkout guidance without the
  user having to remember separate commands.
- `zaxy status`, `zaxy hook-status`, and the dashboard show whether memory was
  captured, checked out, stale, or unavailable.
- No Docker container, Bolt URI, Postgres DSN, graph password, or service start
  command is required for default local use.
- Context injection follows ruthless token discipline: every injected byte
  should be ranked, cited when possible, bounded, and justified by the current
  task.

Bare `zaxy init` now expands to the no-sidecar local Codex path when no
explicit client, output, or backend options are supplied. It selects
`PROJECTION_BACKEND=embedded`, writes the repo-local deterministic capture
config, checks repo-local embedded projection posture, records session genesis
and heartbeat events, and prints the Codex MCP install and verification steps
without requiring Neo4j, pgGraph, Docker, or a graph service endpoint. It does
not start a background watcher unless `--capture start` is supplied. Embedded is
now the production settings default; Neo4j and pgGraph remain explicit sidecar
adapters for control runs, interoperability, and experiments. `--preset
local-embedded-codex` remains available as an explicit spelling for scripts that
should not rely on the bare-init default.

## Runtime Strategy

Eventloom remains the source of truth. Graph backends are projection engines.
The default projection should move from sidecar-first to embedded-first:

1. **Embedded graph default**: Evaluate Kuzu as the first in-process graph
   projection candidate because it keeps graph semantics without a daemon.
2. **LatticeDB candidate track**: Evaluate LatticeDB as a first-class embedded
   candidate because its stated shape is unusually close to Zaxy's target:
   single-file property graph, native HNSW vector search, BM25 full-text search,
   Cypher-style traversal, and durable streams/changefeeds. It must pass the
   same `ProjectionStore` contract and same-harness benchmark gates before it
   can displace Kuzu or become the embedded default.
3. **pgGraph collaboration track**: Keep pgGraph as the Postgres-native path and
   work with upstream on embedded packaging or library mode if that can remove
   sidecar friction without losing traversal quality.
4. **Neo4j control backend**: Keep Neo4j as the quality and performance control
   until an embedded backend passes the same guardrails.

The embedded backend implements the existing projection contract instead of
forking retrieval logic. Any future embedded candidate must preserve that shared
runtime surface before it can displace the current Kuzu default.

## Memory Activation Layer

MCP availability is not a forcing function. Zaxy needs activation mechanics that
make memory use the default behavior of a session:

- A launcher path such as `zaxy activate codex` or `zaxy run codex` injects
  current bootstrap and checkout context before the model starts work.
- Repo instructions installed by `zaxy init` require checkout before roadmap,
  implementation, release, review, resume, and high-context questions.
- Passive capture continues even when the model does not call Zaxy tools.
- Dashboard and CLI status expose last checkout, last capture, stale checkout,
  and sessions doing meaningful work without recent memory use.

Current status: `zaxy activate codex` emits a prompt-ready session-start
activation packet, records a bootstrap activity marker, and includes the
recommended first `memory_checkout` call. `zaxy activate codex --launch` starts
Codex with that activation packet as the initial prompt via the supported Codex
`[PROMPT]` argument; `--dry-run` prints the exact launch command.
- Session genesis now persists a `write_instructions.memory_activation` policy
  for codebase and generic workspaces. The policy names `memory_checkout` as the
  required tool before roadmap, implementation, release, review, resume, and
  high-context questions, includes the activation launcher, resume hook, and CLI
  checkout fallback commands, marks MCP tool availability as runtime-unverified,
  and records a fail-closed blocker when no fresh checkout is present. This keeps
  the checkout rule in Eventloom during `zaxy init` instead of living only in
  docs.
- Codex onboarding next steps now prefer `zaxy activate codex --launch` as the
  session entrypoint and also print the resume boundary and CLI checkout fallback
  commands for sessions recovering after `/resume`, Codex update, or MCP churn.
- `zaxy init` now installs a marker-managed `Zaxy Memory Activation` block in
  `AGENTS.md` by default, with `--no-agent-instructions` as the escape hatch. The
  block gives Codex a model-visible fail-closed activation rule even when MCP
  tools are absent after a resume or tool reload.
- Zaxy emits `memory.reminder.suggested` as the auditable warning/proposal event
  when memory activation appears missing for risky work, and `zaxy hook-status`
  surfaces the latest reminder under `memory_activation.latest_reminder`.
  Dashboard parity now includes a read-only `Latest reminder` metric backed by
  the same `memory_activation.latest_reminder` status field, so operators can
  see whether activation pressure was suggested without opening raw Eventloom
  JSONL.

## Efficiency Targets

The embedded runtime gate reports these metrics for every candidate backend:

- first useful `zaxy init` time
- cold bootstrap and first checkout latency
- append-to-projection latency
- replay/rebuild throughput
- exact, keyword, vector, and traversal query p50/p95/p99
- returned bytes and approximate returned tokens
- LongMemEval Answer@5 and Recall@5
- citation coverage
- dashboard graph load time
- resident memory and on-disk footprint
- recovery time after deleting the projection artifact
- token efficiency: answer quality per returned token and per injected token
- activation efficiency: percentage of high-context sessions with a fresh
  checkout before the first substantive answer

The embedded default must beat sidecar onboarding friction and must not regress
Zaxy's published quality floors. A faster setup that loses citations, temporal
semantics, or traversal quality is not acceptable.

## Frontier Research Tracks

The embedded runtime work should leave room for new techniques when ordinary
retrieval is not enough:

- **Active graph checkout**: assemble context from graph state changes, not only
  user query similarity.
- **Evidence-first token budgeting**: allocate tokens to claims, sources, and
  paths by expected answer impact rather than fixed lane quotas.
- **Continuity pressure**: detect sessions that are acting without fresh memory
  and inject or warn before stale work accumulates.
- **Path-aware compression**: compress graph neighborhoods while preserving
  source paths, temporal windows, and contradiction edges.
- **Self-benchmarking memory**: every backend and activation strategy should
  produce comparable quality, latency, and token-efficiency evidence.

## Runtime Gates

The embedded graph runtime is now the default only because these gates passed:

- `PROJECTION_BACKEND=embedded` works through `MemoryFabric`, MCP, dashboard,
  `zaxy memory status`, `zaxy reproject`, and context assembly.
- Eventloom replay can rebuild the embedded graph deterministically.
- Exact, keyword, vector, and traversal lanes either work or degrade with
  explicit diagnostics.
- Temporal versioning, invalidation, source citations, Event nodes, hash-chain
  links, inferred-edge metadata, and session isolation match Neo4j semantics.
- The same-harness benchmark output compares embedded, LatticeDB, Neo4j,
  pgGraph, and BM25 with latency, quality, citation, and token-efficiency
  fields.
- The default docs show one-command local setup with no external graph service.

Current gate status: embedded runtime: passed and promoted to default. The
adapter, activation status, and shootout harness remain under deeper
same-harness quality and latency runs.
MCP embedded runtime status: wired. With `PROJECTION_BACKEND=embedded`, MCP
startup now uses the embedded local projection runtime instead of probing or
starting Neo4j, while still constructing the graph through the backend-neutral
`ProjectionStore` factory.
Embedded graph-traversal gate status: passed focused 10-subject smoke after
matching Neo4j's undirected traversal semantics and carrying active
relationships forward across temporal entity reassertions. The archived report
at `reports/benchmarks/backend-shootout-graph-traversal-embedded-after-carry-forward`
shows Zaxy embedded mean score `1.000`, `Answer@5=1.000`, `Recall@5=1.000`,
citation coverage `1.000`, and p50 checkout latency `18.31ms` on the
`mempalace-graph-traversal-v1` smoke. This is strong focused evidence, not a
full default-backend decision.
Checked all-backend smoke status: passed for embedded, LatticeDB, Neo4j,
pgGraph, and BM25 on `reports/backend-shootout/sample.eventloom`, with citation
coverage at 1.0 for every backend. This is a reproducibility gate, not
full-scale default-backend evidence.
Medium-scale embedded runtime evidence: the 40-question LongMemEval-compatible embedded shootout
at `reports/backend-shootout/longmemeval-40-backend-shootout.json` now builds
and serves 100 dashboard nodes and 100 dashboard edges from a
repo-local Kuzu projection. Embedded/Kuzu scored `Answer@5=0.6`,
`Recall@5=1.0`, citation coverage `1.0`, checkout p95 `62.387ms`, lane p95s
of exact `4.202ms`, keyword `18.587ms`, vector `3.619ms`, and traversal
`0.006ms`, cold bootstrap `116.946ms`, first checkout `35.33ms`,
append-to-projection p95 `30.696ms`, projection throughput `48.097` events/sec,
first useful init `10928.55ms`, rebuild recovery `10583.365ms`, resident memory
delta `672075776` bytes, on-disk footprint `28762112` bytes. The answer-ready
contract scored `Answer@5=1.0` and `Recall@5=1.0`, versus BM25 `Answer@5=0.55`,
`Recall@5=1.0`, checkout p95 `187.997ms`, and quality plus Answer@5 per 1k
returned/injected tokens `0.1394`. This is operational scale evidence, not a
default-backend gate: the edges are deterministic session-to-document
relationships. Explicit Kuzu bulk projection transactions plus prewarmed
keyword/vector indexes reduced the earlier roughly 120-second projection/rebuild
cost to roughly 10 seconds while keeping vector retrieval enabled.
Embedded read-index warmup is now part of the runtime path: `MemoryFabric.connect()`
preloads current-entity, keyword, vector, and traversal indexes plus the
Eventloom verbatim source index for the configured session, and `retrieve()`
warms each additional requested projection session at most once. This
intentionally moves Kuzu's cold read-index cost and source-lane BM25 build cost
into startup or the first use of a new session so repeated checkouts do not pay
scattered exact/vector/keyword/traversal/source stalls.
100-query scale evidence status: answer-ready quality passed and the embedded
scale guardrail now passes with raw retrieve `Recall@5=0.99`. The archived report at
`reports/backend-shootout/longmemeval-100-backend-shootout.json` covers 100
queries and 1,559 Eventloom events. Embedded/Kuzu reported projection
throughput `51.797` events/sec, first useful init `30590.521ms`, rebuild
recovery `28743.206ms`, raw retrieve checkout p95 `19.904ms`, lane p95s of
exact `0.006ms`, keyword `5.986ms`, vector `8.009ms`, and traversal
`0.006ms`, cold bootstrap `492.029ms`, first checkout `14.707ms`,
append-to-projection p95 `27.127ms`, resident memory delta `1609691136` bytes,
on-disk footprint `57298944` bytes, citation coverage `1.0`, and 100 dashboard
nodes / 100 dashboard edges. The answer-ready contract scored `Answer@5=0.99`
and `Recall@5=1.0`, first checkout `155.983ms`, and checkout p95 `88.94ms`;
raw retrieve improved to `Recall@5=0.99`, while BM25 scored `Answer@5=0.52`
and `Recall@5=0.9`.
Vector retrieval is enabled in this report. Treat this as strong answer-ready
and embedded scale evidence, not as the end of optimization work.
LatticeDB candidate gate status: failed current active-backend gate. The first
adapter slice is routed through the projection factory, MemoryFabric, and
backend shootout as an explicit candidate, and it projects entities and
relationships with exact, keyword, traversal, temporal invalidation, source
retirement, projection status, Eventloom citation metadata, native vector
search, native full-text search, and inferred-edge audit diagnostics. However,
the first graph-traversal smoke result returned `Answer@5=0.0`, mean score
`0.0`, and roughly `6.4s` p50 checkout latency on a three-query smoke. Keep the
adapter available for explicit `--backends latticedb` experiments, but park it
outside the default active shootout until it passes quality and latency gates.

## First Implementation Slice

Build a branch-local backend shootout before changing defaults:

1. Add a Kuzu-backed `ProjectionStore` adapter behind
   `PROJECTION_BACKEND=embedded`. Runtime status: promoted default for exact,
   keyword, traversal, temporal invalidation, source retirement, Eventloom
   projection status, inferred-edge diagnostics, and vector search with
   cached sparse cosine ranking over projected entity embeddings.
2. Add a backend benchmark command or extend the existing guardrail script so it
   can replay one Eventloom workload into embedded, LatticeDB, Neo4j, pgGraph,
   and BM25.
3. Wire embedded graph status into `zaxy init --infra check`, `zaxy status`,
   `zaxy memory status --graph`, and dashboard graph routes. Embedded infra check status: wired. `zaxy init --projection-backend embedded --infra check`
   now reports repo-local projection path readiness instead of asking for Neo4j
   or pgGraph, and `--infra start` only creates the projection parent directory
   needed for lazy embedded graph creation. `zaxy status --projection-backend embedded`
   reports the same local embedded projection posture without probing
   Neo4j. `zaxy memory status --graph --projection-backend embedded --embedded-graph-path <path>`
   inspects Eventloom-vs-embedded projection integrity through the same
   backend-neutral status path. Bare-profile operational commands are also wired:
   `zaxy memory inferred-status --session-id <session>` reads `.env.local` and
   reports embedded inferred-edge audit status without Neo4j, and
   `zaxy reproject .eventloom/<session>.jsonl --session-id <session>` rebuilds
   the repo-local embedded projection without extra backend flags.
4. Add activation status to CLI/dashboard so users can see whether the model
   actually used Zaxy memory. CLI status now reports last checkout, last
   capture, stale checkout, and no-checkout warnings; dashboard parity is wired
   through `/api/status`, `/api/memory-persistence`, and a read-only
   `/api/checkout` endpoint plus Checkout tab controls that render current
   Memory Checkout diagnostics without appending activity events. Activation
   efficiency is now measurable as the percentage of high-context sessions with
   fresh checkout before the first substantive captured event, with session-level
   fresh, stale, and missing checkout diagnostics surfaced in `zaxy hook-status`
  JSON and the dashboard status payload. The dashboard also shows the latest
  activation reminder as a `Latest reminder` metric sourced from
  `memory_activation.latest_reminder`. Checkout activity markers now carry
  numeric `token_efficiency` diagnostics, and both `zaxy hook-status` and the
  dashboard expose latest checkout prompt tokens plus current facts per 1k
  prompt tokens. Activation status now warns when any high-context session in
  the inspected Eventloom set is missing checkout or has only stale checkout,
  even if another session has a recent checkout. The key KPI is fresh checkout before work starts; the stronger product bar is fresh, efficient checkout
  before work starts.
5. Keep Neo4j as the explicit quality-control backend while embedded remains
   the release default.
