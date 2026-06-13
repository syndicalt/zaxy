# Migration Guide

This guide covers the public upgrade path from Zaxy 0.4 through 2.x. Eventloom
logs are append-only; migrations should add events,
projection metadata, indexes, docs, or compatibility shims instead of rewriting
historical records.

## Upgrade Checklist

Before and after each upgrade:

1. Back up `.eventloom/`, projection directories, and any sidecar database.
2. Run `zaxy memory status --graph` to verify Eventloom integrity and graph
   projection lag before changing code.
3. Run `zaxy doctor --beta-readiness` from the target checkout.
4. Rebuild or refresh projections with `zaxy refresh-context` or
   `zaxy reproject` when the release notes mention projection changes.
5. Re-run the documented examples or mission smoke that your integration uses.
6. Capture the final command evidence in Eventloom with
   `verification.recorded` or your normal lifecycle capture path.

## From 0.4 to 0.5

0.5 reframed the public path around Coordinator Memory for Agent Teams. The
memory substrate remains compatible with 0.4 Eventloom logs, but docs and
examples now expect the first-run flow to prove install, init, bootstrap,
checkout, doctor, and example timing.

Required actions:

- Read the updated getting-started and MCP quickstart docs before changing
  client configuration.
- Keep existing Eventloom files in place; do not flatten them into markdown or
  vector-only stores.
- Run `zaxy init` or the appropriate preset only after confirming it will not
  overwrite local MCP config unexpectedly.
- Validate public examples with the release smoke path if you package Zaxy for
  other users.

## From 0.5 to 0.6

0.6 introduced the shared native integration contract and stronger MCP response
shape checks. Existing MCP clients should keep working, but code that parses
checkout metadata should expect the `zaxy.native.v0.6` contract in native
adapter responses.

Required actions:

- Update LangGraph or CrewAI adapters to read the `zaxy.native.v0.6` metadata
  envelope instead of relying on ad hoc keys.
- Prefer `memory_bootstrap`, `memory_checkout`, and `context_assemble` for
  model-facing state instead of assembling prompt memory manually.
- Check `docs/examples/mcp-tool-contract.json` and
  `docs/examples/mcp-response-snapshots.json` when updating client tests.
- Run the MCP and native example smoke tests after changing integration code.

## From 0.6 to 0.7

0.7 added Zaxy Coordinate mission workflows. This is additive for single-agent
memory users, but multi-agent clients should move from informal transcript-only
coordination to explicit mission, worker, assignment, finding, review,
promotion, approval, checkout, and handoff events.

Required actions:

- Use Coordinate APIs or MCP tools for team state instead of storing accepted
  findings as unstructured transcript text.
- Inspect migrated mission state with `zaxy coordinate inspect`.
- Preserve finding evidence and review decisions so promoted memory remains
  auditable.
- If you use benchmark claims, regenerate CoordinationBench reports from
  tracked workloads instead of local scratch files.

## From 0.7 to 0.8

0.8 added outside-MCP model-call wrappers and provider-neutral trace export.
These paths are optional; MCP remains the framework-neutral integration route.

Required actions:

- Use the OpenAI-compatible or Claude-compatible adapters only when model calls
  happen directly in application code and you want checkout injection there.
- Keep model-call capture opt-in and bounded. Do not record raw provider
  secrets or full private request bodies in Eventloom.
- Use `zaxy trace export` when a run needs replay-derived mission, checkout,
  model-call, tool-call, review, promotion, or handoff correlation without a
  tracing vendor.
- Use `zaxy replay --from-seq` and `--to-seq` for bounded inspection of long
  sessions rather than splitting or rewriting logs.

## From 1.0 to 1.1

1.1 adds StateRecoveryBench and replay-derived Coordinate accepted-state
resolution. This is additive for existing memory users, but Coordinate clients
should treat parent-promoted accepted state as the only answerable mission state
by default.

Required actions:

- Use `zaxy state-recovery-benchmark` for accepted-state recovery claims.
  `zaxy experimental state-recovery` remains a research/debug command.
- Keep benchmark claims scoped to the production `memory_fabric_checkout`
  baseline. Associative projection rows are diagnostic only.
- If a client consumes Coordinate proof packets, expect `accepted_finding_ids`,
  review refs, promotion refs, worker source refs, and non-authoritative row
  labels to come from one replay-derived resolver.
- Preserve review and promotion events in the parent mission session; worker
  findings alone are diagnostic until promoted.

## From 0.8 to 0.9

0.9 was the API freeze candidate. The main migration task is to compare your
client usage with `docs/api-inventory.md` and remove dependencies on surfaces
marked `Internal`. Surfaces marked `Experimental` can remain in prototypes, but
they should not be required for a production 1.0 rollout.

Required actions:

- Compare MCP tools, Python imports, CLI commands, Eventloom event types,
  projection backend usage, and benchmark artifacts against
  `docs/api-inventory.md`.
- For `Stable` or `Beta` surfaces, add compatibility tests before changing
  payload keys, command options, event names, or report schemas.
- Treat pgGraph and LatticeDB backend use as experimental unless the current
  release gate explicitly promotes them.
- Do not make public benchmark claims from untracked Eventloom, query, or
  diagnostics files.

## From 0.9 to 2.0

The frozen v0.9 surfaces carry forward; 1.x and 2.0 changes are additive or
follow the freeze policy in `docs/api-inventory.md`. Notable upgrade items:

- The PyPI distribution is `zaxy-memory`; the import package and console
  command remain `zaxy`.
- 1.1.x adopted Eventloom v1 JSONL envelopes while preserving legacy log
  replay; see the changelog for envelope details.
- 2.0 added causal memory contracts, review-gated consolidation, and
  reasoning-loop primitives as new (additive) surfaces.
- Benchmark and evaluation modules moved out of the runtime package into
  `zaxy_benchmarks`; production imports of `zaxy.*benchmark*` modules should
  be removed (they were `Internal`/`Experimental` surfaces).

## From 2.0 to 2.1+

The 2.1–2.3 agent-experience and cognitive-memory surfaces are entirely
additive — no Eventloom envelope changes, no projection migrations, no
removed or renamed tools — but **2.1.0 flips two defaults**, each promoted by
its measurement lane per the roadmap:

1. **`MCP_TOOL_PROFILE` defaults to `core`** (was `full`). The internal
   tool-adoption lane measured the listed tool surface at 8,165 → 1,344
   estimated tokens (an 83.5% reduction) with `memory_checkout` listed first.
   Profiles change listing only; dispatch is never filtered, so every tool
   stays callable by name. **Opt out: `MCP_TOOL_PROFILE=full`** (or
   `zaxy serve --profile full`). Affected: clients that snapshot tool
   listings or pin the listed-tool count — refresh snapshots or pin `full`.
2. **`RETRIEVAL_PROFILE` defaults to `cognitive`** (was `local_fast`). The
   internal forgetting lane measured exact cold-start parity
   (zero-reinforcement corpora rank byte-identically to plain), no-recall-loss
   1.0 (attenuated memories stay reachable by explicit query, labeled in
   diagnostics), pin/authority exemptions 1.0, and ranking lift 1.0 vs 0.0.
   The cognitive profile composes the same local retrieval stack as
   `local_fast` plus the salience-ranking, cue-blending, and graph-walk
   flags. **Opt out: `RETRIEVAL_PROFILE=local_fast`** for the previous plain
   default. Affected: users with reinforcement history or graph structure the
   walk can traverse may see ranking shifts; cold-start users (no
   reinforcement events) see identical ranking. Settings that leave the
   profile unset but explicitly customize embedding/reranker/scoring knobs
   keep resolving to the `custom` profile exactly as before the flip.

Unchanged defaults: the write-time encoding gate stays off
(`ENCODING_GATE_ENABLED=false`, unmeasured) and vector quantization stays
`none`. One default moved defensively: `VECTOR_ANN_THRESHOLD` rose from
`50000` to `1000000` because the internal vector-scale lane measured the
HNSW path below exact search on both recall and latency at 10^5 vectors —
the raise prevents auto-engaging a measured-worse path; lower it explicitly
to opt in. The
feeling-of-knowing surface remains experimental (its calibration lane Brier
margin over the base-rate predictor is +0.001–0.006 — too thin for
promotion). Beyond the two flips above, existing clients upgrade without
action; everything below is opt-in.

New settings (all documented in [configuration.md](configuration.md)):

- `MCP_TOOL_PROFILE` (`core` default since 2.1.0, or `full`): MCP tool
  listing profile. `core` lists only the front-door verb set; profiles change
  listing, never capability — every tool stays callable by name.
- `SALIENCE_HALF_LIFE_DAYS` (default `30.0`): recency-decay half-life for the
  salience reinforcement ledger.
- `SALIENCE_FLOOR` (default `0.15`): attenuation floor under the `cognitive`
  retrieval profile; below-floor memories leave default checkout ranking but
  remain reachable by explicit query.
- `ENCODING_GATE_ENABLED` (default `false`): tag appends as
  `novel`/`reinforcing`/`redundant`; events are always appended and
  hash-chained regardless.
- `VECTOR_ANN_THRESHOLD` (default `1000000` in 2.1.0, lowered to `100000` in
  2.2 — see the 2.2 subsection below): per-scope vector count at which the
  embedded backend uses an engine-native HNSW index; results from the
  approximate path report `exact: false`.
- `VECTOR_QUANTIZATION` (`none` default, or `int8`): opt-in quantized vector
  storage with exact float rerank of oversampled candidates.

New MCP tools (additive; no existing tool was removed or renamed):

- `memory_feeling_of_knowing`: experimental metamemory pre-check, listed in
  the `core` profile; each call appends a non-authoritative
  `metacognition.fok.predicted` calibration marker.
- `memory_consolidation(operation, ...)` and `memory_confidence(operation,
  ...)`: umbrella tools over the existing consolidation and
  confidence/metacognition handlers; the single-purpose tool names remain
  available per the stability commitment.

Both profiles now list `memory_checkout` first, and every core-profile tool
description names checkout as the front door. Clients that snapshot tool
listings or descriptions should refresh against
`docs/examples/mcp-tool-contract.json`.

New CLI surfaces:

- `zaxy serve --profile core|full` overrides `MCP_TOOL_PROFILE` per process.
- `zaxy memory checkout --max-tokens N` (and the `max_tokens` argument on the
  `memory_checkout` and `context_assemble` MCP tools) applies a token budget
  with greedy packing; diagnostics report `budget_requested`, `budget_used`,
  and `elided`, plus `stable_prefix_chars` for the cache-stable consolidated
  prefix.
- `zaxy memory mine-procedures` mines recurring successful tool sequences into
  review-pending procedure candidates; nothing becomes authoritative without
  consolidation review.
- `zaxy memory re-embed` batch-migrates projected vectors to the active
  embedding provider version tag; Eventloom events are never rewritten.
- `zaxy hook-event session-resumed` records `hook.session_resumed` and prints
  a cited compaction-recovery packet; see [hooks.md](hooks.md).
- `zaxy agent-experience-lanes` runs the internal tool-adoption, budget, and
  cache lanes (internal validation labels only; not public claims).

New event kinds and payload conventions (additive Eventloom payloads):

- `memory.reinforcement` with `kind: surfaced | confirmed | promoted |
  invalidated`, batched target event refs, and a source id; salience is fully
  rebuildable by replaying these events.
- `metacognition.fok.predicted`: non-authoritative feeling-of-knowing
  calibration markers.
- `hook.session_resumed`: post-compaction resume lifecycle events.
- Payload conventions: an optional `cues` mapping (`mission`, `workspace`,
  `tool`, `phase`) for encoding-specificity blending, an `encoding` tag
  written only when the gate is enabled, and `"pinned": true` to exempt a
  memory from the attenuation floor.

The cognitive retrieval profile (the default since 2.1.0) enables
salience-blended ranking, cue-overlap blending, and personalized-PageRank
graph-walk retrieval over the unchanged immutable log. Attenuation is
projection policy only — labeled in `diagnostics.attenuation`, exempting
pinned and authority-bearing memories, and reversible by switching to
`RETRIEVAL_PROFILE=local_fast`. See [retrieval.md](retrieval.md). The lean
`core` tool listing is likewise the default since 2.1.0; restore the full
listing with `MCP_TOOL_PROFILE=full` or `zaxy serve --profile full`;
`memory_capabilities` reports the active profile and the
available-but-unlisted tools.

### 2.2: ANN engagement defaults

2.2 changes when the embedded backend's HNSW vector path engages — nothing
else in the upgrade path changes. `VECTOR_ANN_THRESHOLD` drops from `1000000`
to `100000`, gated by a new dimension ceiling `VECTOR_ANN_MAX_DIMENSION`
(default `64`), and engagement gains a second clause: within the ceiling, a
(session, version) vector scope also engages ANN when its exact float64
matrix (count × dimension × 8 bytes) would exceed the 256 MiB vector index
cache byte budget — above 524,288 rows at dimension 64. The lowered count
default is backed by two consecutive vector-scale lane passes at exactly
10^5 vectors (dimension 64) with recall@10 of 1.0 on both the strict and
tie-aware metrics, ANN p50 at-or-better than exact in-run, and resident
bytes improved
(`docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json`/`-r2.json`);
the ceiling is that evidence's measured envelope — at dimension 1536/50k
gaussian the lane measured HNSW recall@10 of 0.6 against exact at 22ms p50
(`ann3-d1536-50k-gauss-crossover.json`; 0.6344 on a rerun, and 0.8438 at
`efs` 800 with worse-than-exact latency), so higher-dimensional scopes stay
on exact (or opted-in int8) search regardless of size. **Opt out:**
`VECTOR_ANN_THRESHOLD=1000000` restores the 2.1 count behavior, and
`VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT=false` disables the byte clause; a
single over-budget exact matrix stays resident as a cache of one (the
budget bounds multi-scope cache totals). An explicit
`VECTOR_QUANTIZATION=int8` opt-in keeps its precedence below the count
threshold and is exempt from the byte clause. `memory_capabilities` reports
the effective rule under `vector_search`.

## From 2.2 to 2.3: embedded engine moves to LadybugDB

2.3 replaces the embedded projection engine. The Kuzu project was archived
upstream in October 2025 with 0.11.3 as its final release: it already ships
no macOS or Windows wheels for Python 3.14, and no Python 3.15 wheels will
ever exist (3.15 lands ~October 2026), so staying was a countdown, not an
option — **there is no opt-out**. Zaxy now exact-pins
[LadybugDB](https://github.com/LadybugDB/ladybug) (`ladybug==0.17.1`), the
actively maintained fork of the same engine. The adoption was gated on
re-running Zaxy's own defect reproductions and the full vector-scale and
graph-walk lanes against the fork.

What changes for you:

- **Storage format.** LadybugDB cannot read a projection file written by the
  pre-fork engine. On first open of an old `embedded.kuzu`, Zaxy
  automatically moves it (and its `.wal`) aside to
  `<path>.pre-ladybug.bak` — **no data is deleted** — opens a fresh store at
  the same path, and logs the rebuild instructions. Projections are derived
  state: replay full history with `zaxy reproject <eventloom log>`, or let
  new checkins project incrementally. `zaxy doctor` reports a leftover
  `.pre-ladybug.bak` with a remediation line; it is safe to delete once the
  rebuilt projection is verified. Eventloom logs — the source of truth — are
  untouched.
- **Paths stay put.** The default projection path remains
  `.eventloom/projections/embedded.kuzu`; the extension is cosmetic and
  changing the default would orphan configured paths.
- **Dependency.** `pip install "zaxy-memory"` now pulls `ladybug==0.17.1`
  instead of `kuzu`. The pin is exact on purpose (single-maintainer fork;
  every defect verdict was verified against this version) — and pins the
  `ladybug` PyPI name, not the stale pre-rename `real-ladybug`.
- **Behavior.** Query semantics, schema, retrieval results, and the ANN
  engagement defaults are unchanged; the vector index code is unchanged
  since the fork, and the 2.3 lane reruns reproduced the 2.2 deterministic
  blocks (identical corpus hashes and recall). Superseded ANN shadow
  generations are now dropped instead of emptied (full space reclaim,
  enabled by an upstream fix verified through Zaxy's own rebuild cycle).
- **Vector extension fetch (local-first, one-time).** Kuzu 0.11.3 bundled the
  vector index; LadybugDB ships it as an official `vector` extension fetched
  on first use (a small one-time download cached under `~/.lbdb`), much like
  installing the package itself. Zaxy fetches it automatically the first time
  the ANN path engages, then it runs entirely on-box. Air-gapped deployments
  that want ANN run `INSTALL vector` once on a networked host and ship the
  `~/.lbdb` cache; with no network and no cache, approximate (HNSW) search is
  simply unavailable and retrieval falls back to **exact** float search —
  correct results, no error, fully on-box. The default exact path is pure
  NumPy and needs nothing fetched.

## Compatibility Tests

The compatibility suite should prove that old public behavior still works or
that a migration path is documented:

- MCP schema tests compare advertised tool names and input schemas.
- Response snapshot tests cover `memory_bootstrap`, `memory_checkout`,
  `memory_query`, `memory_verbatim`, `context_assemble`, `memory_feedback`, and
  coordination checkout outputs.
- Event and graph tests cover hash-chain replay, projection provenance,
  invalidation, temporal versions, inferred-edge metadata, and source
  citations.
- CLI tests cover release smoke, beta readiness, replay windows, trace export,
  Coordinate inspection, and schema migration planning.
- Benchmark guardrail tests reject reports without tracked inputs, query
  diagnostics, citation coverage, latency budgets, and fingerprint validation.

When a compatibility break is intentional, update this guide, the changelog,
the API inventory, and the narrow test that proves the old behavior is rejected
or migrated. For stable or beta surfaces listed in
`docs/examples/v1-schema-freeze.json`, record the change with
`schema.migration.proposed` before implementation and
`schema.migration.applied` after verification.

## Rollback Policy

Rollback should restore code and projections without rewriting Eventloom:

- Revert the application code or package version.
- Restore projection directories or sidecar database snapshots if projection
  schema changed.
- Keep new Eventloom records as audit history. If a fact is no longer valid,
  append `memory_invalidate` or a superseding typed event.
- Run `zaxy memory status --graph` after rollback to confirm the log and graph
  agree.
- Record rollback verification with `verification.recorded` so future checkout
  can cite the operational decision.

Related references: [api-inventory.md](api-inventory.md),
[testing.md](testing.md), [runbook.md](runbook.md), and
[README.md](../README.md).
