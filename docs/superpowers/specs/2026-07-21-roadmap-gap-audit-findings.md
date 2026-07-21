# Roadmap gap audit — findings and execution plan

- **Status:** audit complete; all findings dispositioned. 13 PRs open and CI-green (#139-#151). Remaining work is listed in §8 with evidence.
- **Date:** 2026-07-21
- **Owner:** Claude Opus 4.8 (orchestrator), with five parallel audit agents

## 1. Why this exists

A roadmap-execution request arrived with no roadmap attached. Rather than guess,
all four roadmap surfaces were audited against actual code by five parallel
agents, each required to cite `file:line`, verify a real caller path, and
classify every capability as SHIPPED-DEEP / SHIPPED-THIN / MISSING.

Headline: **the four named roadmaps are substantially delivered.** `ZAXY-3.md`
I1–I8 shipped through 3.0.0, Coordinate phases 1–5 shipped, BETA lanes shipped,
and five of six `fable-findings.md` backlog items closed in 3.2.0. The real
remaining work is not new features — it is **governance and measurement
integrity gaps inside features that already report as done**.

Two audit hypotheses were disproved and are recorded here so they are not
re-litigated: I5b's hash-chain-after-erasure replay test genuinely exists
(`tests/test_forgetting.py:146,246`), and FleetBench runs a real workload rather
than a stub (verified by execution).

## 2. Findings that were reproduced first-hand

These were not taken on an agent's word; each was re-verified directly.

### 2.1 Gate-withheld rules were assembled into the prompt (FIXED)

Under **default** settings a `partial` outcome yields rule confidence 0.7, below
the 0.85 threshold, so the I4 gate withholds it: `memory.rule.proposed`,
`auto_applied: false`, `review_status: pending`. The rule text nonetheless
reached the model.

It leaked on three independent lanes — the graph lane as an
`event:memory.rule.proposed:N` entity, the verbatim lane as raw payload JSON,
and the synthesis lane, which aggregated it into a generated *preference*
statement recommending the unreviewed rule. The generic projection lanes surface
any event by content match and had no notion of the gate, so this was not a
missing per-lane filter — it was a missing chokepoint.

**Impact:** the I4 evolution gate — the roadmap's stated moat, "Autonomy is
opt-in, reversible, and visible" (§9 invariant 6) — was decorative for rule
generation. A rule held for human review was applied to the model's context
exactly as if approved.

**Fix:** a governance exclusion applied where every lane is materialised
(`core/fabric_checkout.py`), covering the retrieved-context prompt, working set,
and recall candidate set together, with a defense-in-depth pass in
`core/checkout_build.py` for assemblies constructed elsewhere (`mcp_payloads.py`
builds its own). Withheld events remain in the replay/audit trail — this is an
assembly-time filter, never a deletion.

**Controlled verification:** identical lesson text and identical query, differing
only in the outcome that drives the gate decision. Auto-applied (`active`)
surfaces; gate-withheld (`pending`) does not. Encoded as four tests in
`tests/test_outcome_learning.py::TestWithheldRuleIsNotAssembled`, including one
asserting the withheld event is still in the log.

### 2.2 `mypy` was broken on master by dependency drift (FIXED)

CI installs `[dev]` unpinned. `anyio` >= 4.2 types `create_memory_object_stream`
as a generic class, so `mypy --strict` could no longer infer the item type at
`mcp_server.py:2678-2679`. Master last ran green 2026-07-07; today's resolution
(`anyio` 4.14.2, `mypy` 2.3.0) fails. **The next push to master would have failed
the type gate**, invisibly to the green run list.

Fixed by parametrizing both streams as MCP's own transports do. The underlying
cause — unpinned dev dependencies — remains and will recur (see §5).

## 3. Findings surfaced by the audit

Each was reproduced before being fixed — the reproduction output is quoted in the
corresponding PR. None was acted on from an agent's report alone.

Ranked by severity as first written; see §4 for disposition.

### 3.1 CoordinationBench scores against its own gold answers (CRITICAL, blocked)

`_accepted_finding_ids` (`zaxy_benchmarks/coordination_benchmark.py:1608-1618`)
reads `case.gold.expected_accepted_claims` and returns exactly those IDs;
`_run_case` (`:1467-1469`) then accepts and promotes each. **The harness tells
Zaxy the answer, then scores Zaxy against that answer.** Zaxy's acceptance policy
is never exercised; baselines get no equivalent oracle. Compounding it,
`accepted_state_synthesis_quality` and `non_authoritative_leakage` default to
`1.0` (`:225-226`) and are never computed on the Zaxy path, while baseline
conflict precision/recall are typed literals `0.0` (`:546-547`).

Mitigating and confirmed: the report is bit-for-bit reproducible, and
`docs/benchmarks.md:123-134` frames CoordinationBench as an internal guardrail,
**not a public claim**. So this is not a live retraction-class violation. But the
Success Criterion at `docs/coordinate-roadmap.md:612-614` ("beating implemented
flat transcript, markdown, and BM25 baselines") is not supported by this harness,
and its token-efficiency clause is contradicted by the repo's own artifact: Zaxy
`injected_tokens 5151` vs bm25 `481`, flat `1265`, markdown `1603` — 3–10x worse.

An honest gold-free adjudicator already exists and is 96% covered but is **not
wired in**: `src/zaxy/coordinationbench_adapter.py:141-232`.

**Blocked:** changing benchmark methodology requires explicit user approval per
CLAUDE.md. See §6.

### 3.2 Promotion has no policy gate (HIGH)

`promote_finding` (`src/zaxy/coordination.py:817-861`) checks neither a prior
`accepted` review nor evidence presence. An auditor promoted an unreviewed,
evidence-free finding into accepted parent state, rc=0, rendered beneath the
banner `Evidence policy: accepted_parent_state_with_citations_required`
(`src/zaxy/purpose.py:401`) — a declarative string, not an enforced gate. This
makes the roadmap's own framing ("promotes only accepted, cited findings",
`docs/coordinate-roadmap.md:9-10`) unenforced, and Trust-Gated Autopromotion
MISSING.

### 3.3 Dashboard mutations unauthenticated off-loopback (HIGH, security)

`_host_is_loopback` exists (`src/zaxy/cli/serving.py:1952`) and guards the SSE
transport (`:2057`) but is never called by the dashboard command
(`:1906-1948`). `zaxy dashboard --host 0.0.0.0 --enable-coordinate-review`
publishes Eventloom-writing endpoints to the LAN with no auth and no warning. A
same-origin check exists and fails closed (`dashboard.py:1259-1264`) but is the
sole control and is trivially bypassed by a non-browser client.

### 3.4 Default backend silently no-ops two quality claims (HIGH)

`projection_backend` defaults to `embedded` (`src/zaxy/config.py:103`), but
inferred-edge trust scoring and checkout `inferred_context` diagnostics are
driven by `_path_inferred_*` properties produced **only** by the optional Neo4j
path (`graph.py:1387-1403`). Embedded traversal emits only
`_path_relation_types`/`_path_length` (`embedded_graph_internals.py:521-535`), so
`query.py:1207` early-returns multiplier 1.0. Uncited inferred paths are never
downweighted on the default install, and no test catches it because every test
hand-injects the properties. `Source`/`CITES_SOURCE` and
`SUPERSEDED_BY`/`PREVIOUS_VERSION` are likewise Neo4j-only.

### 3.5 The fleet plane has no MCP front door (HIGH)

Cross-agent propagation genuinely works and is decisively tested
(`tests/test_fleet_surface.py:617`). But `fleet_ids` does not exist in the
`memory_checkout` MCP tool schema and there is no CLI checkout flag, so the lane
is reachable only from the Python fabric API. CLAUDE.md names MCP the primary
interface and `memory_checkout` the front door — so an agent fleet driven over
MCP cannot consume fleet memory at all.

### 3.6 Silent no-op rollback (MEDIUM) — RESOLVED

`editable.py:44-53` declared six `ROLLBACKABLE_EVENT_TYPES`, but reversal
semantics existed for only one (`consolidation.candidate.reviewed`). Rolling back
the other five appended an accepted, gated event that changed nothing — the
operator got a success response for a rollback that did not happen.

Closed by making every declared type genuinely reversible and dropping the two
that cannot be:

- `memory.rule.generated` / `memory.rule.proposed` — reversed by the
  assembly-time governance exclusion (`_governance_withheld_event_seqs`).
- `memory.corrected` — reversed by the same exclusion, extended to cover the raw
  `# Recent Events` timeline, which was a second lane-independent route into the
  prompt. Rolling back a correction now restores the retained original as the
  current view.
- `evolution.gate.evaluated` — **removed from the set.** A gate decision is an
  audit record; its effect is the event it gated, which is what an operator must
  roll back instead. There is nothing to restore.
- `fleet.promotion.reviewed` — **removed from the set.** `FleetManager.rollback_promotion`
  already reverses promotions and enforces steward-or-original-actor authorization
  plus the rollback window; the memory path was inert and bypassed both checks.

### 3.7 Dead config presented as configurable (MEDIUM)

`MemoryEvolutionPolicy.op_thresholds` / `default_threshold`
(`evolution_policy.py:53-54`) are never populated from Settings; every op is
pinned at 0.85. `zaxy memory evolution-policy --json` prints a per-op threshold
map that is always identical — it looks configurable and is not.

### 3.8 `record_outcome` accepts nonexistent targets (MEDIUM)

`target_seq=999, target_hash="f"*64` against a 6-event log returns
`{"reinforced": "invalidated"}` and appends a reinforcement against nothing,
where `edit_memory`/`rollback_memory`/`verified_forget` all call
`_require_target_event`. Asymmetric and silent.

### 3.9 Other confirmed gaps

- I2 amortized learned-context precompute: MISSING; compaction is report-only
  (`crystallization.py:147`), so I3's "lean on I2's precomputed context" dangles.
- I3 has no MCP/CLI surface; engageable only via a process-wide env flag.
- I3 long-span relevance scoring: MISSING — remote-tier score is authoring
  confidence with a hardcoded 0.5 fallback (`long_horizon.py:170-175`).
- I6 out-of-process plugin loading: MISSING and openly disclaimed in-code
  (`plugins.py:22-24`); code-intelligence was never packaged as the reference
  plugin, so the API has never been proven against a real vertical.
- I8b cross-agent transfer is a `within_mission_proxy`; the benchmark imports
  nothing from `zaxy.fleet` and was never updated after I7 shipped.
  **RESOLVED** (`bench/fleet-wide-cross-agent-transfer`): FleetBench `fleet-v2`
  drives a real `FleetManager`, propagates through the I4 gate, and scores
  enrolled agents' real `checkout_memory` with a never-enrolled negative
  control. Run: `reports/benchmarks/fleet-transfer-v1/`. The same change retired
  the empty-finding filler that made the scaling workload near-degenerate.
- Crystallization has no stage-level error handling — a failing stage aborts
  before the summary event, leaving an unattended cron run with no audit record.
- Benchmark workload fingerprints are computed but **not pinned**; no drift guard.
- `apply_approval_decisions` is non-atomic — a mid-batch failure leaves earlier
  reviews committed to an append-only log with no signal.
- Coordination git capture has no `try`/`except` and no `timeout=`.
- `_check_beta_roadmap` is a tautological keyword grep on BETA.md itself.
- BETA.md was last edited when `version = "0.1.0"`, 309 commits ago; its Neo4j-
  primary framing contradicts the embedded-default architecture throughout.

## 4. Disposition — every finding, with its PR

| # | Finding | Disposition |
|---|---|---|
| §2.2 | `mypy` broken by unpinned `anyio` drift | **#139** |
| §2.1 | Gate-withheld rules reached the prompt | **#140** |
| §3.3 | Dashboard published unauthenticated writes | **#141** |
| §3.8 | `record_outcome` cited nonexistent events | **#142** |
| §3.1 | CoordinationBench scored against its own gold | **#143** + **#149** |
| §3.5 | Fleet plane unreachable from MCP | **#144** |
| — | git capture could hang; crystallization lost its audit record | **#145** |
| §3.2 | Promotion accepted unreviewed/evidence-free findings; batches non-atomic | **#146** |
| §3.6 | Rule rollback was a silent no-op | **#147** |
| — | BETA.md predated 3.0.0; `capture-soak` deprecated form in docs + gates | **#148** |
| §3.4 | Trust scoring / `inferred_context` no-ops on the default backend | **#150** |
| — | Reinforcement write blocked the checkout response path | **#151** |

Nothing was merged; every PR is green and awaiting review.

## 5. Cross-cutting recommendation

CI installs unpinned dev dependencies, which is what broke `mypy` on master
without a code change. A constraints file or pinned dev deps would convert these
silent breakages into deliberate upgrades. Not done unilaterally — it touches
release/CI policy.

## 6. Still open (with evidence, ranked)

1. **Unpinned dev dependencies** — the root cause of #139, which will recur. A
   constraints file or pinned dev deps converts silent breakage into deliberate
   upgrades. Untouched because it changes release/CI policy.
2. ~~**`_project_event` in the reinforcement drain** is dead work.~~
   **RETRACTED 2026-07-21 — this item was wrong, and acting on it would have
   corrupted stores.** The premise was half-true: the `memory.reinforcement`
   extractor does return empty entities and edges. But `_project_event` is not
   only the extractor — `upsert_extraction` unconditionally MERGEs an `Event`
   node and its `NEXT_EVENT`/`PREVIOUS_EVENT` chain edges *before* it consults
   entities (`embedded_graph_store.py:448`). That chain mirror is ~13.6 ms of
   the ~13.8 ms, and it is real work with a real consumer:
   `inspect_event_projection_status` derives `integrity_ok` from chain
   contiguity, reachable from `zaxy memory status --graph`.

   Verified by skipping projection for reinforcement events on an interleaved
   workload ending on a reinforcement (what a checkout turn actually looks like):

   | field | baseline | skipping |
   |---|---|---|
   | `integrity_ok` | True | **False** |
   | `missing_chain_links` | 0 | **4** |
   | `projection_lag` | 0 | **1** |
   | `latest_hash_matches` | True | **False** |

   A skipped event leaves the *following* event's `prev_hash` dangling, so each
   skip corrupts a link it does not own. Users would have seen "your store is
   corrupt."

   Two things worth keeping from the investigation. The cost is **not** an
   unindexed scan — measured roughly flat at 13.6/13.8/15.2 ms across 115/645/2475
   events, so it is fixed per-write transaction overhead and "add an index" would
   not have helped either. And `memory.reinforced` / `memory.evidence.reinforced`
   are *different* event types that **do** project entities, so any rule here must
   match `memory.reinforcement` exactly — a prefix match would silently destroy
   real projection state.

   Remaining unmeasured levers for tight-loop throughput: batching the two chain
   statements into one transaction, or moving the drain off the request path
   entirely. Neither is measured, so neither is claimed.
3. **Rollback is still a no-op for three declared types** —
   `MEMORY_CORRECTED_EVENT_TYPE` verified as such, plus `evolution.gate.evaluated`
   and `fleet.promotion.reviewed`. Documented per-type in #147; making them
   effective is open.
4. **Benchmark workload fingerprints are not pinned** — computed deterministically
   but no drift guard, so a generator edit changes them silently.
5. **`_check_beta_roadmap` is a tautological keyword grep** against BETA.md
   itself; it proves nothing about the code.
6. **I2 amortized learned-context precompute** is MISSING (compaction is
   report-only), so I3's "lean on I2's precomputed context" still dangles.
   **I3 long-span relevance scoring** is MISSING (remote-tier score is authoring
   confidence with a hardcoded 0.5 fallback). **I6 out-of-process plugin loading**
   is MISSING and openly disclaimed in-code; code-intelligence was never packaged
   as the reference plugin, so the API is unproven against a real vertical.
7. **No archived capture-soak evidence** under `reports/`, and no
   `manifest.json` for the CoordinationBench report package.
8. **`<300 ms` warm checkout at real scale is unproven.** #151 measures 23 ms at
   500 events and 48.7 ms at 3,000. The quoted ~1.0-1.6 s figure came from a
   ~78k-event store that was not reproduced; the read side scales with corpus
   size, and no reproducible latency harness exists at that scale. Do not put a
   `<300 ms` number on any public surface until measured where it is claimed.

## 7. Open questions — resolved 2026-07-21

All four were put to the user and answered; each answer is reflected above.

1. **CoordinationBench oracle** → remove it and wire the honest adjudicator
   (#143), and retract the token-efficiency clause. Done. Precision/recall
   stayed 1.0, but earned: the public-signal policy independently selects the
   same finding, proven by corrupting the answer key.
2. **Promotion gate** → enforce it (#146). Requires *both* an accepted review
   and evidence, with an auditable `force` escape. The wider of the two options,
   chosen because the banner, the roadmap, and the approval packet's own
   advisory all promise citations.
3. **Embedded parity vs scoping** → both. Scoped honestly in BETA.md (#148),
   then actually fixed (#150) at parity with the Neo4j UAT's pinned values.
4. **BETA.md** → restated against 3.2.0 rather than retired (#148).

## 8. Done-when

- Every §2 and §3 finding has a regression test that fails without its fix,
  verified by reverting the source change and keeping the test. *(met)*
- Every finding is fixed with tests, or explicitly scoped in the controlling doc
  with the decision recorded. *(met — see §4 and §7)*
- `ruff` + `mypy` clean; local suite green against the documented exclusions.
  *(met — 4180 passed, 17 known-environmental failures)*
- Benchmark-claim changes carry the user's recorded agreement. *(met — §7.1)*
- Remaining work is enumerated with evidence rather than left implicit.
  *(met — §6)*

**Not done-when:** nothing here is merged. All 13 PRs are green and awaiting
review; merging is the user's call.

## 9. A note on the local test baseline

17 tests fail on this machine regardless of any change here, and they are not
regressions: `tests/test_harvey_lab_benchmark.py` (11) needs podman and a host
docx reader; `tests/test_packaging.py` (4) and `tests/test_coverage_ratchet.py`
(2) shell out to bare `python`, which does not exist in this environment (only
`python3` and the venv). Verified identical on clean master before any work
began, so a "17 failed" line is the floor, not a signal. CI, which has `python`
on PATH, is green on all of them.

This is worth adding to CLAUDE.md's local-quirks list alongside the documented
`test_doctor` hang.
