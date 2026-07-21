# Roadmap gap audit — findings and execution plan

- **Status:** in progress — P0 governance fix landed; benchmark-integrity items blocked on user decision (agree before code)
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

## 3. Findings reported by auditors, not yet independently reproduced

Ranked by severity. Each carries the auditor's cited evidence; none has been
acted on yet.

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

### 3.6 Silent no-op rollback (MEDIUM)

`editable.py:44-53` declares six `ROLLBACKABLE_EVENT_TYPES`, but reversal
semantics exist for only one (`consolidation.candidate.reviewed`). Rolling back
the other five appends an accepted, gated event that changes nothing — the
operator gets a success response for a rollback that did not happen.

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
- Crystallization has no stage-level error handling — a failing stage aborts
  before the summary event, leaving an unattended cron run with no audit record.
- Benchmark workload fingerprints are computed but **not pinned**; no drift guard.
- `apply_approval_decisions` is non-atomic — a mid-batch failure leaves earlier
  reviews committed to an append-only log with no signal.
- Coordination git capture has no `try`/`except` and no `timeout=`.
- `_check_beta_roadmap` is a tautological keyword grep on BETA.md itself.
- BETA.md was last edited when `version = "0.1.0"`, 309 commits ago; its Neo4j-
  primary framing contradicts the embedded-default architecture throughout.

## 4. Execution order

Landed:

1. `mypy`/anyio CI fix (§2.2) — branch `fix/mypy-anyio-generic-streams`.
2. Gate-withheld rule assembly exclusion (§2.1) — same branch, 4 tests.

Proceeding without asking (internal correctness, tests in same commit):

3. Dashboard loopback guard (§3.3) — reuse the existing helper.
4. `apply_approval_decisions` atomicity; git capture hardening.
5. `record_outcome` target validation (§3.8).
6. Per-op threshold plumbing, or delete the dead field (§3.7).
7. `fleet_ids` on the `memory_checkout` MCP tool + CLI (§3.5), regenerating
   `docs/examples/mcp-tool-contract.json` in the same PR.
8. Rollback: narrow the declared set to what reverses, or implement the rest
   (§3.6) — narrowing is the honest default.

Requires user decision before any code (§6):

9. CoordinationBench oracle removal (§3.1) and the roadmap Success Criterion.
10. Promotion policy gate (§3.2) — behaviour change to a shipped surface.
11. Embedded-backend parity for trust scoring (§3.4) — large, and the honest
    interim is to scope the claim.

## 5. Cross-cutting recommendation

CI installs unpinned dev dependencies, which is what broke `mypy` on master
without a code change. A constraints file or pinned dev deps would convert these
silent breakages into deliberate upgrades. Not done unilaterally — it touches
release/CI policy.

## 6. Open questions (agree before code)

1. **CoordinationBench oracle.** Remove the gold oracle and wire the existing
   honest adjudicator, accepting that precision/recall will drop below 1.0? And
   should the Success Criterion's token-efficiency clause be retracted, given the
   repo's own artifact contradicts it?
2. **Promotion gate.** Should `promote_finding` refuse unreviewed or
   evidence-free findings by default (with `--force`), or is the current
   always-manual-but-unchecked behaviour intended?
3. **Embedded parity vs scoping.** Implement `_path_inferred_*` on the embedded
   backend (L), or scope the claims in BETA.md/docs to Neo4j and file parity as
   known work?
4. **BETA.md.** It predates 3.0.0 entirely. Rewrite against current reality, or
   retire it the way AGENTS.md's stale checklist was retired?

## 7. Done-when

- Every §2 finding has a regression test that fails without the fix. *(met)*
- Every §3 item is either fixed with tests, or explicitly scoped/retracted in the
  controlling doc with the user's agreement recorded.
- `ruff` + `mypy` clean; local suite green against the documented exclusions.
- No benchmark claim changes without §6.1 resolved.
