# Autonomous Governed Memory-Consolidation Pass — Design

Status: **proposed (design-first; agree before code)**
Date: 2026-06-16
Owner: Zaxy core

## 1. Goal

Give Zaxy an autonomous, scheduled consolidation pass — its answer to Google's
always-on "sleep" memory agent — **without** giving up the properties that make
Zaxy distinct. The pass periodically (and on demand) reviews recent/unconsolidated
events, proposes (a) cross-cutting **insight/connection** candidates and
(b) **compaction** of related state into denser projections, and emits everything
as cited, review-pending candidates. It never decides truth on its own.

It also closes a real operational gap observed in production: `zaxy-default.jsonl`
reached ~115 MB and a *cold* checkout (fresh process) takes ~18 s because it
replays/indexes the whole log. The compaction/snapshot deliverable removes that
cost from the cold path.

## 2. Non-negotiable invariants

These are hard constraints. A change that violates any of them is out of scope.

1. **Nothing the pass produces is authoritative.** Every product of the pass
   enters as a `consolidation.candidate.created` event with
   `review_status="pending"` and `authority_status="non_authoritative"`. Review
   acceptance (`consolidation.candidate.reviewed`) deliberately does **not**
   promote authority — this matches the current contract in
   `src/zaxy/consolidation.py:96`.
2. **Append-only, hash-chained Eventloom stays the source of truth.** Compaction
   produces denser *projections* and *verified snapshots* only. It never mutates,
   rewrites, or deletes history. Full replay/verification from seq 1 must remain
   possible and must still pass `verify_event_chain`.
3. **Every candidate cites its basis events.** No insight without
   `source_events: [{seq, hash}, …]`. Uncited or under-cited proposals are
   dropped, never emitted. (Enforced by the existing
   `consolidation._snapshot_source_events` validation.)
4. **Agent memory, not a second brain.** No document/multimodal ingestion in
   scope. The pass operates only over events already in the log. (See
   `positioning-not-a-second-brain`.)

## 3. What already exists (reuse — do not duplicate)

The substrate is largely present. The pass extends it; it does not reinvent it.

| Need | Existing machinery | Location |
|---|---|---|
| Candidate event contract | `build_consolidation_candidate_event`, types `{episode, claim, procedure}`, deterministic `candidate_id`, cited `source_events`, pending+non-authoritative defaults | `src/zaxy/consolidation.py` |
| Review gate | `build_consolidation_review_event` (accept/reject/defer/conflict; no authority promotion) | `src/zaxy/consolidation.py:75` |
| Proposal pipeline | `ConsolidationSegment` → `ProposedConsolidation` → `.to_candidate_event()`; `select_consolidation_segments`, `generate_consolidation_proposals` | `src/zaxy/consolidation_pipeline.py` |
| Idempotent autonomous proposer (precedent) | `mine_and_propose` — read-only over the log, idempotent via deterministic `candidate_id`, skips duplicates | `src/zaxy/procedure_mining.py:289` |
| Compaction + snapshots | `zaxy compact … --snapshot-every N` (`.snapshot-{n}.json`), medoid/exemplar projections, purpose-based preservation, authority detection | `src/zaxy/compaction.py`, `src/zaxy/cli/serving.py:110` |
| Incremental verified replay | `SessionRetrievalCache.verified_replay` + `_extend_replay` (verifies only the appended tail against a cached prefix) | `src/zaxy/retrieval_cache.py` |
| Salience signals | `memory.reinforcement` (`build_*_reinforcement_event`), `SalienceLedger` | `src/zaxy/salience.py` |
| Semantic extraction | `@register("event.type")` rule extractors; existing extractors for `consolidation.candidate.created/reviewed` | `src/zaxy/extract/rules.py` |
| Lifecycle hooks | `hook.precompact`, `hook.checkpoint`, `hook.heartbeat`, `hook.stop` | `src/zaxy/hooks.py` |

**The genuine gaps** are three: (1) an *autonomous scheduler* — Zaxy has hooks
but no recurring background pass; (2) a *cross-cutting insight* candidate type and
proposer — current proposers work within a single segment; (3) *snapshot loading
on the cold checkout path* — snapshots are written but cold replay still starts
from seq 1.

## 4. Design

### 4.1 New candidate type: `insight`

Extend `CONSOLIDATION_CANDIDATE_TYPES` to `{episode, claim, procedure, insight}`.
The existing `candidate_id` regex already admits any `[a-z_]+` type, so only the
frozenset and its validators change. An `insight` is a cross-cutting connection or
higher-level abstraction drawn across **two or more** segments/events.

Payload (reuses the existing shape; no new event type):

```
candidate_type: "insight"
title:          short claim of the connection
summary:        the insight, phrased as a hypothesis, not a fact
source_events:  [{seq, hash}, …]   # ≥2, spanning ≥2 distinct source segments
confidence:     0.0–1.0
method:         "deterministic-cooccurrence" | "llm-assisted"  # provenance of the proposal
review_status:  "pending"
authority_status: "non_authoritative"
```

Guardrail (hard): an `insight` candidate MUST cite ≥2 source events drawn from ≥2
distinct segments; otherwise it is dropped before append. This is the structural
defense against LLM-invented connections — an insight that cannot point at the
events it connects does not exist.

### 4.2 The proposer — `propose_insights`

New module `src/zaxy/consolidation_insight.py`, modeled directly on
`mine_and_propose`:

- **Read path:** read events through `SessionRetrievalCache` (incremental; no full
  re-read on small growth). Operate over a bounded window (recent + unconsolidated)
  rather than the whole log, so cost is proportional to new material.
- **Candidate generation (Phase 2 = deterministic):** detect co-occurring entities
  / repeated topics / recurring causal links across segments using the existing
  extractor output; emit an `insight` proposal per supported cluster with the
  union of the cluster's source events as citations and a confidence derived from
  support count. No LLM required for the first cut — deterministic and testable.
- **Idempotency:** same deterministic `candidate_id` mechanism; re-running skips
  duplicates (report `skipped_duplicate_count`), exactly like `mine_and_propose`.
- **Compaction proposals:** where a cluster is highly redundant, additionally
  propose a compaction (denser projection) via the existing `compaction.py`
  machinery — also cited, also review-pending.
- **LLM-assisted (Phase 4, optional, opt-in):** an LLM may *phrase* or *rank*
  insights, but only over candidates that already passed the citation guardrail.
  `method="llm-assisted"` records this. The LLM never invents citations.

### 4.3 Scheduler / trigger (the missing piece)

One entry function, `run_consolidation_pass(...)`, reachable three ways:

1. **Manual:** new CLI `zaxy memory consolidate [--insights] [--compact] [--session …]`
   (sibling of `mine-procedures` in `cli/serving.py`). Always available; default
   surface for testing and ad-hoc runs.
2. **Hook-triggered:** invoked from `hook.precompact` / `hook.checkpoint` / idle
   (`hook.heartbeat` with an idle threshold based on last-append timestamp). Idle
   = "sleep": consolidate when the agent is not actively writing.
3. **Background interval (opt-in):** an `asyncio` task in the `serve()` loop runs
   the pass every N minutes. **Off by default** (resource/cost respect); enabled
   by config (interval minutes). This is the closest analog to Google's 30-min
   timer, but opt-in and bounded.

All three share the same idempotent, read-mostly entry function, so triggering it
twice is safe.

### 4.4 Compaction snapshot → cold checkout (the perf deliverable)

Today `verified_replay` starts at seq 1 on a cold process. Plan:

- On compaction with `--snapshot-every`, persist a **verified snapshot**: the
  projection/state plus the `{seq, hash}` of the last event it covers.
- On cold checkout, `SessionRetrievalCache` loads the newest trustworthy snapshot,
  seeds the cached prefix from it, and verifies **only the tail** after the
  snapshot's last event — reusing the existing `_extend_replay` /
  `verify_event_chain` tail-verification path. If the tail fails verification,
  fall back to full replay (safety preserved).
- **Trust model:** the snapshot is a cache, never an authority. Its last-event
  hash must chain-match the live log at that seq, or it is discarded. Audit/replay
  from seq 1 remains available and unchanged.
- **Acceptance metric:** cold checkout on a ~115 MB log drops from ~18 s toward
  the warm ~1.6 s the incremental cache already achieves (target: a clear,
  measured reduction; exact number set during Phase 1 benchmarking).

### 4.5 Surfacing in checkout

Accepted (or high-confidence pending) insights surface in Memory Checkout as
clearly-labeled, **cited candidates** — never as authoritative facts. They ride
the existing checkout assembly and citation rendering; no new authority path.

## 5. Guardrails against hallucinated connections

1. Citation floor (§4.1): ≥2 events across ≥2 segments, or dropped.
2. Confidence floor: below a threshold → dropped or flagged, never silently merged.
3. Non-authoritative always: review acceptance never promotes authority.
4. Deterministic-first: Phase 2 needs no LLM; LLM (Phase 4) only ranks/phrases
   already-cited candidates.
5. Idempotent + auditable: every candidate is a replayable event with a
   deterministic id and explicit `method` provenance.

## 6. Phasing (separate PRs; this doc is Phase 0)

- **Phase 0 — this design doc.** Agree before code.
- **Phase 1 — snapshot → cold-checkout wiring.** Lowest risk, no governance
  change, immediate operational win. Benchmark cold checkout before/after on a
  large log. Ships independently of the rest.
- **Phase 2 — `insight` candidate type + deterministic `propose_insights` +
  `zaxy memory consolidate` (manual) + extractor registration + review-gate
  reuse.** No scheduler yet; manual trigger only.
- **Phase 3 — autonomous scheduler** (hook + opt-in interval, idle detection).
- **Phase 4 (optional) — LLM-assisted ranking/phrasing** behind the Phase 2
  guardrails.

Each phase: ruff + mypy + full suite green; public surfaces preserved.

## 7. Done-when

Design written and agreed (Phase 0); the pass runs on schedule and on demand;
emits cited, review-pending insight + compaction candidates; the review gate
accepts/rejects them; cold-checkout cost on a large log is measurably reduced via
compaction/snapshot; full replay/audit still holds; ruff + mypy + full suite
green. Landed in phases, this design PR first.

## 8. Open questions (resolve during agreement)

1. **Insight clustering signal (Phase 2):** start with co-occurring entities +
   repeated topics across segments? Or also recurring causal edges from the graph?
   (Proposed: entities + topics first; causal edges as a fast-follow.)
2. **Scheduler default:** keep background interval **off by default** (opt-in), with
   hook/idle as the primary autonomous trigger? (Proposed: yes.)
3. **Snapshot cadence & location:** reuse `.eventloom/*.snapshot-*.json`, or a
   dedicated verified-snapshot file with the covered `{seq, hash}` header?
   (Proposed: dedicated header so the trust check is explicit.)
4. **Confidence floor value** for insight emission — pick during Phase 2 from
   real candidate distributions.

## References

- `docs/consolidation.md` — consolidation principles and safety spec
- `src/zaxy/consolidation.py`, `src/zaxy/consolidation_pipeline.py`,
  `src/zaxy/procedure_mining.py`, `src/zaxy/compaction.py`,
  `src/zaxy/retrieval_cache.py`, `src/zaxy/salience.py`, `src/zaxy/hooks.py`
- `goal-checkout-incremental-cache` (shipped v2.4.4) — incremental retrieval cache
- `positioning-not-a-second-brain` — scope boundary
