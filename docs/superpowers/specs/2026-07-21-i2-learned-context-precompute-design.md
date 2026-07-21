# I2 — Amortized learned-context precompute

- **Status:** implemented on `feat/i2-learned-context-precompute` (see §10 for
  the open-question resolutions and the deviations from this design)
- **Date:** 2026-07-21
- **Owner:** Claude Opus 4.8, with the 2026-07-21 roadmap gap audit as input
- **Roadmap refs:** `ZAXY-3.md` §7 I2 (net-new item c), §7 I3 ("lean on I2's
  precomputed learned context"), §9 invariants 1–3 and 5

## 1. The gap, stated precisely

`ZAXY-3.md` I2 promises three net-new pieces. Two shipped: the config-gated
runner (`crystallization.py:113`) and the autonomous metacognitive monitor
(`crystallization.py:299`). The third — **amortized "learned context" precompute
for checkout** — is not implemented, and the audit found nothing under
`grep -rni "learned.context\|precompute\|amortiz" src/` beyond unrelated BM25
and salience precompute.

The machinery to do it already exists and is already *invoked*. The crystallization
pass builds a real compaction projection and then **discards it**:

```python
# crystallization.py:513
def _run_compaction_diagnostic(eventlog: Any) -> dict[str, Any]:
    """Audit the log and, only if safe, build the additive compaction projection."""
    audit = audit_event_log(eventlog)
    if not audit.safe:
        return {...}
    projection = build_compaction_projection(eventlog)
    return {                       # <- counts only; `projection` is dropped here
        "safe": True,
        "record_count": len(projection.records),
        "projection_id": projection.projection_id,
        ...
    }
```

`compaction.py` already ships everything the consumer side needs:
`write_compaction_projection` (`:333`), `load_compaction_projection` (`:347`), and
`search_compaction_projections` (`:353`), which returns cited routing candidates.

**So this initiative is wiring, persistence, and staleness discipline — not new
retrieval science.** That is the main reason to do it now.

### 1a. Why it matters beyond ticking I2

I3's two-tier assembly currently populates its consolidated (remote) tier from
**one** source: accepted `consolidation.candidate.*` events replayed by
`long_horizon.py:102`. On a long session with no consolidation history the remote
tier is empty and two-tier mode degrades to single-tier with a diagnostics stanza.
The compaction projection is the second source that makes the remote tier
non-empty by construction, which is what I3 was written to expect.

## 2. Non-goals

- **Not** a new summarizer. Compaction is medoid/exemplar and source-backed
  (`compaction.py:260`); this proposal adds no generative summarization and
  therefore inherits SSGM drift-resistance (§9 invariant 3). If a future variant
  wants LLM summarization, that is a separate design with its own gate.
- **Not** a cache for the episodic tier. Recent history stays exact replay.
- **Not** authoritative. Nothing here promotes to authority (§9 invariant 2).

## 3. Design

### 3.1 Persist the projection as a rebuildable artifact

Write the projection the pass already builds to
`.eventloom/projections/learned-context/<session_id>.json`, matching the existing
projection-path convention (`config.py:110`, `:114`).

This location matters for a rule the repo already enforces: **projections are
never precious.** Anything under `projections/` may be moved aside and rebuilt
from the log (CLAUDE.md mistake #8). The artifact must therefore be reconstructible
by re-running the pass, with no information that exists only inside it.

### 3.2 Record the build as an event; keep the artifact a cache

The log stays the source of truth (§9 invariant 1). So:

- **Event** (`crystallization.projection.built`, non-authoritative): carries
  `projection_id`, `strategy`, `source_event_count`, `record_count`, the audit's
  `identity_recall` / `citation_coverage`, the covered `{seq, hash}` head, and the
  artifact path. This is the replayable record.
- **Artifact**: the JSON the event describes. Deleting it loses nothing; the next
  pass rebuilds it.

A reader that finds an artifact whose `projection_id` has no corresponding event
must treat the artifact as untrusted and ignore it. That asymmetry is deliberate:
the event is evidence, the file is convenience.

### 3.3 Staleness — the part most likely to go wrong

The projection covers the log up to some seq. The log keeps growing. A stale
projection surfacing old consolidated context as current is exactly the failure
mode this repo cares about.

Adopt the **existing** pattern rather than inventing one: the verbatim index and
replay tip already persist a `{covered_seq, covered_hash}` header and tail-verify
on load, falling back to a full rebuild on any mismatch (PR #82). Reuse it:

- On load, verify `covered_hash` still matches the event at `covered_seq`.
- Any mismatch, missing event, or unreadable artifact ⇒ **ignore the projection
  entirely** and fall back to today's behaviour. Never partially trust it.
- Surface `learned_context.stale: true` in checkout diagnostics when this fires,
  so the degradation is visible rather than silent.

### 3.4 Consumption at checkout

`long_horizon.build_long_horizon_plan` (`long_horizon.py:72`) gains a second
source. Its existing overlap guard is the model to follow — it already refuses to
surface a candidate whose sources all sit inside the recent window, because that
would duplicate rather than consolidate (`long_horizon.py:107-113`).

The same guard must apply to projection records, **plus** a new one: a projection
record and an accepted consolidation candidate can cover the *same* source events.
Surfacing both is the duplication I3 exists to prevent. Dedupe on cited source
event seqs, preferring the accepted candidate (it carries a human review decision;
the projection does not).

Records reach the packet through `search_compaction_projections`, which already
returns citations — so §9 invariant 5 ("everything cites") holds without new work.

### 3.5 Default off, byte-identical when off

Gate on a new `learned_context_enabled` setting, default `false`, mirroring
`crystallization_enabled` (`config.py:173`) and `long_horizon_enabled` (`:212`).

`long_horizon` already has a test proving the default-off path is **byte-identical**
to prior behaviour; this needs the same, and it is the single most important test
in the change. A checkout with the feature off must be indistinguishable from
today.

## 4. Why not the alternatives

| Alternative | Why not |
|---|---|
| Emit projection records as consolidation candidates | They would enter the review queue and require human decisions on machine-derived compaction. I2's output is meant to be additive and non-authoritative, not more review load. |
| Store the projection in the graph | It is a text-routing artifact; `search_compaction_projections` already serves it. Adding node types buys nothing and adds backend-parity work across four backends. |
| Precompute inside checkout on demand | That is the cost I2 exists to amortize, and it puts compaction on the per-turn hot path — the opposite of the #151 direction. |
| Skip persistence, keep it in memory | Dies with the process; crystallization is a cron-triggered pass in a *different* process from the MCP server. |

## 5. Work breakdown

| # | Step | Files | Size |
|---|---|---|---|
| 1 | Persist the projection + emit `crystallization.projection.built` | `crystallization.py:513`, `compaction.py` (reuse `write_compaction_projection`) | S |
| 2 | Covered-head header + tail-verify on load; ignore-on-mismatch | `compaction.py`, new loader seam | M |
| 3 | `learned_context_enabled` setting + path setting | `config.py` | S |
| 4 | Second source in the long-horizon plan, with both dedupe guards | `long_horizon.py:72-120` | M |
| 5 | Checkout diagnostics (`learned_context`: enabled/record_count/stale) | `core/checkout_build.py` | S |
| 6 | Byte-identical default-off test + staleness + dedupe tests | `tests/test_long_horizon.py`, `tests/test_crystallization.py`, `tests/test_compaction.py` | M |

Steps 1–3 are independently landable and inert until step 4 wires consumption,
which is the reviewable ordering: persistence first, consumption last.

## 6. Risks

1. **Silent staleness** — the highest risk, and why §3.3 fails closed rather than
   degrading. A stale projection is worse than none.
2. **Double-surfacing** with accepted consolidation candidates (§3.4). Needs an
   explicit test with a candidate and a projection record covering the same
   sources.
3. **Cron/server process split** — the pass writes the artifact, the server reads
   it. Concurrent read during write must not yield a torn file; write to a temp
   path and rename atomically.
4. **Growth** — projections accumulate per session. The repo has been bitten by
   unbounded projection growth before (the 397MB incident, fixed by the pre-open
   bloat guard in 3.2.0). Bound the artifact and state the bound.

## 7. Open questions

1. **Per-session or per-workspace artifact?** Per-session is simpler and matches
   the session-sharded log, but a fleet with many short sessions produces many
   small files. Per-workspace needs a session dimension inside the records.
2. **Should the projection be I4-gated?** It is non-authoritative and additive, so
   arguably not — but I2's stated contract is "review-gated candidates", and
   `run_crystallization_pass` already routes candidates through
   `evaluate_evolution_gate` (`crystallization.py:199`). Gating the *build* seems
   wrong; gating *consumption* may be right. **I lean: no gate, because nothing is
   promoted and every record cites.** Wants a decision.
3. **Does this change `docs/benchmarks.md` numbers?** If the remote tier gets
   denser, long-horizon retrieval quality moves. Any published number touched by
   this needs a re-run under the same harness — and per CLAUDE.md that needs
   explicit sign-off before it goes outward.
4. **Retention.** How many projections per session are kept, and does an old one
   get deleted or moved aside? "Never delete, move aside" is the house rule for
   `.eventloom`; projections are explicitly exempt, but a stated bound is needed
   for risk 4.

## 8. Done-when

- [ ] Crystallization persists the projection and emits a cited
      `crystallization.projection.built` event; the artifact is reconstructible by
      re-running the pass.
- [ ] A stale, missing, or mismatched artifact is **ignored**, with the
      degradation visible in checkout diagnostics — proven by a test that mutates
      the covered head.
- [ ] With `learned_context_enabled=false`, checkout is **byte-identical** to
      pre-change output — proven by test, mirroring the existing long-horizon
      default-off test.
- [ ] A projection record and an accepted consolidation candidate covering the
      same source events surface **once**, preferring the candidate — proven by test.
- [ ] Every surfaced record carries an `eventloom://…#hash` citation.
- [ ] `ruff` + `mypy` clean; local suite green against the documented 17-failure
      environmental floor; new `src/zaxy` lines covered (coverage ratchet).
- [ ] No benchmark number changes without §7.3 resolved and signed off.

## 9. What this does not close

I3's other two gaps stay open and are **not** fixed by this: horizon-aware
compaction *inside* checkout, and long-span relevance scoring (the remote-tier
score is still an authoring confidence with a hardcoded 0.5 fallback,
`long_horizon.py:170-175`). This work makes the remote tier non-empty; it does not
make it well-ranked. Worth sequencing that immediately after, because a denser
tier with a weak score function may read as a regression.

## 10. Implementation record (2026-07-21)

Implemented on `feat/i2-learned-context-precompute`, together with the I3
long-span relevance scoring §9 flagged as the necessary sequel.

### 10.1 Open questions — resolved

1. **Per-session or per-workspace?** **Per-session**, matching the
   session-sharded log. The artifact is rewritten *in place* rather than
   accumulated, so a fleet of short sessions costs one bounded file each rather
   than one file per pass.
2. **I4-gated?** **No gate**, as this document leaned. Nothing is promoted,
   every surfaced record cites, and consumption is already behind
   `learned_context_enabled`. Gating the build would have made a cache
   review-load; gating consumption would have duplicated the flag.
3. **Benchmark numbers?** **None move.** The feature is off by default and no
   published number was produced with it on, so `docs/benchmarks.md` is
   untouched.
4. **Retention.** One artifact per session, replaced atomically. Bounded by
   construction at **< ~1 MiB per session**: at most 64 records x (4 KiB text +
   2 x 16 x 256 B identity/citation strings), plus a capped audit block. Total
   on-disk cost is O(sessions), not O(passes) — the 397 MB failure mode is
   structurally unavailable.

### 10.2 Deviations from §3 / §5

- **New module rather than a loader seam inside `compaction.py`.** §5 step 2
  said "`compaction.py`, new loader seam". The artifact envelope, the covered-head
  verification, the trust check, and the atomic write are a different concern
  from audit/build/search, and `compaction.py` was already 795 lines. They live in
  `src/zaxy/learned_context.py`; `compaction.py` gained only two public seams
  (`projection_from_payload`, `text_tokens`) so the new module need not reach
  into its privates.
- **No new path setting.** §5 step 3 said "setting + path setting". The path is
  *derived* from `eventloom_dir`, matching how the log itself is located. A
  standalone default path string would not follow a test or deployment that
  moves `eventloom_path`, which is exactly how these artifacts get orphaned.
- **Diagnostics live on the long-horizon plan, not `checkout_build.py`.** §5 step
  5 pointed at `core/checkout_build.py`. The stanza is emitted by
  `LongHorizonPlan.to_diagnostics()` and threaded through the existing
  `long_horizon` section, so the key is absent entirely when the feature is off —
  which is what makes the default-off output byte-identical.
- **`missing` is not `stale`.** §3.3 grouped a missing artifact with mismatched
  and unreadable ones. A never-built artifact is *absence*, not a stale claim, so
  it reports `reason: "missing"` with `stale: false`. Every case that involves a
  file actually claiming to be current still fails closed with `stale: true`.

### 10.3 I3 long-span relevance scoring

The remote-tier score is no longer the authoring `confidence` with a hardcoded
0.5 fallback. `long_horizon.span_relevance` returns a weighted sum in [0, 1] of
four terms, each a deterministic function of data already on the record:

| Term | Weight | Why it is there |
|---|---|---|
| `query_overlap` | 0.40 | The only term that responds to what was asked; a long-span tier that ignores the query is a chronological dump. Uses the same tokenizer as projection search, so both sources are comparable. |
| `span_coverage` | 0.25 | The tier's job is compression: an item standing in for more scrolled-out history buys more context per token. Saturates at 8 source events so one large candidate cannot dominate on size. |
| `horizon_proximity` | 0.20 | `max_source_seq / split_seq`. Everything here is already behind the split; among those, history nearer the split is nearer to current work and less likely superseded. This is the term that makes the score span-aware. |
| `authoring_prior` | 0.15 | Kept but demoted. An accepted candidate's confidence survived human review, so it earns a minority vote — it just no longer *is* the score. Records without one get a neutral 0.5, not a zero. |

Every term, its weight, and the split geometry are echoed into
`metadata["relevance_terms"]`, so the ranking is reconstructible from the packet
alone. Ranking is applied **before** the budget, so the budget keeps the most
relevant remote history rather than the first-replayed.
