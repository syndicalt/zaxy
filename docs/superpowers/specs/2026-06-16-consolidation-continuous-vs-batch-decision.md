# Decision — Continuous salience-based consolidation, not a timed batch pass

Status: **Accepted** (decided 2026-06-16)
Type: decision record
Supersedes: an earlier proposal to build an autonomous timed consolidation pass
(that proposal was analyzed and rejected; this record keeps the analysis)

## Decision

Zaxy will **not** add a separate timed/batch "consolidation pass." Zaxy's
existing continuous **salience + learning + forgetting** layer already provides
memory consolidation — and provides it more faithfully and more safely than a
periodic batch job would.

## Context

Google's `always-on-memory-agent`
(GoogleCloudPlatform/generative-ai, `gemini/agents/always-on-memory-agent`)
popularized an "always-on" memory agent whose `ConsolidateAgent` runs on a timer
(default every 30 min): it reviews unconsolidated memories, finds cross-cutting
connections, generates higher-level insights, and compresses related entries.
The question: should Zaxy build an analogous pass?

We scoped one (design-first), then analyzed it against Zaxy's architecture and
its learning/forgetting systems. The analysis is the reason for this record.

## Rationale

### 1. Zaxy already does this — continuously, not in batches

Google's batch job approximates, on a cron, what Zaxy does online:

| Google batch consolidation (every 30 min) | Zaxy continuous layer |
|---|---|
| Find what's important | Salience reinforcement, updated online on every surface/confirm/promote (`src/zaxy/salience.py`) |
| Compress / drop the rest | Recency decay (30-day half-life) + invalidation (0.2×) — continuous forgetting, no batch (`salience.py`) |
| Find recurring patterns | Procedure mining — cross-session, cited, review-gated (`src/zaxy/procedure_mining.py`) |
| Compress related info | Compaction — provenance-preserving, on demand (`src/zaxy/compaction.py`) |

The neuroscience the "sleep" metaphor borrows from favors the online design:
systems consolidation is *repeated reactivation strengthening some traces while
others downscale* — i.e. reinforcement-on-surfacing plus decay. A timer that
calls an LLM to "generate insights" is closer to journaling than to
consolidation. The continuous layer is the more faithful mechanism.

### 2. A batch pass adds no mechanism — only a harmful feedback loop

This is the load-bearing finding. **Learning and forgetting operate through the
*salience* channel, which is orthogonal to the *authority* channel.** A proposed
pass guarded only authority ("everything stays `non_authoritative`"), but that
buys nothing against salience: non-authoritative candidates are still extracted
into retrievable entities, still surface in checkout, and still trip the
surfaced-reinforcement loop.

Concretely, every checkout reinforces the citations of the facts/evidence it
surfaces (`src/zaxy/core/fabric.py:2654`, `_record_surfaced_reinforcement` →
`build_surfaced_reinforcement_event`, multiplier 1.05×). Therefore any batch-pass
output that surfaces would:

- **Defeat forgetting (resurrection).** A compaction projection cites all its
  source event hashes; an insight cites its sources. Surfacing either
  re-reinforces those citations — including events the forgetting system has
  attenuated (30-day decay) or invalidated (0.2×). The pass becomes a back door
  that un-forgets. There is also no cascade: invalidating a source does not
  update candidates citing it, so a stale insight keeps resurfacing the very
  memory you forgot.
- **Pollute learning (echo chamber / model collapse).** Insights would enter the
  *same* salience and purpose-outcome economy as primary memories — retrieval has
  no exclusion for consolidation entities, `consolidation.candidate.created` is
  not in the encoding-gate skip list, and purpose-outcome aggregates
  (`src/zaxy/core/checkout_build.py`) do not exclude non-authoritative items. The
  system would learn from its own generated abstractions and reinforce them.
- **Compaction is salience-blind.** Exemplar selection in `compaction.py` does
  not consult current salience, and its audit checks *identity recall*, not
  *salience preservation* — so it can bundle a forgotten memory into a projection
  that resurfaces it.

So the batch pass does not add a missing capability; it adds a loop that fights
the continuous layer Zaxy already has. (Existing precedent for the correct
instinct: the `memory.reinforcement` extractor returns empty entities so
reinforcement cannot distort ranking — the system already takes care to keep
synthetic/observability signal out of the learning loop.)

### 3. The one genuinely-different idea is not consolidation, and not Zaxy's

The only thing Google's pass does that Zaxy does not is *invent net-new
cross-cutting abstractions*. But that is **reasoning over memory**, not
consolidation *of* memory. It belongs to the agent or a separate "brain" layer
above Zaxy — see `positioning-not-a-second-brain`. Folding it into the substrate
is precisely what created the conflicts above.

## Carve-outs (do not lose these)

- **Generative cross-cutting synthesis** — out of scope for Zaxy core; an
  agent/brain-layer concern.
- **Cold-checkout / large-log performance** — this is independent engineering
  that was merely *bundled* into the consolidation proposal. A cold checkout on a
  large log (~115 MB session log → ~18 s) is slow because a fresh process
  re-replays the full log across ~8 `read_all()` consumers in
  `src/zaxy/core/fabric.py`. The real fix is a verified derived-state checkpoint
  loaded on cold start with tail-only verification — worth its own goal when
  desired, unrelated to consolidation philosophy.

## Also declined from Google's design

- **Multimodal/document ingestion** (27 file types) — the second-brain layer,
  deliberately out of scope (`positioning-not-a-second-brain`).
- **Read-all-memories-into-LLM retrieval** — does not scale; Zaxy's bounded,
  purpose-conditioned, evidence-bounded checkout is the differentiator.

## Consequences

- Positioning and docs should present **continuous salience + decay** as Zaxy's
  consolidation story, not a future batch feature.
- If a timed/batch consolidation pass is proposed again, this record is the
  answer: it is redundant with the online layer and conflicts via the salience
  channel.
- The cold-checkout performance work remains open and may be picked up
  independently.

## References

- `src/zaxy/salience.py` — reinforcement kinds/multipliers, 30-day half-life decay
- `src/zaxy/core/fabric.py:2654` — surfaced-reinforcement loop (the conflict point)
- `src/zaxy/core/checkout_build.py` — purpose-outcome learning, encoding gate, evidence eligibility
- `src/zaxy/procedure_mining.py`, `src/zaxy/compaction.py` — existing on-demand consolidation
- `src/zaxy/consolidation.py` — non-authoritative candidate/review contract
- `docs/consolidation.md` — consolidation principles
- `goal-checkout-incremental-cache` (shipped v2.4.4) — incremental retrieval cache
- `positioning-not-a-second-brain` — scope boundary
