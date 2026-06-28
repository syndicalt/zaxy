# Governed Sleep-Time Crystallization

Crystallization is Zaxy's optional **sleep-time** reflection pass: in an idle
window it reflects over a session's Eventloom log and emits **additive,
review-gated, non-authoritative, cited** consolidation / skill / metacognition
candidates. It turns the on-demand pipelines (`memory consolidation
propose-from-log`, `memory mine-procedures`, the metacognition primitives, the
compaction audit) into a single governed loop a scheduler can fire.

It is a *scheduler over existing primitives* — it does **not** reimplement
consolidation, procedure mining, compaction, metacognition, or salience. Every
auto-apply decision routes through the shipped Memory Evolution Policy gate (I4),
and nothing it writes is ever authoritative.

## One-shot, not a daemon

Like the memory export push layer, crystallization is **operator-side and
one-shot**. There is no always-on background daemon, and there is **no MCP
tool** — the MCP surface stays pull-only. Recurring scheduling is left to an
external scheduler (cron, a systemd timer, or the OS), which invokes the
one-shot runner:

```cron
# Reflect over the default session every night at 03:30 (machine is idle).
30 3 * * *  cd /srv/agent && CRYSTALLIZATION_ENABLED=1 zaxy crystallize --session-id default --json >> /var/log/zaxy/crystallize.log
```

The runner is **off by default**. It only runs when the operator enables it via
configuration (see below) and invokes it; with crystallization not invoked,
nothing about Zaxy's behavior changes.

## Configuration

A single boolean gate controls availability:

| Setting | Env var | Default | Meaning |
| --- | --- | --- | --- |
| `crystallization_enabled` | `CRYSTALLIZATION_ENABLED` | `false` | Enable the governed crystallization runner (operator/cron-triggered). |

Autonomy is **not** a separate knob. Auto-apply behavior is inherited entirely
from the existing Memory Evolution Policy (`evolution_autonomy_default`,
`evolution_op_autonomy`, `evolution_rollback_window_seconds`) described in
[configuration.md](configuration.md). The `zaxy crystallize` CLI refuses to run
while `crystallization_enabled` is false, so enabling it is a deliberate
operator action.

## What one pass does

`zaxy crystallize` builds a `MemoryFabric` and runs one
`run_crystallization_pass`. Each enabled stage schedules an existing primitive:

1. **Consolidation** — `MemoryFabric.propose_consolidation_candidates` appends
   cited, review-pending `consolidation.candidate.created` events for the
   session's actionable log windows. Scoped to `--session-id`.
2. **Procedure mining** — `procedure_mining.mine_and_propose` mines recurring
   successful tool sequences into review-pending procedure candidates. Procedure
   mining is **inherently cross-session** (a skill must recur across at least two
   distinct sessions), so it considers every thread present in the session log
   rather than a single-session slice.
3. **Auto-apply gating (I4)** — every *freshly proposed* candidate, regardless
   of which session it came from, is routed through
   `MemoryFabric.evaluate_evolution_gate('consolidate', confidence, ...)`. That
   call records an auditable `evolution.gate.evaluated` event and returns the
   tier and decision. When `--auto-apply` is set **and** the gate returns
   `auto_apply` (i.e. the tier is `auto_with_rollback` and confidence meets the
   threshold), the runner emits a `consolidation.candidate.reviewed
   status=accepted` review. Otherwise the candidate is left pending. Auto-apply
   here means *accepted review*, still `non_authoritative` and reversible within
   the rollback window — never a promotion to authority.
4. **Metacognition monitor** — the "autonomous monitor that fires the
   primitives". It reads the replayed metacognition surface
   (`summarize_metacognition_events`) and, for each open gap (an unresolved
   conflict cluster or an open known-unknown) that does not already carry an open
   re-verification request, emits one cited `metacognition.reverify.requested`
   event (`build_reverify_request_event`). Scoped to `--session-id`.
5. **Compaction (opt-in)** — when `--compaction` is set, the runner audits the
   log with `compaction.audit_event_log` and, **only if the audit is `safe`**,
   builds the *additive* `compaction.build_compaction_projection`. This pass is
   report-only and never writes a projection file or rewrites the log — that is
   the job of `zaxy compact`. See [consolidation.md](consolidation.md) for the
   compaction safety model.
6. **Salience diagnostic** — a read-only top-salient listing replayed from the
   log (`SalienceLedger.replay`). The runner never emits reinforcement.

Finally the pass appends one `crystallization.run.completed` summary event
(non-authoritative, citing every candidate id it generated) so the pass itself
is auditable and fully replayable.

### Session scope summary

| Stage | Scope |
| --- | --- |
| Consolidation | `--session-id` only |
| Procedure mining | cross-session (every thread in the session log) |
| Metacognition monitor | `--session-id` only |
| Salience diagnostic | the session log (read-only) |

## Invariants

Crystallization preserves the same guarantees as every other Zaxy write path:

- **The log is the source of truth.** All output is derived by replaying the
  immutable, hash-chained Eventloom log.
- **Non-authoritative + review-pending.** Candidate events are stamped
  `authority_status=non_authoritative` and `review_status=pending` by the
  underlying builders; an auto-applied review is still `non_authoritative` and
  reversible.
- **Nothing auto-promotes to authority.** "Auto-apply" emits an *accepted
  review*; it never changes what is citable or authoritative.
- **Everything cites.** Every candidate carries `seq` + 64-hex content-hash
  citations, and the run summary cites the candidate ids it produced.
- **No destructive summarization.** Compaction only ever uses the additive
  projection path, gated by `audit_event_log(...).safe`; the in-place log
  rewrite is never used here.
- **Idempotent.** The builders use deterministic candidate ids and the
  propose/mine paths skip ids that already exist, so re-running a pass over an
  unchanged log proposes no duplicate candidates. The metacognition monitor
  dedups on the deterministic reverify id (and a gap's open re-verification
  request), so re-runs add nothing.

## CLI

```
zaxy crystallize [OPTIONS]
```

| Option | Default | Description |
| --- | --- | --- |
| `--session-id` | `default` | Session (Eventloom thread) to crystallize. |
| `--eventloom-path` | `.eventloom` | Eventloom directory. |
| `--consolidation / --no-consolidation` | on | Propose review-pending consolidation candidates. |
| `--procedure-mining / --no-procedure-mining` | on | Mine recurring procedures into candidates. |
| `--compaction / --no-compaction` | off | Run the additive compaction audit/projection diagnostic. |
| `--metacognition / --no-metacognition` | on | Run the autonomous metacognition monitor. |
| `--auto-apply / --no-auto-apply` | on | Let the I4 gate auto-accept high-confidence candidates. |
| `--json` | off | Print the machine-readable report. |

`--json` prints the `CrystallizationReport`:

```json
{
  "session_id": "default",
  "consolidation_candidates": 3,
  "procedure_candidates": 1,
  "auto_accepted": 1,
  "left_pending": 3,
  "metacognition": { "unresolved_conflict_cluster_count": 0, "...": "..." },
  "reverify_requested": 0,
  "compaction": null,
  "top_salient": [ { "seq": 1, "hash": "…", "score": 1.5 } ],
  "gate_decisions": [
    { "candidate_id": "consolidation:procedure:…", "op": "consolidate",
      "confidence": 0.85, "auto_apply": true, "tier": "auto_with_rollback" }
  ],
  "latency_ms": 12.4
}
```

`--no-auto-apply` records every gate decision for visibility but emits no
accepted review, leaving all candidates pending for human review via `zaxy
memory consolidation status` / `zaxy memory consolidation review`.

When the embedded projection store is locked by a long-lived server, the runner
degrades the graph lane (null backend) and keeps the appends durable, mirroring
`zaxy memory outcome`.

## References

- [Consolidation safety model](consolidation.md)
- [Configuration](configuration.md)
- [Runbook](runbook.md)
- [README](../README.md)
