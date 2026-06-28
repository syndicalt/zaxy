# Agent Event Taxonomy

Zaxy retrieves best when agents emit typed events with stable payload fields.
These built-in event shapes are deterministic, searchable, and replayable.
Events may also carry optional retrieval-retention metadata such as
`expires_at`, `importance`, `last_reinforced_at`, and `reinforcement_count`.
Retention policies use this metadata only while ranking or filtering retrieved
context; they do not mutate the underlying Eventloom record.
Built-in extractors project this metadata for goals, tasks, decisions, context
policies, and unknown fallback events.

## Decisions

Use `decision.made` for durable choices.

```json
{
  "decision": "Preserve previous chat as a structured Eventloom trace.",
  "summary": "Optional short context.",
  "rationale": ["Typed events are replayable.", "Raw transcript remains available."],
  "alternatives_considered": ["Do nothing", "Store the raw transcript"]
}
```

Projection: `decision` entity linked from the actor with `made_decision`.

## Diagnosed Issues

Use `issue.diagnosed` after root cause is known. Avoid recording speculative
hypotheses as diagnosed issues.

```json
{
  "issue": "memory_query returned no results for recent decisions",
  "root_cause": "Decision payload text was not projected into graph summaries.",
  "evidence": ["memory_replay showed the event existed", "Exact event lookup worked"],
  "fix": "Add typed extractors and reproject Eventloom."
}
```

Projection: `issue` entity with `status=diagnosed`, linked from the actor with
`diagnosed_issue`.

## Verification

Use `verification.recorded` for test, lint, build, smoke, and operational
checks that should survive handoff.

```json
{
  "command": "pytest --no-cov -m \"not integration\"",
  "outcome": "passed",
  "summary": "382 passed, 5 deselected",
  "evidence": ["exit code 0", "ruff clean"]
}
```

Projection: `verification` entity with `outcome`, linked from the actor with
`recorded_verification`.

## Handoffs

Use `handoff.created` when a session, subagent, or branch needs compact state
for a future agent.

```json
{
  "title": "optional stable handoff title",
  "summary": "Zaxy MCP is online with temporal memory.",
  "next_steps": ["Add remote MCP rate limiting", "Add local-first embedding setup"],
  "risks": ["Pathlight traces are currently sparse"]
}
```

Projection: `handoff` entity with `status=created`, linked from the actor with
`created_handoff`.

## Policies

Use `context.policy` for standing project/session guidance that should shape
future behavior. If the instruction should apply to all agents in the repo,
also consider updating `AGENTS.md`.

```json
{
  "source": "AGENTS.md",
  "project": "Zaxy",
  "status": "Active project guidance loaded from resumed chat context.",
  "instructions": ["Write tests first.", "Do not store secrets in Eventloom."]
}
```

Projection: `context_policy` entity linked from the actor with
`set_context_policy`.

## Reinforcement

Use `memory.reinforced` when a still-relevant memory should resist decay. This
does not rewrite the original event; it appends a new event that projects
ranking metadata onto the named entity.

```json
{
  "entity_name": "Use event-sourced retention metadata.",
  "entity_type": "decision",
  "summary": "Still relevant after roadmap review.",
  "importance": 0.9,
  "reinforcement_count": 3
}
```

Projection: target entity with `last_reinforced_at`, `reinforcement_count`, and
optional `importance`, linked from the actor with `reinforced_memory`.
Packet-memory reinforcement may also carry `source_kind=packet_projection`,
`source_event_seq`, `source_event_hash`, `provider_path`, and `model` so the
reinforcement remains tied to the captured packet provenance.

Retention fields are intentionally small and typed:

- `expires_at`: ISO-8601 timestamp after which `filter_expired` hides a result.
- `importance`: float from `0.0` to `1.0` used by decay scoring.
- `last_reinforced_at`: ISO-8601 timestamp used as decay age reference.
- `reinforcement_count`: positive integer that adds a bounded decay boost.

Use `memory.feedback` for audit-only retrieval feedback that should not update
retention metadata. `MemoryFabric.record_context_feedback(...,
feedback="irrelevant")` emits this event.

```json
{
  "entity_name": "Outdated deployment note",
  "entity_type": "decision",
  "feedback": "irrelevant",
  "source": "keyword",
  "score": 0.42,
  "citation": "eventloom://agent-1/events/42#abcdef123456"
}
```

Projection: unknown-event fallback today, preserving the feedback as an
Eventloom-backed audit record without deleting, invalidating, or downranking the
target entity.

Use `memory.reinforcement` for the salience ledger. These events are appended
automatically — checkout surfacing emits one batched `surfaced` event per
checkout, positive `memory_feedback` emits `confirmed`, coordination promotion
emits `promoted`, and `memory_invalidate` emits `invalidated` — and replaying
them fully rebuilds per-memory salience state, so attenuation is always
reversible.

```json
{
  "kind": "surfaced",
  "targets": [{"seq": 42, "hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],
  "source": {"checkout_id": "checkout-123"},
  "authority_status": "non_authoritative"
}
```

`kind` is `surfaced`, `confirmed`, `promoted`, or `invalidated`; `targets`
cites the reinforced memories by event sequence and hash; `source` names the
originating checkout, feedback, or promotion; and an optional `weight`
overrides the default multiplier for the kind. Salience only changes ranking
under the `cognitive` retrieval profile (see [retrieval.md](retrieval.md)) and
never changes what is citable.

Related payload conventions: appends may carry an optional `cues` mapping with
`mission`, `workspace`, `tool`, and `phase` string fields for
encoding-specificity blending; `"pinned": true` exempts a memory from the
attenuation floor; and when `ENCODING_GATE_ENABLED=true`, append payloads gain
an observational `encoding` tag (`novel`, `reinforcing`, or `redundant`) that
never blocks or drops the append.

## Metacognition

Use `metacognition.*` events for uncertainty, confidence trajectories,
conflicting evidence, and re-verification needs. These events are always
diagnostic and non-authoritative. They help an agent decide when to refresh,
ask a user, or inspect conflicting evidence, but they do not update accepted
facts.

Supported beta.2 events:

- `metacognition.unknown.recorded`: records an open known unknown with cited
  source events, a claim key, gap type, and optional reverify query.
- `metacognition.confidence.assessed`: records one append-only confidence point
  for a claim with support/conflict counts and cited evidence.
- `metacognition.conflict.clustered`: records unresolved support and conflict
  source-event sets for a claim.
- `metacognition.reverify.requested`: records an open re-verification request
  with priority and cited source events.
- `metacognition.fok.predicted`: records one non-authoritative
  feeling-of-knowing calibration marker (`query_hash`, `verdict`, `score`)
  each time the experimental `memory_feeling_of_knowing` tool runs, so the
  internal calibration lane can score predictions against actual checkout
  outcomes.

Example known unknown:

```json
{
  "unknown_id": "metacognition:unknown:111111111111111111111111",
  "question": "Which backend caused the latency spike?",
  "reason": "Checkout had conflicting evidence.",
  "claim_key": "backend-latency-cause",
  "gap_type": "conflicting_evidence",
  "status": "open",
  "source_events": [{"seq": 7, "hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],
  "authority_status": "non_authoritative"
}
```

Projection: `known_unknown`, `confidence_assessment`, `conflict_cluster`, and
`reverify_request` entities. Graph properties preserve claim keys, status or
resolution state, confidence, priority, source-event refs, source-event hashes,
and `authority_status=non_authoritative`.

## Skill Memory

Use `skill.*` events for procedural memory that should be retrieved as
applicable guidance, not mixed into factual truth. Checkout is read-only: agents
must append an explicit lifecycle event to change a skill.

Supported lifecycle events:

- `skill.proposed`: draft a new skill version.
- `skill.validated`: mark a version as trusted by cited evidence.
- `skill.revised`: add a new version that supersedes prior guidance.
- `skill.deprecated`: stop recommending a skill version.
- `skill.contradicted`: record evidence that a skill no longer holds.
- `skill.applied`: record that a skill was used on a task.
- `skill.outcome_recorded`: record success, failure, and evidence after use.

Skill definition event:

```json
{
  "skill_id": "python-test-first",
  "name": "Python test-first implementation",
  "version": "1",
  "summary": "Write the failing pytest before implementation.",
  "procedure": ["Write focused failing test", "Run pytest", "Implement minimum code"],
  "applicability": ["Python feature work", "bug fixes"],
  "citations": ["eventloom://agent-1/events/12#abcdef123456"]
}
```

Outcome event:

```json
{
  "skill_id": "python-test-first",
  "version": "1",
  "task": "fix retrieval scoring",
  "success_score": 0.95,
  "feedback": "used",
  "evidence": ["pytest tests/test_query.py::test_scoring -q"]
}
```

Contradiction or deprecation events may also include `failure_modes`,
`rollback`, `contradiction_reason`, and `evidence`. These fields are preserved
on the `SkillVersion` projection so checkout can identify rollback candidates
without deleting the old version or changing the active procedure implicitly.
Checkout separates procedural memory into applicable, diagnostic, and excluded
buckets. Applicable procedures must be cited and validated, revised, or
accepted. Proposed, pending, and deferred procedures remain diagnostic. Rejected,
conflicted, deprecated, contradicted, stale, superseded, closed, or uncited
procedures are excluded from operational instructions.

Projection: `Skill` and `SkillVersion` entities linked with `HAS_VERSION`
style edges, plus lifecycle and outcome entities for application metrics.
Graph properties preserve procedure, applicability, status, citations, outcome
evidence, rollback guidance, contradiction reasons, and version identifiers.

## Fleet Memory Plane

Use `fleet.*` events for governed cross-agent/cross-session knowledge sharing
(Zaxy 3 / I7). The fleet plane is opt-in (`fleet_enabled`, off by default). A
fleet memory is never a copy or mutation of its source: it is a new,
`non_authoritative` event on a dedicated `fleet.<fleet_id>` thread that **cites**
the originating events and the I4 `evolution.gate.evaluated` it routed through.
Promotion raises *visibility scope*, never *authority*; nothing here ever becomes
authoritative.

Two orthogonal controls govern every crossing: a **visibility scope** ladder
(`private`, `session`, `mission`, `fleet`, `global`) carried as event metadata,
and an agent **trust tier** (`untrusted`, `member`, `trusted`, `steward`)
authorizing how far an agent may *propose*. Each crossing composes three
independent gates — trust tier permits the scope, the I4 gate decides auto-apply
vs. review, and a steward reviews held promotions — and a breach of any one stops
the crossing auditably.

Registration and governance events:

- `fleet.created`: create a fleet; the creator is recorded as the implicit steward.
- `fleet.agent.enrolled`: enroll an agent at a trust tier (never escalates trust).
- `fleet.trust.assigned`: a steward assigns a trust tier (reversible, audited).

Propagation events (each carries `promotion_id`, `fleet_id`, `origin_session`,
`origin_actor`, `visibility_scope`, `confidence`, `source_events`, `gate_event`,
`review_status`, and `authority_status="non_authoritative"`):

- `fleet.skill.promoted`: share a validated skill version fleet-wide.
- `fleet.outcome.propagated`: share a success/failure/partial outcome.
- `fleet.rule.propagated`: share a preventive rule (with `trigger`).

Lifecycle (governance, conflict, reversal) events:

- `fleet.promotion.reviewed`: steward `accepted`/`rejected`/`deferred` a held promotion.
- `fleet.memory.superseded`: additive supersession — both prior and superseding
  promotions are retained and replayable.
- `fleet.promotion.rolled_back`: a reversible un-share that lowers effective scope.

Propagation event:

```json
{
  "promotion_id": "fleetpromo:0a1b2c3d4e5f60718293a4b5",
  "fleet_id": "platform",
  "outcome": "failure",
  "summary": "Pre-warm the JWKS cache before the first token refresh",
  "claim_key": "auth.jwks.cache",
  "origin_session": "agent-a",
  "origin_actor": "agent-a",
  "visibility_scope": "fleet",
  "confidence": 0.92,
  "source_events": [{"seq": 12, "hash": "abcdef123456..."}],
  "gate_event": {"seq": 41, "hash": "fedcba654321..."},
  "review_status": "active",
  "authority_status": "non_authoritative"
}
```

Projection: `fleet` plus `fleet_skill` / `fleet_outcome` / `fleet_rule` entities
citing their source events, carrying `visibility_scope`, `fleet_id`,
`review_status`, and `non_authoritative`. The reviewed/superseded/rolled-back
events update `review_status` in place without deleting any promotion. Memory
Checkout adds a distinct, cited, non-authoritative **fleet lane**: an agent
receives a fleet's `active` promotions only when it is enrolled at a tier above
`untrusted` — a non-enrolled or untrusted agent never receives a fleet's promoted
memory, and pending/superseded/rolled-back promotions are never surfaced as active.

## Schema Migrations

Use `schema.migration.proposed` when a stable or beta contract needs a
non-additive change after the v0.9 freeze candidate. The event records the
surface, affected versions, compatibility plan, and tests that prove old data
is rejected, accepted, or migrated intentionally.

```json
{
  "surface": "mcp_tool_contract",
  "from_version": "0.9",
  "to_version": "1.0",
  "change": "Rename a response field with a compatibility alias.",
  "compatibility": "Add alias through 1.x and remove only in a future major release.",
  "tests": ["tests/test_mcp.py::TestToolSchema"],
  "docs": ["docs/migration.md", "docs/api-inventory.md"]
}
```

Use `schema.migration.applied` after the migration ships and verification has
passed. It should cite the proposal, command evidence, and release notes.

```json
{
  "proposal_event_seq": 42,
  "surface": "mcp_tool_contract",
  "verification": ["pytest tests/test_mcp.py -q"],
  "release_note": "CHANGELOG.md#090---unreleased"
}
```

Projection: `schema_migration` entity linked to the affected contract surface
and release version. These events are audit records; they do not rewrite older
Eventloom records.

## Hook Checkpoints

Use `hook.checkpoint` for observer-hook milestones that should remain
searchable during future context assembly.

```json
{
  "trigger": "checkpoint",
  "source": "codex",
  "summary": "Finished hook install mode.",
  "reason": "manual",
  "turn_count": 7,
  "workspace": "/repo"
}
```

Projection: `hook_checkpoint` entity linked from the session with
`recorded_checkpoint`. The summary is indexed for retrieval, while source,
reason, turn count, and workspace remain structured graph properties.

## Lifecycle Hooks

Use lifecycle events for automatic capture around agent execution. These events
store durable metadata and bounded summaries, not raw tool arguments, full
command output, or source file bodies.

Tool call completion:

```json
{
  "tool_name": "shell",
  "status": "succeeded",
  "session_id": "zaxy-default",
  "call_id": "call-123",
  "arguments_redacted": true,
  "argument_keys": ["cmd"],
  "result_summary": "443 passed"
}
```

Projection: `tool_call` entity linked from the session with
`completed_tool_call`.

Command completion:

```json
{
  "command": "pytest -m \"not integration\" --benchmark-disable --no-cov",
  "exit_code": 0,
  "outcome": "passed",
  "session_id": "zaxy-default",
  "duration_ms": 2600,
  "stdout_excerpt": "443 passed, 5 deselected",
  "stderr_excerpt": ""
}
```

Projection: `command_run` entity linked from the session with
`completed_command`.

File edit application:

```json
{
  "path": "src/zaxy/core.py",
  "operation": "modified",
  "session_id": "zaxy-default",
  "summary": "Added lifecycle hook.",
  "line_count": 12
}
```

Projection: `file_edit` entity linked from the session with
`applied_file_edit`.

Compaction completion:

```json
{
  "session_id": "zaxy-default",
  "mode": "rewrite",
  "status": "succeeded",
  "log_path": ".eventloom/zaxy-default.jsonl",
  "event_count": 120,
  "output_path": ".eventloom/zaxy-default.jsonl",
  "snapshot_path": ".eventloom/zaxy-default.snapshot-120.json"
}
```

Projection: `compaction_run` entity linked from the session with
`completed_compaction`.

Subagent completion:

```json
{
  "parent_session_id": "main",
  "subagent_session_id": "worker-1",
  "status": "succeeded",
  "summary": "Worker finished retrieval."
}
```

Projection: `subagent_run` entity linked from the parent session with
`completed_subagent`.

Session end:

```json
{
  "session_id": "zaxy-default",
  "reason": "teardown",
  "status": "succeeded"
}
```

Projection: `session_end` entity linked from the session with `ended_session`.

## Unknown Events

Unknown event types still project as `event` entities. Zaxy includes safe
top-level scalar and list payload text in their summaries, while skipping common
secret-bearing keys and nested objects. Add a typed extractor when an unknown
event becomes part of a public or repeated taxonomy.

Related pages: [eventloom.md](eventloom.md), [retrieval.md](retrieval.md),
[api.md](api.md), and [../README.md](../README.md).
