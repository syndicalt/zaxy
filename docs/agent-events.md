# Agent Event Taxonomy

Zaxy retrieves best when agents emit typed events with stable payload fields.
These built-in event shapes are deterministic, searchable, and replayable.

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

## Unknown Events

Unknown event types still project as `event` entities. Zaxy includes safe
top-level scalar and list payload text in their summaries, while skipping common
secret-bearing keys and nested objects. Add a typed extractor when an unknown
event becomes part of a public or repeated taxonomy.

Related pages: [eventloom.md](eventloom.md), [retrieval.md](retrieval.md),
[api.md](api.md), and [../README.md](../README.md).
