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
