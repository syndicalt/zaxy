# Integrations

MCP remains Zaxy's primary integration surface for agent memory, but Python
applications can also use direct framework helpers. LangGraph now has a
dependency-light native-beta adapter in `zaxy.adapters.langgraph`. CrewAI
has a dependency-light native-preview adapter in `zaxy.adapters.crewai`.
Zaxy also includes a dependency-light OpenAI-compatible model-call adapter in
`zaxy.adapters.openai_compatible` for applications that own their provider
client directly. AutoGen remains a template starter until real usage identifies
the right runtime hooks to maintain.

Generate a starter:

```bash
zaxy setup integration-template langgraph --session-id zaxy-default
zaxy setup integration-template crewai --session-id zaxy-default
zaxy setup integration-template autogen --session-id zaxy-default
```

When an application wants Zaxy to install the framework package too, use the
optional extras. These extras are intentionally separate from the core install
so MCP users and lightweight template users do not inherit framework dependency
trees:

```bash
python -m pip install 'zaxy-memory[langgraph]'
python -m pip install 'zaxy-memory[crewai]'
python -m pip install 'zaxy-memory[autogen]'
python -m pip install 'zaxy-memory[frameworks]'
```

The CLI can print the matching command with a starter:

```bash
zaxy setup integration-template langgraph --install-hint
```

To inspect the current framework support registry:

```bash
zaxy integrations
zaxy integrations --json
zaxy integrations --recommendation --json
```

Each entry reports the framework package, optional extra, starter function,
current maturity, and whether a framework-native adapter package exists.
LangGraph reports `native-beta` with `native_adapter=zaxy.adapters.langgraph`.
CrewAI reports `native-preview` with `native_adapter=zaxy.adapters.crewai`.
AutoGen remains template-only with `native_adapter=not-yet-packaged`.

The current maintained integration recommendation is
`common-native-preview-contract` on the model-facing UX hardening track. The
reasoning is deliberately conservative: LangGraph and CrewAI already exercise
the shared Memory Checkout, observation, and feedback flow, while AutoGen
remains template-only until runtime hooks are validated in real usage. The next
adapter work should stabilize shared payload keys and feedback behavior across
native adapters before promoting another framework-native package.

## Native Integration Contract

The v0.6 native-runtime guardrail is
`docs/examples/native-integration-contract.json`. It defines the shared
`zaxy.native.v0.6` payload contract used by dependency-light framework adapters
outside MCP.

All native adapters use this lifecycle:

1. before model/task call: run Memory Checkout and inject cited context;
2. after model/task call: capture assistant or task output;
3. after tool call: capture a redacted observation;
4. after context use: record feedback for the contexts actually used.

The stable metadata object lives under the adapter payload's `zaxy` key. For
LangGraph that is `state["zaxy"]`; for CrewAI that is `payload["zaxy"]`; for
OpenAI-compatible model calls that is `result["zaxy"]`.
Required keys are `contract`, `framework`, `operation`, `source`, `kind`,
`status`, `session_id`, `query`, `current_fact_count`, `warning_count`,
`diagnostics`, `quality`, `feedback`, and `error`.

Successful checkout payloads set `kind = "memory_checkout"`, `status = "ok"`,
and `error = None`. Failed checkout payloads fail closed: adapters inject empty
context fields, set `status = "error"`, set `error.code = "checkout_failed"`,
and include a required action to retry Memory Checkout before the next model or
task call.

## OpenAI-Compatible Model Calls

Smoke the outside-MCP model-call path without installing a provider SDK:

```bash
python examples/openai_compatible_memory.py
```

Use the adapter when an application already owns an OpenAI-style client and
wants Zaxy to activate memory at the model-call boundary without MCP:

```python
from zaxy.adapters.openai_compatible import OpenAICompatibleMemoryAdapter

adapter = OpenAICompatibleMemoryAdapter(session_id="my-agent")
result = await adapter.chat_completion(
    client,
    model="gpt-compatible-model",
    messages=[{"role": "user", "content": "What should I remember?"}],
)
```

The adapter does not import OpenAI or any other provider package. It accepts any
client with `client.chat.completions.create(**request)`, including synchronous
and asynchronous clients.

`OpenAICompatibleMemoryAdapter.chat_completion()` runs Memory Checkout,
prepends the checkout prompt as a system message, records a bounded
`model.call.requested` event without raw message content, calls the provider,
and persists assistant output as a sanitized `transcript.turn` event. The
returned payload includes:

- `request`: the provider request with injected memory;
- `response`: the raw provider response;
- `assistant_content`: extracted assistant text when present;
- `checkout`: the full Memory Checkout dictionary;
- `zaxy`: the shared `zaxy.native.v0.6` checkout metadata.

`record_tool_call()` records redacted `tool.call.completed` observations for
provider tool calls. `record_feedback()` appends `memory.reinforced` or
`memory.feedback` events for checkout facts used by the model call.

See [../examples/openai_compatible_memory.py](../examples/openai_compatible_memory.py)
for a no-network smoke example.

## Claude-Compatible Model Calls

Smoke the Claude-style outside-MCP model-call path without installing a
provider SDK:

```bash
python examples/claude_compatible_memory.py
```

Use the adapter when an application already owns a Claude-style client and
wants Zaxy to activate memory at the messages-call boundary without MCP:

```python
from zaxy.adapters.claude_compatible import ClaudeCompatibleMemoryAdapter

adapter = ClaudeCompatibleMemoryAdapter(session_id="my-agent")
result = await adapter.messages_create(
    client,
    model="claude-compatible-model",
    messages=[{"role": "user", "content": "What should I remember?"}],
    max_tokens=1024,
)
```

The adapter does not import Anthropic, Claude, or any provider package. It
accepts any client with `client.messages.create(**request)`, including
synchronous and asynchronous clients.

`ClaudeCompatibleMemoryAdapter.messages_create()` runs Memory Checkout,
prepends the checkout prompt through the provider-shaped `system` field,
records a bounded `model.call.requested` event without raw message content,
calls the provider, and persists assistant text blocks as a sanitized
`transcript.turn` event. It returns the same `zaxy.native.v0.6` checkout
metadata under `result["zaxy"]` and exposes the same `record_tool_call()` and
`record_feedback()` helpers as the OpenAI-compatible adapter.

See [../examples/claude_compatible_memory.py](../examples/claude_compatible_memory.py)
for a no-network smoke example.

## Neutral Trace Export

Native adapters and lifecycle hooks append durable Eventloom observations. To
feed a tracing provider or local JSONL pipeline without making that provider a
runtime dependency, export the correlated trace graph:

```bash
zaxy trace export --eventloom-path .eventloom --json
zaxy trace export --eventloom-path .eventloom --format jsonl --output trace.jsonl
```

The export uses the provider-neutral `zaxy.trace.v0.8` format. Each span cites
the source Eventloom session, sequence, hash, timestamp, actor, event type, and
bounded attributes. Edges link missions to checkout, model calls, tool calls,
findings, reviews, promotions, and handoffs, and link reviews or promotions
back to the finding they decided. The output is designed for adapters that
translate into Pathlight, LangSmith, Phoenix, or local JSONL traces. The JSONL
format writes one record per summary, session, span, and edge for simple
append-only ingestion.

## Coordinate Adapter Contract

Zaxy Coordinate has a dependency-light adapter contract for orchestrators that
already own worker spawning, worktrees, containers, or task scheduling:

```python
from zaxy.adapters.coordination import CoordinationAdapter

adapter = CoordinationAdapter(eventloom_path=".eventloom", actor="coordinator")
adapter.start_mission("auth-main", objective="Ship auth refactor")
adapter.create_worker("auth-main", "auth-api")
adapter.assign("auth-main", "auth-api", "Trace API auth failures")
finding = adapter.report_finding(
    "auth-main",
    "auth-api",
    summary="API failures trace to expired JWKS cache handling",
    evidence=[{"kind": "source", "reference": "src/auth.py:12"}],
    claim_key="auth.failure.cause",
    claim_value="expired-jwks-cache",
)
```

The adapter returns JSON-friendly payloads with Eventloom event sequence and
hash metadata. It does not infer findings from transcripts and does not spawn
workers; callers must pass explicit summaries, evidence, confidence, and
claim fields.

Generate coordination starters:

```bash
zaxy coordinate adapter-template codex --mission auth-main --worker auth-api
zaxy coordinate adapter-template langgraph --mission auth-main --worker auth-api
zaxy coordinate adapter-template crewai --mission auth-main --worker auth-api
zaxy coordinate adapter-template mcp --mission auth-main --worker auth-api
```

## LangGraph Native Beta

Smoke the dependency-light LangGraph path without installing LangGraph:

```bash
python examples/langgraph_memory.py
```

Use the native-beta adapter when you want Zaxy to behave like a LangGraph
node without requiring Zaxy to own your graph schema:

```python
from zaxy.adapters.langgraph import (
    create_langgraph_coordination_node,
    LangGraphMemoryAdapter,
    create_langgraph_memory_checkout_node,
    create_langgraph_memory_node,
)

memory_node = create_langgraph_memory_node(session_id="my-agent")
checkout_node = create_langgraph_memory_checkout_node(session_id="my-agent")
coordinate_node = create_langgraph_coordination_node(mission_id="auth-main", worker_id="auth-api")
adapter = LangGraphMemoryAdapter(session_id="my-agent")
```

`create_langgraph_memory_node()` returns an async node that reads the latest
message from `state["messages"]` or `state["latest_message"]`, records the turn
as `transcript.turn`, assembles prompt-ready context, and returns the original
state with:

- `zaxy_context`: prompt text for the next model call;
- `zaxy_contexts`: the retrieved `Context` objects used for feedback;
- `zaxy`: metadata including session, replay count, warnings, and citations.

For production agents, put `create_langgraph_memory_checkout_node()` or
`LangGraphMemoryAdapter.checkout_before_model()` at model boundaries. That
middleware calls `memory_checkout` automatically so long sessions, resumes,
compactions, and roadmap questions reintroduce cited current memory instead of
depending on the model to remember Zaxy.

The checkout node uses the beta native adapter metadata contract
`zaxy.native.v0.6` under `state["zaxy"]`. Stable keys are:

- `contract`, `framework`, `operation`, `source`, `kind`, and `status`;
- `session_id`, `query`, `current_fact_count`, and `warning_count`;
- `diagnostics`, `quality`, and `feedback` from Memory Checkout;
- `error`, which is `None` on success.

If checkout fails, the adapter fails closed: it returns empty `zaxy_context` and
`zaxy_contexts`, sets `status` to `error`, includes
`error.code = "checkout_failed"`, and asks the caller to retry Memory Checkout
or run `zaxy doctor`. It does not inject stale context after a checkout failure.

`LangGraphMemoryAdapter.record_tool_call()` records redacted
`tool.call.completed` observations for tool nodes. `record_assistant_turn()`
persists assistant output as a transcript turn. `record_context_feedback()`
reinforces contexts that were actually projected into state.

See [../examples/langgraph_memory.py](../examples/langgraph_memory.py) for a
smoke example.

## CrewAI Native Preview

Use the CrewAI adapter from task callbacks, custom task wrappers, or any
application-owned lifecycle point. Zaxy does not import CrewAI or require a
specific CrewAI object model:

```python
from zaxy.adapters.crewai import (
    create_crewai_coordination_step,
    CrewAIMemoryAdapter,
    create_crewai_memory_checkout_step,
    create_crewai_memory_step,
)

memory_step = create_crewai_memory_step(session_id="my-crew")
checkout_step = create_crewai_memory_checkout_step(session_id="my-crew")
coordinate_step = create_crewai_coordination_step(mission_id="auth-main", worker_id="auth-api")
adapter = CrewAIMemoryAdapter(session_id="my-crew")
```

`create_crewai_memory_step()` returns an async callable that accepts task input
and returns prompt-ready memory text. `CrewAIMemoryAdapter.before_task()` returns
a richer payload containing:

- `memory`: prompt text for the task;
- `contexts`: retrieved `Context` objects used for feedback;
- `zaxy`: metadata including session, replay count, warnings, crew, agent, and
  task identifiers when supplied.

For production task wrappers, use `create_crewai_memory_checkout_step()` or
`CrewAIMemoryAdapter.checkout_before_task()` at task boundaries. That path calls
`memory_checkout` before task execution and returns the checkout prompt as
`memory`.

CrewAI checkout uses the same `zaxy.native.v0.6` metadata contract under
`payload["zaxy"]` as the LangGraph checkout node. It adds CrewAI-specific
`crew`, `agent`, and `task_id` fields when supplied. If checkout fails, the
adapter fails closed with empty `memory` and `contexts`, `status = "error"`,
`error.code = "checkout_failed"`, and remediation guidance to retry Memory
Checkout or run `zaxy doctor`.

`CrewAIMemoryAdapter.after_task()` persists task output as an assistant turn.
`record_tool_use()` records redacted `tool.call.completed` observations.
`record_context_feedback()` reinforces contexts that were actually used by the
crew task.

```python
payload = await adapter.before_task(
    "Draft beta release notes",
    crew="release",
    agent="writer",
    task_id="release-notes",
)

# Pass payload["memory"] to the CrewAI task context.

await adapter.after_task(
    "Release notes drafted.",
    crew="release",
    agent="writer",
    task_id="release-notes",
)
await adapter.record_context_feedback(payload, feedback="used", importance=0.8)
```

The templates and native adapters use the same durable flow:

1. Create a `MemoryFabric` with the configured Eventloom path.
2. Call Memory Checkout before replying or starting task/model work.
3. Capture completed turns with `after_turn()`.
4. Close the fabric client.

LangGraph starters expose `zaxy_langgraph_memory_node(state)` for users who want
a copy-paste template instead of importing the preview adapter. CrewAI starters
expose `zaxy_crewai_memory_step(message)` and
`zaxy_crewai_record_result(result)` on top of `CrewAIMemoryAdapter`. AutoGen
starters expose `zaxy_autogen_context(message)`, which runs Memory Checkout
before replying and returns a dictionary for agent context variables.
`zaxy_autogen_record_reply(reply)` captures completed replies after the agent
responds.

The same renderer is available from Python:

```python
from zaxy import render_agent_integration_template

template = render_agent_integration_template(
    "langgraph",
    session_id="zaxy-default",
    eventloom_path=".eventloom",
)
```

These starters are intentionally shallow. They avoid framework dependencies and
version-specific APIs so Zaxy can provide a stable adoption path while each
application keeps control of its graph, crew, or agent runtime. For hosted or
multi-language clients, prefer MCP. For Python services that already own their
agent loop, direct templates are the shortest path to durable lifecycle memory.

Related pages: [api.md](api.md), [mcp.md](mcp.md),
[configuration.md](configuration.md), and [site/index.html](../site/index.html).
