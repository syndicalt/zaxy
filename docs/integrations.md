# Integrations

MCP remains Zaxy's primary integration surface for agent memory, but Python
applications can also use direct framework helpers. LangGraph now has a
dependency-light native-preview adapter in `zaxy.adapters.langgraph`. CrewAI
also has a dependency-light native-preview adapter in `zaxy.adapters.crewai`.
AutoGen remains a template starter until real usage identifies the right runtime
hooks to maintain.

Generate a starter:

```bash
zaxy integration-template langgraph --session-id zaxy-default
zaxy integration-template crewai --session-id zaxy-default
zaxy integration-template autogen --session-id zaxy-default
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
zaxy integration-template langgraph --install-hint
```

To inspect the current framework support registry:

```bash
zaxy integrations
zaxy integrations --json
zaxy integrations --recommendation --json
```

Each entry reports the framework package, optional extra, starter function,
current maturity, and whether a framework-native adapter package exists.
LangGraph reports `native-preview` with `native_adapter=zaxy.adapters.langgraph`.
CrewAI reports `native-preview` with `native_adapter=zaxy.adapters.crewai`.
AutoGen remains template-only with `native_adapter=not-yet-packaged`.

The current maintained integration recommendation is
`common-native-preview-contract` on the model-facing UX hardening track. The
reasoning is deliberately conservative: LangGraph and CrewAI already exercise
the shared Memory Checkout, observation, and feedback flow, while AutoGen
remains template-only until runtime hooks are validated in real usage. The next
adapter work should stabilize shared payload keys and feedback behavior across
native-preview adapters before promoting another framework-native package.

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

## LangGraph Native Preview

Use the native-preview adapter when you want Zaxy to behave like a LangGraph
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
