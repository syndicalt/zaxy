# Integrations

MCP remains Zaxy's primary interface for agent memory, but Python applications
can also use direct, dependency-light framework starters. These templates do
not import LangGraph, CrewAI, or AutoGen. They render plain Python functions
that call `MemoryFabric` lifecycle APIs and can be pasted into an existing
agent application.

Generate a starter:

```bash
zaxy integration-template langgraph --session-id zaxy-default
zaxy integration-template crewai --session-id zaxy-default
zaxy integration-template autogen --session-id zaxy-default
```

The templates all use the same durable flow:

1. Create a `MemoryFabric` with the configured Eventloom path.
2. Capture the latest turn with `after_turn()`.
3. Build resumable context with `handoff_bundle()`.
4. Close the fabric client.

LangGraph starters expose `zaxy_langgraph_memory_node(state)`, returning the
original state plus `zaxy_context` and `zaxy_handoff` prompt strings. CrewAI
starters expose `zaxy_crewai_memory_step(message)`, returning a combined prompt
string suitable for task callbacks. AutoGen starters expose
`zaxy_autogen_context(message)`, returning a dictionary for agent context
variables.

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

Related pages: [api.md](api.md), [mcp.md](mcp.md), and [configuration.md](configuration.md).
