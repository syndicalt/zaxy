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
```

Each entry reports the framework package, optional extra, starter function,
current maturity, and whether a framework-native adapter package exists. The
current entries are `template` maturity with `native_adapter=not-yet-packaged`;
that status is deliberate until real usage identifies which framework runtime
APIs should become maintained adapters.

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
