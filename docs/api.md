# Python API

The primary Python entry point is `zaxy.core.MemoryFabric`. It wires the event
log, session manager, graph store, query router, embedding provider, metrics,
and tracer. The MCP server uses the same underlying orchestration, so Python
and MCP callers share behavior.

For the maintained classification of MCP tools, Python exports, CLI commands,
Eventloom events, projection backend contracts, and benchmark artifact schemas
(first frozen at v0.9 and kept current through 2.0), see
[API Inventory](api-inventory.md).

Create a fabric:

```python
from zaxy import MemoryFabric

fabric = MemoryFabric()
await fabric.connect()
```

Append an event:

```python
event = await fabric.append(
    event_type="goal.created",
    actor="planner",
    payload={"title": "Ship memory integration"},
    session_id="agent-1",
)
```

Initialize a workspace session with a durable genesis profile:

```python
profile = await fabric.initialize_session(".", session_id="zaxy-default")
```

Query memory:

```python
context = await fabric.query(
    "Ship memory integration",
    temporal_filter=None,
    limit=5,
    session_id="agent-1",
)
```

Returned context metadata includes Eventloom citations when available and
`score_explanation` details for retrieval debugging. Score explanations include
source weights, the matched query, query-expansion weights when applicable, and
temporal scoring fields for as-of queries.

Query exact source chunks:

```python
sources = await fabric.query_verbatim(
    "Which transcript turn mentioned identity-preserving chunks?",
    session_id="agent-1",
    limit=5,
)
```

`query_verbatim()` reads the Eventloom log directly and does not require Neo4j.
It returns raw document, transcript, or event payload chunks with
`eventloom://...` citations and source metadata. Use it when the agent needs
the exact text that produced a memory rather than a compact graph summary.

Ingest local project documents:

```python
count = await fabric.ingest_documents(
    "docs",
    session_id="agent-1",
    max_lines=80,
)
```

Document ingestion currently supports Markdown, plain text, and reStructuredText
files. It skips hidden directories, chunks content by line range, writes
`document.indexed` Eventloom events, and projects each chunk as a `document`
entity. Retrieved document chunks cite their original file and starting line
with `file://path:line` citations.

Document and transcript projection also emits a role-neutral
`neutral_substrate` graph record with actor, artifact, action, time, source,
quote, uncertainty, permission scope, candidate claim, and Eventloom source
backpointer. Purpose-specific meanings such as `legal_obligation`,
`roadmap_commitment`, `customer_escalation`, or `churn_risk` are projection
views rebuilt from that neutral record, not labels written by ingestion.

Ingest a sanitized transcript:

```python
count = await fabric.ingest_transcript(
    [
        {"role": "user", "content": "What did we decide?"},
        {"role": "assistant", "content": "Use MMR for diversity."},
    ],
    source="codex",
    session_id="agent-1",
)
```

Transcript ingestion writes one `transcript.turn` event per non-empty turn.
Secret-looking content is redacted before the event is appended, and redaction
paths are retained on the event payload for auditability.

Assemble prompt-ready context:

```python
assembly = await fabric.assemble_context(
    "What did we decide about retrieval?",
    session_id="agent-1",
    replay_from_seq=1,
    limit=5,
)
print(assembly.prompt)
```

Context assembly combines recent replayed events, ranked graph retrieval, and a
reserved verbatim Eventloom source-recall lane. It is the first lifecycle API
for building an LLM context window from durable session history, temporal graph
memory, and exact cited source chunks. Returned context metadata includes
`assembly_lane` so clients can distinguish `graph` context from `verbatim`
source recall.
The prompt begins with `# Active Memory Working Set`, a bounded deterministic
projection of goals, decisions, tasks, artifacts, blockers, and cited source
anchors. This gives the model task-relevant memory structure without injecting
the whole event log or graph.
`assembly.assembly_policy` records the policy in force for the call, and
`assembly.context_counts` records graph, verbatim, and replay counts for
observability. `assembly.working_set` exposes the same working-set items as
structured data.

Run lifecycle hooks after a turn or subagent handoff:

```python
next_context = await fabric.after_turn(
    role="assistant",
    content="Use MMR for diversity.",
    session_id="agent-1",
    query="retrieval decisions",
    max_recent_events=20,
)

handoff = await fabric.handoff_bundle(
    session_id="agent-1",
    query="current goals and open tasks",
)

subagent = await fabric.cleanup_subagent(
    parent_session_id="main",
    subagent_session_id="worker-1",
    summary="Indexed retrieval docs and found no blockers.",
)
```

`after_turn` preserves the turn as a `transcript.turn` event before assembling
bounded context. `handoff_bundle` includes summary data, prompt-ready context,
and Eventloom integrity status. `cleanup_subagent` records `subagent.cleaned`
in the subagent session and returns a bundle the parent can import or inspect.

Record feedback for retrieved context:

```python
contexts = await fabric.query("retention decisions", session_id="agent-1")
await fabric.record_context_feedback(
    contexts[:2],
    feedback="used",
    session_id="agent-1",
    actor="assistant",
    importance=0.8,
)
```

Positive feedback (`used` or `helpful`) appends `memory.reinforced` events for
the retrieved entity. Negative feedback (`irrelevant`) appends audit-only
`memory.feedback` events and does not delete or decay existing memory. Feedback
for `packet_memory` context preserves the packet projection citation plus source
packet sequence/hash, provider path, and model when available.

Persist synthesis answer artifacts from Memory Checkout:

```python
checkout = await fabric.checkout_memory(
    "How much did I spend on bike expenses in total?",
    session_id="agent-1",
    purpose="review",
)
artifact_event = await fabric.record_synthesis_artifact(checkout, actor="assistant")
candidate = checkout.diagnostics["synthesis"]["answer_candidates"][0]
await fabric.record_synthesis_candidate(
    checkout,
    candidate=candidate,
    outcome="used",
    actor="assistant",
)
```

`checkout_memory(..., purpose=...)` accepts a preset name such as `coding`,
`review`, `release`, `security`, `research`, `support`, `product`, `sales`,
`legal`, `executive`, or `coordinate`, or a purpose
object with role, task, risk, time horizon, expected action, evidence policy,
retention policy, and ontology lens fields. The normalized profile is returned
in the checkout payload and diagnostics so downstream feedback can distinguish
memory that was useful for a release review from memory that was useful for
implementation. The broader support/product/sales/legal/executive presets are
project-local agent-work policies; they do not imply enterprise connectors,
global permissions, or a full Company Brain product category. Checkout also
enforces purpose suppress rules before projection: for example, `coordinate`
checkout excludes worker-local pending, rejected, and stale unpromoted rows from
current facts while reporting the suppressed counts and reasons in
`diagnostics.purpose_policy`.
Non-general purpose profiles also condition retrieval before projection. Zaxy
adds deterministic emphasis terms from the profile's ontology lens, required
evidence, expected action, and retention policy, selects a purpose-appropriate
scoring profile, and raises the internal recall budget without increasing the
visible prompt limit. The applied policy is reported in
`diagnostics.purpose_retrieval_policy`, including the profile, emphasis terms,
scoring profile, base recall limit, and resolved recall limit.
Feedback templates produced by checkout include the normalized purpose profile
when the profile is not `general`, and both `record_context_feedback(...,
purpose=..., outcome=...)` and MCP `memory_feedback(..., purpose, outcome)`
preserve that useful-for-what metadata in Eventloom. Reinforced memory
projection keeps compact purpose and authority fields so future retrieval can
distinguish accepted Coordinate state that supported a handoff from the same
fact used for a code review or implementation step.

Operators can inspect that purpose layer without starting a graph backend:

```bash
zaxy memory purpose status --eventloom-path .eventloom
zaxy memory purpose lanes --eventloom-path .eventloom --json
zaxy memory purpose feedback --eventloom-path .eventloom --profile coordinate
```

These commands replay Eventloom only. They report the active purpose profile,
checkout evidence-policy status, suppressed row counts and reasons, stale or
missing evidence refresh suggestions, retained positive/negative consequence
history, and Coordinate accepted parent state versus worker-local diagnostics.
The same replay-only summary powers the local dashboard `/api/purpose/status`,
`/api/purpose/lanes`, `/api/purpose/feedback` routes and the static Eventloom
viewer Purpose panel.

`record_synthesis_artifact` appends `memory.synthesis.artifact.created` with a
deterministic `synthesis_artifact_v1` payload: query, checkout quality, purpose
profile, slot plan, answer candidates, auditable ledger rows, support citations,
source groups, snippets, and verification warnings. `record_synthesis_candidate`
writes outcome events such as `memory.synthesis.used`,
`memory.synthesis.rejected`, or `memory.synthesis.corrected` without mutating
the source memory. Candidate and row-level evidence feedback also preserve the
purpose profile that conditioned the checkout.

MCP clients use the same contract through `memory_synthesis_artifact`. The tool
accepts the Memory Checkout payload directly. If a selected `candidate` is
provided, `outcome` must be provided in the same call; `helpful` normalizes to
`used`, while `excluded` records `memory.evidence.excluded`.
Selected candidates are validated against
`diagnostics.synthesis.answer_candidates` and persisted from the canonical
checkout candidate so support ids cannot drift from the artifact.
MCP clients can also call `memory_synthesis_evidence` for one
`diagnostics.synthesis.ledger_rows` item when a specific cited row was used or
excluded, preserving the row fact id, source group, citation, selected candidate,
and reason as row-level feedback.
Synthesis artifacts, answer candidates, ledger rows, and outcome feedback are
projected into graph memory as typed entities and edges for later traversal.

For Coordinate, treat the same artifact as the proof packet for accepted-state
composition:

```python
coordinate_state = await fabric.coordinate_checkout(
    "release-rc1",
    include_diagnostics=True,
)
checkout = await fabric.checkout_memory(
    "Compose the accepted release-rc1 findings into the handoff answer.",
    session_id="release-rc1",
)
proof = await fabric.coordinate_record_synthesis_artifact(
    "release-rc1",
    checkout,
    decision_scope="handoff",
    handoff_id="release-rc1:handoff:9",
    actor="coordinator",
)
```

The Coordinate checkout should show which rows are authoritative before the
coordinator records a mission-scoped proof packet. It carries the `coordinate`
purpose profile, which requires accepted parent state and treats worker-local
pending findings as diagnostic unless explicitly included. The proof packet links the
generic synthesis artifact to accepted findings, review events, promotion
events, worker source events, optional handoff event refs, diagnostic pending
rows, conflicts, and excluded duplicate, stale, or conflicting rows before
promotion or handoff. Handoff-scoped proof packets require `handoff_id` and
cite the matching handoff event sequence and hash.

Render first-run integration payloads:

```python
from zaxy import render_agent_integration_template, render_handoff_adapter, render_mcp_client_config

config = render_mcp_client_config("claude-desktop", eventloom_path=".eventloom")
handoff_payload = render_handoff_adapter(handoff, "langgraph")
template = render_agent_integration_template("langgraph", session_id="agent-1")
```

`render_mcp_client_config()` supports `claude-desktop`, `cursor`, and `vscode`
JSON fragments for local MCP setup. `render_handoff_adapter()` supports
`generic`, `langgraph`, `crewai`, and `autogen` payload shapes without importing
those frameworks.

Customize retrieval policy:

```python
from zaxy.query import HTTPReranker, LateInteractionHTTPReranker, LexicalReranker, QueryRouter

router = QueryRouter(
    fabric.graph,
    scoring_profile="precision",
    reranker=LexicalReranker(),
)
chunks = await router.query("auth decision rationale", session_id="agent-1")
```

Built-in scoring profiles are `balanced`, `precision`, `recall`, and
`temporal`. Rerankers implement an async `rerank(query, results, limit=...)`
method and receive fused, deduplicated graph candidates before final truncation.
Use `LexicalReranker` for deterministic local reranking, `HTTPReranker` for
local/self-hosted model endpoints, `LateInteractionHTTPReranker` for
ColBERT-style token-interaction endpoints, or `OpenAICompatibleReranker` for
OpenAI-compatible chat-completions reranking.

Replay a session:

```python
replay = fabric.replay(session_id="agent-1", from_seq=0)
```

Invalidate a graph fact:

```python
await fabric.invalidate(
    entity_name="Ship memory integration",
    entity_type="Goal",
    invalid_at="2026-05-06T12:00:00Z",
)
```

Always close long-lived clients:

```python
await fabric.close()
```

`MemoryFabric` validates inputs before writing events or querying the graph.
Session IDs are passed through the same safety rules used by MCP. Payload size,
query length, traversal depth, and limits are bounded in `src/zaxy/security.py`.
At the durable `EventLog.append` boundary, payloads are also classified and
common secret fields or secret-looking values are redacted before the event hash
is sealed.

Lower-level modules are public enough for advanced integration but should be
used carefully: `EventLog` for direct JSONL operations, `extract` for rule-based
projection, projection backends for graph operations, `QueryRouter` for
retrieval, and `MemoryTracer` for Pathlight spans. The plain install uses
embedded LadybugDB; install `zaxy-memory[neo4j]` for the optional Neo4j sidecar and
`zaxy-memory[pathlight]` for Pathlight tracing. Prefer `MemoryFabric` unless you
are building tests, migrations, or specialized tooling.

For extractor and schema authoring, use the CLI helpers before editing
production code:

```bash
zaxy extractor-template decision.recorded --entity-type decision --name-key title
zaxy schema-plan
```

Extractor templates validate event names and identifiers before rendering code.
Schema migrations are named, checksum-addressed, and still applied through
`GraphStore.init_schema()` so existing callers keep the same lifecycle.

Errors should be treated as operational signals. Validation errors normally mean
the caller sent an unsafe session ID, oversized payload, invalid limit, or empty
query. Graph errors usually mean the selected projection backend is unavailable,
indexes are missing, or backend credentials are wrong. Event log errors usually
mean filesystem permissions, lock contention, or integrity verification failed.
In all cases, the Eventloom log should remain the recovery anchor: fix the
environment, replay the log, and rebuild projections rather than inventing graph
state by hand.

For long-running processes, create one fabric per service process and reuse it.
Avoid constructing a new fabric for every query because each instance owns graph
and tracer clients. Tests can instantiate directly with mocks, but production
callers should use the lifecycle methods consistently.

Configuration comes from [configuration.md](configuration.md). MCP behavior is
in [mcp.md](mcp.md). Graph details are in [graph-schema.md](graph-schema.md).
The quick install path is in [README.md](../README.md), and the public product
overview is [site/index.html](../site/index.html).
