# Runtime Dashboard Design

## Goal

Add a lightweight local dashboard for runtime memory debugging. The dashboard should make Zaxy's Eventloom capture, graph projection, retrieval diagnostics, and provenance visible without introducing a second mutation path.

## Scope

This slice builds a read-only live local web app for the active workspace. It focuses on runtime memory and debugging: Eventloom sessions, graph projection health, memory checkout diagnostics, recent observations, and interactive graph inspection.

It does not build a hosted multi-tenant dashboard, graph editing tools, memory invalidation workflows, compaction controls, feedback submission, or codebase-intelligence visualization. Those can be added later as explicit follow-up slices.

## Architecture

Add a separate dashboard command:

```bash
zaxy dashboard --host 127.0.0.1 --port 8765
```

The command starts a localhost-only web server with static UI assets and a small read-only JSON API. It should not run inside the MCP server by default and should not append to Eventloom or mutate Neo4j.

Backend:

- New module: `src/zaxy/dashboard.py`
- Typer CLI entry: `zaxy dashboard`
- Static UI serving for bundled HTML, CSS, and JavaScript
- Read-only JSON endpoints backed by existing Eventloom, status, graph, and checkout code
- No mutation endpoints in the MVP

Frontend:

- Plain bundled HTML/CSS/JS for the first slice, consistent with the existing static viewer approach
- Cytoscape.js for the graph canvas
- Dense operational layout rather than a marketing-style dashboard

Primary tabs:

- **Overview**: capture health, latest Eventloom sequence/hash, projection lag, degraded lanes, and warnings
- **Sessions**: Eventloom sessions, threads, recent activity, and selected scope
- **Graph**: interactive runtime memory graph
- **Checkout**: current memory checkout diagnostics, citations, warnings, quality score, and feedback guidance templates
- **Events**: recent Eventloom events with filters and payload detail

## Workspace And Session Scope

`zaxy dashboard` shows only the active workspace by default. It must not crawl local project directories, discover unrelated `.eventloom` folders, or merge projects implicitly.

Default resolution:

1. current working directory
2. workspace root
3. `.eventloom` under that workspace, unless overridden by config or CLI
4. selected session/domain using the same workspace-neutral defaults as `zaxy serve`
5. matching Neo4j graph projection filtered by session/domain

Explicit overrides:

```bash
zaxy dashboard --workspace /path/to/project
zaxy dashboard --eventloom-path /path/to/project/.eventloom
zaxy dashboard --session-id default
zaxy dashboard --domain my-project
```

The dashboard header always shows the resolved workspace, Eventloom path, domain, session, read-only mode, and Neo4j connection status.

An all-project dashboard is out of scope for the MVP. If added later, it should require an explicit workspace registry or command-line flag.

## Graph Model

The graph tab is for runtime memory debugging, not arbitrary Neo4j browsing.

Initial graph views:

- **Provenance**: `Event` nodes connected by `NEXT_EVENT` and `PREVIOUS_EVENT`, with sequence, hash, type, actor, timestamp, and integrity markers
- **Memory**: entities, relationships, `Source` nodes, `CITES_SOURCE` edges, and source-event backpointers
- **Temporal**: current and historical entity versions connected by `SUPERSEDED_BY` and `PREVIOUS_VERSION`
- **Lifecycle**: transcript turns, commands, tool calls, file edits, checkpoints, and task-completion observations
- **Retrieval Path**: query or checkout-selected nodes with score explanations, citation coverage, inferred-edge trust, and path highlighting

Graph interactions:

- Pan, zoom, search, select
- Filter by node type, edge type, event type, actor, session, time range, and inferred/asserted status
- Expand selected nodes by one or two hops
- Toggle temporal validity: current-only, historical-only, or all versions
- Open node and edge detail panels with properties, citations, source event sequence/hash, confidence, inference method, validity interval, and warnings
- Highlight the path from a selected memory item back to Eventloom provenance

Scale constraints:

- Never load the whole graph by default
- Initial graph load is a bounded session summary plus recent or high-signal nodes
- Expansion endpoints enforce session/domain scope, hop limits, and node/edge limits
- Large neighborhoods return capped graph elements plus omitted node/edge counts

## Read-Only API

All MVP endpoints are `GET` and scoped to the resolved workspace/session/domain:

```text
GET /api/status
GET /api/sessions
GET /api/events?session_id=&type=&actor=&q=&limit=&cursor=
GET /api/checkout?session_id=&query=
GET /api/graph/summary?session_id=
GET /api/graph/neighborhood?session_id=&node_id=&view=&hops=&limit=
GET /api/graph/search?session_id=&q=&view=&limit=
GET /api/graph/path-to-event?session_id=&node_id=&limit=
```

Event panels read from Eventloom logs. Status panels reuse existing memory status, doctor-style checks, graph projection status, and inferred-edge status. Graph panels read from Neo4j through read-only query methods. Checkout panels use the same policy module as `MemoryFabric` and MCP checkout so the dashboard shows the same trust guidance that model-facing interfaces receive.

The frontend polls status and event data periodically. Graph requests are user-driven to avoid accidental large reads.

## Read-Only Enforcement

The dashboard must be read-only by construction:

- No `POST`, `PUT`, `PATCH`, or `DELETE` routes in the MVP
- Non-`GET` methods return `405`
- No UI controls for append, invalidate, compact, reproject, or feedback submission
- No direct Cypher input or arbitrary Neo4j browsing
- Localhost bind by default

Any future remote mode must be a separate design with authentication and tenant scoping.

## Error Handling

The dashboard should remain useful when one layer is unavailable:

- No `.eventloom`: show an empty memory state and setup guidance; disable graph views
- Neo4j unavailable: keep Overview, Sessions, and Events working; mark Graph as degraded with the connection error
- Projection lag: show latest Eventloom sequence versus latest projected sequence and warn that graph data may be stale
- Broken Eventloom integrity: flag affected logs/sessions and mark provenance/citation trust as degraded
- Large graph neighborhood: show capped results and omitted element counts
- Unsupported compacted/projection context: reuse existing checkout warnings

## Testing

Tests should cover:

- CLI wiring for `zaxy dashboard`
- Workspace, Eventloom path, session, and domain resolution
- Localhost default binding
- API serialization for status, sessions, events, checkout, and graph responses
- Non-`GET` rejection
- Graph query session/domain filtering
- Graph hop and result limits
- Degraded states for missing Eventloom and unavailable Neo4j
- Static UI smoke checks for core assets and API references

Integration tests with real Neo4j can be marked `integration`; unit tests should mock graph access by default.
