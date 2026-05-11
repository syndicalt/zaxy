# Graph Schema

Neo4j is Zaxy's structured reasoning layer. It is not the source of truth; the
Eventloom log is. The graph stores projections that make memory queryable by
entity, relation, keyword, vector similarity, and temporal validity.

The provenance backbone starts with `Session` and `Event` nodes. Each projected
event is linked through `(:Session)-[:HAS_EVENT]->(:Event)`, and the event then
links to the facts it produced through `PROJECTED_ENTITY` and
`PROJECTED_RELATION`. `Event` nodes also store Eventloom's `hash` and
`prev_hash` fields and project the sealed linear path through
`NEXT_EVENT`/`PREVIOUS_EVENT` relationships. This gives Neo4j a visible audit
spine without changing Eventloom's role as the immutable source of truth.

The central memory fact shape remains `Entity`. Important properties include
`session_id`, `name`, `entity_type`, `summary`, `valid_from`, `valid_to`,
`event_id`, and `embedding`. Zaxy creates a stable version identity from
session, name, type, and `valid_from`. Reasserting a fact creates a new temporal
version instead of overwriting the existing one. The previous current version in
the same session is closed by setting `valid_to`. Immediate entity versions are
also linked with `SUPERSEDED_BY` and `PREVIOUS_VERSION`, so operators and
retrieval code can traverse a fact's temporal lineage without guessing from
timestamps alone.

Projected entities also carry Eventloom provenance: `source_event_seq`,
`source_event_hash`, `source_event_type`, and `source_thread`. Query results use
these fields to emit stable citations back to the immutable event log.

Source-backed entities additionally project a first-class `Source` node keyed by
`session_id` and source path. Document chunks, code files, symbols, imports,
dependencies, calls, and coverage records already carry `source_path` metadata;
when that metadata is present, Zaxy links both the projected entity and its
originating `Event` to the source through `CITES_SOURCE`. Those relationships
carry line range, checksum, and Eventloom provenance where available. This gives
Neo4j an inspectable citation layer while Eventloom remains the authority for
the original event payload.

Edges represent extracted relations between entities. Zaxy stores each edge in
two forms: a compatibility `RELATES` relationship with a `relation_type`
property, and a typed relationship label derived from that relation type, such
as `CALLS_SYMBOL`, `DEFINES_SYMBOL`, or `PROJECTED_LLM_PACKET`. Typed labels are
generated only from strict snake_case identifiers so the graph remains safe to
project through Cypher. Both forms carry `session_id`, event provenance, and
validity windows. This lets query traversal keep using the stable compatibility
edge while Neo4j Browser and direct Cypher inspection can show semantic labels.
For example, an agent can ask about a goal, expand to tasks, expand to
decisions, and still know which facts were valid at the requested time.
Lifecycle observation events can also link back to tasks when they carry an
explicit `task_id` or `taskId`. In those cases Zaxy projects task-to-observation
edges such as `OBSERVED_COMMAND`, `OBSERVED_FILE_EDIT`, `OBSERVED_TOOL_CALL`,
and `HAS_CHECKPOINT`. These edges are deterministic because they require an
explicit task identifier in the event payload; Zaxy does not infer task linkage
from free text.

Indexes matter for production behavior. Zaxy creates lookup constraints for
entity versions, full-text indexes for keyword search, and vector indexes for
embedding similarity when configured. Schema setup is expressed as named,
idempotent migrations in `src/zaxy/schema.py`; each applied migration is
recorded as a `ZaxySchemaMigration` node with a checksum and statement count.
Operators can inspect the current plan without opening a database connection:

```bash
zaxy schema-plan
```

Useful inspection queries:

```cypher
MATCH p=(:Session)-[:HAS_EVENT]->(:Event)-[:CITES_SOURCE]->(:Source)
RETURN p
LIMIT 25;

MATCH p=(:Entity)-[:CITES_SOURCE]->(:Source)
RETURN p
LIMIT 25;

MATCH p=(:Entity)-[:SUPERSEDED_BY]->(:Entity)
RETURN p
LIMIT 25;

MATCH p=(:Entity {entity_type: "task"})-->(observation:Entity)
WHERE observation.entity_type IN ["command_run", "file_edit", "tool_call", "hook_checkpoint"]
RETURN p
LIMIT 25;
```

The manual Cypher file under `scripts/setup_neo4j_indexes.cypher` documents the
operational index setup for environments that apply Cypher separately.

Invalidation does not delete nodes. `memory_invalidate` closes validity windows
at `invalid_at`. This preserves history while preventing default current-time
queries from returning stale facts. Temporal queries can still retrieve the fact
if it was valid at the requested point.

The graph code lives in `src/zaxy/graph.py` and is covered by unit tests with
mocked Neo4j plus integration tests against Docker. Retrieval behavior is
documented in [retrieval.md](retrieval.md). Event provenance is documented in
[eventloom.md](eventloom.md). Production database configuration is covered in
[configuration.md](configuration.md) and [deployment.md](deployment.md).

When changing schema, follow the test-first rule from [testing.md](testing.md):
write mock tests for generated Cypher semantics, integration tests for real
Neo4j behavior, and update this page plus [README.md](../README.md) if the
public contract changes.
