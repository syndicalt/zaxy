# Graph Schema

Neo4j is Zaxy's structured reasoning layer. It is not the source of truth; the
Eventloom log is. The graph stores projections that make memory queryable by
entity, relation, keyword, vector similarity, and temporal validity.

The central node shape is `Entity`. Important properties include `name`,
`entity_type`, `summary`, `valid_from`, `valid_to`, `event_id`, and
`embedding`. Zaxy creates a stable version identity from name, type, and
`valid_from`. Reasserting a fact creates a new temporal version instead of
overwriting the existing one. The previous current version is closed by setting
`valid_to`.

Edges represent extracted relations between entities. They carry
`relation_type`, event provenance, and validity windows. This lets query
traversal answer multi-hop questions while keeping the timeline intact. For
example, an agent can ask about a goal, expand to tasks, expand to decisions,
and still know which facts were valid at the requested time.

Indexes matter for production behavior. Zaxy creates lookup constraints for
entity versions, full-text indexes for keyword search, and vector indexes for
embedding similarity when configured. The manual Cypher file under
`scripts/setup_neo4j_indexes.cypher` documents the operational index setup.

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
