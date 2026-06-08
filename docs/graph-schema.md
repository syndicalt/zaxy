# Graph Schema

The graph projection is Zaxy's structured reasoning layer. It is not the source
of truth; the Eventloom log is. The default implementation is embedded Kuzu,
with Neo4j retained as an optional sidecar and control backend. The graph stores
projections that make memory queryable by entity, relation, keyword, vector
similarity, and temporal validity.

The provenance backbone starts with `Session` and `Event` nodes. Each projected
event is linked through `(:Session)-[:HAS_EVENT]->(:Event)`, and the event then
links to the facts it produced through `PROJECTED_ENTITY` and
`PROJECTED_RELATION`. `Event` nodes also store Eventloom's `hash` and
`prev_hash` fields and project the sealed linear path through
`NEXT_EVENT`/`PREVIOUS_EVENT` relationships. This gives the selected graph
backend a visible audit spine without changing Eventloom's role as the
immutable source of truth.

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
the graph projection an inspectable citation layer while Eventloom remains the
authority for the original event payload.

Edges represent extracted relations between entities. Zaxy stores each edge in
two forms: a compatibility `RELATES` relationship with a `relation_type`
property, and a typed relationship label derived from that relation type, such
as `CALLS_SYMBOL`, `DEFINES_SYMBOL`, or `PROJECTED_LLM_PACKET`. Typed labels are
generated only from strict snake_case identifiers so the graph remains safe to
project through Cypher. Both forms carry `session_id`, event provenance, and
validity windows. Edges also carry an explicit audit classification:
deterministic edges use `inferred=false` and `confidence=1.0`; future inferred
edges must use `inferred=true`, a bounded `confidence`, an `inference_method`,
and optional namespaced `evidence_*` properties. Zaxy writes that metadata to
the `RELATES` edge, the typed relationship edge, and the `PROJECTED_RELATION`
provenance edge so audit and traversal views do not disagree. This lets query
traversal keep using the stable compatibility edge while graph inspection tools
can show semantic labels. For example, an agent can
ask about a goal, expand to tasks, expand to decisions, and still know which
facts were valid at the requested time.
Lifecycle observation events can also link back to tasks when they carry an
explicit `task_id` or `taskId`. In those cases Zaxy projects task-to-observation
edges such as `OBSERVED_COMMAND`, `OBSERVED_FILE_EDIT`, `OBSERVED_TOOL_CALL`,
and `HAS_CHECKPOINT`. These edges are deterministic because they require an
explicit task identifier in the event payload; Zaxy does not infer task linkage
from free text.

The first inferred-edge inlet is the explicit `inference.edge.generated` event.
It does not infer from arbitrary text. It accepts a source entity reference, a
target entity reference, a snake_case `relation_type`, a 0..1 `confidence`, an
`inference_method`, and optional structured evidence. Projection creates the
source and target entities if needed, then writes an inferred edge with the same
audit metadata on every graph relationship surface. This gives downstream
generators a safe contract while keeping Eventloom as the durable record of who
generated the inference and why.

Zaxy's first built-in inferred-edge producer is deliberately narrow. When a
`task.completed` event explicitly cites a decision with `decision`,
`decision_event_seq`, and `decision_event_hash`, Zaxy appends an
`inference.edge.generated` event with relation type
`likely_implemented_decision`, confidence `0.86`, and method
`task_completed_decision_citation_v1`. Without the decision Eventloom citation,
the producer emits nothing.

Operators can audit inferred graph density without writing to memory:

```bash
zaxy memory inferred-status --session-id zaxy-default
zaxy memory inferred-status --session-id zaxy-default --json
```

The status command reports total inferred edges, method counts, confidence
statistics, evidence coverage, missing evidence counts, missing source-event
provenance, and representative samples. This keeps inferred edges inspectable
as a trust surface instead of a hidden retrieval heuristic.

Retrieval uses the same trust surface. When graph traversal crosses inferred
edges, Zaxy computes an inferred-edge trust multiplier from confidence,
non-unknown inference methods, Eventloom source-event provenance, and evidence
coverage. Fully cited inferred edges can modestly boost traversal context;
uncited or weakly evidenced inferred edges are downweighted. `memory_checkout`
and context assembly expose the trust score, multiplier, and coverage factors in
score explanations so inferred relationships remain visible to the model and to
operators.

Memory Checkout also rolls those per-result explanations into an
`inferred_context` diagnostic. The prompt-ready checkout includes inferred
context counts, inferred edge counts, relation labels, inference methods,
average trust, and low-trust guidance so models can distinguish cited inferred
graph context from deterministic memory.

## Causal Projection

Zaxy 2.0 alpha.1 adds an explicit causal projection lane through
`causal.edge.generated` events. These are Eventloom records, not hidden
retrieval heuristics: every causal edge has an append event, actor, payload,
hash-chain position, and graph projection path. The graph is still a projection
of Eventloom, so causal edges remain replayable and auditable from the source
log.

The alpha relation taxonomy is deliberately bounded:

- `caused`
- `enabled`
- `blocked`
- `prevented`
- `regressed`
- `fixed`
- `explained`

Graph relation labels are derived by prefixing the taxonomy value with
`causal_`, such as `causal_caused` or `causal_fixed`. Projection stores the
causal label as graph metadata while preserving the same compatibility
`RELATES` path used by existing traversal code.

Causal edges are inferred, proposed evidence. They are non-authoritative unless
a separate review and promotion workflow explicitly changes the authority state
through the normal gates. Alpha.1 causal projection does not make a causal
statement true by appearing in retrieval, and accepted-looking language in a
method name or summary is not authority promotion. Downstream agents should
treat these edges as cited diagnostics until reviewed authority is present.

Each causal event must preserve evidence and provenance: source and target
entity references, relation type, graph relation type, confidence, causal
method, review status, authority status, Eventloom citation metadata, and
structured evidence. Retrieval exposes the same trust boundary through causal
successor/predecessor results and Memory Checkout diagnostics, including
relation labels, methods, citations, authority status, and low-trust guidance.
Weak, uncited, stale, or distractor-supported causal paths should be diagnosed
or downweighted rather than presented as deterministic memory.

## Consolidation Scaffold

The alpha.1 consolidation scaffold records reviewable memory candidates through
two Eventloom event types:

- `consolidation.candidate.created`
- `consolidation.candidate.reviewed`

Candidate types are `episode`, `claim`, and `procedure`. A candidate is a cited
summary proposal over one or more Eventloom source events. Projection creates a
`consolidation_candidate` entity with the candidate id, type, title, summary,
source-event citations, confidence, method, review status, and
`authority_status=non_authoritative`.

Alpha.2 adds deterministic segment selection and generated proposal creation on
top of this scaffold. Segment selection is event-sourced: Zaxy replays bounded
Eventloom ranges, derives stable segment ids from session id and source event
sequence windows, and emits candidates whose source events retain sequence and
hash citations. The same Eventloom inputs produce the same proposal identities;
no proposal is selected or shaped to fit a benchmark case.

Generated episode, claim, and procedure candidates are review-pending and
non-authoritative. They may help an operator or model inspect possible
compaction targets, but they do not replace the original Eventloom records and
should not be treated as current facts unless a separate authority gate promotes
supported state. Stale, conflicted, rejected, superseded, or `valid_to`-closed
candidates remain auditable graph state, not current authoritative memory.

`consolidation.candidate.reviewed` records a review disposition such as
`accepted`, `rejected`, `deferred`, or `conflicted`, plus rationale and
candidate id. Review events are also replayable Eventloom records and project
as `consolidation_review` entities linked to the candidate. A review
disposition, including `accepted`, does not automatically promote authority;
promotion remains an explicit separate workflow so alpha consolidation cannot
silently collapse source memory into authoritative state. Accepted review means
the proposal passed a review disposition, not that its summary became canonical
truth.

## Reasoning-Loop Primitives

Zaxy 2.0 beta.1 adds reasoning-loop primitive observations as graph-projected
Eventloom evidence. The graph projection may contain observation entities for
`reasoning.primitive.called` with primitive name, deterministic phase, status,
result count, cited evidence count, session, actor, sequence, hash, and source
citations. The supported phase taxonomy is deliberately small:

- `planning`
- `execution`
- `review`
- `reflection`

Phase routing is deterministic policy metadata. It helps checkout and operators
distinguish planning context from execution, review, and reflection context, but
it is not a learned planning policy and it is not a benchmark-specific scoring
shortcut. Primitive observations are reasoning trace evidence: they are
observable, replayable, and useful for audit, but they do not prove the
primitive result is true.

Belief proposals use `belief.update.proposed`. Projection should preserve the
claim, rationale, confidence, source-event citations, phase,
`review_status=pending`, and `authority_status=non_authoritative`. A proposal
is not a fact update. It remains review material until a separate authority path
promotes supported state through the normal gates.

Memory Checkout summarizes these graph surfaces separately from current facts:
`diagnostics.reasoning_primitives` reports context count, phase counts, and
primitive counts; `diagnostics.belief_update_proposals` reports proposal count,
pending count, and the non-authoritative authority status. Prompt guidance must
preserve this separation so a model can inspect the reasoning trace without
treating proposal text as canonical memory.

Zaxy 2.0 beta.2 adds typed metacognition projections for uncertainty and
re-verification workflows:

- `metacognition.unknown.recorded` -> `known_unknown`
- `metacognition.confidence.assessed` -> `confidence_assessment`
- `metacognition.conflict.clustered` -> `conflict_cluster`
- `metacognition.reverify.requested` -> `reverify_request`

These entities preserve claim keys, status or resolution status, confidence,
priority, source event refs, source event hashes, and
`authority_status=non_authoritative`. They are diagnostic graph state, not
accepted facts. Confidence assessment entities are append-only trajectory
points. Conflict clusters remain unresolved until a separate resolution
workflow records authority. Re-verification requests remain open until another
event explicitly closes or supersedes the need.

Beta.2 also hardens procedural memory projection. Skill versions and
procedure candidates preserve procedure steps, applicability, citations,
failure modes, rollback guidance, contradiction reasons, status, and outcome
evidence. Checkout treats validated, revised, or accepted cited procedures as
applicable planning guidance; proposed, pending, or deferred procedures as
diagnostic; and rejected, conflicted, deprecated, contradicted, stale,
superseded, closed, or uncited procedures as excluded.

Example Eventloom payload:

```json
{
  "source": {"name": "task-7", "entity_type": "task"},
  "target": {"name": "Use Memory Checkout as the model-facing state contract", "entity_type": "decision"},
  "relation_type": "likely_informed",
  "confidence": 0.82,
  "inference_method": "operator_review_v1",
  "evidence": {
    "source_event_seq": 6676,
    "reason": "Operator confirmed the task led to the decision."
  }
}
```

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

MATCH p=()-[r:RELATES {inferred: true}]->()
RETURN p, r.confidence, r.inference_method
ORDER BY r.confidence DESC
LIMIT 25;
```

The manual Cypher file under `scripts/setup_neo4j_indexes.cypher` documents the
optional Neo4j sidecar index setup for environments that apply Cypher
separately. Embedded Kuzu creates its projection schema through the embedded
adapter and does not use that Cypher file.

Invalidation does not delete nodes. `memory_invalidate` closes validity windows
at `invalid_at`. This preserves history while preventing default current-time
queries from returning stale facts. Temporal queries can still retrieve the fact
if it was valid at the requested point.

The graph backend code lives in `src/zaxy/embedded_graph_store.py`,
`src/zaxy/graph.py`, and backend adapters, with unit coverage and sidecar
integration tests where applicable. Retrieval behavior is documented in
[retrieval.md](retrieval.md). Event provenance is documented in
[eventloom.md](eventloom.md). Production database configuration is covered in
[configuration.md](configuration.md) and [deployment.md](deployment.md).

When changing schema, follow the test-first rule from [testing.md](testing.md):
write mock tests for generated projection semantics, sidecar integration tests
when backend-specific behavior changes, and update this page plus
[README.md](../README.md) if the public contract changes.
