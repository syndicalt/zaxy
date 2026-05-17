# Skill Memory And pgGraph Backend Evaluation Design

## Context

Zaxy already has a strong memory substrate: Eventloom is the source of truth,
Neo4j is the structured reasoning projection, and Memory Checkout assembles
cited, prompt-ready context. A recent product thesis argues that memory and
skills should be treated as one routed world model instead of separate static
files, plugins, or APIs.

Zaxy mostly satisfies the evidence and provenance side of that thesis, but it
does not yet treat reusable procedures as first-class memory. Separately,
pgGraph is worth evaluating because it could combine Postgres full-text,
pgvector, relational constraints, transactions, and graph traversal in one
operational backend. pgGraph is currently alpha software, so this must be an
evaluation track, not a default backend migration.

## Goals

- Add Skill Memory to the roadmap as the procedural layer of Zaxy's world model.
- Evaluate pgGraph as an experimental backend without risking current Neo4j
  retrieval quality.
- Preserve Eventloom as the immutable source of truth for every backend.
- Keep Neo4j as the default production and benchmark baseline until a new
  backend proves parity or better on the same harness.

## Non-Goals

- Do not replace Neo4j in the current release line.
- Do not make pgGraph production-supported while its upstream docs still mark it
  alpha and warn against production/shared-cluster use.
- Do not auto-amend skills without evaluation, provenance, and rollback.
- Do not split Zaxy into separate "memory" and "skills" products.

## Skill Memory Roadmap

Skill Memory should treat procedures as memory objects with lifecycle,
provenance, outcomes, and versions. The initial event taxonomy should include:

- `skill.proposed`
- `skill.validated`
- `skill.applied`
- `skill.outcome_recorded`
- `skill.revised`
- `skill.deprecated`
- `skill.contradicted`

The graph projection should add `Skill` and `SkillVersion` entities connected to
tasks, tools, commands, files, source citations, success metrics, failure modes,
and applicability conditions. Memory Checkout should return applicable skills in
a distinct lane from factual context so models can see which procedures are
recommended, why they are recommended, when not to apply them, and what evidence
supports them.

Promotion from observed behavior to active skill should be gated by explicit
evaluation. Each active skill version needs citations, scope, confidence,
outcome history, rollback metadata, and a supersession path.

## pgGraph Evaluation Roadmap

pgGraph should be evaluated as a Postgres-local graph acceleration layer over
Zaxy-owned relational tables. Zaxy would still own temporal semantics through
schema design: entity versions, edge versions, sources, Eventloom projections,
validity windows, and invalidation records.

The evaluation should use a backend-neutral projection contract:

- `upsert_extraction`
- `search_exact`
- `search_keyword`
- `search_vector`
- `search_traversal`
- projection integrity and inferred-edge status methods needed by checkout and
  dashboard surfaces

The Neo4j adapter should be wrapped behind that contract first with no behavior
change. The pgGraph adapter should then be built as an experimental backend over
Postgres tables, PostgreSQL full-text search, pgvector, and pgGraph traversal.

## Evaluation Gates

pgGraph can move beyond experimental only if it satisfies all of these:

- Same-harness retrieval quality matches or beats Neo4j on temporal recall,
  source recall, graph traversal, context collapse, and the LongMemEval slice.
- Citation coverage and source provenance remain intact.
- Temporal validity and invalidation semantics are equivalent to Neo4j.
- Local setup and operational recovery are simpler than the current Neo4j path.
- Latency and returned-token tradeoffs are no worse under the published
  benchmark settings.
- Upstream pgGraph production guidance no longer blocks production/shared-cluster
  use for Zaxy's expected workload.

## Risks

- Zaxy's retrieval scoring is heavily tuned around current Neo4j query behavior.
  Backend neutrality must not erase score explanations or lane-specific weights.
- pgGraph's current alpha limitations include SQL contract hardening,
  correctness hardening, sync semantics, persistence hardening, and operational
  visibility. Those are serious enough to keep the backend experimental.
- A relational representation may make temporal queries easier, but graph
  traversal semantics and path hydration will need careful equivalence tests.
- Skill auto-improvement can overfit local anecdotes. Evaluation gates and
  rollback are mandatory.

## Testing And Benchmarking

The first implementation plan should be benchmark-driven:

- Unit-test the backend-neutral contract with a fake projection store.
- Preserve existing Neo4j behavior through adapter tests.
- Add pgGraph integration tests only behind an explicit marker and Docker
  service.
- Run the same benchmark workloads against Neo4j and pgGraph from the same
  Eventloom replay.
- Require guardrail output to report quality, latency, citation coverage,
  retrieval lane composition, and score metadata for both backends.

## Roadmap Placement

Skill Memory should be a product roadmap item after current benchmark and UAT
work because it extends Memory Checkout into procedural context. pgGraph should
be a research/backend track that starts with contract extraction and benchmark
proof, not with a migration.
