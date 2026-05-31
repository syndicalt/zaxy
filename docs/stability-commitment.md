# Stability Commitment

This page defines the v1.0 API and data model stability commitment. It is scoped
to surfaces marked `Stable` or `Beta` in [docs/api-inventory.md](api-inventory.md)
and bound by [docs/examples/v1-schema-freeze.json](examples/v1-schema-freeze.json).

## Stability Commitment

After v1.0, Zaxy will preserve documented public behavior for stable and beta
surfaces within the 1.x line. Breaking changes require a documented migration
path, test coverage, and release notes. Additive fields, new optional commands,
new event types, and new experimental adapters may ship in minor releases when
they do not invalidate existing Eventloom replay or public clients.

## Public API Surfaces

The commitment covers these public surfaces:

- MCP tool names, input schemas, and response contracts.
- Python package exports listed in `zaxy.__all__`.
- CLI command families documented in `docs/api-inventory.md`.
- Eventloom event families documented in `docs/agent-events.md`.
- Memory Bootstrap and Memory Checkout model-facing contracts.
- Zaxy Coordinate mission, finding, approval, checkout, and handoff payloads.
- ProjectionStore behavior required for exact, keyword, vector, traversal,
  invalidation, projection status, and inferred-edge diagnostics.
- Benchmark artifact schemas used by release gates.

## Data Model Commitment

Eventloom is the durable source of truth. Zaxy will not require users to rewrite
historical Eventloom records for a 1.x upgrade. Durable records remain
append-only, hash-linked, replayable, and audit-oriented.

The graph projection remains rebuildable. Projection nodes, edges, indexes, and
backend-specific storage can change internally, but the public data semantics
remain stable: entities carry temporal validity, source citations, Eventloom
provenance, session isolation, and deterministic replay from Eventloom where the
source log is available.

MCP, CLI, and Python clients should treat Eventloom citations as the strongest
provenance contract. Graph IDs, backend row IDs, and private helper names are
not durable identifiers.

## Compatibility Policy

Stable and beta changes should follow this order:

1. Prefer additive fields or optional behavior.
2. Preserve old names as aliases when a rename is unavoidable.
3. Add compatibility tests that exercise old and new behavior.
4. Update `docs/migration.md`, `CHANGELOG.md`, and the affected contract
   snapshot.
5. Use release notes to explain user action, if any.

## Migration Events

Non-additive stable or beta changes require Eventloom migration events:

- `schema.migration.proposed` before implementation, naming the surface,
  affected versions, compatibility plan, tests, and docs.
- `schema.migration.applied` after verification, citing the proposal, command
  evidence, and release note.

These migration events are audit records. They do not mutate older Eventloom
history.

## Non-Commitments

The v1.0 stability commitment does not cover:

- surfaces marked `Experimental` or `Internal` in `docs/api-inventory.md`;
- direct imports from modules that are not exported by `zaxy.__all__`;
- candidate backends such as pgGraph or LatticeDB until promoted by release
  gates;
- private graph storage details, Cypher/Kuzu SQL details, or row IDs;
- local diagnostic files ignored by benchmark contribution policy;
- provider-specific SDK behavior outside Zaxy's dependency-light wrappers.

Related references: [api-inventory.md](api-inventory.md),
[migration.md](migration.md), [agent-events.md](agent-events.md),
[release-validation-checklist.md](release-validation-checklist.md), and
[README.md](../README.md).
