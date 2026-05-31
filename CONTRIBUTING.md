# Contributing

Zaxy is production-oriented agent memory infrastructure. Contributions should
preserve the core design: Eventloom is the append-only source of truth, graph
projections are rebuildable reasoning layers, and public memory claims must be
cited, replayable, and test-backed.

The project bar is explicit: NO HACKS, ONLY DEVELOP PRODUCTION CODE. Small,
well-scoped changes are welcome, but shortcuts that hide state, weaken
provenance, bypass tests, or depend on local-only benchmark artifacts are not.

## Development Rules

- Use test-first development for behavior changes. Write the failing test,
  watch it fail for the expected reason, then implement the smallest production
  change that passes.
- Keep changes aligned with [docs/api-inventory.md](docs/api-inventory.md).
  Stable and beta surfaces need compatibility tests or migration notes when
  their behavior changes.
- Update [docs/migration.md](docs/migration.md) when a user upgrading from
  0.4 through the current release candidate needs a new action.
- Do not rewrite Eventloom history. Append a new event, invalidate a fact, or
  rebuild projection state from the log.
- Keep direct provider integrations dependency-light and optional. MCP remains
  the primary framework-neutral interface.

## Local Setup

```bash
python -m pip install -e ".[dev]"
zaxy status
```

Use Docker services only for explicit integration work or backend comparisons.
Unit tests should mock Neo4j, Pathlight, hosted model providers, and filesystem
side effects unless the test is intentionally marked as integration.

## Verification

Run the narrow tests that cover your change first, then broaden based on blast
radius. Common gates:

```bash
ruff check src tests
mypy src
pytest
zaxy doctor --beta-readiness
scripts/release-check.sh --root .
python scripts/build-site-docs.py --check
scripts/validate-docs.sh --root .
```

For docs-only work, run the docs renderer and validation plus any packaging or
site tests that assert the changed page. For release-surface work, include the
API inventory, migration guide, changelog, and focused compatibility tests.

## Pull Request Checklist

- The failing test existed before implementation for code or behavior changes.
- New public behavior is documented and linked from the relevant hub page.
- Public API, MCP, CLI, Eventloom, projection, or benchmark contract changes
  are reflected in [docs/api-inventory.md](docs/api-inventory.md).
- Upgrade or rollback implications are reflected in
  [docs/migration.md](docs/migration.md).
- Benchmark claims follow
  [docs/benchmark-contributions.md](docs/benchmark-contributions.md) and use
  tracked Eventloom/query inputs.
- Secrets, tokens, raw private provider payloads, and local-only diagnostics are
  not committed.

## Issue Quality

Use the GitHub issue templates. A useful issue includes the Zaxy version,
operating system, Python version, projection backend, exact command or API call,
expected behavior, actual behavior, and a minimal Reproduction. Benchmark issues
must also include artifact paths and whether the inputs are tracked.

Related references: [docs/testing.md](docs/testing.md),
[docs/runbook.md](docs/runbook.md), [docs/benchmarks.md](docs/benchmarks.md),
and [README.md](README.md).
