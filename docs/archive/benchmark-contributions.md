# Benchmark Contributions

Benchmark contributions are welcome when they improve reproducibility,
coverage, or release confidence. Zaxy benchmark evidence must be replayable from
tracked inputs and must not rely on private local state.

## Accepted Contribution Types

- New workload files that exercise memory checkout, temporal retrieval,
  Eventloom source recall, coordination, codebase mapping, or projection
  backend behavior.
- New benchmark report artifacts generated from tracked Eventloom and query
  inputs.
- Guardrail updates that make release checks more representative without hiding
  regressions.
- External comparison disclosures that clearly separate same-harness results
  from adapter feasibility notes.

## Required Evidence

Every benchmark report proposed for release use needs:

- tracked Eventloom inputs under the report or benchmark directory;
- tracked query files or workload files;
- `query_results` diagnostics for per-query audit;
- citation coverage metrics;
- latency metrics for checkout and retrieval lanes when relevant;
- report metadata and workload fingerprints;
- a Markdown sidecar that summarizes the same numbers humans will cite.

Reports under `reports/backend-shootout/` are checked by
`scripts/check-backend-shootout.py`. Release reports must keep query diagnostics,
fingerprint validation, git-tracked inputs, citation coverage, and latency
budgets enabled. Do not promote `reports/benchmarks/*-diagnostics.*` files as
public claims unless the corresponding workload and report contract are also
tracked and documented.

## Running Checks

Use the narrow command for the artifact you changed, then run the broader
release checks when the result supports a public claim:

```bash
python scripts/check-backend-shootout.py reports/backend-shootout/backend-shootout.json \
  --require-report-metadata \
  --require-markdown-report \
  --require-query-results \
  --require-git-tracked-inputs \
  --verify-report-fingerprints

zaxy doctor --beta-readiness
scripts/release-check.sh --root .
```

For Coordinate benchmarks, regenerate from a tracked workload and keep the
report limitations visible. For LongMemEval-compatible reports, preserve the
same-harness BM25 comparison and cite whether hosted embeddings or caches were
used.

## Review Criteria

Reviewers should reject benchmark contributions when:

- the Eventloom or query inputs are absent, generated locally, or untracked;
- `query_results` are missing or replaced with placeholders;
- citation coverage is not reported;
- the report makes a public claim from a candidate backend marked experimental
  in [api-inventory.md](../api-inventory.md);
- latency budgets are relaxed without explanation and changelog coverage;
- the Markdown sidecar disagrees with the JSON report.

Accepted benchmark changes should update [benchmarks.md](../benchmarks.md),
[testing.md](../testing.md), the changelog, and [README.md](../../README.md) when the
public benchmark story changes. Operational release steps live in
[runbook.md](../runbook.md).
