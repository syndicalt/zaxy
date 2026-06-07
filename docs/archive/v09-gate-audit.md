# v0.9 Gate Audit

This audit records the current v0.9 hardening gates and the repo evidence that
proves or blocks each one. It is intentionally separate from the roadmap so
release work can distinguish completed engineering gates from optional external
validation that can arrive after release.

## Proven Gates

### Full Release Gate on Python 3.11, 3.12, and 3.13

Status: proven by CI configuration.

Evidence:

- `.github/workflows/ci.yml` runs the unit test matrix on Python 3.11, 3.12,
  and 3.13.
- The matrix command is
  `pytest -m "not integration" --benchmark-disable --cov --cov-report=xml`.
- Python 3.13 runs `scripts/check-coverage.py --root . --coverage-xml coverage.xml`
  as the canonical coverage ratchet check.

### Coverage Remains At Or Above 92 Percent

Status: proven by project configuration and release checks.

Evidence:

- `pyproject.toml` sets `[tool.coverage.report] fail_under = 92`.
- `pyproject.toml` sets `[tool.zaxy.coverage] min_total_percent = "92.00"`.
- `scripts/release-check.sh` runs coverage XML generation and
  `scripts/check-coverage.py`.

### No Public Benchmark Claim Depends On Untracked Or Local-Only Inputs

Status: proven by beta readiness and release benchmark checks.

Evidence:

- `zaxy doctor --beta-readiness` reports `backend_report_inputs` and
  `benchmark_no_regression`.
- Release benchmark commands require `--require-git-tracked-inputs`,
  `--require-query-results`, `--verify-report-fingerprints`, Markdown sidecars,
  citation coverage, and p95/p99 latency budgets.
- `docs/benchmark-contributions.md` documents tracked Eventloom inputs,
  query files, `query_results`, citation coverage, and report sidecars as
  contribution requirements.

### API Inventory Has No Undocumented Stable Surfaces

Status: proven for the current freeze candidate.

Evidence:

- `docs/api-inventory.md` classifies MCP tools, Python exports, CLI commands,
  Eventloom event families, projection backend contracts, and benchmark
  artifact schemas.
- `docs/examples/v1-schema-freeze.json` binds the candidate contract files.
- Stable or beta non-additive changes require `schema.migration.proposed` and
  `schema.migration.applied`.

### Migration Guide Covers 0.4 To 0.9

Status: proven by documentation and tests.

Evidence:

- `docs/migration.md` covers upgrade bands from 0.4 to 0.5, 0.5 to 0.6,
  0.6 to 0.7, 0.7 to 0.8, and 0.8 to 0.9.
- The guide covers compatibility tests and non-destructive rollback.
- Packaging tests assert the required release-band headings and core commands.

### Release Gate Surface Coverage

Status: proven by beta readiness.

Evidence:

- `zaxy doctor --beta-readiness` reports `release_gate_surface_coverage`.
- `scripts/release-check.sh` names public examples, MCP smoke, LangGraph smoke,
  Coordinate mission smoke, benchmark comparison, docs validation, and beta UAT.
- Intentional omissions must be written as `SKIP:<reason>`.

## Optional Evidence

### External User Feedback

Status: optional for v1.0 release.

External user feedback remains useful evidence, but it is not a v1.0 release
blocker. Current repo evidence proves the first-run and Coordinate paths are
documented and gated; outside-user evidence can be attached after release.

Required evidence:

- a dated note, issue, discussion, or case study from a user outside the current
  implementation session;
- the path they ran, such as `zaxy init`, `zaxy memory bootstrap`,
  `zaxy memory checkout`, `zaxy doctor --beta-readiness`, or
  `python examples/coordinate_three_worker_project.py`;
- result, friction, and whether docs or code changed in response.

Related references: [v1-roadmap.md](v1-roadmap.md), [testing.md](../testing.md),
[first-run-validation.md](../first-run-validation.md), [coordinate-quickstart.md](../coordinate-quickstart.md),
and [README.md](../../README.md).
