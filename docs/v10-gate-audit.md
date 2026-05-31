# v1.0 Gate Audit

This audit records the current v1.0 release gates and the evidence that proves
or blocks each one. It is intentionally command-focused so the release
candidate can distinguish local engineering proof from evidence that requires a
real external user or project.

The audit does not replace the release checklist. It explains what each gate is
expected to prove, where the command is recorded, and which gates still require
evidence before a stable release can be cut.

## Proven Local Gates

### Clean-repo UAT

Status: locally gated.

Evidence:

- `docs/release-validation-checklist.md` requires `scripts/beta-uat.sh`.
- `scripts/release-check.sh` includes the beta UAT command as a named release
  gate, so release runs cannot silently omit the clean-workspace path.
- `zaxy doctor --beta-readiness` reports the release-gate inventory used by
  automation.

### MCP smoke

Status: locally gated.

Evidence:

- The release checklist requires `python scripts/mcp_smoke_test.py`.
- The release gate surface coverage check requires an MCP smoke command in
  `scripts/release-check.sh`.
- The v0.6 MCP contract snapshot and response snapshot tests protect the public
  MCP tool surface used by this smoke path.

### LangGraph smoke

Status: locally gated.

Evidence:

- The release checklist requires
  `pytest tests/test_examples_v05.py::test_langgraph_example_runs_without_langgraph_dependency --no-cov -q`.
- `zaxy doctor --release-smoke` also exercises the dependency-light
  `examples/langgraph_memory.py` path.
- The smoke proves the native-beta checkout contract can be consumed without
  requiring LangGraph as a hard core dependency.

### Direct model integration smoke

This is the direct model integration smoke gate from the release checklist.

Status: locally gated.

Evidence:

- The release checklist requires
  `pytest tests/test_openai_compatible_adapter.py tests/test_claude_compatible_adapter.py --no-cov -q`.
- `tests/test_openai_compatible_adapter.py` covers the OpenAI-compatible memory
  activation helper outside MCP.
- `tests/test_claude_compatible_adapter.py` covers the Claude-compatible memory
  activation helper outside MCP.
- The release-smoke inventory also runs the no-network examples with fake
  provider-shaped clients.

### Coordinate mission smoke

Status: locally gated.

Evidence:

- The release checklist requires
  `pytest tests/test_examples_v05.py::test_coordinate_three_worker_example_runs --no-cov -q`.
- The example exercises a three-worker mission with review, approval, promoted
  state, checkout, audit reporting, and handoff.
- Coordinate CLI, MCP, and adapter tests cover the same workflow at lower
  levels.

### Benchmark guardrails

Status: locally gated.

Evidence:

- The release checklist requires `scripts/benchmark-guardrails.sh`.
- `zaxy doctor --beta-readiness` reports `benchmark_no_regression` and
  `backend_report_inputs`.
- Release benchmark checks require tracked inputs, query results, citation
  coverage, checkout quality floors, and p95/p99 latency budgets before public
  benchmark claims are accepted.

### Docs validation

Status: locally gated.

Evidence:

- The release checklist requires
  `python scripts/build-site-docs.py --check && scripts/validate-docs.sh --root .`.
- Static docs tests require rendered HTML for every core Markdown document and
  require `site/index.html` to link to the rendered pages rather than raw
  Markdown.

### Release smoke

Status: locally gated.

Evidence:

- The release checklist requires `zaxy doctor --release-smoke`.
- The release smoke inventory includes public examples, the LangGraph example,
  and the direct OpenAI-compatible and Claude-compatible examples.

### Coverage remains at or above 92%

Status: locally gated.

Evidence:

- The release checklist requires
  `pytest --cov --cov-report=xml && scripts/check-coverage.py --root . --coverage-xml coverage.xml`.
- `pyproject.toml` configures the coverage floor and release ratchet at 92
  percent.
- CI runs the coverage ratchet on the Python 3.13 matrix lane.

### Public surfaces are tagged

Status: locally gated.

Evidence:

- `docs/api-inventory.md` classifies public MCP, Python, CLI, Eventloom,
  projection-backend, and benchmark artifact surfaces as stable, beta,
  experimental, or internal.
- `docs/examples/v1-schema-freeze.json` binds the v1 freeze candidate contract
  files.
- Non-additive changes to stable or beta surfaces require
  `schema.migration.proposed` and `schema.migration.applied` events.

## Optional Post-Release Evidence

### External validation

Status: optional for v1.0 release.

The v1.0 release may ship without a dated note, issue, discussion, or case study
from a user or project outside the implementation session. The local gates prove
that first-run, MCP, LangGraph, direct model-call, Coordinate, docs, benchmark,
coverage, and stability-contract paths are represented in automation. External
validation remains useful post-release evidence, but it is no longer a release
blocker.

Required evidence:

- who ran the validation and when;
- which path they ran, such as `zaxy init`, `scripts/beta-uat.sh`,
  `python scripts/mcp_smoke_test.py`, `zaxy doctor --release-smoke`, or
  `python examples/coordinate_three_worker_project.py`;
- the result, time-to-first-success, friction, and any docs or code changes made
  in response;
- whether the validation supports the v1.0 positioning as Coordinator Memory
  for Agent Teams.

Use [external-validation.md](external-validation.md) and the External validation
GitHub issue template to collect the report when evidence exists.
Machine-checkable reports should start from
`docs/examples/external-validation-report.example.json`, use the
`zaxy.v1.external-validation-report` contract, record that the validator name must not be a placeholder, sample name, or implementation-session name,
record that validator name must not identify the implementing agent,
record that friction or failure narrative must not be `none`, `n/a`, or placeholder text,
record that shell must be concrete enough to identify the shell used,
record that Python version must be concrete major/minor evidence,
record that install source must be concrete enough to reproduce the install path,
record that Zaxy version or commit must be concrete, not `latest`, `current`, `main`, `master`, `head`, or `stable`,
include at least one evidence link that is an absolute `http` or `https` URL and
includes a concrete artifact path,
points to a reviewable evidence artifact instead of a repository homepage or collection page,
record that GitHub evidence links must use a supported artifact path,
record that GitHub evidence links must not include query strings,
record that GitHub evidence links must not include URL fragments,
record that GitHub evidence links must not include trailing slashes,
record that GitHub evidence links must not include empty path segments,
record that GitHub artifact path keywords must be lowercase,
record that GitHub issue, discussion, and pull-request links must use exact canonical positive-numbered artifact paths,
record that GitHub pull-request links must use `/pull/<number>`,
record that GitHub Actions run links must use exact `/actions/runs/<id>` paths with a concrete canonical positive numeric run ID,
record that GitHub release links must use exact `/releases/tag/<tag>` paths with a concrete non-vague release tag,
record that GitHub commit links must use `/commit/<sha>` with a full 40-character commit SHA,
record that GitHub file links (`blob`, `raw`, or `tree`) must use a full 40-character commit SHA ref and file path instead of a branch or tag,
record that raw.githubusercontent.com links must use a full 40-character commit SHA ref and file path instead of a branch or tag,
uses a fully qualified public hostname,
does not include credentials,
not `localhost`, loopback, link-local, unspecified, or private-network URL,
not an internal-only domain such as `.internal`, `.local`, `.lan`, `.test`, or `.invalid`,
not a reserved example domain, confirm command entries must be single-line strings,
confirm command entries must record executed commands, not echoed command text,
not `echo` or `printf` command text,
not compound shell commands,
not backgrounded shell commands,
not parenthesized shell groups,
not shell comments,
not help or version probes,
record that `first_run_local` reports must include `zaxy init`, `zaxy memory bootstrap`, `zaxy memory checkout`, and `zaxy doctor --beta-readiness`,
record that `other_documented` reports must include at least one substantive Zaxy validation command,
record that arbitrary or unknown `zaxy` command text does not count as validation evidence,
prove `commands` match the selected `validation_path`,
and pass `scripts/check-external-validation.py` before the optional evidence
row is updated.

Related references: [external-validation.md](external-validation.md),
[release-validation-checklist.md](release-validation-checklist.md),
[v09-gate-audit.md](v09-gate-audit.md), [v1-roadmap.md](v1-roadmap.md),
[testing.md](testing.md), [runbook.md](runbook.md), and [README.md](../README.md).
