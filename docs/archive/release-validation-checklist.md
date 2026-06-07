# Release Validation Checklist

Use this checklist before cutting a v1.x release. It records the command-level
evidence required by the roadmap and keeps optional external validation separate
from local automation.

## Required Gates

- [ ] Clean-repo UAT passes.
  Command: `scripts/beta-uat.sh`
- [ ] MCP smoke passes.
  Command: `python scripts/mcp_smoke_test.py`
- [ ] LangGraph smoke passes.
  Command:
  `pytest tests/test_examples_v05.py::test_langgraph_example_runs_without_langgraph_dependency --no-cov -q`
- [ ] At least one direct model integration smoke passes.
  Command:
  `pytest tests/test_openai_compatible_adapter.py tests/test_claude_compatible_adapter.py --no-cov -q`
- [ ] Coordinate mission smoke passes.
  Command:
  `pytest tests/test_examples_v05.py::test_coordinate_three_worker_example_runs --no-cov -q`
- [ ] Benchmark guardrails pass.
  Command: `scripts/benchmark-guardrails.sh`
- [ ] StateRecoveryBench accepted-state guardrails pass.
  Command:
  `zaxy state-recovery-benchmark --output-dir /tmp/zaxy-state-recovery --workload reports/benchmarks/state-recovery-v1/state-recovery-workload.json`
- [ ] Tracked StateRecoveryBench artifact validates.
  Command:
  `python scripts/check-state-recovery-benchmark.py reports/benchmarks/state-recovery-v1/state-recovery-benchmark.json --workload reports/benchmarks/state-recovery-v1/state-recovery-workload.json --require-git-tracked-inputs`
- [ ] Docs validation passes.
  Command:
  `python scripts/build-site-docs.py --check && scripts/validate-docs.sh --root .`
- [ ] Release smoke passes.
  Command: `zaxy doctor --release-smoke`
- [ ] Coverage remains at or above 92%.
  Command: `pytest --cov --cov-report=xml && scripts/check-coverage.py --root . --coverage-xml coverage.xml`
- [ ] Public surfaces are tagged with their stability level.
  Evidence: `docs/api-inventory.md` and `docs/examples/v1-schema-freeze.json`.
## Aggregate Commands

Run the local release inventory:

```bash
zaxy doctor --beta-readiness
```

External validation is optional for v1.0. Without a report, the inventory should
report `external_validation_evidence` as `ok` with an optional evidence message.
After a real report is written to
`reports/external-validation/external-validation-report.json`, the same check
should report `ok` with attached evidence.

Run the full release gate:

```bash
scripts/release-check.sh --root .
```

The release gate may only skip an environment-specific smoke with an explicit
`SKIP:<reason>` command value. Do not remove a gate or leave it blank.

## Artifact Review

- Confirm `CHANGELOG.md` covers every release band from 0.4 through the target
  release.
- Confirm `docs/migration.md` covers upgrades from 0.4 through the release
  candidate.
- Confirm `docs/announcements/zaxy-v1.0.md` includes positioning, examples,
  benchmark evidence, limitations, roadmap beyond 1.0, and optional external
  validation.
- Confirm benchmark claims link to tracked report artifacts and disclose
  baselines, latency, citation coverage, and limitations.
- Confirm StateRecoveryBench claims are scoped to the production
  `memory_fabric_checkout` baseline; associative projection rows remain
  diagnostic research baselines.
- Confirm no public docs recommend internal surfaces from `docs/api-inventory.md`.
- Confirm any attached external validation report uses
  `docs/examples/external-validation-report.example.json` as the source contract
  for `zaxy.v1.external-validation-report`, records that the validator name must not be a placeholder, sample name, or implementation-session name,
  records that validator name must not identify the implementing agent,
  records that friction or failure narrative must not be `none`, `n/a`, or placeholder text,
  records that shell must be concrete enough to identify the shell used,
  records that Python version must be concrete major/minor evidence,
  records that install source must be concrete enough to reproduce the install path,
  records that Zaxy version or commit must be concrete, not `latest`, `current`, `main`, `master`, `head`, or `stable`,
  records that first-useful-checkout timing must be positive,
  includes at least one evidence link that is an absolute `http` or `https` URL and
  includes a concrete artifact path,
  points to a reviewable evidence artifact instead of a repository homepage or collection page,
  records that GitHub evidence links must use a supported artifact path,
  records that GitHub evidence links must not include query strings,
  records that GitHub evidence links must not include URL fragments,
  records that GitHub evidence links must not include trailing slashes,
  records that GitHub evidence links must not include empty path segments,
  records that GitHub artifact path keywords must be lowercase,
  records that GitHub issue, discussion, and pull-request links must use exact canonical positive-numbered artifact paths,
  records that GitHub pull-request links must use `/pull/<number>`,
  records that GitHub Actions run links must use exact `/actions/runs/<id>` paths with a concrete canonical positive numeric run ID,
  records that GitHub release links must use exact `/releases/tag/<tag>` paths with a concrete non-vague release tag,
  records that GitHub commit links must use `/commit/<sha>` with a full 40-character commit SHA,
  records that GitHub file links (`blob`, `raw`, or `tree`) must use a full 40-character commit SHA ref and file path instead of a branch or tag,
  records that raw.githubusercontent.com links must use a full 40-character commit SHA ref and file path instead of a branch or tag,
  uses a fully qualified public hostname,
  does not include credentials,
  not `localhost`, loopback, link-local, unspecified, or private-network URL,
  not an internal-only domain such as `.internal`, `.local`, `.lan`, `.test`, or `.invalid`,
  not a reserved example domain, confirms command entries must be single-line strings,
  confirms command entries must record executed commands, not echoed command text,
  not `echo` or `printf` command text,
  not compound shell commands,
  not backgrounded shell commands,
  not parenthesized shell groups,
  not shell comments,
  not help or version probes,
  records that `first_run_local` reports must include `zaxy init`, `zaxy memory bootstrap`, `zaxy memory checkout`, and `zaxy doctor --beta-readiness`,
  records that `other_documented` reports must include at least one substantive Zaxy validation command,
  records that arbitrary or unknown `zaxy` command text does not count as validation evidence,
  proves `commands` match the selected `validation_path`,
  and passes `scripts/check-external-validation.py`.

Related references: [external-validation.md](../external-validation.md),
[v10-gate-audit.md](v10-gate-audit.md), [v09-gate-audit.md](v09-gate-audit.md),
[testing.md](../testing.md), [runbook.md](../runbook.md), [benchmarks.md](../benchmarks.md),
and [README.md](../../README.md).
