# External Validation

External validation is an optional post-release evidence path for v1.0. Local
automation proves that release commands, examples, docs, benchmarks, and schema
contracts exist; outside-user evidence can later prove that a person outside the
implementation session can adopt the workflow without private context.

Status: optional for the v1.0 release. Attach a dated issue, discussion, case
study, or signed-off note when one becomes available.

## Who Should Run This

Use someone who has not contributed to the current implementation session. The
best validator is a developer who can install Python packages, run shell
commands, and evaluate whether Zaxy's public positioning as Coordinator Memory
for Agent Teams matches the actual workflow.

The validator should work from public docs and release artifacts only. Do not
walk them through private repo history, hidden environment variables, or
untracked local state. If they need help, record the exact gap and whether the
fix belongs in docs, `zaxy doctor`, `zaxy init`, an example, or a release gate.

## Validation Paths

Run at least one path. Run more than one if the release decision depends on a
specific audience.

### First-run local path

```bash
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
zaxy memory checkout "current project memory and next useful action" --eventloom-path .eventloom
zaxy doctor --beta-readiness
```

For the machine-checkable report, `first_run_local` reports must include `zaxy init`, `zaxy memory bootstrap`, `zaxy memory checkout`, and `zaxy doctor --beta-readiness`.

This path validates install, onboarding, Memory Bootstrap, Memory Checkout,
doctor guidance, and the no-sidecar local default.

### Coordinate workflow path

```bash
python examples/coordinate_three_worker_project.py
```

This path validates the v1.0 product thesis: multiple workers can report cited
findings, produce reviewable state, resolve conflicts, promote accepted memory,
assemble checkout context, and create a handoff without reading raw Eventloom
JSONL.

### Clean-repo release UAT path

```bash
scripts/beta-uat.sh
```

This path validates the scripted clean-workspace flow used by the release gate.
It is stronger than the first-run local path, but it should not replace a human
reading the docs and explaining what was confusing.

### Other documented path

If a validator uses another documented v1.0 path, the machine-checkable
`other_documented` reports must include at least one substantive Zaxy validation command.
arbitrary or unknown `zaxy` command text does not count as validation evidence.

## Required Evidence

Attach evidence in a GitHub issue, discussion, release note, or case-study
draft. The evidence must include:

- validator name or project, with enough context to show they are outside the
  implementation session; the validator name must not be a placeholder, sample name, or implementation-session name;
  validator name must not identify the implementing agent;
- date, operating system, shell, Python version, install source, and Zaxy
  version or commit; Zaxy version or commit must be concrete, not `latest`, `current`, `main`, `master`, `head`, or `stable`;
  shell must be concrete enough to identify the shell used;
  Python version must be concrete major/minor evidence;
  install source must be concrete enough to reproduce the install path;
- validation path run and exact commands used; command entries must be single-line strings,
  command entries must record executed commands, not echoed command text,
  not `echo` or `printf` command text,
  not compound shell commands, not backgrounded shell commands, not parenthesized shell groups, not shell comments, and not help or version probes;
- positive measured time to first useful checkout or first successful Coordinate result;
- whether Docker, Neo4j, Postgres, hosted credentials, or a graph password were
  unexpectedly required;
- command output or screenshots for success or failure;
- friction, confusing docs, unclear error messages, or missing remediation;
  friction or failure narrative must not be `none`, `n/a`, or placeholder text;
- at least one evidence link that is an absolute `http` or `https` URL to a
  reviewable issue, discussion, release note, report, or case-study artifact,
  includes a concrete artifact path,
  points to a reviewable evidence artifact instead of a repository homepage or collection page,
  GitHub evidence links must use a supported artifact path,
  GitHub evidence links must not include query strings,
  GitHub evidence links must not include URL fragments,
  GitHub evidence links must not include trailing slashes,
  GitHub evidence links must not include empty path segments,
  GitHub artifact path keywords must be lowercase,
  GitHub issue, discussion, and pull-request links must use exact canonical positive-numbered artifact paths,
  GitHub pull-request links must use `/pull/<number>`,
  GitHub Actions run links must use exact `/actions/runs/<id>` paths with a concrete canonical positive numeric run ID,
  GitHub release links must use exact `/releases/tag/<tag>` paths with a concrete non-vague release tag,
  GitHub commit links must use `/commit/<sha>` with a full 40-character commit SHA,
  GitHub file links (`blob`, `raw`, or `tree`) must use a full 40-character commit SHA ref and file path instead of a branch or tag,
  raw.githubusercontent.com links must use a full 40-character commit SHA ref and file path instead of a branch or tag,
  uses a fully qualified public hostname,
  does not include credentials,
  not `localhost`, loopback, link-local, unspecified, or private-network URL,
  not an internal-only domain such as `.internal`, `.local`, `.lan`, `.test`, or `.invalid`,
  and not a reserved example domain;
- release decision: pass, pass with follow-up, or fail until fixed.

## Report Template

For release-gate use, copy
`docs/examples/external-validation-report.example.json`, change `status` from
`example` to `validated`, replace the placeholder values, and run:

```bash
python scripts/check-external-validation.py path/to/external-validation-report.json
```

For teams that choose to require external evidence, pass the same report path
directly:

```bash
scripts/release-check.sh --root . --require-external-validation --external-validation-report reports/external-validation/external-validation-report.json
```

In required release mode, custom `--external-validation-cmd` values must be a
single `python scripts/check-external-validation.py <report-path>` invocation
with no shell suffixes.

The checker validates the `zaxy.v1.external-validation-report` contract and
rejects pending reports, reports from the current implementation session,
reports whose validator name identifies the implementation session,
reports whose validator name identifies the implementing agent,
unexpected sidecar or credential requirements, failed release decisions, missing
or vague shell evidence, vague Python version evidence, vague install-source evidence, missing
evidence links, evidence links that are not an absolute `http` or `https` URL,
bare-origin, single-label hostname, credential-bearing, local-only, private-network, internal-only domain, or reserved example-domain evidence URLs, reports
with repository home pages, collection pages, unsupported GitHub repo paths, GitHub query strings, URL fragments, trailing slashes, empty path segments, or mixed-case artifact keywords, noncanonical, nonpositive, nonnumeric, or malformed GitHub issue, discussion, or pull-request links, plural `/pulls/<number>` links, malformed GitHub Actions run links, GitHub latest-release redirects, vague or malformed GitHub release tag links, branch-like, short-SHA, or malformed GitHub commit links, GitHub `blob`, `raw`, or `tree` file links with moving refs, short refs, or missing file paths, or raw.githubusercontent.com links with moving refs, short refs, or missing file paths instead of evidence artifacts,
with missing or zero timing for first useful checkout,
where command entries do not record executed commands, reports where command
entries contain multiple lines, reports where command entries contain shell control operators,
including background operators and shell comments,
reports where command entries are only help or version probes, reports where `commands` match the
selected `validation_path` incorrectly, `first_run_local` reports missing any
required first-run command marker, `other_documented` reports missing a
substantive Zaxy validation command, `other_documented` reports with only
unknown `zaxy` command text, and
reports that do not support the v1.0 Coordinator Memory for Agent Teams
positioning. In a valid report, `commands` match the selected `validation_path`.

`zaxy doctor --beta-readiness` reports this as
`external_validation_evidence`. Missing default evidence is OK for the v1.0
release; a validated report at
`reports/external-validation/external-validation-report.json` keeps that check
green with attached evidence. To check a report before moving it into the
default reports directory, run:

```bash
zaxy doctor --beta-readiness --external-validation-report path/to/external-validation-report.json
```

For a hard local policy gate, require that evidence:

```bash
zaxy doctor --beta-readiness --require-external-validation --external-validation-report reports/external-validation/external-validation-report.json
```

```markdown
## External validation report

- Validator:
- Date:
- Project or environment:
- Zaxy version or commit:
- Operating system:
- Shell:
- Python version:
- Install source:
- Validation path:
- Exact commands:
- Time to first useful checkout:
- Time to first successful Coordinate result, if applicable:
- Unexpected sidecar, credential, or password requirement:
- What worked:
- Friction or failure:
- Least useful error or doc section:
- Suggested docs, doctor, init, example, or release-gate change:
- Release decision:
```

## Acceptance Criteria

External validation is acceptable when the report proves that at least one
outside user or project ran a documented v1.0 path, recorded the outcome, and
made a release decision that can be reviewed independently.

Passing validation does not require a perfect experience. It does require
specific evidence and a decision. A failed validation is still useful evidence
for follow-up work, but it does not block the v1.0 release unless maintainers
explicitly opt into `--require-external-validation`.

After the report is attached, update the optional evidence reference in
`docs/archive/v10-gate-audit.md` only if the report proves the requirement.
Then check the matching optional review item in
`docs/archive/release-validation-checklist.md`.

Related references: [first-run-validation.md](first-run-validation.md),
[coordinate-quickstart.md](coordinate-quickstart.md),
[v10-gate-audit.md](archive/v10-gate-audit.md),
[release-validation-checklist.md](archive/release-validation-checklist.md),
[testing.md](testing.md), and [README.md](../README.md).
