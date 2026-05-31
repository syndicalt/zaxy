---
name: External validation
about: Report a v1.0 first-run, Coordinate, or clean-repo UAT validation
title: "[External validation]: "
labels: validation, release
---

## Zaxy version

Version or commit, install source, concrete Python major/minor version, operating system, shell, and
projection backend if visible.
Shell must be concrete enough to identify the shell used.
Install source must be concrete enough to reproduce the install path.
Zaxy version or commit must be concrete, not `latest`, `current`, `main`, `master`, `head`, or `stable`.

## Validation path

Which path did you run: first-run local path, Coordinate workflow path,
clean-repo release UAT, or another documented v1.0 path?
`first_run_local` reports must include `zaxy init`, `zaxy memory bootstrap`, `zaxy memory checkout`, and `zaxy doctor --beta-readiness`.
`other_documented` reports must include at least one substantive Zaxy validation command.
Arbitrary or unknown `zaxy` command text does not count as validation evidence.

The machine-checkable report validator name must not be a placeholder, sample name,
or implementation-session name.
The machine-checkable report validator name must not identify the implementing agent.

## Reproduction

Exact commands, docs pages, examples, and environment variables used.
The machine-checkable report command entries must be single-line strings.
The machine-checkable report command entries must record executed commands, not echoed command text.
They must be workflow commands, not `echo` or `printf` command text, not compound shell commands, not backgrounded shell commands, not parenthesized shell groups, not shell comments, and not help or version probes.

## Time to first useful checkout

How long did it take to reach a useful `memory_checkout` result? If you ran the
Coordinate path instead, include time to first successful promoted checkout or
handoff.

## Evidence link

Link to the reviewable report, discussion, release note, log, screenshot bundle,
or case-study artifact. The machine-checkable report should start from
`docs/examples/external-validation-report.example.json`, and every evidence
link in that JSON report must be an absolute `http` or `https` URL and
include a concrete artifact path,
point to a reviewable evidence artifact instead of a repository homepage or collection page,
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
use a fully qualified public hostname,
does not include credentials,
not `localhost`, loopback, link-local, unspecified, or private-network URL,
not an internal-only domain such as `.internal`, `.local`, `.lan`, `.test`, or `.invalid`,
and not a reserved example domain.

## Friction or failure

Where did you get stuck? Include exact error text, confusing docs, missing
remediation, unexpected sidecar requirements, or unclear output.
Friction or failure narrative must not be `none`, `n/a`, or placeholder text.

## Release decision

Choose one and explain: pass, pass with follow-up, or fail until fixed.
