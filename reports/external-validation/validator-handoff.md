# Zaxy v1.0 External Validation Handoff

This packet is for validators who were not part of the Zaxy implementation
session. Use only the public 1.0.0 docs, README, install instructions, and
`docs/external-validation.md`. Do not rely on private walkthrough context.

Current release commit for this handoff:

```text
4f9e366b1f394434ed17b8cca1c154c1bea47a5f
```

Reviewable validation request:

```text
https://github.com/syndicalt/zaxy/issues/17
```

## Validator Path

Run one documented path and record exact commands, command output or
screenshots, timing, environment, friction, and release decision.

Easiest first-run path:

```bash
zaxy init
zaxy memory bootstrap --eventloom-path .eventloom
zaxy memory checkout "current project memory and next useful action" --eventloom-path .eventloom
zaxy doctor --beta-readiness
```

Stronger Coordinate path:

```bash
python examples/coordinate_three_worker_project.py
```

Stronger clean-repo release UAT path:

```bash
scripts/beta-uat.sh
```

## Evidence To Capture

- Validator or project name, with enough context to show the validator is
  outside the implementation session.
- Date, OS, shell, Python version, install source, and exact Zaxy version or
  commit.
- Exact commands run.
- Time to first useful Memory Checkout or first successful Coordinate result.
- Command output or screenshots.
- What worked.
- What was confusing or failed.
- Whether Docker, Neo4j, Postgres, hosted credentials, or passwords were
  unexpectedly required.
- Release decision: `pass`, `pass_with_follow_up`, or `fail_until_fixed`.
  A `fail_until_fixed` result is useful evidence for follow-up work. Do not
  convert it into the machine-checkable JSON until the blocking issue is fixed
  and a validator records `pass` or `pass_with_follow_up`.

## Reviewable Artifact

Put the evidence in a GitHub issue, GitHub discussion, release note, PR
comment, or public/semi-public case-study document. The final machine-readable
report must include at least one absolute `http` or `https` evidence link to
that concrete artifact.

After the evidence exists, copy
`docs/examples/external-validation-report.example.json` to:

```text
reports/external-validation/external-validation-report.json
```

Change `status` to `validated`, replace every placeholder with the real
evidence, use `pass` or `pass_with_follow_up` for `release_decision`, and
validate it from this repository root:

```bash
python scripts/check-external-validation.py reports/external-validation/external-validation-report.json
zaxy doctor --beta-readiness --require-external-validation --external-validation-report reports/external-validation/external-validation-report.json
```

Do not create `external-validation-report.json` with placeholder or
self-generated data. External validation is optional post-release evidence for
1.0.0; teams that choose to require it can use the two commands above as their
local hard gate.
