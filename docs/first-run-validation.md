# First-Run Validation

Use this checklist with someone who has not worked on Zaxy before. The goal is
to verify that a normal local developer can install, initialize, inspect memory,
and run one example in less than five minutes.

The validation should happen in a clean directory, not in a repo where Zaxy has
already been initialized. Do not pre-create `.eventloom`, `.env.local`, MCP
config files, or graph projection artifacts. The point is to learn whether the
public instructions are enough for a new user to reach a useful first checkout
without private project knowledge.

## Commands

```bash
pipx install zaxy-memory
zaxy init
zaxy bootstrap --eventloom-path .eventloom
zaxy checkout "current project memory and next useful action" --eventloom-path .eventloom
zaxy doctor --eventloom-path .eventloom
python examples/single_agent_memory.py
```

Success means the user can explain where local memory lives, can see bootstrap
guidance, can run checkout without starting a sidecar graph service, and can run
one example that prints structured output. If `zaxy doctor` reports a warning,
record the exact text and whether the remediation was understandable. If a
command requires Docker, Neo4j, Postgres, a graph password, or a hosted service
for the local path, mark the validation as failed.

## Report

- Operating system:
- Shell:
- Python version:
- Install method:
- Time to successful `zaxy doctor`:
- Time to first successful example:
- Did any command require Docker, Neo4j, Postgres, or a graph password?
- Where did you get stuck?
- Which error message was least useful?
- What should the quick start say differently?

For release readiness, copy the measured timings into
`docs/examples/first-run-timing-report.json`. `zaxy doctor --beta-readiness`
checks that `time_to_successful_doctor_seconds` and
`time_to_first_successful_example_seconds` are each at or below 300 seconds and
that the clean local path does not require a sidecar service.

## Follow-Up

Turn every repeated failure into one of three changes: clearer docs, a `doctor`
check with a specific remediation, or a safer default in `zaxy init`. Avoid
adding new setup options as the first response; the v0.5 release gate is about a
boring local path that works without users understanding the full architecture.

Related references: [README.md](../README.md), [Getting Started](getting-started.md),
[MCP Quickstart](mcp-quickstart.md), and [site/index.html](../site/index.html).
