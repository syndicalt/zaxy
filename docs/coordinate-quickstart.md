# Coordinate Quickstart

Coordinate keeps worker-local exploration separate from trusted parent mission
state. worker-local findings are not trusted parent state until the coordinator
reviews and promotes them.

Use this workflow when several agents or people investigate overlapping parts of
a project and you need one accepted, replayable parent history at the end. The
worker sessions can disagree, duplicate each other, or cite stale evidence. The
parent mission should only receive findings after explicit review and promotion.

## Minimal Mission

```bash
zaxy coordinate start "Ship auth refactor" --mission auth-main
zaxy coordinate worker create --mission auth-main --worker auth-api
zaxy coordinate assign --mission auth-main --worker auth-api "Trace API auth failures"
zaxy coordinate report --mission auth-main --worker auth-api \
  --summary "API failures trace to expired JWKS cache handling" \
  --evidence "src/auth/jwks.py:42" \
  --claim-key auth.failure.cause \
  --claim-value expired-jwks-cache
zaxy coordinate brief --mission auth-main
FINDING_ID="$(python - <<'PY'
from pathlib import Path
from zaxy.coordination import CoordinationManager
brief = CoordinationManager(eventloom_path=Path('.eventloom')).brief('auth-main')
print(brief.findings[0].finding_id)
PY
)"
zaxy coordinate decide --mission auth-main --finding "$FINDING_ID" --status accepted
zaxy coordinate promote --mission auth-main --finding "$FINDING_ID"
zaxy coordinate checkout --mission auth-main
zaxy coordinate handoff --mission auth-main --summary "Accepted auth finding promoted."
```

Use `zaxy coordinate approval-packet` when a remote reviewer needs to inspect
pending or conflicted findings before promotion.

## What To Check

`zaxy coordinate brief` is the main review surface. It should show the mission,
workers, assignments, pending findings, accepted findings, stale findings,
conflicts, missing-evidence warnings, and the next coordinator action. If a
finding lacks evidence, keep it pending or reject it. If two workers cite
incompatible source states or claim different current values for the same key,
review the conflict before promoting either claim.

`zaxy coordinate checkout` returns accepted parent state by default. That is the
prompt surface future agents should trust. Use diagnostic checkout only when the
coordinator needs to inspect pending, conflicted, rejected, or stale worker-local
state. This is the main safety boundary: worker logs can be useful evidence, but
they are not shared project memory until review says they are.

For a larger runnable example, use:

```bash
python examples/coordinate_three_worker_project.py
```

That example creates three workers, records conflicting claims, accepts one
evidence-backed finding, promotes it, checks out accepted state, and creates a
handoff. It prints JSON so release smoke tests can verify the workflow without
reading raw Eventloom files.

Related references: [README.md](../README.md),
[Coordinate roadmap](coordinate-roadmap.md), [MCP Quickstart](mcp-quickstart.md),
and [benchmarks](benchmarks.md).
