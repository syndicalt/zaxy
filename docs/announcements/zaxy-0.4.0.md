# Zaxy 0.4.0: Coordinator Memory For Agent Teams

![Zaxy 0.4.0 release header](../assets/zaxy-coordinate-header.png)

Zaxy 0.4.0 is the release where the project stops being only an agent memory
system and becomes a coordinator memory layer for multi-agent work.

Spinning up agents is no longer the hard part. Worktrees, containers, task
runners, and agent CLIs can already isolate ten workers. The hard part starts
when those workers return: one found the current source of truth, one found an
older rule, two duplicated each other, one made an unsupported claim, and the
project now needs a coherent answer.

Zaxy 0.4.0 is built around that coordination problem.

## The New Product Shape

Zaxy Coordinate introduces a parent mission plus isolated worker sessions:

```text
parent mission
  -> worker sessions
  -> assignments
  -> cited findings
  -> conflict and stale-claim review
  -> accepted merge-back
  -> replayable project history
```

That is the important distinction. Worker sessions can investigate freely
without turning every hypothesis into shared truth. The parent mission becomes
the accepted state of the project. Rejected, stale, duplicated, and conflicting
claims stay auditable, but they do not pollute the trusted checkout by default.

This changes Zaxy's core claim:

> Isolated workers. Cited findings. Conflict detection. Accepted merge-back
> into one replayable project history.

That is a sharper product than generic persistent memory. It is designed for
teams of agents that need a reliable control plane, not just a larger
scratchpad.

## What Ships In 0.4.0

### Zaxy Coordinate

The release adds first-class coordination primitives for missions, workers,
assignments, findings, decisions, promotions, handoffs, and approval packets.
The Python and CLI surfaces expose a coordinator workflow that can start a
mission, create scoped workers, collect evidence-backed findings, review those
findings, and promote accepted state into the parent mission.

The key behavior is accepted-state checkout. A parent mission checkout should
answer from promoted project state, while pending worker claims remain visibly
separate. For agent teams, that boundary is the difference between useful
parallelism and shared-memory contamination.

### CoordinationBench

Zaxy 0.4.0 also introduces CoordinationBench as the benchmark lane for this new
direction. CoordinationBench asks whether a system can turn multiple isolated
worker outputs into one governed parent history.

The scorer measures accepted-finding precision and recall, conflict precision
and recall, stale-claim rejection, duplicate consolidation, evidence coverage,
parent-checkout answerability, citation coverage, Eventloom replayability,
token estimates, and coordination latency.

The release deliberately keeps the benchmark story honest:

| Lane | Score | What it means |
|------|-------|---------------|
| CoordinationBench v1 public lane | `1.000` | First-party public-label reproducibility result. |
| CoordinationBench v1-scale public lane | `1.000` | First-party public-label scale reproducibility result. |
| Public-derived holdout mean | `0.606` | Current frozen-adapter generalization baseline. |

The holdout result is the product signal to watch. It shows that the
replay-backed coordination layer is strong at accepted-state precision and
duplicate consolidation, while the frozen adapter still needs better
source-aware final answering, stale-source interpretation, and broader conflict
detection across public-derived cases.

Zaxy should not market a representative perfect CoordinationBench score. The
public v1 scores prove the adapter can satisfy the public contract; the `0.606`
holdout mean shows where the real next work is.

### Embedded Graph Runtime

The plain local path now centers on the embedded Kuzu projection backend. Neo4j,
pgGraph, and LatticeDB remain useful sidecar or evaluation tracks, but the
release direction is zero-friction local memory: install Zaxy, initialize the
workspace, and get Eventloom plus a rebuildable local graph projection without
requiring an external graph database for ordinary use.

That matters for adoption. Agent memory only helps if it is present at the
start of work. A sidecar-first setup creates friction exactly where users need
the system to disappear.

The backend benchmark gates now protect that embedded default with reproducible
reports, fingerprint checks, citation coverage, token-efficiency checks, and
latency budgets. Optional backends remain explicit, not accidental.

### Memory Bootstrap And Checkout

The model-facing memory contract is also tighter in 0.4.0. Zaxy exposes Memory
Bootstrap and Memory Checkout as first-class ways to tell an agent what memory
exists, what can be trusted, what needs refresh, and how to cite or reinforce
retrieved context.

The checkout object now carries diagnostics for answerability, confidence,
source lanes, citation coverage, unsupported compacted context, warnings, and
feedback guidance. That makes memory less magical and more operational: the
agent can see whether it is answering from current cited state, stale state, or
insufficient memory.

### Deterministic Capture And Activation

Zaxy 0.4.0 continues the local-first capture work. Deterministic capture records
session turns, tool calls, commands, and file edits into Eventloom so the memory
system can reconstruct what happened instead of relying only on manually written
summaries.

The release also treats activation as a product KPI. Capturing work after the
fact is not enough; useful memory needs to be checked out before substantive
work starts. The doctor, hook-status, and capture-soak paths now make that
visible with freshness, coverage, and token-efficiency diagnostics.

## Why This Release Matters

Most agent memory products are converging on retrieval: store facts, search
facts, inject facts. That is necessary, but it does not solve the coordination
layer for agent teams.

Multi-agent work needs three extra properties:

1. Isolation, so worker hypotheses do not become global truth immediately.
2. Governance, so findings are accepted, rejected, or deferred with evidence.
3. Replay, so the final project state can be audited back to the events that
   produced it.

Zaxy already had the substrate: Eventloom append-only history, temporal graph
projection, citations, Memory Checkout, MCP, and local capture. Version 0.4.0
connects those pieces into a product direction that is easier to explain and
harder for flat memory systems to copy.

The release is not claiming that Zaxy has solved all of coordination. It is
claiming the right architecture for the problem: parent mission state, worker
session isolation, evidence-backed promotion, and a benchmark that measures the
actual failure modes.

## Install Or Upgrade

The PyPI distribution is `zaxy-memory`; the import package and console command
remain `zaxy`.

```bash
pip install -U zaxy-memory
zaxy init
zaxy doctor
```

For local Codex onboarding:

```bash
zaxy init --preset local-codex
```

For the Coordinate docs:

```bash
zaxy coordinate --help
```

## Read Next

- [Zaxy Coordinate announcement](zaxy-coordinate.md)
- [Coordinate roadmap](../coordinate-roadmap.md)
- [Benchmarks](../benchmarks.md)
- [Getting started](../getting-started.md)
- [MCP interface](../mcp.md)

Zaxy 0.4.0 is the coordination release. The next bar is representative
CoordinationBench performance across real multi-agent workloads, independent
adapter review, and tighter activation so agents start every meaningful session
with current, cited project memory already in context.
