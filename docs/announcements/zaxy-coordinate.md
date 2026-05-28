# Zaxy Coordinate: The Memory Layer For Agent Teams

![Zaxy Coordinate header](../assets/zaxy-coordinate-header.png)

AI agents are getting easier to spawn. That is not the hard part anymore.

Worktrees isolate code. Containers isolate processes. Task runners can launch ten
agents in parallel. The missing layer is coordination: when those agents return
with findings, contradictions, partial evidence, stale assumptions, and competing
recommendations, what turns that activity into one reliable project history?

Zaxy's new direction is built around that problem.

## From Agent Memory To Coordinator Memory

Most memory products start with the same question: how does one agent remember
useful facts?

That matters, but it is not enough for real multi-agent work. A project with ten
agents does not need one shared scratchpad where every claim becomes equally
trusted. It needs a coordinator state layer:

- each worker keeps an isolated session
- each finding carries evidence
- conflicts are visible
- rejected claims stay auditable but do not pollute accepted memory
- accepted findings are promoted into one parent mission history

That is the core idea behind Zaxy Coordinate.

## The New Primitive: Mission And Workers

Zaxy already has the foundations: Eventloom append-only history, session-sharded
logs, temporal graph projection, cited Memory Checkout, replay, and MCP tools.
Zaxy Coordinate packages those foundations into a workflow for agent teams.

```text
Mission session
  -> worker sessions
  -> assignments
  -> findings
  -> evidence
  -> conflicts
  -> decisions
  -> accepted project memory
```

The parent mission session is the trusted project history. Worker sessions are
scoped investigation spaces. The coordinator promotes only accepted findings into
the parent.

That distinction is the product wedge: Zaxy is not just shared memory. It is
governed merge-back for agent work.

## The Command We Want To Matter

The flagship interface is the coordinator brief:

```bash
zaxy coordinate brief --mission auth-refactor
```

The brief should answer the questions every human or coordinator agent needs:

- What is the mission?
- Who is working on what?
- What did each worker find?
- Which findings are backed by evidence?
- Which findings conflict?
- Which findings are stale or duplicated?
- Which findings are ready to promote?
- What decision is needed next?

This is where Zaxy should feel different. A normal memory system retrieves
related context. A coordinator memory system explains the state of the work.

## Accepted-State Checkout

Shared memory is dangerous when worker hypotheses become durable truth too early.

Zaxy Coordinate introduces accepted-state checkout: by default, the parent
mission checkout returns only promoted, cited, current state. Pending worker
findings can still be visible, but they are labeled as unaccepted.

That gives agent teams a trust boundary:

```text
worker-local claim != accepted project memory
```

The distinction is small in a demo and critical in production.

## CoordinationBench

Retrieval benchmarks are useful, but they are becoming table stakes. Agent teams
need a different benchmark: can a system coordinate parallel work into a reliable
outcome?

CoordinationBench will test cases where several workers investigate overlapping
tasks. Some report true findings. Some duplicate each other. Some contradict each
other. Some rely on stale evidence. Some lack evidence entirely.

The benchmark should score:

- accepted-finding precision
- accepted-finding recall
- conflict recall
- stale-claim rejection
- evidence coverage
- parent-checkout answerability
- citation coverage
- replayability
- token cost
- brief latency

The point is not to prove that Zaxy can find related text. The point is to prove
that Zaxy can preserve the shape of coordinated work.

## What We Are Pruning

This direction also clarifies what should move out of the spotlight.

Zaxy should not lead with backend choice. Embedded graph is the default path;
Neo4j, pgGraph, and LatticeDB are advanced adapters and evaluation tracks.

Zaxy should not compete as another generic "persistent memory" library. The
field already has strong products there. Zaxy's sharper claim is agent-team
coordination with replayable evidence.

Zaxy should not let benchmarks sprawl into the product story. LongMemEval remains
important, but CoordinationBench should become the flagship proof for the new
direction.

## The Product Claim

Zaxy is the coordinator memory layer for multi-agent projects:

> Isolated workers. Cited findings. Conflict detection. Accepted merge-back into
> one replayable project history.

That is the hook. Spawn as many agents as you want. Zaxy's job is to keep the
project history coherent.

## Roadmap

The roadmap starts with four deliverables:

1. A first-class coordination event taxonomy for missions, workers, assignments,
   findings, evidence, conflicts, decisions, promotions, and handoffs.
2. A CLI and MCP surface for `zaxy coordinate start`, `worker create`, `assign`,
   `report`, `brief`, `decide`, `promote`, and `handoff`.
3. Accepted-state checkout so parent mission memory excludes unpromoted worker
   claims by default.
4. CoordinationBench, a same-harness benchmark for multi-agent coordination.

The long-term goal is not just better recall. It is better project control:
memory that can tell what happened, what was accepted, what was rejected, what
still conflicts, and why the final answer should be trusted.

