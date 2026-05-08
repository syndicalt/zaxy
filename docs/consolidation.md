# Geometry-Aware Consolidation

Zaxy's current release path treats vectors as a retrieval signal, not as the
source of truth. That distinction becomes more important when memory stores are
compressed. Recent work on the geometry of consolidation argues that replacing
many embedded memories with fewer representatives can preserve broad semantic
coverage while damaging identity-level retrieval when clusters are geometrically
spread. The practical lesson for Zaxy is straightforward: context compaction
must preserve identity, citations, temporal validity, and replayability.

This document is a product/research roadmap note, not a completed compaction
feature claim. The current system already has the primitives needed for
identity-preserving context memory: Eventloom event sequences and hashes, typed
graph entities, source path and line citations, transcript source and turn
identifiers, temporal validity windows, exact lookup, keyword search, vector
search, and traversal. The benchmark harness now includes a consolidation
collapse lane and identity-recall metric, and `zaxy compact --audit` applies
the same safety model before any log rewrite or snapshot operation.

## Motivation

Agent systems eventually need to compact context. Raw transcripts grow,
documents change, project histories become long, and a context engine needs a
way to keep retrieval fast and context windows small. The unsafe shortcut is to
cluster memories and replace each cluster with a centroid or summary that is
treated as the memory itself.

That shortcut is dangerous when the user later needs a specific identity:

- the exact event that introduced a fact;
- the document path and line where a requirement appears;
- the transcript turn where a decision was made;
- the previous value of a now-updated preference;
- the task connected to a goal;
- the source record behind an assembled context chunk.

A centroid, summary, or semantic rollup can remain topically correct while
losing the specific identity needed for accountable retrieval. Zaxy should use
compression to accelerate discovery and reduce token load, but it should never
allow compression to become the only authority for truth.

## Design Principle

The core policy is:

> Vectors discover neighborhoods. Eventloom, graph identity, temporal validity,
> and source citations establish truth.

In practice, this means consolidated representations are allowed to point to
source records, rank neighborhoods, summarize clusters, and accelerate query
routing. They are not allowed to erase the ability to recover the original
event, document chunk, transcript turn, or graph fact that justified an answer.

## Planned Capabilities

### 1. Consolidation Risk Audit

Before compacting a cluster, Zaxy should measure whether the cluster is safe to
compress. The first audit can be deliberately simple and deterministic:

- mean within-cluster cosine distance;
- nearest-neighbor preservation before and after consolidation;
- identity recall for every source item under the intended query policy;
- count of unique event/document/transcript identifiers in the cluster;
- citation coverage after compaction;
- temporal-window diversity inside the cluster.

Clusters with high spread, many distinct identifiers, mixed validity windows, or
poor identity recall should be marked unsafe for centroid-style consolidation.

### 2. Identity Invariant Tests

Every compaction operator should produce a test plan. For each source record in
the compacted set, Zaxy should be able to ask: can the system still retrieve
this identity when the query names its durable identifier?

Examples:

- `eventloom://default/events/42#abcdef123456`;
- `docs/architecture.md:18`;
- `session-0007:turn-2`;
- `user-0007:theme`;
- `task-0007`;
- `Goal 0007`.

If a compacted representation improves topic recall but fails these identity
checks, it is not acceptable as an authoritative memory replacement.

### 3. Medoid And Exemplar Compaction

When a cluster is safe, a centroid or summary may be useful as a routing
artifact. When a cluster is not safe, Zaxy should prefer medoids, exemplars, or
small cited representative sets. A medoid has the advantage of being a real
source item with provenance. Exemplars can preserve multiple identities when a
cluster contains several distinct facts that only look semantically similar.

Zaxy should store compacted artifacts as projections with explicit backpointers,
not as replacements for the Eventloom log or cited source chunks.

### 4. Geometry-Aware Benchmark Lane

The benchmark harness includes a consolidation lane that compares identity
preservation for raw retrieval, vector-style baselines, live Zaxy retrieval, and
a centroid baseline that keeps one representative text. The lane uses
near-duplicate source records with distinct durable identifiers so reports can
separate topical coverage from exact identity recall.

Future expansions should compare:

- centroid summaries;
- medoid representatives;
- exemplar sets;
- vector-only compacted retrieval;
- Zaxy replay plus graph-backed retrieval;
- Zaxy compacted projections with identity invariants.

The lane scores both coverage and identity. Coverage asks whether the right
topic or cluster is found. Identity asks whether the exact source record is
recoverable. Zaxy should optimize for both, with identity treated as a release
invariant for authoritative context.

### 5. Context Assembly Policy

Future context assembly should make compaction status visible. A compacted
summary can be included in a prompt as orientation, but final facts should be
assembled from cited events, graph facts, document chunks, and transcript turns.
If only a summary is available, the answer should be marked as degraded or
unsupported by source-level evidence.

## Non-Goals For The Current Release

The current release should not attempt to implement a full spectral
consolidation algorithm, learned compression system, or claim direct superiority
over external consolidation methods. The near-term objective is to keep the
release candidate stable while documenting the safety model and adding future
benchmark coverage.

The current release also should not claim that Zaxy avoids latent-space failure
automatically. Zaxy avoids making latent vectors authoritative only when callers
use the identity-preserving retrieval and assembly policies described here.

## Current Safety Check

Run `zaxy compact PATH --audit` before compaction. The command is
non-destructive: it reads the Eventloom log, verifies hash-chain integrity,
measures source identity recall for a one-representative compaction candidate,
reports citation coverage for document and transcript sources, and reports mean
within-cluster embedding distance using the deterministic local embedding
provider. Unsafe reports exit non-zero and list missing identities or citation
gaps.

Use `zaxy compact PATH --audit --json` for automation.

## Roadmap

1. Expand the consolidation-collapse benchmark with mixed temporal validity and
   transcript/session identities.
2. Add medoid/exemplar projection storage with backpointers to Eventloom and
   source citations.
3. Add context assembly warnings when output depends on a compacted summary
   without source-level support.

## References

- [Retrieval](retrieval.md)
- [Benchmark review](benchmark-review.md)
- [Architecture](architecture.md)
- [Testing](testing.md)
- [README](../README.md)
- Geometry of Consolidation repository:
  <https://github.com/niashwin/geometry-of-consolidation>
