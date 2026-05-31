---
name: Benchmark contribution
about: Propose a benchmark workload, report, guardrail, or comparison update
title: "[Benchmark]: "
labels: benchmark
---

## Zaxy version

Version, install source, Python version, operating system, projection backend,
embedding provider, and reranker provider.

## Reproduction

Exact command used to generate the workload or report.

## Benchmark artifact path

Path under `benchmarks/` or `reports/` for the workload, Eventloom input, query
file, JSON report, and Markdown sidecar.

## Tracked inputs

Confirm whether Eventloom files, query files, workload files, and source
fingerprints are tracked by git.

## Metrics changed

Answer@5, Recall@5, citation coverage, checkout latency, retrieval lane
latency, token efficiency, memory footprint, rebuild recovery, or dashboard
load metrics affected.

## Public claim impact

State whether this changes README, docs, release gates, or external comparison
language.
