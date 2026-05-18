# Changelog

All notable Zaxy release changes are recorded here.

## 0.2.2 - 2026-05-18

- Added pgGraph projection integrity and inferred-edge audit status support.
- Routed read-only memory graph status commands through the backend selector so
  pgGraph can use the same operator diagnostics as Neo4j.
- Expanded dashboard and pgGraph test coverage to preserve the release coverage
  ratchet.

## 0.2.1 - 2026-05-15

- Added first-class Hermes Agent MCP config rendering and explicit `config.yaml`
  merge support through `zaxy ide-config hermes`.
- Kept Hermes Agent onboarding workspace-neutral so global MCP config does not
  pin `EVENTLOOM_PATH`, `EVENTLOOM_THREAD`, or `ZAXY_DOMAIN` to one repository.
- Added PyYAML packaging support and documentation for Hermes Agent MCP install
  targets.

## 0.2.0 - 2026-05-15

- Promoted the beta release to a stable package so default `pip install zaxy-memory` resolves to the current Zaxy release without prerelease flags.
- Preserved the 0.2.0 beta release evidence and benchmark claims while making the same production-ready memory, capture, checkout, graph, and benchmark hardening available as the latest stable PyPI version.

## 0.2.0b1 - 2026-05-15

- Promoted Zaxy to its first beta packaging track with clean CI, release smoke, beta readiness, and trusted PyPI publishing gates.
- Hardened model-facing memory UX with Memory Bootstrap, Memory Checkout diagnostics, feedback guidance, source-aware context assembly, and shared checkout policy across core and MCP paths.
- Expanded deterministic capture and onboarding with local Codex capture, hook status coverage, leak detection, happy-path infrastructure profiles, and clean-repo UAT.
- Improved graph projection and auditability with hash-linked Eventloom event paths, source citation edges, temporal entity version edges, inferred-edge audit metadata, and graph projection integrity checks.
- Added and archived MemPalace-comparable benchmark evidence, including guardrails for mean score, Answer@5, Recall@5, citation coverage, and latency budgets.
- Hardened long-memory retrieval and synthesis to reach the current archived beta benchmark report: mean 0.950, Answer@5 0.950, citation coverage 1.000, and R@1/R@5/R@10 0.990.

## 0.1.0 - 2026-05-11

- Published the first public `zaxy-memory` package on PyPI.
- Added the `zaxy` console script for local onboarding, memory inspection, MCP serving, capture, projection, benchmarking, and release operations.
- Switched the publish workflow to PyPI Trusted Publishing so future releases use GitHub OIDC instead of long-lived PyPI API tokens.
- Shipped the current alpha memory substrate: Eventloom-backed provenance, Neo4j projection, Memory Checkout, deterministic capture, local onboarding, hooks, packet capture as an optional path, and benchmark tooling.
