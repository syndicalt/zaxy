# Changelog

All notable Zaxy release changes are recorded here.

## 0.1.0 - 2026-05-11

- Published the first public `zaxy-memory` package on PyPI.
- Added the `zaxy` console script for local onboarding, memory inspection, MCP serving, capture, projection, benchmarking, and release operations.
- Switched the publish workflow to PyPI Trusted Publishing so future releases use GitHub OIDC instead of long-lived PyPI API tokens.
- Shipped the current alpha memory substrate: Eventloom-backed provenance, Neo4j projection, Memory Checkout, deterministic capture, local onboarding, hooks, packet capture as an optional path, and benchmark tooling.
