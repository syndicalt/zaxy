# FleetBench (scaffold)

- version: `fleet-v1`
- fingerprint: `d4619d5737705c3ed67d62aaebbe68c9a22acfeac1efff072590407e536cfa30` (scored fields only; latency excluded)
- cross_agent_transfer scope: `within_mission_proxy` (within-mission proxy; fleet-wide pending I7)

Fleet/coordination scaling axes measured over real CoordinationBench runs. Each row is one scale point (worker count); every axis is exact-scored and deterministic except `latency_ms` (real wall-clock).

## Scaling axes

| worker_count | mission_count | coordination_quality | governance_correctness | cross_agent_transfer (proxy) | returned_tokens | injected_tokens | token_efficiency | latency_ms |
|--------------|---------------|----------------------|------------------------|------------------------------|-----------------|-----------------|------------------|------------|
| 3 | 1 | 0.907407 | 1.0 | 1.0 | 400 | 186 | 0.535 | 19.967 |
| 5 | 1 | 0.907407 | 1.0 | 1.0 | 450 | 186 | 0.586667 | 9.146 |
| 8 | 1 | 0.907407 | 1.0 | 1.0 | 525 | 186 | 0.645714 | 10.32 |
| **mean** | 1 | 0.907407 | 1.0 | 1.0 | 458 | 186 | 0.589127 | 13.144 |

Axis directions: `coordination_quality`, `governance_correctness`, `cross_agent_transfer`, and `token_efficiency` are higher-is-better in [0, 1]. `token_efficiency` = fraction of raw worker-log tokens NOT injected into the governed brief (`1 - injected/returned`); `returned_tokens` = raw worker logs, `injected_tokens` = governed accepted-state brief.

## Metric provenance

| Metric | Status |
|--------|--------|
| coordination_quality | REAL (exact, deterministic) |
| governance_correctness | REAL (exact, deterministic) |
| returned_tokens / injected_tokens / token_efficiency | REAL (deterministic `_approx_tokens` estimates) |
| cross_agent_transfer | PROXY / SCAFFOLD (within_mission_proxy; fleet-wide pending I7) |
| mission_count / worker_count | REAL (scaling point) |
| latency_ms | REAL wall-clock (non-deterministic; excluded from fingerprint) |

## Scaffold caveats

- cross_agent_transfer is a SCAFFOLD within-mission PROXY (scope=within_mission_proxy): it scores worker->parent promotion propagation inside a single mission, NOT fleet-wide cross-agent transfer. Fleet-wide cross-agent transfer is realized in I7.
- latency_ms is REAL wall-clock and environment-dependent; it is EXCLUDED from the fingerprint and from equality/determinism checks.
- returned_tokens and injected_tokens are _approx_tokens estimates (len//4), not tokenizer-exact counts.
- coordination_quality and governance_correctness are REAL exact-scored, deterministic aggregates of CoordinationBench signals.

## Reproduction

Regenerate this scaffold report over real CoordinationBench cases:

```bash
env PYTHONPATH=src EMBEDDING_ENABLED=true EMBEDDING_PROVIDER=hash EMBEDDING_DIMENSION=1536 \
  python -c 'from pathlib import Path; from zaxy_benchmarks.fleet_benchmark import run_fleet_benchmark; run_fleet_benchmark(Path("reports/experimental/fleet-benchmark-scaffold"))'
```

The `zaxy` CLI command for FleetBench is wired separately by the orchestrator.
