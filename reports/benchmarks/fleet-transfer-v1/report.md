# FleetBench

- version: `fleet-v2`
- workload fingerprint: `887f206c12a29b8dd37fbd9b439e07d6b6a2b639915a5762013d2e197104dd66`
- fingerprint: `868b0b964b8de1319802e3575392098e773b9f7950144d125183627b7a079d9d` (scored fields only; latency excluded)
- cross_agent_transfer scope: `fleet_wide` (real fleet-wide transfer with a never-enrolled negative control)

Fleet/coordination scaling axes measured over real CoordinationBench missions and real FleetManager propagation. Each row is one scale point (worker count); every axis is exact-scored and deterministic except `latency_ms`.

## Scaling axes

| worker_count | mission_count | coordination_quality | governance_correctness | cross_agent_transfer | control | receivers | promotions | gated | returned_tokens | injected_tokens | token_efficiency | latency_ms |
|--------------|---------------|----------------------|------------------------|----------------------|---------|-----------|------------|-------|-----------------|-----------------|------------------|------------|
| 3 | 1 | 0.907407 | 1.0 | 1.0 | 1.0 | 2 | 1 | 0 | 400 | 207 | 0.4825 | 389.76 |
| 5 | 1 | 0.922222 | 1.0 | 1.0 | 1.0 | 4 | 3 | 0 | 841 | 558 | 0.336504 | 572.556 |
| 8 | 1 | 0.923977 | 1.0 | 1.0 | 1.0 | 7 | 6 | 0 | 1580 | 1085 | 0.313291 | 783.255 |
| **mean** | 1 | 0.917869 | 1.0 | 1.0 | 1.0 | 4 | 3 | 0 | 940 | 617 | 0.377432 | 581.857 |

Axis directions: `coordination_quality`, `governance_correctness`, `cross_agent_transfer`, `cross_agent_transfer_control`, and `token_efficiency` are higher-is-better in [0, 1]. `token_efficiency` = fraction of raw worker-log tokens NOT injected into the governed brief (`1 - injected/returned`).

## Metric provenance

| Metric | Status |
|--------|--------|
| coordination_quality | REAL (exact, deterministic) |
| governance_correctness | REAL (exact, deterministic) |
| returned_tokens / injected_tokens / token_efficiency | REAL (deterministic `_approx_tokens` estimates) |
| cross_agent_transfer | REAL (fleet_wide; governed delivery, not relevance) |
| cross_agent_transfer_control | REAL (never-enrolled negative control) |
| mission_count / worker_count | REAL (scaling point) |
| latency_ms | REAL wall-clock (non-deterministic; excluded from fingerprint) |

## Caveats

- cross_agent_transfer is REAL fleet-wide transfer (scope=fleet_wide): the mission's accepted findings are propagated by the origin agent through the real I4 gate and every OTHER enrolled agent's real checkout_memory is scored on whether it received them.
- cross_agent_transfer_control is the mandatory negative control: a never-enrolled stranger runs the identical checkout. It is 1.0 only when the stranger receives NOTHING. A transfer number with control < 1.0 is not creditable, because a metric that cannot tell an enrolled agent from a stranger measures nothing.
- SCOPE LIMIT: the checkout fleet lane returns every ACTIVE promotion an enrolled agent is entitled to, and is not filtered by query relevance. cross_agent_transfer therefore measures governed DELIVERY (does propagated memory reach the right agents and only them), NOT retrieval ranking or relevance. No relevance claim may cite this number.
- fleet_promotion_gated_count counts propagations the I4 gate held for review; they are excluded from the transfer denominator, because withholding them is the gate working rather than a transfer failure.
- latency_ms is REAL wall-clock and environment-dependent; it is EXCLUDED from the fingerprint and from equality/determinism checks.
- returned_tokens and injected_tokens are _approx_tokens estimates (len//4), not tokenizer-exact counts.
- coordination_quality and governance_correctness are REAL exact-scored, deterministic aggregates of CoordinationBench signals.

## Reproduction

```bash
env PYTHONPATH=src EMBEDDING_ENABLED=true EMBEDDING_PROVIDER=hash EMBEDDING_DIMENSION=1536 \
  python -c 'from pathlib import Path; from zaxy_benchmarks.fleet_benchmark import run_fleet_benchmark; run_fleet_benchmark(Path("reports/benchmarks/fleet-transfer-v1"))'
```
