# CoordinationBench

- version: `coordination-real-v1`
- workload: `b63e156150b92c9aa7d8895604741484f85f4c231d4c7d2097a2467fdaf14bd0`

| Metric | Value |
|--------|-------|
| accepted_finding_precision | 1.0 |
| accepted_finding_recall | 1.0 |
| conflict_precision | 1.0 |
| conflict_recall | 1.0 |
| stale_claim_rejection | 1.0 |
| duplicate_consolidation | 1.0 |
| evidence_coverage | 1.0 |
| parent_checkout_answerability | 1.0 |
| citation_coverage | 1.0 |
| eventloom_replayable | True |
| returned_tokens | 24057 |
| injected_tokens | 24057 |
| brief_latency_ms | 4.017 |
| promotion_latency_ms | 125.48 |
| accepted_state_synthesis_quality | 1.0 |
| non_authoritative_leakage | 1.0 |
| purpose_feedback_coverage | 1.0 |

## Coordinate Purpose/Synthesis Gate

- status: `passed`
- message: Coordinate accepted-state synthesis is proof-backed with citations, Coordinate-purpose feedback, replayable Eventloom provenance, parent checkout answerability, and no non-authoritative worker-row leakage.

| Required metric | Required value |
|-----------------|----------------|
| accepted_state_synthesis_quality | 1.0 |
| non_authoritative_leakage | 1.0 |
| purpose_feedback_coverage | 1.0 |
| citation_coverage | 1.0 |
| parent_checkout_answerability | 1.0 |
| eventloom_replayable | True |

## Competitor Claim Gate

- status: `blocked`
- required adapters: `quarq, hybi`
- completed adapters: `none`
- message: Public same-harness competitor claims are blocked until required adapters have completed, locally scored, fingerprinted results.

| Adapter | Blocker |
|---------|---------|
| Semantic Reach / HyperBinder / Hybi | adapter status is not_run/disclosure_only; same-harness public claims require completed locally scored results |
| Quarq | adapter status is not_run/disclosure_only; same-harness public claims require completed locally scored results |

## Baselines

| Baseline | Description | accepted_finding_precision | conflict_recall | stale_claim_rejection | duplicate_consolidation | parent_checkout_answerability | citation_coverage | injected_tokens | returned_tokens |
|----------|-------------|----------------------------|-----------------|-----------------------|-------------------------|-------------------------------|-------------------|-----------------|-----------------|
| bm25_worker_logs | Rank rendered worker findings with local BM25 and score the retrieved worker context. | 0.666667 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 481 | 481 |
| flat_transcript | Concatenate every worker finding and treat the combined transcript as accepted state. | 0.272727 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1265 | 1265 |
| markdown_notes | Render worker findings as shared markdown notes without promotion or conflict semantics. | 0.636364 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1603 | 1603 |

## Competitor Adapter Disclosures

| Adapter | Contract | Status | Claim status | Blockers |
|---------|----------|--------|--------------|----------|
| ActiveGraph | coordinationbench-v1 | not_run | disclosure_only | No pinned adapter package/version and same-harness workload replay contract has been configured. |
| Agent Memory | coordinationbench-v1 | not_run | disclosure_only | No pinned adapter package/version and same-harness workload replay contract has been configured. |
| Semantic Reach / HyperBinder / Hybi | coordinationbench-v1 | not_run | disclosure_only | Pinned hybi SDK metadata exists, but no completed same-harness HyperBinder runtime result has been locally scored. |
| Mem0 | coordinationbench-v1 | not_run | disclosure_only | No pinned adapter package/version and same-harness workload replay contract has been configured. |
| Quarq | coordinationbench-v1 | not_run | disclosure_only | Pinned Quarq source metadata exists, but no completed same-harness CoordinationBench result has been locally scored. |

## Limitations

CoordinationBench is a coordination-specific benchmark, not a universal memory benchmark.
It measures accepted-state precision, conflict handling, stale-claim rejection, evidence grounding, parent checkout answerability, and replayability for multi-agent coordination workflows.
The report should not be used as a claim about generic document RAG, open-domain QA, or all memory systems.
Competitor rows marked `disclosure_only` or `not_run` are adapter-status disclosures, not scores.

## Reproduction

Regenerate this report with the CoordinationBench CLI:

```bash
zaxy coordinate benchmark --output-dir reports/benchmarks/coordination-real-v1 --workload reports/benchmarks/coordination-real-v1/coordination-workload.json --json
```

For generated seed workloads, omit `--workload` and pass `--missions 1 --workers 3`.
For external systems, replace disclosure-only templates with pinned runner manifests or pinned result files.
