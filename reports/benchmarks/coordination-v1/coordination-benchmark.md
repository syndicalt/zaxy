# CoordinationBench

- version: `coordination-v1`
- workload: `4b6f01f5a0e9275bd6cd0238d439ee326d471483d5da3cc1dcc9a258d21bfafc`

| Metric | Value |
|--------|-------|
| accepted_finding_precision | 1.0 |
| accepted_finding_recall | 1.0 |
| conflict_precision | 0.166667 |
| conflict_recall | 1.0 |
| stale_claim_rejection | 1.0 |
| duplicate_consolidation | 1.0 |
| evidence_coverage | 1.0 |
| parent_checkout_answerability | 1.0 |
| citation_coverage | 1.0 |
| eventloom_replayable | True |
| returned_tokens | 4507 |
| injected_tokens | 4507 |
| brief_latency_ms | 0.789 |
| promotion_latency_ms | 19.437 |

## Baselines

| Baseline | Description | accepted_finding_precision | conflict_recall | stale_claim_rejection | duplicate_consolidation | parent_checkout_answerability | citation_coverage | injected_tokens | returned_tokens |
|----------|-------------|----------------------------|-----------------|-----------------------|-------------------------|-------------------------------|-------------------|-----------------|-----------------|
| bm25_worker_logs | Rank rendered worker findings with local BM25 and score the retrieved worker context. | 0.333333 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 244 | 244 |
| flat_transcript | Concatenate every worker finding and treat the combined transcript as accepted state. | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 334 | 334 |
| markdown_notes | Render worker findings as shared markdown notes without promotion or conflict semantics. | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 377 | 377 |

## Competitor Adapter Disclosures

| Adapter | Contract | Status | Claim status | Blockers |
|---------|----------|--------|--------------|----------|
| ActiveGraph | coordinationbench-v1 | not_run | disclosure_only | No pinned adapter package/version and same-harness workload replay contract has been configured. |
| Agent Memory | coordinationbench-v1 | not_run | disclosure_only | No pinned adapter package/version and same-harness workload replay contract has been configured. |
| Mem0 | coordinationbench-v1 | not_run | disclosure_only | No pinned adapter package/version and same-harness workload replay contract has been configured. |

## Limitations

CoordinationBench is a coordination-specific benchmark, not a universal memory benchmark.
It measures accepted-state precision, conflict handling, stale-claim rejection, evidence grounding, parent checkout answerability, and replayability for multi-agent coordination workflows.
The report should not be used as a claim about generic document RAG, open-domain QA, or all memory systems.
Competitor rows marked `disclosure_only` or `not_run` are adapter-status disclosures, not scores.

## Reproduction

Regenerate this report with the CoordinationBench CLI:

```bash
zaxy coordinate benchmark --output-dir reports/benchmarks/coordination-v1 --workload reports/benchmarks/coordination-v1/coordination-workload.json --json
```

For generated seed workloads, omit `--workload` and pass `--missions 1 --workers 3`.
For external systems, replace disclosure-only templates with pinned runner manifests or pinned result files.
