# LongMemBench External Run Plan

- Source: https://github.com/xiaowu0162/LongMemEval
- Dataset: `data/longmemeval_oracle.json`
- Evaluator model: `gpt-4o`
- Expected questions: `500`

## Commands

### bootstrap

```bash
zaxy longmembench-bootstrap --worktree path/to/LongMemEval
```

### doctor

```bash
zaxy longmembench-doctor path/to/LongMemEval
```

### ready_before_run

```bash
zaxy longmembench-ready --longmemeval-worktree path/to/LongMemEval --dataset path/to/LongMemEval/data/longmemeval_oracle.json --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json --answer-mode openai-compatible
```

### diagnostic

```bash
zaxy benchmark --output-dir reports/benchmarks/longmembench-external/diagnostic --embedding-provider hash --workload longmemeval --dataset path/to/LongMemEval/data/longmemeval_oracle.json --questions 500 --runs 1 --limit 10 --baseline-backends bm25 --zaxy-backend checkout --embedding-cache .cache/zaxy/longmemeval-embeddings.json
```

### generate_hypotheses

```bash
zaxy longmembench-generate-hypotheses --dataset path/to/LongMemEval/data/longmemeval_oracle.json --output reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl --report reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json --questions 500 --answer-mode openai-compatible --model gpt-4o --embedding-provider hash --embedding-cache .cache/zaxy/longmemeval-embeddings.json
```

### official_eval

```bash
zaxy longmembench-evaluate-official --longmemeval-worktree path/to/LongMemEval --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl --dataset path/to/LongMemEval/data/longmemeval_oracle.json --evaluator-model gpt-4o --output-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o --run-report reports/benchmarks/longmembench-external/official-eval-run.json
```

### official_metrics

```bash
python3 path/to/LongMemEval/src/evaluation/print_qa_metrics.py reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o path/to/LongMemEval/data/longmemeval_oracle.json
```

### validator_evidence

```bash
zaxy longmembench-validator-evidence --longmemeval-worktree path/to/LongMemEval --dataset path/to/LongMemEval/data/longmemeval_oracle.json --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl --official-eval-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o --output reports/benchmarks/longmembench-external/validator-evidence.json --evaluator-model gpt-4o --official-eval-command ZAXY_OFFICIAL_EVAL_COMMAND --print-metrics-command ZAXY_PRINT_METRICS_COMMAND --validator-name "Independent Validator" --validator-evidence-url https://validation.openmemory.dev/reviewable-run --validator-run-id validator-run-001 --validator-relation independent-third-party
```

### import

```bash
zaxy longmembench-import --longmemeval-worktree path/to/LongMemEval --dataset path/to/LongMemEval/data/longmemeval_oracle.json --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl --official-eval-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o --diagnostic-report reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json --output-dir reports/benchmarks/longmembench-external
```

### gate

```bash
zaxy longmembench-gate reports/benchmarks/longmembench-external/longmembench-report.json --require-official-sota
```

### audit

```bash
zaxy longmembench-audit --longmemeval-worktree path/to/LongMemEval --dataset path/to/LongMemEval/data/longmemeval_oracle.json --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl --official-eval-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o --diagnostic-report reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json --report reports/benchmarks/longmembench-external/longmembench-report.json --hypothesis-report reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json --official-eval-run-report reports/benchmarks/longmembench-external/official-eval-run.json --output reports/benchmarks/longmembench-external/longmembench-audit.json
```

### publish

```bash
zaxy longmembench-publish reports/benchmarks/longmembench-external/longmembench-report.json --audit reports/benchmarks/longmembench-external/longmembench-audit.json --output reports/benchmarks/longmembench-external/publishable-statistics.md
```
