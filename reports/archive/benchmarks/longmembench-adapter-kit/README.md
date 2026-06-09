# Zaxy LongMemBench Adapter Kit

This kit is for an external LongMemEval checkout. It keeps official QA
evaluation outside Zaxy while letting Zaxy publish an audited report that
separates official answer accuracy from Zaxy retrieval/checkout diagnostics.

Official LongMemEval testing requires:

1. Feed timestamped histories to the memory system.
2. Write a JSONL hypotheses file with one object per line:
   `{"question_id": "...", "hypothesis": "..."}`.
3. Run LongMemEval's official evaluator:

   ```bash
   cd path/to/LongMemEval/src/evaluation
   python3 evaluate_qa.py gpt-4o path/to/zaxy-hypotheses.jsonl ../../data/longmemeval_oracle.json
   python3 print_qa_metrics.py path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_oracle.json
   ```

Write the completed validator evidence record:

```bash
zaxy longmembench-validator-evidence \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --hypotheses path/to/zaxy-hypotheses.jsonl \
  --official-eval-log path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --output reports/benchmarks/longmembench-external/validator-evidence.json \
  --evaluator-model gpt-4o \
  --official-eval-command "python3 evaluate_qa.py gpt-4o path/to/zaxy-hypotheses.jsonl ../../data/longmemeval_oracle.json" \
  --print-metrics-command "python3 print_qa_metrics.py path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_oracle.json" \
  --validator-name "Independent Validator" \
  --validator-evidence-url https://validation.openmemory.dev/reviewable-run \
  --validator-run-id validator-run-001 \
  --validator-relation independent-third-party
```

Then import the official evaluator log into Zaxy:

```bash
zaxy longmembench-import \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --hypotheses path/to/zaxy-hypotheses.jsonl \
  --official-eval-log path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --diagnostic-report reports/benchmarks/longmemeval-500-current74-zaxyonly-gated-relative-temporal-anchor-embedded-reuse-20260604/live-benchmark.json \
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \
  --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json \
  --output-dir reports/benchmarks/longmembench-external

zaxy longmembench-validate reports/benchmarks/longmembench-external/longmembench-report.json --require-official-full
zaxy longmembench-gate reports/benchmarks/longmembench-external/longmembench-report.json --require-official-sota
zaxy longmembench-audit \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --hypotheses path/to/zaxy-hypotheses.jsonl \
  --official-eval-log path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --diagnostic-report reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json \
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \
  --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json \
  --report reports/benchmarks/longmembench-external/longmembench-report.json \
  --hypothesis-report reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json \
  --official-eval-run-report reports/benchmarks/longmembench-external/official-eval-run.json \
  --output reports/benchmarks/longmembench-external/longmembench-audit.json
```

For CI systems that cannot pass a completed validator JSON file, the equivalent
manual fields are `--validator-name`, `--validator-evidence-url`,
`--validator-run-id`, and `--validator-relation`.

Do not claim official LongMemEval SOTA from Zaxy retrieval diagnostics alone.
The SOTA gate requires official evaluator evidence over the full 500-question
dataset.
