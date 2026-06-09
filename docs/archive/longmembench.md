# LongMemBench External Validation

LongMemBench is Zaxy's external validation lane for official LongMemEval-style
testing. It exists because Zaxy's internal LongMemEval-compatible checkout
reports are useful but do not, by themselves, prove an official LongMemEval
end-to-end assistant score.

## Claim Levels

Use precise language:

- **LongMemEval-compatible checkout:** Zaxy's adapted Eventloom checkout run.
  Metrics include Recall@k, Answer@k, citation coverage, latency, and token
  estimates.
- **Official LongMemEval QA:** hypotheses are evaluated by the official
  LongMemEval `evaluate_qa.py` script.
- **Official LongMemEval SOTA candidate:** official QA evidence covers all 500
  questions and is ready for comparison against the current accepted external
  leaderboard or maintainer-recognized evidence.

Do not claim official SOTA from retrieval diagnostics alone.

## External Run Flow

Prepare an external LongMemEval checkout:

```bash
zaxy longmembench-bootstrap --worktree .cache/zaxy/benchmarks/LongMemEval
```

From the Zaxy checkout, generate the validation kit and manifest:

```bash
zaxy longmembench-doctor path/to/LongMemEval
zaxy longmembench-adapter-kit --output-dir reports/benchmarks/longmembench-adapter-kit
zaxy longmembench-plan --output-dir reports/benchmarks/longmembench-external
zaxy longmembench-ready \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \
  --answer-mode openai-compatible
```

The adapter kit includes `validator-checklist.md` and
`validator-evidence-template.json`. The checklist defines the minimum external
artifacts required for an independent SOTA claim; the template maps directly to
the validator fields imported by `longmembench-import`.

Use the generated runner for repeatable execution. Smoke mode validates
bootstrap, readiness, diagnostic retrieval, and hypothesis generation without
importing or gating official evidence:

```bash
ANSWER_MODE=extractive QUESTIONS=1 RUN_OFFICIAL_EVAL=0 RUN_DIAGNOSTIC=0 \
  reports/benchmarks/longmembench-external/run-longmembench-zaxy.sh path/to/LongMemEval
```

Smoke outputs are written under
`reports/benchmarks/longmembench-external/smoke/` by default so they cannot
overwrite official 500-question artifacts.

Full official mode requires evaluator credentials and runs through import plus
the strict SOTA gate:

```bash
OPENAI_API_KEY=... \
VALIDATOR_NAME="Independent Validator" \
VALIDATOR_EVIDENCE_URL=https://validation.openmemory.dev/reviewable-run \
VALIDATOR_RUN_ID=validator-run-001 \
VALIDATOR_RELATION=independent-third-party \
  reports/benchmarks/longmembench-external/run-longmembench-zaxy.sh path/to/LongMemEval
```

Run the Zaxy diagnostic benchmark as supporting evidence:

```bash
zaxy benchmark \
  --output-dir reports/benchmarks/longmembench-external/diagnostic \
  --embedding-provider hash \
  --workload longmemeval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --questions 500 \
  --runs 1 \
  --limit 10 \
  --baseline-backends bm25 \
  --zaxy-backend checkout \
  --embedding-cache .cache/zaxy/longmemeval-embeddings.json
```

Generate official hypotheses with Zaxy checkout evidence. For a SOTA-candidate
run, use model-backed answer generation:

```bash
zaxy longmembench-generate-hypotheses \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --output reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl \
  --report reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json \
  --answer-mode openai-compatible \
  --model gpt-4o \
  --embedding-provider hash \
  --embedding-cache .cache/zaxy/longmemeval-embeddings.json
```

For local smoke tests, use `--answer-mode extractive`; that mode proves the
artifact contract but should not be used for an official SOTA submission unless
the resulting evaluator score is competitive. The official output is JSONL:

```json
{"question_id": "gpt4_2655b836", "hypothesis": "The GPS system was not functioning correctly."}
```

Evaluate with LongMemEval's official script:

```bash
export OPENAI_API_KEY=...
zaxy longmembench-evaluate-official \
  --longmemeval-worktree path/to/LongMemEval \
  --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --evaluator-model gpt-4o \
  --output-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --run-report reports/benchmarks/longmembench-external/official-eval-run.json

cd path/to/LongMemEval/src/evaluation
python3 print_qa_metrics.py /absolute/path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_oracle.json
```

Write the completed validator evidence record from the official artifacts:

```bash
zaxy longmembench-validator-evidence \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl \
  --official-eval-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --output reports/benchmarks/longmembench-external/validator-evidence.json \
  --evaluator-model gpt-4o \
  --official-eval-command "python3 evaluate_qa.py gpt-4o /absolute/path/to/zaxy-hypotheses.jsonl ../../data/longmemeval_oracle.json" \
  --print-metrics-command "python3 print_qa_metrics.py /absolute/path/to/zaxy-hypotheses.jsonl.eval-results-gpt-4o ../../data/longmemeval_oracle.json" \
  --validator-name "Independent Validator" \
  --validator-evidence-url https://validation.openmemory.dev/reviewable-run \
  --validator-run-id validator-run-001 \
  --validator-relation independent-third-party
```

Import and gate the evidence:

```bash
zaxy longmembench-import \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl \
  --official-eval-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --diagnostic-report reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json \
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \
  --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json \
  --output-dir reports/benchmarks/longmembench-external

zaxy longmembench-validate \
  reports/benchmarks/longmembench-external/longmembench-report.json \
  --require-official-full

zaxy longmembench-gate \
  reports/benchmarks/longmembench-external/longmembench-report.json \
  --require-official-sota

zaxy longmembench-audit \
  --longmemeval-worktree path/to/LongMemEval \
  --dataset path/to/LongMemEval/data/longmemeval_oracle.json \
  --hypotheses reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl \
  --official-eval-log reports/benchmarks/longmembench-external/zaxy-hypotheses.jsonl.eval-results-gpt-4o \
  --diagnostic-report reports/benchmarks/longmembench-external/diagnostic/live-benchmark.json \
  --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json \
  --validator-evidence reports/benchmarks/longmembench-external/validator-evidence.json \
  --report reports/benchmarks/longmembench-external/longmembench-report.json \
  --hypothesis-report reports/benchmarks/longmembench-external/zaxy-hypotheses-report.json \
  --official-eval-run-report reports/benchmarks/longmembench-external/official-eval-run.json \
  --output reports/benchmarks/longmembench-external/longmembench-audit.json
```

Only after this gate passes should a publishable report be rendered:

```bash
zaxy longmembench-publish \
  reports/benchmarks/longmembench-external/longmembench-report.json \
  --audit reports/benchmarks/longmembench-external/longmembench-audit.json \
  --output reports/benchmarks/longmembench-external/publishable-statistics.md
```

## Evidence Requirements

A valid official full-set report must include:

- official LongMemEval worktree commit;
- Zaxy source checkout commit under validation;
- official dataset path and SHA-256;
- 500 dataset questions;
- 500 generated hypotheses;
- 500 official evaluator log rows;
- SHA-256 hashes for the dataset, hypotheses, official evaluator log, and
  generated LongMemBench report artifacts;
- official evaluator model and command;
- correct count and accuracy;
- external SOTA baseline JSON using an official LongMemEval QA accuracy metric;
- external SOTA baseline `checked_at` freshness evidence, no older than 30 days
  unless the gate default is explicitly changed in code;
- independent validator provenance: name, reviewable evidence URL, run ID, and
  relation to Zaxy;
- passing `longmembench-audit` JSON output with schema version
  `zaxy.longmembench-audit.v1`, `generated_at`, and SHA-256 hashes for the
  complete artifact set;
- optional Zaxy diagnostic report with checkout/citation metrics.

The preferred provenance handoff is a completed
`validator-evidence-template.json` imported with `--validator-evidence`. The
manual `--validator-name`, `--validator-evidence-url`, `--validator-run-id`, and
`--validator-relation` flags remain available for CI systems that inject
provenance through environment variables. If both are supplied, nonblank
conflicts are rejected rather than silently rewritten.

When `--validator-evidence` is supplied, `longmembench-import` cross-checks the
completed evidence file against the imported official evaluator log. The
validated system name, Zaxy commit, LongMemEval commit, dataset SHA-256,
dataset question count, hypotheses SHA-256, official evaluator log SHA-256,
evaluator model, evaluated count, correct count, accuracy, and official
evaluator command must match the report built from the local artifacts.

The strict `--require-official-sota` gate requires that cross-check and a
validator-bound Zaxy commit. A report with only manual validator fields can be a
useful intermediate artifact, but it cannot pass the official SOTA gate.

The strict gate does not decide whether Zaxy is globally SOTA. It proves the
artifact is an official full-set LongMemEval candidate and, when
`--require-official-sota` is used, that it beats the imported external baseline.
The imported baseline must also be recently checked. The final public claim
should still cite the baseline source and any LongMemEval maintainer or
leaderboard process available at publication time.

## SOTA Baseline Schema

For a strict SOTA claim, write a baseline JSON file:

```json
{
  "system": "Current accepted best system",
  "accuracy": 0.966,
  "metric": "official_longmemeval_task_averaged_accuracy",
  "evidence_url": "https://validation.openmemory.dev/reviewable-artifact",
  "evidence_date": "2026-06-01",
  "checked_at": "2026-06-07",
  "currentness_url": "https://validation.openmemory.dev/leaderboard-or-review",
  "source_type": "public-reproduction",
  "question_count": 500,
  "evaluator_model": "gpt-4o",
  "notes": "Use official-leaderboard, maintainer-accepted, peer-reviewed-paper, public-reproduction, or vendor-disclosure."
}
```

`zaxy longmembench-gate --require-official-sota` fails if the official Zaxy
accuracy is less than or equal to this baseline. Retrieval metrics such as
Recall@5 are deliberately rejected for this gate; the baseline must be an
official LongMemEval QA accuracy metric over the full 500-question set.
Strict SOTA also fails when `checked_at` is missing, in the future, or older
than 30 days at gate time. `expires_at` is optional, but when present it must not
have passed.
Baseline and validator evidence URLs must be public, reviewable HTTP(S) URLs.
Reserved placeholder domains such as `example.com`, local URLs, and private IP
addresses are rejected by the strict evidence validators.

Strict official SOTA also requires external validator provenance in the imported
report. Local smoke runs and internally generated evaluator logs can validate
the pipeline, but they cannot pass `--require-official-sota` without a
reviewable independent validator record.
