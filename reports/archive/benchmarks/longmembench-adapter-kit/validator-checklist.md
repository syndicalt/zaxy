# LongMemBench Validator Checklist

Use this checklist when producing independent evidence for a Zaxy official
LongMemEval SOTA claim.

## Required Inputs

- Official LongMemEval checkout URL and commit.
- Official `longmemeval_oracle.json` dataset with 500 questions.
- Zaxy source checkout commit under validation.
- `reports/benchmarks/longmembench-external/sota-baseline.json` with a
  `checked_at` date no older than 30 days for strict SOTA gating.
- Evaluator credentials for LongMemEval's official `evaluate_qa.py`.

## Required Run Steps

1. Run `zaxy longmembench-bootstrap --worktree <LongMemEval>`.
2. Run `zaxy longmembench-doctor <LongMemEval>` and record the commit.
3. Generate 500 Zaxy hypotheses in openai-compatible mode.
4. Run LongMemEval's official evaluator over the generated hypotheses.
5. Run `print_qa_metrics.py` on the evaluator result.
6. Run `zaxy longmembench-validator-evidence` to complete
   `validator-evidence.json`, then import the result with `--validator-evidence`.
7. Run `zaxy longmembench-gate <report> --require-official-sota`.
8. Run `zaxy longmembench-audit ...` over the complete artifact set.

## Required Evidence Artifacts

- `zaxy-hypotheses.jsonl` with exactly 500 rows.
- `zaxy-hypotheses-report.json`.
- `zaxy-hypotheses.jsonl.eval-results-<model>` with exactly 500 rows.
- `official-eval-run.json`.
- `longmembench-report.json`.
- `longmembench-report.md`.
- `longmembench-audit.json` with SHA-256 hashes for the complete artifact set.
- Current SOTA baseline JSON with official QA metric, full-set question count,
  reviewable evidence URL, and fresh `checked_at`.
- Terminal transcript or CI log showing the gate command and result.
- Terminal transcript or CI log showing the audit command and result.
- Completed `validator-evidence-template.json`.

The completed validator evidence must match the imported report: validated
system name, Zaxy commit, LongMemEval commit, dataset SHA-256, question count,
hypotheses SHA-256, official evaluator log SHA-256, evaluator model, evaluated
count, correct count, accuracy, and official evaluator command are checked
during `longmembench-import`.

Manual validator fields alone cannot pass `--require-official-sota`; the strict
gate requires a cross-checked `validator-evidence.json` import bound to a Zaxy
commit.

## Non-Negotiable Claim Boundary

Do not treat Zaxy retrieval diagnostics, smoke runs, or internally generated
partial reports as official SOTA evidence. The publishable claim requires full
official QA evaluator evidence and independent validator provenance.
