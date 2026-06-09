#!/usr/bin/env bash
set -euo pipefail

LONGMEMEVAL_WORKTREE=${LONGMEMEVAL_WORKTREE:-${1:-.cache/zaxy/benchmarks/LongMemEval}}
ANSWER_MODE=${ANSWER_MODE:-openai-compatible}
EVALUATOR_MODEL=${EVALUATOR_MODEL:-gpt-4o}
QUESTIONS=${QUESTIONS:-500}
RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL:-1}
RUN_DIAGNOSTIC=${RUN_DIAGNOSTIC:-${RUN_OFFICIAL_EVAL}}
VALIDATOR_NAME=${VALIDATOR_NAME:-}
VALIDATOR_EVIDENCE_URL=${VALIDATOR_EVIDENCE_URL:-}
VALIDATOR_RUN_ID=${VALIDATOR_RUN_ID:-}
VALIDATOR_RELATION=${VALIDATOR_RELATION:-}
if [[ -z "${RUN_OUTPUT_DIR:-}" ]]; then
  if [[ "${RUN_OFFICIAL_EVAL}" == "0" ]]; then
    RUN_OUTPUT_DIR=reports/benchmarks/longmembench-external/smoke
  else
    RUN_OUTPUT_DIR=reports/benchmarks/longmembench-external
  fi
fi
OFFICIAL_EVAL_COMMAND="python3 evaluate_qa.py ${EVALUATOR_MODEL} ${RUN_OUTPUT_DIR}/zaxy-hypotheses.jsonl ${LONGMEMEVAL_WORKTREE}/data/longmemeval_oracle.json"
PRINT_METRICS_COMMAND="python3 ${LONGMEMEVAL_WORKTREE}/src/evaluation/print_qa_metrics.py ${RUN_OUTPUT_DIR}/zaxy-hypotheses.jsonl.eval-results-${EVALUATOR_MODEL} ${LONGMEMEVAL_WORKTREE}/data/longmemeval_oracle.json"

if [[ "${RUN_OFFICIAL_EVAL}" != "0" && "${ANSWER_MODE}" == "openai-compatible" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo 'OPENAI_API_KEY is required for official openai-compatible LongMemBench runs.' >&2
  exit 2
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  if [[ -z "${VALIDATOR_NAME}" || -z "${VALIDATOR_EVIDENCE_URL}" || -z "${VALIDATOR_RUN_ID}" || -z "${VALIDATOR_RELATION}" ]]; then
    echo 'VALIDATOR_NAME, VALIDATOR_EVIDENCE_URL, VALIDATOR_RUN_ID, and VALIDATOR_RELATION are required for official SOTA runs.' >&2
    exit 2
  fi
fi

run_step() {
  local name=$1
  local command=$2
  echo "[$name] $command"
  eval "$command"
}

run_step bootstrap 'zaxy longmembench-bootstrap --worktree "${LONGMEMEVAL_WORKTREE}"'
run_step doctor 'zaxy longmembench-doctor "${LONGMEMEVAL_WORKTREE}"'
run_step ready_before_run 'zaxy longmembench-ready --longmemeval-worktree "${LONGMEMEVAL_WORKTREE}" --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json --answer-mode "${ANSWER_MODE}"'
if [[ "${RUN_DIAGNOSTIC}" != "0" ]]; then
  run_step diagnostic 'zaxy benchmark --output-dir "${RUN_OUTPUT_DIR}"/diagnostic --embedding-provider hash --workload longmemeval --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --questions "${QUESTIONS}" --runs 1 --limit 10 --baseline-backends bm25 --zaxy-backend checkout --embedding-cache .cache/zaxy/longmemeval-embeddings.json'
else
  echo '[diagnostic] skipped because RUN_DIAGNOSTIC=0'
fi
run_step generate_hypotheses 'zaxy longmembench-generate-hypotheses --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --output "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl --report "${RUN_OUTPUT_DIR}"/zaxy-hypotheses-report.json --questions "${QUESTIONS}" --answer-mode "${ANSWER_MODE}" --model "${EVALUATOR_MODEL}" --embedding-provider hash --embedding-cache .cache/zaxy/longmemeval-embeddings.json'
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step official_eval 'zaxy longmembench-evaluate-official --longmemeval-worktree "${LONGMEMEVAL_WORKTREE}" --hypotheses "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --evaluator-model "${EVALUATOR_MODEL}" --output-log "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl.eval-results-"${EVALUATOR_MODEL}" --run-report "${RUN_OUTPUT_DIR}"/official-eval-run.json'
else
  echo '[official_eval] skipped because RUN_OFFICIAL_EVAL=0'
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step official_metrics 'python3 "${LONGMEMEVAL_WORKTREE}"/src/evaluation/print_qa_metrics.py "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl.eval-results-"${EVALUATOR_MODEL}" "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json'
else
  echo '[official_metrics] skipped because RUN_OFFICIAL_EVAL=0'
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step validator_evidence 'zaxy longmembench-validator-evidence --longmemeval-worktree "${LONGMEMEVAL_WORKTREE}" --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --hypotheses "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl --official-eval-log "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl.eval-results-"${EVALUATOR_MODEL}" --output "${RUN_OUTPUT_DIR}"/validator-evidence.json --evaluator-model "${EVALUATOR_MODEL}" --official-eval-command "${OFFICIAL_EVAL_COMMAND}" --print-metrics-command "${PRINT_METRICS_COMMAND}" --validator-name "${VALIDATOR_NAME}" --validator-evidence-url "${VALIDATOR_EVIDENCE_URL}" --validator-run-id "${VALIDATOR_RUN_ID}" --validator-relation "${VALIDATOR_RELATION}"'
else
  echo '[validator_evidence] skipped because RUN_OFFICIAL_EVAL=0'
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step import 'zaxy longmembench-import --longmemeval-worktree "${LONGMEMEVAL_WORKTREE}" --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --hypotheses "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl --official-eval-log "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl.eval-results-"${EVALUATOR_MODEL}" --diagnostic-report "${RUN_OUTPUT_DIR}"/diagnostic/live-benchmark.json --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json --validator-evidence "${RUN_OUTPUT_DIR}"/validator-evidence.json --output-dir "${RUN_OUTPUT_DIR}"'
else
  echo '[import] skipped because RUN_OFFICIAL_EVAL=0'
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step gate 'zaxy longmembench-gate "${RUN_OUTPUT_DIR}"/longmembench-report.json --require-official-sota'
else
  echo '[gate] skipped because RUN_OFFICIAL_EVAL=0'
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step audit 'zaxy longmembench-audit --longmemeval-worktree "${LONGMEMEVAL_WORKTREE}" --dataset "${LONGMEMEVAL_WORKTREE}"/data/longmemeval_oracle.json --hypotheses "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl --official-eval-log "${RUN_OUTPUT_DIR}"/zaxy-hypotheses.jsonl.eval-results-"${EVALUATOR_MODEL}" --diagnostic-report "${RUN_OUTPUT_DIR}"/diagnostic/live-benchmark.json --sota-baseline reports/benchmarks/longmembench-external/sota-baseline.json --validator-evidence "${RUN_OUTPUT_DIR}"/validator-evidence.json --report "${RUN_OUTPUT_DIR}"/longmembench-report.json --hypothesis-report "${RUN_OUTPUT_DIR}"/zaxy-hypotheses-report.json --official-eval-run-report "${RUN_OUTPUT_DIR}"/official-eval-run.json --output "${RUN_OUTPUT_DIR}"/longmembench-audit.json'
else
  echo '[audit] skipped because RUN_OFFICIAL_EVAL=0'
fi
if [[ "${RUN_OFFICIAL_EVAL}" != "0" ]]; then
  run_step publish 'zaxy longmembench-publish "${RUN_OUTPUT_DIR}"/longmembench-report.json --audit "${RUN_OUTPUT_DIR}"/longmembench-audit.json --output "${RUN_OUTPUT_DIR}"/publishable-statistics.md'
else
  echo '[publish] skipped because RUN_OFFICIAL_EVAL=0'
fi
