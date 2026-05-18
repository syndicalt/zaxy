#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZAXY_BIN="${ZAXY_BIN:-zaxy}"
ZAXY_CMD=(${ZAXY_BIN})
DATASET="${ZAXY_LONGMEMEVAL_DATASET:-.cache/zaxy/benchmarks/longmemeval_oracle.json}"
EMBEDDING_CACHE="${ZAXY_LONGMEMEVAL_EMBEDDING_CACHE:-.cache/zaxy/longmemeval-embeddings.json}"

cd "${ROOT}"

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing required benchmark artifact: ${path}" >&2
        exit 1
    fi
}

require_file "${DATASET}"
require_file "${EMBEDDING_CACHE}"
require_file "reports/benchmarks/longmemeval-500-hash/live-benchmark.json"
require_file "reports/benchmarks/longmemeval-500-neo4j-current-checkout/live-benchmark.json"
require_file "reports/benchmarks/longmemeval-500-pggraph-comparison/live-benchmark.json"

"${ZAXY_CMD[@]}" benchmark-inventory --json >/dev/null

"${ZAXY_CMD[@]}" benchmark-compare reports/benchmarks/longmemeval-500-hash/live-benchmark.json \
    --backend zaxy-checkout \
    --min-mean-score 0.626 \
    --min-answer-recall-at-5 0.608 \
    --min-recall-at-5 0.956 \
    --min-citation-coverage 1.0 \
    --max-p95-ms 15000 \
    --max-p99-ms 23000

"${ZAXY_CMD[@]}" benchmark-compare reports/benchmarks/longmemeval-500-neo4j-current-checkout/live-benchmark.json \
    --backend zaxy-checkout \
    --min-mean-score 0.714 \
    --min-answer-recall-at-5 0.626 \
    --min-recall-at-5 0.958 \
    --min-citation-coverage 1.0 \
    --max-p95-ms 1200 \
    --max-p99-ms 2500

"${ZAXY_CMD[@]}" benchmark-compare reports/benchmarks/longmemeval-500-pggraph-comparison/live-benchmark.json \
    --backend zaxy-checkout \
    --min-mean-score 0.714 \
    --min-answer-recall-at-5 0.626 \
    --min-recall-at-5 0.958 \
    --min-citation-coverage 1.0 \
    --max-p95-ms 1200 \
    --max-p99-ms 3000

"${ZAXY_CMD[@]}" benchmark-compare reports/benchmarks/longmemeval-500-pggraph-comparison/live-benchmark.json \
    --backend zaxy \
    --min-mean-score 0.626 \
    --min-answer-recall-at-5 0.608 \
    --min-recall-at-5 0.956 \
    --min-citation-coverage 1.0 \
    --max-p95-ms 15000 \
    --max-p99-ms 23000

echo "Benchmark guardrails passed."
