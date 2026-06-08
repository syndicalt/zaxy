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
require_file "reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json"
require_file "reports/benchmarks/longmemeval-500-publish-20260607/run-config.md"

"${ZAXY_CMD[@]}" benchmark-inventory --json >/dev/null
"${ZAXY_CMD[@]}" benchmark-freeze --json >/dev/null

"${ZAXY_CMD[@]}" benchmark-compare reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json \
    --backend zaxy-checkout \
    --min-mean-score 0.95 \
    --min-answer-recall-at-5 0.90 \
    --min-recall-at-5 0.99 \
    --min-citation-coverage 1.0 \
    --max-p95-ms 2500 \
    --max-p99-ms 3000

echo "Benchmark guardrails passed."
