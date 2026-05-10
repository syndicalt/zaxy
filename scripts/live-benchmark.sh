#!/usr/bin/env bash
# Run live retrieval benchmarks for md/BM25/vector/md+vector/Zaxy.

set -euo pipefail

ROOT="$(pwd)"
OUTPUT_DIR="reports/benchmarks"
RUNS="5"
LIMIT="10"
EMBEDDING_PROVIDER="openai"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7688}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-testpassword}"
RESET_GRAPH="false"
WORKLOAD="fixture"
SUBJECTS="100"
DOCUMENTS="250"
SESSIONS="50"
DATASET=""
QUESTIONS=""

usage() {
    cat <<USAGE
Usage: scripts/live-benchmark.sh [--root PATH] [--output-dir PATH] [--runs N] [--limit N] [--embedding-provider openai|hash] [--workload fixture|statistical|frozen|suite|consolidation|longmemeval] [--dataset PATH] [--questions N] [--subjects N] [--documents N] [--sessions N] [--reset-graph]

Runs zaxy benchmark against markdown, BM25, vector, markdown+vector, and live Zaxy retrieval.
OpenAI mode requires OPENAI_API_KEY or OPENAI_API_KEY_FILE.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --embedding-provider)
            EMBEDDING_PROVIDER="$2"
            shift 2
            ;;
        --workload)
            WORKLOAD="$2"
            shift 2
            ;;
        --subjects)
            SUBJECTS="$2"
            shift 2
            ;;
        --documents)
            DOCUMENTS="$2"
            shift 2
            ;;
        --sessions)
            SESSIONS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --questions)
            QUESTIONS="$2"
            shift 2
            ;;
        --reset-graph)
            RESET_GRAPH="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "${ROOT}"

args=(
    zaxy benchmark
    --output-dir "${OUTPUT_DIR}"
    --runs "${RUNS}"
    --limit "${LIMIT}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --workload "${WORKLOAD}"
    --subjects "${SUBJECTS}"
    --documents "${DOCUMENTS}"
    --sessions "${SESSIONS}"
    --neo4j-uri "${NEO4J_URI}"
    --neo4j-user "${NEO4J_USER}"
    --neo4j-password "${NEO4J_PASSWORD}"
)

if [[ "${RESET_GRAPH}" == "true" ]]; then
    args+=(--reset-graph)
fi
if [[ -n "${DATASET}" ]]; then
    args+=(--dataset "${DATASET}")
fi
if [[ -n "${QUESTIONS}" ]]; then
    args+=(--questions "${QUESTIONS}")
fi

"${args[@]}"
