#!/usr/bin/env bash
# Run live retrieval benchmarks for md/BM25/vector/md+vector/Zaxy.

set -euo pipefail

ROOT="$(pwd)"
OUTPUT_DIR="reports/benchmarks"
RUNS="5"
LIMIT="10"
EMBEDDING_PROVIDER="hash"
PROJECTION_BACKEND="${PROJECTION_BACKEND:-embedded}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7688}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-testpassword}"
PGGRAPH_DSN="${PGGRAPH_DSN:-}"
RESET_GRAPH="false"
WORKLOAD="fixture"
SUBJECTS="100"
DOCUMENTS="250"
SESSIONS="50"
DATASET=""
QUESTIONS=""
EMBEDDING_CACHE=""
EXTERNAL_RESULTS=""
BASELINE_BACKENDS=""
ZAXY_BACKEND=""
PROGRESS="false"
DRY_RUN="false"
REUSE_PROJECTION="false"

validate_positive_integer() {
    local flag="$1"
    local value="$2"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid ${flag}: ${value}; expected a positive integer" >&2
        exit 2
    fi
}

require_value() {
    local flag="$1"
    local value="${2-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "Missing value for ${flag}" >&2
        exit 2
    fi
}

usage() {
    cat <<USAGE
Usage: scripts/live-benchmark.sh [--root PATH] [--output-dir PATH] [--runs N] [--limit N] [--embedding-provider openai|hash|local-http|sentence-transformers] [--embedding-cache PATH] [--external-results PATH] [--baseline-backends LIST] [--zaxy-backend graph|checkout|both] [--progress] [--dry-run] [--reuse-projection] [--projection-backend embedded|neo4j|pggraph|latticedb] [--neo4j-uri URI] [--neo4j-user USER] [--neo4j-password PASSWORD] [--pggraph-dsn DSN] [--workload fixture|statistical|frozen|suite|consolidation|context-collapse|graph-traversal|source-recall|temporal-recall|longmemeval] [--dataset PATH] [--questions N] [--subjects N] [--documents N] [--sessions N] [--reset-graph]

Runs zaxy benchmark against markdown, BM25, vector, markdown+vector, and live Zaxy retrieval.
Uses deterministic hash embeddings by default for offline reproducibility.
Uses embedded projection by default; set PROJECTION_BACKEND or --projection-backend to compare external backends.
OpenAI mode is opt-in with --embedding-provider openai and requires OPENAI_API_KEY or OPENAI_API_KEY_FILE.
Neo4j and pgGraph connection flags are passed only when those projection backends are selected.
Use --dry-run to print the exact zaxy benchmark command without executing it.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            require_value "$1" "${2-}"
            ROOT="$2"
            shift 2
            ;;
        --output-dir)
            require_value "$1" "${2-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --runs)
            require_value "$1" "${2-}"
            RUNS="$2"
            shift 2
            ;;
        --limit)
            require_value "$1" "${2-}"
            LIMIT="$2"
            shift 2
            ;;
        --embedding-provider)
            require_value "$1" "${2-}"
            EMBEDDING_PROVIDER="$2"
            shift 2
            ;;
        --embedding-cache)
            require_value "$1" "${2-}"
            EMBEDDING_CACHE="$2"
            shift 2
            ;;
        --external-results)
            require_value "$1" "${2-}"
            EXTERNAL_RESULTS="$2"
            shift 2
            ;;
        --baseline-backends)
            require_value "$1" "${2-}"
            BASELINE_BACKENDS="$2"
            shift 2
            ;;
        --zaxy-backend)
            require_value "$1" "${2-}"
            ZAXY_BACKEND="$2"
            shift 2
            ;;
        --progress)
            PROGRESS="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --reuse-projection)
            REUSE_PROJECTION="true"
            shift
            ;;
        --projection-backend)
            require_value "$1" "${2-}"
            PROJECTION_BACKEND="$2"
            shift 2
            ;;
        --pggraph-dsn)
            require_value "$1" "${2-}"
            PGGRAPH_DSN="$2"
            shift 2
            ;;
        --neo4j-uri)
            require_value "$1" "${2-}"
            NEO4J_URI="$2"
            shift 2
            ;;
        --neo4j-user)
            require_value "$1" "${2-}"
            NEO4J_USER="$2"
            shift 2
            ;;
        --neo4j-password)
            require_value "$1" "${2-}"
            NEO4J_PASSWORD="$2"
            shift 2
            ;;
        --workload)
            require_value "$1" "${2-}"
            WORKLOAD="$2"
            shift 2
            ;;
        --subjects)
            require_value "$1" "${2-}"
            SUBJECTS="$2"
            shift 2
            ;;
        --documents)
            require_value "$1" "${2-}"
            DOCUMENTS="$2"
            shift 2
            ;;
        --sessions)
            require_value "$1" "${2-}"
            SESSIONS="$2"
            shift 2
            ;;
        --dataset)
            require_value "$1" "${2-}"
            DATASET="$2"
            shift 2
            ;;
        --questions)
            require_value "$1" "${2-}"
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

case "${PROJECTION_BACKEND}" in
    embedded|neo4j|pggraph|latticedb)
        ;;
    *)
        echo "Invalid projection backend: ${PROJECTION_BACKEND}; expected one of: embedded, neo4j, pggraph, latticedb" >&2
        exit 2
        ;;
esac

case "${EMBEDDING_PROVIDER}" in
    openai|hash|local-http|sentence-transformers)
        ;;
    *)
        echo "Invalid embedding provider: ${EMBEDDING_PROVIDER}; expected one of: openai, hash, local-http, sentence-transformers" >&2
        exit 2
        ;;
esac

case "${WORKLOAD}" in
    fixture|statistical|frozen|suite|consolidation|context-collapse|graph-traversal|source-recall|temporal-recall|longmemeval)
        ;;
    *)
        echo "Invalid workload: ${WORKLOAD}; expected one of: fixture, statistical, frozen, suite, consolidation, context-collapse, graph-traversal, source-recall, temporal-recall, longmemeval" >&2
        exit 2
        ;;
esac

if [[ -n "${ZAXY_BACKEND}" ]]; then
    case "${ZAXY_BACKEND}" in
        graph|checkout|both)
            ;;
        *)
            echo "Invalid --zaxy-backend: ${ZAXY_BACKEND}; expected one of: graph, checkout, both" >&2
            exit 2
            ;;
    esac
fi

validate_positive_integer "--runs" "${RUNS}"
validate_positive_integer "--limit" "${LIMIT}"
validate_positive_integer "--subjects" "${SUBJECTS}"
validate_positive_integer "--documents" "${DOCUMENTS}"
validate_positive_integer "--sessions" "${SESSIONS}"
if [[ -n "${QUESTIONS}" ]]; then
    validate_positive_integer "--questions" "${QUESTIONS}"
fi

cd "${ROOT}"

args=(
    zaxy benchmark
    --output-dir "${OUTPUT_DIR}"
    --runs "${RUNS}"
    --limit "${LIMIT}"
    --embedding-provider "${EMBEDDING_PROVIDER}"
    --projection-backend "${PROJECTION_BACKEND}"
    --workload "${WORKLOAD}"
    --subjects "${SUBJECTS}"
    --documents "${DOCUMENTS}"
    --sessions "${SESSIONS}"
)

if [[ "${PROJECTION_BACKEND}" == "neo4j" ]]; then
    args+=(--neo4j-uri "${NEO4J_URI}")
    args+=(--neo4j-user "${NEO4J_USER}")
    args+=(--neo4j-password "${NEO4J_PASSWORD}")
fi
if [[ "${PROJECTION_BACKEND}" == "pggraph" && -n "${PGGRAPH_DSN}" ]]; then
    args+=(--pggraph-dsn "${PGGRAPH_DSN}")
fi
if [[ "${RESET_GRAPH}" == "true" ]]; then
    args+=(--reset-graph)
fi
if [[ -n "${DATASET}" ]]; then
    args+=(--dataset "${DATASET}")
fi
if [[ -n "${QUESTIONS}" ]]; then
    args+=(--questions "${QUESTIONS}")
fi
if [[ -n "${EMBEDDING_CACHE}" ]]; then
    args+=(--embedding-cache "${EMBEDDING_CACHE}")
fi
if [[ -n "${EXTERNAL_RESULTS}" ]]; then
    args+=(--external-results "${EXTERNAL_RESULTS}")
fi
if [[ -n "${BASELINE_BACKENDS}" ]]; then
    args+=(--baseline-backends "${BASELINE_BACKENDS}")
fi
if [[ -n "${ZAXY_BACKEND}" ]]; then
    args+=(--zaxy-backend "${ZAXY_BACKEND}")
fi
if [[ "${PROGRESS}" == "true" ]]; then
    args+=(--progress)
fi
if [[ "${REUSE_PROJECTION}" == "true" ]]; then
    args+=(--reuse-projection)
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    printf '%q' "${args[0]}"
    for arg in "${args[@]:1}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
    exit 0
fi

"${args[@]}"
