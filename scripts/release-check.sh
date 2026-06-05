#!/usr/bin/env bash
# Run the Zaxy go-live release gate.

set -euo pipefail

ROOT="$(pwd)"
RUFF_CMD="ruff"
MYPY_CMD="mypy"
PYTEST_CMD="pytest"
COVERAGE_CMD="python scripts/check-coverage.py"
PACKET_SMOKE_CMD="pytest tests/test_packet_memory_e2e.py --no-cov -q"
EXAMPLES_SMOKE_CMD="pytest tests/test_examples_v05.py --no-cov -q"
MCP_SMOKE_CMD="python scripts/mcp_smoke_test.py"
LANGGRAPH_SMOKE_CMD="pytest tests/test_examples_v05.py::test_langgraph_example_runs_without_langgraph_dependency --no-cov -q"
COORDINATE_SMOKE_CMD="pytest tests/test_examples_v05.py::test_coordinate_three_worker_example_runs --no-cov -q"
PACKAGE_CMD="scripts/build-dist.sh"
DOCS_CMD="python scripts/build-site-docs.py --check && scripts/validate-docs.sh"
VALIDATE_CMD="scripts/validate-deployment.sh"
BETA_UAT_CMD="scripts/beta-uat.sh"
STATE_RECOVERY_CMD='python scripts/check-state-recovery-benchmark.py reports/benchmarks/state-recovery-v1/state-recovery-benchmark.json --workload reports/benchmarks/state-recovery-v1/state-recovery-workload.json --require-git-tracked-inputs && tmpdir=$(mktemp -d) && python -m zaxy state-recovery-benchmark --output-dir "${tmpdir}" --workload reports/benchmarks/state-recovery-v1/state-recovery-workload.json --json >/dev/null && python scripts/check-state-recovery-benchmark.py "${tmpdir}/state-recovery-benchmark.json" --workload "${tmpdir}/state-recovery-workload.json" && rm -rf "${tmpdir}"'
EXTERNAL_VALIDATION_REPORT=""
EXTERNAL_VALIDATION_CMD="SKIP:external validation is optional for v1.0 release"
REQUIRE_EXTERNAL_VALIDATION=0
HOOK_STATUS_CMD="PYTHONPATH=src python -m zaxy hook-status --eventloom-path reports/activation-release --now 2026-05-20T12:00:00+00:00 --min-activation-rate 1.0 --max-checkout-prompt-tokens 5000 --min-checkout-facts-per-1k-tokens 0.1"
BACKEND_SHOOTOUT_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/backend-shootout.json --require-report-metadata --require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints --require-backends embedded,bm25 --forbid-backends neo4j,pggraph,latticedb --require-labeled-metrics --require-dashboard-source embedded=embedded --min-answer-at-5 0.5 --min-recall-at-5 0.5 --min-citation-coverage 1.0 --min-quality-per-1k-injected-tokens embedded=1.0 --min-answer-at-5-per-1k-injected-tokens embedded=1.0 --max-cold-bootstrap-ms embedded=250 --max-first-checkout-ms embedded=25 --max-append-to-projection-p95-ms embedded=50 --max-resident-memory-delta-bytes embedded=256000000 --max-on-disk-footprint-bytes embedded=256000000 --max-dashboard-graph-load-ms embedded=250 --max-checkout-p99-ms embedded=25 --max-exact-p99-ms embedded=10 --max-keyword-p99-ms embedded=5 --max-vector-p99-ms embedded=5 --max-traversal-p99-ms embedded=5"
BACKEND_PERFORMANCE_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/longmemeval-40-backend-shootout.json --require-report-metadata --require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints --require-backends embedded,bm25 --forbid-backends neo4j,pggraph,latticedb --require-labeled-metrics --require-dashboard-source embedded=embedded --min-citation-coverage 1.0 --min-projection-events-per-second embedded=40 --max-cold-bootstrap-ms embedded=250 --max-first-useful-init-ms embedded=15000 --max-first-checkout-ms embedded=50 --max-append-to-projection-p95-ms embedded=35 --max-resident-memory-delta-bytes embedded=768000000 --max-on-disk-footprint-bytes embedded=256000000 --max-dashboard-graph-load-ms embedded=500 --max-rebuild-recovery-ms embedded=15000 --max-checkout-p95-ms embedded=100 --max-checkout-p99-ms embedded=85 --min-quality-per-1k-returned-tokens embedded=0.10 --min-answer-at-5-per-1k-returned-tokens embedded=0.10 --min-quality-per-1k-injected-tokens embedded=0.10 --min-answer-at-5-per-1k-injected-tokens embedded=0.10 --max-exact-p95-ms embedded=15 --max-exact-p99-ms embedded=10 --max-keyword-p95-ms embedded=75 --max-keyword-p99-ms embedded=40 --max-vector-p95-ms embedded=25 --max-vector-p99-ms embedded=35 --max-traversal-p95-ms embedded=10 --max-traversal-p99-ms embedded=10"
BACKEND_SCALE_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/longmemeval-100-backend-shootout.json --require-report-metadata --require-markdown-report --require-query-results --require-git-tracked-inputs --verify-report-fingerprints --require-backends embedded,bm25 --forbid-backends neo4j,pggraph,latticedb --require-labeled-metrics --require-dashboard-source embedded=embedded --min-recall-at-5 0.90 --min-citation-coverage 1.0 --min-projection-events-per-second embedded=35 --max-cold-bootstrap-ms embedded=600 --max-first-useful-init-ms embedded=45000 --max-first-checkout-ms embedded=150 --max-append-to-projection-p95-ms embedded=40 --max-resident-memory-delta-bytes embedded=1700000000 --max-on-disk-footprint-bytes embedded=512000000 --max-dashboard-graph-load-ms embedded=500 --max-rebuild-recovery-ms embedded=45000 --max-checkout-p95-ms embedded=200 --max-checkout-p99-ms embedded=250 --min-quality-per-1k-returned-tokens embedded=0.15 --min-answer-at-5-per-1k-returned-tokens embedded=0.15 --min-quality-per-1k-injected-tokens embedded=0.15 --min-answer-at-5-per-1k-injected-tokens embedded=0.15 --max-exact-p95-ms embedded=10 --max-exact-p99-ms embedded=12 --max-keyword-p95-ms embedded=20 --max-keyword-p99-ms embedded=15 --max-vector-p95-ms embedded=15 --max-vector-p99-ms embedded=20 --max-traversal-p95-ms embedded=10 --max-traversal-p99-ms embedded=10"

usage() {
    cat <<USAGE
Usage: scripts/release-check.sh [--root PATH] [--ruff-cmd CMD] [--mypy-cmd CMD] [--pytest-cmd CMD] [--coverage-cmd CMD] [--packet-smoke-cmd CMD] [--examples-smoke-cmd CMD] [--mcp-smoke-cmd CMD] [--langgraph-smoke-cmd CMD] [--coordinate-smoke-cmd CMD] [--package-cmd CMD] [--docs-cmd CMD] [--validate-cmd CMD] [--hook-status-cmd CMD] [--backend-shootout-cmd CMD] [--backend-performance-cmd CMD] [--backend-scale-cmd CMD] [--state-recovery-cmd CMD] [--beta-uat-cmd CMD] [--external-validation-report PATH] [--external-validation-cmd CMD] [--require-external-validation]

Runs ruff, mypy, the full pytest suite, coverage ratchet, packet-memory smoke coverage, public examples, MCP smoke, LangGraph smoke, Coordinate mission smoke, packaging validation, docs validation, deployment validation, activation guardrails, backend shootout validation, medium-scale backend performance validation, 100-query backend scale validation, StateRecoveryBench guardrails, beta UAT, and optional external validation evidence.
Set any smoke command to SKIP:<reason> to print an explicit skip reason.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --ruff-cmd)
            RUFF_CMD="$2"
            shift 2
            ;;
        --mypy-cmd)
            MYPY_CMD="$2"
            shift 2
            ;;
        --pytest-cmd)
            PYTEST_CMD="$2"
            shift 2
            ;;
        --coverage-cmd)
            COVERAGE_CMD="$2"
            shift 2
            ;;
        --packet-smoke-cmd)
            PACKET_SMOKE_CMD="$2"
            shift 2
            ;;
        --examples-smoke-cmd)
            EXAMPLES_SMOKE_CMD="$2"
            shift 2
            ;;
        --mcp-smoke-cmd)
            MCP_SMOKE_CMD="$2"
            shift 2
            ;;
        --langgraph-smoke-cmd)
            LANGGRAPH_SMOKE_CMD="$2"
            shift 2
            ;;
        --coordinate-smoke-cmd)
            COORDINATE_SMOKE_CMD="$2"
            shift 2
            ;;
        --package-cmd)
            PACKAGE_CMD="$2"
            shift 2
            ;;
        --docs-cmd)
            DOCS_CMD="$2"
            shift 2
            ;;
        --validate-cmd)
            VALIDATE_CMD="$2"
            shift 2
            ;;
        --hook-status-cmd)
            HOOK_STATUS_CMD="$2"
            shift 2
            ;;
        --backend-shootout-cmd)
            BACKEND_SHOOTOUT_CMD="$2"
            shift 2
            ;;
        --backend-performance-cmd)
            BACKEND_PERFORMANCE_CMD="$2"
            shift 2
            ;;
        --backend-scale-cmd)
            BACKEND_SCALE_CMD="$2"
            shift 2
            ;;
        --state-recovery-cmd)
            STATE_RECOVERY_CMD="$2"
            shift 2
            ;;
        --beta-uat-cmd)
            BETA_UAT_CMD="$2"
            shift 2
            ;;
        --external-validation-report)
            EXTERNAL_VALIDATION_REPORT="$2"
            EXTERNAL_VALIDATION_CMD="python scripts/check-external-validation.py ${EXTERNAL_VALIDATION_REPORT}"
            shift 2
            ;;
        --external-validation-cmd)
            EXTERNAL_VALIDATION_CMD="$2"
            shift 2
            ;;
        --require-external-validation)
            REQUIRE_EXTERNAL_VALIDATION=1
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

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_gate() {
    local label="$1"
    local command="$2"
    if [[ "${command}" == SKIP:* ]]; then
        local reason="${command#SKIP:}"
        if [[ -z "${reason}" ]]; then
            echo "Skip requested for ${label} without a reason" >&2
            return 2
        fi
        echo "Skipping ${label}: ${reason}"
        return 0
    fi
    echo "Running ${label}..."
    bash -c "${command}"
}

echo "Running release gate..."

"${RUFF_CMD}" check src tests
"${MYPY_CMD}" src
"${PYTEST_CMD}" --tb=short --cov-report=xml
bash -c "${COVERAGE_CMD} --root \"${ROOT}\" --coverage-xml coverage.xml"
run_gate "packet-memory smoke" "${PACKET_SMOKE_CMD}"
run_gate "public examples smoke" "${EXAMPLES_SMOKE_CMD}"
run_gate "MCP smoke" "${MCP_SMOKE_CMD}"
run_gate "LangGraph smoke" "${LANGGRAPH_SMOKE_CMD}"
run_gate "Coordinate mission smoke" "${COORDINATE_SMOKE_CMD}"
"${PACKAGE_CMD}" --root "${ROOT}"
run_gate "docs validation" "${DOCS_CMD} --root \"${ROOT}\""
"${VALIDATE_CMD}" --root "${ROOT}"
bash -c "${HOOK_STATUS_CMD}"
bash -c "${BACKEND_SHOOTOUT_CMD}"
bash -c "${BACKEND_PERFORMANCE_CMD}"
bash -c "${BACKEND_SCALE_CMD}"
run_gate "StateRecoveryBench guardrail" "${STATE_RECOVERY_CMD}"
run_gate "beta UAT" "${BETA_UAT_CMD}"
if [[ "${REQUIRE_EXTERNAL_VALIDATION}" == "1" && "${EXTERNAL_VALIDATION_CMD}" == SKIP:* ]]; then
    echo "External validation is required; pass --external-validation-report PATH or --external-validation-cmd CMD." >&2
    exit 2
fi
if [[ "${REQUIRE_EXTERNAL_VALIDATION}" == "1" && ! "${EXTERNAL_VALIDATION_CMD}" =~ ^[[:space:]]*python[[:space:]]+scripts/check-external-validation\.py[[:space:]]+[^[:space:]\;\|\&\<\>\`]+[[:space:]]*$ ]]; then
    echo "External validation must run scripts/check-external-validation.py in required release mode." >&2
    exit 2
fi
run_gate "external validation" "${EXTERNAL_VALIDATION_CMD}"

echo "Release check passed"
