#!/usr/bin/env bash
# Run the Zaxy go-live release gate.

set -euo pipefail

ROOT="$(pwd)"
RUFF_CMD="ruff"
MYPY_CMD="mypy"
PYTEST_CMD="pytest"
COVERAGE_CMD="python scripts/check-coverage.py"
PACKET_SMOKE_CMD="pytest tests/test_packet_memory_e2e.py --no-cov -q"
PACKAGE_CMD="scripts/build-dist.sh"
DOCS_CMD="scripts/validate-docs.sh"
VALIDATE_CMD="scripts/validate-deployment.sh"
HOOK_STATUS_CMD="PYTHONPATH=src python -m zaxy hook-status --eventloom-path reports/activation-release --now 2026-05-20T12:00:00+00:00 --min-activation-rate 1.0 --max-checkout-prompt-tokens 5000 --min-checkout-facts-per-1k-tokens 0.1"
BACKEND_SHOOTOUT_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/backend-shootout.json --require-report-metadata --require-markdown-report --verify-report-fingerprints --require-backends embedded,bm25 --forbid-backends latticedb --require-labeled-metrics --require-dashboard-source embedded=embedded --min-answer-at-5 0.5 --min-recall-at-5 0.5 --min-citation-coverage 1.0 --min-quality-per-1k-injected-tokens embedded=1.0 --min-answer-at-5-per-1k-injected-tokens embedded=1.0 --max-cold-bootstrap-ms embedded=250 --max-first-checkout-ms embedded=25 --max-append-to-projection-p95-ms embedded=50 --max-resident-memory-delta-bytes embedded=256000000 --max-on-disk-footprint-bytes embedded=256000000 --max-dashboard-graph-load-ms embedded=250 --max-checkout-p99-ms embedded=25 --max-exact-p99-ms embedded=10 --max-keyword-p99-ms embedded=5 --max-vector-p99-ms embedded=5 --max-traversal-p99-ms embedded=5"
BACKEND_PERFORMANCE_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/longmemeval-40-backend-shootout.json --require-report-metadata --require-markdown-report --verify-report-fingerprints --require-backends embedded,bm25 --forbid-backends latticedb --require-labeled-metrics --require-dashboard-source embedded=embedded --min-citation-coverage 1.0 --min-projection-events-per-second embedded=40 --max-cold-bootstrap-ms embedded=200 --max-first-useful-init-ms embedded=15000 --max-first-checkout-ms embedded=50 --max-append-to-projection-p95-ms embedded=30 --max-resident-memory-delta-bytes embedded=768000000 --max-on-disk-footprint-bytes embedded=256000000 --max-dashboard-graph-load-ms embedded=500 --max-rebuild-recovery-ms embedded=15000 --max-checkout-p95-ms embedded=100 --max-checkout-p99-ms embedded=75 --min-quality-per-1k-returned-tokens embedded=0.10 --min-answer-at-5-per-1k-returned-tokens embedded=0.10 --min-quality-per-1k-injected-tokens embedded=0.10 --min-answer-at-5-per-1k-injected-tokens embedded=0.10 --max-exact-p95-ms embedded=15 --max-exact-p99-ms embedded=10 --max-keyword-p95-ms embedded=75 --max-keyword-p99-ms embedded=40 --max-vector-p95-ms embedded=25 --max-vector-p99-ms embedded=35 --max-traversal-p95-ms embedded=10 --max-traversal-p99-ms embedded=10"
BACKEND_SCALE_CMD="python scripts/check-backend-shootout.py reports/backend-shootout/longmemeval-100-backend-shootout.json --require-report-metadata --require-markdown-report --verify-report-fingerprints --require-backends embedded,bm25 --forbid-backends latticedb --require-labeled-metrics --require-dashboard-source embedded=embedded --min-citation-coverage 1.0 --min-projection-events-per-second embedded=40 --max-cold-bootstrap-ms embedded=200 --max-first-useful-init-ms embedded=40000 --max-first-checkout-ms embedded=100 --max-append-to-projection-p95-ms embedded=40 --max-resident-memory-delta-bytes embedded=1536000000 --max-on-disk-footprint-bytes embedded=512000000 --max-dashboard-graph-load-ms embedded=500 --max-rebuild-recovery-ms embedded=40000 --max-checkout-p95-ms embedded=125 --max-checkout-p99-ms embedded=175 --min-quality-per-1k-returned-tokens embedded=0.15 --min-answer-at-5-per-1k-returned-tokens embedded=0.15 --min-quality-per-1k-injected-tokens embedded=0.15 --min-answer-at-5-per-1k-injected-tokens embedded=0.15 --max-exact-p95-ms embedded=10 --max-exact-p99-ms embedded=12 --max-keyword-p95-ms embedded=60 --max-keyword-p99-ms embedded=80 --max-vector-p95-ms embedded=15 --max-vector-p99-ms embedded=20 --max-traversal-p95-ms embedded=10 --max-traversal-p99-ms embedded=10"

usage() {
    cat <<USAGE
Usage: scripts/release-check.sh [--root PATH] [--ruff-cmd CMD] [--mypy-cmd CMD] [--pytest-cmd CMD] [--coverage-cmd CMD] [--packet-smoke-cmd CMD] [--package-cmd CMD] [--docs-cmd CMD] [--validate-cmd CMD] [--hook-status-cmd CMD] [--backend-shootout-cmd CMD] [--backend-performance-cmd CMD] [--backend-scale-cmd CMD]

Runs ruff, mypy, the full pytest suite, coverage ratchet, packet-memory smoke coverage, packaging validation, docs validation, deployment validation, activation guardrails, backend shootout validation, medium-scale backend performance validation, and 100-query backend scale validation.
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

echo "Running release gate..."

"${RUFF_CMD}" check src tests
"${MYPY_CMD}" src
"${PYTEST_CMD}" --tb=short --cov-report=xml
bash -c "${COVERAGE_CMD} --root \"${ROOT}\" --coverage-xml coverage.xml"
bash -c "${PACKET_SMOKE_CMD}"
"${PACKAGE_CMD}" --root "${ROOT}"
"${DOCS_CMD}" --root "${ROOT}"
"${VALIDATE_CMD}" --root "${ROOT}"
bash -c "${HOOK_STATUS_CMD}"
bash -c "${BACKEND_SHOOTOUT_CMD}"
bash -c "${BACKEND_PERFORMANCE_CMD}"
bash -c "${BACKEND_SCALE_CMD}"

echo "Release check passed"
