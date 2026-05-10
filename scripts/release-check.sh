#!/usr/bin/env bash
# Run the Zaxy go-live release gate.

set -euo pipefail

ROOT="$(pwd)"
RUFF_CMD="ruff"
MYPY_CMD="mypy"
PYTEST_CMD="pytest"
PACKET_SMOKE_CMD="pytest tests/test_packet_memory_e2e.py --no-cov -q"
PACKAGE_CMD="scripts/build-dist.sh"
DOCS_CMD="scripts/validate-docs.sh"
VALIDATE_CMD="scripts/validate-deployment.sh"

usage() {
    cat <<USAGE
Usage: scripts/release-check.sh [--root PATH] [--ruff-cmd CMD] [--mypy-cmd CMD] [--pytest-cmd CMD] [--packet-smoke-cmd CMD] [--package-cmd CMD] [--docs-cmd CMD] [--validate-cmd CMD]

Runs ruff, mypy, the full pytest suite, packet-memory smoke coverage, packaging validation, docs validation, and deployment validation.
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

echo "Running release gate..."

"${RUFF_CMD}" check src tests
"${MYPY_CMD}" src
"${PYTEST_CMD}" --tb=short
"${PACKET_SMOKE_CMD}"
"${PACKAGE_CMD}" --root "${ROOT}"
"${DOCS_CMD}" --root "${ROOT}"
"${VALIDATE_CMD}" --root "${ROOT}"

echo "Release check passed"
