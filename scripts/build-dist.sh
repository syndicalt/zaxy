#!/usr/bin/env bash
# Build and validate Python distribution artifacts.

set -euo pipefail

ROOT="$(pwd)"
DIST_DIR="dist"
BUILD_CMD="python -m build"
TWINE_CMD="python -m twine"

usage() {
    cat <<USAGE
Usage: scripts/build-dist.sh [--root PATH] [--dist-dir PATH] [--build-cmd CMD] [--twine-cmd CMD]

Builds sdist and wheel artifacts, then validates distribution metadata with twine.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        --dist-dir)
            DIST_DIR="$2"
            shift 2
            ;;
        --build-cmd)
            BUILD_CMD="$2"
            shift 2
            ;;
        --twine-cmd)
            TWINE_CMD="$2"
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

echo "Building package artifacts..."

mkdir -p "${DIST_DIR}"
${BUILD_CMD} --sdist --wheel --outdir "${DIST_DIR}" "${ROOT}"
${TWINE_CMD} check "${DIST_DIR}"/*

echo "Package artifacts passed"
