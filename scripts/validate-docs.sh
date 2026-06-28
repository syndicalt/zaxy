#!/usr/bin/env bash
# Validate the public documentation site (Astro project under web/).
# Syncs the repo docs into the Astro content collection, runs `astro build`,
# then the postbuild gate (web/scripts/check-build.mjs) fails on a broken
# internal link or a missing required doc. Replaces the retired
# scripts/build-site-docs.py + pre-rendered site/ tree.

set -euo pipefail

ROOT="$(pwd)"

usage() {
    cat <<USAGE
Usage: scripts/validate-docs.sh [--root PATH]

Builds the Astro docs site under web/ (docs sync + astro build) and runs the
broken-link / required-doc gate.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
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

cd "${ROOT}/web"
npm ci
npm run build
