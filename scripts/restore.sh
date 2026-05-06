#!/usr/bin/env bash
# Restore a Zaxy backup archive after validating its checksum manifest.

set -euo pipefail

ARCHIVE=""
MANIFEST=""
TARGET=""
FORCE=false

usage() {
    cat <<USAGE
Usage: scripts/restore.sh --archive PATH --manifest PATH --target PATH [--force]

Validates MANIFEST, then extracts ARCHIVE into TARGET.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive)
            ARCHIVE="$2"
            shift 2
            ;;
        --manifest)
            MANIFEST="$2"
            shift 2
            ;;
        --target)
            TARGET="$2"
            shift 2
            ;;
        --force)
            FORCE=true
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

if [[ -z "${ARCHIVE}" || -z "${MANIFEST}" || -z "${TARGET}" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Archive not found: ${ARCHIVE}" >&2
    exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
    echo "Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

ARCHIVE_DIR="$(cd "$(dirname "${ARCHIVE}")" && pwd)"
ARCHIVE_BASENAME="$(basename "${ARCHIVE}")"
MANIFEST_ABS="$(cd "$(dirname "${MANIFEST}")" && pwd)/$(basename "${MANIFEST}")"

(
    cd "${ARCHIVE_DIR}"
    if ! sha256sum --check "${MANIFEST_ABS}" >/dev/null 2>&1; then
        echo "Archive checksum validation failed" >&2
        exit 1
    fi
)

if [[ "${FORCE}" != true && -e "${TARGET}/.eventloom" ]]; then
    echo "Target event log already exists: ${TARGET}/.eventloom" >&2
    exit 1
fi

mkdir -p "${TARGET}"

tar \
    --extract \
    --gzip \
    --file "${ARCHIVE_DIR}/${ARCHIVE_BASENAME}" \
    --directory "${TARGET}"

echo "Restored backup: ${ARCHIVE} -> ${TARGET}"
