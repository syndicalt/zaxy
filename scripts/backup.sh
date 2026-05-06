#!/usr/bin/env bash
# Create an operational backup archive for Zaxy event logs and non-secret config.

set -euo pipefail

ROOT="$(pwd)"
OUTPUT_DIR=""
NAME="zaxy-backup-$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
    cat <<USAGE
Usage: scripts/backup.sh [--root PATH] [--output-dir PATH] [--name NAME]

Creates NAME.tar.gz and NAME.sha256 in OUTPUT_DIR.
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
        --name)
            NAME="$2"
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

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${ROOT}/backups"
fi

mkdir -p "${OUTPUT_DIR}"

ARCHIVE="${OUTPUT_DIR}/${NAME}.tar.gz"
MANIFEST="${OUTPUT_DIR}/${NAME}.sha256"

if [[ -e "${ARCHIVE}" || -e "${MANIFEST}" ]]; then
    echo "Backup output already exists: ${NAME}" >&2
    exit 1
fi

INCLUDES=()
for candidate in ".eventloom" ".env.example" "AGENTS.md" "README.md" "docs/runbook.md"; do
    if [[ -e "${ROOT}/${candidate}" ]]; then
        INCLUDES+=("${candidate}")
    fi
done

if [[ ${#INCLUDES[@]} -eq 0 ]]; then
    echo "No backup inputs found under ${ROOT}" >&2
    exit 1
fi

tar \
    --create \
    --gzip \
    --file "${ARCHIVE}" \
    --directory "${ROOT}" \
    --exclude="secrets" \
    --exclude=".certs" \
    --exclude="backups" \
    "${INCLUDES[@]}"

(
    cd "${OUTPUT_DIR}"
    sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${MANIFEST}")"
)

echo "Created backup: ${ARCHIVE}"
echo "Created manifest: ${MANIFEST}"
