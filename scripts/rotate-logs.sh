#!/usr/bin/env bash
# Rotate a Zaxy Eventloom JSONL log into a checksumed archive file.

set -euo pipefail

LOG_PATH=""
ARCHIVE_DIR=""
NAME=""

usage() {
    cat <<USAGE
Usage: scripts/rotate-logs.sh --log PATH [--archive-dir PATH] [--name NAME]

Copies LOG into ARCHIVE_DIR/NAME.jsonl, writes NAME.sha256, then truncates LOG.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log)
            LOG_PATH="$2"
            shift 2
            ;;
        --archive-dir)
            ARCHIVE_DIR="$2"
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

if [[ -z "${LOG_PATH}" ]]; then
    usage >&2
    exit 2
fi

if [[ ! -f "${LOG_PATH}" ]]; then
    echo "Log not found: ${LOG_PATH}" >&2
    exit 1
fi

if [[ ! -s "${LOG_PATH}" ]]; then
    echo "Log is empty: ${LOG_PATH}" >&2
    exit 1
fi

LOG_DIR="$(cd "$(dirname "${LOG_PATH}")" && pwd)"
LOG_BASENAME="$(basename "${LOG_PATH}")"

if [[ -z "${ARCHIVE_DIR}" ]]; then
    ARCHIVE_DIR="${LOG_DIR}/archive"
fi

if [[ -z "${NAME}" ]]; then
    stem="${LOG_BASENAME%.jsonl}"
    NAME="${stem}-$(date -u +%Y%m%dT%H%M%SZ)"
fi

mkdir -p "${ARCHIVE_DIR}"

ARCHIVE="${ARCHIVE_DIR}/${NAME}.jsonl"
MANIFEST="${ARCHIVE_DIR}/${NAME}.sha256"

if [[ -e "${ARCHIVE}" || -e "${MANIFEST}" ]]; then
    echo "Rotation output already exists: ${NAME}" >&2
    exit 1
fi

cp "${LOG_PATH}" "${ARCHIVE}"

(
    cd "${ARCHIVE_DIR}"
    sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${MANIFEST}")"
)

: > "${LOG_PATH}"

echo "Rotated log: ${LOG_PATH} -> ${ARCHIVE}"
echo "Created manifest: ${MANIFEST}"
