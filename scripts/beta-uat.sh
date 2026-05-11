#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR=""
INSTALL_SPEC="${ZAXY_BETA_INSTALL_SPEC:-${ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cleanup() {
    if [[ -n "${PROJECT:-}" && -d "${PROJECT}" ]]; then
        zaxy capture stop --workspace "${PROJECT}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${WORKDIR}" && -d "${WORKDIR}" ]]; then
        rm -rf "${WORKDIR}"
    fi
}
trap cleanup EXIT

WORKDIR="$(mktemp -d)"
RUN_ID="$(basename "${WORKDIR}" | tr '[:upper:]' '[:lower:]')"
DOMAIN="${ZAXY_BETA_DOMAIN:-zaxy-beta-uat-${RUN_ID}}"
SESSION_ID="${DOMAIN}-default"
PROJECT="${WORKDIR}/workspace"
mkdir -p "${PROJECT}"
cat > "${PROJECT}/README.md" <<'MARKDOWN'
# Zaxy Beta UAT Workspace

This throwaway repository verifies the clean first-run memory path.
MARKDOWN

pushd "${PROJECT}" >/dev/null

"${PYTHON_BIN}" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "${INSTALL_SPEC}"

zaxy init . \
    --domain "${DOMAIN}" \
    --preset local-codex \
    --capture start \
    --infra check

zaxy memory bootstrap --session-id "${SESSION_ID}"
CHECKOUT_OUTPUT="$(zaxy memory checkout "current workspace memory state" --session-id "${SESSION_ID}")"
echo "${CHECKOUT_OUTPUT}"
grep -q "Answerability: answer_from_memory" <<<"${CHECKOUT_OUTPUT}"
grep -Eq "Citations: [1-9]" <<<"${CHECKOUT_OUTPUT}"
zaxy doctor --eventloom-path .eventloom
zaxy hook-status --eventloom-path .eventloom
zaxy capture status --workspace .
zaxy capture-soak --eventloom-path .eventloom --session-id "${SESSION_ID}"
zaxy memory status --eventloom-path .eventloom
zaxy doctor --beta-readiness --project-root "${ROOT}"
zaxy capture stop --workspace .

popd >/dev/null

echo "Beta UAT passed for ${PROJECT}"
