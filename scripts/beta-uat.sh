#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR=""
INSTALL_SPEC="${ZAXY_BETA_INSTALL_SPEC:-${ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTIVE_PROJECTS=()

cleanup() {
    for PROJECT in "${ACTIVE_PROJECTS[@]}"; do
        [[ -d "${PROJECT}" ]] || continue
        zaxy capture stop --workspace "${PROJECT}" >/dev/null 2>&1 || true
    done
    if [[ -n "${WORKDIR}" && -d "${WORKDIR}" ]]; then
        rm -rf "${WORKDIR}"
    fi
}
trap cleanup EXIT

WORKDIR="$(mktemp -d)"
RUN_ID="$(basename "${WORKDIR}" | tr '[:upper:]' '[:lower:]')"
BASE_DOMAIN="${ZAXY_BETA_DOMAIN:-zaxy-beta-uat-${RUN_ID}}"

run_workspace() {
    local label="$1"
    local preset="$2"
    local capture_action="$3"
    local project="${WORKDIR}/${label}-workspace"
    local domain="${BASE_DOMAIN}-${label}"
    local session_id="${domain}-default"

    ACTIVE_PROJECTS+=("${project}")
    mkdir -p "${project}"
    cat > "${project}/README.md" <<'MARKDOWN'
# Zaxy Beta UAT Workspace

This throwaway repository verifies the clean first-run memory path.
MARKDOWN

    pushd "${project}" >/dev/null

    "${PYTHON_BIN}" -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install "${INSTALL_SPEC}"

    zaxy init . \
        --domain "${domain}" \
        --preset "${preset}" \
        --capture "${capture_action}" \
        --infra check

    BOOTSTRAP_OUTPUT="$(zaxy memory bootstrap --session-id "${session_id}")"
    echo "${BOOTSTRAP_OUTPUT}"
    grep -q "Call memory_checkout before answering roadmap or implementation questions." <<<"${BOOTSTRAP_OUTPUT}"
    grep -q "Call memory_feedback when cited checkout context was used." <<<"${BOOTSTRAP_OUTPUT}"
    CHECKOUT_OUTPUT="$(zaxy memory checkout "current workspace memory state" --session-id "${session_id}")"
    echo "${CHECKOUT_OUTPUT}"
    grep -q "Answerability: answer_from_memory" <<<"${CHECKOUT_OUTPUT}"
    grep -Eq "Citations: [1-9]" <<<"${CHECKOUT_OUTPUT}"
    grep -q "Suggested next call: memory_checkout" <<<"${CHECKOUT_OUTPUT}"
    grep -q "Feedback: call memory_feedback" <<<"${CHECKOUT_OUTPUT}"
    zaxy hook-event command \
        --eventloom-path .eventloom \
        --session-id "${session_id}" \
        --source "${label}-local" \
        --workspace "${project}" \
        --command "zaxy memory bootstrap" \
        --exit-code 0
    zaxy hook-event file-edit \
        --eventloom-path .eventloom \
        --session-id "${session_id}" \
        --source "${label}-local" \
        --workspace "${project}" \
        --path README.md \
        --operation created \
        --summary "Created clean UAT workspace README"
    zaxy hook-event tool-call \
        --eventloom-path .eventloom \
        --session-id "${session_id}" \
        --source "${label}-local" \
        --workspace "${project}" \
        --tool-name memory_checkout \
        --tool-status ok \
        --result-summary "Checkout returned cited model-facing memory guidance"
    zaxy hook-event transcript-turn \
        --eventloom-path .eventloom \
        --session-id "${session_id}" \
        --source "${label}-local" \
        --role assistant \
        --content "Verified memory bootstrap and memory checkout guidance in a clean workspace." \
        --turn-index 1
    zaxy doctor --eventloom-path .eventloom
    zaxy hook-status --eventloom-path .eventloom
    zaxy capture status --workspace .
    zaxy capture-soak --eventloom-path .eventloom --workspace-root . --session-id "${session_id}"
    zaxy memory status --eventloom-path .eventloom
    zaxy doctor --beta-readiness --project-root "${ROOT}"
    zaxy capture stop --workspace .

    popd >/dev/null

    echo "Beta UAT passed for ${project}"
}

run_workspace "codex" "local-codex" "start"
run_workspace "claude-code" "local-claude" "status"
