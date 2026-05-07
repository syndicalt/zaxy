#!/usr/bin/env bash
# Validate a production Zaxy deployment directory before remote MCP/SSE use.

set -euo pipefail

ROOT="$(pwd)"

usage() {
    cat <<USAGE
Usage: scripts/validate-deployment.sh [--root PATH]

Checks production env, Neo4j TLS config, remote MCP auth, and secret file permissions.
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

ENV_FILE="${ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing deployment env file: ${ENV_FILE}" >&2
    exit 1
fi

declare -A ENV
while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    value="${value%\"}"
    value="${value#\"}"
    value="${value%\'}"
    value="${value#\'}"
    ENV["${key}"]="${value}"
done < "${ENV_FILE}"

failures=0

fail() {
    echo "$1" >&2
    failures=$((failures + 1))
}

resolve_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${ROOT}/${path}"
    fi
}

require_secret_file() {
    local env_name="$1"
    local path="${ENV[${env_name}]:-}"
    if [[ -z "${path}" ]]; then
        return
    fi
    local resolved
    resolved="$(resolve_path "${path}")"
    if [[ ! -f "${resolved}" ]]; then
        fail "${env_name} does not exist: ${path}"
        return
    fi
    local mode
    mode="$(stat -c '%a' "${resolved}")"
    local other_permissions=$((8#${mode} & 7))
    if [[ ${other_permissions} -ne 0 ]]; then
        fail "${env_name} is world-readable or world-writable: ${path}"
    fi
}

if [[ "${ENV[ZAXY_ENV]:-}" != "production" ]]; then
    fail "ZAXY_ENV must be production"
fi

neo4j_uri="${ENV[NEO4J_URI]:-}"
neo4j_ca_cert="${ENV[NEO4J_CA_CERT]:-}"
if [[ "${neo4j_uri}" == bolt://* && -z "${neo4j_ca_cert}" ]]; then
    fail "NEO4J_CA_CERT is required when NEO4J_URI uses bolt:// in production"
fi

remote_token="${ENV[MCP_REMOTE_AUTH_TOKEN]:-}"
remote_token_file="${ENV[MCP_REMOTE_AUTH_TOKEN_FILE]:-}"
if [[ -z "${remote_token}" && -z "${remote_token_file}" ]]; then
    fail "MCP_REMOTE_AUTH_TOKEN or MCP_REMOTE_AUTH_TOKEN_FILE is required"
fi

admin_token="${ENV[MCP_ADMIN_TOKEN]:-}"
admin_token_file="${ENV[MCP_ADMIN_TOKEN_FILE]:-}"
if [[ -z "${admin_token}" && -z "${admin_token_file}" ]]; then
    fail "MCP_ADMIN_TOKEN or MCP_ADMIN_TOKEN_FILE is required"
fi

session_header="${ENV[MCP_REMOTE_SESSION_HEADER]:-x-zaxy-session-id}"
if [[ -z "${session_header}" ]]; then
    fail "MCP_REMOTE_SESSION_HEADER must not be empty"
fi

for secret_env in \
    NEO4J_PASSWORD_FILE \
    MCP_ADMIN_TOKEN_FILE \
    MCP_REMOTE_AUTH_TOKEN_FILE \
    OPENAI_API_KEY_FILE \
    PATHLIGHT_ACCESS_TOKEN_FILE; do
    require_secret_file "${secret_env}"
done

if [[ ${failures} -gt 0 ]]; then
    echo "Deployment validation failed with ${failures} issue(s)" >&2
    exit 1
fi

echo "Deployment validation passed"
