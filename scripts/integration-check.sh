#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="skip-if-unavailable"
PYTEST_ARGS=("-q")
REQUIRED_PORTS=(7688 7689)

usage() {
  cat <<'USAGE'
Usage: scripts/integration-check.sh [--start|--require|--skip-if-unavailable] [-- PYTEST_ARGS...]

Run the local test suite with explicit Neo4j integration-service handling.

Modes:
  --start                Generate TLS certs, start neo4j-test and neo4j-tls, then run pytest.
  --require              Require neo4j-test and neo4j-tls to already be reachable, then run pytest.
  --skip-if-unavailable  Run pytest without tests/test_graph.py when Neo4j test ports are closed.

Default mode: --skip-if-unavailable

Examples:
  scripts/integration-check.sh --start
  scripts/integration-check.sh --require -- -m integration --no-cov -q
  scripts/integration-check.sh --skip-if-unavailable -- --no-cov -q
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      MODE="start"
      shift
      ;;
    --require)
      MODE="require"
      shift
      ;;
    --skip-if-unavailable)
      MODE="skip-if-unavailable"
      shift
      ;;
    --)
      shift
      PYTEST_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

port_open() {
  local port="$1"
  (exec 3<>"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

services_ready() {
  local port
  for port in "${REQUIRED_PORTS[@]}"; do
    if ! port_open "${port}"; then
      return 1
    fi
  done
}

print_start_hint() {
  cat >&2 <<'HINT'
Neo4j integration services are not reachable.
Start them with:
  ./scripts/generate-certs.sh .certs
  docker compose up -d neo4j-test neo4j-tls

Or let this helper start them:
  scripts/integration-check.sh --start
HINT
}

docker_compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "${ROOT}/docker-compose.yml" "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "${ROOT}/docker-compose.yml" "$@"
    return
  fi

  echo "Docker Compose is required to start Neo4j integration services." >&2
  return 127
}

wait_for_services() {
  local timeout_seconds="${ZAXY_INTEGRATION_TIMEOUT:-120}"
  local deadline=$((SECONDS + timeout_seconds))

  until services_ready; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for Neo4j test services on ports 7688 and 7689." >&2
      return 1
    fi
    sleep 2
  done
}

start_services() {
  "${ROOT}/scripts/generate-certs.sh" "${ROOT}/.certs"
  docker_compose up -d neo4j-test neo4j-tls
  wait_for_services
}

cd "${ROOT}"

case "${MODE}" in
  start)
    start_services
    pytest "${PYTEST_ARGS[@]}"
    ;;
  require)
    if ! services_ready; then
      print_start_hint
      exit 1
    fi
    pytest "${PYTEST_ARGS[@]}"
    ;;
  skip-if-unavailable)
    if services_ready; then
      pytest "${PYTEST_ARGS[@]}"
    else
      print_start_hint
      echo "Running without Neo4j-backed graph integration tests." >&2
      pytest "${PYTEST_ARGS[@]}" --ignore=tests/test_graph.py
    fi
    ;;
  *)
    echo "Unknown integration-check mode: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
