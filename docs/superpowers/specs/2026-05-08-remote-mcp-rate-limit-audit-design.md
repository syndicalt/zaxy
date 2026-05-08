# Remote MCP Rate Limit And Audit Export Design

## Goal

Protect remote MCP/SSE deployments with session-scoped request rate limiting and append-only audit export.

## Scope

This slice applies only to remote HTTP/SSE transport. Stdio remains unchanged. The limiter is in-memory and process-local. Distributed/shared rate limiting is outside this slice.

## Architecture

Remote requests already flow through `MCPTransportAuth.authorize()` before `/sse` and `/messages/` continue. The new design keeps that boundary:

1. authenticate headers and resolve a remote session ID;
2. apply a session-first fixed-window rate limiter;
3. export a compact audit JSONL record for allowed, auth-denied, and rate-limited requests;
4. return `401` for auth/session failures and `429` for rate-limit failures.

The rate-limit key is the validated remote session ID. If a request cannot produce a session ID, the client host may be used only for audit context, not as the primary tenant boundary.

## Configuration

- `MCP_RATE_LIMIT_ENABLED`: default `true`
- `MCP_RATE_LIMIT_REQUESTS`: default `120`
- `MCP_RATE_LIMIT_WINDOW_SECONDS`: default `60`
- `MCP_AUDIT_ENABLED`: default `false`
- `MCP_AUDIT_PATH`: default `.eventloom/remote_audit.jsonl`

## Data

Audit records are newline-delimited JSON with:

- timestamp
- session_id when known
- route
- method
- outcome: `allowed`, `denied_auth`, or `denied_rate_limit`
- reason
- client_host when available

No bearer tokens, JWTs, raw payloads, or query text are exported.

## Tests

Unit tests cover limiter allow/deny behavior, window reset, audit JSONL export, and disabled audit no-op behavior. MCP tests cover request authorization paths returning `429` after a configured limit and writing audit records for allowed and denied requests.
