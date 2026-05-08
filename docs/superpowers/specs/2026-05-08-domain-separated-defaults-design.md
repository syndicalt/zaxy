# Domain-Separated Defaults Design

## Goal

Prevent accidental cross-project memory sharing when MCP clients omit `session_id`.

## Scope

This slice fixes default scoping. It does not change Neo4j schema or migrate existing sessions.

## Architecture

Zaxy keeps session IDs as the graph and Eventloom isolation key. A project domain now feeds the default session:

- `ZAXY_DOMAIN=zaxy`
- `EVENTLOOM_THREAD=zaxy-default`

Generated MCP configs include both values. MCP handlers use the configured `EVENTLOOM_THREAD` instead of hardcoded `default` when a tool call omits `session_id`.

## Behavior

`zaxy ide-config ...` accepts `--domain`. If omitted, it derives a safe domain slug from the current working directory. Remote SSE configs use the derived default session in `x-zaxy-session-id`.

Existing explicit `session_id` values keep working.

## Tests

Tests cover domain slugging, generated stdio/SSE config defaults, and MCP handler fallback to `EVENTLOOM_THREAD`.
