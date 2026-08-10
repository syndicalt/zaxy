# mcp 2.0 migration — design

**Date:** 2026-08-10 · **Kind:** design · **Status:** agreed (pre-implementation)

## Background

mcp 2.0.0 (2026-07-28) is a breaking major of the MCP Python SDK: wire types
moved to snake_case via the new exact-pinned `mcp-types` package, the lowlevel
`Server` decorator handlers were removed in favor of constructor `on_*`
params, `McpError` was renamed `MCPError`, and the message unions
(`JSONRPCMessage` & friends) are plain unions validated through
`TypeAdapter`s instead of `RootModel`s. The weekly dependency-drift workflow
caught the release immediately (runs 31365647515, 30802063102: 66 mypy
errors); PR #166 capped the dependency at `mcp>=1.0.0,<2` as the stopgap.
The 1.x line is maintenance-only (security fixes), so the cap buys time, not
a steady state. This spec is the deliberate migration the cap's comment
promised.

Reference: official v1→v2 migration guide
(https://py.sdk.modelcontextprotocol.io/v2/migration/), read in full.

## Blast radius (inventoried)

Zaxy's SDK surface is server-only and small:

- `src/zaxy/mcp_tool_specs.py` — 61 × `Tool(name=..., description=...,
  inputSchema={...})`. camelCase kwargs still parse under v2 but fail mypy
  strict; attribute access must be snake_case.
- `src/zaxy/mcp_server.py` — the only real consumer:
  - module-level `app = Server("zaxy-memory")` (L280);
  - `@app.list_tools()` / `@app.call_tool()` decorators registered inside
    `main()` (L2927–2954), closing over the local `active_server`;
  - custom Unix-socket transport `_socket_mcp_transport` (L2683–2726) using
    `types.JSONRPCMessage.model_validate_json` and
    `model_dump_json(by_alias=True)`;
  - `stdio_server()`, `app.run()`, `app.create_initialization_options()`,
    `SseServerTransport` — all **unchanged** in v2 per the guide.
- `zaxy_benchmarks/agent_experience_lanes.py:165` — one `tool.inputSchema`
  read.
- Tests: ~80 `.inputSchema` attribute reads (`tests/test_mcp.py`,
  `tests/test_mcp_server.py:863–889`, `tests/test_fleet_surface.py:819`);
  `tests/test_mcp.py:4029–4032` uses `JSONRPCMessage.model_validate_json` +
  `SessionMessage`; ~10 test classes patch `mcp_server.app.run` /
  `mcp_server.stdio_server`.
- No FastMCP, no client transports, no `McpError`, no `isError` results —
  zaxy returns its own JSON error payloads as `TextContent`.

## Decisions

1. **Minimal lowlevel port.** Keep the lowlevel `Server`, the SSE transport
   (`SseServerTransport` is still supported in v2), the Unix-socket owner
   transport, and every wire contract byte-identical. No move to the
   high-level `MCPServer` (zaxy needs dynamic `visible_tools()`, capture
   hooks, and its own error-payload format); no SSE→streamable-http
   migration; no adoption of v2 features (`_meta` envelopes, middleware,
   tasks). Those are separate decisions, deliberately not now.

2. **Handler registration moves to the constructor.** v2 replaces the
   decorators with `Server(..., on_list_tools=..., on_call_tool=...)`.
   Handlers receive `(ctx: ServerRequestContext, params)` and return full
   result types (`ListToolsResult`, `CallToolResult`); unwrapped
   `list[Tool]` / `list[TextContent]` returns are no longer accepted.
   Because `app` is module-level (and tests patch `mcp_server.app.run`), the
   handlers become module-level functions that resolve the active
   `ZaxyMCPServer` through a module global; `main()` / `main_sse()` assign
   the resolved instance to that global before `app.run(...)`, preserving
   today's registration semantics (including the socket-owner path, which
   relies on `main()` having registered the handlers).

3. **Error contract preserved.** v1's decorator wrapped handler exceptions
   into `isError: true` results and performed jsonschema argument
   validation; v2 does neither (uncaught exceptions become JSON-RPC errors).
   Zaxy already catches everything into `_mcp_error_result` and validates
   its own arguments, so wrapping those payloads in
   `CallToolResult(content=...)` keeps client-visible behavior identical.
   Verified during implementation that no test relies on the removed
   SDK-level validation.

4. **Socket transport uses the adapter.** `JSONRPCMessage.model_validate_json`
   → `jsonrpc_message_adapter.validate_json`; `SessionMessage(...)` takes
   the union member directly (same call shape); outbound keeps
   `model_dump_json(by_alias=True, exclude_none=True)`.

5. **Explicit server version.** v2 reports an empty `serverInfo.version`
   for unversioned servers (v1 reported the mcp package version — wrong
   anyway). Pass the `zaxy-memory` distribution version explicitly.

6. **Dependency floor.** `pyproject.toml` bound becomes `mcp>=2.0.0,<3`;
   `constraints/ci.txt` recompiled. v2 pulls `mcp-types` (exact-pinned),
   `httpx2`, `opentelemetry-api`, and raises floors (pydantic≥2.12,
   sse-starlette≥3.0.0, anyio≥4.9, typing-extensions≥4.13). Zaxy's direct
   `httpx` dependency is unaffected (the two coexist).

7. **Snapshots expected unchanged.** `mcp-tool-contract.json` keys are
   already snake_case and the 61 schemas are untouched, so it should be
   byte-identical (regenerate only if it diffs); `mcp-response-snapshots.json`
   holds zaxy's own payloads and should not move. Tool count stays 61.

## Test plan

- Test-first: the new module-level `_on_list_tools` / `_on_call_tool`
  handlers get direct behavioral tests (fake ctx/params) — also protects
  the 92.00% coverage ratchet, since the v1 closures' coverage moves.
- Mechanical test updates: `.inputSchema` → `.input_schema` (~80 reads),
  socket-transport test → adapter.
- Gates in CI order: ruff → mypy strict → pytest 3.11/3.12/3.13 → coverage
  ratchet. Local runs exclude `tests/test_doctor.py` and pass
  `--benchmark-disable` per CLAUDE.md.
- `scripts/mcp_smoke_test.py` (raw JSON-RPC, handshake-era
  `protocolVersion: "2024-11-05"`) must pass — v2 serves both protocol eras.
- Final proof: dispatch `dependency-drift.yml` on the migration branch;
  the unpinned job must go green with mcp 2.0.0 resolving.

## Risks

- Coverage ratchet drift from moved handler code → mitigated by the new
  direct handler tests.
- Hidden reliance on v1 decorator arg validation → checked explicitly.
- sse-starlette 3.x co-install → zaxy doesn't import it directly; covered
  by CI + smoke test.
- Stricter v2 frame validation on the socket transport → covered by
  `tests/test_mcp.py` socket-transport tests + smoke test.
