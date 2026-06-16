# Embedded MCP Single-Owner Enforcement + Orphan Reaping — Investigation & Design

Status: **investigation complete; design proposed (agree before Phase 2 code)**
Date: 2026-06-16
Owner: Zaxy core

## 1. Incident

For an entire working session, every `memory_checkout` appeared to "hang"
(~18-22s+ then the MCP server disconnected). After misattributing it to checkout
performance (real but unrelated work shipped as #81/#82/#83) and a stale install
(server was on 2.2.0), the true cause was found: **the embedded LadybugDB
projection was corrupted** — opening it throws
`RuntimeError: Assertion failed ... wal_record.cpp ... UNREACHABLE_CODE` — and
**four `zaxy serve` processes** (three orphans from prior sessions + the live
one), all with `EVENTLOOM_PATH=.../zaxy/.eventloom`, were associated with that
one store. Manual recovery: kill the this-workspace servers + move the corrupt
store aside + reconnect. See memory `incident-mcp-orphan-projection-corruption`.

This note root-causes *why the safeguards let that happen* and proposes the
durable fix. The embedded backend takes an **exclusive write lock**; concurrent
writers corrupt the WAL. The design intends **one owner + N proxies** per
workspace (`EmbeddedMcpRuntimeCoordinator`: "Coordinates a single embedded MCP
owner per Eventloom directory"; non-owners run `mcp_server.proxy_main` over the
owner's unix socket).

## 2. Method

Static trace of `cli/serving.py:serve` (the owner/proxy decision) and
`mcp_runtime.py` (the coordinator), plus a reproduction of the coordinator's
claim behavior.

## 3. Findings (each confirmed by code and/or reproduction)

### F1 — The coordinator only guards the `serve` stdio path; every other store-open bypasses it
Owner claim happens only here:
`cli/serving.py:1242` — `if transport == "stdio" and projection_backend == "embedded": coordinator.try_claim_owner()`.
The CLI commands that open the embedded store directly — `memory checkout` (its
projection path / `#83` lock-fallback), `reproject`, `doctor`, `dashboard` —
build a `MemoryFabric`/store and `connect()` with **no owner claim**. So any
non-serve store access is uncoordinated: it can open the store as a second writer
alongside a serve-owner (or another CLI), or hit the lock and fall back.

### F2 — The owner lock and the store are keyed independently; divergence defeats single-owner
- Owner lock key: `EmbeddedMcpRuntimeCoordinator.from_eventloom_path(resolved_eventloom_path)`
  where `resolved_eventloom_path = --eventloom-path arg or $EVENTLOOM_PATH or cwd/.eventloom` (`serving.py:1220`).
- Store path: `settings.embedded_graph_path` (`serving.py:1228`, from cwd config / `$EMBEDDED_GRAPH_PATH`).

These are resolved separately. When they diverge — mismatched env, cwd-relative
settings, or an explicit `--embedded-graph-path` that doesn't match the
eventloom — single-owner is enforced on the **wrong key**. **Reproduced:** two
coordinators on different eventloom paths *both* successfully claim owner; if both
then open one shared store, that's two writers → corruption. (Same-key correctly
refuses the second claimant.)

### F3 — No orphan reaping; recovery only handles *dead* owners, not *live-but-broken* ones
`try_claim_owner` uses `fcntl.flock(LOCK_EX|LOCK_NB)` held for process lifetime —
correct for live coordination. But:
- A **live orphan** (its client session ended, process never exited) holds the
  flock forever; nothing detects or reaps it. New clients then *proxy to the
  orphan* — so a hung/broken orphan-owner makes **every** client's checkout fail
  (the observed symptom).
- `repair_stale_runtime` (`mcp_runtime.py:139`) only cleans when the lock is
  **free** (owner dead). If the lock is held but the owner socket is unhealthy it
  returns a `warning` telling the human to "Fully exit stale processes" — i.e.
  exactly the manual step we performed. No automatic recovery from a live-broken
  owner, and no reaping of accumulated orphans.

### F4 — Unclean owner death leaves a dirty WAL; the next open crashes instead of self-healing
If an owner is `SIGKILL`ed or crashes mid-write (or a CLI store-write is
timeout-killed — F1), LadybugDB's WAL is left uncheckpointed/dirty. The next
open replays it and can hit `UNREACHABLE_CODE`. `EmbeddedGraphStore.connect`
only handles the pre-fork-Kuzu *format-incompatibility* case (moves it aside);
it re-raises everything else, so WAL corruption propagates and **crashes the
checkout** rather than quarantining the (derived) store and rebuilding.

## 4. Why the incident happened (synthesis)

The four orphans shared one `EVENTLOOM_PATH`, so F2 alone wouldn't multi-own
them. The corruption is best explained by **F1 + F4 + F3 compounding**:
uncoordinated direct store-opens (CLI checkouts, including timeout-killed ones
mid-write) and/or an uncleanly-killed owner left a dirty WAL (F4); orphaned
servers accumulated and were never reaped (F3); and once the store was corrupt,
every owner/proxy that opened it crashed (F4) — the perpetual hang+disconnect.
The exact historical sequence can't be fully reconstructed (processes gone), but
all four gaps are confirmed and each independently enables this corruption class.

## 5. Design — the durable fix (phased; agree before code)

Invariants (hard): multiple clients in one repo (e.g. two Claude instances) must
work as exactly **one owner + N proxies**; the store is opened by **at most one
process**; the Eventloom log stays authority and the projection is rebuildable;
reaping never touches live or cross-workspace servers.

- **Phase 2 — single-owner enforcement that actually covers the store (F1, F2).**
  Key the owner lock on the **canonical store path** (resolve eventloom ⟺ store
  once; assert they agree, or derive the lock from the store path) so coordination
  can't guard the wrong thing. Route **every** embedded-store open through the
  claim: `serve` proxies (today's behavior); CLI store-opens (`checkout`,
  `reproject`, `doctor`, `dashboard`) must either proxy to the owner or refuse /
  run graph-degraded — **never** open the store as a second writer. (The `#83`
  null-backend degrade is the graph-degraded primitive for the CLI case.)
- **Phase 3 — orphan reaping + live-broken recovery (F3).** Detect owners that are
  dead *or* live-but-unhealthy (lock held but socket not accepting), scoped to the
  workspace; reap/recover safely (startup self-heal and/or `zaxy doctor --repair`),
  never killing healthy or other-workspace servers. Extend `repair_stale_runtime`
  to handle the live-broken case instead of only advising a human.
- **Related (recommend folding in) — WAL-corruption resilience (F4).** On open,
  detect WAL/store corruption and quarantine-and-rebuild the derived projection
  from the log (as `connect` already does for format incompatibility), so a dirty
  store degrades gracefully instead of crashing checkout. This is the gap that
  turned the latent race into a session-long outage.

## 6. Done-when

Root cause documented (this note); a concurrent two-instance start against one
eventloom yields exactly one owner + proxies with **zero** store corruption
(a concurrency/stress test); divergent eventloom/store keys can no longer produce
two owners of one store; orphaned dead-session servers are reaped or cannot
accumulate; a corrupt store self-heals rather than crashing checkout; ruff + mypy
+ full suite green after each phase.

## 7. Open questions

1. Lock key: derive strictly from the resolved store path, or keep the eventloom
   key but **assert** `dirname(store) == eventloom/projections`? (Leaning: key on
   the store path — it's what actually needs protecting.)
2. CLI-while-no-owner: should a lone CLI `checkout` become a transient owner
   (claim → open → release) so it's still coordinated, or always graph-degrade?
   (Leaning: transient owner-claim so a solo CLI still gets full graph + can't
   race a second CLI.)
3. Reaping trigger: startup self-heal only, an explicit `doctor --repair`, or
   both? Liveness signal for "orphan" (no client) vs "legitimate idle owner".

## References

- `src/zaxy/mcp_runtime.py` — `EmbeddedMcpRuntimeCoordinator`, `try_claim_owner`,
  `repair_stale_runtime`, `EmbeddedMcpOwnerClaim`
- `src/zaxy/cli/serving.py:1219-1267` — `serve` owner/proxy decision + key resolution
- `src/zaxy/mcp_server.py:proxy_main` — the proxy client path
- `src/zaxy/embedded_graph_store.py:connect` — store open + (only) format-incompat recovery
- `src/zaxy/null_projection_store.py` — the graph-degraded primitive (#83)
- memory `incident-mcp-orphan-projection-corruption`, `perf-cold-checkout`
