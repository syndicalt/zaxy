# Zaxy Codebase Review — Security, Performance, Maintainability

**Date:** 2026-06-09
**Version reviewed:** 2.0.0rc1 (branch `master`, commit `ae1164a`)
**Scope:** full repository — 106 source modules (~94k lines under `src/zaxy/`), tests,
deployment artifacts (`Dockerfile`, `docker-compose*.yml`, `scripts/`), and documentation.

## How this review was conducted

Four parallel passes over the codebase, each finding verified against the cited code
before inclusion:

1. **Security** — authentication and authorization (`mcp_server.py`, `security.py`,
   `config.py`), input validation, secrets handling, web surfaces (`dashboard.py`,
   `viewer.py`), and deployment posture.
2. **Performance** — event-log I/O, projection backends (embedded Kuzu, pgGraph,
   Neo4j), query/retrieval pipeline, pagination, and server handlers.
3. **Architecture & dead code** — module sizing, layering, import graph, duplication,
   unused modules.
4. **Style, typing, documentation** — tool configuration, lint/type-check results,
   docs drift.

Tooling runs performed during the review: `ruff check src/ tests/` (clean) and
`mypy src/zaxy` under `strict = true` (clean, all 106 files).

## Executive summary

Zaxy is in notably good shape on the dimensions that are hardest to retrofit:
**typing (mypy strict passes with zero issues), lint cleanliness, layering (no
circular imports, core never imports server/CLI), and the remote authentication
model (correctly pinned JWT validation, constant-time token comparison, rate
limiting, audit trail).** No injection-class vulnerabilities were found: there is no
`eval`/`exec`/`pickle`/`shell=True` anywhere in `src/`, all subprocess calls use
list-form arguments, and graph queries parameterize user input (the single
interpolated value — traversal depth — is range-validated to 1–5 first).

The most important problems found, in order:

1. **Performance: the append-only JSONL event log is treated as a random-access
   database.** Nearly every hot operation re-parses (and often re-hashes) the log
   from byte 0. Three compounding defects (`event.py:227`, `event.py:359`,
   `mcp_server.py:3247`) make per-operation cost O(total history) instead of
   O(new events) across the MCP server, checkout, and coordination paths.
2. **Security: two real web-surface flaws in the dashboard** — a stored-XSS gap in
   two `innerHTML` templates that skip the otherwise-consistent `escapeHtml`
   (`dashboard.py:2326-2329`), and CSRF-able state-changing POST endpoints with no
   auth or origin check (`dashboard.py:1238-1241`).
3. **Maintainability: god modules and ~20% benchmark code in the runtime package.**
   `retrieval_plan.py` is 9,285 lines, `synthesis.py` 7,407, `__main__.py` 6,687;
   benchmark/eval modules total ~18.6k lines shipped inside the production wheel.

A prioritized remediation roadmap is at the end of this document.

---

## 1. Security findings

### Severity: High

#### S1 — Stored XSS in dashboard sessions/events tables

**Where:** `src/zaxy/dashboard.py:2325-2330`

Every other dashboard template routes interpolated values through `escapeHtml`
(e.g. lines 2394, 2437, 2440, 2511), but the two templates for the sessions table
and recent-events table do not:

```javascript
document.getElementById("sessions-body").innerHTML = status.memory.sessions.map((session) => `
  <tr><td><code>${session.session_id}</code></td><td>${session.event_count}</td><td>${session.latest_type || ""}</td>...`
document.getElementById("events-body").innerHTML = events.events.map((event) => `
  <tr><td><code>${event.session_id}</code></td><td>${event.seq}</td><td>${event.type}</td><td>${event.actor}</td><td>${event.summary || ""}</td></tr>`
```

`event.type`, `event.actor`, and `event.summary` originate from event payloads —
which are written by captured tool calls, worker sessions, and ingested documents.
Any agent or ingested content that can append an event with HTML in those fields
executes script in the operator's browser when they open the dashboard. Because the
dashboard also exposes approval endpoints (see S2), XSS here escalates to silently
approving/merging coordination findings.

*Exploitability is tempered by the default `127.0.0.1` bind (`dashboard.py:39-40`),
but the threat model for this product is explicitly "multiple semi-trusted agents
writing to shared memory," so event fields must be treated as untrusted.*

**Fix:** wrap all interpolated fields in these two templates with `escapeHtml`,
identical to the surrounding templates. One-line-per-field change.

### Severity: Medium

#### S2 — CSRF on dashboard state-changing endpoints

**Where:** `src/zaxy/dashboard.py:1238-1244`, server at `dashboard.py:2614`

`POST /api/coordinate/review`, `POST /api/coordinate/review-finding`, and
`POST /api/coordinate/apply-approval` mutate coordination state (approve/reject
findings, apply approval packets). The dashboard's `ThreadingHTTPServer` has no
authentication, no CSRF token, and no `Origin`/`Host` header validation. A malicious
webpage open in the operator's browser can fire cross-origin POSTs at
`http://127.0.0.1:8765/api/coordinate/apply-approval` — the browser blocks reading
the *response*, but the request still executes. Review parameters are passed in the
query string (`dashboard.py:2468`), which makes forging them trivial.

**Fix (any of):** validate the `Origin`/`Host` headers against the configured bind
address; issue a per-process random token embedded in the dashboard HTML and require
it on POSTs; or require the MCP admin token for mutation endpoints.

#### S3 — Unauthenticated-by-default MCP transport + `0.0.0.0` Docker default

**Where:** `src/zaxy/mcp_server.py:2597`, `Dockerfile` final `CMD`

When neither `MCP_REMOTE_AUTH_TOKEN` nor the OIDC triplet is configured,
`MCPTransportAuth.authorize` accepts any request and scopes it to a
**client-chosen** session header (falling back to `"default"`). The Dockerfile's
default command is `serve --transport sse --host 0.0.0.0 --port 8080`. The
production guardrail is real — `config.py:456-476` hard-fails startup without
auth tokens **when `ZAXY_ENV=production`**, and `docker-compose.prod.yml:33` sets
it — but a bare `docker run` of the image (or any compose file that forgets
`ZAXY_ENV`) serves unauthenticated, network-reachable memory read/write with
client-selected session scope.

**Fix:** set `ENV ZAXY_ENV=production` in the Dockerfile (forcing explicit opt-out
for dev containers), or refuse to bind non-loopback hosts when no remote auth is
configured.

### Severity: Low

#### S4 — Admin token compared with `!=` instead of constant-time comparison

**Where:** `src/zaxy/mcp_server.py:2487-2489`

```python
if self._admin_token and arguments.get("admin_token") != self._admin_token:
    raise PermissionError("admin_token is required for this tool")
```

The bearer-token path correctly uses `hmac.compare_digest`
(`mcp_server.py:2591`); the admin-token gate for destructive/bulk-read tools does
not, leaving a (hard-to-exploit, but free-to-fix) timing oracle.

**Fix:** `hmac.compare_digest(str(arguments.get("admin_token") or ""), self._admin_token)`.

#### S5 — Dashboard has no authentication at all

**Where:** `src/zaxy/dashboard.py:39-40` (`host: str = "127.0.0.1"`, `port: 8765`)

The localhost-only default is a reasonable posture for a local ops tool, but
anything else on the same machine (other users on a shared host, containers with
host networking) can read full memory contents and drive approvals. Worth an
optional token and a documented warning in `docs/operations.md` against rebinding
to non-loopback interfaces.

#### S6 — Latent cache-key bug in `SessionManager.get`

**Where:** `src/zaxy/session.py:45-54`

```python
safe_id = validate_session_id(session_id)
if safe_id not in self._sessions:
    ...
    self._sessions[session_id] = Session(...)   # stores raw key
return self._sessions[safe_id]                  # reads validated key
```

Today `validate_session_id` returns its input unchanged when valid, so the keys
coincide — but if validation ever normalizes (case-folding, trimming), every `get`
raises `KeyError` or silently re-creates sessions. Store under `safe_id`.

### Security: what is done well

These are worth preserving as the codebase evolves:

- **Input validation is centralized and consistently applied**
  (`src/zaxy/security.py`): session-ID allowlist regex + resolved-path containment
  check (`eventlog_path`, lines 64-71) blocks path traversal; payload size cap
  (1 MiB), query length cap (4,096), result-limit cap (100), replay cap (1,000),
  and traversal-depth validation (1–5) *before* the only string interpolation into
  Cypher (`validate_traversal_depth`, line 185).
- **No dangerous primitives:** zero `eval`/`exec`/`pickle.load`/`yaml.load`(unsafe)/
  `shell=True`/`os.system` in `src/`; all subprocess invocations use list-form argv
  (`coordination_git.py:148`, `capture_manager.py:75`, `release.py`, `__main__.py:2535`).
- **JWT/OIDC validation is textbook** (`mcp_server.py:2603-2631`): algorithms pinned
  to `["RS256", "ES256"]`, `exp`/`iat`/`iss`/`aud` required, audience and issuer
  asserted, scope checked, session taken from a verified claim, JWKS key resolution.
- **Static bearer auth** uses `hmac.compare_digest` (`mcp_server.py:2591`); remote
  requests are rate-limited per session and audited, including denials
  (`RemoteRequestGuard`, `mcp_server.py:2642-2706`).
- **Secrets hygiene:** `_FILE`-suffixed env indirection for all sensitive settings
  (`config.py:440-454`); payload redaction by key pattern *and* value pattern
  (OpenAI/PyPI/AWS/JWT shapes) with sensitivity classification
  (`security.py:99-149`); `secrets/`, `.certs/`, `.env`, `.env.local` are all
  gitignored — only `.env.example` is tracked (verified against `git ls-files`).
- **Production startup validation** (`config.py:456-476`) rejects the default Neo4j
  password, non-TLS bolt without a CA cert, and missing admin/remote tokens.
- **Container hardening:** multi-stage build, non-root `zaxy` user, no secrets baked
  into the image, healthcheck; prod compose injects tokens via Docker secrets files.

---

## 2. Performance findings

The dominant theme: **the append-only JSONL log is re-read from byte 0 by nearly
every operation**, while its append-only nature makes tail reads, offset
checkpoints, and signature-keyed caches cheap to apply. The pattern is already
proven in-repo — `MemoryFabric._verbatim_index` caches by `(mtime_ns, size)`
(`core.py:1769-1778`, `core.py:4326-4332`) — it just isn't used everywhere.

### High impact

#### P1 — `EventLog.append_many` reads the entire file on every append

**Where:** `src/zaxy/event.py:227-237`

The append path opens the log in `a+`, then `fh.seek(0); fh.readlines()` and
JSON-parses the last line just to recover `seq`/`prev_hash`. Every append is O(n)
in file size; ingesting n events is O(n²) total bytes read. Hit by
`MemoryFabric.append` (`core.py:1142`), every `ingest_*` loop
(`core.py:1193-1229, 1350-1367`), and per-tool-call lifecycle capture (P3).

**Fix:** seek backwards from EOF to read only the last line, or cache
`(last_seq, last_hash, file_size)` per `EventLog` instance and revalidate via
`fstat` while holding the existing exclusive lock. Low effort.

#### P2 — `replay()` parses the log twice and SHA-256-verifies every event, in hot paths

**Where:** `src/zaxy/event.py:359-374` (calls `read_all()`, then `verify()` which
calls `read_all()` again and recomputes a per-event SHA-256, lines 315-357)

Hot callers that pay two full parses plus n hashes per call:

- `mcp_server.py:2433` — every `context_assemble` / `memory_checkout`;
- `mcp_server.py:2527` — `_resolve_checkout_ref("HEAD")` replays the whole log to
  read the **last** event;
- `core.py:1789, 2087, 2416`;
- `coordination.py:855, 897, 930, 975, 999, 1154, 1161, 1306, 1337` — the dashboard
  mission view (`dashboard.py:1467-1489`) triggers ~8–10 full replay+verify passes
  per page load.

At 100k events this is ~200k JSON parses + 100k SHA-256 hashes per checkout,
synchronously, inside async handlers — blocking every concurrent MCP/SSE client.

**Fix:** make integrity verification opt-in on `replay()` (or verify only the chain
tail since the last verified offset); cache parsed events per `EventLog` keyed by
`(st_mtime_ns, st_size)`; resolve `HEAD` from a tail read.

#### P3 — Per-tool-call lifecycle capture invalidates all projection read caches

**Where:** `src/zaxy/mcp_server.py:3247-3265` → `_append_lifecycle_event`
(`:1180-1196`); cache clear at `src/zaxy/embedded_graph_store.py:298`

Enabled by default, every MCP tool call — including pure reads like
`memory_query` — appends a `tool.call.completed` event (O(n) file read per P1) and
calls `upsert_extraction`, whose first action in the embedded backend is
`_clear_read_caches(session_id)`. Interleaved query/capture traffic therefore
rebuilds the keyword/vector/traversal/entity indexes from full Kuzu scans on
**every query**, defeating otherwise-good caching.

**Fix:** skip invalidation when the extraction projected no entities/edges
(lifecycle events typically project only an Event node); or batch lifecycle capture
behind a write-behind queue; or apply incremental cache updates.

#### P4 — Embedded vector search is brute-force pure-Python over dense vectors

**Where:** `src/zaxy/embedded_graph_store.py:854-929`; `_row_to_entity` at `:1483`

The sparse-postings design only helps for sparse vectors; the default
`HashEmbeddingProvider` and any sentence-transformers/OpenAI provider produce dense
1024–3072-dim vectors, so the query loop performs `dimensions × entities` Python
float multiply-adds (~15M ops at 10k entities). Index rebuild also JSON-parses every
entity's full embedding out of `properties_json` — and per P3, rebuilds happen
after every write.

**Fix:** store embeddings in a numpy matrix per session (one `argpartition`
cosine top-k is ~100× faster) or adopt Kuzu's native vector index; move embeddings
out of `properties_json` into a typed column.

#### P5 — Synchronous HTTP clients and `time.sleep` inside async paths

**Where:** `src/zaxy/query.py:208, 254, 306` (rerankers create sync `httpx.Client`
inside `async def rerank`); `src/zaxy/embedding.py:104, 121-143, 191`
(`_post_with_retries` calls `time.sleep` with up to 6 retries and 10s 429 backoff)

Called from `core._project_event` (`core.py:1156-1160`) and `retrieve`
(`core.py:1458`) — a rate-limited embedding endpoint can freeze the entire MCP
server event loop for ~60 seconds.

**Fix:** `httpx.AsyncClient` + `await asyncio.sleep`, or wrap provider calls in
`asyncio.to_thread`.

#### P6 — Dashboard opens a new backend per HTTP request

**Where:** `src/zaxy/dashboard.py:1422-1458` (`_checkout_body` constructs a new
`MemoryFabric` — new Kuzu `Database`/`Connection`, cold caches — under a fresh
`asyncio.run` per request); `dashboard.py:990-1026` and `:697-737`
(`connect()`/`close()` per method call); `:1491-1495` (new `CoordinationManager`
per request, compounding P2)

**Fix:** hold one connected store/fabric for the `DashboardApp` lifetime (with a
lock for the threaded HTTP server), or run the dashboard natively async.

### Medium impact

| # | Finding | Where | Fix sketch |
|---|---------|-------|------------|
| P7 | `len(eventlog.read_all()) + 1` parses the full log to mint an ID; `propose_consolidation_candidates` calls `read_all()` twice back-to-back | `coordination.py:745, 1085`; `core.py:1935, 1943` | tail-read last `seq` / reuse the first read |
| P8 | `refs.resolve()`/`list_refs()` scan the full refs log per call (on the checkout path) | `refs.py:73, 92` | in-memory `{name: ref}` cache keyed by file signature |
| P9 | `inspect_memory_status` does `read_all()` + `verify()` for **every** `*.jsonl`, on every `memory_capabilities`/`memory_bootstrap` call | `memory_status.py:282-298`, `capabilities.py:26` | cache by file signature; verify lazily |
| P10 | MCP server rebuilds the BM25 verbatim index per `memory_verbatim`/`context_assemble` call, bypassing core's existing signature-keyed cache | `mcp_server.py:1990, 2446-2452` | reuse `MemoryFabric._verbatim_index` |
| P11 | Causal-neighbor search loads *all* causal edges per call, BFS in Python, uncached (both embedded and pgGraph) | `embedded_graph_store.py:611-728`; `pggraph_store.py:524-598, 1036-1060` | cache adjacency like `_traversal_index`; push BFS into Postgres (recursive CTE) |
| P12 | pgGraph vector column is undimensioned (`vector`), so no HNSW/IVFFlat possible; `ORDER BY <=>` sequential-scans | `pggraph_store.py:27-95, 600-650` | type the column `vector(N)` + HNSW index |
| P13 | Embeddings computed one HTTP round-trip per entity, no batching, no cache across re-assertions | `embedding.py:320-339` | batch `input` lists; LRU by embedding text |
| P14 | Cursor pagination re-executes the full hybrid query (including paid rerankers) per page and slices | `core.py:1689-1740`; `mcp_server.py:1892-1919` | short-TTL ranked-result cache keyed by (query, session, profile) |
| P15 | `retire_source_projections` (embedded) full-scans all entities, JSON-parses each, then 2 statements per retired node | `embedded_graph_store.py:1008-1054` | project `source_path` to a real column (Neo4j backend already does — `graph.py:930-979`) |
| P16 | Dashboard/viewer/status event listings parse entire logs then slice the tail; `viewer.py` embeds *every* event into the HTML export unbounded | `dashboard.py:289-300`; `viewer.py:29-84`; `memory_status.py:122-154` | tail-read last N lines; add `--limit` to viewer |
| P17 | N+1 write statements per entity/edge in `upsert_extraction` outside bulk mode | `embedded_graph_store.py:337-431`; `graph.py:484-498` | `UNWIND $rows` / multi-row upserts |
| P18 | Sync file I/O + hashing directly inside async MCP handlers stalls all SSE clients | `mcp_server.py:1188, 2239, 2433, 2527` | `asyncio.to_thread`, or made moot by P1/P2 |

### Performance: what is done well

- MMR is properly bounded (pool ≤ 4×limit) with incremental max-similarity
  tracking and a shared token cache (`query.py:597-659, 1241-1289`).
- `VerbatimIndex` precomputes IDF/length norms, uses postings + `heapq.nlargest`
  (`verbatim.py:45-113`); core caches it by file signature.
- The embedded store has session-keyed caches for entity/keyword/vector/traversal
  lookups, `warm_session`, and bulk-projection transactions with one-shot state
  load (`embedded_graph_store.py:88-107, 206, 266-293, 1295-1313`).
- Neo4j schema ships correct constraints plus lookup, vector, and fulltext indexes
  (`schema.py:45-87`); pgGraph keyword search uses a GIN tsvector index.
- Query candidate budgets are capped (`MAX_QUERY_LIMIT`, `_candidate_limit` of 50).

---

## 3. Architecture and dead code

### A1 — God modules

| Module | Lines | Concern |
|--------|------:|---------|
| `retrieval_plan.py` | 9,285 | Largest module in the package — nearly 10% of all source |
| `synthesis.py` | 7,407 | |
| `__main__.py` | 6,687 | 72 CLI commands in one file |
| `harvey_lab_benchmark.py` | 4,694 | Benchmark code (see A2) |
| `core.py` | 4,356 | `MemoryFabric` orchestrates everything |
| `live_benchmark.py` | 4,079 | Benchmark code |
| `mcp_server.py` | 3,630 | Tool schemas + handlers + transport + auth in one file |
| `extract.py` | 3,583 | |
| `longmembench.py` | 3,319 | Benchmark code |

**Recommendation:** split along existing seams. `__main__.py` → one typer sub-app
module per command family (memory, coordinate, benchmark, release); `mcp_server.py`
→ tool schema definitions / handlers / transport+auth; `retrieval_plan.py` and
`synthesis.py` → stage-per-module packages. Mechanical refactors, protected by the
existing test suite (~100 test files) and strict mypy.

### A2 — ~20% of the shipped package is benchmark/eval code

Benchmark-named modules (`*benchmark*.py`, `longmembench.py`,
`rc_benchmark_freeze.py`, `external_validation.py`) total **18,587 lines** inside
`src/zaxy/` — code that installs into every user's environment via the
`zaxy-memory` wheel but exists to produce marketing/benchmark evidence
(`harvey_lab_benchmark`, `live_benchmark`, `causal_benchmark`,
`coordination_benchmark`, `consolidation_benchmark`, `purpose_benchmark`,
`reasoning_benchmark`, `statistical` benchmarks…).

**Recommendation:** extract to a `zaxy-benchmarks` companion package or a
non-packaged top-level `benchmarks/` tree (one already exists at the repo root),
keeping only the thin CLI shims. This shrinks the wheel, the import surface, and
the audit surface in one move.

### A3 — Dead / test-only modules

- `src/zaxy/coordinationbench_adapter.py` — imported by nothing in `src/` and no
  CLI wiring; referenced only by `tests/test_coordinationbench_adapter.py`.
- `src/zaxy/feature_evidence.py` — same pattern; only
  `tests/test_feature_evidence.py` references it.

**Recommendation:** delete them (with their tests), or wire them into the CLI if
they are meant to be user-facing. Code whose only consumer is its own test is
maintenance cost with no product value.

### A4 — Layering and imports: clean (positive)

- Core domain modules (`core.py`, `event.py`, `query.py`) never import server, CLI,
  or dashboard modules — verified by grep.
- Full-package import probe succeeds with no circular-import failures.
- `__main__.py` defers all `zaxy.*` imports into command bodies/`TYPE_CHECKING`,
  keeping CLI startup fast; `__init__.py` uses a lazy `_LAZY_EXPORTS` table with an
  explicit `__all__`.

### A5 — Test layout

The suite mirrors the source roughly one-to-one (~100 test files). Modules without
a same-named test file: `benchmark`, `coordination_git`, `coordination_templates`,
`evidence_candidates`, `external_validation`, `hooks`, `log`, `mcp_server`,
`packet_guidance`, `projection_backends`, `recall`, `release`, `retrieval_intent`,
`retrieval_profile`, `schema`. Several are covered indirectly (`mcp_server` via
`test_mcp_runtime.py` and `test_remote_security.py`; `recall` via query tests), but
`coordination_git` (runs subprocesses) and `hooks` deserve direct tests.

---

## 4. Style, typing, and documentation

### What the tooling run showed (positive)

- **`ruff check src/ tests/` — clean.** Rule sets enabled: E, F, I, N, W, UP, B,
  C4, SIM, with the formatter handling line length.
- **`mypy src/zaxy` under `strict = true` — zero issues across 106 files.** For a
  94k-line codebase this is exceptional and worth protecting in CI (CI workflows
  exist: `ci.yml`, `pages.yml`, `publish.yml`).
- Consistent `from __future__ import annotations`, `X | None` unions, pathlib
  usage, and Google-style docstrings in the modules sampled.

### ST1 — Docstring linting is configured but never runs

`pyproject.toml` sets `[tool.ruff.lint.pydocstyle] convention = "google"`, but the
`select` list does not include any `D` rules — so the convention setting is dead
configuration and docstring coverage/format is unenforced.

**Fix:** add `"D"` to `select` with pragmatic ignores (e.g. `D1` for tests via
per-file-ignores), or delete the pydocstyle block to stop implying enforcement.

### ST2 — Docs drift

- `docs/api.md` (line 8) still points readers to the **"v0.9 freeze-candidate"**
  classification in `api-inventory.md`, while the package is at **2.0.0rc1**. The
  inventory language predates two major versions.
- Recommend a docs sweep keyed to the 2.0.0 release: re-validate `api.md` /
  `api-inventory.md` against the actual `__all__` (`src/zaxy/__init__.py:86`) and
  the 72 CLI commands in `__main__.py`; `scripts/validate-docs.sh` exists and could
  gain an export-vs-inventory check.

### ST3 — No pre-commit configuration

CI runs the checks, but there is no `.pre-commit-config.yaml`, so contributors
discover ruff/mypy failures only after pushing. Adding ruff (check + format) and
mypy hooks would shorten the loop cheaply.

---

## 5. Prioritized remediation roadmap

Ordered by (risk × effort). The first block is all small, surgical changes.

### Now (small fixes, high value)

| # | Action | Refs | Effort |
|---|--------|------|--------|
| 1 | Add `escapeHtml` to the sessions/events dashboard templates | S1 | trivial |
| 2 | Add Origin/Host validation or a CSRF token to dashboard POST routes | S2 | small |
| 3 | Tail-read instead of full-file read in `EventLog.append_many` | P1 | small |
| 4 | Make `replay()` verification opt-in; resolve `HEAD` from tail | P2 | small-medium |
| 5 | Skip read-cache invalidation for no-op lifecycle extractions | P3 | small |
| 6 | `hmac.compare_digest` for the admin token; fix `session.py` cache key | S4, S6 | trivial |
| 7 | `ENV ZAXY_ENV=production` in the Dockerfile | S3 | trivial |

### Next (structural performance)

| # | Action | Refs | Effort |
|---|--------|------|--------|
| 8 | Signature-keyed parsed-event cache on `EventLog`; apply to refs/status/capabilities | P2, P7-P9 | medium |
| 9 | Async HTTP clients (or `to_thread`) for rerankers and embedding providers; batch embeddings | P5, P13 | small-medium |
| 10 | Persistent fabric/store in the dashboard; tail-read event listings | P6, P16 | small-medium |
| 11 | numpy (or Kuzu-native) vector index for the embedded store; embeddings out of `properties_json` | P4 | medium |
| 12 | pgvector: typed column + HNSW index; causal BFS in-database | P11, P12 | medium |

### Later (maintainability)

| # | Action | Refs | Effort |
|---|--------|------|--------|
| 13 | Extract benchmark/eval modules from the shipped wheel | A2 | medium |
| 14 | Split `__main__.py` and `mcp_server.py` along command-family/concern seams | A1 | medium |
| 15 | Delete or wire in `coordinationbench_adapter.py`, `feature_evidence.py` | A3 | trivial |
| 16 | Enable ruff `D` rules; pre-commit config; 2.0.0 docs sweep | ST1-ST3 | small |
| 17 | Decompose `retrieval_plan.py` / `synthesis.py` into stage packages | A1 | large |

---

## Appendix: verification notes

- Secrets exposure check: `git ls-files` confirms only `.env.example` is tracked;
  `secrets/`, `.certs/`, `.env`, `.env.local` are present locally but gitignored.
- Dangerous-primitive grep (`shell=True`, `os.system`, `eval(`, `exec(`,
  `pickle.load`, unsafe `yaml.load`): zero true hits in `src/` (matches were
  substrings of `*_eval(` function names).
- Import-graph probes and module-usage greps were run against `src/` and `tests/`;
  "dead module" claims mean no import from any other `src/` module *and* no CLI or
  entry-point wiring.
- Line counts via `wc -l`; function count via AST walk (3,707 functions in
  `src/zaxy/`).
