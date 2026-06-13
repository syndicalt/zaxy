# Zaxy ↔ Kuzu Usage Audit (2.3 backend-continuity input)

Date: 2026-06-11. Repo: `/home/cheapseatsecon/Projects/Personal/zaxy` (read-only audit).
Context: Kuzu 0.11.3 is the final upstream release; repo archived, docs offline
(`docs/superpowers/specs/2026-06-11-zaxy-2-2-ann-engineering-plan.md:48-51`). Evaluating
LadybugDB or alternatives is formally on the 2.3 agenda (same spec, lines 214, 273-276).

All `Lnnn` references without a path are into `src/zaxy/embedded_graph_store.py` (2,660 LOC,
the only module that *executes* Kuzu statements besides `doctor.py`).

---

## 1. Complete Cypher/Kuzu API surface

`import kuzu` exists in exactly two places: `src/zaxy/embedded_graph_store.py:239-245`
(`find_spec` probe + `kuzu.Database`/`kuzu.Connection`) and `src/zaxy/doctor.py:362-367`
(read-only `kuzu.Database(..., read_only=True)` for a doctor check). The dependency is
`kuzu>=0.11.0` at `pyproject.toml:42`; in practice everything is engineered against the
frozen 0.11.3.

Every store statement funnels through one choke point, `_execute` (L1916-1931), which
regex-validates `$parameter` bindings (L107-110) because **unbound parameters segfault
Kuzu 0.11.3 rather than raising**. 48 `self._execute(` call sites total.

### 1.1 DDL — startup + ANN shadow rebuilds

| Statement | Where | Hotness | Portability |
|---|---|---|---|
| `CREATE NODE TABLE IF NOT EXISTS Entity(... PRIMARY KEY(node_key))` | L258-273 | startup (`init_schema`) | Kuzu DDL dialect; concept is a plain table → SQL-expressible. Neo4j has no table DDL (its `init_schema` differs, `graph.py:276`) |
| `CREATE NODE TABLE Event(...)`, `BenchmarkProjection(...)` | L274-298 | startup | same |
| `CREATE REL TABLE RELATES(FROM Entity TO Entity, ...)`, `NEXT_EVENT`, `PREVIOUS_EVENT` | L299-331 | startup | Kuzu-specific REL TABLE; SQL: edge table with two FKs |
| `CREATE NODE TABLE <shadow>(entity_row INT64, vec FLOAT[dim], PRIMARY KEY(entity_row))` | L1367-1375 | per ANN generation rebuild | **Kuzu-specific**: `FLOAT[dim]` fixed-size list column. No SQL/Neo4j equivalent without a vector extension |

Schema observations: all relationship metadata is scalar columns + JSON-in-string
(`properties_json`, `evidence_json` — e.g. L265, L312). Embeddings are **not** typed columns;
they live inside `properties_json` (parsed by `_embedding_vector`, L2634). The only typed
vector column is the ANN shadow table.

### 1.2 Transactions

`BEGIN TRANSACTION` / `COMMIT` / `ROLLBACK` as literal statements, only for bulk Eventloom
replay: `begin/commit/rollback_bulk_projection` (L392-419), driven via `getattr` feature
detection from `src/zaxy/cli/serving.py:1342-1356`. Everything else relies on Kuzu's implicit
auto-commit per statement. Portability: trivial in SQL; Neo4j would use driver tx API.

### 1.3 DML (write path)

| Family | Where | Hotness | Portability |
|---|---|---|---|
| `MERGE (ev:Event {event_key}) SET ...` | L425-443 | **per appended event** | standard Cypher MERGE; SQL UPSERT |
| Event chain `MATCH (prev),(current) ... MERGE (prev)-[:NEXT_EVENT]->(current)` + `PREVIOUS_EVENT` | L445-459 | per event with prev_hash | standard / SQL insert |
| Close superseded entity `SET e.valid_to = $valid_to` | L472-481 | per changed entity | standard / SQL UPDATE |
| `MERGE (e:Entity {node_key}) SET ...` (new version row) | L482-506 | per changed entity | standard / SQL UPSERT |
| Edge upsert `MATCH ... MERGE (s)-[r:RELATES {relation_type}]->(t) SET ...` | L528-556, `_merge_relationship` L2106-2133 | per extracted edge; also during relationship version-copy | standard / SQL UPSERT on (src,dst,relation_type) |
| Relationship carry-forward reads + re-merge (`_copy_active_relationships_to_new_version`) | L1997-2089 | per superseded entity version | 2 single-hop reads + N merges; SQL joins |
| `invalidate_entity` SET valid_to | L1656-1671 | per invalidation call | SQL UPDATE |
| `retire_source_projections` scan + per-node SET (entity + incident edges) | L1683-1713 | per source retirement | SQL UPDATE |
| `re_embed_session` scan (`contains(e.properties_json, '"embedding"')`) + per-row SET | L1514-1556 | maintenance CLI (`zaxy memory re-embed`, `cli/workspace.py:887-905`) | `contains()` → SQL LIKE |
| Benchmark marker `MERGE (p:BenchmarkProjection) SET ...` / presence probe | L353-390 | benchmark harness only | trivial |
| ANN shadow `UNWIND $rows AS row CREATE (:<table> {...})` batched 1024 | L1446-1457 | ANN delta inserts + pyarrow-less fallback | standard Cypher; SQL executemany |
| **`COPY <table> FROM '<parquet path>'`** | L1429 | per full ANN generation rebuild | **Kuzu-specific** bulk loader. Note: must round-trip through a parquet *tempfile* — in-memory Arrow `COPY FROM` **segfaults 0.11.3** (L1394-1431) |
| `MATCH (n:<old_gen>) DETACH DELETE n` | L1390 | per generation swap | standard; needed only because `DROP_VECTOR_INDEX` leaves un-checkpointed state (kuzu#6040) and `DROP TABLE` is binder-rejected while indexed (L1316-1325) — generation tables are emptied, never dropped |

### 1.4 Read queries (MATCH patterns)

**Every MATCH in the codebase is depth ≤ 1.** Patterns are node scans
(`MATCH (e:Entity) WHERE ...`) or single-hop edges
(`MATCH (source:Entity)-[r:RELATES]->(target:Entity) WHERE ...`). There are **no
variable-length paths, no shortest-path, no graph algorithms in Cypher**. Aggregation is
limited to `count(r)` (L988, L1771, L1781) plus `ORDER BY`/`LIMIT` (L1739, L1958-1959).
Multi-hop traversal is done by Python BFS over in-memory adjacency (see §2).

| Query | Where | Hotness | Portability |
|---|---|---|---|
| Active-entity scan (`valid_to IS NULL`) | `_current_entities` L644-652 | once per cache (re)build after a projection change | trivial SQL |
| Temporal-point entity scan (`valid_from <= $t AND (valid_to IS NULL OR $t < valid_to)`) | `_temporal_entities` L675-685 | once per (session, temporal_point), cached | trivial SQL range predicate |
| Active edge scan (full 21-column row pairs) | `_build_traversal_index` L886-957 | once per traversal-cache build | single-hop join → SQL |
| Temporal edge scan | L922-957 | per (session, point) cache build | SQL |
| **Causal edge scan** (`r.relation_type STARTS WITH 'causal_'`) | `_causal_edge_rows` L814-868 | **per `search_causal_neighbors` call — NOT cached** | `STARTS WITH` → SQL LIKE 'causal\_%' |
| Edge-count existence probe | `has_traversal_edges` L984-992 | only when traversal cache cold | SQL COUNT |
| Adjacency snapshot node + edge key scans | `_build_adjacency_snapshot` L1020-1054 | once per session per projection change (cached, L1013-1018) | SQL |
| Active-entity point lookup (`ORDER BY valid_from DESC LIMIT 1`) | `_active_entity_state` L1950-1966 | per uncached (session,type,name) during upsert; bulk preload variant L1981-1988 | SQL |
| Event projection integrity (`ORDER BY e.seq`, 2× `count(r)`) | L1733-1785 | per status/doctor inspection | SQL |
| Inferred-edge audit scan (`r.inferred = true`) | L1824-1838 | per audit inspection | SQL |

### 1.5 CALL functions — all Kuzu-specific, all ANN-scoped

| Call | Where | Hotness |
|---|---|---|
| `CALL QUERY_VECTOR_INDEX('<table>','<index>', $query_vector, $k, efs := N) RETURN node.entity_row` | L1117-1121 | per vector query **only when ANN engaged** (see §2) |
| `CALL CREATE_VECTOR_INDEX('<table>','<index>','vec', metric := 'cosine')` | L1474-1476 | once per shadow generation |
| `CALL SHOW_TABLES() RETURN name` | L1461 | per shadow rebuild (generation discovery) |
| `CALL SHOW_FUNCTIONS() WHERE name = 'CREATE_VECTOR_INDEX'` | L1486-1488 | once per process (capability probe) |

No `PROJECT_GRAPH` (deliberately avoided — per-scope dedicated tables instead, L1101-1110),
no `THREADS` (considered in the 2.2 spec, not adopted).

### 1.6 Kuzu-0.11.3-specific engineering that wouldn't port (or wouldn't need to)

1. Unbound-parameter segfault guard — L107-110, L1916-1931.
2. In-memory-Arrow `COPY FROM` segfault → parquet tempfile round-trip — L1394-1431.
3. `DROP_VECTOR_INDEX`/`DROP TABLE` unusable → append-only generation tables emptied via
   `DETACH DELETE` — L1310-1392.
4. Live-index delete+reinsert silently corrupts searches → insert-only delta policy with
   content digests (`_ANN_DELTA_REBUILD_FRACTION`, L94-105, L1342-1360).
5. `FLOAT[dim]` fixed-size-list column type — L1371.
6. Lock-contention error sniffing on the `.kuzu` path string for checkout fallback —
   `src/zaxy/cli/workspace.py:325-336`, used at `src/zaxy/cli/serving.py:374-377`.

### 1.7 Kuzu touches outside the store

- `src/zaxy/doctor.py:354-394` — opens the DB read-only with raw `kuzu` and runs one
  active-entity scan to sample embedding versions (maintenance-time).
- `src/zaxy/dashboard.py:929-1201` — `EmbeddedDashboardGraphProvider` constructs an
  `EmbeddedGraphStore` (L933-936) but then **pierces the abstraction** via
  `self._store._require_connection()` to run 6 raw queries (L1007-1034, L1022-1034,
  L1060-1083, L1115-1126, L1151-1163). All are portable single-hop/scan Cypher with
  `LIMIT`; nothing Kuzu-specific in the query text.
- `src/zaxy/cli/workspace.py:894` — re-embed CLI constructs the store directly (via the
  patchable seam `src/zaxy/cli/runtime.py:53-59`).
- Path-string-only references (`embedded.kuzu` default path): `config.py:107`,
  `local_profile.py:24,42`, `core.py:356`, `mcp_server.py:1395`, `onboarding.py:1103`,
  `integrations.py:674`, `release.py:1330`, `associative_memory.py:1183`,
  `dashboard.py:1223`, `cli/serving.py:1223-1224`. Naming only, no API coupling.

---

## 2. Load-bearing vs incidental: what actually runs in Kuzu at request time

The 2.2 architecture is **read-through caches built from one-shot Kuzu scans, with all
retrieval compute in Python/numpy**:

- **Exact search**: in-memory dict lookup (`search_exact` L562-574, lookup built once per
  cache, L657-668). Kuzu: one scan per cache build.
- **Keyword search**: pure-Python BM25 with postings, IDF, candidate-budgeting
  (`search_keyword` L576-611, helpers L2328-2462). Kuzu: zero at query time.
- **Traversal**: Python BFS (depth clamped 1–5, L717) over `_TraversalIndex` adjacency
  (L708-749) built from **one single-hop edge scan** (L886-957) and cached per session
  (L870-884). Kuzu never executes a multi-hop query.
- **Graph walk (personalized PageRank)**: `fetch_adjacency` returns a content-hash-signed
  `AdjacencySnapshot` (L994-1054), cached per session; PageRank runs in numpy
  (`graph_walk.py:144+`), walk results cached on snapshot signature
  (`query.py:640-700`).
- **Vector search** (L1056-1091): three strategies, one shared float64 numpy rerank.
  - Dense exact: numpy matmul (L2577-2606). Default path.
  - int8 (opt-in `VECTOR_QUANTIZATION=int8`): numpy int8 dot products (L2488-2508).
  - Kuzu HNSW: engages **only** when dimension ≤ `vector_ann_max_dimension` (default
    **64**, `config.py:421-432`) **and** (count ≥ `vector_ann_threshold` (default
    **100,000**, `config.py:397-417`) or float64 matrix > 256 MiB byte budget) —
    `_ann_engagement_reason` L1159-1204. Real embedding providers ship 256–3072 dims, so
    **on default config with real embeddings the HNSW path never engages**; it exists for
    the lane-proven d64/10^5 envelope. Even there it only does candidate selection
    (`QUERY_VECTOR_INDEX`, L1117) — ordering is numpy rerank (L2511-2574). And per the
    config's own evidence comments (`config.py:390-396`), ANN p50 at the only proven
    envelope was a wash vs the exact matrix (24.17 vs 24.20 ms; 26.67 vs 30.82 ms) — the
    win was resident bytes (0 vs 51.2 MB), not latency.
- **Feeling-of-knowing index**: `active_entity_names` served from the entity cache
  (L629-638; consumed via getattr at `mcp_server.py:2768-2776`).

**What remains genuinely Kuzu-resident:**

1. **Durable storage** of the projection (Entity/Event/RELATES bitemporal rows) and its
   crash-safe write path — `upsert_extraction` runs ~3–8 statements per appended event
   (L421-560). This is the largest real dependency.
2. **Cold-start / cache-rebuild scans** — every projection change invalidates the session's
   read caches (L1622-1629), so the next read of each kind costs one Kuzu scan;
   `warm_session` (L333-339) front-loads them.
3. **`search_causal_neighbors`** — the only retrieval path that hits Kuzu **on every call**
   (`_causal_edge_rows` is uncached, L765 + L814-868); BFS itself is still Python.
4. **Per-(session, temporal_point) reads** — cached but unbounded key space; first query at
   each new temporal point is a Kuzu scan.
5. **HNSW candidate selection** in the narrow d≤64 / ≥10^5 envelope (plus shadow-table
   rebuild machinery, ~600 LOC: L92-110, L143-177, L1093-1122, L1288-1493, L2465-2485).
6. **Maintenance/ops surfaces**: doctor sampling (raw kuzu), dashboard provider raw queries,
   re-embed scan/update, bulk-replay transactions, benchmark markers, checkout
   lock-contention fallback (a Kuzu file-lock behavior, `cli/workspace.py:325-336`).

**Quantified**: of the five retrieval families (exact, keyword, traversal/walk, vector,
causal), four execute **zero** Kuzu statements on a warm cache under default config; the
fifth (causal) executes one single-hop scan per call. Steady-state request-time compute is
≈100% Python/numpy; Kuzu's request-time role is storage, first-touch scans, causal scans,
and the per-event write path.

---

## 3. ProjectionStore protocol and conformance state

### 3.1 The contract

`src/zaxy/projection.py:17-135` — a structural `Protocol` with **14 methods**: `connect`,
`close`, `init_schema`, `upsert_extraction`, `search_exact`, `search_keyword`,
`search_traversal`, `search_causal_neighbors`, `has_traversal_edges`, `search_vector`,
`invalidate_entity`, `retire_source_projections`, `inspect_event_projection_status`,
`inspect_inferred_edge_status`.

A second protocol, `AdjacencyProvider` (`graph_walk.py:130-141`), defines
`fetch_adjacency`; its docstring promises "Embedded (Kuzu), Neo4j, and Postgres
implementations land in the backend wave" — **only embedded has landed**
(grep: `fetch_adjacency` exists in no other backend).

### 3.2 Backend implementations

Constructed in `src/zaxy/projection_backends.py:35-66` (embedded | neo4j | pggraph | latticedb).

| Capability | embedded (`embedded_graph_store.py`) | neo4j (`graph.py:209+`) | pggraph (`pggraph_store.py:178+`) | latticedb (`latticedb_store.py:32+`) |
|---|---|---|---|---|
| 14 protocol methods | yes | yes (explicit subclass, `graph.py:209`) | yes (structural) | yes (structural) |
| `fetch_adjacency` (graph walk) | **yes** L994 | no | no | no |
| `warm_session` | **yes** L333 | no | no | no |
| `active_entity_names` (FoK) | **yes** L629 | no | no | no |
| bulk projection tx (begin/commit/rollback) | **yes** L392-419 | no | no | no |
| `reset_benchmark_projection` | yes L341 | no | yes `pggraph_store.py:670` | yes `latticedb_store.py:71` |
| benchmark markers (present/mark) | yes L353-390 | no | no | no |
| `re_embed_session`, `embedding_version_counts` | **yes** L1495, L1571 | no | no | no |
| extra | — | `invalidate_edge` `graph.py:904` | — | — |
| `search_vector(embedding_version=…)` kwarg | **yes** L1056-1063 (protocol omits it) | no | no | no |

All extras are consumed through `getattr` feature detection — `core.py:1140-1148`
(warm_session), `query.py:655-659` (fetch_adjacency), `mcp_server.py:2768-2776`
(active_entity_names), `cli/serving.py:1331,1342-1344` (reset/bulk). So a non-embedded
default silently loses graph-walk ranking, FoK pre-check, warmup, and bulk replay — they
degrade rather than fail, which masks regressions.

### 3.3 Test coverage today

- `tests/test_projection.py` (231 lines): **typing + routing only** — a
  `FakeProjectionStore` pinned with `assert_type` (L103) and four `build_projection_store`
  routing tests (L120-231). No behavior.
- Per-backend bespoke suites, no shared scenarios:
  `tests/test_embedded_graph_store.py` (3,820 LOC, **102 tests**, ~38 vector/ANN-related);
  `tests/test_graph.py` (63 tests, but **Neo4j driver fully mocked** — file docstring
  line 3); `tests/test_pggraph_store.py` (27 tests); `tests/test_latticedb_store.py`
  (19 tests).
- `tests/test_backend_shootout.py` (6,310 LOC): a CLI benchmark harness whose tests are
  ~90% report/guardrail validation; default active backends are embedded+bm25 only
  (L115-123), latticedb candidate only when installed (L577). It measures quality/latency,
  not contract semantics.
- Scale lanes (`zaxy_benchmarks/vector_scale_lane.py`) construct `EmbeddedGraphStore`
  directly and poke privates (`_vector_index`, `_current_entity_index_cache` —
  L205-214, L378-402) — embedded-coupled, not backend-generic.

### 3.4 Distance from a conformance gate

There is **no backend-parametrized behavioral suite**. "Any backend passing this is a valid
default" is currently unachievable because the things that make embedded the default are
exactly the untested-in-common extras. Gap list:

1. **Parametrized parity suite over the 14 protocol methods**: shared fixture events →
   identical assertions on exact/keyword/traversal/causal/vector results, temporal-point
   visibility, entity versioning + relationship carry-forward (L1997-2089 semantics),
   invalidation, source retirement, both inspect statuses. Today each backend asserts its
   own bespoke subset.
2. **Capability-extras conformance**: fetch_adjacency snapshot semantics (undirected
   doubling, signature stability — L994-1054 docstring), warm_session, active_entity_names,
   bulk tx, reset — currently embedded-only, and the getattr seams mean absence is silent.
3. **Protocol drift repairs**: `embedding_version` kwarg on embedded `search_vector` not in
   the protocol; dashboard/doctor/re-embed bypass the protocol entirely (raw connection /
   raw kuzu / concrete class).
4. **Real-backend truth for neo4j**: unit suite is mocked; parity claims for the "control
   backend" rest on integration environments not in CI.
5. **Scale/latency lanes parametrized by backend** (the shootout harness is close — it
   already runs retrieval workloads per backend — but gates reports, not semantics).

Estimated effort to build the gate: one `tests/projection_conformance/` suite of ~40-60
scenarios parametrized over store factories (~1,500-2,500 LOC), much of it liftable from
existing `test_embedded_graph_store.py` scenario setups.

---

## 4. Migration blast radius

### Option A — LadybugDB drop-in

- **Files/LOC**: if the fork keeps the `kuzu` module name and storage format:
  `pyproject.toml:42` (1 line). If renamed: + `embedded_graph_store.py:239-244`
  (`find_spec("kuzu")`, `import kuzu`), `doctor.py:362`. Total ≤ 3 files, < 20 LOC.
- **Risk (qualitative)**: mechanically trivial; behaviorally medium. The store encodes four
  0.11.3 bug workarounds (§1.6 items 1–4) that must be re-verified against the fork —
  ideally some can be *deleted* if fixed upstream-of-fork. On-disk format compatibility of
  existing `embedded.kuzu` artifacts matters less than it looks: projections are derived
  state, rebuildable from the Eventloom log (`cli/serving.py` reprojection path), so the
  worst case is a one-time reproject.
- **Test leverage**: maximal — all 102 embedded tests, the vector-scale/graph-scale lanes,
  and the shootout harness run unchanged and directly certify the swap.

### Option B — second embedded backend implementing ProjectionStore

- **Surface to implement**: 14 protocol methods + the 8 load-bearing extras
  (fetch_adjacency, warm_session, active_entity_names, bulk trio, reset, re_embed +
  version counts) ≈ 22 methods; plus a dashboard provider counterpart (~300 LOC,
  `dashboard.py:929-1201`), doctor check, workspace lock-fallback analogue,
  `projection_backends.py` + config registration.
- **Files/LOC**: new store ~2,000–2,700 LOC (the embedded store is 2,660) + bespoke tests
  ~3,000+ LOC, because there is no conformance suite to inherit; ~8 integration files
  touched.
- **Risk**: high. The subtle semantics (bitemporal windows, entity version chains with
  relationship carry-forward, causal metadata contracts, adjacency signature stability)
  are specified only by the embedded implementation + its tests.
- **Test leverage**: low until the §3.4 gate exists. Strictly dominated by Option C unless
  the chosen engine brings something Zaxy doesn't already do in numpy — and §2 shows there
  is almost nothing left in that category.

### Option C — shrink the ask: Zaxy-owned store (SQLite/parquet) + existing numpy machinery

- **What actually needs reimplementation** (per §2): the durable row store and ~20 distinct
  statement shapes, all single-hop scans, equality/range/`IS NULL`/prefix predicates, and
  UPSERTs — every one trivially SQL (two tables + indexes on
  `(session_id, valid_to)`, `(session_id, name, entity_type, valid_to)`). The compute
  layer (BM25, dense/int8 vector, BFS, PageRank, FoK, rerank) transfers untouched.
- **ANN**: deletable. The HNSW envelope is d≤64 ∧ ≥10^5 where measured p50 was a wash vs
  exact (`config.py:390-396`); dropping it costs ~51 MB resident at that corner, and
  ~600 LOC of generation/digest/COPY machinery (L92-110, L143-177, L1093-1122,
  L1288-1493, L2465-2485) plus all four 0.11.3 workarounds disappear. (hnswlib/usearch is a
  later opt-in if a real need appears.)
- **Files/LOC**: rewrite the persistence half of `embedded_graph_store.py` (~1,000 LOC
  changed, ~600 deleted); `dashboard.py` provider queries (~100 LOC); `doctor.py:354-394`
  (~40); `cli/workspace.py:325-336` + `cli/serving.py:374-377` lock fallback (simplified —
  SQLite WAL handles the concurrent-checkout case the fallback exists for); path-default
  strings (~10 files, cosmetic). Drops the `kuzu` C++ wheel from `dependencies`.
- **Risk**: medium. Concurrency/durability semantics move onto Zaxy (SQLite is a
  well-understood substrate; projections remain rebuildable from the log, capping
  worst-case). Biggest hazard is re-deriving the bitemporal/version-copy semantics —
  mitigated by keeping the class and its public behavior identical.
- **Test leverage**: high. Most of the 102 embedded tests assert public behavior
  (project → search → invalidate) and port as-is; the ~38 vector tests keep dense/int8 and
  retire the ANN-shadow subset. The conformance suite from §3.4 doubles as the migration
  gate.

### Summary table

| | A: LadybugDB drop-in | B: new ProjectionStore backend | C: Zaxy-owned SQLite store |
|---|---|---|---|
| Files touched | 1–3 | ~10 + new module | ~6 + 1 rewritten module |
| LOC | < 20 | ~5,000–6,000 (store + tests) | ~1,200 changed / ~600 deleted |
| Kuzu workarounds | re-verify 4 | gone, new engine's quirks instead | gone |
| C++ dep | keeps (fork) | depends | removed |
| Risk | low-mech / med-behavioral | high | medium |
| Test leverage | full suite as-is | near zero today | most of suite + conformance gate |
| Strategic position | bridge; still hostage to fork health | dominated | terminal; owns the default |

---

## 5. Architectural read

The codebase has already voted. 2.2 moved every retrieval computation that matters into
Python/numpy behind per-session caches; Kuzu was left holding (1) durable bitemporal rows,
(2) single-hop scans that hydrate caches, (3) a per-event MERGE write path, and (4) an HNSW
index that the default configuration prevents real embedding workloads from ever reaching.
No query in the repo uses a capability that distinguishes a graph database from a relational
table with two indexes — the deepest Cypher pattern is one hop, and the store's own design
(single `_execute` choke point, no Cypher passthrough in the public surface, derived-state
projections rebuildable from the event log) makes the engine swappable by construction.

Recommended 2.3 shape: **A as the immediate continuity move** (one-line dependency change,
full test-suite certification, contingent on verifying the four 0.11.3 workarounds against
the fork), **C as the destination** the code's structure favors — it deletes the frozen-
runtime risk class entirely rather than transferring it to a young fork, and it removes the
last C++ wheel from the default install. **B is dominated** in every column. In all cases,
the first artifact to build is the backend-parametrized conformance suite (§3.4): it is the
cheapest way to convert each option's behavioral risk into mechanical verification, and it
is prerequisite to ever calling any backend "a valid default."
