# Zaxy 2.3 Backend Continuity Options — Research Report

- Date: 2026-06-11
- Scope: stay-and-contain on frozen Kuzu 0.11.3 vs. adopt the LadybugDB fork vs. stand up a second embedded backend
- Evidence basis: every claim carries a source URL or a locally-verified observation.
  Local verification environment: Linux x86_64, Python 3.13, test scripts at
  `/tmp/zaxy-23-research/defect_tests.py` and `/tmp/zaxy-23-research/defect_tests2.py`,
  run against `ladybug==0.17.1` and `kuzu==0.11.3` in dedicated venvs
  (`/tmp/zaxy-23-research/venv17`, `/tmp/zaxy-23-research/venvk`).
- Defect list under audit (from `docs/superpowers/specs/2026-06-11-zaxy-2-2-ann-engineering-plan.md`
  and `docs/research/ann-engineering-2026-06.md`):
  1. kuzu#5965 — `SET` on indexed columns rejected
  2. kuzu#6040 — `DROP_VECTOR_INDEX` metadata corruption
  3. kuzu#6047 — sequential HNSW load can block DB open
  4. `COPY FROM` in-memory Arrow with fixed-size-list column segfaults
  5. unbound `$parameter` segfaults instead of raising
  6. silent search breakage after in-place mutation under a live HNSW index

---

## 1. LadybugDB fork — in depth

### 1.1 Identity, history, governance

- Repo: <https://github.com/LadybugDB/ladybug>. Created **2025-10-07** — three days before
  the Kuzu archive (kuzudb/kuzu `archived: true`, last push 2025-10-10; verified via GitHub API).
  Not archived, MIT license, 1,277 stars, 65 open issues, last push **2026-06-11** (today).
  (GitHub API, verified locally.)
- Founder/maintainer: **Arun Sharma** (`adsharma`, ex-Facebook/ex-Google), GitHub company field
  "Ladybug Memory, Inc" — there is a corporate entity behind it, announced as "a community-driven
  fork of the Kuzu repo but with a vision to be a full one-to-one replacement of Kuzu"
  (The Register: <https://www.theregister.com/2025/10/14/kuzudb_abandoned/>;
  gdotv weekly: <https://gdotv.com/blog/weekly-edge-kuzu-forks-duckdb-graph-cypher-24-october-2025/>).
  The original Kuzu team is **not** involved post-fork (their names appear only in inherited
  pre-fork commit history).
- **Bus factor — the weakest signal.** Of the last 200 commits: `adsharma` 162, `aheev` 34,
  all others ≤ 2 (GitHub API, verified). Every release since v0.12.0 was cut by `adsharma`
  (or the actions bot). This is a one-engineer project with one regular collaborator.
  Counterweights: ~monthly releases for 8 consecutive months, same-week responses on issues
  (e.g. #351, #452 threads), CodeQL workflow and SECURITY.md added (PRs #397, #385 in
  v0.15.4.2 release notes), nightly dev wheels on PyPI. Active, but if `adsharma` stops,
  the project stops.

### 1.2 Releases and wheels

- GitHub releases: v0.12.0 (2025-11-04) → v0.13.x (Dec) → v0.14.x (Jan) → v0.15.x (Feb–Apr)
  → v0.16.x (Apr–May) → **v0.17.1 (2026-06-02)** — roughly monthly cadence
  (<https://github.com/LadybugDB/ladybug/releases>, verified via GitHub API).
- PyPI naming churn (a real migration hazard to pin around):
  - First published as **`real-ladybug`** (0.0.1.dev1 on 2025-10-30 → 0.15.3 on 2026-04-01),
    because `ladybug` was occupied (<https://pypi.org/project/real-ladybug/>).
  - Renamed to **`ladybug`** at v0.15.4.2 (PR #374 "rename: real_ladybug -> ladybug");
    PyPI `ladybug` now carries 0.15.3 → **0.17.1 (2026-06-02)** plus nightly
    `0.17.0.devYYYYMMDD` wheels (<https://pypi.org/project/ladybug/>, verified via PyPI JSON).
    `real-ladybug` is now stale at 0.15.3 — anyone pinning the old name silently froze in April.
- **Wheel matrix for `ladybug` 0.17.1 (verified from PyPI):** cp310–cp314 ×
  {macosx_13_0_arm64, macosx_13_0_x86_64, manylinux_2_26_aarch64, manylinux_2_27_x86_64,
  musllinux_1_2 (both arches), win_amd64, win_arm64} + sdist. That is **strictly broader**
  than frozen kuzu 0.11.3 (adds cp314 on macOS/Windows, musllinux, win_arm64).
  Python 3.14 support landed in PR #221 (v0.15.0 release notes). One regression: macOS
  floor moved from 11.0 (kuzu) to 13.0 (ladybug).
- License: MIT (GitHub API + PyPI metadata, verified).

### 1.3 API compatibility posture

- **Drop-in at the Python API level, different import name.** `ladybug` 0.17.1 exposes
  `Database`, `Connection`, `AsyncConnection`, `PreparedStatement`, `QueryResult`,
  `torch_geometric_result_converter` — the kuzu surface — plus new `ArrowQueryResult`/CSR
  APIs (verified locally by import). Issue reporters routinely write
  `import real_ladybug as kuzu` and run unmodified Kuzu code
  (e.g. <https://github.com/LadybugDB/ladybug/issues/377>).
- **Locally verified:** every Cypher/DDL form exercised in my defect battery ran unmodified
  on ladybug 0.17.1 — `CREATE NODE TABLE`, `CREATE REL TABLE`, `MATCH`, parameterized
  `CREATE`/`SET`/`DELETE`, `INSTALL vector; LOAD vector;`, `CALL CREATE_VECTOR_INDEX`,
  `CALL QUERY_VECTOR_INDEX`, `COPY ... FROM`, `CHECKPOINT`, `EXPORT/IMPORT DATABASE`.
- **Storage format is NOT compatible.** Ladybug 0.17.1 refuses a Kuzu-0.11.3-written
  database: `RuntimeError: ... The file is not a valid Lbug database file!`
  (locally verified). However the **export path works**: kuzu 0.11.3
  `EXPORT DATABASE (format='parquet')` → ladybug 0.17.1 `IMPORT DATABASE` round-tripped
  nodes, rels, and schema correctly (locally verified). For Zaxy this is nearly moot:
  the embedded store is a projection that can be rebuilt from the source of truth.
- Divergence is beginning (Arrow-backed tables, open-type graphs, multiple labels,
  extension-ABI break at 0.16.0 per release notes) — today it is a superset; in a year
  the dialects may drift. Pin exact versions.

### 1.4 The defect ledger — fix evidence, one by one

All "locally verified" rows were run against `ladybug==0.17.1` (scripts in
`/tmp/zaxy-23-research/`), each case in an isolated subprocess so a segfault would be
caught as a signal exit.

| # | Frozen-Kuzu defect | Status in ladybug 0.17.1 | Evidence |
|---|---|---|---|
| 1 | kuzu#5965: `SET` on indexed column rejected | **NOT fixed** (still rejected, cleanly) | Locally verified: `RuntimeError: Cannot set property vec in table embeddings because it is used in one or more indexes.` Open feature request: <https://github.com/LadybugDB/ladybug/issues/377>. (Cosmetic bug: error message names the wrong table.) Zaxy already builds indexes after bulk load, so this is non-blocking — and the failure mode is an exception, not corruption. |
| 2 | kuzu#6040: `DROP_VECTOR_INDEX` metadata corruption | **FIXED** | Fix commit `029d7aef25` (2025-10-28) "fix: DROP_VECTOR_INDEX persistence bug preventing property updates (#22)" <https://github.com/LadybugDB/ladybug/commit/029d7aef25>; tracking issue closed 2025-11-12: <https://github.com/LadybugDB/ladybug/issues/69>. **Locally verified:** create index → `DROP_VECTOR_INDEX` → `CHECKPOINT` → close → reopen → `SET` on the formerly-indexed column succeeds. Zaxy's drop-free generation-swap could be retired under ladybug (but see §1.5 before doing so). |
| 3 | kuzu#6047: sequential HNSW load can block DB open | **No direct evidence of a fix** | No matching issue/commit found in the fork (searched "HNSW load", "blocking database open", "6047"). Adjacent but distinct work exists (non-blocking concurrent checkpoint, PRs #332/#371, v0.15.4.2 notes). Treat as inherited until the Zaxy lane measures cold-start at 10^5. Zaxy's threshold-gated cold-start guard stays. |
| 4 | In-memory Arrow `COPY FROM` fixed-size-list segfault | **FIXED / not reproducible** | **Locally verified:** `COPY V FROM $tbl` where `$tbl` is an in-memory `pyarrow.Table` with a `pa.list_(pa.float32(), 4)` column succeeded (exit 0, rows queryable) — the exact shape that segfaults kuzu 0.11.3 per the 2.2 E1 finding. No single fix commit identified; related closed fixes: param-binding FLOAT[N] segfault <https://github.com/LadybugDB/ladybug/issues/376> (closed 2026-04-12) and substantial Arrow-path work in 0.15–0.17. The parquet round-trip could be removed under ladybug after lane confirmation. |
| 5 | Unbound `$parameter` segfault | **Crash fixed; semantics now silent** | **Locally verified:** an unbound `$missing` no longer segfaults — it evaluates to `NULL` and the query runs "successfully". That converts a crash into a silent-wrong-answer hazard. **Zaxy's `_QUERY_PARAMETER_RE` choke point must stay** regardless of backend. |
| 6 | Silent search breakage after in-place mutation under live HNSW | **Substantially fixed; one residual hole** | Fix lineage: issue #67 "vector index: poor recall after many deletions" (closed 2025-11-13) <https://github.com/LadybugDB/ladybug/issues/67>, fixed by PR #68 "vector index: fix checkpoint corruption" (root cause: `OverflowFile::checkpoint()` page allocation) <https://github.com/LadybugDB/ladybug/issues/68>. **Locally verified (positive):** delete + reinsert of all 100 rows one-by-one under a live index, then `QUERY_VECTOR_INDEX` → 5/5 overlap with exact ground truth. **Locally verified (residual):** `MATCH (v:V) DELETE v` (delete-ALL in one statement) then reinsert → the index **permanently returns 0 rows** on 0.17.1. Zaxy never does delete-all under a live index on the active generation (deltas are pure extensions; rebuilds are fresh generations), so this is avoided by existing design — but it means "empty superseded generation tables" remains the right pattern even though dropping is now safe. |

Bonus row — the defect that motivated 2.3 in the first place:

| The G3 failure (HNSW recall collapse at high dim / large N) | **NOT addressed — by the maintainer's own statement** | "There have been no significant changes to the vector indexing code since [the fork]" — adsharma, 2026-03-31, in <https://github.com/LadybugDB/ladybug/issues/351> (that issue's ~18%-recall report was traced to a downstream Rust-crate packaging error, not core HNSW). Ladybug ships the same NaviX HNSW that Zaxy's lane measured at 0.60–0.63 recall@10 at 50k×1536. **Switching backends does not buy back high-dimension ANN; the numpy paths stay primary.** |

### 1.5 Counter-signals (be blunt)

- **Open issue #452** (2026-05-04, still open): `Database::~Database()` SIGSEGV during
  checkpoint flush plus per-write throughput collapse on a ~60K-node graph on 0.16.x —
  17K upserts unfinished after 13 minutes
  (<https://github.com/LadybugDB/ladybug/issues/452>). Maintainer triaged within days
  (binding lifetime bug; "LadybugDB isn't optimized for small writes" —
  <https://github.com/LadybugDB/ladybug/discussions/301>). Zaxy's COPY-based vector path
  aligns with the bulk-write posture, but any Zaxy sync path still doing row-at-a-time
  UNWIND mutation at 10^4–10^5 scale must be lane-tested on ladybug before adoption.
- Closed-but-recent segfault reports in the bindings (#380 checkpoint segfault, #390
  interpreter-shutdown segfault, #483 numpy `$param` AttributeError) show binding-layer
  churn. They get fixed quickly, but the layer Zaxy sits on is the one moving.
- v0.15.0 release notes: "fixing known vulnerabilities via third-party dependencies"
  (<https://github.com/LadybugDB/ladybug/releases/tag/v0.15.0>) — i.e., the inherited
  dependency tree had known vulns. Ladybug patched them; **frozen Kuzu still carries them**
  (see §4).
- One person cuts every release. Monthly cadence for 8 months is real but not a guarantee.

### 1.6 Migration blast radius for Zaxy (qualitative, locally grounded)

Small. `import kuzu` appears in exactly two files (`src/zaxy/embedded_graph_store.py:241`,
`src/zaxy/doctor.py:362`, both lazy imports — verified); the pin is `kuzu>=0.11.0` in
`pyproject.toml`. ~51 `execute()` call sites in the store run a dialect that executed
unmodified in my tests. No file-format migration: projections rebuild. The 2.2 defect
accommodations (parquet round-trip, generation swap, param choke point, cold-start guard)
can all be **kept** under ladybug at zero cost — retire them only as the lane proves each
one unnecessary.

---

## 2. Other Kuzu forks and continuations

| Fork | Status | Verdict |
|---|---|---|
| **ryugraph** (predictable-labs) <https://github.com/predictable-labs/ryugraph> | 137 stars; last push **2026-01-20**; PyPI `ryugraph` 25.9.2 last published 2025-12-06 (verified) | Stalled ~5 months. Early energy (the LadybugDB DROP_VECTOR_INDEX fix arrived "via ryugraph", per issue #69), now apparently abandoned. Not a candidate. |
| **Vela-Engineering/kuzu** <https://github.com/Vela-Engineering/kuzu> | Active (pushed 2026-06-07); tagged `v0.12.0-vela.*` releases (2026-06-07); claims concurrent multi-writer support; MIT | Interesting engineering, but **no PyPI distribution** (verified: no `vela-kuzu`/`kuzu-vela` packages), 37 stars, single-company fork for their own agent-memory product. Watch, don't adopt. |
| **Bighorn** (Kineviz) <https://github.com/Kineviz/bighorn> | 130 stars; last push **2025-10-11** — the day after the archive; not on PyPI (verified) | Dead on arrival. |

LadybugDB is the only fork with releases, wheels, users, and a defect-fix record.
(Fork landscape context: <https://gdotv.com/blog/weekly-edge-kuzu-forks-duckdb-graph-cypher-24-october-2025/>.)

---

## 3. Alternative embedded backends (scoped to Zaxy's profile)

Profile assumed: property-graph entities with temporal validity windows, typed
relationships, traversal queries, exact + vector search — where **2.2 proved the numpy
paths beat Kuzu HNSW at every measured scale**, so the vector workload does **not**
need to live in the graph engine.

### 3.1 DuckDB (+ property-graph approaches)

- Maintenance: excellent. 1.5.3 on PyPI 2026-05-20, cp310–cp314 wheels, MIT,
  `requires_python >=3.10` (PyPI JSON, verified). Backed by DuckDB Labs + the independent
  DuckDB Foundation.
- Embedded story: first-class, in-process, no sidecar.
- Graph-query fit: the catch. Native option is recursive CTEs (SQL). The property-graph
  option, **DuckPGQ** (SQL/PGQ), is a CWI research community extension —
  <https://github.com/cwida/duckpgq-extension>: 416 stars, MIT, last commit 2026-03-02,
  last repo push 2026-04-09, no formal releases (verified via GitHub API). Adopting it
  would swap one weak-maintenance dependency for another, now in the query dialect itself.
- Migration blast radius: **high** — every Cypher call site rewritten to SQL(/PGQ);
  traversals and the temporal-window predicates redesigned. Vector: DuckDB VSS exists but
  is unnecessary given the numpy paths.

### 3.2 SQLite (stdlib `sqlite3`, JSON + recursive CTEs as graph substrate)

- Maintenance: the best on Earth — public-domain, supported "through 2050"
  (<https://www.sqlite.org/lts.html>), ships inside CPython, **zero new dependencies**.
- Embedded story: definitional.
- Graph-query fit: adjacency tables + recursive CTEs (documented graph pattern:
  <https://www.sqlite.org/lang_with.html>). Typed relationships = rel tables; temporal
  validity windows = indexed columns (arguably a better fit in SQL than in Cypher).
  Loses Cypher expressiveness; fine if the audited traversal surface is shallow
  (single-/few-hop), painful if there are deep variable-length path queries.
- Migration blast radius: **high** (full store rewrite behind the projection contract),
  but it is the only option whose dependency risk is effectively zero forever.

### 3.3 LanceDB / lance-graph

- LanceDB: healthy — 0.33.0 on PyPI 2026-05-28, Apache-2.0, embedded, abi3 wheels
  (PyPI JSON, verified). But it is a vector/table store: **no graph queries at all** —
  wrong shape, and Zaxy doesn't even need its vector engine (numpy wins per 2.2 lanes).
- **lance-graph** (<https://github.com/lance-format/lance-graph>): "Run Graph Queries with
  Lance" — Apache-2.0, created 2025-09-29, pushed 2026-06-06, 154 stars (verified).
  Notable: ladybug contributor `aheev` also contributes here, and ex-Kuzu-DevRel
  prrao87 benchmarks "Kuzu, Ladybug and lance-graph" together
  (<https://github.com/prrao87/graph-benchmark>). The ecosystem's center of gravity may
  move here, but it is months old. Watch for 2.4+, not a 2.3 candidate.

### 3.4 Shrink the ask (Zaxy-owned projection store: SQLite/parquet + existing numpy vectors)

- Honest framing: 2.2 already moved the hard part out of the graph engine. The lane
  proved exact/int8 numpy superior to Kuzu HNSW at every measured scale, and ANN is
  gated to dim ≤ 64 (`docs/research/ann-engineering-2026-06.md` §6). What remains in
  Kuzu is entity/relationship storage + traversal — a workload SQLite handles.
- Maintenance signal: inherits SQLite's (§3.2). No C++ graph engine, no fork-watching,
  no wheel-matrix anxiety, ever.
- Blast radius: **the largest engineering effort** (a new store implementation + parity
  tests behind the projection contract), but the smallest permanent risk surface. The
  2.2 strategy-pipeline consolidation and the single-choke-point design make the store
  more swappable than it was a release ago.

---

## 4. Risk timeline: staying on frozen Kuzu 0.11.3

- **Python 3.13: covered.** cp313 (+cp313t linux) wheels exist on all platforms (PyPI, verified).
- **Python 3.14: already broken on macOS/Windows.** kuzu 0.11.3 ships cp314/cp314t wheels
  for **manylinux only** — there are **no macOS and no Windows cp314 wheels**
  (<https://pypi.org/project/kuzu/0.11.3/#files>, verified via PyPI JSON). A macOS-arm64
  developer on Python 3.14 cannot `pip install kuzu` today; they'd need a from-source
  build of an archived C++ tree (viability on current Xcode/clang: unverified, and nobody
  will fix it if it breaks).
- **Python 3.15: hard cliff, ~October 2026.** CPython 3.15.0 final is scheduled
  2026-10-01 (PEP 790: <https://peps.python.org/pep-0790/>). No 3.15 wheels will ever
  exist for kuzu; the sdist pins a pybind11 generation that predates 3.15. Zaxy declares
  `requires-python >=3.11` with no upper bound — the first user on 3.15 hits an
  uninstallable default backend in ~4 months.
- **Platform drift.** Wheels target macosx_11_0 and manylinux_2_26/2_27 — fine for
  existing platforms; the risk is new toolchains/SDKs breaking the *source* build, the
  only escape hatch an archived repo has. New platform targets (e.g. win_arm64, musl)
  will never appear (ladybug already ships both — §1.2).
- **Security posture.** kuzudb/kuzu has zero published security advisories (GitHub API,
  verified) — which means no process, not no bugs. Concretely: LadybugDB's v0.15.0
  release notes say it focused on "fixing known vulnerabilities via third-party
  dependencies" (<https://github.com/LadybugDB/ladybug/releases/tag/v0.15.0>) — those
  same vulnerable vendored deps are frozen forever in 0.11.3. Zaxy's exposure is
  partially contained (embedded, local-first, no network-facing query surface; the
  unbound-param and Arrow-segfault choke points already guard the known crash inputs),
  but "unmaintained C++ parsing attacker-influenceable strings" is a liability that
  only grows.
- Net: stay-and-contain is not a strategy, it is a countdown with a visible expiry
  (~Oct 2026, or the first macOS-3.14 user — i.e., now).

---

## 5. Decision-ready comparison and recommendation

| Option | Continuity | Defect posture | Vector fit | Blast radius | Existential risk |
|---|---|---|---|---|---|
| Stay-and-contain (kuzu 0.11.3) | None; expires ~Oct 2026 (3.15), already broken for macOS py3.14 | All 6 defects permanent (all contained by 2.2 engineering) | numpy primary (proven) | Zero | High and rising |
| **Adopt LadybugDB** | Monthly releases; broader wheel matrix incl. cp314 all-platform | #6040 fixed, Arrow-COPY fixed, param segfault fixed(→NULL), mutation breakage fixed (delete-all residual), #5965 unchanged, #6047 unproven | Same NaviX HNSW — G3 unchanged; numpy stays primary | Small (2 import sites, pin change, projection rebuild; dialect ran unmodified) | Bus factor ≈ 1 |
| Second backend: SQLite (shrink the ask) | Effectively infinite | Entire defect class eliminated | numpy primary (unchanged) | Large (new store impl) | Lowest possible |
| DuckDB (+DuckPGQ) | DuckDB excellent; DuckPGQ research-grade | n/a | numpy primary | Large (SQL rewrite) + weak link in dialect | Medium (DuckPGQ) |
| LanceDB / lance-graph | LanceDB healthy; lance-graph embryonic | n/a | Redundant with numpy | Large + graph gap | High (immaturity) |

### Recommendation (ranked)

1. **Adopt LadybugDB as the 2.3 default backend — confidence: medium-high.**
   It is the only continuation with releases, full-matrix wheels (incl. the cp314
   macOS/Windows gap that frozen Kuzu will never close), a verified drop-in API, a
   working EXPORT→IMPORT migration path, and commit-level fixes for three of the six
   documented defects (plus a crash→exception downgrade on a fourth) — all re-verified
   locally against 0.17.1 in this audit. Conditions that should be non-negotiable:
   exact-version pin (`ladybug==`, never the stale `real-ladybug` name); **keep every
   2.2 defect accommodation** (param choke point — unbound params now silently NULL;
   generation-swap empties; cold-start guard — #6047 unproven) and retire each only on
   lane evidence; re-run the full vector-scale + graph-scale lanes on ladybug as the
   formal acceptance bar (the open #452 write-throughput report at exactly Zaxy-relevant
   scale is the thing the lane must clear); numpy paths remain the primary vector
   machinery — the fork does not change NaviX and does not reopen G3.
   The bus-factor-of-one is real and is the reason for item 2.

2. **Start the shrink-the-ask design study in parallel (2.3 research, 2.4+ delivery) —
   confidence: high that it removes the risk class; medium on cost.** Ladybug fixes the
   continuity emergency, not the structural one (Zaxy's default path depends on whether
   one person keeps merging). A Zaxy-owned SQLite/parquet projection store + the existing
   numpy vector machinery is the only option whose maintenance risk goes to ~zero, and
   2.2's strategy-pipeline consolidation already shrank the surface a second backend must
   implement. The pending Cypher-surface audit decides feasibility; if the traversal
   surface is shallow, this should become the long-term default.

3. **Stay-and-contain — acceptable only as the fallback if the ladybug lane runs fail —
   confidence: high in the assessment.** It works today because 2.2 engineered around
   everything, but it has a dated expiry: no macOS/Windows wheels on Python 3.14 *now*,
   no Python 3.15 path in ~4 months, and a frozen vendored-dependency tree with known
   (since-patched-in-ladybug) vulnerabilities. If the lane rejects ladybug, pin hard,
   add an upper Python bound, and accelerate option 2.

4. **DuckDB / LanceDB as graph substrate — not recommended for 2.3 — confidence: high.**
   DuckDB itself is the healthiest dependency in this report, but the property-graph
   story (DuckPGQ) is research-grade with no releases and a 3-month-old last commit;
   lance-graph is eight months old. Either would trade a verified small migration for a
   large rewrite onto a weaker link. Reassess lance-graph at 2.4+.
