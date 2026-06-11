# Embeddings

Embeddings are optional but important for semantic retrieval. Zaxy supports a
deterministic local hash provider and an OpenAI-compatible hosted provider. Both
produce fixed-size vectors attached to extracted entities and query text, then
selected projection backend vector search participates in result fusion.

The hash provider is designed for tests, offline development, and deterministic
behavior. It does not provide high-quality semantic meaning, but it allows vector
code paths to run without network access or secrets. This is valuable for CI and
for local contributors who only need to verify mechanics.

For the local-first profile:

```bash
zaxy local-profile
zaxy local-profile --output .env.local
zaxy local-profile --projection-backend embedded --output .env.local
zaxy local-profile --check
```

The generated profile uses `PROJECTION_BACKEND=embedded`,
`EMBEDDING_PROVIDER=hash`, `RERANKER_PROVIDER=lexical`, and sidecar autostart
disabled. It intentionally omits hosted API key variables so the default local
path stays offline and deterministic. Use an explicit sidecar profile only when
you need a Neo4j or pgGraph comparison target.

The hosted provider is selected with:

```bash
EMBEDDING_ENABLED=true
EMBEDDING_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY_FILE=secrets/openai_api_key.txt
```

`OPENAI_BASE_URL` can point at any OpenAI-compatible embeddings endpoint. Keep
`EMBEDDING_DIMENSION` aligned with the model and the selected backend vector
index. If the dimension changes, rebuild the vector index and replay affected
events so entity vectors are regenerated consistently.

Secrets should be supplied through `OPENAI_API_KEY_FILE` in production. Direct
`OPENAI_API_KEY` is convenient for local testing but can leak through process
inspection or shell history. See [security.md](security.md) and
[configuration.md](configuration.md).

Embedding generation happens after extraction. The provider receives structured
entity text, not arbitrary raw payload dumps. This reduces the chance of
embedding secrets and improves result quality by keeping the vector input close
to the graph fact being stored.

Stored vectors in the embedded backend carry the producing provider's version
tag (for example `hash@<fingerprint>-dim1536`). Search never compares vectors
across version tags, so changing providers or models cannot silently return
garbage matches; `zaxy doctor` reports mixed-version corpora, and
`zaxy memory re-embed --session-id <session>` batch-migrates stale-version
vectors to the active provider without touching Eventloom. Scale engagement
is never silently lossy and follows a dimension-gated two-clause rule since
2.2: the embedded backend uses exact dense-matrix search unless the group's
vector dimension is at or below `VECTOR_ANN_MAX_DIMENSION` (default `64`)
and either **(a)** the (session, version) vector group reaches
`VECTOR_ANN_THRESHOLD` vectors (default `100000`, lowered from `1000000` on
gate-G4 evidence — two consecutive vector-scale lane passes at exactly 10^5
vectors/dimension 64 with recall@10 of 1.0 on both the strict and tie-aware
metrics, ANN p50 at-or-better than exact in-run, and resident bytes
improved; see
`docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json`/`-r2.json`), or
**(b)** the group's exact float64 matrix would exceed the 256 MiB vector
index cache byte budget (count × dimension × 8 bytes — the budget fits
524,288 rows at dimension 64, 87,381 at 384, and 21,845 at 1536; disable
this clause with `VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT=false`). The dimension
ceiling is the measured envelope of the G4 evidence: at dimension 1536 with
50k gaussian vectors the lane measured HNSW recall@10 of 0.6 at `efs`
400 (0.6344 on a rerun, and only 0.8438 at `efs` 800 with p50 rising past
exact) while the
exact matrix answered in 22ms p50 despite sitting 2.29x over
the byte budget
(`docs/research/artifacts/ann-2026-06/ann3-d1536-50k-gauss-crossover.json`)
— the cache eviction always keeps the newest matrix resident, so a single
over-budget scope is a cache of one, not a thrash. High-dimension corpora
therefore stay on exact float64 — or explicitly opted-in
`VECTOR_QUANTIZATION=int8`, which keeps its precedence below the count
threshold and is exempt from clause (b) — unless the ceiling is raised
explicitly with lane evidence. When ANN engages, the (session, version)
vector group syncs
into its own Kuzu-native HNSW shadow table that queries hit directly — no
per-query graph projection and no predicate scan; indexes are built and
loaded only on engagement, so default-path users below the rule pay no
cold-start cost. The HNSW query retrieves an
oversampled candidate set (`VECTOR_ANN_EFS`, default `400`, controls the
index's query-time candidate list) and the store reranks those candidates
with exact float64 scores from the resident entity vectors, so approximate
selection never decides the final ordering; results still report
`exact: false` because the candidate set is approximate. The `efs` default
matches the measured high-dimension requirement: candidate recall is
efs-bound around dimension 1536 (0.8531 recall@10 at `efs` 200 versus 0.9875
at 400 on a realistic distribution), and each step costs only a couple of
milliseconds at the corpus sizes where ANN engages. Set `VECTOR_ANN_EFS=800`
for maximum recall (see [configuration.md](configuration.md) for the
measured sweep).
`VECTOR_QUANTIZATION=int8` opts in to quantized storage that shares the same
exact-rerank pipeline over int8 dot-product candidates. See
[configuration.md](configuration.md) for the settings.

Shadow-table sync is delta-aware. When a projection change invalidates the
vector index, an unchanged vector corpus reuses the resident shadow
generation with no writes; a small append-only delta (up to 10% of the
resident rows, verified by content digest) inserts only the new rows into
the live HNSW index; anything larger rebuilds into a fresh generation table
via a bulk `COPY` load (with the HNSW index built after the load) and swaps
it in atomically. Full rebuilds therefore cost seconds, not minutes, at
100k-vector scale. pyarrow — normally present as a transitive dependency —
enables the fast `COPY` load; without it the build falls back to batched
inserts, which are correct but slower.

If hosted embedding calls fail, treat the event log as the recovery source. Fix
configuration, replay the Eventloom log, and rebuild graph projections. Do not
manually patch vectors in a projection backend unless you are doing a controlled
maintenance operation documented in [operations.md](operations.md).

Related pages: [retrieval.md](retrieval.md), [graph-schema.md](graph-schema.md),
[deployment.md](deployment.md), and [README.md](../README.md). The public site
summary is [site/index.html](../site/index.html).
