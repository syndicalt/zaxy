# Configuration

Zaxy configuration is centralized in `src/zaxy/config.py`. Settings load from
process environment variables, `.env`, defaults, and Docker/Kubernetes-style
secret files. Direct environment values win over `*_FILE` values. This keeps
development simple while allowing production deployments to avoid plaintext
secrets in environment dumps.

Projection settings are `PROJECTION_BACKEND`, `EMBEDDED_GRAPH_PATH`,
`LATTICEDB_PATH`, `PGGRAPH_DSN`, and the Neo4j settings used only when
`PROJECTION_BACKEND=neo4j`. The default backend is `embedded`, which stores the
repo-local LadybugDB projection at `.eventloom/projections/embedded.kuzu` and does
not start Docker or require a graph endpoint.

Neo4j settings are `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`, `NEO4J_CA_CERT`, `NEO4J_TRUST_ALL`, and
`NEO4J_AUTO_START`. Development defaults target `bolt://localhost:7687` with
password `testpassword`. When `PROJECTION_BACKEND=neo4j` in localhost
development, install `zaxy-memory[neo4j]`; then `NEO4J_AUTO_START=true` lets
Zaxy start or reuse a named Docker container, `zaxy-neo4j`, when Bolt is not
reachable. The local embedded happy path leaves `NEO4J_AUTO_START=false`,
`NEO4J_CA_CERT` empty, and `NEO4J_PASSWORD_FILE` empty. Production mode rejects
the default password and requires TLS evidence only when Neo4j is the selected
backend. Use `bolt+s://` or set `NEO4J_CA_CERT` to a trusted certificate path
for production Neo4j.

Eventloom settings are `EVENTLOOM_PATH`, `EVENTLOOM_THREAD`, and `ZAXY_DOMAIN`.
The path is the directory containing session JSONL logs. The thread is the
default session name when callers do not provide an explicit session. Generated
MCP configs derive a domain-prefixed default such as `zaxy-default` so separate
projects do not collide on raw `default`. Session identifiers are validated
before becoming filenames.

MCP settings include `SERVER_NAME`, `MCP_ADMIN_TOKEN`,
`MCP_REMOTE_AUTH_TOKEN`, and `MCP_REMOTE_SESSION_HEADER`. The remote bearer
token protects SSE endpoints. The session header scopes remote clients so one
client cannot query or replay another client's session by accident. Production
mode requires an admin token so replay and invalidation cannot be left open.
Public multi-tenant deployments should use OIDC instead by setting
`MCP_OIDC_ISSUER`, `MCP_OIDC_AUDIENCE`, `MCP_OIDC_JWKS_URL`,
`MCP_OIDC_REQUIRED_SCOPE`, and `MCP_OIDC_SESSION_CLAIM`. Remote MCP/SSE rate
limiting is controlled by `MCP_RATE_LIMIT_ENABLED`,
`MCP_RATE_LIMIT_REQUESTS`, and `MCP_RATE_LIMIT_WINDOW_SECONDS`. Remote request
audit export is controlled by `MCP_AUDIT_ENABLED` and `MCP_AUDIT_PATH`.
Automatic MCP tool-call lifecycle events are controlled by
`MCP_LIFECYCLE_CAPTURE_ENABLED`; when enabled, Zaxy records redacted
`tool.call.completed` metadata for successful and failed tool dispatch.
`MCP_TOOL_PROFILE` selects the MCP tool listing profile: `core` (default
since 2.1.0) lists only the front-door verb set centered on
`memory_checkout`, while `full` lists every tool. The default flipped from
`full` in 2.1.0 backed by the internal tool-adoption lane (listing surface
8,165 → 1,344 estimated tokens, an 83.5% reduction); set
`MCP_TOOL_PROFILE=full` to restore the previous listing. Profiles change
listing, not capability: every tool remains callable by name under either
profile, and `zaxy serve --profile core|full` overrides the setting per
process. `memory_capabilities` reports the active profile and any tools that
are available but unlisted.

Embedding settings include `EMBEDDING_ENABLED`, `EMBEDDING_PROVIDER`,
`EMBEDDING_DIMENSION`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`. The deterministic hash provider is useful for local tests and
offline development. The hosted OpenAI-compatible provider is useful when vector
similarity quality matters. See [embeddings.md](embeddings.md).

Embedded vector index scale settings include `VECTOR_ANN_THRESHOLD`,
`VECTOR_ANN_MAX_DIMENSION`, `VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT`,
`VECTOR_ANN_EFS`, and `VECTOR_QUANTIZATION`. Stored vectors carry the
producing provider's version tag (for example `hash@<fingerprint>-dim1536`);
search never compares vectors across version tags, `zaxy doctor` reports
mixed-version corpora, and `zaxy memory re-embed --session-id <session>`
migrates stale-version vectors to the active provider.

ANN engagement is a two-clause rule since 2.2, gated by a dimension ceiling.
A (session, version) vector scope whose dimension is at or below
`VECTOR_ANN_MAX_DIMENSION` (default `64`) switches from the exact dense
matrix to an engine-native (LadybugDB) HNSW index when **(a)** the scope's vector count
reaches `VECTOR_ANN_THRESHOLD` (default `100000`), or **(b)** the scope's
exact float64 matrix (count × dimension × 8 bytes) would exceed the 256 MiB
vector index cache byte budget. Scopes above the dimension ceiling always
use exact float64 (or explicitly opted-in int8) search regardless of corpus
size. The count default came down from `1000000` in 2.2 on gate-G4 evidence:
two consecutive vector-scale lane runs at exactly 10^5 vectors (dimension
64) passed every ANN exit criterion — recall@10 of 1.0 on both the strict
and tie-aware metrics, ANN p50 at-or-better than the exact matrix in-run,
resident index bytes improved, and full `COPY`-based index builds of 92–98s
(`docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json`/`-r2.json`).
The dimension ceiling default is the measured envelope of that evidence —
the lane's all-criteria double pass exists only at dimension 64, and the
conclusion does not transfer upward: at dimension 1536 with 50k gaussian
vectors the lane measured HNSW recall@10 of 0.6 even at `efs` 400 while the
exact matrix answered in 22ms p50
(`docs/research/artifacts/ann-2026-06/ann3-d1536-50k-gauss-crossover.json`;
a rerun measured 0.6344, `ann4-d1536-50k-gauss-r2.json`), and raising `efs`
to 800 recovered recall only to 0.8438 with ANN p50 worse than exact
(`ann4-d1536-50k-gauss-efs800.json`) — tuning does not rescue
high-dimension scale.
Raise `VECTOR_ANN_MAX_DIMENSION` only with lane evidence for your dimension
and distribution.

Clause (b) is the dimension-aware memory guard within the ceiling; at the
256 MiB budget the exact matrix crosses it above these row counts:

| Dimension | Exact float64 rows that fit the byte budget |
| --- | --- |
| 64 | 524,288 |
| 384 | 87,381 |
| 1536 | 21,845 |

`VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT` (default `true`) controls clause (b)
alone. An explicit `VECTOR_ANN_THRESHOLD` remains an absolute count override
for clause (a), but within the dimension ceiling the byte clause applies
regardless of it — to force the exact float64 matrix above budget, also set
`VECTOR_ANN_BYTE_BUDGET_ENGAGEMENT=false` (or opt in to
`VECTOR_QUANTIZATION=int8`, which keeps its precedence below the count
threshold and stores ~1/8 of the float64 bytes). Exact search above budget
is viable — and above the dimension ceiling it is the measured
recommendation — because the cache's LRU byte eviction always keeps the
newest matrix resident: a single over-budget scope degrades to a cache of
one (a 100k-vector corpus at dimension 1536 is a 1.23 GB matrix, 4.58× the
budget, and stays resident); the budget bounds multi-scope cache totals, so
only workloads alternating across several jointly over-budget scopes rebuild
per switch. The HNSW path oversamples candidates from the index and reranks
them with exact float64 scores read from the resident entity vectors, so the
final ordering is exact while results still report `exact: false` (the
candidate set is approximate). Each (session, version) group queries its own
shadow table directly — no per-query graph projection or predicate scan.
`VECTOR_ANN_EFS` (default `400`) is the HNSW query-time candidate-list size
(`efs`), the primary recall knob: higher values trade latency for recall, and
the effective value is never below the oversampled candidate count a query
requests. The default moved from `200` to `400` in 2.2 on lane evidence: an
internal sweep on a realistic embedding distribution (10k vectors, dimension
1536) measured recall@10 of 0.8531 at `efs` 200, 0.9875 at 400, and 1.0 at
800, with roughly 2ms of added p50 per step, so the default now matches the
high-dimension recommendation. Set `VECTOR_ANN_EFS=800` for maximum recall
when the extra few milliseconds are acceptable.
`VECTOR_QUANTIZATION=int8` (default `none`) opts in to
int8 matrix storage with per-vector scale factors; quantized search oversamples
candidates with integer dot products and reranks them with exact float scores.

Retrieval settings include `RETRIEVAL_PROFILE`, `QUERY_DEFAULT_LIMIT`,
`QUERY_SCORING_PROFILE`, `RETENTION_POLICY`,
`RETENTION_DECAY_HALF_LIFE_DAYS`, `RETENTION_EXPIRED_WEIGHT`,
`CONTEXT_VERBATIM_ENABLED`, `CONTEXT_VERBATIM_SLOTS`, `RERANKER_PROVIDER`,
`RERANKER_URL`, `RERANKER_API_KEY`, `OPENAI_RERANK_MODEL`, and
`OPENAI_BASE_URL`. `RETRIEVAL_PROFILE` names the retrieval profile:
`cognitive` (default since 2.1.0), `local_fast`, `local_sota`, `hosted_sota`,
or `custom`. The default flipped from `local_fast` in 2.1.0 backed by the
internal forgetting lane (exact cold-start parity, no-recall-loss 1.0,
pin/authority exemptions 1.0, ranking lift 1.0 vs 0.0); the cognitive profile
composes the same local stack as `local_fast` plus the salience-ranking,
cue-blending, and graph-walk flags. Set `RETRIEVAL_PROFILE=local_fast` to
restore the previous plain ranking. Leaving the profile unset while
explicitly customizing embedding/reranker/scoring knobs still resolves to the
`custom` profile, exactly as before the flip. See
[retrieval.md](retrieval.md).
`RETENTION_POLICY=none` is the default and preserves current retrieval
behavior. `filter_expired` removes expired candidates at query time, while
`decay` keeps candidates eligible but downranks stale or expired memory without
mutating Eventloom or projected facts. `RERANKER_PROVIDER=lexical` enables
deterministic local reranking. `RERANKER_PROVIDER=http` sends fused candidates
to a local/self-hosted endpoint. `RERANKER_PROVIDER=openai` uses an
OpenAI-compatible chat-completions model and `OPENAI_API_KEY`.
`CONTEXT_VERBATIM_ENABLED=true` reserves exact Eventloom source recall during
context assembly, and `CONTEXT_VERBATIM_SLOTS` controls how many assembled
context slots are reserved for those cited source chunks.
`CONTEXT_PACKET_MEMORY_ENABLED=true` adds a bounded proactive lane for recent
`llm.packet.projected` memory, and `CONTEXT_PACKET_MEMORY_SLOTS` controls how
many assembled context slots are reserved for that packet-derived memory.

Cognitive memory settings include `SALIENCE_HALF_LIFE_DAYS`, `SALIENCE_FLOOR`,
and `ENCODING_GATE_ENABLED`. `SALIENCE_HALF_LIFE_DAYS` (default `30.0`, must
be greater than zero) is the exponential recency-decay half-life used by the
salience reinforcement ledger. `SALIENCE_FLOOR` (default `0.15`) only takes
effect under the `RETRIEVAL_PROFILE=cognitive` profile (the default since
2.1.0): memories whose
replayed salience falls below the floor leave default checkout ranking,
labeled `attenuated` in checkout diagnostics, while staying fully reachable
via explicit `memory_query`/`memory_replay`; authority-accepted state and
events appended with `"pinned": true` payload metadata are exempt.
`ENCODING_GATE_ENABLED` (default `false`) tags each append's payload with a
`novel`/`reinforcing`/`redundant` encoding classification computed from
verbatim-index and entity-name overlap (no embedding calls); events are
always appended and hash-chained regardless, redundant appends additionally
emit a weak reinforcement toward the duplicated memory, and the tags are
purely observational — re-projecting with the gate off yields identical
ranking state. See [retrieval.md](retrieval.md) for the `cognitive`
retrieval profile.

Supported secret-file variants are `NEO4J_PASSWORD_FILE`,
`MCP_ADMIN_TOKEN_FILE`, `MCP_REMOTE_AUTH_TOKEN_FILE`,
`MCP_OIDC_CLIENT_SECRET_FILE`, `OPENAI_API_KEY_FILE`, `RERANKER_API_KEY_FILE`,
and `PATHLIGHT_ACCESS_TOKEN_FILE`. Production setup writes these references
into `.env`; the settings loader resolves them during initialization. Secret
files must not be world-readable.

Validation commands:

```bash
scripts/validate-deployment.sh --root .
scripts/release-check.sh --root .
```

The deployment validator checks production mode, selected sidecar TLS posture,
remote MCP auth, admin-token configuration, and secret-file permissions. Embedded
LadybugDB production deployments do not need Neo4j certificate material unless
`PROJECTION_BACKEND=neo4j` is selected. The full release gate also runs tests,
package validation, and documentation validation. See
[deployment.md](deployment.md), [security.md](security.md), and
[runbook.md](runbook.md). The short setup path is still documented in
[README.md](../README.md).
