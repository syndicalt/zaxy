# Configuration

Zaxy configuration is centralized in `src/zaxy/config.py`. Settings load from
process environment variables, `.env`, defaults, and Docker/Kubernetes-style
secret files. Direct environment values win over `*_FILE` values. This keeps
development simple while allowing production deployments to avoid plaintext
secrets in environment dumps.

Core Neo4j settings are `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`, `NEO4J_CA_CERT`, `NEO4J_TRUST_ALL`, and
`NEO4J_AUTO_START`. Development defaults target `bolt://localhost:7687` with
password `testpassword`. For localhost development MCP startup,
`NEO4J_AUTO_START=true` lets Zaxy start or reuse a named Docker container,
`zaxy-neo4j`, when Bolt is not reachable. Production mode rejects the default
password and requires TLS evidence when using a `bolt://` URI. Use `bolt+s://`
or set `NEO4J_CA_CERT` to a trusted certificate path.

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

Embedding settings include `EMBEDDING_ENABLED`, `EMBEDDING_PROVIDER`,
`EMBEDDING_DIMENSION`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`. The deterministic hash provider is useful for local tests and
offline development. The hosted OpenAI-compatible provider is useful when vector
similarity quality matters. See [embeddings.md](embeddings.md).

Retrieval settings include `QUERY_DEFAULT_LIMIT`, `QUERY_SCORING_PROFILE`,
`RETENTION_POLICY`, `RETENTION_DECAY_HALF_LIFE_DAYS`,
`RETENTION_EXPIRED_WEIGHT`, `CONTEXT_VERBATIM_ENABLED`,
`CONTEXT_VERBATIM_SLOTS`, `RERANKER_PROVIDER`, `RERANKER_URL`,
`RERANKER_API_KEY`, `OPENAI_RERANK_MODEL`, and `OPENAI_BASE_URL`.
`RETENTION_POLICY=none` is the default and preserves current retrieval
behavior. `filter_expired` removes expired candidates at query time, while
`decay` keeps candidates eligible but downranks stale or expired memory without
mutating Eventloom or Neo4j facts. `RERANKER_PROVIDER=lexical` enables
deterministic local reranking. `RERANKER_PROVIDER=http` sends fused candidates
to a local/self-hosted endpoint. `RERANKER_PROVIDER=openai` uses an
OpenAI-compatible chat-completions model and `OPENAI_API_KEY`.
`CONTEXT_VERBATIM_ENABLED=true` reserves exact Eventloom source recall during
context assembly, and `CONTEXT_VERBATIM_SLOTS` controls how many assembled
context slots are reserved for those cited source chunks.

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

The deployment validator checks production mode, TLS configuration, remote MCP
auth, admin-token configuration, and secret-file permissions. The full release
gate also runs tests, package validation, and documentation validation. See
[deployment.md](deployment.md), [security.md](security.md), and
[runbook.md](runbook.md). The short setup path is still documented in
[README.md](../README.md).
