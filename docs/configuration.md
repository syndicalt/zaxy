# Configuration

Zaxy configuration is centralized in `src/zaxy/config.py`. Settings load from
process environment variables, `.env`, defaults, and Docker/Kubernetes-style
secret files. Direct environment values win over `*_FILE` values. This keeps
development simple while allowing production deployments to avoid plaintext
secrets in environment dumps.

Core Neo4j settings are `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`NEO4J_DATABASE`, `NEO4J_CA_CERT`, and `NEO4J_TRUST_ALL`. Development defaults
target `bolt://localhost:7687` with password `testpassword`. Production mode
rejects the default password and requires TLS evidence when using a `bolt://`
URI. Use `bolt+s://` or set `NEO4J_CA_CERT` to a trusted certificate path.

Eventloom settings are `EVENTLOOM_PATH` and `EVENTLOOM_THREAD`. The path is the
directory containing session JSONL logs. The thread is the default session name
when callers do not provide an explicit session. Session identifiers are
validated before becoming filenames.

MCP settings include `SERVER_NAME`, `MCP_ADMIN_TOKEN`,
`MCP_REMOTE_AUTH_TOKEN`, and `MCP_REMOTE_SESSION_HEADER`. The remote bearer
token protects SSE endpoints. The session header scopes remote clients so one
client cannot query or replay another client's session by accident.

Embedding settings include `EMBEDDING_ENABLED`, `EMBEDDING_PROVIDER`,
`EMBEDDING_DIMENSION`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_BASE_URL`, and
`OPENAI_API_KEY`. The deterministic hash provider is useful for local tests and
offline development. The hosted OpenAI-compatible provider is useful when vector
similarity quality matters. See [embeddings.md](embeddings.md).

Supported secret-file variants are `NEO4J_PASSWORD_FILE`,
`MCP_ADMIN_TOKEN_FILE`, `MCP_REMOTE_AUTH_TOKEN_FILE`, `OPENAI_API_KEY_FILE`,
and `PATHLIGHT_ACCESS_TOKEN_FILE`. Production setup writes these references into
`.env`; the settings loader resolves them during initialization. Secret files
must not be world-readable.

Validation commands:

```bash
scripts/validate-deployment.sh --root .
scripts/release-check.sh --root .
```

The deployment validator checks production mode, TLS configuration, remote MCP
auth, and secret-file permissions. The full release gate also runs tests,
package validation, and documentation validation. See [deployment.md](deployment.md),
[security.md](security.md), and [runbook.md](runbook.md). The short setup path
is still documented in [README.md](../README.md).
