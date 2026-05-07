# Security

Zaxy handles durable memory, so security defaults matter. The most important
rule is to keep secrets out of Eventloom payloads. Eventloom logs are designed
to persist, replay, and export. Store redacted summaries, references, and
non-sensitive identifiers instead of credentials, API keys, bearer tokens, or
private customer data.

Eventloom appends apply a final safety net before an event is sealed. Payload
keys such as `password`, `api_key`, `authorization`, `token`, `secret`, and
`private_key` are replaced with `[REDACTED]`, and common secret-looking string
values are redacted even when the field name is generic. The event records a
`security` classification with `sensitivity` and `redacted_paths`, so replay
can show where redaction occurred without exposing the original value. Treat
this as defense in depth, not as permission to send raw credentials.

Production configuration should use secret files. Supported file variables are
`NEO4J_PASSWORD_FILE`, `MCP_ADMIN_TOKEN_FILE`,
`MCP_REMOTE_AUTH_TOKEN_FILE`, `OPENAI_API_KEY_FILE`, and
`PATHLIGHT_ACCESS_TOKEN_FILE`. The settings loader reads these paths from
process environment or `.env`. Direct env variables take precedence. Secret
files should be mode `600` and stored outside the repository when possible.

Remote MCP/SSE must be authenticated. Configure `MCP_REMOTE_AUTH_TOKEN` or
`MCP_REMOTE_AUTH_TOKEN_FILE`; clients must send `Authorization: Bearer <token>`.
Also configure `MCP_REMOTE_SESSION_HEADER`, defaulting to `x-zaxy-session-id`,
so each remote client is scoped to a validated session ID. Remote clients should
not be able to replay or query another session by choosing a different payload
field.

Production also requires `MCP_ADMIN_TOKEN` or `MCP_ADMIN_TOKEN_FILE`.
Replay and invalidation are bulk-read or state-changing operations, so they
must fail closed when no admin token is configured. Remote sessions still remain
session-scoped after admin authorization.

Graph projections are session-scoped. Zaxy stores `session_id` on projected
entities and relationships, includes it in the temporal uniqueness constraint,
and applies it to exact, keyword, vector, traversal, replay, and invalidation
paths. This prevents one remote client from retrieving another client's memory
through the shared Neo4j database.

Neo4j should use TLS in production. Development compose binds ports to
localhost. Production compose enables Bolt TLS and mounts certificates generated
by `scripts/generate-certs.sh` or supplied by your platform. The deployment
validator rejects production plaintext Bolt usage without a CA certificate.

Input validation and payload sanitization live in `src/zaxy/security.py`. They
bound payload size, query length, traversal depth, result limits, and session ID
shape, then classify and redact payloads before durable writes. These limits
protect the graph and filesystem from accidental or hostile inputs.

Observability also needs care. Pathlight tracing should avoid raw query text
unless `TRACE_RAW_QUERIES=true` is explicitly chosen. Raw prompts and queries may
contain sensitive context. Prefer hashes, summaries, and structured metadata for
routine production traces.

Before promotion run:

```bash
scripts/validate-deployment.sh --root .
scripts/release-check.sh --root .
```

See [configuration.md](configuration.md), [deployment.md](deployment.md),
[operations.md](operations.md), [mcp.md](mcp.md), and [README.md](../README.md)
for the surrounding procedures.
