# Deployment

Production deployment starts with explicit configuration and secret files. Run:

```bash
./scripts/setup.sh --production
docker compose -f docker-compose.prod.yml up -d
```

Production setup writes `.env` and secret files under `./secrets`. The `.env`
file references those secret files with `*_FILE` variables. Generated local
secrets are useful for validation and staging, but real deployments should
replace them with platform-managed secrets or files mounted by Docker,
Kubernetes, or a vault sidecar.

The default production backend is the embedded LadybugDB projection, which requires
no graph service endpoint and does not start Neo4j. Optional sidecar backends
must be configured explicitly. For Neo4j, install `zaxy-memory[neo4j]`, generate
certificate material, and start the compose profile:

```bash
./scripts/generate-certs.sh .certs
docker compose -f docker-compose.prod.yml --profile neo4j up -d zaxy-neo4j
```

Production compose mounts the Neo4j certificate material only for that profile;
`NEO4J_CA_CERT` tells Zaxy which certificate to trust. Do not set
`NEO4J_TRUST_ALL=true` in production unless you are doing a temporary emergency
recovery with a written risk acceptance.

Remote MCP/SSE should only be exposed behind authentication. For single-tenant
or private deployments, configure `MCP_REMOTE_AUTH_TOKEN_FILE` and require
clients to send the token. For public multi-tenant deployments, configure OIDC
with `MCP_OIDC_ISSUER`, `MCP_OIDC_AUDIENCE`, and `MCP_OIDC_JWKS_URL`; Zaxy
validates JWTs and scopes clients from the configured session claim. Configure
`MCP_ADMIN_TOKEN_FILE` as well; production config rejects deployments that leave
admin replay/invalidation unprotected. Place the service behind your normal
ingress controls and prefer private network exposure over direct public internet
exposure unless OIDC and rate limiting are both in place.

The production container starts the SSE transport on `0.0.0.0:8080` so Docker
and orchestration platforms can route traffic to it. Local development still
defaults to stdio and the SSE CLI host defaults to `127.0.0.1`.

Before promoting a deployment:

```bash
scripts/validate-deployment.sh --root .
scripts/release-check.sh --root .
```

The release gate must pass locally and in CI. It checks linting, types, tests,
coverage, package metadata, docs links, deployment configuration, backend
shootout evidence, injected-token efficiency floors, and 100-query embedded
scale validation. A tagged release should use artifacts built by
`scripts/build-dist.sh`. Public Python releases publish to PyPI as
`zaxy-memory`; the import package and CLI remain `zaxy`.

Backups must be configured before production traffic. At minimum, persist
Eventloom logs and secret material recovery procedures. Embedded and optional
sidecar projection backups can speed restore, but Eventloom replay is the
correctness path.

For operational recovery, see [operations.md](operations.md) and
[runbook.md](runbook.md). For environment variables, see
[configuration.md](configuration.md). For threat model and secret handling, see
[security.md](security.md). The quick setup remains in [README.md](../README.md).
