# Deployment

Production deployment starts with explicit configuration and secret files. Run:

```bash
./scripts/setup.sh --production
./scripts/generate-certs.sh .certs
docker compose -f docker-compose.prod.yml up -d
```

Production setup writes `.env` and secret files under `./secrets`. The `.env`
file references those secret files with `*_FILE` variables. Generated local
secrets are useful for validation and staging, but real deployments should
replace them with platform-managed secrets or files mounted by Docker,
Kubernetes, or a vault sidecar.

Neo4j must be reachable through an encrypted Bolt configuration. Production
compose enables Bolt TLS and mounts certificate material. `NEO4J_CA_CERT` tells
Zaxy which certificate to trust. Do not set `NEO4J_TRUST_ALL=true` in
production unless you are doing a temporary emergency recovery with a written
risk acceptance.

Remote MCP/SSE should only be exposed behind bearer auth. Configure
`MCP_REMOTE_AUTH_TOKEN_FILE` and require clients to send the token. Require a
session header such as `x-zaxy-session-id` for every remote client. Place the
service behind your normal ingress controls and prefer private network exposure
over direct public internet exposure.

Before promoting a deployment:

```bash
scripts/validate-deployment.sh --root .
scripts/release-check.sh --root .
```

The release gate must pass locally and in CI. It checks linting, types, tests,
coverage, package metadata, docs links, and deployment configuration. A tagged
release should use artifacts built by `scripts/build-dist.sh`.

Backups must be configured before production traffic. At minimum, persist
Eventloom logs and secret material recovery procedures. Neo4j backups are
recommended for faster restore, but Eventloom replay is the correctness path.

For operational recovery, see [operations.md](operations.md) and
[runbook.md](runbook.md). For environment variables, see
[configuration.md](configuration.md). For threat model and secret handling, see
[security.md](security.md). The quick setup remains in [README.md](../README.md).
