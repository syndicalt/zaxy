# Getting Started

Zaxy is a local-first memory fabric for AI agents. It keeps the immutable
work history in Eventloom JSONL files and projects structured facts into Neo4j
so agents can retrieve connected, temporal context through MCP tools. The
fastest path is to run Neo4j with Docker, start the MCP server, append a few
typed events, and query them back.

Start from the repository root:

```bash
pip install zaxy-memory
zaxy status
```

For local development from a checkout:

```bash
pip install -e ".[dev]"
./scripts/setup.sh
docker compose up -d
zaxy status
```

`./scripts/setup.sh` creates `.env`, `.eventloom`, and local runtime
directories. Development mode uses `neo4j/testpassword` and localhost-bound
ports. Production mode is different: `./scripts/setup.sh --production` writes
secret files under `./secrets`, configures `ZAXY_ENV=production`, and expects a
TLS-enabled Neo4j profile. See [deployment.md](deployment.md) before exposing
remote SSE.

For an offline retrieval profile with no hosted services or API keys:

```bash
zaxy local-profile --output .env.local
zaxy local-profile --check
```

This configures deterministic hash embeddings, lexical reranking, and local
Neo4j auto-start. It is the recommended baseline for local development before
switching to hosted embeddings or model-backed rerankers.

Check local onboarding prerequisites before wiring an agent:

```bash
zaxy doctor
zaxy doctor --json
```

The doctor command verifies Eventloom writeability, local embedding/reranker
construction, static viewer generation, MCP default-session posture, Neo4j
configuration posture, and production-mode warnings. It does not start Docker
or require a live Neo4j connection; use `zaxy status` when you want a live graph
connectivity test.

To start the default stdio MCP server:

```bash
zaxy serve
```

When stdio starts in local development mode, Zaxy checks
`bolt://localhost:7687`. If Neo4j is not reachable and Docker is available, it
starts a `zaxy-neo4j` container automatically and waits for Bolt before serving
MCP tools. This is the default in generated MCP client config. Set
`NEO4J_AUTO_START=false` to opt out.

To run the SSE transport for daemon-style clients:

```bash
zaxy serve --transport sse --port 8080
```

To let Zaxy observe client lifecycle checkpoints without proxying tool
execution, generate hook adapter config:

```bash
zaxy hooks claude-code \
  --eventloom-path .eventloom \
  --domain my-project \
  --output .claude/settings.local.json
```

The stable hook contract is documented in [hooks.md](hooks.md).

The MCP tool names are stable: `memory_append`, `memory_query`,
`memory_feedback`, `memory_replay`, and `memory_invalidate`. A simple client can
append a typed `goal.created` or `task.proposed` event, query for the goal
title, record whether retrieved context was useful, and receive compact context
chunks from the graph. Zaxy also exposes a Python API through `MemoryFabric`;
see [api.md](api.md) for constructor and method details.

For day-to-day validation, run:

```bash
ruff check src tests
mypy src
pytest
scripts/release-check.sh --root .
```

The full pytest command enforces the 90 percent coverage gate configured in
`pyproject.toml`. Integration tests require Docker Neo4j services. The release
gate adds package artifact checks, documentation link validation, and deployment
preflight checks. The current public overview is [site/index.html](../site/index.html),
and the operational checklist remains in [runbook.md](runbook.md). The
[README.md](../README.md) is intentionally short; these docs are the detailed
operator and integrator reference.

The mental model is simple: append operational facts, extract graph facts,
retrieve connected context, and replay from the original event stream whenever
you need auditability. Do not treat Neo4j as the source of truth. Neo4j is the
reasoning layer. Eventloom is the durable history. MCP is the interface agents
call.
