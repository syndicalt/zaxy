# Getting Started

Zaxy is a local-first memory fabric for AI agents. It keeps the immutable
work history in Eventloom JSONL files and projects structured facts into Neo4j
so agents can retrieve connected, temporal context through MCP tools. The
fastest path is to run Neo4j with Docker, start the MCP server, append a few
typed events, and query them back.

Start from the repository root:

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

To start the default stdio MCP server:

```bash
zaxy serve
```

To run the SSE transport for daemon-style clients:

```bash
zaxy serve --transport sse --port 8080
```

The MCP tool names are stable: `memory_append`, `memory_query`,
`memory_replay`, and `memory_invalidate`. A simple client can append a typed
`goal.created` or `task.proposed` event, then query for the goal title and
receive compact context chunks from the graph. Zaxy also exposes a Python API
through `MemoryFabric`; see [api.md](api.md) for constructor and method
details.

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
