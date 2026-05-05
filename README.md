# Zaxy

**Event-sourced temporal knowledge graph fabric for AI agent memory.**

Zaxy replaces markdown files + vector DBs with a structured, replayable,
bi-temporal memory system built on Eventloom, Neo4j, and Pathlight.

## Quick Start

```bash
# 1. Setup (generates .env and directories)
./scripts/setup.sh

# 2. Start everything (Neo4j + Zaxy MCP server)
docker compose up -d

# 3. Verify
zaxy status
pytest -m integration

# 4. Test drive (no agent needed)
python scripts/test_drive.py
```

## Architecture

```
Agent (LangGraph / Any MCP Client)
    |
    v
MCP Server — memory_append / memory_query / memory_replay / memory_invalidate
    |
    v
Eventloom (immutable JSONL log)  →  Hybrid Extraction  →  Neo4j (temporal KG)
    |                                                               |
    +—————— Pathlight traces every operation ————————→  Query Router
                                                              |
                                                    Hybrid Retrieval
                                                    (exact + BM25 + traversal)
```

## Key Features

- **Immutable audit trail**: Eventloom append-only JSONL with SHA-256 hash chains.
- **Bi-temporal graph**: Facts have validity windows (`valid_from`, `valid_to`).
- **Hybrid extraction**: Rule-based for typed events (60–80% cost reduction), LLM fallback.
- **Hybrid retrieval**: Exact + keyword + graph traversal with configurable fusion weights.
- **MCP-native**: Drop-in memory for any MCP-compatible agent framework.
- **Observable**: Pathlight traces, breakpoints, and diff support.

## Project Structure

| File | Purpose |
|------|---------|
| `src/zaxy/event.py` | Eventloom JSONL I/O + hash chain integrity |
| `src/zaxy/extract.py` | Hybrid extraction engine + rule registry |
| `src/zaxy/graph.py` | Neo4j bi-temporal wrapper |
| `src/zaxy/query.py` | Hybrid retrieval router |
| `src/zaxy/mcp_server.py` | MCP stdio server |
| `src/zaxy/trace.py` | Pathlight observability hooks |
| `src/zaxy/core.py` | MemoryFabric orchestrator |
| `src/zaxy/__main__.py` | CLI (`zaxy serve`, `zaxy replay`, etc.) |

## Development

- **Tests first** (Karpathy rule). Every public function has a test.
- **Unit tests** mock Neo4j/Pathlight. **Integration tests** use Docker.
- **Coverage gate: ≥90%** enforced by CI.
- **Lint/format**: `ruff`. **Types**: `mypy`.

```bash
# Run unit tests
pytest

# Run integration tests (requires Docker)
pytest -m integration

# Lint and type-check
ruff check src tests
mypy src
```

## License

MIT
