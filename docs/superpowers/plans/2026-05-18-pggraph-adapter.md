# pgGraph Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real experimental `PROJECTION_BACKEND=pggraph` adapter that stores Zaxy projections in PostgreSQL tables and uses pgGraph for bounded traversal, while Neo4j remains the default.

**Architecture:** Eventloom remains the source of truth. The pgGraph adapter owns a small relational projection schema, registers its entity and edge tables with pgGraph, and implements the existing `ProjectionStore` contract. Exact and keyword retrieval use ordinary PostgreSQL; traversal uses `graph.traverse`; vector search stays explicitly unavailable until pgvector ranking is added and benchmarked.

**Tech Stack:** Python 3.11+, optional `psycopg[binary]`, PostgreSQL, pgGraph SQL functions (`graph.add_table`, `graph.add_edge`, `graph.build`, `graph.traverse`), pytest/ruff/mypy.

---

## File Structure

- Create `src/zaxy/pggraph_store.py`: async pgGraph/PostgreSQL implementation of `ProjectionStore`.
- Modify `src/zaxy/projection_backends.py`: route `backend="pggraph"` to `PgGraphStore`.
- Modify `src/zaxy/config.py`: add `pggraph_dsn` setting.
- Modify `pyproject.toml`: add optional `pggraph` extra for `psycopg[binary]`.
- Test in `tests/test_pggraph_store.py`, `tests/test_projection.py`, and `tests/test_config.py`.
- Document in `docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md` and `docs/benchmarks.md`.

## Task 1: pgGraph Config And Factory

**Files:**
- Modify: `src/zaxy/config.py`
- Modify: `src/zaxy/projection_backends.py`
- Modify: `pyproject.toml`
- Test: `tests/test_config.py`, `tests/test_projection.py`

- [ ] **Step 1: Write failing config and factory tests**

Add to `tests/test_config.py`:

```python
def test_pggraph_dsn_defaults_to_local_postgres() -> None:
    settings = Settings(_env_file=None)

    assert settings.pggraph_dsn == "postgresql://postgres:postgres@localhost:5432/zaxy"
```

Update `tests/test_projection.py`:

```python
from zaxy.pggraph_store import PgGraphStore


def test_build_projection_store_routes_pggraph_to_adapter() -> None:
    store = build_projection_store(
        ProjectionBackendConfig(
            backend="pggraph",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpassword",
            neo4j_ca_cert=None,
            neo4j_trust_all=False,
            pggraph_dsn="postgresql://postgres:postgres@localhost:5432/zaxy",
        )
    )

    assert isinstance(store, PgGraphStore)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_config.py tests/test_projection.py -q --no-cov -k "pggraph"
```

Expected: fails because `pggraph_dsn` and `PgGraphStore` do not exist and the factory still rejects pgGraph.

- [ ] **Step 3: Implement minimal config and factory routing**

Add `pggraph_dsn` to `Settings`:

```python
pggraph_dsn: str = Field(
    default="postgresql://postgres:postgres@localhost:5432/zaxy",
    description="Experimental pgGraph PostgreSQL DSN",
)
```

Extend `ProjectionBackendConfig`:

```python
pggraph_dsn: str | None = None
```

Route pgGraph:

```python
if backend == "pggraph":
    from zaxy.pggraph_store import PgGraphStore

    if not config.pggraph_dsn:
        raise ValueError("pgGraph backend requires pggraph_dsn")
    return PgGraphStore(config.pggraph_dsn)
```

Add `pyproject.toml` optional extra:

```toml
pggraph = [
    "psycopg[binary]>=3.2.0",
]
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/test_config.py tests/test_projection.py -q --no-cov -k "pggraph"
ruff check src/zaxy/config.py src/zaxy/projection_backends.py tests/test_config.py tests/test_projection.py
PYTHONPATH=src mypy src/zaxy/config.py src/zaxy/projection_backends.py tests/test_projection.py
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/zaxy/config.py src/zaxy/projection_backends.py tests/test_config.py tests/test_projection.py
git commit -m "feat: route pggraph projection backend"
```

## Task 2: pgGraph Store Schema And Connection

**Files:**
- Create: `src/zaxy/pggraph_store.py`
- Test: `tests/test_pggraph_store.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_pggraph_store.py` with fake async connection objects:

```python
from __future__ import annotations

from typing import Any

import pytest

from zaxy.pggraph_store import PgGraphStore


class FakeCursor:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def __aenter__(self) -> "FakeCursor":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def execute(self, sql: str, params: tuple[object, ...] | dict[str, object] | None = None) -> None:
        self.connection.statements.append((sql, params))

    async def fetchall(self) -> list[dict[str, Any]]:
        return self.rows


class FakeConnection:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []
        self.cursor_obj = FakeCursor()
        self.cursor_obj.connection = self
        self.commits = 0
        self.closed = False

    def cursor(self, *, row_factory: object | None = None) -> FakeCursor:
        return self.cursor_obj

    async def execute(self, sql: str, params: tuple[object, ...] | dict[str, object] | None = None) -> None:
        self.statements.append((sql, params))

    async def commit(self) -> None:
        self.commits += 1

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_pggraph_store_init_schema_creates_projection_tables_and_registers_pggraph() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)

    await store.init_schema()

    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_entities" in sql
    assert "CREATE TABLE IF NOT EXISTS zaxy_pggraph_edges" in sql
    assert "graph.add_table" in sql
    assert "graph.add_edge" in sql
    assert "graph.build" in sql
    assert connection.commits == 1
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py -q --no-cov
```

Expected: fails because `zaxy.pggraph_store` does not exist.

- [ ] **Step 3: Implement schema and lifecycle**

Create `src/zaxy/pggraph_store.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from zaxy.extract import ExtractionResult
from zaxy.graph import GraphEntity, GraphEventProjectionStatus, GraphInferredEdgeStatus, SearchResult
from zaxy.security import validate_session_id


@dataclass(frozen=True)
class PgGraphRow:
    values: dict[str, Any]


class PgGraphStore:
    def __init__(self, dsn: str, *, connection: Any | None = None) -> None:
        self._dsn = dsn
        self._connection = connection

    async def connect(self) -> None:
        if self._connection is not None:
            return
        try:
            from psycopg import AsyncConnection
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("pgGraph backend requires installing zaxy-memory[pggraph]") from exc
        self._connection = await AsyncConnection.connect(self._dsn, row_factory=dict_row)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def init_schema(self) -> None:
        connection = self._require_connection()
        await connection.execute(PGGRAPH_SCHEMA_SQL)
        await connection.commit()

    def _require_connection(self) -> Any:
        if self._connection is None:
            raise AssertionError("Call connect() first")
        return self._connection
```

Define `PGGRAPH_SCHEMA_SQL` with projection tables, indexes, and registration:

```sql
CREATE TABLE IF NOT EXISTS zaxy_pggraph_entities (...);
CREATE TABLE IF NOT EXISTS zaxy_pggraph_edges (...);
CREATE INDEX IF NOT EXISTS zaxy_pggraph_entities_lookup_idx ON zaxy_pggraph_entities (...);
CREATE INDEX IF NOT EXISTS zaxy_pggraph_edges_source_idx ON zaxy_pggraph_edges (...);
SELECT graph.add_table('zaxy_pggraph_entities'::regclass, 'node_key', ARRAY['name', 'summary', 'entity_type'], 'session_id')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'add_table' AND pronamespace = 'graph'::regnamespace);
SELECT graph.add_edge('zaxy_pggraph_edges'::regclass, 'source_node_key', 'zaxy_pggraph_entities'::regclass, 'node_key', 'relates', false, NULL, 'relation_type')
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'add_edge' AND pronamespace = 'graph'::regnamespace);
SELECT graph.build()
WHERE EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'build' AND pronamespace = 'graph'::regnamespace);
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py -q --no-cov
ruff check src/zaxy/pggraph_store.py tests/test_pggraph_store.py
PYTHONPATH=src mypy src/zaxy/pggraph_store.py
```

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/pggraph_store.py tests/test_pggraph_store.py
git commit -m "feat: add pggraph projection store schema"
```

## Task 3: Upsert, Exact, Keyword, Invalidate

**Files:**
- Modify: `src/zaxy/pggraph_store.py`
- Test: `tests/test_pggraph_store.py`

- [ ] **Step 1: Write failing projection and retrieval tests**

Add tests that project one extraction and assert SQL parameters preserve session, temporal fields, provenance, and properties:

```python
@pytest.mark.asyncio
async def test_pggraph_store_upsert_extraction_writes_entities_edges_and_events() -> None:
    connection = FakeConnection()
    store = PgGraphStore("postgresql://test", connection=connection)
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="Zaxy",
                entity_type="project",
                observed_at="2026-05-18T00:00:00Z",
                summary="Memory product",
                properties={"path": "README.md"},
            )
        ],
        edges=[
            ExtractedEdge(
                source="Zaxy",
                target="pgGraph",
                relation_type="evaluates",
                valid_from="2026-05-18T00:00:00Z",
            )
        ],
        source_event_seq=7,
        source_event_hash="a" * 64,
        source_event_type="decision.created",
    )

    await store.upsert_extraction(result, session_id="agent-1")

    sql = "\n".join(statement for statement, _params in connection.statements)
    assert "INSERT INTO zaxy_pggraph_events" in sql
    assert "INSERT INTO zaxy_pggraph_entities" in sql
    assert "INSERT INTO zaxy_pggraph_edges" in sql
    assert connection.commits == 1
```

Add exact and keyword tests with fake rows:

```python
@pytest.mark.asyncio
async def test_pggraph_store_search_exact_maps_rows_to_graph_entities() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "name": "Zaxy",
            "entity_type": "project",
            "valid_from": "2026-05-18T00:00:00Z",
            "valid_to": None,
            "properties": {"summary": "Memory product"},
            "session_id": "agent-1",
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_exact("Zaxy", session_id="agent-1")

    assert results == [
        GraphEntity(
            name="Zaxy",
            entity_type="project",
            valid_from="2026-05-18T00:00:00Z",
            valid_to=None,
            properties={"summary": "Memory product"},
            session_id="agent-1",
        )
    ]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py -q --no-cov -k "upsert or search_exact"
```

Expected: fails because methods are not implemented.

- [ ] **Step 3: Implement relational upsert and retrieval**

Implement:

```python
async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None: ...
async def search_exact(...) -> list[GraphEntity]: ...
async def search_keyword(...) -> list[SearchResult]: ...
async def invalidate_entity(...) -> None: ...
```

Use deterministic `node_key = f"{session_id}\x1f{entity_type}\x1f{name}\x1f{observed_at}"` and JSON-encoded property payloads. Keyword search should query `name ILIKE` or `summary ILIKE`, order exact prefix matches first, and return `SearchResult(source="keyword")`.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py tests/test_projection.py -q --no-cov
ruff check src/zaxy/pggraph_store.py tests/test_pggraph_store.py
PYTHONPATH=src mypy src/zaxy/pggraph_store.py tests/test_pggraph_store.py
```

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/pggraph_store.py tests/test_pggraph_store.py
git commit -m "feat: project memories into pggraph postgres tables"
```

## Task 4: Traversal And Degraded Vector Lane

**Files:**
- Modify: `src/zaxy/pggraph_store.py`
- Test: `tests/test_pggraph_store.py`, `tests/test_query.py`

- [ ] **Step 1: Write failing traversal and vector tests**

Add a traversal test that verifies `graph.traverse` is used and hydrated rows are mapped to `GraphEntity`. Add a vector test that verifies vector search raises a clear runtime error:

```python
@pytest.mark.asyncio
async def test_pggraph_store_search_traversal_uses_pggraph_traverse() -> None:
    connection = FakeConnection()
    connection.cursor_obj.rows = [
        {
            "node": {
                "name": "pgGraph",
                "entity_type": "backend",
                "valid_from": "2026-05-18T00:00:00Z",
                "valid_to": None,
                "properties": {},
                "session_id": "agent-1",
            }
        }
    ]
    store = PgGraphStore("postgresql://test", connection=connection)

    results = await store.search_traversal("Zaxy", relation_type="evaluates", session_id="agent-1")

    assert results[0].name == "pgGraph"
    assert "graph.traverse" in connection.statements[-1][0]


@pytest.mark.asyncio
async def test_pggraph_store_search_vector_is_explicitly_unavailable() -> None:
    store = PgGraphStore("postgresql://test", connection=FakeConnection())

    with pytest.raises(RuntimeError, match="pgGraph vector search requires pgvector"):
        await store.search_vector([0.1, 0.2])
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py -q --no-cov -k "traversal or vector"
```

Expected: fails because traversal/vector methods are missing.

- [ ] **Step 3: Implement traversal and explicit vector degradation**

Use `graph.search(... hydrate := false)` to resolve the active start node key, then `graph.traverse(... hydrate := true)` to fetch neighbors. For vector search:

```python
raise RuntimeError("pgGraph vector search requires pgvector support and has not passed Zaxy benchmark gates")
```

`QueryRouter` already records lane degradation when a store raises, so this must not crash overall retrieval.

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py tests/test_query.py -q --no-cov -k "pggraph or vector_search_unavailable"
ruff check src/zaxy/pggraph_store.py tests/test_pggraph_store.py
PYTHONPATH=src mypy src/zaxy/pggraph_store.py tests/test_pggraph_store.py
```

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/pggraph_store.py tests/test_pggraph_store.py tests/test_query.py
git commit -m "feat: add pggraph traversal search"
```

## Task 5: Docs, Integration Marker, And Guardrail

**Files:**
- Modify: `docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md`
- Modify: `docs/benchmarks.md`
- Modify: `tests/test_docs_site.py`

- [ ] **Step 1: Write failing docs tests**

Add assertions that docs state pgGraph is implemented as experimental, vector is degraded, and release remains blocked on same-harness gates.

- [ ] **Step 2: Update docs**

Document:

- Install with `pip install "zaxy-memory[pggraph]"`.
- Configure `PROJECTION_BACKEND=pggraph` and `PGGRAPH_DSN=...`.
- Neo4j remains the only production and published benchmark backend.
- pgGraph adapter supports exact, keyword, projection, invalidation, and traversal; vector is unavailable until pgvector is implemented and benchmarked.

- [ ] **Step 3: Run final verification**

Run:

```bash
PYTHONPATH=src pytest tests/test_pggraph_store.py tests/test_projection.py tests/test_config.py tests/test_query.py tests/test_docs_site.py -q --no-cov
PYTHONPATH=src pytest -q --no-cov
ruff check .
PYTHONPATH=src mypy src
PYTHONPATH=src python -m zaxy benchmark-compare reports/benchmarks/longmemeval-500-hash/live-benchmark.json --backend zaxy-checkout --min-mean-score 0.626 --min-answer-recall-at-5 0.608 --min-recall-at-5 0.956 --min-citation-coverage 1.0 --max-p95-ms 15000 --max-p99-ms 23000
```

- [ ] **Step 4: Commit**

```bash
git add docs tests src pyproject.toml
git commit -m "docs: document experimental pggraph adapter"
```

## Self-Review

- Spec coverage: The plan keeps Eventloom as source of truth, preserves Neo4j as default, implements pgGraph only behind `PROJECTION_BACKEND=pggraph`, and keeps benchmark gates explicit.
- Placeholder scan: There are no unresolved placeholder markers. Vector search is intentionally unavailable with a precise runtime error until pgvector support is implemented and benchmarked.
- Type consistency: `PgGraphStore` implements the same `ProjectionStore` method names and return types used by `QueryRouter`, `MemoryFabric`, and MCP.
