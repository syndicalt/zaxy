# Projection Backend Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a backend-neutral projection contract and backend factory so Neo4j remains the default while pgGraph can be evaluated behind the same retrieval and projection surface.

**Architecture:** Eventloom remains the source of truth. `GraphStore` becomes the first concrete implementation of a typed `ProjectionStore` contract, and all high-level construction paths use a backend factory instead of instantiating Neo4j directly. The pgGraph path is explicitly experimental and unavailable until a real adapter passes the same contract and benchmark gates.

**Tech Stack:** Python 3.11+, typing `Protocol`, Pydantic settings, existing Neo4j `GraphStore`, pytest/ruff/mypy, benchmark guardrail CLI.

---

## File Structure

- Modify `src/zaxy/projection.py`: replace loose `object` returns with concrete projection-store method signatures and add read-only/invalidation methods used by checkout, CLI, and dashboard surfaces.
- Create `src/zaxy/projection_backends.py`: central projection backend factory and config dataclass.
- Modify `src/zaxy/config.py`: add `projection_backend` setting with default `neo4j`.
- Modify `src/zaxy/core.py`, `src/zaxy/mcp_server.py`, `src/zaxy/live_benchmark.py`, and `src/zaxy/__main__.py`: instantiate graph projection stores through the factory where behavior must stay backend-neutral.
- Test in `tests/test_projection.py`, `tests/test_config.py`, `tests/test_core.py`, `tests/test_mcp.py`, and targeted benchmark CLI tests if constructor wiring changes.
- Modify `docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md`, `docs/benchmarks.md`, and `AGENTS.md`: record Skill Memory completion and the pgGraph contract-first evaluation state.

## Task 1: Typed Projection Contract

**Files:**
- Modify: `src/zaxy/projection.py`
- Test: `tests/test_projection.py`

- [ ] **Step 1: Write failing protocol tests**

Add `tests/test_projection.py`:

```python
from __future__ import annotations

from typing import assert_type

from zaxy.extract import ExtractionResult
from zaxy.graph import GraphEdge, GraphEntity, GraphEventProjectionStatus, GraphInferredEdgeStatus, SearchResult
from zaxy.projection import ProjectionStore


class FakeProjectionStore:
    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def init_schema(self) -> None:
        pass

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None:
        pass

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None:
        pass

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        return []

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]:
        return []

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus:
        raise NotImplementedError

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus:
        raise NotImplementedError

    async def get_overview_graph(self, session_id: str, *, limit: int = 100) -> tuple[list[GraphEntity], list[GraphEdge]]:
        return [], []


def test_fake_projection_store_satisfies_contract() -> None:
    store: ProjectionStore = FakeProjectionStore()
    assert store is not None


def test_projection_store_search_types_are_concrete(store: ProjectionStore) -> None:
    assert_type(store, ProjectionStore)
```

- [ ] **Step 2: Run test and mypy to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_projection.py -q --no-cov
PYTHONPATH=src mypy tests/test_projection.py
```

Expected: pytest may pass structurally, but mypy fails because `ProjectionStore` still returns `object` and lacks the extended methods.

- [ ] **Step 3: Implement the typed protocol**

Update `src/zaxy/projection.py`:

```python
from __future__ import annotations

from typing import Protocol

from zaxy.extract import ExtractionResult
from zaxy.graph import GraphEdge, GraphEntity, GraphEventProjectionStatus, GraphInferredEdgeStatus, SearchResult


class ProjectionStore(Protocol):
    """Backend contract for projecting Eventloom facts into a queryable memory index."""

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def init_schema(self) -> None: ...

    async def upsert_extraction(self, result: ExtractionResult, session_id: str = "default") -> None: ...

    async def invalidate_entity(
        self,
        name: str,
        entity_type: str,
        invalid_at: str,
        session_id: str = "default",
    ) -> None: ...

    async def search_exact(
        self,
        name: str,
        entity_type: str | None = None,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]: ...

    async def search_keyword(
        self,
        query: str,
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]: ...

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]: ...

    async def search_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[SearchResult]: ...

    async def inspect_event_projection_status(
        self,
        session_id: str,
        *,
        eventloom_latest_seq: int | None = None,
        eventloom_latest_hash: str | None = None,
    ) -> GraphEventProjectionStatus: ...

    async def inspect_inferred_edge_status(
        self,
        session_id: str,
        *,
        limit: int = 10,
    ) -> GraphInferredEdgeStatus: ...

    async def get_overview_graph(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> tuple[list[GraphEntity], list[GraphEdge]]: ...
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/test_projection.py tests/test_graph.py -q --no-cov -k "projection_store or graph_store_satisfies"
PYTHONPATH=src mypy src/zaxy/projection.py tests/test_projection.py
```

Expected: tests pass and mypy succeeds.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/zaxy/projection.py tests/test_projection.py
git commit -m "refactor: type projection store contract"
```

## Task 2: Backend Factory And Setting

**Files:**
- Create: `src/zaxy/projection_backends.py`
- Modify: `src/zaxy/config.py`
- Test: `tests/test_projection.py`, `tests/test_config.py`

- [ ] **Step 1: Write failing factory tests**

Append to `tests/test_projection.py`:

```python
import pytest

from zaxy.graph import GraphStore
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store


def test_build_projection_store_defaults_to_neo4j() -> None:
    store = build_projection_store(
        ProjectionBackendConfig(
            backend="neo4j",
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="testpassword",
            neo4j_ca_cert=None,
            neo4j_trust_all=False,
        )
    )

    assert isinstance(store, GraphStore)


def test_build_projection_store_rejects_pggraph_until_adapter_exists() -> None:
    with pytest.raises(NotImplementedError, match="pgGraph backend is experimental"):
        build_projection_store(
            ProjectionBackendConfig(
                backend="pggraph",
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="testpassword",
                neo4j_ca_cert=None,
                neo4j_trust_all=False,
            )
        )
```

Add to `tests/test_config.py`:

```python
def test_projection_backend_defaults_to_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROJECTION_BACKEND", raising=False)
    get_settings.cache_clear()

    assert get_settings().projection_backend == "neo4j"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_projection.py tests/test_config.py -q --no-cov -k "projection_backend or build_projection_store"
```

Expected: fails because `projection_backends.py` and `projection_backend` setting do not exist.

- [ ] **Step 3: Implement the factory**

Create `src/zaxy/projection_backends.py`:

```python
"""Projection backend construction.

Neo4j is the production default. pgGraph is exposed only as an explicit
experimental target until an adapter passes the same contract and benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zaxy.graph import GraphStore
from zaxy.projection import ProjectionStore

ProjectionBackendName = Literal["neo4j", "pggraph"]


@dataclass(frozen=True)
class ProjectionBackendConfig:
    backend: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_ca_cert: str | None
    neo4j_trust_all: bool


def build_projection_store(config: ProjectionBackendConfig) -> ProjectionStore:
    backend = config.backend.casefold().strip()
    if backend == "neo4j":
        return GraphStore(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password,
            ca_cert=config.neo4j_ca_cert,
            trust_all=config.neo4j_trust_all,
        )
    if backend == "pggraph":
        raise NotImplementedError(
            "pgGraph backend is experimental and has no adapter yet. "
            "Keep PROJECTION_BACKEND=neo4j until pgGraph passes the projection contract and benchmark gates."
        )
    raise ValueError("projection backend must be one of: neo4j, pggraph")
```

Add to `Settings` in `src/zaxy/config.py`:

```python
projection_backend: str = Field(default="neo4j", validation_alias="PROJECTION_BACKEND")
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/test_projection.py tests/test_config.py -q --no-cov -k "projection_backend or build_projection_store"
ruff check src/zaxy/projection_backends.py src/zaxy/config.py tests/test_projection.py tests/test_config.py
PYTHONPATH=src mypy src/zaxy/projection_backends.py
```

Expected: tests pass, ruff clean, mypy succeeds.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/zaxy/projection_backends.py src/zaxy/config.py tests/test_projection.py tests/test_config.py
git commit -m "feat: add projection backend factory"
```

## Task 3: Route High-Level Construction Through The Factory

**Files:**
- Modify: `src/zaxy/core.py`
- Modify: `src/zaxy/mcp_server.py`
- Modify: `src/zaxy/live_benchmark.py`
- Modify: `src/zaxy/__main__.py`
- Test: `tests/test_core.py`, `tests/test_mcp.py`, `tests/test_live_benchmark.py`, `tests/test_cli.py`

- [ ] **Step 1: Write failing wiring tests**

Add tests that patch `zaxy.core.build_projection_store`, `zaxy.mcp_server.build_projection_store`, and `zaxy.live_benchmark.build_projection_store`, then instantiate or run the existing constructor paths and assert the factory receives `backend="neo4j"` from settings.

Use this pattern in each file:

```python
@patch("zaxy.core.build_projection_store")
def test_memory_fabric_constructs_projection_store_through_factory(mock_build: MagicMock, tmp_path: Path) -> None:
    mock_build.return_value = MagicMock()

    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), tracer_disabled=True)

    assert fabric.graph is mock_build.return_value
    assert mock_build.call_args.args[0].backend == "neo4j"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_core.py tests/test_mcp.py tests/test_live_benchmark.py -q --no-cov -k "projection_store_through_factory or projection_backend"
```

Expected: fails because construction still instantiates `GraphStore` directly.

- [ ] **Step 3: Replace direct construction where safe**

Import and use:

```python
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store
```

Build config from resolved settings:

```python
ProjectionBackendConfig(
    backend=resolved_settings.projection_backend,
    neo4j_uri=neo4j_uri or resolved_settings.neo4j_uri,
    neo4j_user=neo4j_user or resolved_settings.neo4j_user,
    neo4j_password=neo4j_password or resolved_settings.neo4j_password,
    neo4j_ca_cert=neo4j_ca_cert if neo4j_ca_cert is not None else resolved_settings.neo4j_ca_cert,
    neo4j_trust_all=neo4j_trust_all if neo4j_trust_all is not None else resolved_settings.neo4j_trust_all,
)
```

Leave direct `GraphStore` construction in migration/schema commands that inspect Neo4j-only internals such as `_driver`, and document those as Neo4j-only until the pgGraph adapter has equivalent operations.

- [ ] **Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_core.py tests/test_mcp.py tests/test_live_benchmark.py -q --no-cov -k "projection_store_through_factory or projection_backend"
PYTHONPATH=src pytest tests/test_core.py tests/test_mcp.py tests/test_query.py -q --no-cov
ruff check src/zaxy/core.py src/zaxy/mcp_server.py src/zaxy/live_benchmark.py src/zaxy/__main__.py
PYTHONPATH=src mypy src/zaxy/core.py src/zaxy/mcp_server.py src/zaxy/live_benchmark.py
```

Expected: tests pass, ruff clean, mypy succeeds.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/zaxy/core.py src/zaxy/mcp_server.py src/zaxy/live_benchmark.py src/zaxy/__main__.py tests/test_core.py tests/test_mcp.py tests/test_live_benchmark.py
git commit -m "refactor: construct projection stores through backend factory"
```

## Task 4: Documentation And Roadmap State

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/benchmarks.md`
- Modify: `docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md`
- Test: `tests/test_docs_site.py`

- [ ] **Step 1: Write failing docs assertions**

Add to `tests/test_docs_site.py`:

```python
def test_pggraph_docs_keep_backend_experimental_and_contract_first() -> None:
    spec = Path("docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "PROJECTION_BACKEND=neo4j" in spec
    assert "pgGraph backend is experimental" in spec
    assert "Projection backend contract and Neo4j factory" in agents
    assert "Skill Memory procedural world-model layer" in agents
```

- [ ] **Step 2: Run docs test to verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_docs_site.py -q --no-cov -k pggraph_docs
```

Expected: fails because docs do not yet mention the factory state or updated roadmap completion.

- [ ] **Step 3: Update docs**

Update `AGENTS.md` status:

```markdown
- [x] Skill Memory procedural world-model layer with lifecycle events, checkout routing, MCP helper, docs, and full-set quality guardrail verification
- [x] Projection backend contract and Neo4j factory for pgGraph evaluation without changing the default backend
```

Move the Skill Memory next step out of the active list. Update pgGraph next step to:

```markdown
Build the experimental pgGraph adapter behind `PROJECTION_BACKEND=pggraph` only after the backend-neutral Neo4j factory and contract tests are green.
```

Update the pgGraph spec with the current upstream posture from the docs checked on May 18, 2026:

```markdown
As of May 18, 2026, pgGraph docs describe version 0.1.0, PostgreSQL 13-18 support, and alpha status for experimentation, demos, benchmarks, and early feedback. Zaxy therefore keeps `PROJECTION_BACKEND=neo4j` as the default and treats `PROJECTION_BACKEND=pggraph` as unavailable until an adapter passes the benchmark gates.
```

- [ ] **Step 4: Run docs tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_docs_site.py -q --no-cov
```

Expected: docs tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add AGENTS.md docs/benchmarks.md docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md tests/test_docs_site.py
git commit -m "docs: update pggraph evaluation roadmap state"
```

## Task 5: Final Verification

**Files:**
- Read-only verification over changed code and docs.

- [ ] **Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=src pytest tests/test_projection.py tests/test_config.py tests/test_core.py tests/test_mcp.py tests/test_query.py tests/test_docs_site.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 2: Run static checks**

Run:

```bash
ruff check src/zaxy/projection.py src/zaxy/projection_backends.py src/zaxy/config.py src/zaxy/core.py src/zaxy/mcp_server.py src/zaxy/live_benchmark.py tests/test_projection.py tests/test_config.py tests/test_core.py tests/test_mcp.py tests/test_docs_site.py
PYTHONPATH=src mypy src/zaxy/projection.py src/zaxy/projection_backends.py src/zaxy/config.py src/zaxy/core.py src/zaxy/mcp_server.py src/zaxy/live_benchmark.py
```

Expected: ruff clean and mypy succeeds.

- [ ] **Step 3: Run benchmark quality guardrail on archived full-500 report**

Run:

```bash
PYTHONPATH=src python -m zaxy benchmark-compare reports/benchmarks/longmemeval-500-hash/live-benchmark.json \
  --backend zaxy-checkout \
  --min-mean-score 0.626 \
  --min-answer-recall-at-5 0.608 \
  --min-recall-at-5 0.956 \
  --min-citation-coverage 1.0 \
  --max-p95-ms 15000 \
  --max-p99-ms 23000
```

Expected: PASS on the archived report. If a live run is performed, compare quality floors separately from noisy local latency and do not replace archived floors without an explicit new report.

- [ ] **Step 4: Confirm no direct accidental pgGraph production enablement**

Run:

```bash
rg -n "PROJECTION_BACKEND|pggraph|GraphStore\\(" src/zaxy tests docs AGENTS.md
```

Expected: pgGraph is documented as experimental, default backend remains Neo4j, and any remaining direct `GraphStore(` construction is either in the Neo4j adapter/factory, tests, or Neo4j-specific CLI/schema code.

- [ ] **Step 5: Commit final fixes if needed**

```bash
git status --short
```

Expected: clean worktree. If verification caused edits, commit them with `chore: finalize projection backend contract`.

## Self-Review

- Spec coverage: backend-neutral contract, Neo4j default preservation, pgGraph experimental gate, same-harness guardrails, and roadmap state are mapped to tasks.
- Placeholder scan: no TBD/TODO/fill-in markers remain.
- Type consistency: `ProjectionStore`, `ProjectionBackendConfig`, `build_projection_store`, and `PROJECTION_BACKEND` are used consistently across tasks.
- Scope check: this plan intentionally stops before a pgGraph adapter implementation. That adapter is a separate plan after this contract slice is merged and verified.
