# Zero-Friction Embedded Graph Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a prototype embedded graph runtime that can replace sidecar-first local onboarding without weakening Zaxy's temporal, cited, graph-backed retrieval quality.

**Architecture:** Keep Eventloom as the source of truth and add `PROJECTION_BACKEND=embedded` behind the existing `ProjectionStore` contract. The first candidate is a Kuzu-backed adapter stored under `.eventloom/projections/embedded.kuzu`; Neo4j remains the default and quality control backend until same-harness gates pass.

**Tech Stack:** Python 3.11+, Kuzu optional dependency, existing `ProjectionStore`, Eventloom replay, Typer CLI, pytest, ruff, mypy.

---

## File Structure

- Create `src/zaxy/embedded_graph_store.py`: Kuzu-backed `ProjectionStore` implementation.
- Modify `src/zaxy/projection_backends.py`: route `backend="embedded"` to the embedded store.
- Modify `src/zaxy/config.py`: add embedded graph path/settings and document accepted backend values.
- Modify `src/zaxy/__main__.py`: expose embedded backend in CLI options/help where projection backend is accepted.
- Modify `src/zaxy/dashboard.py`: make graph routes display embedded backend metadata and projection status.
- Modify `pyproject.toml`: add optional dependency group `embedded = ["kuzu>=0.11"]`.
- Create `tests/test_embedded_graph_store.py`: adapter contract tests with real Kuzu when installed and skip otherwise.
- Create `tests/test_projection_backends_embedded.py`: factory/config tests.
- Create or extend `tests/test_cli.py`: help/status/init coverage for embedded backend.
- Create `scripts/backend-shootout.py`: repeatable same-harness backend comparison runner.
- Modify `docs/zero-friction-runtime-roadmap.md`, `docs/configuration.md`, `docs/getting-started.md`, and `docs/benchmarks.md`: document prototype status and gates.

## Task 1: Backend Factory And Settings

**Files:**
- Modify: `src/zaxy/config.py`
- Modify: `src/zaxy/projection_backends.py`
- Modify: `pyproject.toml`
- Test: `tests/test_projection_backends_embedded.py`

- [x] **Step 1: Write the failing factory tests**

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store


def test_build_projection_store_routes_embedded_backend(tmp_path: Path) -> None:
    config = ProjectionBackendConfig(
        backend="embedded",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpassword",
        neo4j_ca_cert=None,
        neo4j_trust_all=False,
        embedded_graph_path=tmp_path / ".eventloom" / "projections" / "embedded.kuzu",
    )

    with patch("zaxy.embedded_graph_store.EmbeddedGraphStore") as store_cls:
        store = build_projection_store(config)

    assert store is store_cls.return_value
    store_cls.assert_called_once_with(tmp_path / ".eventloom" / "projections" / "embedded.kuzu")


def test_embedded_backend_requires_path() -> None:
    config = ProjectionBackendConfig(
        backend="embedded",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpassword",
        neo4j_ca_cert=None,
        neo4j_trust_all=False,
    )

    with pytest.raises(ValueError, match="embedded backend requires embedded_graph_path"):
        build_projection_store(config)
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_projection_backends_embedded.py --no-cov -q`

Expected: fails because `embedded_graph_path` and `EmbeddedGraphStore` do not exist.

- [x] **Step 3: Implement minimal factory support**

Add `embedded_graph_path: Path | None = None` to `ProjectionBackendConfig`, route `backend == "embedded"` to `zaxy.embedded_graph_store.EmbeddedGraphStore`, and update the error text to include `neo4j, pggraph, embedded`.

- [x] **Step 4: Add package metadata**

Add this optional dependency group:

```toml
embedded = [
    "kuzu>=0.11.0",
]
```

- [x] **Step 5: Run focused checks**

Run:

```bash
ruff check src/zaxy/projection_backends.py tests/test_projection_backends_embedded.py
PYTHONPATH=src pytest tests/test_projection_backends_embedded.py --no-cov -q
```

Expected: ruff clean and tests pass.

## Task 2: Embedded Store Schema And Projection

**Files:**
- Create: `src/zaxy/embedded_graph_store.py`
- Test: `tests/test_embedded_graph_store.py`

- [x] **Step 1: Write schema/projection tests**

```python
import importlib.util
from pathlib import Path

import pytest

from zaxy.embedded_graph_store import EmbeddedGraphStore
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kuzu") is None, reason="kuzu is not installed")


@pytest.mark.asyncio
async def test_embedded_store_projects_entities_and_edges(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()

    result = ExtractionResult(
        entities=[
            ExtractedEntity(name="Goal A", entity_type="goal", properties={"summary": "ship memory"}),
            ExtractedEntity(name="Task B", entity_type="task", properties={"summary": "write tests"}),
        ],
        edges=[
            ExtractedEdge(
                source_name="Task B",
                source_type="task",
                target_name="Goal A",
                target_type="goal",
                relation_type="SUPPORTS",
                properties={"source_event_seq": 1, "source_event_hash": "abc"},
            )
        ],
        event_metadata={"seq": 1, "hash": "abc", "prev_hash": None, "event_type": "task.proposed"},
    )

    await store.upsert_extraction(result, session_id="agent-1")

    exact = await store.search_exact("Goal A", session_id="agent-1")
    neighbors = await store.search_traversal("Task B", depth=1, session_id="agent-1")

    assert exact[0].name == "Goal A"
    assert any(entity.name == "Goal A" for entity in neighbors)
    await store.close()
```

- [x] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_embedded_graph_store.py --no-cov -q`

Expected: fails because `EmbeddedGraphStore` does not exist, or skips if Kuzu is not installed.

- [x] **Step 3: Implement minimal Kuzu-backed store**

Implement `connect`, `close`, `init_schema`, `upsert_extraction`, `search_exact`, `search_keyword`, `search_traversal`, and `has_traversal_edges`. Use explicit node/relationship tables for `Entity`, `Event`, `Source`, `RELATES`, `NEXT_EVENT`, `PREVIOUS_EVENT`, `CITES_SOURCE`, `SUPERSEDED_BY`, and `PREVIOUS_VERSION` where supported by the first Kuzu schema.

- [x] **Step 4: Run focused checks**

Run:

```bash
ruff check src/zaxy/embedded_graph_store.py tests/test_embedded_graph_store.py
PYTHONPATH=src pytest tests/test_embedded_graph_store.py --no-cov -q
```

Expected: ruff clean and tests pass or Kuzu-dependent tests skip cleanly when Kuzu is absent.

## Task 3: Temporal Semantics And Rebuild

**Files:**
- Modify: `src/zaxy/embedded_graph_store.py`
- Test: `tests/test_embedded_graph_store.py`

- [x] **Step 1: Add temporal and invalidation tests**

Add tests proving reasserted entities create active versions with `valid_from`, old versions get `valid_to`, `invalidate_entity` closes active rows, and `inspect_event_projection_status` reports latest projected Eventloom sequence/hash.

- [x] **Step 2: Run tests to verify failure**

Run: `PYTHONPATH=src pytest tests/test_embedded_graph_store.py --no-cov -q`

Expected: temporal and status assertions fail until implemented.

- [x] **Step 3: Implement temporal behavior**

Mirror Neo4j semantics in the embedded store: preserve session isolation, close previous active versions, create supersession relationships, and make projection status compare Event nodes against Eventloom latest sequence/hash.

- [x] **Step 4: Run focused checks**

Run:

```bash
ruff check src/zaxy/embedded_graph_store.py tests/test_embedded_graph_store.py
PYTHONPATH=src pytest tests/test_embedded_graph_store.py --no-cov -q
```

Expected: tests pass or skip only for missing Kuzu.

## Task 4: Activation And Status Surfaces

**Files:**
- Modify: `src/zaxy/__main__.py`
- Modify: `src/zaxy/dashboard.py`
- Modify: `src/zaxy/hooks.py`
- Test: `tests/test_cli.py`

- [x] **Step 1: Add failing CLI assertions**

Add tests asserting `zaxy init --help`, `zaxy dashboard --help`, and `zaxy memory status --help` surface `embedded` as a projection backend option, and `hook-status` output includes last checkout and stale/no-checkout activation status when available.

- [x] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/test_cli.py --no-cov -q`

Expected: new assertions fail until help/status surfaces are updated.

- [x] **Step 3: Implement activation status**

Add read-only status fields for latest checkout event, latest capture event, stale checkout threshold, and no-memory-use warning. Keep JSON output stable and add clearer human output.

- [x] **Step 4: Run focused checks**

Run:

```bash
ruff check src/zaxy/__main__.py src/zaxy/dashboard.py src/zaxy/hooks.py tests/test_cli.py
mypy src/zaxy/__main__.py src/zaxy/dashboard.py src/zaxy/hooks.py
PYTHONPATH=src pytest tests/test_cli.py --no-cov -q
```

Expected: ruff, mypy, and CLI tests pass.

## Task 5: Backend Shootout Harness

**Files:**
- Create: `scripts/backend-shootout.py`
- Modify: `docs/benchmarks.md`
- Test: `tests/test_docs_site.py`

- [x] **Step 1: Add docs test for shootout contract**

Add assertions that benchmark docs mention embedded, Neo4j, pgGraph, BM25, first useful init time, checkout p95/p99, returned tokens, citation coverage, and recovery time.

- [x] **Step 2: Run test to verify failure**

Run: `PYTHONPATH=src pytest tests/test_docs_site.py::test_zero_friction_runtime_roadmap_sets_frontier_bar --no-cov -q`

Expected: fails until docs and script exist.

- [x] **Step 3: Implement the shootout script**

Create a CLI that accepts `--eventloom-path`, `--session-id`, `--backends embedded,neo4j,pggraph,bm25`, `--queries-file`, and `--output`. It should replay Eventloom into each backend, run identical query workloads, and emit JSON plus markdown summary with latency, quality, citation, and token metrics.

- [x] **Step 4: Run script smoke test**

Run:

```bash
python scripts/backend-shootout.py --help
ruff check scripts/backend-shootout.py tests/test_docs_site.py
PYTHONPATH=src pytest tests/test_docs_site.py::test_zero_friction_runtime_roadmap_sets_frontier_bar --no-cov -q
```

Expected: help renders, ruff clean, docs test passes.

## Task 6: Gate Review

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/zero-friction-runtime-roadmap.md`
- Modify: `docs/benchmarks.md`

- [x] **Step 1: Run full relevant verification**

Run:

```bash
ruff check src/zaxy tests scripts
mypy src/zaxy
PYTHONPATH=src pytest tests/test_projection_backends_embedded.py tests/test_embedded_graph_store.py tests/test_cli.py tests/test_docs_site.py --no-cov -q
```

Expected: checks pass, with Kuzu tests skipping only when Kuzu is not installed.

- [x] **Step 2: Record gate status**

Update the roadmap with one of these statuses:

```text
embedded prototype: blocked - Kuzu missing
embedded prototype: failed - quality gate regression
embedded prototype: passed - eligible for broader benchmark
```

- [x] **Step 3: Commit**

Run:

```bash
git add pyproject.toml src/zaxy tests docs scripts
git commit -m "feat: prototype embedded graph runtime"
```

Expected: commit succeeds only after the gate status is recorded.

## Self-Review

- Spec coverage: The plan covers embedded graph backend, activation/status, benchmark shootout, docs, and guardrails from `docs/zero-friction-runtime-roadmap.md`.
- Placeholder scan: No unresolved placeholder markers are intentionally left in the implementation steps.
- Type consistency: The plan uses the existing `ProjectionStore` contract and names the new backend `embedded` consistently.
