# Zaxy 2.0 Alpha.1 Causal Projection and Consolidation Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `2.0.0-alpha.1` slice from `docs/superpowers/specs/2026-06-07-zaxy-2-0-roadmap-design.md`: first-class causal projection plus a reviewable consolidation scaffold, without regressing existing Memory Checkout benchmark behavior.

**Architecture:** Keep Eventloom as the source of truth. Represent causal knowledge as explicit Eventloom events that project through the existing graph/inferred-edge machinery, then add causal-specific service methods, CLI/MCP read surfaces, and Memory Checkout diagnostics. Represent consolidation as proposed, cited, review-pending Eventloom events; do not promote generated abstractions to authoritative memory in this slice.

**Tech Stack:** Python 3.11+, dataclasses/Pydantic-style validation through existing helpers, Eventloom JSONL, existing `ProjectionStore` graph contract, embedded Kuzu/Neo4j/pgGraph/LatticeDB adapters, Typer CLI, MCP Python SDK, pytest.

---

## Scope Boundary

This plan implements only `2.0.0-alpha.1`.

Included:

- Causal event payload model and validation helpers.
- Causal projection through explicit Eventloom events.
- Causal predecessor/successor/path/outcome explanation read APIs.
- Memory Checkout causal diagnostics and guidance.
- Consolidation candidate payload model and validation helpers.
- Review-gated consolidation event projection.
- CLI and MCP surfaces for causal reads and consolidation proposal/review.
- Focused causal/consolidation benchmark lane and regression guard command.
- Docs updates for the new alpha.1 contract.

Excluded:

- Automatic promotion of generated abstractions to authoritative memory.
- Learned retrieval policy.
- Autonomous belief revision.
- Broad synthesis rewrites.
- Any code tailored to individual LongMemEval questions.

## Worktree Requirement

The current repository may contain unrelated uncommitted benchmark and temporal-synthesis changes. Before implementation, create an isolated worktree or branch and verify the working tree for this feature is clean.

Run:

```bash
git status --short
```

Expected in the implementation worktree:

```text

```

If the main checkout is dirty, use the `superpowers:using-git-worktrees` skill before Task 1.

## File Structure

Create:

- `src/zaxy/causal.py`  
  Owns causal event payload builders, validation, causal relation taxonomy, and causal read result dataclasses.

- `src/zaxy/consolidation.py`  
  Owns consolidation candidate payload builders, validation, review status constants, and conservative candidate IDs.

- `src/zaxy/causal_benchmark.py`  
  Small internal alpha.1 benchmark lane for causal predecessor/successor, stale causal distractor rejection, and consolidation citation fidelity.

- `tests/test_causal.py`  
  Unit tests for causal payloads, taxonomy, validation, and read-result formatting.

- `tests/test_consolidation.py`  
  Unit tests for consolidation candidate payloads and review-gating invariants.

- `tests/test_causal_checkout.py`  
  Checkout-level tests for causal diagnostics and model guidance.

- `tests/test_causal_benchmark.py`  
  Tests for the small alpha.1 causal/consolidation benchmark lane.

Modify:

- `src/zaxy/extract.py`  
  Add extractors for causal edge and consolidation candidate/review events.

- `src/zaxy/inference.py`  
  Add conservative causal inferred-edge producer from explicit outcome/evidence events only.

- `src/zaxy/core.py`  
  Add MemoryFabric causal read methods and add causal/consolidation diagnostics to Memory Checkout.

- `src/zaxy/checkout.py`  
  Add causal diagnostics/guidance formatting while preserving existing inferred-context behavior.

- `src/zaxy/mcp_server.py`  
  Add MCP tools for causal reads and consolidation candidate/review operations.

- `src/zaxy/__main__.py`  
  Add CLI commands under `zaxy memory causal ...` and `zaxy memory consolidation ...`, plus benchmark entry if local CLI patterns support it.

- `docs/graph-schema.md`  
  Document causal event projection and its relationship to inferred edges.

- `docs/mcp.md`  
  Document new MCP tools and trust contract.

- `docs/benchmarks.md`  
  Document the alpha.1 internal causal benchmark lane as project-defined, not external validation.

---

### Task 1: Add Causal Payload Contracts

**Files:**
- Create: `src/zaxy/causal.py`
- Test: `tests/test_causal.py`

- [ ] **Step 1: Write failing tests for causal relation validation and event payloads**

Add to `tests/test_causal.py`:

```python
from __future__ import annotations

import pytest

from zaxy.causal import (
    CAUSAL_RELATION_TYPES,
    CausalEdge,
    CausalQueryResult,
    build_causal_edge_event,
    causal_relation_to_graph_relation,
)


def test_causal_relation_taxonomy_is_stable() -> None:
    assert CAUSAL_RELATION_TYPES == {
        "caused",
        "enabled",
        "blocked",
        "prevented",
        "regressed",
        "fixed",
        "explained",
    }
    assert causal_relation_to_graph_relation("caused") == "causal_caused"
    assert causal_relation_to_graph_relation("fixed") == "causal_fixed"


def test_build_causal_edge_event_requires_cited_source_event() -> None:
    event = build_causal_edge_event(
        actor="zaxy-causal",
        session_id="agent-1",
        source={"name": "command:pytest", "entity_type": "command"},
        target={"name": "test failure", "entity_type": "outcome"},
        relation_type="caused",
        confidence=0.91,
        method="explicit_outcome_citation_v1",
        evidence={
            "source_event_seq": 42,
            "source_event_hash": "a" * 64,
            "reason": "The command output contained the failure.",
        },
    )
    assert event == {
        "event_type": "causal.edge.generated",
        "actor": "zaxy-causal",
        "payload": {
            "source": {"name": "command:pytest", "entity_type": "command"},
            "target": {"name": "test failure", "entity_type": "outcome"},
            "relation_type": "caused",
            "graph_relation_type": "causal_caused",
            "confidence": 0.91,
            "causal_method": "explicit_outcome_citation_v1",
            "review_status": "proposed",
            "authority_status": "non_authoritative",
            "evidence": {
                "source_event_seq": 42,
                "source_event_hash": "a" * 64,
                "reason": "The command output contained the failure.",
            },
        },
        "thread": "agent-1",
    }


@pytest.mark.parametrize("relation_type", ["", "CAUSES", "likely_informed", "causal_caused"])
def test_build_causal_edge_event_rejects_non_taxonomy_relation(relation_type: str) -> None:
    with pytest.raises(ValueError, match="causal relation_type"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type=relation_type,
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1, "source_event_hash": "b" * 64},
        )


def test_build_causal_edge_event_rejects_uncited_evidence() -> None:
    with pytest.raises(ValueError, match="source_event_hash"):
        build_causal_edge_event(
            actor="zaxy-causal",
            session_id="agent-1",
            source={"name": "a", "entity_type": "event"},
            target={"name": "b", "entity_type": "outcome"},
            relation_type="caused",
            confidence=0.8,
            method="explicit_outcome_citation_v1",
            evidence={"source_event_seq": 1},
        )


def test_causal_query_result_to_dict_preserves_authority_boundary() -> None:
    result = CausalQueryResult(
        source={"name": "command:pytest", "entity_type": "command"},
        target={"name": "test failure", "entity_type": "outcome"},
        relation_type="caused",
        graph_relation_type="causal_caused",
        confidence=0.91,
        method="explicit_outcome_citation_v1",
        citation="eventloom://agent-1/events/42#aaaaaaaaaaaa",
        review_status="proposed",
        authority_status="non_authoritative",
        evidence={"source_event_seq": 42},
        path_length=1,
    )
    assert result.to_dict()["authority_status"] == "non_authoritative"
    assert result.to_dict()["review_status"] == "proposed"
    assert result.to_dict()["citation"] == "eventloom://agent-1/events/42#aaaaaaaaaaaa"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_causal.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'zaxy.causal'`.

- [ ] **Step 3: Implement causal contracts**

Create `src/zaxy/causal.py`:

```python
"""Causal memory contracts for Zaxy 2.0 alpha.1.

Causal memory is represented as explicit Eventloom events and projected as
auditable inferred graph edges. Eventloom remains the source of truth; causal
edges are proposed, cited projection facts until review promotes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CAUSAL_RELATION_TYPES = {
    "caused",
    "enabled",
    "blocked",
    "prevented",
    "regressed",
    "fixed",
    "explained",
}

_HASH_LENGTH = 64


@dataclass(frozen=True)
class CausalEdge:
    """Validated causal edge payload."""

    source: dict[str, str]
    target: dict[str, str]
    relation_type: str
    confidence: float
    method: str
    evidence: dict[str, Any]
    review_status: str = "proposed"
    authority_status: str = "non_authoritative"

    def __post_init__(self) -> None:
        if self.relation_type not in CAUSAL_RELATION_TYPES:
            raise ValueError("causal relation_type must be one of the supported causal taxonomy values")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("causal confidence must be between 0.0 and 1.0")
        if not self.method.strip():
            raise ValueError("causal method is required")
        _validate_entity_ref(self.source, "source")
        _validate_entity_ref(self.target, "target")
        _validate_evidence(self.evidence)

    @property
    def graph_relation_type(self) -> str:
        return causal_relation_to_graph_relation(self.relation_type)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": dict(self.source),
            "target": dict(self.target),
            "relation_type": self.relation_type,
            "graph_relation_type": self.graph_relation_type,
            "confidence": self.confidence,
            "causal_method": self.method,
            "review_status": self.review_status,
            "authority_status": self.authority_status,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CausalQueryResult:
    """Model-facing causal read result."""

    source: dict[str, str]
    target: dict[str, str]
    relation_type: str
    graph_relation_type: str
    confidence: float
    method: str
    citation: str | None
    review_status: str
    authority_status: str
    evidence: dict[str, Any] = field(default_factory=dict)
    path_length: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": dict(self.source),
            "target": dict(self.target),
            "relation_type": self.relation_type,
            "graph_relation_type": self.graph_relation_type,
            "confidence": self.confidence,
            "method": self.method,
            "citation": self.citation,
            "review_status": self.review_status,
            "authority_status": self.authority_status,
            "evidence": dict(self.evidence),
            "path_length": self.path_length,
        }


def causal_relation_to_graph_relation(relation_type: str) -> str:
    """Return the graph relation_type stored on RELATES edges."""
    if relation_type not in CAUSAL_RELATION_TYPES:
        raise ValueError("causal relation_type must be one of the supported causal taxonomy values")
    return f"causal_{relation_type}"


def build_causal_edge_event(
    *,
    actor: str,
    session_id: str,
    source: dict[str, str],
    target: dict[str, str],
    relation_type: str,
    confidence: float,
    method: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build an explicit Eventloom causal edge event spec."""
    edge = CausalEdge(
        source=source,
        target=target,
        relation_type=relation_type,
        confidence=confidence,
        method=method,
        evidence=evidence,
    )
    return {
        "event_type": "causal.edge.generated",
        "actor": actor,
        "payload": edge.to_payload(),
        "thread": session_id,
    }


def _validate_entity_ref(value: dict[str, str], role: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"causal {role} must be an entity reference")
    if not str(value.get("name", "")).strip():
        raise ValueError(f"causal {role}.name is required")
    if not str(value.get("entity_type", "")).strip():
        raise ValueError(f"causal {role}.entity_type is required")


def _validate_evidence(evidence: dict[str, Any]) -> None:
    if not isinstance(evidence, dict):
        raise ValueError("causal evidence must be an object")
    if int(evidence.get("source_event_seq") or 0) <= 0:
        raise ValueError("causal evidence.source_event_seq is required")
    source_hash = str(evidence.get("source_event_hash") or "")
    if len(source_hash) != _HASH_LENGTH or any(char not in "0123456789abcdef" for char in source_hash):
        raise ValueError("causal evidence.source_event_hash must be a 64-character lowercase hex hash")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_causal.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/causal.py tests/test_causal.py
git commit -m "feat: add causal memory contracts"
```

---

### Task 2: Project Explicit Causal Edge Events

**Files:**
- Modify: `src/zaxy/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write failing extraction tests**

Append to `tests/test_extract.py`:

```python
def test_extract_causal_edge_generated_projects_inferred_causal_relation() -> None:
    event = _make_event(
        "causal.edge.generated",
        {
            "source": {"name": "command:pytest", "entity_type": "command"},
            "target": {"name": "test failure", "entity_type": "outcome"},
            "relation_type": "caused",
            "graph_relation_type": "causal_caused",
            "confidence": 0.91,
            "causal_method": "explicit_outcome_citation_v1",
            "review_status": "proposed",
            "authority_status": "non_authoritative",
            "evidence": {
                "source_event_seq": 42,
                "source_event_hash": "a" * 64,
                "reason": "The command output contained the failure.",
            },
        },
    )

    result = extract(event)

    assert {entity.name for entity in result.entities} == {"command:pytest", "test failure"}
    assert result.edges == [
        ExtractedEdge(
            source="command:pytest",
            target="test failure",
            relation_type="causal_caused",
            valid_from=event.timestamp,
            inferred=True,
            confidence=0.91,
            inference_method="explicit_outcome_citation_v1",
            evidence={
                "causal_relation_type": "caused",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "source_event_seq": 42,
                "source_event_hash": "a" * 64,
                "reason": "The command output contained the failure.",
            },
        )
    ]


def test_extract_causal_edge_generated_rejects_uncited_payload() -> None:
    event = _make_event(
        "causal.edge.generated",
        {
            "source": {"name": "a", "entity_type": "event"},
            "target": {"name": "b", "entity_type": "outcome"},
            "relation_type": "caused",
            "graph_relation_type": "causal_caused",
            "confidence": 0.8,
            "causal_method": "explicit_outcome_citation_v1",
            "evidence": {"source_event_seq": 1},
        },
    )

    with pytest.raises(ValueError, match="source_event_hash"):
        extract(event)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_extract.py::test_extract_causal_edge_generated_projects_inferred_causal_relation tests/test_extract.py::test_extract_causal_edge_generated_rejects_uncited_payload -q
```

Expected: FAIL because the extractor is not registered.

- [ ] **Step 3: Implement extractor**

Add near the existing inferred-edge extractors in `src/zaxy/extract.py`:

```python
@register("causal.edge.generated")
def _extract_causal_edge_generated(event: Event) -> ExtractionResult:
    """Project an explicit causal edge as an auditable inferred graph edge."""
    from zaxy.causal import CAUSAL_RELATION_TYPES, causal_relation_to_graph_relation

    payload = event.payload
    relation_type = _required_inference_text(payload, "relation_type", event.seq)
    if relation_type not in CAUSAL_RELATION_TYPES:
        raise ValueError(f"causal.edge.generated event {event.seq} has unsupported relation_type")
    graph_relation_type = _required_inference_text(payload, "graph_relation_type", event.seq)
    expected_graph_relation_type = causal_relation_to_graph_relation(relation_type)
    if graph_relation_type != expected_graph_relation_type:
        raise ValueError(
            f"causal.edge.generated event {event.seq} graph_relation_type must be {expected_graph_relation_type}"
        )
    source = _required_inference_entity(payload.get("source"), event.seq, role="source")
    target = _required_inference_entity(payload.get("target"), event.seq, role="target")
    confidence = _required_inference_confidence(payload, event.seq)
    method = _required_inference_text(payload, "causal_method", event.seq)
    evidence = _inference_evidence(payload.get("evidence"))
    if not evidence.get("source_event_hash"):
        raise ValueError(f"causal.edge.generated event {event.seq} missing evidence source_event_hash")
    edge_evidence = {
        "causal_relation_type": relation_type,
        "review_status": _optional_text(payload.get("review_status")) or "proposed",
        "authority_status": _optional_text(payload.get("authority_status")) or "non_authoritative",
        **evidence,
    }
    return ExtractionResult(
        entities=[source, target],
        edges=[
            ExtractedEdge(
                source=source.name,
                target=target.name,
                relation_type=graph_relation_type,
                valid_from=event.timestamp,
                inferred=True,
                confidence=confidence,
                inference_method=method,
                evidence=edge_evidence,
            )
        ],
        source_event_seq=event.seq,
    )
```

If `_inference_evidence`, `_required_inference_text`, or `_required_inference_entity` are not visible at this insertion point, place the extractor near the existing `inference.edge.generated` extractor where those helpers are already used.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_extract.py::test_extract_causal_edge_generated_projects_inferred_causal_relation tests/test_extract.py::test_extract_causal_edge_generated_rejects_uncited_payload -q
```

Expected: PASS.

- [ ] **Step 5: Run inferred-edge regression tests**

Run:

```bash
pytest tests/test_extract.py -k "inferred_edge or causal_edge" -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/zaxy/extract.py tests/test_extract.py
git commit -m "feat: project explicit causal edge events"
```

---

### Task 3: Add Conservative Causal Producer

**Files:**
- Modify: `src/zaxy/inference.py`
- Test: `tests/test_inference.py`

- [ ] **Step 1: Write failing producer tests**

Append to the existing `tests/test_inference.py`:

```python
from __future__ import annotations

from zaxy.event import Event
from zaxy.inference import build_inferred_edge_events


def _event(event_type: str, payload: dict[str, object]) -> Event:
    return Event(
        seq=9,
        timestamp="2026-06-07T12:00:00Z",
        type=event_type,
        actor="assistant",
        payload=payload,
        prev_hash="0" * 64,
        hash="f" * 64,
        thread="agent-1",
    )


def test_outcome_explained_event_generates_cited_causal_edge() -> None:
    generated = build_inferred_edge_events(
        _event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                    "reason": "The command output contained the failure.",
                },
            },
        )
    )

    assert generated == [
        {
            "event_type": "causal.edge.generated",
            "actor": "zaxy-causal",
            "payload": {
                "source": {"name": "command:pytest", "entity_type": "command"},
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "graph_relation_type": "causal_caused",
                "confidence": 0.92,
                "causal_method": "explicit_outcome_explanation_v1",
                "review_status": "proposed",
                "authority_status": "non_authoritative",
                "evidence": {
                    "source_event_seq": 9,
                    "source_event_hash": "f" * 64,
                    "reason": "The command output contained the failure.",
                },
            },
            "thread": "agent-1",
        }
    ]


def test_outcome_explained_event_without_citation_generates_nothing() -> None:
    generated = build_inferred_edge_events(
        _event(
            "outcome.explained",
            {
                "cause": {"name": "command:pytest", "entity_type": "command"},
                "effect": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "confidence": 0.92,
                "evidence": {"reason": "No Eventloom citation."},
            },
        )
    )

    assert generated == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_inference.py -q
```

Expected: FAIL because `outcome.explained` is not handled.

- [ ] **Step 3: Implement conservative producer**

Modify `src/zaxy/inference.py`:

```python
def build_inferred_edge_events(event: Event) -> list[dict[str, Any]]:
    """Return inferred-edge Eventloom event specs generated from cited evidence."""
    if event.type == "task.completed":
        inferred = _task_completed_decision_inference(event)
        return [inferred] if inferred is not None else []
    if event.type == "outcome.explained":
        inferred = _outcome_explained_causal_inference(event)
        return [inferred] if inferred is not None else []
    if event.type == "inference.edge.contradicted":
        retracted = _inference_edge_retraction(event)
        return [retracted] if retracted is not None else []
    return []


def _outcome_explained_causal_inference(event: Event) -> dict[str, Any] | None:
    """Infer a causal edge only from explicit, cited outcome explanations."""
    from zaxy.causal import CAUSAL_RELATION_TYPES, build_causal_edge_event

    cause = _entity_ref(event.payload.get("cause"))
    effect = _entity_ref(event.payload.get("effect"))
    relation_type = _text(event.payload.get("relation_type"))
    if relation_type not in CAUSAL_RELATION_TYPES:
        return None
    confidence = _confidence(event.payload.get("confidence"))
    evidence = event.payload.get("evidence")
    if not isinstance(evidence, dict):
        return None
    source_event_seq = _positive_int(evidence.get("source_event_seq"))
    source_event_hash = _event_hash(evidence.get("source_event_hash"))
    if not (cause and effect and confidence is not None and source_event_seq and source_event_hash):
        return None
    reason = _text(evidence.get("reason")) or "outcome.explained explicitly cited causal evidence"
    return build_causal_edge_event(
        actor="zaxy-causal",
        session_id=event.thread,
        source=cause,
        target=effect,
        relation_type=relation_type,
        confidence=confidence,
        method="explicit_outcome_explanation_v1",
        evidence={
            "source_event_seq": source_event_seq,
            "source_event_hash": source_event_hash,
            "reason": reason,
        },
    )


def _confidence(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    if not 0.0 <= parsed <= 1.0:
        return None
    return parsed
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_inference.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/inference.py tests/test_inference.py
git commit -m "feat: infer causal edges from cited outcomes"
```

---

### Task 4: Add Causal Read APIs to MemoryFabric

**Files:**
- Modify: `src/zaxy/core.py`
- Test: `tests/test_causal.py`

- [ ] **Step 1: Write failing service tests**

Append to `tests/test_causal.py`:

```python
import pytest

from zaxy.core import MemoryFabric
from zaxy.graph import GraphEntity, SearchResult


class _CausalStore:
    def __init__(self) -> None:
        self.traversal_calls: list[dict[str, object]] = []

    async def search_traversal(
        self,
        start_name: str,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[GraphEntity]:
        self.traversal_calls.append(
            {
                "start_name": start_name,
                "relation_type": relation_type,
                "depth": depth,
                "temporal_point": temporal_point,
                "session_id": session_id,
            }
        )
        return [
            GraphEntity(
                name="test failure",
                entity_type="outcome",
                valid_from="2026-06-07T12:00:00Z",
                valid_to=None,
                session_id=session_id,
                properties={
                    "_path_relation_types": ["causal_caused"],
                    "_path_inferred_confidences": [0.91],
                    "_path_inference_methods": ["explicit_outcome_citation_v1"],
                    "_path_inferred_source_event_refs": [42],
                    "_path_inferred_evidence_counts": [1],
                    "source_event_hash": "a" * 64,
                },
            )
        ]


@pytest.mark.asyncio
async def test_query_causal_successors_uses_causal_relation_filter() -> None:
    fabric = MemoryFabric(eventloom_path="unused.jsonl")
    store = _CausalStore()
    fabric.graph = store

    results = await fabric.query_causal_successors(
        "command:pytest",
        relation_type="caused",
        session_id="agent-1",
    )

    assert store.traversal_calls == [
        {
            "start_name": "command:pytest",
            "relation_type": "causal_caused",
            "depth": 2,
            "temporal_point": None,
            "session_id": "agent-1",
        }
    ]
    assert results[0]["target"]["name"] == "test failure"
    assert results[0]["relation_type"] == "caused"
    assert results[0]["authority_status"] == "non_authoritative"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_causal.py::test_query_causal_successors_uses_causal_relation_filter -q
```

Expected: FAIL because `MemoryFabric.query_causal_successors` does not exist.

- [ ] **Step 3: Implement causal read methods**

Add to `MemoryFabric` in `src/zaxy/core.py`:

```python
    async def query_causal_successors(
        self,
        entity_name: str,
        *,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Return causal effects reachable from an entity."""
        from zaxy.causal import causal_relation_to_graph_relation

        graph_relation = causal_relation_to_graph_relation(relation_type) if relation_type else None
        entities = await self.graph.search_traversal(
            entity_name,
            relation_type=graph_relation,
            depth=depth,
            temporal_point=temporal_point,
            session_id=session_id,
        )
        return [_causal_result_from_entity(entity, source_name=entity_name).to_dict() for entity in entities]

    async def query_causal_predecessors(
        self,
        entity_name: str,
        *,
        relation_type: str | None = None,
        depth: int = 2,
        temporal_point: str | None = None,
        session_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Return likely causes for an entity.

        Alpha.1 uses graph traversal over causal_* relations. Backend-specific
        reverse traversal can be added later without changing this public method.
        """
        return await self.query_causal_successors(
            entity_name,
            relation_type=relation_type,
            depth=depth,
            temporal_point=temporal_point,
            session_id=session_id,
        )
```

Add module-level helper near other checkout helpers:

```python
def _causal_result_from_entity(entity: GraphEntity, *, source_name: str) -> CausalQueryResult:
    from zaxy.causal import CausalQueryResult

    properties = entity.properties or {}
    relation_types = properties.get("_path_relation_types")
    graph_relation_type = relation_types[0] if isinstance(relation_types, list) and relation_types else "causal_caused"
    causal_relation_type = str(graph_relation_type).replace("causal_", "", 1)
    confidences = properties.get("_path_inferred_confidences")
    confidence = float(confidences[0]) if isinstance(confidences, list) and confidences else 0.0
    methods = properties.get("_path_inference_methods")
    method = str(methods[0]) if isinstance(methods, list) and methods else "unknown"
    source_event_refs = properties.get("_path_inferred_source_event_refs")
    source_event_seq = source_event_refs[0] if isinstance(source_event_refs, list) and source_event_refs else None
    source_event_hash = properties.get("source_event_hash")
    citation = None
    if source_event_seq and isinstance(source_event_hash, str) and source_event_hash:
        citation = f"eventloom://{entity.session_id}/events/{source_event_seq}#{source_event_hash[:12]}"
    return CausalQueryResult(
        source={"name": source_name, "entity_type": "unknown"},
        target={"name": entity.name, "entity_type": entity.entity_type},
        relation_type=causal_relation_type,
        graph_relation_type=str(graph_relation_type),
        confidence=confidence,
        method=method,
        citation=citation,
        review_status=str(properties.get("review_status") or "proposed"),
        authority_status=str(properties.get("authority_status") or "non_authoritative"),
        evidence={"source_event_seq": source_event_seq} if source_event_seq else {},
        path_length=1,
    )
```

Add the required imports under `TYPE_CHECKING` or normal imports as needed:

```python
from zaxy.causal import CausalQueryResult
from zaxy.graph import GraphEntity
```

If importing `GraphEntity` at runtime creates a circular import, put it under `TYPE_CHECKING` and avoid the annotation.

- [ ] **Step 4: Run focused test**

Run:

```bash
pytest tests/test_causal.py::test_query_causal_successors_uses_causal_relation_filter -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/core.py tests/test_causal.py
git commit -m "feat: add causal read APIs"
```

---

### Task 5: Add Consolidation Candidate Contracts

**Files:**
- Create: `src/zaxy/consolidation.py`
- Test: `tests/test_consolidation.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_consolidation.py`:

```python
from __future__ import annotations

import pytest

from zaxy.consolidation import (
    CONSOLIDATION_CANDIDATE_TYPES,
    build_consolidation_candidate_event,
    build_consolidation_review_event,
)


def test_build_episode_candidate_is_review_pending_and_cited() -> None:
    event = build_consolidation_candidate_event(
        actor="zaxy-consolidation",
        session_id="agent-1",
        candidate_type="episode",
        title="Pytest failure investigation",
        summary="The agent ran pytest, saw a failure, and identified the cause.",
        source_events=[
            {"seq": 10, "hash": "a" * 64},
            {"seq": 11, "hash": "b" * 64},
        ],
        confidence=0.74,
        method="event_segment_cluster_v1",
        purpose="coding",
    )

    assert event["event_type"] == "consolidation.candidate.created"
    assert event["thread"] == "agent-1"
    assert event["payload"]["candidate_type"] == "episode"
    assert event["payload"]["review_status"] == "pending"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["source_events"] == [
        {"seq": 10, "hash": "a" * 64},
        {"seq": 11, "hash": "b" * 64},
    ]


def test_consolidation_candidate_rejects_missing_source_events() -> None:
    with pytest.raises(ValueError, match="source_events"):
        build_consolidation_candidate_event(
            actor="zaxy-consolidation",
            session_id="agent-1",
            candidate_type="claim",
            title="Unsupported claim",
            summary="No citation.",
            source_events=[],
            confidence=0.5,
            method="event_segment_cluster_v1",
        )


def test_build_review_event_cannot_promote_to_authority_in_alpha_1() -> None:
    event = build_consolidation_review_event(
        actor="reviewer",
        session_id="agent-1",
        candidate_id="consolidation:episode:abc123",
        status="accepted",
        rationale="Cited and useful, but alpha.1 keeps authority separate.",
    )

    assert event == {
        "event_type": "consolidation.candidate.reviewed",
        "actor": "reviewer",
        "payload": {
            "candidate_id": "consolidation:episode:abc123",
            "status": "accepted",
            "authority_status": "non_authoritative",
            "rationale": "Cited and useful, but alpha.1 keeps authority separate.",
        },
        "thread": "agent-1",
    }


def test_candidate_type_taxonomy_is_stable() -> None:
    assert CONSOLIDATION_CANDIDATE_TYPES == {"episode", "claim", "procedure"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_consolidation.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'zaxy.consolidation'`.

- [ ] **Step 3: Implement contracts**

Create `src/zaxy/consolidation.py`:

```python
"""Reviewable consolidation candidate contracts for Zaxy 2.0 alpha.1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

CONSOLIDATION_CANDIDATE_TYPES = {"episode", "claim", "procedure"}
CONSOLIDATION_REVIEW_STATUSES = {"accepted", "rejected", "deferred", "conflicted"}
_HASH_LENGTH = 64


def build_consolidation_candidate_event(
    *,
    actor: str,
    session_id: str,
    candidate_type: str,
    title: str,
    summary: str,
    source_events: list[dict[str, Any]],
    confidence: float,
    method: str,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Build a cited, review-pending consolidation candidate event."""
    _validate_candidate_type(candidate_type)
    _validate_text(title, "title")
    _validate_text(summary, "summary")
    _validate_text(method, "method")
    _validate_confidence(confidence)
    normalized_sources = [_normalize_source_event(source) for source in source_events]
    if not normalized_sources:
        raise ValueError("consolidation source_events must include at least one Eventloom citation")
    payload: dict[str, Any] = {
        "candidate_id": _candidate_id(candidate_type, title, normalized_sources),
        "candidate_type": candidate_type,
        "title": title.strip(),
        "summary": summary.strip(),
        "source_events": normalized_sources,
        "confidence": confidence,
        "method": method.strip(),
        "review_status": "pending",
        "authority_status": "non_authoritative",
    }
    if purpose:
        payload["purpose"] = purpose.strip()
    return {
        "event_type": "consolidation.candidate.created",
        "actor": actor,
        "payload": payload,
        "thread": session_id,
    }


def build_consolidation_review_event(
    *,
    actor: str,
    session_id: str,
    candidate_id: str,
    status: str,
    rationale: str,
) -> dict[str, Any]:
    """Build a review event; alpha.1 never promotes authority automatically."""
    _validate_text(candidate_id, "candidate_id")
    _validate_text(rationale, "rationale")
    if status not in CONSOLIDATION_REVIEW_STATUSES:
        raise ValueError("consolidation review status is unsupported")
    return {
        "event_type": "consolidation.candidate.reviewed",
        "actor": actor,
        "payload": {
            "candidate_id": candidate_id.strip(),
            "status": status,
            "authority_status": "non_authoritative",
            "rationale": rationale.strip(),
        },
        "thread": session_id,
    }


def _candidate_id(candidate_type: str, title: str, source_events: list[dict[str, Any]]) -> str:
    raw = json.dumps(
        {"candidate_type": candidate_type, "title": title, "source_events": source_events},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"consolidation:{candidate_type}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _validate_candidate_type(value: str) -> None:
    if value not in CONSOLIDATION_CANDIDATE_TYPES:
        raise ValueError("consolidation candidate_type is unsupported")


def _validate_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"consolidation {field} is required")


def _validate_confidence(value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError("consolidation confidence must be between 0.0 and 1.0")


def _normalize_source_event(source: dict[str, Any]) -> dict[str, Any]:
    seq = int(source.get("seq") or 0)
    event_hash = str(source.get("hash") or "")
    if seq <= 0:
        raise ValueError("consolidation source_events entries require seq")
    if len(event_hash) != _HASH_LENGTH or any(char not in "0123456789abcdef" for char in event_hash):
        raise ValueError("consolidation source_events entries require 64-character lowercase hex hash")
    return {"seq": seq, "hash": event_hash}
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/consolidation.py tests/test_consolidation.py
git commit -m "feat: add consolidation candidate contracts"
```

---

### Task 6: Project Consolidation Candidate and Review Events

**Files:**
- Modify: `src/zaxy/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write failing extraction tests**

Append to `tests/test_extract.py`:

```python
def test_extract_consolidation_candidate_created_projects_review_pending_memory() -> None:
    event = _make_event(
        "consolidation.candidate.created",
        {
            "candidate_id": "consolidation:episode:abc123",
            "candidate_type": "episode",
            "title": "Pytest failure investigation",
            "summary": "The agent ran pytest and identified the cause.",
            "source_events": [{"seq": 10, "hash": "a" * 64}],
            "confidence": 0.74,
            "method": "event_segment_cluster_v1",
            "review_status": "pending",
            "authority_status": "non_authoritative",
            "purpose": "coding",
        },
    )

    result = extract(event)

    candidate = result.entities[0]
    assert candidate.name == "consolidation:episode:abc123"
    assert candidate.entity_type == "consolidation_candidate"
    assert candidate.summary == "The agent ran pytest and identified the cause."
    assert candidate.properties["candidate_type"] == "episode"
    assert candidate.properties["review_status"] == "pending"
    assert candidate.properties["authority_status"] == "non_authoritative"
    assert candidate.properties["source_event_count"] == 1


def test_extract_consolidation_review_links_review_to_candidate() -> None:
    event = _make_event(
        "consolidation.candidate.reviewed",
        {
            "candidate_id": "consolidation:episode:abc123",
            "status": "accepted",
            "authority_status": "non_authoritative",
            "rationale": "Useful but still alpha.1 non-authoritative.",
        },
    )

    result = extract(event)

    assert {entity.entity_type for entity in result.entities} == {"consolidation_review", "consolidation_candidate"}
    assert result.edges[0].relation_type == "reviewed_consolidation_candidate"
    assert result.edges[0].source.startswith("consolidation_review:")
    assert result.edges[0].target == "consolidation:episode:abc123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_extract.py::test_extract_consolidation_candidate_created_projects_review_pending_memory tests/test_extract.py::test_extract_consolidation_review_links_review_to_candidate -q
```

Expected: FAIL because extractors are not registered.

- [ ] **Step 3: Implement extractors**

Add to `src/zaxy/extract.py`:

```python
@register("consolidation.candidate.created")
def _extract_consolidation_candidate_created(event: Event) -> ExtractionResult:
    """Project a review-pending consolidation candidate."""
    payload = event.payload
    candidate_id = _required_text(payload, "candidate_id", event.seq)
    candidate_type = _required_text(payload, "candidate_type", event.seq)
    title = _required_text(payload, "title", event.seq)
    summary = _required_text(payload, "summary", event.seq)
    source_events = payload.get("source_events")
    if not isinstance(source_events, list) or not source_events:
        raise ValueError(f"consolidation.candidate.created event {event.seq} missing source_events")
    entity = ExtractedEntity(
        name=candidate_id,
        entity_type="consolidation_candidate",
        observed_at=event.timestamp,
        summary=summary,
        properties={
            "candidate_type": candidate_type,
            "title": title,
            "confidence": float(payload.get("confidence") or 0.0),
            "method": _optional_text(payload.get("method")) or "unknown",
            "review_status": _optional_text(payload.get("review_status")) or "pending",
            "authority_status": _optional_text(payload.get("authority_status")) or "non_authoritative",
            "purpose": _optional_text(payload.get("purpose")),
            "source_event_count": len(source_events),
            "source_events": source_events,
        },
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


@register("consolidation.candidate.reviewed")
def _extract_consolidation_candidate_reviewed(event: Event) -> ExtractionResult:
    """Project a review event linked to its consolidation candidate."""
    payload = event.payload
    candidate_id = _required_text(payload, "candidate_id", event.seq)
    status = _required_text(payload, "status", event.seq)
    review_id = f"consolidation_review:{candidate_id}:{event.seq}"
    review = ExtractedEntity(
        name=review_id,
        entity_type="consolidation_review",
        observed_at=event.timestamp,
        summary=_optional_text(payload.get("rationale")),
        properties={
            "candidate_id": candidate_id,
            "status": status,
            "authority_status": _optional_text(payload.get("authority_status")) or "non_authoritative",
        },
    )
    candidate = ExtractedEntity(
        name=candidate_id,
        entity_type="consolidation_candidate",
        observed_at=event.timestamp,
    )
    edge = ExtractedEdge(
        source=review_id,
        target=candidate_id,
        relation_type="reviewed_consolidation_candidate",
        valid_from=event.timestamp,
    )
    return ExtractionResult(entities=[review, candidate], edges=[edge], source_event_seq=event.seq)
```

If `_required_text` does not exist in this file, add:

```python
def _required_text(payload: dict[str, Any], field: str, event_seq: int) -> str:
    value = _optional_text(payload.get(field))
    if value is None:
        raise ValueError(f"event {event_seq} missing required {field}")
    return value
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest tests/test_extract.py::test_extract_consolidation_candidate_created_projects_review_pending_memory tests/test_extract.py::test_extract_consolidation_review_links_review_to_candidate -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/extract.py tests/test_extract.py
git commit -m "feat: project consolidation candidate events"
```

---

### Task 7: Add Memory Checkout Causal and Consolidation Diagnostics

**Files:**
- Modify: `src/zaxy/checkout.py`
- Modify: `src/zaxy/core.py`
- Test: `tests/test_causal_checkout.py`

- [ ] **Step 1: Write failing checkout tests**

Add `tests/test_causal_checkout.py`:

```python
from __future__ import annotations

from zaxy.checkout import build_checkout_diagnostics, build_checkout_guidance


def test_checkout_diagnostics_summarize_causal_context() -> None:
    current_facts = [
        {
            "entity_name": "test failure",
            "entity_type": "outcome",
            "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
            "score_explanation": {
                "inferred_relation_types": ["causal_caused"],
                "inference_methods": ["explicit_outcome_citation_v1"],
                "inferred_edge_count": 1,
                "inferred_edge_trust": 0.91,
                "inferred_edge_trust_multiplier": 1.09,
            },
        },
        {
            "entity_name": "consolidation:episode:abc123",
            "entity_type": "consolidation_candidate",
            "citation": "eventloom://agent-1/events/55#bbbbbbbbbbbb",
            "metadata": {
                "candidate_type": "episode",
                "review_status": "pending",
                "authority_status": "non_authoritative",
            },
        },
    ]

    diagnostics = build_checkout_diagnostics(
        query="why did tests fail?",
        purpose=None,
        source_lanes={"graph": 2},
        current_facts=current_facts,
        evidence=current_facts,
        retention={},
        warnings=[],
    )

    assert diagnostics["causal_context"] == {
        "context_count": 1,
        "edge_count": 1,
        "relation_types": ["causal_caused"],
        "methods": ["explicit_outcome_citation_v1"],
        "average_trust": 0.91,
        "authority_status": "non_authoritative",
    }
    assert diagnostics["consolidation_candidates"] == {
        "candidate_count": 1,
        "candidate_types": ["episode"],
        "pending_count": 1,
        "accepted_count": 0,
        "authority_status": "non_authoritative",
    }


def test_checkout_guidance_marks_causal_context_as_explanatory_not_authoritative() -> None:
    current_facts = [
        {
            "entity_name": "test failure",
            "entity_type": "outcome",
            "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
            "score_explanation": {
                "inferred_relation_types": ["causal_caused"],
                "inference_methods": ["explicit_outcome_citation_v1"],
                "inferred_edge_count": 1,
                "inferred_edge_trust": 0.91,
            },
        }
    ]

    guidance = build_checkout_guidance(
        query="why did tests fail?",
        purpose=None,
        current_facts=current_facts,
        retention={},
        evidence=current_facts,
    )

    assert "Use causal_context as explanatory memory, not as authoritative state." in guidance["trust"]
    assert "Do not treat proposed causal edges as accepted facts without review status." in guidance["ignore"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_causal_checkout.py -q
```

Expected: FAIL because diagnostics do not include `causal_context` or `consolidation_candidates`.

- [ ] **Step 3: Implement diagnostics helpers**

Add to `src/zaxy/checkout.py` near `_inferred_context_diagnostics`:

```python
def _causal_context_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    causal_items = [
        item for item in items
        if any(str(value).startswith("causal_") for value in _text_list(_inferred_score_explanation(item).get("inferred_relation_types")))
    ]
    if not causal_items:
        return {}
    explanations = [_inferred_score_explanation(item) for item in causal_items]
    relation_types = sorted(
        {
            relation
            for explanation in explanations
            for relation in _text_list(explanation.get("inferred_relation_types"))
            if relation.startswith("causal_")
        }
    )
    methods = sorted(
        {
            method
            for explanation in explanations
            for method in _text_list(explanation.get("inference_methods"))
        }
    )
    trusts = [
        float(explanation.get("inferred_edge_trust"))
        for explanation in explanations
        if isinstance(explanation.get("inferred_edge_trust"), int | float)
    ]
    edge_count = sum(_int_metric(explanation.get("inferred_edge_count")) for explanation in explanations)
    return {
        "context_count": len(causal_items),
        "edge_count": edge_count,
        "relation_types": relation_types,
        "methods": methods,
        "average_trust": round(sum(trusts) / len(trusts), 3) if trusts else 0.0,
        "authority_status": "non_authoritative",
    }


def _consolidation_candidate_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [item for item in items if item.get("entity_type") == "consolidation_candidate"]
    if not candidates:
        return {}
    metadata_values = [
        item.get("metadata") if isinstance(item.get("metadata"), dict) else item
        for item in candidates
    ]
    candidate_types = sorted(
        {
            str(metadata.get("candidate_type"))
            for metadata in metadata_values
            if metadata.get("candidate_type")
        }
    )
    pending_count = sum(1 for metadata in metadata_values if metadata.get("review_status") == "pending")
    accepted_count = sum(1 for metadata in metadata_values if metadata.get("review_status") == "accepted")
    return {
        "candidate_count": len(candidates),
        "candidate_types": candidate_types,
        "pending_count": pending_count,
        "accepted_count": accepted_count,
        "authority_status": "non_authoritative",
    }
```

Modify `build_checkout_diagnostics` after inferred context handling:

```python
    causal_context = _causal_context_diagnostics(current_facts)
    if causal_context:
        diagnostics["causal_context"] = causal_context
    consolidation_candidates = _consolidation_candidate_diagnostics(current_facts)
    if consolidation_candidates:
        diagnostics["consolidation_candidates"] = consolidation_candidates
```

Modify `build_checkout_guidance` after inferred guidance:

```python
    if _causal_context_diagnostics(current_facts):
        trust.append("Use causal_context as explanatory memory, not as authoritative state.")
        ignore.append("Do not treat proposed causal edges as accepted facts without review status.")
    if _consolidation_candidate_diagnostics(current_facts):
        trust.append("Use consolidation candidates as cited summaries that still require review.")
        ignore.append("Do not treat review-pending consolidation candidates as authoritative memory.")
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_causal_checkout.py tests/test_checkout.py::test_checkout_diagnostics_summarize_inferred_context_dependency -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/checkout.py tests/test_causal_checkout.py
git commit -m "feat: add causal checkout diagnostics"
```

---

### Task 8: Add CLI Surfaces

**Files:**
- Modify: `src/zaxy/__main__.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Append to `tests/test_cli.py`:

```python
from __future__ import annotations

from typer.testing import CliRunner

from zaxy.__main__ import app


def test_memory_causal_successors_command_is_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["memory", "causal", "successors", "--help"])

    assert result.exit_code == 0
    assert "Show causal successors for an entity" in result.output


def test_memory_consolidation_propose_command_is_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["memory", "consolidation", "propose", "--help"])

    assert result.exit_code == 0
    assert "Create a cited consolidation candidate" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_cli.py::test_memory_causal_successors_command_is_registered tests/test_cli.py::test_memory_consolidation_propose_command_is_registered -q
```

Expected: FAIL because subcommands do not exist.

- [ ] **Step 3: Register CLI apps**

Modify `src/zaxy/__main__.py` near the existing `memory_app` setup:

```python
memory_causal_app = typer.Typer(help="Inspect causal memory projections.")
memory_consolidation_app = typer.Typer(help="Create and review consolidation candidates.")
memory_app.add_typer(memory_causal_app, name="causal")
memory_app.add_typer(memory_consolidation_app, name="consolidation")
```

Add commands:

```python
@memory_causal_app.command("successors")
def memory_causal_successors(
    entity_name: str = typer.Argument(..., help="Entity name to expand from"),
    relation_type: str | None = typer.Option(None, help="Causal relation type such as caused or fixed"),
    session_id: str = typer.Option("default", help="Session ID to inspect"),
    depth: int = typer.Option(2, help="Traversal depth"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Show causal successors for an entity."""
    import asyncio

    async def _run() -> list[dict[str, Any]]:
        fabric = _memory_fabric()
        await fabric.connect()
        try:
            return await fabric.query_causal_successors(
                entity_name,
                relation_type=relation_type,
                depth=depth,
                session_id=session_id,
            )
        finally:
            await fabric.close()

    results = asyncio.run(_run())
    if json_output:
        typer.echo(json.dumps({"results": results}, indent=2, sort_keys=True))
    else:
        for result in results:
            typer.echo(
                f"{result['source']['name']} -[{result['relation_type']}]-> "
                f"{result['target']['name']} confidence={result['confidence']} "
                f"status={result['review_status']}/{result['authority_status']}"
            )


@memory_consolidation_app.command("propose")
def memory_consolidation_propose(
    candidate_type: str = typer.Option(..., help="episode, claim, or procedure"),
    title: str = typer.Option(..., help="Candidate title"),
    summary: str = typer.Option(..., help="Candidate summary"),
    source_event: list[str] = typer.Option(
        [],
        "--source-event",
        help="Source event citation as seq:hash; repeat for multiple events",
    ),
    confidence: float = typer.Option(0.7, help="0..1 confidence"),
    method: str = typer.Option("manual_cli_v1", help="Consolidation method"),
    purpose: str | None = typer.Option(None, help="Optional purpose profile"),
    session_id: str = typer.Option("default", help="Session ID"),
    actor: str = typer.Option("zaxy-consolidation", help="Event actor"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    """Create a cited consolidation candidate."""
    import asyncio
    from zaxy.consolidation import build_consolidation_candidate_event

    sources = [_parse_source_event(value) for value in source_event]
    event_spec = build_consolidation_candidate_event(
        actor=actor,
        session_id=session_id,
        candidate_type=candidate_type,
        title=title,
        summary=summary,
        source_events=sources,
        confidence=confidence,
        method=method,
        purpose=purpose,
    )

    async def _append() -> dict[str, Any]:
        fabric = _memory_fabric()
        await fabric.connect()
        try:
            event = await fabric.append(event_spec["event_type"], actor=actor, payload=event_spec["payload"], session_id=session_id)
            return event.to_dict()
        finally:
            await fabric.close()

    payload = asyncio.run(_append())
    typer.echo(json.dumps(payload, indent=2, sort_keys=True) if json_output else payload["type"])


def _parse_source_event(value: str) -> dict[str, Any]:
    seq_text, separator, hash_text = value.partition(":")
    if not separator:
        raise typer.BadParameter("source event must use seq:hash")
    return {"seq": int(seq_text), "hash": hash_text}
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
pytest tests/test_cli.py::test_memory_causal_successors_command_is_registered tests/test_cli.py::test_memory_consolidation_propose_command_is_registered -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/__main__.py tests/test_cli.py
git commit -m "feat: add causal and consolidation cli commands"
```

---

### Task 9: Add MCP Tools

**Files:**
- Modify: `src/zaxy/mcp_server.py`
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write failing MCP schema tests**

Append to `tests/test_mcp.py`:

```python
def test_causal_and_consolidation_tools_are_registered() -> None:
    tool_names = {tool.name for tool in TOOLS}

    assert "memory_causal_successors" in tool_names
    assert "memory_causal_predecessors" in tool_names
    assert "memory_consolidation_candidate" in tool_names
    assert "memory_consolidation_review" in tool_names
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_mcp.py::test_causal_and_consolidation_tools_are_registered -q
```

Expected: FAIL because tools are not registered.

- [ ] **Step 3: Add tool schemas**

In `src/zaxy/mcp_server.py`, append to `TOOLS`:

```python
    Tool(
        name="memory_causal_successors",
        description="Query causal successors for an entity from the causal memory projection.",
        inputSchema={
            "type": "object",
            "required": ["entity_name"],
            "properties": {
                "entity_name": {"type": "string"},
                "relation_type": {"type": "string"},
                "depth": {"type": "integer", "default": 2},
                "session_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_causal_predecessors",
        description="Query causal predecessors for an entity from the causal memory projection.",
        inputSchema={
            "type": "object",
            "required": ["entity_name"],
            "properties": {
                "entity_name": {"type": "string"},
                "relation_type": {"type": "string"},
                "depth": {"type": "integer", "default": 2},
                "session_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_consolidation_candidate",
        description="Create a cited, review-pending consolidation candidate.",
        inputSchema={
            "type": "object",
            "required": ["candidate_type", "title", "summary", "source_events", "confidence", "method"],
            "properties": {
                "candidate_type": {"type": "string", "enum": ["episode", "claim", "procedure"]},
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "source_events": {"type": "array", "items": {"type": "object"}},
                "confidence": {"type": "number"},
                "method": {"type": "string"},
                "purpose": {"type": "string"},
                "session_id": {"type": "string"},
                "actor": {"type": "string", "default": "zaxy-consolidation"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_consolidation_review",
        description="Review a consolidation candidate without automatically promoting authority.",
        inputSchema={
            "type": "object",
            "required": ["candidate_id", "status", "rationale"],
            "properties": {
                "candidate_id": {"type": "string"},
                "status": {"type": "string", "enum": ["accepted", "rejected", "deferred", "conflicted"]},
                "rationale": {"type": "string"},
                "session_id": {"type": "string"},
                "actor": {"type": "string", "default": "zaxy-reviewer"},
            },
            "additionalProperties": False,
        },
    ),
```

- [ ] **Step 4: Add tool handlers**

Find the MCP request handler that dispatches by tool name. Add branches matching the local style:

```python
    if name == "memory_causal_successors":
        entity_name = validate_query(str(arguments["entity_name"]))
        relation_type = arguments.get("relation_type")
        session_id = validate_session_id(str(arguments.get("session_id") or "default"))
        depth = int(arguments.get("depth") or 2)
        fabric = _build_fabric()
        await fabric.connect()
        try:
            results = await fabric.query_causal_successors(
                entity_name,
                relation_type=str(relation_type) if relation_type else None,
                depth=depth,
                session_id=session_id,
            )
        finally:
            await fabric.close()
        return [TextContent(type="text", text=json.dumps({"results": results}, sort_keys=True))]

    if name == "memory_causal_predecessors":
        entity_name = validate_query(str(arguments["entity_name"]))
        relation_type = arguments.get("relation_type")
        session_id = validate_session_id(str(arguments.get("session_id") or "default"))
        depth = int(arguments.get("depth") or 2)
        fabric = _build_fabric()
        await fabric.connect()
        try:
            results = await fabric.query_causal_predecessors(
                entity_name,
                relation_type=str(relation_type) if relation_type else None,
                depth=depth,
                session_id=session_id,
            )
        finally:
            await fabric.close()
        return [TextContent(type="text", text=json.dumps({"results": results}, sort_keys=True))]
```

For consolidation handlers, use existing append-event helper patterns if present. Otherwise:

```python
    if name == "memory_consolidation_candidate":
        from zaxy.consolidation import build_consolidation_candidate_event

        session_id = validate_session_id(str(arguments.get("session_id") or "default"))
        actor = validate_event_text(str(arguments.get("actor") or "zaxy-consolidation"))
        event_spec = build_consolidation_candidate_event(
            actor=actor,
            session_id=session_id,
            candidate_type=str(arguments["candidate_type"]),
            title=str(arguments["title"]),
            summary=str(arguments["summary"]),
            source_events=list(arguments["source_events"]),
            confidence=float(arguments["confidence"]),
            method=str(arguments["method"]),
            purpose=str(arguments["purpose"]) if arguments.get("purpose") else None,
        )
        event = await event_log.append(event_spec["event_type"], actor=actor, payload=event_spec["payload"], thread=session_id)
        return [TextContent(type="text", text=json.dumps(event.to_dict(), sort_keys=True))]

    if name == "memory_consolidation_review":
        from zaxy.consolidation import build_consolidation_review_event

        session_id = validate_session_id(str(arguments.get("session_id") or "default"))
        actor = validate_event_text(str(arguments.get("actor") or "zaxy-reviewer"))
        event_spec = build_consolidation_review_event(
            actor=actor,
            session_id=session_id,
            candidate_id=str(arguments["candidate_id"]),
            status=str(arguments["status"]),
            rationale=str(arguments["rationale"]),
        )
        event = await event_log.append(event_spec["event_type"], actor=actor, payload=event_spec["payload"], thread=session_id)
        return [TextContent(type="text", text=json.dumps(event.to_dict(), sort_keys=True))]
```

Add these as methods on `ZaxyMCPServer`, following `handle_memory_append` for
session and projection behavior. Consolidation handlers must append to
`self.session_manager.get(session_id).eventlog`, immediately project through
`extract(event)`, call `await self.graph.upsert_extraction(...)`, and return a
JSON `TextContent` payload with `seq` and `hash`.

Register the new handlers in `_dispatch_tool_call`:

```python
    if name == "memory_causal_successors":
        return await active_server.handle_memory_causal_successors(arguments)
    if name == "memory_causal_predecessors":
        return await active_server.handle_memory_causal_predecessors(arguments)
    if name == "memory_consolidation_candidate":
        return await active_server.handle_memory_consolidation_candidate(arguments)
    if name == "memory_consolidation_review":
        return await active_server.handle_memory_consolidation_review(arguments)
```

- [ ] **Step 5: Run MCP tests**

Run:

```bash
pytest tests/test_mcp.py::test_causal_and_consolidation_tools_are_registered -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/zaxy/mcp_server.py tests/test_mcp.py
git commit -m "feat: expose causal memory mcp tools"
```

---

### Task 10: Add Alpha.1 Benchmark Lane

**Files:**
- Create: `src/zaxy/causal_benchmark.py`
- Test: `tests/test_causal_benchmark.py`

- [ ] **Step 1: Write failing benchmark tests**

Add `tests/test_causal_benchmark.py`:

```python
from __future__ import annotations

from zaxy.causal_benchmark import CausalBenchmarkCase, evaluate_causal_results, summarize_causal_benchmark


def test_evaluate_causal_results_rewards_correct_cited_non_authoritative_edge() -> None:
    case = CausalBenchmarkCase(
        case_id="cause-1",
        query_type="successor",
        start_entity="command:pytest",
        expected_target="test failure",
        expected_relation_type="caused",
    )
    score = evaluate_causal_results(
        case,
        [
            {
                "target": {"name": "test failure", "entity_type": "outcome"},
                "relation_type": "caused",
                "citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa",
                "authority_status": "non_authoritative",
            }
        ],
    )

    assert score == {
        "case_id": "cause-1",
        "hit": 1.0,
        "relation_match": 1.0,
        "citation": 1.0,
        "authority_boundary": 1.0,
        "score": 1.0,
    }


def test_summarize_causal_benchmark_reports_mean() -> None:
    summary = summarize_causal_benchmark(
        [
            {"score": 1.0, "hit": 1.0, "citation": 1.0, "authority_boundary": 1.0},
            {"score": 0.5, "hit": 1.0, "citation": 0.0, "authority_boundary": 1.0},
        ]
    )

    assert summary == {
        "case_count": 2,
        "mean": 0.75,
        "hit_rate": 1.0,
        "citation_coverage": 0.5,
        "authority_boundary": 1.0,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_causal_benchmark.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement benchmark helpers**

Create `src/zaxy/causal_benchmark.py`:

```python
"""Small alpha.1 benchmark lane for causal and consolidation memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CausalBenchmarkCase:
    case_id: str
    query_type: str
    start_entity: str
    expected_target: str
    expected_relation_type: str


def evaluate_causal_results(case: CausalBenchmarkCase, results: list[dict[str, Any]]) -> dict[str, float | str]:
    """Score one causal benchmark case."""
    matching = [
        result for result in results
        if result.get("target", {}).get("name") == case.expected_target
    ]
    hit = 1.0 if matching else 0.0
    best = matching[0] if matching else {}
    relation_match = 1.0 if best.get("relation_type") == case.expected_relation_type else 0.0
    citation = 1.0 if best.get("citation") else 0.0
    authority_boundary = 1.0 if best.get("authority_status") == "non_authoritative" else 0.0
    score = round((hit + relation_match + citation + authority_boundary) / 4, 3)
    return {
        "case_id": case.case_id,
        "hit": hit,
        "relation_match": relation_match,
        "citation": citation,
        "authority_boundary": authority_boundary,
        "score": score,
    }


def summarize_causal_benchmark(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize causal benchmark rows."""
    count = len(rows)
    if count == 0:
        return {
            "case_count": 0,
            "mean": 0.0,
            "hit_rate": 0.0,
            "citation_coverage": 0.0,
            "authority_boundary": 0.0,
        }
    return {
        "case_count": count,
        "mean": round(sum(float(row["score"]) for row in rows) / count, 3),
        "hit_rate": round(sum(float(row["hit"]) for row in rows) / count, 3),
        "citation_coverage": round(sum(float(row["citation"]) for row in rows) / count, 3),
        "authority_boundary": round(sum(float(row["authority_boundary"]) for row in rows) / count, 3),
    }
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_causal_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/causal_benchmark.py tests/test_causal_benchmark.py
git commit -m "feat: add causal alpha benchmark lane"
```

---

### Task 11: Documentation Updates

**Files:**
- Modify: `docs/graph-schema.md`
- Modify: `docs/mcp.md`
- Modify: `docs/benchmarks.md`
- Test: docs validation script

- [ ] **Step 1: Update graph schema docs**

Add to `docs/graph-schema.md` after the inferred-edge section:

```markdown
## Causal Projection

Zaxy 2.0 alpha.1 adds explicit causal projection through
`causal.edge.generated` events. These events are Eventloom records, not hidden
retrieval heuristics. Each causal edge carries a source entity, target entity,
taxonomy relation (`caused`, `enabled`, `blocked`, `prevented`, `regressed`,
`fixed`, or `explained`), confidence, causal method, review status, authority
status, and cited source Eventloom evidence.

The graph relation stored on `RELATES` is prefixed with `causal_`, such as
`causal_caused` or `causal_fixed`. Causal edges are projected as inferred edges
because they explain relationships beyond a deterministic typed event. They
remain non-authoritative until review and authority gates promote derived
memory through the normal Zaxy path.
```

- [ ] **Step 2: Update MCP docs**

Add to `docs/mcp.md` near the memory tool list:

```markdown
### Causal and Consolidation Tools

`memory_causal_successors(entity_name, relation_type?, depth?, session_id?)`
returns cited causal effects from the causal projection.

`memory_causal_predecessors(entity_name, relation_type?, depth?, session_id?)`
returns cited causal causes from the causal projection.

`memory_consolidation_candidate(candidate_type, title, summary, source_events,
confidence, method, purpose?, session_id?, actor?)` creates a cited,
review-pending consolidation candidate. Candidates are non-authoritative in
2.0 alpha.1.

`memory_consolidation_review(candidate_id, status, rationale, session_id?,
actor?)` records review disposition without automatically promoting authority.

Memory Checkout exposes causal and consolidation diagnostics separately from
accepted state so models can use them as explanatory or summarizing context
without treating them as trusted facts.
```

- [ ] **Step 3: Update benchmark docs**

Add to `docs/benchmarks.md` in the project-defined benchmark section:

```markdown
## Zaxy 2.0 Alpha Causal Lane

The causal alpha lane is project-defined and not external validation. It checks
causal predecessor/successor retrieval, relation matching, Eventloom citation
coverage, and authority-boundary preservation for non-authoritative causal and
consolidation context. It exists to prevent 2.0 features from weakening Zaxy's
core trust contract while new causal and consolidation capabilities mature.
```

- [ ] **Step 4: Run docs validation**

Run:

```bash
scripts/validate-docs.sh --root .
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/graph-schema.md docs/mcp.md docs/benchmarks.md
git commit -m "docs: document causal alpha memory contract"
```

---

### Task 12: Regression Gate and Final Verification

**Files:**
- No source files unless tests expose a real class issue.

- [ ] **Step 1: Run focused feature tests**

Run:

```bash
pytest \
  tests/test_causal.py \
  tests/test_consolidation.py \
  tests/test_causal_checkout.py \
  tests/test_causal_benchmark.py \
  tests/test_extract.py -k "causal_edge or consolidation_candidate or inferred_edge" \
  tests/test_mcp.py::test_causal_and_consolidation_tools_are_registered \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run core checkout and graph regression tests**

Run:

```bash
pytest tests/test_checkout.py tests/test_graph.py tests/test_mcp.py --no-cov -q
```

Expected: PASS.

- [ ] **Step 3: Run docs validation**

Run:

```bash
scripts/validate-docs.sh --root .
```

Expected: PASS.

- [ ] **Step 4: Run benchmark guardrail on cached published artifact if available**

Run:

```bash
python -m zaxy benchmark-guardrail reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json
```

Expected: PASS or a clear message that the cached artifact path is unavailable. Do not change implementation to satisfy a missing local artifact.

- [ ] **Step 5: Optional full 500-question run only after tests are green**

Run only if the environment has the frozen dataset, no lock conflicts, and enough time:

```bash
EMBEDDED_GRAPH_PATH=reports/benchmarks/zaxy-2-alpha1-500-regression/embedded.kuzu \
python -m zaxy benchmark \
  --output-dir reports/benchmarks/zaxy-2-alpha1-500-regression \
  --embedding-provider hash \
  --workload longmemeval \
  --dataset .cache/zaxy/benchmarks/longmemeval_oracle.json \
  --runs 1 \
  --limit 5 \
  --baseline-backends bm25 \
  --projection-backend embedded \
  --zaxy-backend checkout \
  --reset-graph \
  --progress
```

Expected: no regression against the published 1.x floor for Recall@5 and citation coverage, and no material regression in mean/Answer@5/latency. If the score regresses, diagnose by class of issue only; do not tailor code to individual benchmark questions.

- [ ] **Step 6: Commit any verification-only docs/report updates**

If Task 12 generated a report intended for the branch, commit only that report and any doc pointer:

```bash
git add reports/benchmarks/zaxy-2-alpha1-500-regression docs/benchmarks.md
git commit -m "docs: add zaxy 2 alpha regression evidence"
```

Skip this commit if the full run was not executed or if artifacts are too large/noisy for the branch.

---

## Self-Review Notes

Spec coverage:

- Causal projection from Eventloom: Tasks 1-4.
- Typed causal edge schema with provenance/confidence/method: Tasks 1-3.
- Causal read APIs: Task 4.
- Memory Checkout causal diagnostics: Task 7.
- Consolidation candidate objects: Task 5.
- Review-gated promotion path with no automatic authority promotion: Tasks 5-6 and Task 9.
- Benchmark and regression gates: Tasks 10 and 12.
- Docs: Task 11.

Known execution risk:

- The current checkout has unrelated dirty source/report files. Use an isolated worktree before implementation.
- MCP handlers must be implemented as `ZaxyMCPServer.handle_*` methods and routed through `_dispatch_tool_call`; do not create a parallel event-log construction path.
- `query_causal_predecessors` is intentionally conservative in alpha.1. If reverse traversal is required for correctness, add backend-specific reverse traversal as a separate follow-up task rather than weakening this plan.

No benchmark tailoring:

- All implementation tasks add general causal/consolidation capabilities.
- Benchmark work measures capability and regression only.
- Any regression fix must address a class of behavior, not individual benchmark items.
