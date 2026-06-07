# Zaxy 2.0 Alpha.2 Review-Gated Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the alpha.1 consolidation scaffold into a usable review-gated consolidation MVP that proposes cited episodes, claims, and procedures from Eventloom history without silently promoting generated abstractions to authoritative memory.

**Architecture:** Eventloom remains the source of truth. Alpha.2 adds deterministic segment selection and candidate-generation services that append `consolidation.candidate.created` events through the existing alpha.1 contract, then projects and surfaces review state through the graph, MCP, CLI, checkout diagnostics, and internal guardrails. Generated candidates remain `authority_status=non_authoritative`; accepted reviews are dispositions, not promotion.

**Tech Stack:** Python 3.11+, Eventloom JSONL, existing `MemoryFabric`, `SessionManager`, graph projection backends, Typer CLI, MCP Python SDK, pytest, ruff.

---

## Scope Boundary

Included:

- Deterministic event segment selection from replayed Eventloom sessions.
- Reviewable episode candidates built from related event windows.
- Reviewable claim candidates built from cited source-event clusters.
- Reviewable procedure candidates built from successful or failed workflow traces.
- Candidate supersession, conflict, rejection, and stale diagnostics.
- CLI and MCP proposal/status surfaces.
- Checkout diagnostics for generated candidates without authority promotion.
- Internal alpha.2 guardrail tests for source-event fidelity and authority gating.

Excluded:

- Autonomous authority promotion.
- Learned consolidation policies.
- Cross-project global procedure sharing.
- LongMemEval-specific synthesis changes.
- Code tailored to individual benchmark cases.

## File Structure

Create:

- `src/zaxy/consolidation_pipeline.py`  
  Owns segment selection, deterministic candidate proposal, stale/conflict diagnostics, and append specs that reuse `zaxy.consolidation`.

- `tests/test_consolidation_pipeline.py`  
  Unit tests for segment selection, candidate generation, conflict/stale detection, and no-authority-promotion invariants.

- `src/zaxy/consolidation_benchmark.py`  
  Internal alpha.2 guardrail lane for consolidation proposal fidelity, review gating, stale rejection, and source-event coverage.

- `tests/test_consolidation_benchmark.py`  
  Tests for the alpha.2 guardrail scorer.

Modify:

- `src/zaxy/core.py`  
  Add `MemoryFabric.propose_consolidation_candidates(...)` and `MemoryFabric.consolidation_status(...)`.

- `src/zaxy/__main__.py`  
  Add CLI commands under `zaxy memory consolidation propose-from-log` and `zaxy memory consolidation status`.

- `src/zaxy/mcp_server.py`  
  Add MCP tools for consolidation proposal generation and status.

- `src/zaxy/checkout.py`  
  Extend consolidation diagnostics with generated/stale/conflict/supersession fields while preserving alpha.1 trust guidance.

- `docs/graph-schema.md`, `docs/mcp.md`, `docs/benchmarks.md`  
  Document the alpha.2 pipeline, new tools, and internal guardrail boundary.

---

### Task 1: Add Consolidation Pipeline Contracts

**Files:**
- Create: `src/zaxy/consolidation_pipeline.py`
- Test: `tests/test_consolidation_pipeline.py`

- [ ] **Step 1: Write failing contract tests**

Add `tests/test_consolidation_pipeline.py`:

```python
from __future__ import annotations

import pytest

from zaxy.consolidation_pipeline import (
    ConsolidationSegment,
    ProposedConsolidation,
    build_segment_id,
)


def test_segment_requires_cited_source_events() -> None:
    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000003",
        event_type_counts={"tool.call.completed": 2, "file.edit.applied": 1},
        source_events=[
            {"seq": 1, "hash": "a" * 64, "event_type": "tool.call.completed", "summary": "pytest failed"},
            {"seq": 2, "hash": "b" * 64, "event_type": "file.edit.applied", "summary": "patched checkout"},
            {"seq": 3, "hash": "c" * 64, "event_type": "tool.call.completed", "summary": "pytest passed"},
        ],
    )

    assert segment.source_event_refs == [
        "1:" + "a" * 64,
        "2:" + "b" * 64,
        "3:" + "c" * 64,
    ]
    assert segment.source_event_count == 3


def test_segment_rejects_missing_source_citations() -> None:
    with pytest.raises(ValueError, match="hash"):
        ConsolidationSegment(
            session_id="agent-1",
            segment_id="segment:agent-1:000001-000001",
            event_type_counts={"tool.call.completed": 1},
            source_events=[{"seq": 1, "hash": "short", "event_type": "tool.call.completed"}],
        )


def test_proposed_consolidation_preserves_non_authoritative_boundary() -> None:
    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000002",
        event_type_counts={"tool.call.completed": 2},
        source_events=[
            {"seq": 1, "hash": "a" * 64, "event_type": "tool.call.completed", "summary": "run failed"},
            {"seq": 2, "hash": "b" * 64, "event_type": "tool.call.completed", "summary": "run passed"},
        ],
    )
    proposal = ProposedConsolidation(
        segment=segment,
        candidate_type="episode",
        title="Test run recovery",
        summary="The agent observed a failed run and then a passing run.",
        confidence=0.72,
        method="deterministic_segment_summary_v1",
        purpose="coding",
    )

    event = proposal.to_candidate_event(actor="zaxy-consolidation")

    assert event["event_type"] == "consolidation.candidate.created"
    assert event["thread"] == "agent-1"
    assert event["payload"]["candidate_type"] == "episode"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["review_status"] == "pending"
    assert event["payload"]["source_events"] == [
        {"seq": 1, "hash": "a" * 64},
        {"seq": 2, "hash": "b" * 64},
    ]


def test_build_segment_id_is_stable_and_session_scoped() -> None:
    assert build_segment_id("agent-1", [3, 4, 9]) == "segment:agent-1:000003-000009"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_consolidation_pipeline.py --no-cov -q
```

Expected: fail with `ModuleNotFoundError: No module named 'zaxy.consolidation_pipeline'`.

- [ ] **Step 3: Implement contracts**

Create `src/zaxy/consolidation_pipeline.py` with:

```python
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zaxy.consolidation import build_consolidation_candidate_event
from zaxy.security import validate_session_id

_EVENT_HASH_LEN = 64


@dataclass(frozen=True)
class ConsolidationSegment:
    session_id: str
    segment_id: str
    event_type_counts: Mapping[str, int]
    source_events: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        validate_session_id(self.session_id)
        if not isinstance(self.segment_id, str) or not self.segment_id.startswith(f"segment:{self.session_id}:"):
            raise ValueError("segment_id must be session-scoped")
        if not self.source_events:
            raise ValueError("source_events must be non-empty")
        for index, event in enumerate(self.source_events):
            _validate_source_event(event, index=index)
        for event_type, count in self.event_type_counts.items():
            if not isinstance(event_type, str) or not event_type:
                raise ValueError("event_type_counts keys must be non-empty strings")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("event_type_counts values must be non-negative integers")

    @property
    def source_event_refs(self) -> list[str]:
        return [f"{event['seq']}:{event['hash']}" for event in self.source_events]

    @property
    def source_event_count(self) -> int:
        return len(self.source_events)

    def candidate_source_events(self) -> list[dict[str, Any]]:
        return [{"seq": int(event["seq"]), "hash": str(event["hash"])} for event in self.source_events]


@dataclass(frozen=True)
class ProposedConsolidation:
    segment: ConsolidationSegment
    candidate_type: str
    title: str
    summary: str
    confidence: float
    method: str
    purpose: str | None = None

    def to_candidate_event(self, *, actor: str) -> dict[str, Any]:
        return build_consolidation_candidate_event(
            actor=actor,
            session_id=self.segment.session_id,
            candidate_type=self.candidate_type,
            title=self.title,
            summary=self.summary,
            source_events=self.segment.candidate_source_events(),
            confidence=self.confidence,
            method=self.method,
            purpose=self.purpose,
        )


def build_segment_id(session_id: str, event_seqs: Sequence[int]) -> str:
    sid = validate_session_id(session_id)
    if not event_seqs:
        raise ValueError("event_seqs must be non-empty")
    seqs = sorted(_validate_seq(seq) for seq in event_seqs)
    return f"segment:{sid}:{seqs[0]:06d}-{seqs[-1]:06d}"


def event_type_counts(source_events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index, event in enumerate(source_events):
        _validate_source_event(event, index=index)
        event_type = event.get("event_type")
        if isinstance(event_type, str) and event_type:
            counts[event_type] += 1
    return dict(sorted(counts.items()))


def _validate_seq(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("event seq must be a positive integer")
    return value


def _validate_source_event(event: Mapping[str, Any], *, index: int) -> None:
    if not isinstance(event, Mapping):
        raise ValueError(f"source_events[{index}] must be a mapping")
    _validate_seq(event.get("seq"))
    event_hash = event.get("hash")
    if not isinstance(event_hash, str) or len(event_hash) != _EVENT_HASH_LEN:
        raise ValueError(f"source_events[{index}].hash must be 64 lowercase hex")
    if any(char not in "0123456789abcdef" for char in event_hash):
        raise ValueError(f"source_events[{index}].hash must be 64 lowercase hex")
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation_pipeline.py --no-cov -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/zaxy/consolidation_pipeline.py tests/test_consolidation_pipeline.py
git commit -m "feat: add consolidation pipeline contracts"
```

---

### Task 2: Select Deterministic Event Segments

**Files:**
- Modify: `src/zaxy/consolidation_pipeline.py`
- Test: `tests/test_consolidation_pipeline.py`

- [ ] **Step 1: Add failing segment-selection tests**

Append:

```python
from datetime import UTC, datetime


class EventLike:
    def __init__(self, seq: int, event_hash: str, event_type: str, payload: dict[str, object]) -> None:
        self.seq = seq
        self.hash = event_hash
        self.event_type = event_type
        self.payload = payload
        self.timestamp = datetime(2026, 6, 7, 12, seq, tzinfo=UTC)


def test_select_consolidation_segments_groups_adjacent_relevant_events() -> None:
    from zaxy.consolidation_pipeline import select_consolidation_segments

    events = [
        EventLike(1, "a" * 64, "tool.call.completed", {"tool_name": "pytest", "status": "failed"}),
        EventLike(2, "b" * 64, "file.edit.applied", {"path": "src/zaxy/checkout.py"}),
        EventLike(3, "c" * 64, "tool.call.completed", {"tool_name": "pytest", "status": "succeeded"}),
        EventLike(4, "d" * 64, "memory.checkout.completed", {"query": "unrelated"}),
    ]

    segments = select_consolidation_segments(events, session_id="agent-1", window_size=3)

    assert len(segments) == 1
    assert segments[0].segment_id == "segment:agent-1:000001-000003"
    assert segments[0].event_type_counts == {
        "file.edit.applied": 1,
        "tool.call.completed": 2,
    }
    assert [event["seq"] for event in segments[0].source_events] == [1, 2, 3]


def test_select_consolidation_segments_ignores_non_actionable_noise() -> None:
    from zaxy.consolidation_pipeline import select_consolidation_segments

    events = [
        EventLike(1, "a" * 64, "memory.checkout.completed", {"query": "status"}),
        EventLike(2, "b" * 64, "tool.call.completed", {"tool_name": "pytest", "status": "failed"}),
    ]

    segments = select_consolidation_segments(events, session_id="agent-1", window_size=2)

    assert len(segments) == 1
    assert [event["seq"] for event in segments[0].source_events] == [2]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k select --no-cov -q
```

Expected: fail because `select_consolidation_segments` is missing.

- [ ] **Step 3: Implement deterministic selector**

Add:

```python
ACTIONABLE_EVENT_TYPES = frozenset({
    "tool.call.completed",
    "command.completed",
    "command.result",
    "file.edit.applied",
    "task.completed",
    "coordination.handoff.created",
    "coordination.finding.reported",
})


def select_consolidation_segments(
    events: Sequence[Any],
    *,
    session_id: str,
    window_size: int = 8,
) -> list[ConsolidationSegment]:
    sid = validate_session_id(session_id)
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    source_events = [_source_event_from_event(event) for event in events]
    actionable = [event for event in source_events if event["event_type"] in ACTIONABLE_EVENT_TYPES]
    segments: list[ConsolidationSegment] = []
    for start in range(0, len(actionable), window_size):
        window = actionable[start : start + window_size]
        if not window:
            continue
        segments.append(
            ConsolidationSegment(
                session_id=sid,
                segment_id=build_segment_id(sid, [int(event["seq"]) for event in window]),
                event_type_counts=event_type_counts(window),
                source_events=window,
            )
        )
    return segments


def _source_event_from_event(event: Any) -> dict[str, Any]:
    seq = _validate_seq(getattr(event, "seq", None))
    event_hash = getattr(event, "hash", None)
    if not isinstance(event_hash, str) or len(event_hash) != _EVENT_HASH_LEN:
        raise ValueError("event hash must be 64 lowercase hex")
    if any(char not in "0123456789abcdef" for char in event_hash):
        raise ValueError("event hash must be 64 lowercase hex")
    event_type = getattr(event, "event_type", None)
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("event_type must be a non-empty string")
    payload = getattr(event, "payload", {})
    return {
        "seq": seq,
        "hash": event_hash,
        "event_type": event_type,
        "summary": _event_summary(event_type, payload),
    }


def _event_summary(event_type: str, payload: object) -> str:
    if not isinstance(payload, Mapping):
        return event_type
    parts = [event_type]
    for key in ("status", "tool_name", "path", "summary", "query", "title"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " | ".join(parts[:4])
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k select --no-cov -q
```

Expected: selection tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/zaxy/consolidation_pipeline.py tests/test_consolidation_pipeline.py
git commit -m "feat: select consolidation event segments"
```

---

### Task 3: Generate Episode, Claim, and Procedure Candidates

**Files:**
- Modify: `src/zaxy/consolidation_pipeline.py`
- Test: `tests/test_consolidation_pipeline.py`

- [ ] **Step 1: Add failing candidate-generation tests**

Append:

```python
def test_generate_consolidation_proposals_creates_episode_claim_and_procedure() -> None:
    from zaxy.consolidation_pipeline import generate_consolidation_proposals

    segment = ConsolidationSegment(
        session_id="agent-1",
        segment_id="segment:agent-1:000001-000003",
        event_type_counts={"tool.call.completed": 2, "file.edit.applied": 1},
        source_events=[
            {"seq": 1, "hash": "a" * 64, "event_type": "tool.call.completed", "summary": "pytest failed"},
            {"seq": 2, "hash": "b" * 64, "event_type": "file.edit.applied", "summary": "patched checkout"},
            {"seq": 3, "hash": "c" * 64, "event_type": "tool.call.completed", "summary": "pytest succeeded"},
        ],
    )

    proposals = generate_consolidation_proposals([segment], purpose="coding")

    assert [proposal.candidate_type for proposal in proposals] == ["episode", "claim", "procedure"]
    assert all(proposal.segment == segment for proposal in proposals)
    assert all(proposal.purpose == "coding" for proposal in proposals)
    assert all(proposal.method.startswith("deterministic_") for proposal in proposals)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k generate --no-cov -q
```

Expected: fail because `generate_consolidation_proposals` is missing.

- [ ] **Step 3: Implement deterministic candidate generator**

Add:

```python
def generate_consolidation_proposals(
    segments: Sequence[ConsolidationSegment],
    *,
    purpose: str | None = None,
) -> list[ProposedConsolidation]:
    proposals: list[ProposedConsolidation] = []
    for segment in segments:
        proposals.append(_episode_proposal(segment, purpose=purpose))
        if _has_claim_signal(segment):
            proposals.append(_claim_proposal(segment, purpose=purpose))
        if _has_procedure_signal(segment):
            proposals.append(_procedure_proposal(segment, purpose=purpose))
    return proposals


def _episode_proposal(segment: ConsolidationSegment, *, purpose: str | None) -> ProposedConsolidation:
    return ProposedConsolidation(
        segment=segment,
        candidate_type="episode",
        title=f"Episode {segment.segment_id.rsplit(':', 1)[-1]}",
        summary=_segment_summary(segment),
        confidence=0.68,
        method="deterministic_episode_segment_v1",
        purpose=purpose,
    )


def _claim_proposal(segment: ConsolidationSegment, *, purpose: str | None) -> ProposedConsolidation:
    return ProposedConsolidation(
        segment=segment,
        candidate_type="claim",
        title=f"Claim from {segment.segment_id}",
        summary=f"Candidate claim supported by {segment.source_event_count} cited source events: {_segment_summary(segment)}",
        confidence=0.62,
        method="deterministic_claim_signal_v1",
        purpose=purpose,
    )


def _procedure_proposal(segment: ConsolidationSegment, *, purpose: str | None) -> ProposedConsolidation:
    return ProposedConsolidation(
        segment=segment,
        candidate_type="procedure",
        title=f"Procedure from {segment.segment_id}",
        summary=f"Candidate procedure inferred from observed workflow steps: {_segment_summary(segment)}",
        confidence=0.58,
        method="deterministic_procedure_trace_v1",
        purpose=purpose,
    )


def _segment_summary(segment: ConsolidationSegment) -> str:
    summaries = [str(event.get("summary", "")).strip() for event in segment.source_events]
    compact = [summary for summary in summaries if summary]
    return " -> ".join(compact[:4]) or segment.segment_id


def _has_claim_signal(segment: ConsolidationSegment) -> bool:
    return segment.source_event_count >= 2


def _has_procedure_signal(segment: ConsolidationSegment) -> bool:
    return segment.event_type_counts.get("tool.call.completed", 0) >= 2 or (
        segment.event_type_counts.get("file.edit.applied", 0) >= 1
        and segment.event_type_counts.get("tool.call.completed", 0) >= 1
    )
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k generate --no-cov -q
```

Expected: generator tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/zaxy/consolidation_pipeline.py tests/test_consolidation_pipeline.py
git commit -m "feat: generate reviewable consolidation proposals"
```

---

### Task 4: Add MemoryFabric Consolidation Services

**Files:**
- Modify: `src/zaxy/core.py`
- Test: `tests/test_consolidation_pipeline.py`

- [ ] **Step 1: Add failing fabric service tests**

Append:

```python
import pytest


@pytest.mark.asyncio
async def test_memory_fabric_proposes_consolidation_candidates_from_session_events(tmp_path) -> None:
    from zaxy.core import MemoryFabric

    fabric = MemoryFabric(eventloom_path=str(tmp_path / ".eventloom"), projection_backend="embedded")
    await fabric.connect()
    try:
        await fabric.append("tool.call.completed", actor="agent", payload={"tool_name": "pytest", "status": "failed"}, session_id="agent-1")
        await fabric.append("file.edit.applied", actor="agent", payload={"path": "src/zaxy/checkout.py"}, session_id="agent-1")
        await fabric.append("tool.call.completed", actor="agent", payload={"tool_name": "pytest", "status": "succeeded"}, session_id="agent-1")

        result = await fabric.propose_consolidation_candidates(
            session_id="agent-1",
            actor="zaxy-consolidation",
            purpose="coding",
            window_size=3,
        )
    finally:
        await fabric.close()

    assert result["session_id"] == "agent-1"
    assert result["segment_count"] == 1
    assert result["candidate_count"] >= 1
    assert all(item["event_type"] == "consolidation.candidate.created" for item in result["events"])
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k memory_fabric --no-cov -q
```

Expected: fail because `MemoryFabric.propose_consolidation_candidates` is missing.

- [ ] **Step 3: Implement fabric services**

Add to `src/zaxy/core.py`:

```python
    async def propose_consolidation_candidates(
        self,
        *,
        session_id: str = "default",
        actor: str = "zaxy-consolidation",
        purpose: str | None = None,
        window_size: int = 8,
    ) -> dict[str, Any]:
        from zaxy.consolidation_pipeline import (
            generate_consolidation_proposals,
            select_consolidation_segments,
        )

        sid = validate_session_id(session_id)
        events = self.session_manager.get(sid).eventlog.read_all()
        segments = select_consolidation_segments(events, session_id=sid, window_size=window_size)
        proposals = generate_consolidation_proposals(segments, purpose=purpose)
        appended: list[dict[str, Any]] = []
        for proposal in proposals:
            event_spec = proposal.to_candidate_event(actor=actor)
            event = await self.append(
                event_spec["event_type"],
                actor=event_spec["actor"],
                payload=event_spec["payload"],
                session_id=sid,
            )
            appended.append({"event_type": event.event_type, "seq": event.seq, "hash": event.hash})
        return {
            "session_id": sid,
            "segment_count": len(segments),
            "candidate_count": len(appended),
            "events": appended,
        }
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k memory_fabric --no-cov -q
```

Expected: fabric service test passes.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/zaxy/core.py tests/test_consolidation_pipeline.py
git commit -m "feat: propose consolidation candidates from memory fabric"
```

---

### Task 5: Add Stale, Conflict, and Supersession Diagnostics

**Files:**
- Modify: `src/zaxy/consolidation_pipeline.py`
- Modify: `src/zaxy/checkout.py`
- Test: `tests/test_consolidation_pipeline.py`
- Test: `tests/test_causal_checkout.py`

- [ ] **Step 1: Add failing diagnostic tests**

Add:

```python
def test_candidate_diagnostics_counts_stale_conflicted_and_superseded() -> None:
    from zaxy.consolidation_pipeline import consolidation_candidate_diagnostics

    rows = [
        {"entity_type": "consolidation_candidate", "review_status": "pending", "authority_status": "non_authoritative"},
        {"entity_type": "consolidation_candidate", "review_status": "conflicted", "authority_status": "non_authoritative"},
        {"entity_type": "consolidation_candidate", "review_status": "accepted", "authority_status": "non_authoritative", "valid_to": "2026-06-07T00:00:00Z"},
        {"entity_type": "consolidation_candidate", "review_status": "rejected", "authority_status": "non_authoritative"},
    ]

    assert consolidation_candidate_diagnostics(rows) == {
        "candidate_count": 4,
        "pending_count": 1,
        "accepted_count": 1,
        "rejected_count": 1,
        "conflicted_count": 1,
        "stale_count": 1,
        "authority_status": "non_authoritative",
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_consolidation_pipeline.py -k diagnostics --no-cov -q
```

Expected: fail because diagnostics helper is missing.

- [ ] **Step 3: Implement diagnostics and wire checkout**

Add to `src/zaxy/consolidation_pipeline.py`:

```python
def consolidation_candidate_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("entity_type") == "consolidation_candidate"]
    statuses = [str(row.get("review_status") or row.get("status") or "") for row in candidates]
    stale_count = sum(1 for row in candidates if row.get("valid_to") or row.get("stale") is True)
    return {
        "candidate_count": len(candidates),
        "pending_count": statuses.count("pending"),
        "accepted_count": statuses.count("accepted"),
        "rejected_count": statuses.count("rejected"),
        "conflicted_count": statuses.count("conflicted"),
        "stale_count": stale_count,
        "authority_status": "non_authoritative",
    }
```

In `src/zaxy/checkout.py`, preserve existing fields and add `rejected_count`, `conflicted_count`, and `stale_count` to consolidation diagnostics. Keep the guidance from alpha.1:

```python
"Do not treat consolidation candidates as authoritative memory without a separate promotion event."
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation_pipeline.py tests/test_causal_checkout.py -k "diagnostics or consolidation" --no-cov -q
```

Expected: diagnostics and checkout tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/zaxy/consolidation_pipeline.py src/zaxy/checkout.py tests/test_consolidation_pipeline.py tests/test_causal_checkout.py
git commit -m "feat: add consolidation candidate diagnostics"
```

---

### Task 6: Add CLI and MCP Proposal Surfaces

**Files:**
- Modify: `src/zaxy/__main__.py`
- Modify: `src/zaxy/mcp_server.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Add failing CLI/MCP tests**

Add CLI tests:

```python
def test_memory_consolidation_propose_from_log_help_is_registered() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["memory", "consolidation", "propose-from-log", "--help"])

    assert result.exit_code == 0
    assert "--window-size" in result.output
    assert "--purpose" in result.output
```

Add MCP schema test:

```python
def test_memory_consolidation_propose_from_log_tool_is_registered() -> None:
    tool_names = {tool.name for tool in TOOLS}

    assert "memory_consolidation_propose_from_log" in tool_names
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_cli.py -k propose_from_log --no-cov -q
pytest tests/test_mcp.py -k propose_from_log --no-cov -q
```

Expected: fail because commands/tools are missing.

- [ ] **Step 3: Implement CLI**

Add under `memory_consolidation_app`:

```python
@memory_consolidation_app.command("propose-from-log")
def memory_consolidation_propose_from_log(
    session_id: str = typer.Option("default", help="Session ID to consolidate"),
    actor: str = typer.Option("zaxy-consolidation", help="Actor writing candidate events"),
    purpose: str | None = typer.Option(None, help="Optional consolidation purpose"),
    window_size: int = typer.Option(8, min=1, max=50, help="Actionable event window size"),
    eventloom_path: Path = typer.Option(".eventloom", help="Eventloom directory"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
) -> None:
    import asyncio

    async def _run() -> dict[str, object]:
        fabric = MemoryFabric(eventloom_path=str(eventloom_path))
        try:
            await fabric.connect()
            return await fabric.propose_consolidation_candidates(
                session_id=session_id,
                actor=actor,
                purpose=purpose,
                window_size=window_size,
            )
        finally:
            with suppress(Exception):
                await fabric.close()

    result = asyncio.run(_run())
    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(
            f"Created {result['candidate_count']} consolidation candidates "
            f"from {result['segment_count']} segments for {result['session_id']}."
        )
```

- [ ] **Step 4: Implement MCP**

Add tool schema:

```python
Tool(
    name="memory_consolidation_propose_from_log",
    description="Generate cited, review-pending consolidation candidates from one session log.",
    inputSchema={
        "type": "object",
        "required": [],
        "properties": {
            "session_id": {"type": "string"},
            "actor": {"type": "string", "default": "zaxy-consolidation"},
            "purpose": {"type": "string"},
            "window_size": {"type": "integer", "default": 8, "minimum": 1, "maximum": 50},
        },
        "additionalProperties": False,
    },
)
```

Add `ZaxyMCPServer.handle_memory_consolidation_propose_from_log` that calls a configured `MemoryFabric` or shared internal service without creating a second Eventloom path. Register it in `_dispatch_tool_call`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_cli.py tests/test_mcp.py -k "consolidation and propose" --no-cov -q
```

Expected: CLI and MCP tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/zaxy/__main__.py src/zaxy/mcp_server.py tests/test_cli.py tests/test_mcp.py docs/examples/mcp-tool-contract.json
git commit -m "feat: expose consolidation proposal pipeline"
```

---

### Task 7: Add Alpha.2 Consolidation Guardrail

**Files:**
- Create: `src/zaxy/consolidation_benchmark.py`
- Test: `tests/test_consolidation_benchmark.py`

- [ ] **Step 1: Add failing guardrail tests**

Create `tests/test_consolidation_benchmark.py`:

```python
from __future__ import annotations

from zaxy.consolidation_benchmark import (
    ConsolidationGuardrailCase,
    evaluate_consolidation_guardrail,
    summarize_consolidation_guardrail,
)


def test_guardrail_scores_cited_non_authoritative_candidate() -> None:
    case = ConsolidationGuardrailCase(
        case_id="claim-1",
        candidate_type="claim",
        required_source_events=[{"seq": 1, "hash": "a" * 64}, {"seq": 2, "hash": "b" * 64}],
    )
    row = evaluate_consolidation_guardrail(
        case,
        {
            "candidate_type": "claim",
            "source_events": [{"seq": 1, "hash": "a" * 64}, {"seq": 2, "hash": "b" * 64}],
            "review_status": "pending",
            "authority_status": "non_authoritative",
            "confidence": 0.7,
        },
    )

    assert row == {
        "case_id": "claim-1",
        "type_match": 1.0,
        "source_event_fidelity": 1.0,
        "authority_boundary": 1.0,
        "review_gate": 1.0,
        "score": 1.0,
    }


def test_guardrail_penalizes_authority_promotion() -> None:
    case = ConsolidationGuardrailCase(
        case_id="claim-2",
        candidate_type="claim",
        required_source_events=[{"seq": 1, "hash": "a" * 64}],
    )
    row = evaluate_consolidation_guardrail(
        case,
        {
            "candidate_type": "claim",
            "source_events": [{"seq": 1, "hash": "a" * 64}],
            "review_status": "accepted",
            "authority_status": "authoritative",
            "confidence": 0.7,
        },
    )

    assert row["authority_boundary"] == 0.0
    assert row["review_gate"] == 1.0
    assert row["score"] == 0.75
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_consolidation_benchmark.py --no-cov -q
```

Expected: fail because module is missing.

- [ ] **Step 3: Implement guardrail scorer**

Create `src/zaxy/consolidation_benchmark.py` with deterministic scoring over type match, source event fidelity, authority boundary, and review gate. Validate source event hashes as 64 lowercase hex and review statuses as `pending`, `accepted`, `rejected`, `deferred`, or `conflicted`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_consolidation_benchmark.py --no-cov -q
```

Expected: guardrail tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/zaxy/consolidation_benchmark.py tests/test_consolidation_benchmark.py
git commit -m "feat: add consolidation alpha guardrail"
```

---

### Task 8: Documentation and Regression Gate

**Files:**
- Modify: `docs/graph-schema.md`
- Modify: `docs/mcp.md`
- Modify: `docs/benchmarks.md`
- Modify generated `site/docs/*.html` if `scripts/validate-docs.sh` requires it.

- [ ] **Step 1: Update docs**

Document:

- alpha.2 segment selection is deterministic and event-sourced;
- generated episode, claim, and procedure candidates remain non-authoritative;
- accepted reviews are dispositions, not promotion;
- `memory_consolidation_propose_from_log` is a proposal tool, not an authority tool;
- consolidation guardrail is internal/project-defined and not external validation.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest \
  tests/test_consolidation.py \
  tests/test_consolidation_pipeline.py \
  tests/test_consolidation_benchmark.py \
  tests/test_causal_checkout.py \
  tests/test_cli.py \
  tests/test_mcp.py \
  -k "consolidation" \
  --no-cov -q
```

Expected: all selected tests pass.

- [ ] **Step 3: Run core regression tests**

Run:

```bash
pytest tests/test_checkout.py tests/test_graph.py tests/test_mcp.py --no-cov -q
```

Expected: pass.

- [ ] **Step 4: Run docs validation and lint**

Run:

```bash
scripts/validate-docs.sh --root .
ruff check src/zaxy/consolidation.py src/zaxy/consolidation_pipeline.py src/zaxy/consolidation_benchmark.py src/zaxy/core.py src/zaxy/__main__.py src/zaxy/mcp_server.py tests/test_consolidation.py tests/test_consolidation_pipeline.py tests/test_consolidation_benchmark.py tests/test_cli.py tests/test_mcp.py
```

Expected: pass.

- [ ] **Step 5: Run cached public benchmark guardrail**

Run:

```bash
python -m zaxy benchmark-compare \
  reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.json \
  --backend zaxy-checkout \
  --min-mean-score 0.95 \
  --min-answer-recall-at-5 0.90 \
  --min-recall-at-5 0.99 \
  --min-citation-coverage 1.0 \
  --max-p95-ms 2500 \
  --max-p99-ms 3000
```

Expected: PASS against the cached published artifact. Do not change implementation to satisfy missing artifacts.

- [ ] **Step 6: Commit**

Run:

```bash
git add docs/graph-schema.md docs/mcp.md docs/benchmarks.md site/docs src/zaxy tests
git commit -m "docs: document consolidation alpha mvp"
```

---

## Self-Review Notes

Spec coverage:

- Cluster related event segments into reviewable episodes: Tasks 1-3.
- Synthesize cited candidate claims and procedures: Task 3.
- Preserve source event backpointers: Tasks 1, 3, 4, and 7.
- Track review status, authority status, confidence, scope, and purpose: Tasks 1, 3, 4, and 5.
- Add checkout and status diagnostics: Task 5.
- Add CLI/MCP surfaces: Task 6.
- Add rejection, stale, supersession, and conflict behavior: Task 5.
- Protect 1.x benchmark claims: Task 8.

No benchmark tailoring:

- The pipeline operates on Eventloom event classes and source citations, not LongMemEval questions.
- The guardrail measures consolidation fidelity and authority boundaries only.
- LongMemEval-compatible checks remain regression gates, not implementation targets.

Known risks:

- The first generator is deterministic and conservative; it will likely produce useful review candidates but not polished narrative summaries.
- Backend parity for projected candidate status should be watched after each graph adapter change.
- LLM-assisted synthesis can be planned later only after deterministic source-event fidelity is stable.
