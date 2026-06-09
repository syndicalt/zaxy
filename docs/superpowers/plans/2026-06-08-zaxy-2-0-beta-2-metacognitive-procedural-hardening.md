# Zaxy 2.0 Beta.2 Metacognitive and Procedural Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-ready metacognitive state tracking and first-class procedural planning diagnostics without turning uncertainty, confidence, or procedures into authoritative facts.

**Architecture:** Eventloom remains the source of truth. Beta.2 adds strict metacognition event contracts, replay-derived query surfaces, checkout diagnostics, and procedural planning buckets that reuse existing Skill Memory and consolidation projections. All generated metacognitive/procedural outputs are non-authoritative, cited, observable, and excluded from claim-support scoring unless a separate authority path has accepted them.

**Tech Stack:** Python 3.11+, Eventloom JSONL, existing `MemoryFabric`, Memory Checkout, Skill Memory, consolidation candidates, Typer CLI, MCP Python SDK, pytest, ruff.

---

## Scope Boundary

Included:

- Track known unknowns through `metacognition.unknown.recorded`.
- Track conflicting evidence clusters through `metacognition.conflict.clustered`.
- Track append-only confidence trajectories through `metacognition.confidence.assessed`.
- Track re-verification needs through `metacognition.reverify.requested`.
- Add `MemoryFabric` query surfaces for known unknowns, conflict clusters, confidence trajectories, and re-verification needs.
- Add optional confidence-assessment recording to `get_claim_confidence` without self-reinforcing prior metacognition or primitive trace events.
- Promote procedural memory into explicit applicable/diagnostic/excluded planning buckets.
- Preserve rollback and contradiction diagnostics for procedural memory.
- Extend `memory_skill` to persist `failure_modes`, `rollback`, and `contradiction_reason`.
- Add CLI/MCP surfaces, docs, generated site pages, and internal beta.2 guardrail checks.

Excluded:

- Autonomous belief revision.
- Silent authority promotion.
- Learned confidence models.
- Benchmark-specific retrieval or scoring logic.
- Cross-project global procedure sharing.

## File Structure

Create:

- `src/zaxy/metacognition.py`  
  Strict contracts, builders, deterministic IDs, replay summaries, and eligibility helpers for metacognitive state.

- `tests/test_metacognition.py`  
  Unit tests for builders, validation, replay summaries, and authority boundaries.

Modify:

- `src/zaxy/core.py`  
  Add `MemoryFabric` metacognition methods, confidence trajectory recording, reverify queries, and procedural planning buckets.

- `src/zaxy/extract.py`  
  Add typed extraction for metacognition events and preserve non-authoritative status.

- `src/zaxy/checkout.py`  
  Add metacognition and procedural planning diagnostics/guidance.

- `src/zaxy/reasoning_benchmark.py`  
  Add beta.2 guardrail scoring for metacognition and procedural planning contracts.

- `src/zaxy/__main__.py`  
  Add `zaxy memory reasoning` commands for uncertainty/reverification/procedural planning.

- `src/zaxy/mcp_server.py` and `docs/examples/mcp-tool-contract.json`  
  Add MCP tools and contract snapshot updates.

- `docs/mcp.md`, `docs/graph-schema.md`, `docs/benchmarks.md`, `docs/agent-events.md`, generated `site/docs/*.html`  
  Document beta.2 boundaries and surfaces.

---

### Task 1: Add Metacognition Contracts

**Files:**
- Create: `src/zaxy/metacognition.py`
- Test: `tests/test_metacognition.py`

- [ ] **Step 1: Write failing contract tests**

Add tests proving:

```python
from zaxy.metacognition import (
    build_confidence_assessment_event,
    build_conflict_cluster_event,
    build_known_unknown_event,
    build_reverify_request_event,
    summarize_metacognition_events,
)


SOURCE = [{"seq": 7, "hash": "a" * 64}]


def test_known_unknown_event_is_open_cited_and_non_authoritative() -> None:
    event = build_known_unknown_event(
        actor="agent",
        session_id="agent-1",
        question="Which projection backend caused the latency spike?",
        reason="Checkout had conflicting backend evidence.",
        source_events=SOURCE,
        claim_key="projection-latency-cause",
        gap_type="conflicting_evidence",
        reverify_query="latest cited projection latency cause",
    )

    assert event["event_type"] == "metacognition.unknown.recorded"
    assert event["thread"] == "agent-1"
    assert event["payload"]["status"] == "open"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["source_events"] == SOURCE


def test_confidence_assessment_event_tracks_append_only_point() -> None:
    event = build_confidence_assessment_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        claim="Projection stale caused failure",
        confidence=0.42,
        support_count=1,
        conflict_count=2,
        evidence=[
            {"citation": "eventloom://agent-1/events/7#aaaaaaaaaaaa", "stance": "support"},
            {"citation": "eventloom://agent-1/events/8#bbbbbbbbbbbb", "stance": "conflict"},
        ],
        method="deterministic_token_overlap_v1",
        requires_reverify=True,
    )

    assert event["event_type"] == "metacognition.confidence.assessed"
    assert event["payload"]["confidence"] == 0.42
    assert event["payload"]["requires_reverify"] is True
    assert event["payload"]["authority_status"] == "non_authoritative"


def test_conflict_cluster_event_preserves_support_and_conflict_sources() -> None:
    event = build_conflict_cluster_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        claim_key="projection-latency-cause",
        claim="Projection stale caused failure",
        supporting_source_events=[{"seq": 7, "hash": "a" * 64}],
        conflicting_source_events=[{"seq": 8, "hash": "b" * 64}],
        confidence=0.5,
        reason="Support and conflict evidence both present.",
    )

    assert event["event_type"] == "metacognition.conflict.clustered"
    assert event["payload"]["resolution_status"] == "unresolved"
    assert event["payload"]["authority_status"] == "non_authoritative"


def test_reverify_request_event_is_open_and_cited() -> None:
    event = build_reverify_request_event(
        actor="zaxy-reasoning",
        session_id="agent-1",
        query="Re-check cited projection latency cause",
        reason="Low confidence and conflict count above zero.",
        source_events=SOURCE,
        priority="high",
        claim_key="projection-latency-cause",
    )

    assert event["event_type"] == "metacognition.reverify.requested"
    assert event["payload"]["status"] == "open"
    assert event["payload"]["priority"] == "high"


def test_summarize_metacognition_events_returns_open_counts() -> None:
    events = [
        build_known_unknown_event(
            actor="agent",
            session_id="agent-1",
            question="What changed?",
            reason="No cited answer.",
            source_events=SOURCE,
            claim_key="change",
            gap_type="missing_evidence",
            reverify_query="what changed",
        ),
        build_reverify_request_event(
            actor="agent",
            session_id="agent-1",
            query="what changed",
            reason="missing evidence",
            source_events=SOURCE,
            priority="normal",
            claim_key="change",
        ),
    ]

    summary = summarize_metacognition_events(events)

    assert summary["unknown_count"] == 1
    assert summary["open_unknown_count"] == 1
    assert summary["reverify_needed_count"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_metacognition.py --no-cov -q
```

Expected: fail because `zaxy.metacognition` does not exist.

- [ ] **Step 3: Implement contracts**

Create `src/zaxy/metacognition.py` with:

- `build_known_unknown_event(...)`
- `build_confidence_assessment_event(...)`
- `build_conflict_cluster_event(...)`
- `build_reverify_request_event(...)`
- `summarize_metacognition_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]`
- strict validation for non-empty text, 0..1 confidence, positive source event seqs, 64-lowercase-hex source hashes, Eventloom citations with exact 12 or 64 hash characters, and allowed priorities `low|normal|high|urgent`.

Every builder must set `authority_status="non_authoritative"`. Known unknowns and reverify requests must set `status="open"`. Conflict clusters must set `resolution_status="unresolved"`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_metacognition.py --no-cov -q
```

Expected: all metacognition contract tests pass.

---

### Task 2: Add MemoryFabric Metacognition Services

**Files:**
- Modify: `src/zaxy/core.py`
- Modify: `tests/test_metacognition.py`
- Modify: `tests/test_reasoning_primitives.py`

- [ ] **Step 1: Add failing MemoryFabric tests**

Add async tests proving:

- `record_known_unknown(...)` appends `metacognition.unknown.recorded` and records `reasoning.primitive.called`.
- `get_claim_confidence(..., record_assessment=True)` appends `metacognition.confidence.assessed` after scoring cited support/conflict evidence.
- `list_confidence_trajectory(claim=...)` replays confidence assessments for a claim without requiring Neo4j.
- `list_reverification_needs(...)` returns open unknowns, low-confidence assessments, conflict clusters, and reverify requests.
- Confidence scoring excludes metacognition events, reasoning primitive observations, and belief proposals from support/conflict evidence.

Use embedded/Eventloom-only tests. Monkeypatch projection where necessary.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_metacognition.py tests/test_reasoning_primitives.py -k "metacognition or confidence" --no-cov -q
```

Expected: fail because `MemoryFabric` methods and assessment recording are missing.

- [ ] **Step 3: Implement MemoryFabric services**

Add methods:

```python
async def record_known_unknown(
    self,
    question: str,
    *,
    reason: str,
    source_events: list[dict[str, Any]],
    claim_key: str,
    gap_type: str = "missing_evidence",
    reverify_query: str | None = None,
    phase: str = "review",
    session_id: str = "default",
    actor: str = "zaxy-reasoning",
) -> dict[str, Any]: ...

async def list_known_unknowns(
    self,
    *,
    session_id: str = "default",
    status: str = "open",
    limit: int = 10,
) -> dict[str, Any]: ...

async def list_conflict_clusters(
    self,
    *,
    session_id: str = "default",
    unresolved_only: bool = True,
    limit: int = 10,
) -> dict[str, Any]: ...

async def list_confidence_trajectory(
    self,
    claim: str,
    *,
    session_id: str = "default",
    limit: int = 10,
) -> dict[str, Any]: ...

async def list_reverification_needs(
    self,
    query: str | None = None,
    *,
    session_id: str = "default",
    limit: int = 10,
    min_confidence: float = 0.7,
) -> dict[str, Any]: ...
```

Update `get_claim_confidence(...)` to accept `record_assessment: bool = True` and append `metacognition.confidence.assessed` when true. If support and conflict are both present, append `metacognition.conflict.clustered`. If confidence is below `min_confidence` or conflict count is non-zero, append `metacognition.reverify.requested`.

Implementation requirements:

- Replay Eventloom logs for list methods; do not require graph availability.
- Do not close or resolve unknowns automatically.
- Do not count metacognition events, `reasoning.primitive.called`, or `belief.update.proposed` as support evidence.
- Every method that is model-callable records `reasoning.primitive.called`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_metacognition.py tests/test_reasoning_primitives.py -k "metacognition or confidence" --no-cov -q
```

Expected: all Task 2 tests pass.

---

### Task 3: Add Typed Metacognition Projection and Checkout Diagnostics

**Files:**
- Modify: `src/zaxy/extract.py`
- Modify: `src/zaxy/checkout.py`
- Modify: `tests/test_extract.py`
- Modify: `tests/test_causal_checkout.py`
- Modify: `tests/test_checkout.py`

- [ ] **Step 1: Add failing extractor and checkout tests**

Tests must prove:

- `metacognition.unknown.recorded`, `metacognition.confidence.assessed`, `metacognition.conflict.clustered`, and `metacognition.reverify.requested` project to typed entities.
- Projected entities preserve `authority_status="non_authoritative"`, status/resolution fields, confidence, source event refs, and claim keys.
- Extractors reject `authority_status="authoritative"`.
- Checkout diagnostics include `metacognition.unknown_count`, `open_unknown_count`, `conflict_cluster_count`, `low_confidence_count`, and `reverify_needed_count`.
- Prompt guidance tells the model unresolved/high-uncertainty metacognition is diagnostic, not fact authority.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_extract.py tests/test_causal_checkout.py tests/test_checkout.py -k "metacognition or unknown or reverify" --no-cov -q
```

Expected: fail because typed extraction and diagnostics are missing.

- [ ] **Step 3: Implement typed extraction**

Add extractors in `src/zaxy/extract.py`:

- `@register("metacognition.unknown.recorded")`
- `@register("metacognition.confidence.assessed")`
- `@register("metacognition.conflict.clustered")`
- `@register("metacognition.reverify.requested")`

Use entity types:

- `known_unknown`
- `confidence_assessment`
- `conflict_cluster`
- `reverify_request`

Use deterministic entity names from event payload IDs. Use the existing source-event snapshot helpers for backend-safe `source_event_refs`, `source_event_seqs`, and `source_event_hashes`.

- [ ] **Step 4: Implement checkout diagnostics/guidance**

Add deterministic diagnostics in `src/zaxy/checkout.py`:

```python
diagnostics["metacognition"] = {
    "unknown_count": ...,
    "open_unknown_count": ...,
    "conflict_cluster_count": ...,
    "unresolved_conflict_count": ...,
    "low_confidence_count": ...,
    "reverify_needed_count": ...,
    "authority_status": "non_authoritative",
}
```

Guidance must include:

- unresolved known unknowns require re-verification or user clarification;
- confidence assessments are trajectory evidence, not truth;
- conflict clusters are diagnostic until resolved by a separate authority path.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_extract.py tests/test_causal_checkout.py tests/test_checkout.py -k "metacognition or unknown or reverify" --no-cov -q
```

Expected: pass.

---

### Task 4: Harden Procedural Planning Lane

**Files:**
- Modify: `src/zaxy/core.py`
- Modify: `src/zaxy/checkout.py`
- Modify: `src/zaxy/mcp_server.py`
- Modify: `tests/test_reasoning_primitives.py`
- Modify: `tests/test_checkout.py`
- Modify: `tests/test_mcp.py`

- [ ] **Step 1: Add failing procedural tests**

Tests must prove:

- `retrieve_similar_procedures(...)` returns separate `applicable`, `diagnostic`, and `excluded` buckets.
- Only validated/revised/accepted cited procedures are applicable.
- Pending, deferred, deprecated, contradicted, stale, superseded, rejected, conflicted, uncited, and failed-outcome procedures are excluded or diagnostic, not operational instructions.
- Checkout diagnostics include `procedural_memory.applicable_count`, `diagnostic_count`, `excluded_count`, `rollback_candidate_count`, `contradiction_count`, and `excluded_reasons`.
- `memory_skill` accepts and persists `failure_modes`, `rollback`, and `contradiction_reason`.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_reasoning_primitives.py tests/test_checkout.py tests/test_mcp.py -k "procedure or skill" --no-cov -q
```

Expected: fail because the procedural planning lane is still flat and `memory_skill` does not pass all rollback fields.

- [ ] **Step 3: Implement procedural buckets**

Update helper logic near `_procedure_contexts`:

- return `{"applicable": [...], "diagnostic": [...], "excluded": [...], "excluded_reasons": {...}}`;
- require citation for applicable procedures;
- applicable statuses: `validated`, `revised`, `accepted`;
- diagnostic statuses: `proposed`, `pending`, `deferred`;
- excluded statuses: `rejected`, `conflicted`, `deprecated`, `contradicted`, `stale`;
- exclude `valid_to` closed and `superseded_by` contexts;
- preserve `rollback`, `failure_modes`, `contradiction_reason`, and latest citation.

Update `retrieve_similar_procedures` to keep `procedures` as a backward-compatible alias for `applicable`, and add `applicable`, `diagnostic`, `excluded`, and `procedural_memory`.

- [ ] **Step 4: Extend Skill Memory MCP helper**

In `src/zaxy/mcp_server.py`, extend `memory_skill` schema and handler to accept:

- `failure_modes: list[str]`
- `rollback: str`
- `contradiction_reason: str`

Pass these fields into the payload for `skill.contradicted`, `skill.deprecated`, and other actions when supplied. Preserve existing validation for list fields.

- [ ] **Step 5: Add checkout procedural diagnostics**

Add/extend checkout diagnostics:

```python
diagnostics["procedural_memory"] = {
    "applicable_count": ...,
    "diagnostic_count": ...,
    "excluded_count": ...,
    "rollback_candidate_count": ...,
    "contradiction_count": ...,
    "excluded_reasons": {...},
    "authority_status": "non_authoritative",
}
```

Prompt guidance must say applicable procedures are planning guidance, not authoritative facts, and rollback/contradiction candidates should be avoided or explicitly reviewed.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_reasoning_primitives.py tests/test_checkout.py tests/test_mcp.py -k "procedure or skill" --no-cov -q
```

Expected: pass.

---

### Task 5: Add CLI, MCP, Docs, and Beta.2 Guardrail

**Files:**
- Modify: `src/zaxy/__main__.py`
- Modify: `src/zaxy/mcp_server.py`
- Modify: `src/zaxy/reasoning_benchmark.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_mcp.py`
- Modify: `tests/test_reasoning_benchmark.py`
- Modify: `docs/examples/mcp-tool-contract.json`
- Modify: `docs/mcp.md`
- Modify: `docs/graph-schema.md`
- Modify: `docs/benchmarks.md`
- Modify: `docs/agent-events.md`
- Modify generated `site/docs/*.html`

- [ ] **Step 1: Add failing public-surface tests**

Add CLI tests for:

- `zaxy memory reasoning record-unknown`
- `zaxy memory reasoning known-unknowns`
- `zaxy memory reasoning confidence-trajectory`
- `zaxy memory reasoning reverify-needed`
- `zaxy memory reasoning plan-from-procedures`

Add MCP schema/handler/dispatch tests for:

- `memory_record_known_unknown`
- `memory_known_unknowns`
- `memory_confidence_trajectory`
- `memory_reverification_needs`
- `memory_plan_from_procedures`

Add guardrail tests proving beta.2 scoring includes:

- metacognition observability;
- reverify open status;
- procedural citation presence;
- planning phase match;
- authority boundary.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_cli.py tests/test_mcp.py tests/test_reasoning_benchmark.py -k "metacognition or unknown or reverify or trajectory or plan_from_procedures or beta2" --no-cov -q
```

Expected: fail because public surfaces and guardrail are missing.

- [ ] **Step 3: Implement CLI and MCP surfaces**

CLI command names:

- `record-unknown`
- `known-unknowns`
- `confidence-trajectory`
- `reverify-needed`
- `plan-from-procedures`

MCP tool names:

- `memory_record_known_unknown`
- `memory_known_unknowns`
- `memory_confidence_trajectory`
- `memory_reverification_needs`
- `memory_plan_from_procedures`

All CLI/MCP handlers must instantiate configured `MemoryFabric`, call the matching core method, close the fabric in `finally`, and return JSON-compatible dicts.

- [ ] **Step 4: Implement beta.2 guardrail**

Add `score_metacognition_guardrail(...)` or extend `score_reasoning_guardrail(...)` with explicit beta.2 fields:

- `observable_metacognition`
- `open_reverify_status`
- `procedural_citation_presence`
- `planning_phase_match`
- `authority_boundary`
- `score`

The scorer must inspect contract fields only. It must not score task answers or benchmark labels.

- [ ] **Step 5: Update docs and generated site**

Document:

- metacognitive events are diagnostic and non-authoritative;
- confidence trajectories are append-only and do not overwrite facts;
- reverify requests remain open until separately resolved;
- procedural planning lane separates applicable/diagnostic/excluded procedures;
- rollback/contradiction diagnostics are avoid/review signals;
- beta.2 guardrail is internal/project-defined, not external validation.

Run:

```bash
python scripts/build-site-docs.py
scripts/validate-docs.sh --root .
```

- [ ] **Step 6: Run public-surface tests**

Run:

```bash
pytest tests/test_cli.py tests/test_mcp.py tests/test_reasoning_benchmark.py -k "metacognition or unknown or reverify or trajectory or plan_from_procedures or beta2" --no-cov -q
```

Expected: pass.

---

## Final Regression Gate

After all tasks:

```bash
pytest \
  tests/test_metacognition.py \
  tests/test_reasoning_primitives.py \
  tests/test_reasoning_benchmark.py \
  tests/test_causal_checkout.py \
  tests/test_checkout.py \
  tests/test_extract.py \
  tests/test_cli.py \
  tests/test_mcp.py \
  -k "metacognition or unknown or reverify or trajectory or procedure or skill or beta2" \
  --no-cov -q

pytest tests/test_checkout.py tests/test_graph.py tests/test_mcp.py tests/test_extract.py --no-cov -q

ruff check \
  src/zaxy/metacognition.py \
  src/zaxy/reasoning_primitives.py \
  src/zaxy/reasoning_benchmark.py \
  src/zaxy/core.py \
  src/zaxy/__main__.py \
  src/zaxy/mcp_server.py \
  src/zaxy/checkout.py \
  src/zaxy/extract.py \
  tests/test_metacognition.py \
  tests/test_reasoning_primitives.py \
  tests/test_reasoning_benchmark.py \
  tests/test_causal_checkout.py \
  tests/test_checkout.py \
  tests/test_cli.py \
  tests/test_mcp.py \
  tests/test_extract.py

scripts/validate-docs.sh --root .

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

Expected:

- focused beta.2 tests pass;
- core checkout/graph/MCP/extract regressions pass;
- ruff clean;
- docs validation passes;
- cached LongMemBench guardrail passes without implementation code tailored to that artifact.

## Self-Review Notes

Spec coverage:

- Known unknowns: Tasks 1-3 and Task 5.
- Conflicting evidence clusters: Tasks 1-3 and Task 5.
- Confidence trajectories: Tasks 1-3 and Task 5.
- Re-verification surfaces: Tasks 1-3 and Task 5.
- Procedural memory first-class planning lane: Task 4 and Task 5.
- Rollback/contradiction diagnostics: Task 4 and docs in Task 5.
- No authority promotion: every task includes explicit non-authoritative checks.
- No benchmark tailoring: final benchmark comparison is a guardrail only.

Known risks:

- Initial conflict clustering is deterministic and conservative; it is a diagnostic primitive, not a belief-revision engine.
- Replay-derived metacognition lists may need future pagination for very long sessions.
- Procedural planning buckets depend on existing Skill Memory and consolidation projection quality.
