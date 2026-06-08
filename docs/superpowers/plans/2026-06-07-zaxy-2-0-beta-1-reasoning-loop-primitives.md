# Zaxy 2.0 Beta.1 Reasoning-Loop Memory Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-ready, observable reasoning-loop memory primitives for planning, execution, review, and reflection without granting generated updates authority.

**Architecture:** Eventloom remains the source of truth. Beta.1 adds a deterministic `reasoning_primitives` service layer that composes existing causal, checkout, consolidation, and Skill Memory capabilities into agent-callable primitives, records every primitive call as an observation event, and appends belief updates only as review-pending proposals. Public CLI and MCP surfaces delegate to `MemoryFabric` so all runtime paths share validation, purpose conditioning, citations, and authority boundaries.

**Tech Stack:** Python 3.11+, Eventloom JSONL, existing `MemoryFabric`, causal graph APIs, consolidation candidates, Skill Memory graph contexts, Typer CLI, MCP Python SDK, pytest, ruff.

---

## Scope Boundary

Included:

- `explain_outcome` primitive over causal predecessors plus cited checkout context.
- `propose_belief_update` primitive that appends a cited `belief.update.proposed` event only.
- `get_claim_confidence` primitive that scores support/conflict from cited checkout evidence.
- `retrieve_similar_procedures` primitive over Skill Memory and consolidation procedure candidates.
- Reasoning phase handling for `planning`, `execution`, `review`, and `reflection`.
- Observation events for every primitive call, including phase, status, result counts, and cited evidence counts.
- CLI and MCP surfaces for the new primitives.
- Docs and internal beta.1 guardrail tests.

Excluded:

- Autonomous belief revision.
- Authority promotion from belief proposals.
- Learned planning policies.
- New benchmark-specific retrieval logic.
- LongMemEval-specific answer synthesis changes.

## File Structure

Create:

- `src/zaxy/reasoning_primitives.py`  
  Pure contracts, validators, phase profiles, result payload helpers, claim confidence scoring, and procedure filtering.

- `tests/test_reasoning_primitives.py`  
  Unit and MemoryFabric tests for primitive contracts, observation events, non-authoritative belief proposals, phase profiles, and procedure retrieval.

- `src/zaxy/reasoning_benchmark.py`  
  Internal beta.1 guardrail scorer for observable primitive calls, authority boundary, citation presence, and phase routing.

- `tests/test_reasoning_benchmark.py`  
  Guardrail scorer tests.

Modify:

- `src/zaxy/core.py`  
  Add `MemoryFabric.explain_outcome`, `propose_belief_update`, `get_claim_confidence`, and `retrieve_similar_procedures`.

- `src/zaxy/extract.py`  
  Add deterministic extraction for `belief.update.proposed` and `reasoning.primitive.called` if projection support is needed by tests.

- `src/zaxy/__main__.py`  
  Add `zaxy memory reasoning ...` CLI commands.

- `src/zaxy/mcp_server.py` and `docs/examples/mcp-tool-contract.json`  
  Add MCP tools and snapshot coverage.

- `src/zaxy/checkout.py`  
  Add diagnostics/guidance for belief proposals and reasoning primitive observations when present in checkout context.

- `docs/mcp.md`, `docs/graph-schema.md`, `docs/benchmarks.md`, generated `site/docs/*.html`  
  Document beta.1 primitive boundaries and internal guardrail status.

---

### Task 1: Add Reasoning Primitive Contracts

**Files:**
- Create: `src/zaxy/reasoning_primitives.py`
- Test: `tests/test_reasoning_primitives.py`

- [ ] **Step 1: Write failing contract tests**

Add tests proving:

```python
from zaxy.reasoning_primitives import (
    REASONING_PHASES,
    ReasoningPrimitiveCall,
    build_belief_update_proposal_event,
    phase_purpose_profile,
)


def test_reasoning_phase_taxonomy_is_stable() -> None:
    assert REASONING_PHASES == {"planning", "execution", "review", "reflection"}


def test_phase_purpose_profiles_are_distinct() -> None:
    assert phase_purpose_profile("planning").task == "planning"
    assert phase_purpose_profile("execution").task == "execution"
    assert phase_purpose_profile("review").risk == "high"
    assert phase_purpose_profile("reflection").expected_action == "revise_or_record_learning"


def test_reasoning_call_event_is_observable_and_cited() -> None:
    call = ReasoningPrimitiveCall(
        primitive="explain_outcome",
        phase="planning",
        session_id="agent-1",
        query="Why did the test fail?",
        result_count=2,
        evidence=[{"citation": "eventloom://agent-1/events/42#aaaaaaaaaaaa", "content": "failure cause"}],
        status="succeeded",
    )

    event = call.to_event(actor="zaxy-reasoning")

    assert event["event_type"] == "reasoning.primitive.called"
    assert event["thread"] == "agent-1"
    assert event["payload"]["primitive"] == "explain_outcome"
    assert event["payload"]["phase"] == "planning"
    assert event["payload"]["evidence_count"] == 1


def test_belief_update_proposal_is_never_authoritative() -> None:
    event = build_belief_update_proposal_event(
        actor="agent",
        session_id="agent-1",
        claim="The failure was caused by a stale projection.",
        rationale="Cited causal predecessor indicates stale projection.",
        confidence=0.74,
        source_events=[{"seq": 42, "hash": "a" * 64}],
        phase="reflection",
    )

    assert event["event_type"] == "belief.update.proposed"
    assert event["thread"] == "agent-1"
    assert event["payload"]["authority_status"] == "non_authoritative"
    assert event["payload"]["review_status"] == "pending"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_reasoning_primitives.py --no-cov -q
```

Expected: fail because `zaxy.reasoning_primitives` does not exist.

- [ ] **Step 3: Implement contracts**

Create `src/zaxy/reasoning_primitives.py` with strict validation:

- `REASONING_PHASES = {"planning", "execution", "review", "reflection"}`
- `validate_reasoning_phase(value: object) -> str`
- `phase_purpose_profile(phase: str) -> PurposeProfile`
- `ReasoningPrimitiveCall.to_event(...)`
- `build_belief_update_proposal_event(...)`
- source events must use positive integer `seq` and 64-lowercase-hex `hash`
- belief proposal payload must include `authority_status="non_authoritative"` and `review_status="pending"`

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_reasoning_primitives.py --no-cov -q
```

Expected: all Task 1 tests pass.

---

### Task 2: Add MemoryFabric Reasoning Services

**Files:**
- Modify: `src/zaxy/core.py`
- Modify: `tests/test_reasoning_primitives.py`

- [ ] **Step 1: Add failing MemoryFabric tests**

Add async tests proving:

- `explain_outcome(outcome, phase="planning")` calls causal predecessors, returns cited explanation payload, and appends `reasoning.primitive.called`.
- `propose_belief_update(...)` appends `belief.update.proposed` and then records a primitive call.
- `get_claim_confidence(claim, phase="review")` returns confidence, support count, conflict count, evidence, and records a primitive call.
- `retrieve_similar_procedures(query, phase="planning")` returns Skill Memory or consolidation `procedure` candidates only and records a primitive call.

Use embedded/Eventloom-only tests and fake graph query results where needed; do not require Neo4j.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_reasoning_primitives.py -k "memory_fabric" --no-cov -q
```

Expected: fail because `MemoryFabric` methods do not exist.

- [ ] **Step 3: Implement MemoryFabric methods**

Add methods:

```python
async def explain_outcome(self, outcome: str, *, phase: str = "planning", session_id: str = "default", depth: int = 2) -> dict[str, Any]: ...
async def propose_belief_update(self, claim: str, *, rationale: str, confidence: float, source_events: list[dict[str, Any]], phase: str = "reflection", session_id: str = "default", actor: str = "zaxy-reasoning") -> dict[str, Any]: ...
async def get_claim_confidence(self, claim: str, *, phase: str = "review", session_id: str = "default", limit: int = 5) -> dict[str, Any]: ...
async def retrieve_similar_procedures(self, query: str, *, phase: str = "planning", session_id: str = "default", limit: int = 5) -> dict[str, Any]: ...
```

Implementation requirements:

- All methods validate phase and session.
- All methods append `reasoning.primitive.called` via a private helper.
- `explain_outcome` uses `query_causal_predecessors` and falls back to checkout evidence when no causal rows exist.
- `propose_belief_update` appends `belief.update.proposed`; it must never append an authoritative fact.
- `get_claim_confidence` uses cited checkout evidence and deterministic support/conflict token scoring.
- `retrieve_similar_procedures` searches current memory for `procedure`, Skill Memory, and consolidation procedure candidates; rejected/stale/conflicted candidates are excluded.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_reasoning_primitives.py --no-cov -q
```

Expected: all reasoning primitive tests pass.

---

### Task 3: Add CLI and MCP Reasoning Surfaces

**Files:**
- Modify: `src/zaxy/__main__.py`
- Modify: `src/zaxy/mcp_server.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_mcp.py`
- Modify: `docs/examples/mcp-tool-contract.json`

- [ ] **Step 1: Add failing CLI/MCP tests**

Add CLI help/delegation tests for:

- `zaxy memory reasoning explain-outcome`
- `zaxy memory reasoning propose-belief-update`
- `zaxy memory reasoning claim-confidence`
- `zaxy memory reasoning similar-procedures`

Add MCP schema/handler/dispatch tests for:

- `memory_explain_outcome`
- `memory_propose_belief_update`
- `memory_claim_confidence`
- `memory_similar_procedures`

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_cli.py tests/test_mcp.py -k "reasoning or explain_outcome or belief_update or claim_confidence or similar_procedures" --no-cov -q
```

Expected: fail because commands/tools are missing.

- [ ] **Step 3: Implement surfaces**

Add a nested `memory_reasoning_app = typer.Typer(...)` and wire commands through configured `MemoryFabric` helpers.

MCP handlers must instantiate the same configured `MemoryFabric` path/service pattern used by consolidation proposal/status, call the corresponding method, close the fabric safely, and return JSON.

Update `docs/examples/mcp-tool-contract.json` from `zaxy.mcp_server.TOOLS`.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/test_cli.py tests/test_mcp.py -k "reasoning or explain_outcome or belief_update or claim_confidence or similar_procedures" --no-cov -q
```

Expected: all beta.1 CLI/MCP tests pass.

---

### Task 4: Add Checkout Diagnostics, Docs, and Beta.1 Guardrail

**Files:**
- Create: `src/zaxy/reasoning_benchmark.py`
- Test: `tests/test_reasoning_benchmark.py`
- Modify: `src/zaxy/checkout.py`
- Modify: `tests/test_causal_checkout.py`
- Modify: `docs/mcp.md`, `docs/graph-schema.md`, `docs/benchmarks.md`
- Modify generated `site/docs/*.html` after docs build.

- [ ] **Step 1: Add failing diagnostics and guardrail tests**

Tests must prove:

- checkout diagnostics count `belief.update.proposed` and `reasoning.primitive.called` context without treating either as authoritative;
- prompt guidance says belief updates are proposals until reviewed;
- `evaluate_reasoning_guardrail(...)` scores observable call event, phase match, citation presence, and authority boundary.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/test_causal_checkout.py tests/test_reasoning_benchmark.py -k "reasoning or belief" --no-cov -q
```

Expected: fail because diagnostics/guardrail are missing.

- [ ] **Step 3: Implement diagnostics and guardrail**

Add deterministic diagnostics:

- `reasoning_primitives.context_count`
- `reasoning_primitives.phase_counts`
- `reasoning_primitives.primitive_counts`
- `belief_update_proposals.proposal_count`
- `belief_update_proposals.pending_count`
- `belief_update_proposals.authority_status == "non_authoritative"`

Guardrail rows should include:

- `observable_call`
- `phase_match`
- `citation_presence`
- `authority_boundary`
- `score`

- [ ] **Step 4: Update docs and generated site**

Document:

- beta.1 primitives are reasoning-loop tools, not authority tools;
- primitive calls are observable/replayable;
- belief updates remain `belief.update.proposed`;
- purpose phase routing is deterministic;
- beta.1 guardrail is internal/project-defined, not external validation.

Run:

```bash
python scripts/build-site-docs.py
scripts/validate-docs.sh --root .
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_causal_checkout.py tests/test_reasoning_benchmark.py -k "reasoning or belief" --no-cov -q
```

Expected: pass.

---

## Final Regression Gate

After all tasks:

```bash
pytest \
  tests/test_reasoning_primitives.py \
  tests/test_reasoning_benchmark.py \
  tests/test_causal_checkout.py \
  tests/test_cli.py \
  tests/test_mcp.py \
  -k "reasoning or belief or explain_outcome or claim_confidence or similar_procedures" \
  --no-cov -q

pytest tests/test_checkout.py tests/test_graph.py tests/test_mcp.py --no-cov -q

ruff check \
  src/zaxy/reasoning_primitives.py \
  src/zaxy/reasoning_benchmark.py \
  src/zaxy/core.py \
  src/zaxy/__main__.py \
  src/zaxy/mcp_server.py \
  src/zaxy/checkout.py \
  tests/test_reasoning_primitives.py \
  tests/test_reasoning_benchmark.py \
  tests/test_causal_checkout.py \
  tests/test_cli.py \
  tests/test_mcp.py

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

- focused beta.1 tests pass;
- core checkout/graph/MCP regressions pass;
- ruff clean;
- docs validation passes;
- cached LongMemBench guardrail passes without code tailored to that artifact.

## Self-Review Notes

Spec coverage:

- Causal predecessor/successor lookup: uses existing causal APIs and `explain_outcome`.
- Explain-outcome: Task 2.
- Propose-belief-update: Tasks 1-3, proposal-only.
- Get-claim-confidence: Task 2.
- Retrieve similar procedures: Task 2.
- Observable/replayable primitive calls: Tasks 1-4.
- Phase-aware purpose profiles: Tasks 1-2.
- No silent authority promotion: Tasks 1, 2, 4.
- No benchmark tailoring: final gate only, not implementation logic.

Known risks:

- Initial claim confidence is deterministic and conservative; it is a calibration primitive, not a truth oracle.
- Similar procedure retrieval depends on existing Skill Memory and consolidation projection quality.
- LLM-assisted planning policy is intentionally deferred until deterministic primitive behavior is stable.
