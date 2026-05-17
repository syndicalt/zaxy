# Skill Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class procedural memory lane where skills are event-sourced, projected into the graph, and routed through Memory Checkout without mutating existing factual memory.

**Architecture:** Eventloom remains the source of truth. New `skill.*` events project `Skill` and `SkillVersion` entities plus lifecycle edges into the existing graph, and Memory Checkout exposes applicable skill guidance in a distinct diagnostics/prompt lane. The first release is deterministic and read-only at checkout time; automatic skill amendment is only represented by explicit events.

**Tech Stack:** Python 3.11+, Pydantic/Eventloom models, existing extractor registry in `src/zaxy/extract.py`, Neo4j projection through `GraphStore`, Memory Checkout in `src/zaxy/core.py`, MCP tools in `src/zaxy/mcp_server.py`, pytest/ruff/mypy.

---

## File Structure

- Modify `src/zaxy/extract.py`: add deterministic extractors for `skill.proposed`, `skill.validated`, `skill.applied`, `skill.outcome_recorded`, `skill.revised`, `skill.deprecated`, and `skill.contradicted`.
- Modify `src/zaxy/core.py`: add Skill Memory checkout extraction from ranked contexts and prompt formatting inputs.
- Modify `src/zaxy/mcp_server.py`: expose a read/write MCP helper for appending skill lifecycle events, using the existing append/projection path.
- Modify `docs/agent-events.md`: document the skill lifecycle event taxonomy.
- Modify `docs/mcp.md`: document the MCP-facing skill helper and checkout lane.
- Modify `docs/benchmarks.md`: add the Skill Memory quality guardrail: existing LongMemEval floors must not regress.
- Test in `tests/test_extract.py`, `tests/test_core.py`, `tests/test_mcp.py`, and docs tests.

## Task 1: Skill Event Extractors

**Files:**
- Modify: `src/zaxy/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write failing tests for skill lifecycle extraction**

Add tests near the other typed extractor tests:

```python
def test_extract_skill_proposed_projects_skill_and_version(event_factory):
    event = event_factory(
        type="skill.proposed",
        actor="agent",
        payload={
            "skill_id": "python-test-first",
            "name": "Python test-first implementation",
            "version": "1",
            "summary": "Write the failing pytest before implementation.",
            "procedure": ["Write focused failing test", "Run pytest", "Implement minimum code"],
            "applicability": ["Python feature work", "bug fixes"],
            "citations": ["eventloom://zaxy-default/events/12#abc"],
        },
    )

    result = extract(event)

    assert {entity.entity_type for entity in result.entities} >= {"skill", "skill_version"}
    skill = next(entity for entity in result.entities if entity.entity_type == "skill")
    version = next(entity for entity in result.entities if entity.entity_type == "skill_version")
    assert skill.name == "skill:python-test-first"
    assert skill.summary == "Python test-first implementation"
    assert version.name == "skill:python-test-first:v1"
    assert version.properties["procedure"] == [
        "Write focused failing test",
        "Run pytest",
        "Implement minimum code",
    ]
    assert any(edge.relation_type == "has_version" for edge in result.edges)
```

Add a second test:

```python
def test_extract_skill_outcome_records_application_metrics(event_factory):
    event = event_factory(
        type="skill.outcome_recorded",
        actor="agent",
        payload={
            "skill_id": "python-test-first",
            "version": "1",
            "task": "fix retrieval scoring",
            "success_score": 0.95,
            "feedback": "used",
            "evidence": ["pytest tests/test_query.py::test_scoring -q"],
        },
    )

    result = extract(event)

    assert any(entity.entity_type == "skill_outcome" for entity in result.entities)
    assert any(edge.relation_type == "recorded_outcome" for edge in result.edges)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_extract.py -q --no-cov -k skill_
```

Expected: both tests fail because the extractors are not registered and the generic fallback only creates an `event` entity.

- [ ] **Step 3: Implement minimal deterministic extractors**

Add helpers in `src/zaxy/extract.py`:

```python
def _skill_id(payload: dict[str, Any], event_seq: int) -> str:
    return _required_text(payload.get("skill_id"), field="skill_id", event_seq=event_seq)


def _skill_version(payload: dict[str, Any]) -> str:
    value = payload.get("version")
    return str(value).strip() if value is not None and str(value).strip() else "1"


def _skill_entity(skill_id: str, event: Event, *, summary: str | None = None) -> ExtractedEntity:
    return ExtractedEntity(
        name=f"skill:{skill_id}",
        entity_type="skill",
        observed_at=event.timestamp,
        summary=summary or skill_id,
        properties={"skill_id": skill_id},
    )
```

Register `skill.proposed`:

```python
@register("skill.proposed")
def _extract_skill_proposed(event: Event) -> ExtractionResult:
    skill_id = _skill_id(event.payload, event.seq)
    version = _skill_version(event.payload)
    skill = _skill_entity(skill_id, event, summary=_optional_text_value(event.payload.get("name")))
    version_entity = ExtractedEntity(
        name=f"skill:{skill_id}:v{version}",
        entity_type="skill_version",
        observed_at=event.timestamp,
        summary=_optional_text_value(event.payload.get("summary")),
        properties={
            "skill_id": skill_id,
            "version": version,
            "procedure": _string_list(event.payload.get("procedure")),
            "applicability": _string_list(event.payload.get("applicability")),
            "citations": _string_list(event.payload.get("citations")),
            "status": "proposed",
        },
    )
    return ExtractionResult(
        entities=[skill, version_entity],
        edges=[
            ExtractedEdge(
                source=skill.name,
                target=version_entity.name,
                relation_type="has_version",
                valid_from=event.timestamp,
            )
        ],
        source_event_seq=event.seq,
    )
```

Register `skill.outcome_recorded` similarly, with a `skill_outcome` entity and `recorded_outcome` edge.

- [ ] **Step 4: Run extractor tests to verify green**

Run:

```bash
pytest tests/test_extract.py -q --no-cov -k skill_
```

Expected: skill extractor tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/zaxy/extract.py tests/test_extract.py
git commit -m "feat: project skill memory events"
```

## Task 2: Memory Checkout Skill Lane

**Files:**
- Modify: `src/zaxy/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing checkout test**

Add:

```python
def test_memory_checkout_surfaces_applicable_skills() -> None:
    assembly = ContextAssembly(
        session_id="agent-1",
        prompt="# Context",
        contexts=[
            Context(
                content="Skill Python test-first implementation applies to Python feature work.",
                source="graph",
                score=0.95,
                valid_from="2026-05-17T00:00:00Z",
                valid_to=None,
                metadata={
                    "entity_name": "skill:python-test-first:v1",
                    "entity_type": "skill_version",
                    "citation": "eventloom://agent-1/events/4#abcd",
                    "skill_id": "python-test-first",
                    "procedure": ["Write failing test", "Run pytest", "Implement minimum code"],
                    "applicability": ["Python feature work"],
                    "status": "validated",
                },
            )
        ],
        replay_event_count=0,
        context_counts={"graph": 1, "verbatim": 0, "packet_memory": 0, "replay": 0},
    )

    checkout = build_memory_checkout(query="implement a Python feature", assembly=assembly)

    assert checkout.diagnostics["skills"]["count"] == 1
    assert checkout.diagnostics["skills"]["items"][0]["skill_id"] == "python-test-first"
    assert "## Applicable Skills" in checkout.prompt
    assert "Write failing test" in checkout.prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_core.py -q --no-cov -k applicable_skills
```

Expected: FAIL with missing `skills` diagnostics or missing prompt section.

- [ ] **Step 3: Implement skill selection and prompt formatting**

Add a helper in `src/zaxy/core.py`:

```python
def _checkout_skills(contexts: list[Context], query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    query_tokens = _checkout_tokens(query)
    skills: list[dict[str, Any]] = []
    for context in contexts:
        metadata = context.metadata or {}
        if metadata.get("entity_type") != "skill_version":
            continue
        applicability = metadata.get("applicability")
        applicability_text = " ".join(applicability) if isinstance(applicability, list) else ""
        overlap = len(query_tokens & _checkout_tokens(f"{context.content} {applicability_text}"))
        if overlap <= 0:
            continue
        procedure = metadata.get("procedure")
        skills.append(
            {
                "skill_id": metadata.get("skill_id") or metadata.get("entity_name"),
                "name": metadata.get("entity_name"),
                "score": context.score,
                "citation": _context_citation(context),
                "procedure": procedure if isinstance(procedure, list) else [],
                "applicability": applicability if isinstance(applicability, list) else [],
                "status": metadata.get("status"),
            }
        )
    return sorted(skills, key=lambda row: float(row["score"]), reverse=True)[:limit]
```

Call it inside `build_memory_checkout`, then add `diagnostics = {**diagnostics, "skills": {"count": len(skills), "items": skills}}` when skills exist. Extend `format_memory_checkout_prompt` to render:

```text
## Applicable Skills
- skill_id=python-test-first status=validated citation=eventloom://...
  procedure:
  1. Write failing test
  2. Run pytest
  3. Implement minimum code
```

- [ ] **Step 4: Run checkout test to verify green**

Run:

```bash
pytest tests/test_core.py -q --no-cov -k applicable_skills
```

Expected: PASS.

- [ ] **Step 5: Run existing checkout tests**

Run:

```bash
pytest tests/test_core.py -q --no-cov -k checkout_memory
```

Expected: existing checkout behavior still passes.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/zaxy/core.py tests/test_core.py
git commit -m "feat: route skill memory through checkout"
```

## Task 3: MCP Skill Helper

**Files:**
- Modify: `src/zaxy/mcp_server.py`
- Test: `tests/test_mcp.py`

- [ ] **Step 1: Write failing MCP test**

Add:

```python
async def test_memory_skill_event_appends_and_projects(mcp_server):
    graph = AsyncMock()
    mcp_server.graph = graph

    response = await mcp_server.handle_memory_skill(
        {
            "action": "proposed",
            "skill_id": "python-test-first",
            "name": "Python test-first implementation",
            "procedure": ["Write failing test", "Run pytest"],
            "session_id": "agent-1",
        }
    )

    payload = json.loads(response[0].text)
    assert payload["event_type"] == "skill.proposed"
    graph.upsert_extraction.assert_awaited_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_mcp.py -q --no-cov -k memory_skill
```

Expected: FAIL because `handle_memory_skill` does not exist.

- [ ] **Step 3: Implement MCP tool registration and handler**

Add tool metadata near `memory_feedback`:

```python
Tool(
    name="memory_skill",
    description="Append a skill lifecycle event and project it into memory.",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["proposed", "validated", "applied", "outcome_recorded", "revised", "deprecated", "contradicted"]},
            "skill_id": {"type": "string"},
            "name": {"type": "string"},
            "procedure": {"type": "array", "items": {"type": "string"}},
            "session_id": {"type": "string"},
        },
        "required": ["action", "skill_id"],
    },
)
```

Add handler:

```python
async def handle_memory_skill(self, arguments: dict[str, Any]) -> list[TextContent]:
    action = _required_text(arguments.get("action"), "action")
    if action not in {"proposed", "validated", "applied", "outcome_recorded", "revised", "deprecated", "contradicted"}:
        raise ValueError("unsupported skill action")
    session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
    payload = validate_payload({key: value for key, value in arguments.items() if key not in {"action", "session_id"}})
    eventlog = self.session_manager.get(session_id).eventlog
    event_type = f"skill.{action}"
    event = eventlog.append(event_type, actor="zaxy", payload=payload, thread=session_id)
    await self.graph.upsert_extraction(extract(event), session_id=session_id)
    return [TextContent(type="text", text=json.dumps({"seq": event.seq, "hash": event.hash, "event_type": event_type}))]
```

Route it in the central MCP call dispatcher.

- [ ] **Step 4: Run MCP test to verify green**

Run:

```bash
pytest tests/test_mcp.py -q --no-cov -k memory_skill
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/zaxy/mcp_server.py tests/test_mcp.py
git commit -m "feat: add memory skill mcp tool"
```

## Task 4: Documentation And Guardrails

**Files:**
- Modify: `docs/agent-events.md`
- Modify: `docs/mcp.md`
- Modify: `docs/benchmarks.md`
- Test: `tests/test_docs_site.py`, `tests/test_packaging.py`

- [ ] **Step 1: Document skill events**

Add to `docs/agent-events.md`:

```markdown
## Skill Memory

Use `skill.proposed` when a reusable procedure is identified. Use
`skill.validated` only after explicit evaluation. Use `skill.applied` and
`skill.outcome_recorded` to connect a procedure to a task and outcome. Use
`skill.revised`, `skill.deprecated`, and `skill.contradicted` to preserve
version history without rewriting earlier events.
```

- [ ] **Step 2: Document MCP and checkout behavior**

Add to `docs/mcp.md`:

```markdown
`memory_skill(action, skill_id, ...)` appends skill lifecycle events. Memory
Checkout may return an `Applicable Skills` section when graph retrieval finds
validated skill versions relevant to the current task.
```

- [ ] **Step 3: Document benchmark guardrail**

Add to `docs/benchmarks.md`:

```markdown
Skill Memory changes must pass the full 500-question no-regression guardrail:
mean score >= 0.626, Answer@5 >= 0.608, R@5 >= 0.956, citation coverage = 1.000.
```

- [ ] **Step 4: Run docs tests**

Run:

```bash
pytest tests/test_docs_site.py tests/test_packaging.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 5: Run benchmark guardrail command**

Run:

```bash
python -m zaxy benchmark-compare reports/benchmarks/longmemeval-500-hash/live-benchmark.json \
  --backend zaxy-checkout \
  --min-mean-score 0.626 \
  --min-answer-recall-at-5 0.608 \
  --min-recall-at-5 0.956 \
  --min-citation-coverage 1.0 \
  --max-p95-ms 15000 \
  --max-p99-ms 23000
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add docs/agent-events.md docs/mcp.md docs/benchmarks.md
git commit -m "docs: document skill memory lifecycle"
```

## Task 5: Final Verification

**Files:**
- Read-only verification over changed files.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_extract.py tests/test_core.py tests/test_mcp.py tests/test_docs_site.py tests/test_packaging.py -q --no-cov
```

Expected: PASS.

- [ ] **Step 2: Run static checks**

Run:

```bash
ruff check src/zaxy/extract.py src/zaxy/core.py src/zaxy/mcp_server.py tests/test_extract.py tests/test_core.py tests/test_mcp.py
mypy src/zaxy/extract.py src/zaxy/core.py src/zaxy/mcp_server.py
```

Expected: ruff clean and mypy success.

- [ ] **Step 3: Run benchmark no-regression guardrail**

Run:

```bash
python -m zaxy benchmark-compare reports/benchmarks/longmemeval-500-hash/live-benchmark.json \
  --backend zaxy-checkout \
  --min-mean-score 0.626 \
  --min-answer-recall-at-5 0.608 \
  --min-recall-at-5 0.956 \
  --min-citation-coverage 1.0 \
  --max-p95-ms 15000 \
  --max-p99-ms 23000
```

Expected: PASS.

- [ ] **Step 4: Final commit if needed**

If verification required doc fixes, commit them:

```bash
git add .
git commit -m "chore: finalize skill memory rollout"
```

Expected: no uncommitted changes remain after the final commit.

## Self-Review

- Spec coverage: skill lifecycle events, graph projection, checkout skill routing, outcome tracking, docs, and no-regression guardrails are each mapped to a task.
- pgGraph is intentionally excluded; it has a separate backend evaluation design because it touches projection infrastructure rather than procedural memory.
- Placeholder scan: no unresolved markers or unspecified implementation steps remain.
- Type consistency: `skill_id`, `version`, `procedure`, `applicability`, `status`, and `skill_version` are used consistently across extractor, checkout, MCP, and docs tasks.
