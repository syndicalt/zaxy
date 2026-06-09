# Codebase Review High-Value Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the high-value findings from `docs/research/codebase-review-2026-06-09.md` without reward-hacking, broad rewrites, or unnecessary product surface expansion.

**Architecture:** Fix security issues first, then remove the event-log and projection-cache hot-path costs that affect MCP checkout/query latency. Keep Eventloom as the source of truth, keep graph projections rebuildable, and preserve all release benchmark guardrails. Defer large maintainability refactors until after performance/security regressions are measured and the package boundary is designed.

**Tech Stack:** Python 3.11+, Typer, MCP Python SDK, Eventloom JSONL, embedded Kuzu `ProjectionStore`, pytest, ruff, mypy, docs validation.

---

## Scope Boundary

This plan covers the review items with the best risk-to-effort ratio:

- S1 dashboard stored-XSS gap.
- S2 dashboard CSRF/origin weakness for state-changing routes.
- S3 Docker non-loopback unauthenticated default.
- S4 admin-token timing comparison.
- S6 `SessionManager.get` cache-key bug.
- P1 `EventLog.append_many` full-file tail lookup.
- P2 hot-path `replay()` double parse and mandatory verify behavior.
- P3 lifecycle capture invalidating all embedded read caches on no-op projections.
- P5 sync HTTP sleeps/clients inside async embedding/reranker paths.
- P6 dashboard request-lifetime backend churn.
- A2 benchmark/eval code shipped in the runtime wheel.

Explicit non-scope for this plan:

- Rewriting `retrieval_plan.py` or `synthesis.py`.
- Replacing embedded Kuzu vector search.
- pgGraph HNSW/causal BFS work.
- Changing benchmark scoring logic.
- Removing product features solely to shrink line count.

## Worktree and Safety Requirements

- Start from clean synced `master`.
- Do not touch existing untracked assets unless explicitly assigned.
- Create one branch for this remediation series:

```bash
git checkout master
git pull --ff-only origin master
git switch -c hardening/codebase-review-2026-06-09
git status --short
```

Expected status before Task 1:

```text

```

If untracked generated media or review artifacts are present, leave them untracked
or move them outside the repo before beginning. Do not delete them.

## Validation Gates

Run focused tests after each task. Before opening a PR, run:

```bash
ruff check src/ tests/
mypy src/zaxy
pytest tests/test_dashboard.py tests/test_event.py tests/test_session.py tests/test_mcp.py tests/test_embedded_graph_store.py -q
python scripts/build-site-docs.py --check
scripts/validate-docs.sh
```

For the final PR gate, run the release check if time permits:

```bash
scripts/release-check.sh --root .
```

If the full release check is too expensive for the PR iteration, state exactly
which focused checks passed and why the full gate was deferred.

---

## File Structure

Modify:

- `src/zaxy/dashboard.py`  
  Escape untrusted dashboard event/session fields. Add state-changing route
  request validation. Prepare for a longer-lived dashboard app backend in a
  later task.

- `tests/test_dashboard.py`  
  Add XSS regression tests and state-changing POST origin/host validation tests.

- `src/zaxy/event.py`  
  Add tail-read helpers and replay verification controls.

- `tests/test_event.py`  
  Prove append no longer reads the whole file to discover the tail and prove
  replay can skip integrity verification by default where callers permit it.

- `src/zaxy/session.py`  
  Store sessions under the validated key.

- `tests/test_session.py`  
  Add a normalization-resilience regression test.

- `src/zaxy/mcp_server.py`  
  Use constant-time admin token comparison and avoid unnecessary full-log replay
  for `HEAD` where possible.

- `tests/test_mcp.py` or `tests/test_remote_security.py`  
  Cover admin token comparison behavior and any public MCP behavior changes.

- `Dockerfile`  
  Set production environment by default or block unauthenticated non-loopback
  binding.

- `tests/test_packaging.py` or `tests/test_config.py`  
  Cover Dockerfile/config posture.

- `src/zaxy/embedded_graph_store.py`  
  Avoid clearing read caches when an extraction projects only an `Event` node and
  no entities or edges.

- `tests/test_embedded_graph_store.py`  
  Add no-op extraction cache-preservation tests.

- `src/zaxy/query.py`, `src/zaxy/embedding.py`  
  Remove blocking sync HTTP/sleep from async paths using async clients or
  `asyncio.to_thread`.

- `tests/test_query.py`, `tests/test_embedding.py`  
  Add non-blocking behavior tests around retry sleep/client calls.

- `pyproject.toml`, possible new `benchmarks/` package tree, CLI shims in
  `src/zaxy/__main__.py`  
  Move benchmark implementation code out of the production import surface while
  preserving CLI compatibility.

---

## Task 1: Dashboard XSS and POST Request Guard

**Files:**
- Modify: `src/zaxy/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing dashboard HTML escaping tests**

Add tests that verify the dashboard shell escapes every untrusted field rendered
through the sessions and recent-events templates. The current bug is in the
JavaScript template literals, so the regression test should inspect
`render_dashboard_html()` rather than execute a browser.

```python
def test_dashboard_session_and_event_templates_escape_untrusted_fields() -> None:
    html = render_dashboard_html()

    assert "${escapeHtml(session.session_id)}" in html
    assert "${escapeHtml(session.latest_type || \"\")}" in html
    assert "${escapeHtml(event.session_id)}" in html
    assert "${escapeHtml(event.type)}" in html
    assert "${escapeHtml(event.actor)}" in html
    assert "${escapeHtml(event.summary || \"\")}" in html
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
pytest tests/test_dashboard.py::test_dashboard_session_and_event_templates_escape_untrusted_fields -q
```

Expected before implementation: FAIL because raw `${session.session_id}`,
`${event.type}`, `${event.actor}`, or `${event.summary || ""}` appears in the
dashboard template.

- [ ] **Step 3: Escape the affected dashboard fields**

In `render_dashboard_html()`, update the two vulnerable templates:

```javascript
document.getElementById("sessions-body").innerHTML = status.memory.sessions.map((session) => `
  <tr><td><code>${escapeHtml(session.session_id)}</code></td><td>${session.event_count}</td><td>${escapeHtml(session.latest_type || "")}</td><td>${session.integrity_ok ? "OK" : "FAILED"}</td></tr>
`).join("");
document.getElementById("events-body").innerHTML = events.events.map((event) => `
  <tr><td><code>${escapeHtml(event.session_id)}</code></td><td>${event.seq}</td><td>${escapeHtml(event.type)}</td><td>${escapeHtml(event.actor)}</td><td>${escapeHtml(event.summary || "")}</td></tr>
`).join("");
```

Do not escape numeric fields with `escapeHtml`; keep them numeric in the API and
render directly.

- [ ] **Step 4: Write failing POST guard tests**

Add route-level tests for state-changing dashboard POST routes. Prefer `Host`
and `Origin` validation because it works without introducing persistent auth
state or a new dashboard login.

```python
def test_dashboard_coordinate_post_rejects_cross_origin_request(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status, _headers, body = app.handle_api(
        "POST",
        "/api/coordinate/apply-approval",
        "",
        body=json.dumps({"mission_id": "m1", "decisions": []}),
        request_headers={"host": "127.0.0.1:8765", "origin": "https://attacker.example"},
    )

    assert status == 403
    assert body["error"] == "forbidden_origin"


def test_dashboard_coordinate_post_allows_same_origin_request(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status, _headers, body = app.handle_api(
        "POST",
        "/api/coordinate/apply-approval",
        "",
        body=json.dumps({"mission_id": "missing", "decisions": []}),
        request_headers={"host": "127.0.0.1:8765", "origin": "http://127.0.0.1:8765"},
    )

    assert status != 403
```

If the current `handle_api` signature does not accept headers, this test should
fail with `TypeError`.

- [ ] **Step 5: Add request-header plumbing and validation**

Change `DashboardApp.handle_api` to accept optional request headers:

```python
def handle_api(
    self,
    method: str,
    path: str,
    query: str,
    *,
    body: str | bytes | None = None,
    request_headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], dict[str, Any]]:
```

Add helper functions in `dashboard.py`:

```python
def _dashboard_origin_allowed(scope: DashboardScope, headers: Mapping[str, str] | None) -> bool:
    normalized = {key.lower(): value for key, value in (headers or {}).items()}
    host = normalized.get("host", "")
    origin = normalized.get("origin")
    expected_host = f"{scope.host}:{scope.port}"
    if host and host != expected_host:
        return False
    if origin is None:
        return True
    return origin in {f"http://{expected_host}", f"https://{expected_host}"}
```

Before handling `/api/coordinate/review`,
`/api/coordinate/review-finding`, or `/api/coordinate/apply-approval`, reject
when `_dashboard_origin_allowed` returns false:

```python
if method.upper() == "POST" and path.startswith("/api/coordinate/"):
    if not _dashboard_origin_allowed(self.scope, request_headers):
        return 403, headers, {"error": "forbidden_origin"}
```

In `create_dashboard_handler.do_POST`, pass `dict(self.headers.items())` into
`handle_api`.

- [ ] **Step 6: Run dashboard tests**

Run:

```bash
pytest tests/test_dashboard.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/zaxy/dashboard.py tests/test_dashboard.py
git commit -m "fix: harden dashboard mutation routes"
```

---

## Task 2: Small Security and Runtime Posture Fixes

**Files:**
- Modify: `src/zaxy/mcp_server.py`
- Modify: `src/zaxy/session.py`
- Modify: `Dockerfile`
- Test: `tests/test_session.py`
- Test: `tests/test_mcp.py` or `tests/test_remote_security.py`
- Test: `tests/test_packaging.py` or `tests/test_config.py`

- [ ] **Step 1: Add a session cache-key regression test**

Add to `tests/test_session.py`:

```python
def test_get_stores_session_under_validated_id(monkeypatch: pytest.MonkeyPatch, tmp_base: str) -> None:
    import zaxy.session as session_module

    monkeypatch.setattr(session_module, "validate_session_id", lambda value: value.strip())
    mgr = SessionManager(base_path=tmp_base)

    session = mgr.get(" agent-1 ")

    assert session.session_id == "agent-1"
    assert mgr.get("agent-1") is session
    assert mgr.list_sessions() == ["agent-1"]
```

- [ ] **Step 2: Fix `SessionManager.get`**

Change the store key:

```python
if safe_id not in self._sessions:
    log_path = eventlog_path(self._base, safe_id)
    self._sessions[safe_id] = Session(
        session_id=safe_id,
        eventlog=EventLog(str(log_path)),
    )
return self._sessions[safe_id]
```

- [ ] **Step 3: Add admin-token behavior coverage**

Add a focused test around `_require_admin` that proves equivalent tokens pass
and missing/wrong tokens fail. Do not test timing directly.

```python
def test_mcp_admin_token_gate_accepts_exact_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = ZaxyMCPServer()
    server._admin_token = "secret-admin-token"

    server._require_admin({"admin_token": "secret-admin-token"})


def test_mcp_admin_token_gate_rejects_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    server = ZaxyMCPServer()
    server._admin_token = "secret-admin-token"

    with pytest.raises(PermissionError, match="admin_token"):
        server._require_admin({"admin_token": "wrong"})
```

If direct construction is too heavy for the test module, use the same fixture or
factory already used for other `ZaxyMCPServer` tests.

- [ ] **Step 4: Use constant-time admin-token comparison**

Update `_require_admin`:

```python
def _require_admin(self, arguments: dict[str, Any]) -> None:
    """Require an admin token for destructive or bulk-read tools when configured."""
    if not self._admin_token:
        return
    supplied = str(arguments.get("admin_token") or "")
    if not hmac.compare_digest(supplied, self._admin_token):
        raise PermissionError("admin_token is required for this tool")
```

- [ ] **Step 5: Add Docker posture test**

Add a packaging/config test that reads `Dockerfile` and requires production
environment by default:

```python
def test_dockerfile_defaults_to_production_environment() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ENV ZAXY_ENV=production" in dockerfile
```

- [ ] **Step 6: Set Docker production default**

Add after `WORKDIR /app` in the final runtime stage:

```dockerfile
ENV ZAXY_ENV=production
```

This keeps `docker run` from accidentally exposing unauthenticated SSE on
`0.0.0.0`; local dev can still override with `-e ZAXY_ENV=development`.

- [ ] **Step 7: Run focused tests**

Run:

```bash
pytest tests/test_session.py tests/test_mcp.py tests/test_packaging.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/zaxy/session.py src/zaxy/mcp_server.py Dockerfile tests/test_session.py tests/test_mcp.py tests/test_packaging.py
git commit -m "fix: tighten local security defaults"
```

---

## Task 3: EventLog Tail Read and Replay Verification Controls

**Files:**
- Modify: `src/zaxy/event.py`
- Test: `tests/test_event.py`

- [ ] **Step 1: Add tail-read behavior tests**

Add tests that fail if append reads the whole file just to discover the last
event. The safest regression is to patch `_event_from_json_line` and assert it
is called once for tail discovery, not once per line.

```python
def test_append_many_reads_only_tail_event_for_sequence(monkeypatch: pytest.MonkeyPatch, tmp_eventlog: EventLog) -> None:
    for index in range(50):
        tmp_eventlog.append("seed", actor="tester", payload={"index": index})

    import zaxy.event as event_module

    calls = 0
    original = event_module._event_from_json_line

    def counting_event_from_json_line(line: str, *, seq_hint: int | None = None) -> Event:
        nonlocal calls
        calls += 1
        return original(line, seq_hint=seq_hint)

    monkeypatch.setattr(event_module, "_event_from_json_line", counting_event_from_json_line)

    tmp_eventlog.append_many([{"event_type": "tail", "actor": "tester"}])

    assert calls == 1
```

- [ ] **Step 2: Add replay verification control tests**

Add:

```python
def test_replay_can_skip_integrity_verification(monkeypatch: pytest.MonkeyPatch, tmp_eventlog: EventLog) -> None:
    tmp_eventlog.append("a", actor="tester")
    tmp_eventlog.append("b", actor="tester")

    def fail_verify() -> object:
        raise AssertionError("verify should not run when verify_integrity=False")

    monkeypatch.setattr(tmp_eventlog, "verify", fail_verify)

    result = tmp_eventlog.replay(from_seq=2, verify_integrity=False)

    assert [event.type for event in result.events] == ["b"]
    assert result.integrity is None
```

This requires `ReplayResult.integrity` to allow `None`, or a new lightweight
status object. Prefer `IntegrityReport | None` because callers that need
integrity can request it explicitly.

- [ ] **Step 3: Implement a private locked tail helper**

Add a helper in `event.py`:

```python
def _read_last_line(fh: TextIO) -> str | None:
    fh.seek(0, os.SEEK_END)
    end = fh.tell()
    if end == 0:
        return None
    position = end - 1
    while position >= 0:
        fh.seek(position)
        char = fh.read(1)
        if char == "\n" and position != end - 1:
            break
        position -= 1
    fh.seek(max(position + 1, 0))
    line = fh.readline()
    return line or None
```

Import `os` and `TextIO` as needed. Account for a trailing newline.

- [ ] **Step 4: Use the tail helper in `append_many`**

Replace the full `fh.readlines()` tail lookup with:

```python
last_line = _read_last_line(fh)
seq = 1
prev_hash: str | None = None
if last_line:
    last = _event_from_json_line(last_line)
    seq = last.seq + 1
    prev_hash = last.hash
write_v1 = _should_write_eventloom_v1_from_tail(last_line, items)
```

If `_should_write_eventloom_v1` currently needs all lines, split it into a
tail-aware helper. Preserve legacy compatibility behavior with a focused test.

- [ ] **Step 5: Add `verify_integrity` parameter to `replay`**

Change:

```python
def replay(
    self,
    from_seq: int = 1,
    to_seq: int | None = None,
    *,
    verify_integrity: bool = True,
) -> ReplayResult:
```

Then:

```python
integrity = self.verify() if verify_integrity else None
return ReplayResult(events=filtered, integrity=integrity)
```

Update `ReplayResult` accordingly.

- [ ] **Step 6: Keep CLI/admin paths verifying integrity**

Search callers:

```bash
rg -n "\\.replay\\(" src tests
```

Keep user-facing integrity commands and administrative replay paths on the
default `verify_integrity=True`. For hot read-only internal paths that do their
own status checks, pass `verify_integrity=False` in later tasks.

- [ ] **Step 7: Run event tests**

Run:

```bash
pytest tests/test_event.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/zaxy/event.py tests/test_event.py
git commit -m "perf: avoid full log reads on append"
```

---

## Task 4: Apply Fast Replay to MCP Checkout, Refs, and Status Hot Paths

**Files:**
- Modify: `src/zaxy/mcp_server.py`
- Modify: `src/zaxy/session.py`
- Modify: `src/zaxy/refs.py`
- Modify: `src/zaxy/memory_status.py`
- Test: `tests/test_mcp.py`
- Test: `tests/test_session.py`
- Test: `tests/test_memory_status.py` if present, otherwise add focused coverage to the closest existing status test file.

- [ ] **Step 1: Add a `verify_integrity` passthrough to `SessionManager.replay`**

Test:

```python
def test_session_manager_replay_can_skip_integrity(tmp_base: str, monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = SessionManager(base_path=tmp_base)
    session = mgr.get("agent-1")
    session.eventlog.append("goal.created", "user", {"title": "t1"})

    result = mgr.replay("agent-1", from_seq=1, verify_integrity=False)

    assert len(result.events) == 1
    assert result.integrity is None
```

Implementation:

```python
def replay(self, session_id: str, from_seq: int = 1, *, verify_integrity: bool = True) -> Any:
    session = self.get(session_id)
    return session.eventlog.replay(from_seq=from_seq, verify_integrity=verify_integrity)
```

- [ ] **Step 2: Use unverified replay in model-facing read paths that do not report integrity**

Update MCP checkout/context paths where integrity is not returned to the model:

```python
replay = self.session_manager.replay(
    session_id,
    from_seq=replay_from_seq,
    verify_integrity=False,
)
```

Do not change `memory_replay` unless the tool schema adds an explicit
`verify_integrity` option. Users who call replay expect integrity reporting.

- [ ] **Step 3: Resolve checkout `HEAD` from a tail read**

Add an `EventLog.tail()` or `last_event()` helper in `event.py` if Task 3 did
not already expose one:

```python
def last_event(self) -> Event | None:
    with open(self.path, "a+", encoding="utf-8") as fh:
        self._lock(fh.fileno(), exclusive=False)
        try:
            last_line = _read_last_line(fh)
            return _event_from_json_line(last_line) if last_line else None
        finally:
            self._unlock(fh.fileno())
```

Use it in `_resolve_checkout_ref("HEAD")` instead of replaying from seq 1.

- [ ] **Step 4: Add signature-keyed read caches only where cheap and local**

For `refs.py` and `memory_status.py`, add small per-instance caches keyed by:

```python
signature = (path.stat().st_mtime_ns, path.stat().st_size)
```

Do not add global mutable caches. Keep invalidation obvious.

- [ ] **Step 5: Run focused tests**

Run:

```bash
pytest tests/test_mcp.py tests/test_session.py tests/test_event.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/zaxy/event.py src/zaxy/session.py src/zaxy/mcp_server.py src/zaxy/refs.py src/zaxy/memory_status.py tests/test_event.py tests/test_session.py tests/test_mcp.py
git commit -m "perf: avoid verified full replay in checkout hot paths"
```

---

## Task 5: Preserve Embedded Read Caches for No-Op Lifecycle Projections

**Files:**
- Modify: `src/zaxy/embedded_graph_store.py`
- Test: `tests/test_embedded_graph_store.py`

- [ ] **Step 1: Write failing cache-preservation test**

Add:

```python
@pytest.mark.asyncio
async def test_upsert_extraction_preserves_read_caches_for_event_only_projection(tmp_path: Path) -> None:
    store = EmbeddedGraphStore(tmp_path / "embedded.kuzu")
    await store.connect()
    await store.init_schema()
    store._current_entity_index_cache["agent-1"] = []
    store._keyword_index_cache["agent-1"] = embedded_graph_store._KeywordIndex([], [], {}, {}, [])
    result = ExtractionResult(
        source_event_seq=1,
        source_event_hash="a" * 64,
        source_event_type="tool.call.completed",
        source_thread="agent-1",
        entities=[],
        edges=[],
    )

    await store.upsert_extraction(result, session_id="agent-1")

    assert "agent-1" in store._current_entity_index_cache
    assert "agent-1" in store._keyword_index_cache
    await store.close()
```

Adjust constructor arguments to match the current `ExtractionResult` signature.

- [ ] **Step 2: Move cache invalidation after projected-content check**

In `upsert_extraction`, compute whether the extraction mutates entity/edge read
state:

```python
mutates_read_indexes = bool(result.entities or result.edges)
if mutates_read_indexes:
    self._clear_read_caches(session_id)
if self._bulk_projection_open and mutates_read_indexes:
    self._dirty_bulk_sessions.add(session_id)
```

Still project the `Event` node and `NEXT_EVENT`/`PREVIOUS_EVENT` chain for
lifecycle events. Only avoid clearing entity/keyword/vector/traversal caches
when no entities or edges are projected.

- [ ] **Step 3: Run embedded tests**

Run:

```bash
pytest tests/test_embedded_graph_store.py -q
```

Expected: PASS or SKIP if Kuzu is not installed. If skipped locally, run the CI
integration job or a Kuzu-enabled environment before merging.

- [ ] **Step 4: Commit**

```bash
git add src/zaxy/embedded_graph_store.py tests/test_embedded_graph_store.py
git commit -m "perf: preserve caches for event-only projections"
```

---

## Task 6: Remove Blocking HTTP and Sleep From Async Provider Paths

**Files:**
- Modify: `src/zaxy/query.py`
- Modify: `src/zaxy/embedding.py`
- Test: `tests/test_query.py`
- Test: `tests/test_embedding.py`

- [ ] **Step 1: Write tests that patch `time.sleep` to fail inside async provider calls**

Add a retry test in `tests/test_embedding.py` using a fake response sequence. The
test should fail if `_post_with_retries` calls `time.sleep` while invoked from
an async path.

```python
@pytest.mark.asyncio
async def test_hosted_embedding_provider_retry_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    import zaxy.embedding as embedding_module

    def fail_sleep(_seconds: float) -> None:
        raise AssertionError("time.sleep must not run in async provider paths")

    monkeypatch.setattr(embedding_module.time, "sleep", fail_sleep)

    # Use the existing hosted-provider fake HTTP test pattern here. The provider
    # should call an async sleep or be executed through asyncio.to_thread.
```

Complete the fake using the current hosted embedding tests in
`tests/test_embedding.py`.

- [ ] **Step 2: Convert hosted rerankers to `httpx.AsyncClient`**

In `query.py`, replace sync client construction inside `async def rerank` with:

```python
async with httpx.AsyncClient(timeout=self.timeout) as client:
    response = await client.post(url, headers=headers, json=payload)
```

If the provider class is intentionally sync elsewhere, isolate the async method
with `await asyncio.to_thread(self._rerank_sync, query, candidates)`.

- [ ] **Step 3: Convert embedding retry sleeps**

Prefer an async helper:

```python
async def _apost_with_retries(...) -> dict[str, Any]:
    ...
    await asyncio.sleep(delay)
```

If broad async conversion would sprawl, use `asyncio.to_thread` at the call site
as an interim production fix:

```python
return await asyncio.to_thread(self._embed_sync, texts)
```

The important invariant is that MCP/SSE event loop handlers do not block on
`time.sleep` or sync network I/O.

- [ ] **Step 4: Run provider tests**

Run:

```bash
pytest tests/test_query.py tests/test_embedding.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/zaxy/query.py src/zaxy/embedding.py tests/test_query.py tests/test_embedding.py
git commit -m "perf: avoid blocking provider calls in async paths"
```

---

## Task 7: Dashboard Backend Lifetime and Tail Event Listings

**Files:**
- Modify: `src/zaxy/dashboard.py`
- Test: `tests/test_dashboard.py`

- [ ] **Step 1: Add tests for tail-limited event listing**

Add:

```python
def test_dashboard_events_reads_tail_limit_without_loading_full_history(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    log = EventLog(workspace / ".eventloom" / "default.jsonl")
    for index in range(20):
        log.append("event.recorded", actor="tester", payload={"summary": f"event {index}"})

    app = DashboardApp(resolve_dashboard_scope(DashboardConfig(workspace=workspace)))

    status, _headers, body = app.handle_api("GET", "/api/events", "session_id=default&limit=3")

    assert status == 200
    assert [event["summary"] for event in body["events"]] == ["event 17", "event 18", "event 19"]
```

This locks behavior before optimizing implementation.

- [ ] **Step 2: Add a reusable `EventLog.tail_events(limit)` helper if not already present**

Use the same backward line-reading primitive from Task 3. Return events in
ascending sequence order.

```python
def tail_events(self, limit: int) -> list[Event]:
    if limit <= 0:
        return []
    ...
    return list(reversed(events))
```

- [ ] **Step 3: Use tail reads for dashboard event listings**

Update dashboard event listing code to call `EventLog.tail_events(limit)` when
there is no `from_seq` or broad filter that requires a full scan. Fall back to
`read_all()` only when filters require it.

- [ ] **Step 4: Plan persistent fabric/store separately**

Do not fold a full persistent dashboard backend into this task unless the tail
change is already passing. Create a follow-up issue or task for:

- one `MemoryFabric` per `DashboardApp`;
- one connected embedded store per dashboard lifetime;
- a `threading.RLock` around store access;
- clean close on dashboard shutdown.

- [ ] **Step 5: Run dashboard tests**

Run:

```bash
pytest tests/test_dashboard.py tests/test_event.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/zaxy/dashboard.py src/zaxy/event.py tests/test_dashboard.py tests/test_event.py
git commit -m "perf: tail-read dashboard event listings"
```

---

## Task 8: Benchmark Code Packaging Boundary Design

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/zaxy/__main__.py`
- Move or create: `benchmarks/zaxy_benchmarks/`
- Test: `tests/test_packaging.py`
- Test: `tests/test_cli.py`

### 2026-06-09 Package-Boundary Migration Result

The follow-up slice moved benchmark/eval implementation modules out of the
runtime wheel package while keeping source-checkout eval workflows intact.

Implemented boundary:

- Extracted the runtime-shared event text formatter into `src/zaxy/event_context.py`.
- Moved heavy eval modules from `src/zaxy/` to the top-level source package
  `zaxy_benchmarks/`.
- Updated benchmark modules and tests to import through `zaxy_benchmarks.*`.
- Kept Hatch wheel packaging limited to `packages = ["src/zaxy"]`, so end-user
  installs do not ship benchmark/eval implementations.
- Added `zaxy_benchmarks` to the sdist include list, so source distributions
  retain eval tooling for users who intentionally work from source.
- Updated benchmark CLI commands to resolve `zaxy_benchmarks.*` through a runtime
  helper that raises a clear `typer.BadParameter` when eval tooling is not
  installed/available, instead of a raw `ModuleNotFoundError`.
- Added packaging coverage proving moved eval modules are absent from `src/zaxy`
  and present in `zaxy_benchmarks`.
- Verified built wheel contents contain no `zaxy_benchmarks/` entries and no
  runtime benchmark/LongMemBench modules.

Explicitly out of scope:

- `external_validation.py` remains in `src/zaxy` because it is release/package
  validation tooling imported by `src/zaxy/release.py`, scripts, and packaging
  tests. Moving it should be a separate release-tooling boundary decision.

- [ ] **Step 1: Inventory benchmark modules and CLI entry points**

Run:

```bash
find src/zaxy -maxdepth 1 -type f \( -name '*benchmark*.py' -o -name 'longmembench.py' -o -name 'rc_benchmark_freeze.py' -o -name 'external_validation.py' \) -print
rg -n "benchmark|longmembench|rc_benchmark|harvey|external_validation" src/zaxy/__main__.py tests
```

Record which modules are used by public CLI commands and which are internal
helpers.

- [ ] **Step 2: Add packaging tests for runtime wheel boundary**

Before moving code, add tests that define the desired boundary:

```python
def test_runtime_package_does_not_ship_heavy_benchmark_modules() -> None:
    packaged = Path("src/zaxy")
    heavy_modules = {
        "harvey_lab_benchmark.py",
        "live_benchmark.py",
        "longmembench.py",
        "rc_benchmark_freeze.py",
    }

    assert not any((packaged / module).exists() for module in heavy_modules)
```

This test should fail before the move.

- [ ] **Step 3: Move implementation modules to a non-runtime benchmark tree**

Use `git mv` for modules that are not required at runtime:

```bash
mkdir -p benchmarks/zaxy_benchmarks
git mv src/zaxy/harvey_lab_benchmark.py benchmarks/zaxy_benchmarks/harvey_lab_benchmark.py
git mv src/zaxy/live_benchmark.py benchmarks/zaxy_benchmarks/live_benchmark.py
git mv src/zaxy/longmembench.py benchmarks/zaxy_benchmarks/longmembench.py
git mv src/zaxy/rc_benchmark_freeze.py benchmarks/zaxy_benchmarks/rc_benchmark_freeze.py
```

Move additional benchmark modules only after confirming no runtime imports.

- [ ] **Step 4: Preserve CLI compatibility with lazy optional imports**

In CLI commands that invoke benchmarks, import from the benchmark tree and emit
a clear installation/source-tree error if unavailable:

```python
try:
    from zaxy_benchmarks.longmembench import run_longmembench
except ModuleNotFoundError as exc:
    raise typer.BadParameter(
        "Benchmark commands require the repository benchmark package. "
        "Run from a source checkout or install the zaxy-benchmarks extra."
    ) from exc
```

Do not import benchmark modules during `zaxy --help`.

- [ ] **Step 5: Update packaging config**

Exclude `benchmarks/` from the production wheel unless a separate
`zaxy-benchmarks` package is intentionally added. Keep source-tree tests able to
import the moved modules by adding a test-only path adjustment or package
configuration.

- [ ] **Step 6: Run CLI and packaging tests**

Run:

```bash
pytest tests/test_cli.py tests/test_packaging.py -q
python -m zaxy --help >/tmp/zaxy-help.txt
```

Expected: PASS; help command must not import benchmark runtime modules.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/zaxy/__main__.py benchmarks tests/test_cli.py tests/test_packaging.py
git commit -m "refactor: move heavy benchmarks out of runtime package"
```

---

## Task 9: Final Verification and PR

**Files:**
- No source changes unless verification reveals failures.

- [ ] **Step 1: Run lint and type checks**

```bash
ruff check src/ tests/
mypy src/zaxy
```

Expected: both pass.

- [ ] **Step 2: Run focused regression suite**

```bash
pytest tests/test_dashboard.py tests/test_event.py tests/test_session.py tests/test_mcp.py tests/test_embedded_graph_store.py tests/test_query.py tests/test_embedding.py tests/test_packaging.py tests/test_cli.py -q
```

Expected: PASS or documented Kuzu skip for embedded tests when Kuzu is not
installed locally.

- [ ] **Step 3: Run docs validation**

```bash
python scripts/build-site-docs.py --check
scripts/validate-docs.sh
```

Expected: PASS.

- [ ] **Step 4: Run release check or document why it is deferred**

Preferred:

```bash
scripts/release-check.sh --root .
```

Expected: PASS.

If deferred, include the exact focused checks in the PR body.

- [ ] **Step 5: Open PR**

```bash
git push -u origin hardening/codebase-review-2026-06-09
gh pr create \
  --title "Harden dashboard and event-log hot paths" \
  --body "$(cat <<'EOF'
## Summary
- fixes dashboard XSS/CSRF posture and local security defaults
- removes full-log append/replay hot-path costs where safe
- preserves embedded read caches for event-only lifecycle projection
- starts benchmark/runtime package boundary cleanup

## Test Plan
- [ ] ruff check src/ tests/
- [ ] mypy src/zaxy
- [ ] pytest tests/test_dashboard.py tests/test_event.py tests/test_session.py tests/test_mcp.py tests/test_embedded_graph_store.py tests/test_query.py tests/test_embedding.py tests/test_packaging.py tests/test_cli.py -q
- [ ] python scripts/build-site-docs.py --check
- [ ] scripts/validate-docs.sh
- [ ] scripts/release-check.sh --root .
EOF
)"
```

---

## Follow-Up Backlog After This Plan

These are valid review findings but should not block the first remediation PR:

- Native or numpy-backed embedded vector index.
- pgGraph `vector(N)` + HNSW index.
- Dashboard persistent backend lifetime with locking.
- Decomposition of `retrieval_plan.py`, `synthesis.py`, `__main__.py`, and
  `mcp_server.py`.
- Direct tests for `coordination_git`, `hooks`, and other indirectly covered
  modules.
- Ruff `D` docstring policy decision and pre-commit config.
- Full 2.0 docs/API inventory sweep.

Each follow-up should get its own plan because these are structural changes with
different risk profiles.
