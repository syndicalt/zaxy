# MemoryFabric decomposition — design

Status: approved; phase 1 (ReasoningOps) implemented 2026-07-07 — see `src/zaxy/core/fabric_reasoning.py`
Date: 2026-07-06
Owner: maintainer + Claude Fable (assessment session)

## 1. Problem

`src/zaxy/core/fabric.py` is 4,120 lines, ~3,850 of which are one stateful
god-class, `MemoryFabric` (~90 methods). It is the largest module in the
codebase and the hub every surface depends on (MCP handlers delegate to it
since the #71/#72 parity work). The June first-pass decomposition split the
old `core.py` *module* into a package but left the *class* intact; the class
is where the coupling lives, so line count keeps growing back.

## 2. Evidence

Structural mapping (2026-07-06, verified against source):

| Cluster | Methods (sample) | Lines (approx) |
|---|---|---|
| C1 lifecycle/connection | `__init__` :281, `connect` :422, `close` :437, session warmup | ~350 |
| C2 append/write/evolution | `append` :1175, `append_batch`, gate/outcome/edit/rollback/forget, ingest | ~1,100 |
| C3 query/read | `query` :2273, `retrieve`, `query_page` + cache, verbatim/cue indices | ~700 |
| C4 reasoning/metacognition | causal queries, belief/claim/unknowns/trajectory | ~700 |
| C5 consolidation/checkout/feedback | `checkout_memory` :3417, `assemble_context`, synthesis, `after_turn` | ~750 |
| C6 coordination/fleet/handoff | `coordinate_*` (16), fleet lanes, handoff/cleanup | ~350 |

Coupling: C2 (write path) is the hub — C4/C5/C6 all append through it. All
clusters share `self` state (projection store, session manager, tracer,
query-page cache, verbatim/cue indices, embedding provider).

External patch surface: 10 module-level names in `zaxy.core.fabric` are
`patch()`ed by tests (`build_projection_store` ×15, `QueryRouter` ×9,
`MemoryTracer` ×9, `build_reranker` ×9, `get_metrics` ×7, `SessionManager` ×5,
`build_embedding_provider`, `source_synthesis_bundle_result`, `EventLog`,
`build_memory_checkout`). Any decomposition must keep these names bound in
`zaxy.core.fabric` AND keep moved code resolving them through that namespace.

## 3. Goals / non-goals

Goals: (a) no source file over ~1,500 lines on the fabric path; (b) each
cluster independently readable and testable; (c) zero behavior change per
phase, enforced by the existing parity tests + full suite; (d) the 10-name
patch surface keeps working unchanged throughout.

Non-goals: changing the public `MemoryFabric` API or the MCP delegation
contract; performance work (bounded checkout window etc. is a separate
workstream); touching Eventloom/event semantics.

## 4. Rejected approach: sibling-module mixins

Mechanically splitting the class into `_WriteMixin`/`_QueryMixin`/... sibling
modules was considered and REJECTED: every method still shares undeclared
`self` state, so cohesion does not actually improve; the ~10 patched globals
would have to be resolved through `import zaxy.core.fabric as _f` indirection
in every moved method (easy to get subtly wrong — a bare-name reference in a
moved module silently escapes existing `patch()` targets); and MRO/`__init__`
attribute-ordering becomes an invisible contract between files. It reduces
line counts without reducing coupling — refactor theater that adds debt.

## 5. Design: collaborator extraction, hub-last

Extract clusters into real collaborator objects with *declared* dependencies,
one phase per PR, leaving `MemoryFabric` as the thin coordinator that owns
construction and delegates. Pattern per phase:

```python
# core/fabric_reasoning.py
class ReasoningOps:
    """Metacognition/causal primitives over the fabric's write + read seams."""
    def __init__(self, *, appender: Appender, querier: Querier, tracer: Any) -> None: ...
```

`MemoryFabric.__init__` builds each collaborator against a narrow structural
protocol. Phase 1 refined this: tests monkeypatch fabric *instance* attributes
(`checkout_memory`, `query`, `query_causal_predecessors`) after construction,
so seams must **late-bind** — the collaborator holds the host and resolves
every lookup at call time (one `ReasoningHost` protocol declaring the exact
surface), and intra-cluster calls to *public* primitives route back through
the host so instance patches stay the single dispatch point. Public fabric methods become one-line delegations, so the public API,
the MCP parity contract, and every patch target are untouched (collaborators
receive the patched objects at construction time from the facade's namespace,
preserving `patch("zaxy.core.fabric.X")` interception — construction happens
per-fabric-instance, after patches apply).

Phasing (dependency-ordered, hub last):

1. **C4 ReasoningOps** — most self-contained; depends only on append + query
   seams. Proves the pattern. (~700 lines out)
2. **C6 CoordinationOps** — delegates to CoordinationManager/FleetManager
   already; thin extraction. (~350 lines out)
3. **C5 CheckoutOps** — checkout/assembly/feedback/synthesis; depends on C2/C3
   seams. (~750 lines out)
4. **C3 QueryEngine** — query paths + the page cache + verbatim/cue indices
   move behind one object owning its caches. (~700 lines out)
5. **C2 WriteEngine** — the hub, LAST, once everything else consumes it
   through the narrow seam. (~1,100 lines out)

End state: `fabric.py` ≈ lifecycle + construction + delegation (~600-800
lines), five collaborator modules each under ~1,200.

## 6. Verification per phase

- Full suite + `tests/test_parity.py` (MCP == fabric) green; ruff + mypy
  strict clean.
- The 10 patched names verified by running the specific test files that patch
  them (they fail loudly if a moved call site escapes the patch).
- Coverage ratchet unchanged or better (moves are line-neutral; delegation
  shims are covered by existing tests).
- No `import zaxy.core.fabric` from any collaborator module (cycle guard).

## 7. Open questions

1. Should collaborator seams be `typing.Protocol`s (mypy-checked contracts)
   or plain callables? Protocols preferred; adds ~50 lines of contract per
   phase.
2. `assemble_context`/`checkout_memory` straddle C3/C5 — final boundary to be
   settled in phase 3 with the code in front of us.
3. Does `dashboard.py` (2,717 lines) join this program afterward? Same
   pattern applies; out of scope here.

## 8. Done-when

- [x] Phase 1 (ReasoningOps) merged green — [ ] remaining four phases
- [ ] `fabric.py` under ~800 lines; no fabric-path module over ~1,500
- [ ] All 10 patch targets demonstrably still intercept (test files that patch
      them pass unmodified)
- [ ] `docs/codebase.md` updated with the collaborator map

## References

- 2026-06-16 codebase review finding #3 (first-pass module split, PRs #74-77)
- fable-findings.md (2026-07-06) — gardening backlog item 1
- Structural mapping evidence: session transcript 2026-07-06 (cluster/patch
  tables reproduced in §2)
