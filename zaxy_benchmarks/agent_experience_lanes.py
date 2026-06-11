"""Internal Phase 1 agent-experience lanes: tool-adoption, budget, and cache.

These lanes measure the Zaxy 2.1 agent-experience surfaces with deterministic,
no-LLM proxy metrics:

- Tool-adoption lane: static MCP listing-surface metrics for the ``core``
  versus ``full`` tool profiles. It does not simulate agent transcripts and
  makes no claim about tool-selection accuracy.
- Budget lane: ``apply_checkout_budget`` graceful-degradation contract over a
  real seeded in-temp-dir fabric across a token-budget sweep.
- Cache lane: stable-prefix byte invariance across repeated checkouts and the
  prefix change after a consolidated-tier append. The cache-hit fraction is
  arithmetic (``stable_prefix_tokens / prompt_tokens``), not a provider
  measurement.

Every result is labeled ``"validation": "internal"`` per the
external-validation policy: these are project-defined engineering lanes, not
public benchmark claims.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from zaxy.checkout import apply_checkout_budget, checkout_stable_prefix_chars
from zaxy.token_budget import estimate_tokens

if TYPE_CHECKING:
    from mcp.types import Tool

    from zaxy.core import MemoryFabric

AGENT_EXPERIENCE_LANES_VERSION = "agent-experience-v1"
VALIDATION_LABEL = "internal"
FRONT_DOOR_TOOL = "memory_checkout"
AGENT_EXPERIENCE_LANE_NAMES: tuple[str, ...] = ("tool-adoption", "budget", "cache")

#: Budget sweep in estimated tokens; ``None`` means unlimited (no packing).
DEFAULT_BUDGET_SWEEP: tuple[int | None, ...] = (256, 512, 1024, 2048, 4096, 8192, None)
DEFAULT_CACHE_REPEATS = 5

_LANE_SESSION_ID = "agent-lane"
_LANE_QUERY = "How do we deploy the payments api and what rollback steps apply?"
_LANE_CHECKOUT_LIMIT = 10

#: Deterministic mixed-content seed: goal, tasks, decisions, a superseded
#: preference, an indexed document, a diagnosed issue, and one validated skill
#: so the consolidated stable prefix carries real procedural memory.
_SEED_EVENTS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    (
        "goal.created",
        "user",
        {
            "title": "Ship the payments API",
            "description": "Deliver the payments API rollout with safe deploys and audited rollbacks.",
        },
    ),
    ("task.proposed", "agent", {"taskId": "t1", "summary": "Prepare canary deploy for payments API"}),
    ("task.claimed", "agent", {"taskId": "t1"}),
    (
        "decision.made",
        "agent",
        {"decision": "Use blue-green deploys for the payments API to keep rollback under five minutes."},
    ),
    ("user.preference_changed", "user", {"userId": "u1", "key": "deploy_window", "value": "weekday-only"}),
    ("user.preference_changed", "user", {"userId": "u1", "key": "deploy_window", "value": "weekday-mornings"}),
    (
        "decision.made",
        "agent",
        {"decision": "Database migrations must ship one release ahead of code that depends on them."},
    ),
    (
        "document.indexed",
        "agent",
        {
            "path": "docs/runbooks/deploy.md",
            "summary": (
                "Deploy runbook: preflight checks, canary thresholds, regional rollout order, "
                "and rollback triggers for the payments API."
            ),
        },
    ),
    ("task.completed", "agent", {"taskId": "t1"}),
    (
        "issue.diagnosed",
        "agent",
        {"summary": "Canary deploy alarm traced to a missing payments API migration on replica shards."},
    ),
    (
        "skill.validated",
        "agent",
        {
            "skill_id": "deploy-runbook",
            "version": "2",
            "summary": "Deploy and rollback runbook steps for the payments api",
            "procedure": ["Run the deploy preflight", "Apply database migrations", "Roll out by region"],
            "applicability": ["deploy", "rollback", "payments api"],
        },
    ),
)

#: Consolidated-tier append used by the cache lane to invalidate the prefix.
_CACHE_APPEND_EVENT: tuple[str, str, dict[str, Any]] = (
    "skill.validated",
    "agent",
    {
        "skill_id": "hotfix-rollback",
        "version": "1",
        "summary": "Hotfix rollback drill for payments api deploy failures",
        "procedure": ["Freeze deploys", "Roll back to last green build", "Verify payments api health"],
        "applicability": ["rollback", "deploy", "payments api"],
    },
)


# ----------------------------------------------------------------------
# Tool-adoption lane
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSurfaceMetrics:
    """Static MCP listing-surface metrics for one tool profile.

    ``front_door_rank`` is the 1-based position of the front-door tool in the
    listed order, or None when it is not listed. ``front_door_reference_count``
    counts the other listed tools whose descriptions mention the front door.
    """

    profile: str
    listed_tool_count: int
    schema_bytes: int
    estimated_schema_tokens: int
    front_door_listed: bool
    front_door_rank: int | None
    front_door_reference_count: int
    front_door_reference_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "profile": self.profile,
            "listed_tool_count": self.listed_tool_count,
            "schema_bytes": self.schema_bytes,
            "estimated_schema_tokens": self.estimated_schema_tokens,
            "front_door_listed": self.front_door_listed,
            "front_door_rank": self.front_door_rank,
            "front_door_reference_count": self.front_door_reference_count,
            "front_door_reference_fraction": self.front_door_reference_fraction,
        }


def _tool_schema_text(tool: Tool) -> str:
    """Serialize one listed tool exactly as its agent-facing surface fields."""
    return json.dumps(
        {
            "name": tool.name,
            "description": tool.description or "",
            "inputSchema": tool.inputSchema,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _listed_tools_for_profile(profile_name: str) -> list[Tool]:
    """Return the Tool table filtered exactly like the MCP server listing."""
    from zaxy.mcp_server import TOOLS
    from zaxy.tool_profiles import resolve_profile

    listed = resolve_profile(profile_name)
    if listed is None:
        return list(TOOLS)
    return [tool for tool in TOOLS if tool.name in listed]


def measure_tool_surface(profile_name: str) -> ToolSurfaceMetrics:
    """Measure the static listing surface one profile presents to agents."""
    tools = _listed_tools_for_profile(profile_name)
    schema_texts = [_tool_schema_text(tool) for tool in tools]
    names = [tool.name for tool in tools]
    front_door_rank = names.index(FRONT_DOOR_TOOL) + 1 if FRONT_DOOR_TOOL in names else None
    referencing = [
        tool.name
        for tool in tools
        if tool.name != FRONT_DOOR_TOOL and FRONT_DOOR_TOOL in (tool.description or "")
    ]
    other_count = sum(1 for name in names if name != FRONT_DOOR_TOOL)
    return ToolSurfaceMetrics(
        profile=profile_name,
        listed_tool_count=len(tools),
        schema_bytes=sum(len(text.encode("utf-8")) for text in schema_texts),
        estimated_schema_tokens=sum(estimate_tokens(text) for text in schema_texts),
        front_door_listed=front_door_rank is not None,
        front_door_rank=front_door_rank,
        front_door_reference_count=len(referencing),
        front_door_reference_fraction=(
            round(len(referencing) / other_count, 4) if other_count else 0.0
        ),
    )


def run_tool_adoption_lane() -> dict[str, Any]:
    """Run the tool-adoption lane: core vs full static listing-surface metrics.

    These are honest surface proxies (listed count, schema bytes/tokens,
    front-door rank and description references). The lane does not simulate
    agent transcripts and makes no tool-selection-accuracy claim.
    """
    core = measure_tool_surface("core")
    full = measure_tool_surface("full")
    return {
        "lane": "tool_adoption",
        "version": AGENT_EXPERIENCE_LANES_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "Static MCP listing-surface metrics per tool profile; "
            "not simulated agent behavior or tool-selection accuracy."
        ),
        "front_door_tool": FRONT_DOOR_TOOL,
        "profiles": {"core": core.to_dict(), "full": full.to_dict()},
        "deltas": {
            "listed_tool_count": full.listed_tool_count - core.listed_tool_count,
            "schema_bytes": full.schema_bytes - core.schema_bytes,
            "estimated_schema_tokens": full.estimated_schema_tokens - core.estimated_schema_tokens,
            "schema_token_reduction_fraction": (
                round(1 - core.estimated_schema_tokens / full.estimated_schema_tokens, 4)
                if full.estimated_schema_tokens
                else 0.0
            ),
        },
    }


# ----------------------------------------------------------------------
# Seeded-fabric helpers shared by the budget and cache lanes
# ----------------------------------------------------------------------


async def _seed_lane_fabric(workdir: Path) -> MemoryFabric:
    """Build a real MemoryFabric in ``workdir`` and seed it via the write path.

    Uses the embedded projection backend and pins the deterministic hash
    embedding provider so lane content never depends on network providers.
    """
    from zaxy.core import MemoryFabric
    from zaxy.embedding import HashEmbeddingProvider

    eventloom_path = workdir / ".eventloom"
    fabric = MemoryFabric(
        eventloom_path=str(eventloom_path),
        projection_backend="embedded",
        embedded_graph_path=eventloom_path / "projections" / "embedded.kuzu",
        tracer_disabled=True,
    )
    fabric.embedding_provider = HashEmbeddingProvider(
        dimension=fabric.settings.embedding_dimension
    )
    await fabric.connect()
    for event_type, actor, payload in _SEED_EVENTS:
        await fabric.append(
            event_type,
            actor=actor,
            payload=copy.deepcopy(payload),
            thread=_LANE_SESSION_ID,
            session_id=_LANE_SESSION_ID,
        )
    return fabric


def _citation_fields_preserved(payload: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    """Return whether budget packing left every citation-bearing field intact."""
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False
    if any(not item.get("citation") for item in evidence):
        return False
    diagnostics = payload.get("diagnostics")
    baseline_diagnostics = baseline.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or not isinstance(baseline_diagnostics, Mapping):
        return False
    return (
        payload.get("evidence") == baseline.get("evidence")
        and payload.get("current_facts") == baseline.get("current_facts")
        and payload.get("provenance") == baseline.get("provenance")
        and diagnostics.get("citation_count") == baseline_diagnostics.get("citation_count")
        and diagnostics.get("current_citation_count")
        == baseline_diagnostics.get("current_citation_count")
    )


def _sorted_budgets(budgets: Sequence[int | None]) -> list[int | None]:
    """Return the sweep sorted ascending with unlimited (None) last."""
    finite = sorted({budget for budget in budgets if budget is not None})
    if any(budget is None for budget in budgets):
        return [*finite, None]
    return list(finite)


# ----------------------------------------------------------------------
# Budget lane
# ----------------------------------------------------------------------


def run_budget_lane(
    workdir: Path,
    *,
    budgets: Sequence[int | None] = DEFAULT_BUDGET_SWEEP,
    query: str = _LANE_QUERY,
    limit: int = _LANE_CHECKOUT_LIMIT,
) -> dict[str, Any]:
    """Run the budget lane against the real checkout path in ``workdir``.

    Seeds a deterministic fabric, performs one real ``memory_checkout``, then
    sweeps ``apply_checkout_budget`` over ``budgets`` and verifies the
    graceful-degradation contract: citation fields survive every budget and
    elisions are monotone non-increasing as the budget grows.
    """
    if not budgets:
        raise ValueError("budgets must include at least one budget")
    return asyncio.run(_run_budget_lane_async(workdir, budgets=budgets, query=query, limit=limit))


async def _run_budget_lane_async(
    workdir: Path,
    *,
    budgets: Sequence[int | None],
    query: str,
    limit: int,
) -> dict[str, Any]:
    fabric = await _seed_lane_fabric(workdir)
    try:
        checkout = await fabric.checkout_memory(query, session_id=_LANE_SESSION_ID, limit=limit)
    finally:
        await fabric.close()
    baseline = checkout.to_dict()
    unbudgeted_tokens = estimate_tokens(baseline["prompt"])

    sweep: list[dict[str, Any]] = []
    elided_count_monotone = True
    elided_kinds_monotone = True
    citations_preserved_everywhere = True
    previous_elided_count: int | None = None
    previous_elided_kinds: set[str] | None = None
    for budget in _sorted_budgets(budgets):
        payload = apply_checkout_budget(copy.deepcopy(baseline), max_tokens=budget)
        diagnostics = payload["diagnostics"]
        elided = diagnostics.get("elided", {"count": 0, "kinds": []})
        elided_count = int(elided["count"])
        elided_kinds = {str(kind) for kind in elided["kinds"]}
        citations_preserved = _citation_fields_preserved(payload, baseline)
        citations_preserved_everywhere &= citations_preserved
        if previous_elided_count is not None and elided_count > previous_elided_count:
            elided_count_monotone = False
        if previous_elided_kinds is not None and not elided_kinds <= previous_elided_kinds:
            elided_kinds_monotone = False
        previous_elided_count = elided_count
        previous_elided_kinds = elided_kinds
        sweep.append(
            {
                "budget_requested": budget,
                "budget_used": diagnostics.get("budget_used"),
                "prompt_estimated_tokens": estimate_tokens(payload["prompt"]),
                "elided_count": elided_count,
                "elided_kinds": sorted(elided_kinds),
                "stable_prefix_chars": diagnostics["stable_prefix_chars"],
                "citation_fields_preserved": citations_preserved,
            }
        )

    contract = {
        "citation_fields_preserved_at_every_budget": citations_preserved_everywhere,
        "elided_count_monotone_non_increasing": elided_count_monotone,
        "elided_kinds_monotone_non_increasing": elided_kinds_monotone,
        "status": (
            "pass"
            if citations_preserved_everywhere and elided_count_monotone and elided_kinds_monotone
            else "fail"
        ),
    }
    return {
        "lane": "budget",
        "version": AGENT_EXPERIENCE_LANES_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "apply_checkout_budget graceful-degradation contract over a real seeded "
            "memory_checkout payload; deterministic token estimates, no LLM scoring."
        ),
        "fixture": {
            "seed_event_count": len(_SEED_EVENTS),
            "session_id": _LANE_SESSION_ID,
            "query": query,
            "checkout_limit": limit,
            "unbudgeted_prompt_estimated_tokens": unbudgeted_tokens,
            "evidence_count": len(baseline["evidence"]),
            "current_fact_count": len(baseline["current_facts"]),
        },
        "sweep": sweep,
        "contract": contract,
    }


# ----------------------------------------------------------------------
# Cache lane
# ----------------------------------------------------------------------


def run_cache_lane(
    workdir: Path,
    *,
    repeats: int = DEFAULT_CACHE_REPEATS,
    query: str = _LANE_QUERY,
    limit: int = _LANE_CHECKOUT_LIMIT,
) -> dict[str, Any]:
    """Run the cache lane against repeated real checkouts in ``workdir``.

    Measures consolidated stable-prefix length, byte-identical prefix reuse
    across ``repeats`` checkouts without appends, and the prefix change after
    a consolidated-tier (validated skill) append. The reported cache-hit
    fraction is estimated arithmetic, not a provider measurement.
    """
    if repeats < 2:
        raise ValueError("repeats must be >= 2 to measure prefix reuse")
    return asyncio.run(_run_cache_lane_async(workdir, repeats=repeats, query=query, limit=limit))


async def _run_cache_lane_async(
    workdir: Path,
    *,
    repeats: int,
    query: str,
    limit: int,
) -> dict[str, Any]:
    fabric = await _seed_lane_fabric(workdir)
    try:
        prompts: list[str] = []
        for _ in range(repeats):
            checkout = await fabric.checkout_memory(query, session_id=_LANE_SESSION_ID, limit=limit)
            prompts.append(checkout.prompt)
        event_type, actor, payload = _CACHE_APPEND_EVENT
        await fabric.append(
            event_type,
            actor=actor,
            payload=copy.deepcopy(payload),
            thread=_LANE_SESSION_ID,
            session_id=_LANE_SESSION_ID,
        )
        after_checkout = await fabric.checkout_memory(query, session_id=_LANE_SESSION_ID, limit=limit)
    finally:
        await fabric.close()

    prefix_chars = checkout_stable_prefix_chars(prompts[0])
    prefix_text = prompts[0][:prefix_chars]
    prefix_identical = all(
        checkout_stable_prefix_chars(prompt) == prefix_chars
        and prompt[:prefix_chars] == prefix_text
        for prompt in prompts
    )
    prompts_identical = all(prompt == prompts[0] for prompt in prompts)
    prompt_tokens = estimate_tokens(prompts[0])
    prefix_tokens = estimate_tokens(prefix_text)

    after_prompt = after_checkout.prompt
    after_prefix_chars = checkout_stable_prefix_chars(after_prompt)
    after_prefix_text = after_prompt[:after_prefix_chars]
    prefix_changed = after_prefix_text != prefix_text

    # The cache contract is prefix-scoped. Full prompts may legitimately
    # differ between repeats because checkout records salience reinforcement
    # events whose replay lands in the volatile tail; that bool stays
    # informational.
    contract_pass = prefix_identical and prefix_changed
    return {
        "lane": "cache",
        "version": AGENT_EXPERIENCE_LANES_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "Stable-prefix byte invariance across repeated real checkouts plus the prefix "
            "change after a consolidated-tier append. The cache-hit fraction is estimated "
            "arithmetic (stable_prefix_tokens / prompt_tokens), not a provider measurement."
        ),
        "fixture": {
            "seed_event_count": len(_SEED_EVENTS),
            "session_id": _LANE_SESSION_ID,
            "query": query,
            "checkout_limit": limit,
            "repeats": repeats,
        },
        "stable_prefix_chars": prefix_chars,
        "prompt_chars": len(prompts[0]),
        "stable_prefix_ratio": round(prefix_chars / len(prompts[0]), 4) if prompts[0] else 0.0,
        "stable_prefix_estimated_tokens": prefix_tokens,
        "prompt_estimated_tokens": prompt_tokens,
        "estimated_provider_cache_hit_fraction": (
            round(prefix_tokens / prompt_tokens, 4) if prompt_tokens else 0.0
        ),
        "prefix_byte_identical_across_repeats": prefix_identical,
        "prompt_byte_identical_across_repeats": prompts_identical,
        "append": {
            "event_type": event_type,
            "stable_prefix_chars": after_prefix_chars,
            "prompt_chars": len(after_prompt),
            "prefix_changed": prefix_changed,
            "prefix_grew": after_prefix_chars > prefix_chars,
        },
        "contract": {
            "prefix_byte_identical_across_repeats": prefix_identical,
            "consolidated_append_changes_prefix": prefix_changed,
            "status": "pass" if contract_pass else "fail",
        },
    }


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def run_agent_experience_lanes(
    workdir: Path,
    *,
    lanes: Sequence[str] = AGENT_EXPERIENCE_LANE_NAMES,
    budgets: Sequence[int | None] = DEFAULT_BUDGET_SWEEP,
    repeats: int = DEFAULT_CACHE_REPEATS,
) -> dict[str, Any]:
    """Run the selected agent-experience lanes and return one labeled report.

    ``workdir`` hosts the seeded lane fabrics; each fabric-backed lane uses its
    own subdirectory so embedded projections never collide.
    """
    selected = tuple(lanes)
    unknown = sorted(set(selected) - set(AGENT_EXPERIENCE_LANE_NAMES))
    if unknown:
        valid = ", ".join(AGENT_EXPERIENCE_LANE_NAMES)
        raise ValueError(f"Unknown agent-experience lane(s): {', '.join(unknown)}. Valid lanes: {valid}.")
    if not selected:
        raise ValueError("at least one agent-experience lane must be selected")
    results: dict[str, Any] = {}
    if "tool-adoption" in selected:
        results["tool_adoption"] = run_tool_adoption_lane()
    if "budget" in selected:
        results["budget"] = run_budget_lane(workdir / "budget", budgets=budgets)
    if "cache" in selected:
        results["cache"] = run_cache_lane(workdir / "cache", repeats=repeats)
    return {
        "version": AGENT_EXPERIENCE_LANES_VERSION,
        "validation": VALIDATION_LABEL,
        "lanes": results,
    }
