"""Prompt context primitives and source-aware assembly policy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from zaxy.retrieval_plan import build_evidence_plan, source_lane_candidate_limit
from zaxy.token_budget import BudgetSection, PackResult, pack_sections

TIER_CONSOLIDATED = "consolidated"
TIER_SESSION = "session"
TIER_VOLATILE = "volatile"
STABILITY_TIER_ORDER: tuple[str, ...] = (TIER_CONSOLIDATED, TIER_SESSION, TIER_VOLATILE)
_STABILITY_TIER_RANK = {tier: rank for rank, tier in enumerate(STABILITY_TIER_ORDER)}


@dataclass(frozen=True)
class Context:
    """A piece of retrieved context for injection into an agent prompt."""

    content: str
    source: str
    score: float
    valid_from: str | None = None
    valid_to: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ContextAssemblyPolicy:
    """Budget and merge retrieved context from source-aware assembly lanes."""

    verbatim_enabled: bool = True
    verbatim_slots: int = 1
    packet_memory_enabled: bool = True
    packet_memory_slots: int = 1

    def should_query_verbatim(self, *, limit: int) -> bool:
        """Return whether assembly should read the verbatim source lane."""
        return self.verbatim_enabled and self.verbatim_slots > 0 and limit > 0

    def verbatim_candidate_limit(self, *, query: str | None, limit: int) -> int:
        """Return the source candidate budget needed before final assembly."""
        if not self.should_query_verbatim(limit=limit):
            return 0
        if query is None:
            return limit
        return source_lane_candidate_limit(query, limit=limit)

    def describe(self) -> dict[str, bool | int]:
        """Return a stable client-facing policy description."""
        return {
            "verbatim_enabled": self.verbatim_enabled,
            "verbatim_slots": self.verbatim_slots,
            "packet_memory_enabled": self.packet_memory_enabled,
            "packet_memory_slots": self.packet_memory_slots,
        }

    def with_verbatim_enabled(self, enabled: bool) -> ContextAssemblyPolicy:
        """Return a copy with the verbatim lane enabled or disabled."""
        return ContextAssemblyPolicy(
            verbatim_enabled=enabled,
            verbatim_slots=self.verbatim_slots,
            packet_memory_enabled=self.packet_memory_enabled,
            packet_memory_slots=self.packet_memory_slots,
        )

    def assemble(
        self,
        graph_contexts: list[Context],
        verbatim_contexts: list[Context],
        packet_memory_contexts: list[Context] | None = None,
        *,
        limit: int,
        query: str | None = None,
    ) -> list[Context]:
        """Merge graph and verbatim contexts with a reserved source-recall lane."""
        if limit <= 0:
            return []
        packet_memory_contexts = packet_memory_contexts or []
        desired_verbatim_slots = self.verbatim_slots
        if query is not None:
            plan = build_evidence_plan(query, limit=limit)
            if plan.needs_source_lane:
                desired_verbatim_slots = max(desired_verbatim_slots, plan.source_lane_slots)
        verbatim_limit = (
            min(desired_verbatim_slots, limit, len(verbatim_contexts))
            if self.verbatim_enabled
            else 0
        )
        packet_memory_limit = (
            min(
                self.packet_memory_slots,
                limit - verbatim_limit,
                len(packet_memory_contexts),
            )
            if self.packet_memory_enabled
            else 0
        )
        selected_verbatim = [
            _with_assembly_lane(context, "verbatim")
            for context in _dedupe_contexts(verbatim_contexts)[:verbatim_limit]
        ]
        selected_packet_memory = [
            _with_assembly_lane(context, "packet_memory")
            for context in _dedupe_contexts(packet_memory_contexts)[:packet_memory_limit]
        ]
        selected_keys = {_context_dedupe_key(context) for context in selected_verbatim}
        selected_keys.update(_context_dedupe_key(context) for context in selected_packet_memory)
        graph_limit = limit - len(selected_verbatim) - len(selected_packet_memory)
        selected_graph: list[Context] = []
        for context in _dedupe_contexts(graph_contexts):
            if len(selected_graph) >= graph_limit:
                break
            if _context_dedupe_key(context) in selected_keys:
                continue
            selected_graph.append(_with_assembly_lane(context, "graph"))
        return [*selected_graph, *selected_verbatim, *selected_packet_memory]


def context_counts(contexts: list[Context], *, replay_count: int) -> dict[str, int]:
    """Return client-facing counts for assembled context lanes."""
    counts = {"graph": 0, "verbatim": 0, "packet_memory": 0, "replay": replay_count}
    for context in contexts:
        metadata = context.metadata or {}
        if metadata.get("assembly_lane") == "verbatim" or context.source == "verbatim":
            counts["verbatim"] += 1
        elif metadata.get("assembly_lane") == "packet_memory" or context.source == "packet_memory":
            counts["packet_memory"] += 1
        else:
            counts["graph"] += 1
    return counts


def _dedupe_contexts(contexts: list[Context]) -> list[Context]:
    seen: set[str] = set()
    deduped: list[Context] = []
    for context in contexts:
        key = _context_dedupe_key(context)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
    return deduped


def _context_dedupe_key(context: Context) -> str:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    if isinstance(citation, str) and citation:
        return f"citation:{citation}"
    return f"content:{' '.join(context.content.split()).casefold()}"


def _with_assembly_lane(context: Context, lane: str) -> Context:
    metadata = dict(context.metadata or {})
    metadata["assembly_lane"] = lane
    return Context(
        content=context.content,
        source=context.source,
        score=context.score,
        valid_from=context.valid_from,
        valid_to=context.valid_to,
        metadata=metadata,
    )


@dataclass(frozen=True)
class PromptSection:
    """One stability-tiered, budget-packable section of a rendered prompt."""

    section_id: str
    kind: str
    tier: str
    text: str
    weight: float = 0.5
    mandatory: bool = False


@dataclass(frozen=True)
class PromptSectionSpec:
    """Canonical marker, tier, and budget metadata for one prompt section kind.

    ``marker`` is matched against whole lines (or as a line prefix when
    ``prefix`` is true). Spec tuples are declared in canonical render order so
    splitting a canonically rendered prompt is a forward-only line scan.
    """

    marker: str
    kind: str
    tier: str
    weight: float = 0.5
    mandatory: bool = False
    prefix: bool = False

    def matches(self, line: str) -> bool:
        """Return whether a prompt line starts this section."""
        if self.prefix:
            return line.startswith(self.marker)
        return line == self.marker


ASSEMBLY_PROMPT_SECTION_SPECS: tuple[PromptSectionSpec, ...] = (
    PromptSectionSpec("# Active Memory Working Set", "working_set", TIER_SESSION, weight=0.65),
    PromptSectionSpec("# Recent Events", "recent_events", TIER_SESSION, weight=0.45),
    PromptSectionSpec("# Retrieved Context", "retrieved_context", TIER_VOLATILE, weight=0.7),
    PromptSectionSpec("# Context Warnings", "context_warnings", TIER_VOLATILE, weight=0.75, mandatory=True),
)


def split_prompt_sections(
    prompt: str,
    specs: Sequence[PromptSectionSpec],
    *,
    preamble_kind: str = "preamble",
    preamble_tier: str = TIER_SESSION,
    preamble_mandatory: bool = True,
) -> list[PromptSection]:
    """Split a canonically rendered prompt into tiered sections.

    Markers are matched in spec order with a forward-only scan, so a content
    line can only be mistaken for a section heading when it exactly matches a
    canonical marker that has not appeared yet; even then every line is kept in
    some section, so splitting never loses prompt content. Text before the
    first marker becomes a mandatory preamble section.
    """
    sections: list[PromptSection] = []
    current_spec: PromptSectionSpec | None = None
    current_lines: list[str] = []
    next_spec_index = 0

    def close_block() -> None:
        text = "\n".join(current_lines).strip()
        if not text:
            return
        if current_spec is None:
            sections.append(
                PromptSection(
                    section_id=preamble_kind,
                    kind=preamble_kind,
                    tier=preamble_tier,
                    text=text,
                    mandatory=preamble_mandatory,
                )
            )
            return
        sections.append(
            PromptSection(
                section_id=current_spec.kind,
                kind=current_spec.kind,
                tier=current_spec.tier,
                text=text,
                weight=current_spec.weight,
                mandatory=current_spec.mandatory,
            )
        )

    for line in prompt.splitlines():
        matched: tuple[int, PromptSectionSpec] | None = None
        for spec_index in range(next_spec_index, len(specs)):
            if specs[spec_index].matches(line):
                matched = (spec_index, specs[spec_index])
                break
        if matched is None:
            current_lines.append(line)
            continue
        close_block()
        next_spec_index, current_spec = matched[0] + 1, matched[1]
        current_lines = [line]
    close_block()
    return sections


def order_prompt_sections(sections: Sequence[PromptSection]) -> list[PromptSection]:
    """Return sections in stability-tier order, stable within each tier."""
    return sorted(
        sections,
        key=lambda section: _STABILITY_TIER_RANK.get(section.tier, len(STABILITY_TIER_ORDER)),
    )


def render_prompt_sections(sections: Sequence[PromptSection]) -> str:
    """Render sections into one prompt with canonical blank-line separators."""
    return "\n\n".join(section.text for section in sections if section.text).strip()


def stable_prefix_chars(sections: Sequence[PromptSection]) -> int:
    """Return the rendered length of the leading consolidated-tier prefix."""
    consolidated: list[str] = []
    for section in order_prompt_sections(sections):
        if section.tier != TIER_CONSOLIDATED:
            break
        if section.text:
            consolidated.append(section.text)
    return len("\n\n".join(consolidated))


def pack_prompt_sections(
    sections: Sequence[PromptSection],
    *,
    max_tokens: int,
) -> tuple[list[PromptSection], PackResult]:
    """Pack tier-ordered sections into a token budget.

    Returns the kept sections in stability-tier render order plus the packer
    result for diagnostics. Section ids must be unique, which holds for every
    canonical splitter in this codebase (one section per spec kind).
    """
    ordered = order_prompt_sections(sections)
    result = pack_sections(
        [
            BudgetSection(
                section_id=section.section_id,
                kind=section.kind,
                text=section.text,
                weight=section.weight,
                mandatory=section.mandatory,
            )
            for section in ordered
        ],
        max_tokens,
    )
    kept_ids = {section.section_id for section in result.sections}
    return [section for section in ordered if section.section_id in kept_ids], result


def budget_diagnostics(result: PackResult) -> dict[str, Any]:
    """Return client-facing budget diagnostics for one packing result."""
    return {
        "budget_requested": result.budget_requested,
        "budget_used": result.budget_used,
        "elided": {
            "count": len(result.elided),
            "kinds": sorted({record.kind for record in result.elided}),
            "sections": [record.to_dict() for record in result.elided],
        },
    }


def split_assembly_prompt(prompt: str) -> list[PromptSection]:
    """Split an assembled context prompt into its canonical tiered sections."""
    return split_prompt_sections(prompt, ASSEMBLY_PROMPT_SECTION_SPECS)


def apply_assembly_prompt_budget(prompt: str, *, max_tokens: int) -> tuple[str, dict[str, Any]]:
    """Pack an assembled context prompt into a token budget.

    Returns the packed prompt (tier-ordered, mandatory sections always kept)
    and the budget diagnostics payload reporting what was elided.
    """
    kept, result = pack_prompt_sections(split_assembly_prompt(prompt), max_tokens=max_tokens)
    return render_prompt_sections(kept), budget_diagnostics(result)
