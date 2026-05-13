"""Prompt context primitives and source-aware assembly policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zaxy.retrieval_plan import build_evidence_plan, source_lane_candidate_limit


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
