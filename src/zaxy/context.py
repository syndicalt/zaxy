"""Prompt context primitives and source-aware assembly policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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

    verbatim_slots: int = 1

    def assemble(
        self,
        graph_contexts: list[Context],
        verbatim_contexts: list[Context],
        *,
        limit: int,
    ) -> list[Context]:
        """Merge graph and verbatim contexts with a reserved source-recall lane."""
        if limit <= 0:
            return []
        verbatim_limit = min(self.verbatim_slots, limit, len(verbatim_contexts))
        selected_verbatim = [
            _with_assembly_lane(context, "verbatim")
            for context in _dedupe_contexts(verbatim_contexts)[:verbatim_limit]
        ]
        selected_keys = {_context_dedupe_key(context) for context in selected_verbatim}
        graph_limit = limit - len(selected_verbatim)
        selected_graph: list[Context] = []
        for context in _dedupe_contexts(graph_contexts):
            if len(selected_graph) >= graph_limit:
                break
            if _context_dedupe_key(context) in selected_keys:
                continue
            selected_graph.append(_with_assembly_lane(context, "graph"))
        return [*selected_graph, *selected_verbatim]


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
