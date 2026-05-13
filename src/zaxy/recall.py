"""Internal recall candidate sets for model-facing memory checkout."""

from __future__ import annotations

from dataclasses import dataclass

from zaxy.context import Context
from zaxy.evidence import evidence_source_id


@dataclass(frozen=True)
class RecallCandidate:
    """A retrieved memory candidate before checkout projection."""

    context: Context
    lane: str
    rank: int
    evidence_bearing: bool
    source_id: str | None


@dataclass(frozen=True)
class RecallCandidateSet:
    """An internal high-recall pool kept separate from prompt budgets."""

    candidates: list[RecallCandidate]
    budget: int

    def contexts(self) -> list[Context]:
        """Return candidate contexts in recall order."""
        return [candidate.context for candidate in self.candidates]

    def to_diagnostics(self) -> dict[str, object]:
        """Return stable checkout diagnostics for the recall layer."""
        lanes: dict[str, int] = {}
        source_ids: set[str] = set()
        evidence_count = 0
        for candidate in self.candidates:
            lanes[candidate.lane] = lanes.get(candidate.lane, 0) + 1
            if candidate.evidence_bearing:
                evidence_count += 1
                if candidate.source_id:
                    source_ids.add(candidate.source_id)
        return {
            "candidate_count": len(self.candidates),
            "evidence_count": evidence_count,
            "source_group_count": len(source_ids),
            "budget": self.budget,
            "lanes": lanes,
        }


def build_recall_candidate_set(
    contexts: list[Context],
    *,
    budget: int,
) -> RecallCandidateSet:
    """Build a source-first internal recall set from retrieved contexts."""
    deduped: list[tuple[int, Context]] = []
    seen: set[str] = set()
    for index, context in enumerate(contexts):
        key = _context_key(context)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((index, context))

    ranked_contexts = sorted(
        deduped,
        key=lambda item: _recall_sort_key(item[1], item[0]),
    )[: max(0, budget)]
    candidates = [
        RecallCandidate(
            context=context,
            lane=_context_lane(context),
            rank=rank,
            evidence_bearing=_context_citation(context) is not None,
            source_id=_context_source_id(context),
        )
        for rank, (_index, context) in enumerate(ranked_contexts, start=1)
    ]
    return RecallCandidateSet(candidates=candidates, budget=max(0, budget))


def empty_recall_candidate_set(*, budget: int = 0) -> RecallCandidateSet:
    """Return an empty recall set for default dataclass construction."""
    return RecallCandidateSet(candidates=[], budget=budget)


def _recall_sort_key(context: Context, index: int) -> tuple[int, int, float, int]:
    citation_rank = 0 if _context_citation(context) else 1
    return (citation_rank, _lane_priority(_context_lane(context)), -context.score, index)


def _context_key(context: Context) -> str:
    citation = _context_citation(context)
    if citation is not None:
        return f"citation:{citation}"
    return f"content:{' '.join(context.content.split()).casefold()}"


def _context_citation(context: Context) -> str | None:
    metadata = context.metadata or {}
    citation = metadata.get("citation")
    return citation if isinstance(citation, str) and citation else None


def _context_lane(context: Context) -> str:
    metadata = context.metadata or {}
    lane = metadata.get("assembly_lane")
    if isinstance(lane, str) and lane:
        return lane
    if context.source in {"verbatim", "packet_memory", "projection", "eventloom"}:
        return context.source
    return "graph"


def _context_source_id(context: Context) -> str | None:
    citation = _context_citation(context)
    if citation is None:
        return None
    return evidence_source_id(
        {
            "content": context.content,
            "source": context.source,
            "citation": citation,
            "source_lane": _context_lane(context),
            "score": context.score,
        }
    )


def _lane_priority(lane: str) -> int:
    priorities = {
        "verbatim": 0,
        "eventloom": 1,
        "projection": 2,
        "graph": 3,
        "packet_memory": 4,
    }
    return priorities.get(lane, 5)
