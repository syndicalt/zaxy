"""Two-tier (episodic + consolidated) long-horizon checkout assembly.

For very long ("never-ending thread") sessions, Memory Checkout can split a
session's history into two explicit tiers:

* the EPISODIC (recent) tier — the most recent ``recent_window`` events, kept at
  full detail exactly as the single-tier checkout does today; and
* the CONSOLIDATED (remote) tier — older history represented *not* by raw old
  events but by the session's already-cited I2 consolidation artifacts: the
  ACCEPTED/ACTIVE consolidation candidates whose source events have scrolled out
  of the recent window. Each surfaces as a bounded, cited, non-authoritative
  context item carrying its source-event citation(s).

Nothing is summarized afresh here (no destructive summarization): the remote
tier only replays the consolidation candidates the consolidation pipeline
(``consolidation.py`` / ``consolidation_pipeline.py``) already produced and that
a review event accepted. This module owns the partition logic;
:meth:`zaxy.core.MemoryFabric.assemble_context` builds the plan when the
long-horizon flag is engaged and the session exceeds the window, and
:func:`zaxy.core.checkout_build.build_memory_checkout` renders the
``long_horizon`` diagnostics section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zaxy.compaction import search_compaction_projections, text_tokens
from zaxy.context import Context
from zaxy.learned_context import LearnedContextLoad

#: Source-lane / assembly-lane marker for consolidated remote-tier contexts.
CONSOLIDATED_LANE = "consolidated"

# ---------------------------------------------------------------------------
# I3 long-span relevance scoring.
#
# Before this, a remote-tier item scored by its stored authoring ``confidence``
# with a hardcoded 0.5 fallback. That is how confident the *author* was when the
# candidate was written — it says nothing about whether the item is relevant to
# THIS query at THIS point in a long session, and it is constant for the life of
# the candidate. With two sources now feeding the tier (accepted candidates and
# I2 projection records), an authoring prior also cannot rank them against each
# other: projection records have no confidence at all and would all tie at 0.5.
#
# The replacement is a weighted sum of four terms that are each a deterministic
# function of data already on the record, so the ranking is reconstructible from
# the packet (every term is echoed into ``metadata["relevance_terms"]``). The
# weights sum to 1.0, keeping the score in [0, 1] and comparable across sources.
# ---------------------------------------------------------------------------

#: How well the item's text answers the query. Weighted highest because a
#: long-span tier that ignores the query is just a chronological dump; this is
#: the only term that responds to what was actually asked.
_W_QUERY_OVERLAP = 0.40

#: How much scrolled-out history the item stands in for. The remote tier's job
#: is compression, so an item representing more old events buys more context per
#: token than one citing a single event.
_W_SPAN_COVERAGE = 0.25

#: How close the item's newest source sits to the horizon split. Everything in
#: this tier is already behind the split; among those, history nearer the split
#: is nearer to current work, while deep history is likelier superseded. This is
#: the term that makes the score span-AWARE rather than a generic text match.
_W_HORIZON_PROXIMITY = 0.20

#: The stored authoring confidence, kept but demoted. An accepted candidate's
#: confidence is real information (it survived human review), so it earns a
#: minority vote — it just no longer IS the score, which was the I3 defect.
_W_AUTHORING_PRIOR = 0.15

#: Source-event count at which span coverage saturates. Saturating rather than
#: scaling linearly stops one very large candidate from dominating the tier on
#: size alone.
_SPAN_COVERAGE_SATURATION = 8

#: Neutral authoring prior for records that carry no confidence field (every I2
#: projection record). Neutral, not zero, so they are neither rewarded nor
#: punished on a dimension they do not have.
_NEUTRAL_AUTHORING_PRIOR = 0.5

#: Consolidation review statuses whose candidates may surface in the remote tier.
#: ``accepted`` is the consolidation-candidate review disposition; ``active`` is
#: accepted too for forward-compatibility with promotion-style lifecycles.
SURFACEABLE_REVIEW_STATUSES = frozenset({"accepted", "active"})

_CANDIDATE_CREATED = "consolidation.candidate.created"
_CANDIDATE_REVIEWED = "consolidation.candidate.reviewed"


@dataclass(frozen=True)
class LongHorizonPlan:
    """Resolved episodic/consolidated partition for one checkout."""

    enabled: bool
    recent_window: int
    episodic_count: int
    horizon_split_seq: int | None
    consolidated_contexts: list[Context] = field(default_factory=list)
    learned_context: dict[str, Any] | None = None

    @property
    def consolidated_count(self) -> int:
        return len(self.consolidated_contexts)

    def to_diagnostics(self) -> dict[str, Any]:
        """Return the split summary threaded into ``ContextAssembly.long_horizon``.

        The consolidated count and the per-item list are recomputed by the
        checkout builder from the ranked contexts so the diagnostics reflect what
        actually reached the packet; only the immutable split facts live here.
        """
        diagnostics: dict[str, Any] = {
            "enabled": self.enabled,
            "recent_window": self.recent_window,
            "episodic_count": self.episodic_count,
            "horizon_split_seq": self.horizon_split_seq,
        }
        if self.learned_context is not None:
            # Only present when I2 is engaged, so the default-off diagnostics
            # stanza stays byte-identical to the pre-I2 contract.
            diagnostics["learned_context"] = self.learned_context
        return diagnostics


def build_long_horizon_plan(
    session_events: list[Any],
    *,
    session_id: str,
    recent_window: int,
    budget: int,
    learned_context: LearnedContextLoad | None = None,
    query: str = "",
) -> LongHorizonPlan:
    """Partition a session into episodic + consolidated tiers.

    ``session_events`` is the full (as-of-filtered) replay in seq order. The most
    recent ``recent_window`` events form the episodic tier; older history is
    represented by ACCEPTED/ACTIVE consolidation candidates whose source events
    fall at or before the horizon split. ``budget`` (> 0) bounds the number of
    consolidated items surfaced.

    ``learned_context`` (I2) is an optional *second* source for the remote tier:
    a verified compaction projection whose records cover older history. It is
    consumed only when it loaded cleanly, and never at the expense of an accepted
    consolidation candidate covering the same source events — the candidate wins,
    because it carries a human review decision the projection does not. Passing
    ``None`` (the default) reproduces the pre-I2 single-source behaviour exactly.

    When the session does not exceed the window there is no older region: the
    plan is engaged but the remote tier is empty (graceful — just episodic).
    """
    total = len(session_events)
    learned_diagnostics = None if learned_context is None else learned_context.to_diagnostics()
    if recent_window <= 0 or total <= recent_window:
        return LongHorizonPlan(
            enabled=True,
            recent_window=recent_window,
            episodic_count=total,
            horizon_split_seq=None,
            consolidated_contexts=[],
            learned_context=learned_diagnostics,
        )
    older_events = session_events[: total - recent_window]
    split_seq = _event_seq(older_events[-1])
    query_tokens = text_tokens(query) if query else set()

    consolidated: list[Context] = []
    # Source seqs already represented by a surfaced candidate. A projection
    # record covering one of these would double-surface the same history, which
    # is precisely the duplication the two-tier design exists to prevent.
    covered_source_seqs: set[int] = set()
    for candidate in _replay_consolidation_candidates(session_events, session_id=session_id):
        if candidate["review_status"] not in SURFACEABLE_REVIEW_STATUSES:
            continue
        max_source_seq = candidate["max_source_seq"]
        if max_source_seq is None or max_source_seq > split_seq:
            # No older source events (none cited, or all still inside the recent
            # window): the episodic tier already carries them, so surfacing the
            # candidate here would duplicate rather than consolidate.
            continue
        consolidated.append(
            _consolidated_context(candidate, query_tokens=query_tokens, split_seq=split_seq)
        )
        covered_source_seqs.update(candidate["source_event_seqs"])

    if learned_context is not None and learned_context.projection is not None:
        consolidated.extend(
            _learned_context_contexts(
                learned_context.projection,
                query=query,
                query_tokens=query_tokens,
                split_seq=split_seq,
                covered_source_seqs=covered_source_seqs,
                budget=budget,
            )
        )

    # Rank by the I3 long-span relevance score before the budget bites, so the
    # budget keeps the most relevant remote history rather than the first-replayed.
    consolidated.sort(key=lambda context: -context.score)
    if budget > 0:
        consolidated = consolidated[:budget]
    return LongHorizonPlan(
        enabled=True,
        recent_window=recent_window,
        episodic_count=recent_window,
        horizon_split_seq=split_seq,
        consolidated_contexts=consolidated,
        learned_context=learned_diagnostics,
    )


def _learned_context_contexts(
    projection: Any,
    *,
    query: str,
    query_tokens: set[str],
    split_seq: int,
    covered_source_seqs: set[int],
    budget: int,
) -> list[Context]:
    """Project verified I2 projection records into cited remote-tier contexts.

    Records reach the packet through :func:`search_compaction_projections`, which
    already returns per-record citations, so every surfaced item cites without
    new work here.
    """
    if not query_tokens:
        # The projection search is query-driven; with no query there is no
        # routing signal and the accepted candidates alone carry the tier.
        return []
    # Deliberately NOT the budget: the budget must be applied after span ranking,
    # so the search has to hand back every routable record. Truncating here would
    # let the projection's own text score silently pre-empt the I3 ranking.
    # Records are bounded at write time, so this stays cheap.
    limit = len(projection.records) or 1
    contexts: list[Context] = []
    for result in search_compaction_projections([projection], query, limit=limit):
        record = result.record
        if record.event_seq > split_seq:
            # Still inside the episodic window: the recent tier carries it.
            continue
        if record.event_seq in covered_source_seqs:
            # An accepted consolidation candidate already represents this source
            # event. Prefer the candidate — it carries a human review decision.
            continue
        contexts.append(
            _projection_context(
                result,
                query_tokens=query_tokens,
                split_seq=split_seq,
            )
        )
    return contexts


def _replay_consolidation_candidates(
    session_events: list[Any], *, session_id: str
) -> list[dict[str, Any]]:
    """Replay candidate.created/reviewed events into current candidate state.

    Mirrors :meth:`MemoryFabric.consolidation_status`'s replay: a candidate's
    current ``review_status`` is the latest review disposition (or ``pending``).
    """
    candidates: dict[str, dict[str, Any]] = {}
    for event in session_events:
        event_type = getattr(event, "type", None)
        if event_type == _CANDIDATE_CREATED:
            payload = getattr(event, "payload", None) or {}
            candidate_id = payload.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidates:
                continue
            thread = getattr(event, "thread", None) or session_id
            source_events = _source_event_rows(payload.get("source_events"))
            source_citations = [
                citation
                for src in source_events
                if (citation := _event_citation(thread, src["seq"], src["hash"])) is not None
            ]
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_type": payload.get("candidate_type"),
                "summary": payload.get("summary") or payload.get("title") or candidate_id,
                "confidence": payload.get("confidence"),
                "review_status": str(payload.get("review_status") or "pending"),
                "citation": _event_citation(
                    thread, getattr(event, "seq", None), getattr(event, "hash", None)
                ),
                "source_event_citations": source_citations,
                "source_event_seqs": [src["seq"] for src in source_events],
                "max_source_seq": max((src["seq"] for src in source_events), default=None),
            }
        elif event_type == _CANDIDATE_REVIEWED:
            payload = getattr(event, "payload", None) or {}
            candidate_id = payload.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id in candidates:
                status = payload.get("status")
                if isinstance(status, str) and status:
                    candidates[candidate_id]["review_status"] = status
    return list(candidates.values())


def span_relevance(
    *,
    text: str,
    query_tokens: set[str],
    source_event_count: int,
    max_source_seq: int | None,
    split_seq: int,
    confidence: object = None,
) -> tuple[float, dict[str, Any]]:
    """Score one remote-tier item over a long span, returning the score and its terms.

    The four terms and their weights are documented at the top of this module.
    Every input is data the item already carries, and the returned terms mapping
    is echoed into the context metadata, so the score is reconstructible from the
    packet alone — no hidden state, no model call, deterministic for a given
    (item, query, split) triple.
    """
    item_tokens = text_tokens(text)
    query_overlap = (
        len(query_tokens & item_tokens) / len(query_tokens) if query_tokens else 0.0
    )
    span_coverage = min(1.0, source_event_count / _SPAN_COVERAGE_SATURATION)
    if max_source_seq is None or split_seq <= 0:
        horizon_proximity = 0.0
    else:
        horizon_proximity = min(1.0, max(0.0, max_source_seq / split_seq))
    authoring_prior = (
        float(confidence)
        if isinstance(confidence, int | float) and not isinstance(confidence, bool)
        else _NEUTRAL_AUTHORING_PRIOR
    )
    score = (
        _W_QUERY_OVERLAP * query_overlap
        + _W_SPAN_COVERAGE * span_coverage
        + _W_HORIZON_PROXIMITY * horizon_proximity
        + _W_AUTHORING_PRIOR * authoring_prior
    )
    terms = {
        "query_overlap": round(query_overlap, 6),
        "span_coverage": round(span_coverage, 6),
        "horizon_proximity": round(horizon_proximity, 6),
        "authoring_prior": round(authoring_prior, 6),
        "weights": {
            "query_overlap": _W_QUERY_OVERLAP,
            "span_coverage": _W_SPAN_COVERAGE,
            "horizon_proximity": _W_HORIZON_PROXIMITY,
            "authoring_prior": _W_AUTHORING_PRIOR,
        },
        "span_coverage_saturation": _SPAN_COVERAGE_SATURATION,
        "horizon_split_seq": split_seq,
        "max_source_seq": max_source_seq,
        "source_event_count": source_event_count,
    }
    return round(score, 6), terms


def _consolidated_context(
    candidate: dict[str, Any],
    *,
    query_tokens: set[str],
    split_seq: int,
) -> Context:
    """Project one accepted/active candidate into a cited, non-authoritative Context."""
    confidence = candidate["confidence"]
    source_citations = list(candidate["source_event_citations"])
    score, relevance_terms = span_relevance(
        text=str(candidate["summary"]),
        query_tokens=query_tokens,
        source_event_count=len(candidate["source_event_seqs"]),
        max_source_seq=candidate["max_source_seq"],
        split_seq=split_seq,
        confidence=confidence,
    )
    return Context(
        content=str(candidate["summary"]),
        source=CONSOLIDATED_LANE,
        score=score,
        metadata={
            "relevance_terms": relevance_terms,
            "assembly_lane": CONSOLIDATED_LANE,
            "source_lane": CONSOLIDATED_LANE,
            "tier": CONSOLIDATED_LANE,
            "citation": candidate["citation"],
            "source_event_citations": source_citations,
            "source_event_count": len(source_citations),
            "non_authoritative": True,
            "authority_status": "non_authoritative",
            "review_status": candidate["review_status"],
            "candidate_id": candidate["candidate_id"],
            "candidate_type": candidate["candidate_type"],
            "confidence": confidence,
            "entity_type": "consolidation_candidate",
            "entity_name": candidate["candidate_id"],
        },
    )


def _projection_context(
    result: Any,
    *,
    query_tokens: set[str],
    split_seq: int,
) -> Context:
    """Project one I2 projection-search hit into a cited, non-authoritative Context.

    The synthetic ``candidate_id`` keeps the checkout summarizer
    (:func:`zaxy.core.checkout_build._checkout_long_horizon`) working unchanged —
    it dedupes on that key — while staying stable across rebuilds because both
    the projection id and the source event seq are deterministic.
    """
    record = result.record
    source_citations = [citation for citation in record.citations if citation]
    score, relevance_terms = span_relevance(
        text=record.text,
        query_tokens=query_tokens,
        source_event_count=1,
        max_source_seq=record.event_seq,
        split_seq=split_seq,
        # No authoring confidence exists on a projection record: it was derived
        # mechanically, not asserted by anyone. The neutral prior applies.
        confidence=None,
    )
    return Context(
        content=record.text,
        source=CONSOLIDATED_LANE,
        score=score,
        metadata={
            "relevance_terms": relevance_terms,
            "assembly_lane": CONSOLIDATED_LANE,
            "source_lane": CONSOLIDATED_LANE,
            "tier": CONSOLIDATED_LANE,
            "citation": record.event_ref,
            "source_event_citations": source_citations,
            "source_event_count": len(source_citations),
            "non_authoritative": True,
            "authority_status": "non_authoritative",
            "review_status": None,
            "candidate_id": f"learned-context:{result.projection_id[:12]}:{record.event_seq}",
            "candidate_type": f"projection_{record.kind}",
            "confidence": None,
            "entity_type": "compaction_projection_record",
            "entity_name": record.event_ref,
            "learned_context": True,
            "projection_id": result.projection_id,
            "projection_strategy": result.strategy,
            "projection_search_score": result.score,
        },
    )


def _source_event_rows(value: object) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            seq = item.get("seq")
            event_hash = item.get("hash")
            if (
                isinstance(seq, int)
                and not isinstance(seq, bool)
                and seq > 0
                and isinstance(event_hash, str)
                and event_hash
            ):
                rows.append({"seq": seq, "hash": event_hash})
    return rows


def _event_citation(thread: object, seq: object, event_hash: object) -> str | None:
    if (
        not isinstance(thread, str)
        or not thread
        or not isinstance(seq, int)
        or isinstance(seq, bool)
        or seq <= 0
        or not isinstance(event_hash, str)
        or not event_hash
    ):
        return None
    return f"eventloom://{thread}/events/{seq}#{event_hash[:12]}"


def _event_seq(event: Any) -> int:
    seq = getattr(event, "seq", None)
    return seq if isinstance(seq, int) and not isinstance(seq, bool) else 0
