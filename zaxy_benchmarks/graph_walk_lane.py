"""Internal PPR graph-walk lane: entity-bridge lift and single-hop non-regression.

This lane measures the 2.2-beta.1 bounded personalized-PageRank retrieval
stage (``zaxy.query.QueryRouter`` with ``graph_walk_enabled``) on a real
seeded embedded fabric:

- **Bridge lift**: deterministic multi-hop clusters where the correct memory
  (the *bridge target*) is connected to the query-matched anchor entity only
  through one or two intermediate hops, while a *distractor* memory carries an
  exactly tied lexical signal (same matched query terms, same token counts)
  but no graph path to the anchor. Because the lexical scores tie exactly,
  the plain arm's target/distractor *score margin* is zero — plain ranking
  cannot distinguish the bridged answer from the distractor and resolves the
  tie by arbitrary storage order. The walk arm must produce a strictly
  positive margin toward the bridged target; that margin is attributable to
  graph evidence alone.
- **Single-hop non-regression**: direct-relevance queries whose target memory
  is lexically dominant. The walk arm must keep every direct hit in the top-k
  at a rank no worse than the plain arm.
- **Walk determinism and cache**: the full query set is executed twice on the
  walk arm; ranked content must be identical and the second pass must be
  served from the router's walk cache (hits observable via the router's
  ``graph_walk_cache_hits``/``graph_walk_cache_misses`` diagnostics).

The two arms are identical ``QueryRouter`` configurations over the *same*
projected store (balanced scoring, lexical reranker — the router wiring both
the ``local_fast`` and ``cognitive`` profiles share) differing only in
``graph_walk_enabled``. This isolates the roadmap PPR lane comparison
("embedded vs. PPR-blended ranking") at the exact seam the cognitive profile
arms; full cognitive checkouts additionally blend wall-clock-dependent
salience decay, which would break two-run byte-determinism and confound the
walk measurement. Headline bridge queries run without a query embedding so
the target/distractor lexical tie is exact; an ``embedding_variant`` block
reports the same fractions with the hash-vector lane enabled, where
token-hash noise on near-ties is uncontrolled.

Every result is labeled ``"validation": "internal"``: synthetic fixture,
hash embeddings, project-defined metrics — not a public benchmark claim.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from zaxy.core import MemoryFabric
    from zaxy.embedded_graph_store import EmbeddedGraphStore
    from zaxy.embedding import HashEmbeddingProvider
    from zaxy.query import ContextChunk, QueryRouter

GRAPH_WALK_LANE_VERSION = "graph-walk-lane-v1"
VALIDATION_LABEL = "internal"
DEFAULT_TOP_K = 5

_LANE_SESSION_ID = "walk-lane"
_QUERY_LIMIT = 10
_LANE_INFERENCE_METHOD = "lane_fixture"
#: Parallel relation types between the last intermediate and the bridge
#: target. Parallel RELATES rows proportionally weight the walk transition
#: (documented in AdjacencySnapshot), concentrating mass on the target.
_TARGET_RELATION_TYPES = ("resolved_by", "mitigated_by", "documented_by")

_Entity = tuple[str, str, str]  # (name, entity_type, summary)


@dataclass(frozen=True)
class BridgeCase:
    """One entity-bridge probe: anchor -> intermediates -> target vs distractor.

    ``target`` and ``distractor`` summaries are token-balanced: identical
    matched query terms and identical token counts, so BM25 and the lexical
    reranker score them exactly equal. The distractor lives on an isolated
    two-node island (distractor + pad) with no path to the anchor.
    """

    name: str
    query: str
    anchor: _Entity
    intermediates: tuple[_Entity, ...]
    target: _Entity
    distractor: _Entity
    pad: _Entity


@dataclass(frozen=True)
class DirectCase:
    """One single-hop probe: the query matches the target memory directly."""

    name: str
    query: str
    target: _Entity
    satellite: _Entity


BRIDGE_CASES: tuple[BridgeCase, ...] = (
    BridgeCase(
        name="telemetry-outage",
        query="Which fix did we adopt for the telemetry pipeline outage?",
        anchor=(
            "telemetry pipeline outage",
            "incident",
            "Telemetry pipeline outage under active investigation.",
        ),
        intermediates=(
            (
                "collector saturation finding",
                "finding",
                "Collector saturation surfaced while investigating.",
            ),
        ),
        target=(
            "remedy ledger alpha",
            "decision",
            "Adopt the buffered shipper fix for ingestion.",
        ),
        distractor=(
            "remedy ledger beta",
            "decision",
            "Adopt the replay backlog fix for archiving.",
        ),
        pad=(
            "archive housekeeping memo",
            "note",
            "Quarterly housekeeping reminder for old folders.",
        ),
    ),
    BridgeCase(
        name="billing-reconciliation",
        query="Which patch did we select for the billing reconciliation failure?",
        anchor=(
            "billing reconciliation failure",
            "incident",
            "Billing reconciliation failure flagged by finance.",
        ),
        intermediates=(
            (
                "invoice drift finding",
                "finding",
                "Invoice drift surfaced across monthly summaries.",
            ),
        ),
        target=(
            "remedy ledger gamma",
            "decision",
            "Select the idempotent posting patch for invoices.",
        ),
        distractor=(
            "remedy ledger delta",
            "decision",
            "Select the nightly export patch for receipts.",
        ),
        pad=(
            "vendor onboarding memo",
            "note",
            "Welcome packet checklist for new vendors.",
        ),
    ),
    BridgeCase(
        name="login-regression",
        query="Which change did we approve for the login session regression?",
        anchor=(
            "login session regression",
            "incident",
            "Login session regression reported after rollout.",
        ),
        intermediates=(
            (
                "token expiry finding",
                "finding",
                "Token expiry mismatch traced in the gateway.",
            ),
        ),
        target=(
            "remedy ledger epsilon",
            "decision",
            "Approve the sliding window change for cookies.",
        ),
        distractor=(
            "remedy ledger zeta",
            "decision",
            "Approve the password rotation change for vaults.",
        ),
        pad=(
            "desk allocation memo",
            "note",
            "Seating allocation reminder for the second floor.",
        ),
    ),
    BridgeCase(
        name="object-storage",
        query="Which approach did we choose for the object storage incident?",
        anchor=(
            "object storage incident",
            "incident",
            "Object storage incident raised by the platform.",
        ),
        intermediates=(
            (
                "compaction stall finding",
                "finding",
                "Compaction stalls observed in the write path.",
            ),
            (
                "bucket pressure review",
                "review",
                "Bucket pressure review escalated for capacity.",
            ),
        ),
        target=(
            "remedy ledger eta",
            "decision",
            "Choose the tiered compaction approach for buckets.",
        ),
        distractor=(
            "remedy ledger theta",
            "decision",
            "Choose the cold archive approach for snapshots.",
        ),
        pad=(
            "team offsite memo",
            "note",
            "Agenda outline draft for the spring offsite.",
        ),
    ),
    BridgeCase(
        name="search-relevance",
        query="Which upgrade did we endorse for the search relevance complaint?",
        anchor=(
            "search relevance complaint",
            "incident",
            "Search relevance complaint escalated by support.",
        ),
        intermediates=(
            (
                "synonym gap finding",
                "finding",
                "Synonym coverage gap traced in tokenizers.",
            ),
            (
                "analyzer drift review",
                "review",
                "Analyzer drift review completed by the crew.",
            ),
        ),
        target=(
            "remedy ledger iota",
            "decision",
            "Endorse the stemming dictionary upgrade for recall.",
        ),
        distractor=(
            "remedy ledger kappa",
            "decision",
            "Endorse the index rebuild upgrade for speed.",
        ),
        pad=(
            "library catalog memo",
            "note",
            "Shelf reorganization reminder for the lobby.",
        ),
    ),
    BridgeCase(
        name="canary-alarm",
        query="Which adjustment did we confirm for the canary rollout alarm?",
        anchor=(
            "canary rollout alarm",
            "incident",
            "Canary rollout alarm paged the release crew.",
        ),
        intermediates=(
            (
                "threshold drift finding",
                "finding",
                "Threshold drift spotted during release review.",
            ),
        ),
        target=(
            "remedy ledger lambda",
            "decision",
            "Confirm the gradual ramp adjustment for paging.",
        ),
        distractor=(
            "remedy ledger mu",
            "decision",
            "Confirm the silent window adjustment for nights.",
        ),
        pad=(
            "cafeteria menu memo",
            "note",
            "Seasonal menu rotation reminder for the kitchen.",
        ),
    ),
)

DIRECT_CASES: tuple[DirectCase, ...] = (
    DirectCase(
        name="rollback-playbook",
        query="Where is the rollback playbook for staged deploys?",
        target=(
            "rollback playbook",
            "runbook",
            "Rollback playbook validated for staged deploys.",
        ),
        satellite=(
            "release checklist appendix",
            "note",
            "Checklist appendix for release operations.",
        ),
    ),
    DirectCase(
        name="migration-freeze",
        query="What did we decide about the database migration freeze?",
        target=(
            "database migration freeze",
            "policy",
            "Database migration freeze decided for quarter end.",
        ),
        satellite=(
            "calendar coordination memo",
            "note",
            "Quarter end coordination reminder for leads.",
        ),
    ),
    DirectCase(
        name="oncall-ladder",
        query="Show the oncall escalation ladder for overnight incidents.",
        target=(
            "oncall escalation ladder",
            "runbook",
            "Oncall escalation ladder covering overnight incidents.",
        ),
        satellite=(
            "rotation roster appendix",
            "note",
            "Rotation roster appendix for the support desk.",
        ),
    ),
    DirectCase(
        name="retention-policy",
        query="What is the data retention policy for audit logs?",
        target=(
            "data retention policy",
            "policy",
            "Data retention policy covering audit logs.",
        ),
        satellite=(
            "compliance binder appendix",
            "note",
            "Compliance binder appendix for the archive room.",
        ),
    ),
)


def _entity_payload(entity: _Entity) -> dict[str, str]:
    name, entity_type, summary = entity
    return {"name": name, "entity_type": entity_type, "summary": summary}


def _edge_event_payload(
    source: _Entity,
    target: _Entity,
    relation_type: str,
) -> dict[str, Any]:
    return {
        "source": _entity_payload(source),
        "target": _entity_payload(target),
        "relation_type": relation_type,
        "inference_method": _LANE_INFERENCE_METHOD,
        "confidence": 0.9,
        "evidence": {"fixture": GRAPH_WALK_LANE_VERSION},
    }


def lane_seed_event_payloads() -> list[dict[str, Any]]:
    """Return the ordered ``inference.edge.generated`` payloads for the fixture.

    Per bridge case the order is: distractor island first, then the
    anchor-to-intermediate chain, then the parallel intermediate-to-target
    edges. Direct cases attach each target to one neutral satellite. The
    margin metrics below do not depend on which way the plain arm's
    zero-margin tie happens to resolve.
    """
    payloads: list[dict[str, Any]] = []
    for case in BRIDGE_CASES:
        payloads.append(_edge_event_payload(case.distractor, case.pad, "filed_with"))
        chain = (case.anchor, *case.intermediates)
        for source, target in zip(chain, chain[1:], strict=False):
            payloads.append(_edge_event_payload(source, target, "triggered_review"))
        for relation_type in _TARGET_RELATION_TYPES:
            payloads.append(_edge_event_payload(case.intermediates[-1], case.target, relation_type))
    for direct in DIRECT_CASES:
        payloads.append(_edge_event_payload(direct.target, direct.satellite, "attached_to"))
    return payloads


async def _seed_lane_fabric(workdir: Path) -> tuple[MemoryFabric, HashEmbeddingProvider]:
    """Build a real embedded MemoryFabric in ``workdir`` and seed the fixture.

    Mirrors the house lane pattern: embedded projection backend, deterministic
    hash embedding provider, real append path (extraction + projection).
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
    provider = HashEmbeddingProvider(dimension=fabric.settings.embedding_dimension)
    fabric.embedding_provider = provider
    await fabric.connect()
    for payload in lane_seed_event_payloads():
        await fabric.append(
            "inference.edge.generated",
            actor="lane",
            payload=payload,
            thread=_LANE_SESSION_ID,
            session_id=_LANE_SESSION_ID,
        )
    return fabric, provider


def _build_router(store: Any, *, graph_walk_enabled: bool) -> QueryRouter:
    """Build one measurement arm; both arms differ only in the walk flag."""
    from zaxy.query import LexicalReranker, QueryRouter

    return QueryRouter(
        store,
        default_limit=_QUERY_LIMIT,
        session_id=_LANE_SESSION_ID,
        scoring_profile="balanced",
        reranker=LexicalReranker(),
        graph_walk_enabled=graph_walk_enabled,
    )


def _ranked(chunks: list[ContextChunk]) -> list[dict[str, Any]]:
    return [
        {
            "entity_name": chunk.entity_name,
            "entity_type": chunk.entity_type,
            "score": chunk.score,
        }
        for chunk in chunks
    ]


def _rank_of(ranked: list[dict[str, Any]], entity_name: str) -> int | None:
    for position, item in enumerate(ranked, start=1):
        if item["entity_name"] == entity_name:
            return position
    return None


def _score_of(ranked: list[dict[str, Any]], entity_name: str) -> float | None:
    for item in ranked:
        if item["entity_name"] == entity_name:
            return float(item["score"])
    return None


def _arm_metrics(
    ranked: list[dict[str, Any]],
    *,
    target_name: str,
    distractor_name: str,
    top_k: int,
) -> dict[str, Any]:
    target_rank = _rank_of(ranked, target_name)
    distractor_rank = _rank_of(ranked, distractor_name)
    target_score = _score_of(ranked, target_name)
    distractor_score = _score_of(ranked, distractor_name)
    margin = (
        round(target_score - distractor_score, 4)
        if target_score is not None and distractor_score is not None
        else None
    )
    return {
        "target_rank": target_rank,
        "distractor_rank": distractor_rank,
        "target_score": target_score,
        "distractor_score": distractor_score,
        # Positive margin means ranking *evidence* (not tie order) puts the
        # bridged target above the distractor; zero means indistinguishable.
        "target_distractor_margin": margin,
        "target_in_top_k": target_rank is not None and target_rank <= top_k,
        "target_above_distractor": (
            target_rank is not None
            and (distractor_rank is None or target_rank < distractor_rank)
        ),
    }


def _fraction(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


async def _query_ranked(
    router: QueryRouter,
    query: str,
    *,
    embedding: list[float] | None,
) -> list[dict[str, Any]]:
    chunks = await router.query(
        query,
        limit=_QUERY_LIMIT,
        embedding=embedding,
        session_id=_LANE_SESSION_ID,
    )
    return _ranked(chunks)


async def _bridge_results(
    plain_router: QueryRouter,
    walk_router: QueryRouter,
    *,
    top_k: int,
    embed: Any | None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case in BRIDGE_CASES:
        embedding = embed(case.query) if embed is not None else None
        plain_ranked = await _query_ranked(plain_router, case.query, embedding=embedding)
        walk_ranked = await _query_ranked(walk_router, case.query, embedding=embedding)
        plain = _arm_metrics(
            plain_ranked, target_name=case.target[0], distractor_name=case.distractor[0], top_k=top_k
        )
        walk = _arm_metrics(
            walk_ranked, target_name=case.target[0], distractor_name=case.distractor[0], top_k=top_k
        )
        rank_improved = (walk["target_in_top_k"] and not plain["target_in_top_k"]) or (
            walk["target_rank"] is not None
            and (plain["target_rank"] is None or walk["target_rank"] < plain["target_rank"])
        )
        margin_gained = (
            walk["target_distractor_margin"] is not None
            and plain["target_distractor_margin"] is not None
            and walk["target_distractor_margin"] > plain["target_distractor_margin"]
        )
        cases.append(
            {
                "case": case.name,
                "query": case.query,
                "intermediate_hops": len(case.intermediates),
                "plain": plain,
                "walk": walk,
                "rank_improved": rank_improved,
                "margin_gained": margin_gained,
            }
        )
    total = len(cases)
    fractions = {
        "target_in_top_k_plain": _fraction(
            sum(1 for case in cases if case["plain"]["target_in_top_k"]), total
        ),
        "target_in_top_k_walk": _fraction(
            sum(1 for case in cases if case["walk"]["target_in_top_k"]), total
        ),
        "target_above_distractor_plain": _fraction(
            sum(1 for case in cases if case["plain"]["target_above_distractor"]), total
        ),
        "target_above_distractor_walk": _fraction(
            sum(1 for case in cases if case["walk"]["target_above_distractor"]), total
        ),
        # Evidence-based separation: margin strictly positive (the arm ranks
        # the bridged target above the distractor for a scored reason).
        "positive_margin_plain": _fraction(
            sum(
                1
                for case in cases
                if case["plain"]["target_distractor_margin"] is not None
                and case["plain"]["target_distractor_margin"] > 0.0
            ),
            total,
        ),
        "positive_margin_walk": _fraction(
            sum(
                1
                for case in cases
                if case["walk"]["target_distractor_margin"] is not None
                and case["walk"]["target_distractor_margin"] > 0.0
            ),
            total,
        ),
        "margin_gained_fraction": _fraction(
            sum(1 for case in cases if case["margin_gained"]), total
        ),
        "rank_improved_fraction": _fraction(
            sum(1 for case in cases if case["rank_improved"]), total
        ),
    }
    return {"cases": cases, "fractions": fractions}


async def _direct_results(
    plain_router: QueryRouter,
    walk_router: QueryRouter,
    *,
    top_k: int,
    embed: Any | None,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    divergences: list[dict[str, Any]] = []
    for case in DIRECT_CASES:
        embedding = embed(case.query) if embed is not None else None
        plain_ranked = await _query_ranked(plain_router, case.query, embedding=embedding)
        walk_ranked = await _query_ranked(walk_router, case.query, embedding=embedding)
        plain_rank = _rank_of(plain_ranked, case.target[0])
        walk_rank = _rank_of(walk_ranked, case.target[0])
        retained = walk_rank is not None and walk_rank <= top_k
        rank_not_worse = (
            walk_rank is not None and (plain_rank is None or walk_rank <= plain_rank)
        )
        entry = {
            "case": case.name,
            "query": case.query,
            "plain_target_rank": plain_rank,
            "walk_target_rank": walk_rank,
            "retained_in_top_k": retained,
            "rank_not_worse": rank_not_worse,
        }
        cases.append(entry)
        if plain_rank != walk_rank:
            divergences.append(
                {"case": case.name, "plain_target_rank": plain_rank, "walk_target_rank": walk_rank}
            )
    non_regression = all(case["retained_in_top_k"] and case["rank_not_worse"] for case in cases)
    return {"cases": cases, "divergences": divergences, "non_regression": non_regression}


def run_graph_walk_lane(
    workdir: Path,
    *,
    top_k: int = DEFAULT_TOP_K,
    include_embedding_variant: bool = True,
) -> dict[str, Any]:
    """Run the PPR graph-walk lane in ``workdir`` and return one labeled report.

    The report is fully deterministic: ranks and rounded scores derive from
    BM25/lexical/walk arithmetic over the seeded projection, with no wall-clock
    or network dependence. Two runs in different directories are equal.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    return asyncio.run(
        _run_graph_walk_lane_async(
            workdir, top_k=top_k, include_embedding_variant=include_embedding_variant
        )
    )


async def _run_graph_walk_lane_async(
    workdir: Path,
    *,
    top_k: int,
    include_embedding_variant: bool,
) -> dict[str, Any]:
    fabric, provider = await _seed_lane_fabric(workdir)
    try:
        # The lane builds the embedded backend explicitly, so the adjacency
        # provider surface is present by construction.
        store = cast("EmbeddedGraphStore", fabric.graph)
        plain_router = _build_router(store, graph_walk_enabled=False)
        walk_router = _build_router(store, graph_walk_enabled=True)
        snapshot = await store.fetch_adjacency(session_id=_LANE_SESSION_ID)

        bridge = await _bridge_results(plain_router, walk_router, top_k=top_k, embed=None)
        single_hop = await _direct_results(plain_router, walk_router, top_k=top_k, embed=None)

        first_pass_hits = walk_router.graph_walk_cache_hits
        first_pass_misses = walk_router.graph_walk_cache_misses

        # Second identical pass over every query on the walk arm: content must
        # be identical and walks must replay from the signature-keyed cache.
        first_pass_ranked: list[list[dict[str, Any]]] = []
        second_pass_ranked: list[list[dict[str, Any]]] = []
        all_queries = [case.query for case in BRIDGE_CASES] + [case.query for case in DIRECT_CASES]
        for query in all_queries:
            first_pass_ranked.append(await _query_ranked(walk_router, query, embedding=None))
        hits_before_second = walk_router.graph_walk_cache_hits
        for query in all_queries:
            second_pass_ranked.append(await _query_ranked(walk_router, query, embedding=None))
        second_pass_hits = walk_router.graph_walk_cache_hits - hits_before_second

        embedding_variant: dict[str, Any] | None = None
        if include_embedding_variant:
            variant_bridge = await _bridge_results(
                plain_router, walk_router, top_k=top_k, embed=provider.embed
            )
            variant_direct = await _direct_results(
                plain_router, walk_router, top_k=top_k, embed=provider.embed
            )
            embedding_variant = {
                "note": (
                    "Same probes with the hash-vector lane enabled (query embedding "
                    "passed). Hash-token noise between near-tie pairs is uncontrolled "
                    "here, so these fractions are reported for context only and do "
                    "not gate the exit criteria."
                ),
                "bridge_fractions": variant_bridge["fractions"],
                "single_hop_non_regression": variant_direct["non_regression"],
                "single_hop_divergences": variant_direct["divergences"],
            }
    finally:
        await fabric.close()

    # Multi-hop lift: the walk arm must separate every lexically tied
    # bridge pair by a strictly positive margin (graph evidence decides),
    # gain margin over plain on every case, and never lose top-k coverage.
    multi_hop_lift = (
        bridge["fractions"]["positive_margin_walk"]
        > bridge["fractions"]["positive_margin_plain"]
        and bridge["fractions"]["margin_gained_fraction"] > 0.0
        and bridge["fractions"]["target_in_top_k_walk"]
        >= bridge["fractions"]["target_in_top_k_plain"]
    )
    exit_criteria = {
        "multi_hop_lift": multi_hop_lift,
        "single_hop_non_regression": single_hop["non_regression"],
        "status": "pass" if multi_hop_lift and single_hop["non_regression"] else "fail",
    }
    result: dict[str, Any] = {
        "lane": "graph_walk",
        "version": GRAPH_WALK_LANE_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "Plain vs graph-walk QueryRouter ranking over one seeded embedded "
            "fabric; bridge targets and distractors are exactly tied lexically "
            "so rank flips are attributable to the bounded personalized-PageRank "
            "stage. Synthetic fixture, hash embeddings, no LLM scoring."
        ),
        "fixture": {
            "session_id": _LANE_SESSION_ID,
            "seed_event_count": len(lane_seed_event_payloads()),
            "bridge_case_count": len(BRIDGE_CASES),
            "direct_case_count": len(DIRECT_CASES),
            "top_k": top_k,
            "query_limit": _QUERY_LIMIT,
            "adjacency": {
                "node_count": snapshot.node_count,
                "edge_count": snapshot.edge_count,
                "signature": snapshot.signature,
            },
        },
        "bridge": bridge,
        "single_hop": single_hop,
        "determinism": {
            "repeat_pass_identical": first_pass_ranked == second_pass_ranked,
            "walk_cache": {
                "first_pass_hits": first_pass_hits,
                "first_pass_misses": first_pass_misses,
                "second_pass_hits": second_pass_hits,
                "second_pass_served_from_cache": second_pass_hits == len(all_queries),
            },
        },
        "exit_criteria": exit_criteria,
    }
    if embedding_variant is not None:
        result["embedding_variant"] = embedding_variant
    return result
