"""Bounded personalized-PageRank graph walk over projected adjacency snapshots.

This module is the pure-computation core of the 2.2-beta.1 graph-walk
retrieval stage: backends materialize an :class:`AdjacencySnapshot` of one
session's active entity graph (node identity is the projected ``node_key``
string), :func:`personalized_pagerank` runs a bounded numpy power iteration
with restart mass concentrated on query-matched seed entities, and
:func:`blend_walk_scores` folds the resulting walk mass into existing
candidate scores. Backend fetching is declared here as the
:class:`AdjacencyProvider` contract; concrete implementations land with the
backend wave.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, eq=False)
class AdjacencySnapshot:
    """Immutable CSR adjacency over one session's projected entity graph.

    Representation — CSR (compressed sparse row) over directed edges:

    - ``node_ids``: stable, provider-defined ordering of node identities
      (``Entity.node_key`` strings for the embedded backend). Position in
      this tuple is the node index used by ``indptr``/``indices``.
    - ``indptr``: ``int64`` array of length ``len(node_ids) + 1``; the
      out-edges of node ``i`` occupy ``indices[indptr[i]:indptr[i + 1]]``.
    - ``indices``: ``int64`` array with one entry per directed edge, holding
      the target node index. Parallel edges (e.g. one ``RELATES`` edge per
      relation type between the same pair) may appear multiple times and
      proportionally increase transition probability toward that target.

    CSR is the memory-honest representation for this workload: it costs one
    ``int64`` per edge plus ``n + 1`` row pointers — no ``n x n`` dense
    matrix and no per-node Python list/dict overhead — while keeping power
    iteration fully vectorizable (out-degrees are ``np.diff(indptr)``; the
    per-edge source array is a single ``np.repeat``).

    ``signature`` is an opaque caller-supplied cache-invalidation token
    (the log-signature pattern used by the projection caches). This module
    never computes signatures; providers stamp snapshots so cached walks can
    be discarded when the underlying event log advances.
    """

    node_ids: tuple[str, ...]
    indptr: npt.NDArray[np.int64]
    indices: npt.NDArray[np.int64]
    signature: str

    def __post_init__(self) -> None:
        node_count = len(self.node_ids)
        if len(set(self.node_ids)) != node_count:
            raise ValueError("snapshot node_ids must be unique")
        if self.indptr.ndim != 1 or self.indptr.shape[0] != node_count + 1:
            raise ValueError("snapshot indptr must be one-dimensional with len(node_ids) + 1 rows")
        if self.indices.ndim != 1:
            raise ValueError("snapshot indices must be one-dimensional")
        if node_count == 0:
            if self.indices.size != 0 or int(self.indptr[0]) != 0:
                raise ValueError("empty snapshot must carry no edges")
            return
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.indices.shape[0]:
            raise ValueError("snapshot indptr must start at 0 and end at the edge count")
        if bool(np.any(np.diff(self.indptr) < 0)):
            raise ValueError("snapshot indptr must be monotonically non-decreasing")
        if self.indices.size and (
            int(self.indices.min()) < 0 or int(self.indices.max()) >= node_count
        ):
            raise ValueError("snapshot indices must reference positions within node_ids")

    @classmethod
    def from_edges(
        cls,
        node_ids: Sequence[str],
        edges: Iterable[tuple[str, str]],
        *,
        signature: str,
    ) -> AdjacencySnapshot:
        """Build a snapshot from ``(source_id, target_id)`` directed edges.

        ``node_ids`` fixes the stable node ordering; every edge endpoint must
        appear in it. Edge multiplicity is preserved (parallel edges weight
        the transition). This is the constructor providers and tests share so
        CSR assembly stays in one audited place.
        """
        ids = tuple(node_ids)
        index = {node_id: position for position, node_id in enumerate(ids)}
        source_positions: list[int] = []
        target_positions: list[int] = []
        for source_id, target_id in edges:
            source_position = index.get(source_id)
            target_position = index.get(target_id)
            if source_position is None or target_position is None:
                missing = source_id if source_position is None else target_id
                raise ValueError(f"edge endpoint {missing!r} is not in node_ids")
            source_positions.append(source_position)
            target_positions.append(target_position)
        sources = np.asarray(source_positions, dtype=np.int64)
        targets = np.asarray(target_positions, dtype=np.int64)
        order = np.argsort(sources, kind="stable")
        indptr = np.zeros(len(ids) + 1, dtype=np.int64)
        if len(ids):
            indptr[1:] = np.cumsum(np.bincount(sources, minlength=len(ids)))
        return cls(
            node_ids=ids,
            indptr=indptr,
            indices=targets[order],
            signature=signature,
        )

    @property
    def node_count(self) -> int:
        """Number of nodes in the snapshot."""
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        """Number of directed edges in the snapshot."""
        return int(self.indices.shape[0])


class AdjacencyProvider(Protocol):  # pragma: no cover
    """Backend contract for fetching one session's adjacency snapshot.

    Implementations project the session's active entity graph (the same
    edges the traversal index sees) into an :class:`AdjacencySnapshot`,
    stamping ``signature`` with the backend's log-signature token so callers
    can cache walks until the log advances. Embedded (LadybugDB), Neo4j, and
    Postgres implementations land in the backend wave.
    """

    async def fetch_adjacency(self, session_id: str = "default") -> AdjacencySnapshot:
        """Fetch the active-graph adjacency snapshot for one session."""
        ...


def personalized_pagerank(
    snapshot: AdjacencySnapshot,
    seeds: Iterable[str],
    *,
    alpha: float = 0.85,
    iterations: int = 20,
    tol: float = 1e-8,
    top_n: int,
) -> list[tuple[str, float]]:
    """Rank snapshot nodes by personalized-PageRank mass from ``seeds``.

    Runs bounded power iteration on the row-stochastic transition matrix
    implied by the snapshot, restarting with probability ``1 - alpha`` into
    a distribution uniform over the seed nodes present in the snapshot.

    Design decisions:

    - **Dangling nodes** (zero out-degree) hand their entire mass to the
      restart distribution rather than to a uniform distribution. This keeps
      total mass at exactly 1 while keeping the walk personalized: a dead end
      sends the walker back near the query seeds instead of leaking relevance
      uniformly across the whole graph.
    - **Absent seeds** are ignored; a :class:`ValueError` is raised only when
      *no* seed matches a node in a non-empty snapshot (that signals the
      seeding stage failed, not normal sparsity). An **empty snapshot**
      returns ``[]`` without error — fresh sessions have no graph yet and
      that is not a caller bug.
    - **Unreachable nodes** (disconnected from every seed) receive exactly
      zero mass and are omitted from the result; the returned list may
      therefore be shorter than ``top_n``.
    - **Determinism**: iteration order is fixed by the snapshot; ties are
      broken by sorting on ``(-mass, node_id)``. Top-``top_n`` selection uses
      ``argpartition`` plus a threshold re-scan so boundary ties resolve
      identically to a full sort.

    The loop exits early once the L1 change between iterations drops below
    ``tol``, and is always bounded by ``iterations``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    seed_ids = list(seeds)
    if not seed_ids:
        raise ValueError("at least one seed is required")
    node_count = snapshot.node_count
    if node_count == 0:
        return []
    position = {node_id: index for index, node_id in enumerate(snapshot.node_ids)}
    seed_positions = sorted({position[seed] for seed in seed_ids if seed in position})
    if not seed_positions:
        raise ValueError("none of the seeds are present in the snapshot")

    restart = np.zeros(node_count, dtype=np.float64)
    restart[seed_positions] = 1.0 / len(seed_positions)

    out_degree = np.diff(snapshot.indptr)
    edge_source = np.repeat(np.arange(node_count, dtype=np.int64), out_degree)
    dangling = out_degree == 0
    inverse_out_degree = np.zeros(node_count, dtype=np.float64)
    inverse_out_degree[~dangling] = 1.0 / out_degree[~dangling]

    mass: npt.NDArray[np.float64] = restart.copy()
    for _ in range(iterations):
        contributions = mass[edge_source] * inverse_out_degree[edge_source]
        pushed = np.asarray(
            np.bincount(snapshot.indices, weights=contributions, minlength=node_count),
            dtype=np.float64,
        )
        dangling_mass = float(mass[dangling].sum())
        next_mass: npt.NDArray[np.float64] = (
            alpha * (pushed + dangling_mass * restart) + (1.0 - alpha) * restart
        )
        delta = float(np.abs(next_mass - mass).sum())
        mass = next_mass
        if delta < tol:
            break

    positive = np.flatnonzero(mass > 0.0)
    keep = min(top_n, int(positive.size))
    if keep == 0:
        return []
    if keep < positive.size:
        positive_masses = mass[positive]
        partition = np.argpartition(-positive_masses, keep - 1)
        threshold = float(positive_masses[partition[keep - 1]])
        selected = positive[positive_masses >= threshold]
    else:
        selected = positive
    ranked = sorted(
        ((snapshot.node_ids[int(node)], float(mass[int(node)])) for node in selected),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:keep]


def blend_walk_scores(
    base_scores: Mapping[str, float],
    walk_mass: Mapping[str, float],
    *,
    weight: float,
) -> dict[str, float]:
    """Linearly blend base candidate scores with normalized walk mass.

    Walk masses are **max-normalized** (each divided by the largest mass) so
    the strongest walk hit maps to 1.0 regardless of graph size — raw PPR
    masses shrink as graphs grow, and max-normalization puts them on the same
    ``[0, 1]`` scale as similarity scores. Sum-normalization would instead
    make the walk signal vanish next to similarity scores on large graphs.

    The result covers the union of keys, computed as
    ``(1 - weight) * base + weight * normalized_walk`` with missing values
    treated as 0; keys are emitted in sorted order for determinism. With
    ``weight == 0`` base scores pass through unchanged; with ``weight == 1``
    the result is the pure normalized walk. If ``walk_mass`` is empty or all
    zero, the walk contributes nothing.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")
    if any(value < 0.0 for value in walk_mass.values()):
        raise ValueError("walk masses must be non-negative")
    maximum_mass = max(walk_mass.values(), default=0.0)
    normalized = (
        {key: value / maximum_mass for key, value in walk_mass.items()}
        if maximum_mass > 0.0
        else {}
    )
    return {
        key: (1.0 - weight) * base_scores.get(key, 0.0) + weight * normalized.get(key, 0.0)
        for key in sorted(base_scores.keys() | normalized.keys())
    }
