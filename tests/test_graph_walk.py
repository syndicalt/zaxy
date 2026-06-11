"""Tests for the personalized-PageRank graph-walk core."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from zaxy.graph_walk import (
    AdjacencyProvider,
    AdjacencySnapshot,
    blend_walk_scores,
    personalized_pagerank,
)

ALPHA = 0.85


def _snapshot(
    node_ids: Sequence[str],
    edges: Iterable[tuple[str, str]],
    signature: str = "sig",
) -> AdjacencySnapshot:
    return AdjacencySnapshot.from_edges(node_ids, edges, signature=signature)


def _mass_dict(ranked: list[tuple[str, float]]) -> dict[str, float]:
    return dict(ranked)


def _assert_masses_sum_to_one(ranked: list[tuple[str, float]]) -> None:
    assert sum(mass for _, mass in ranked) == pytest.approx(1.0, abs=1e-12)


def _dense_reference_ppr(
    adjacency: npt.NDArray[np.float64],
    restart: npt.NDArray[np.float64],
    *,
    alpha: float,
    iterations: int,
) -> npt.NDArray[np.float64]:
    """Dense-matrix reference implementation with the same dangling policy."""
    node_count = adjacency.shape[0]
    out_degree = adjacency.sum(axis=1)
    transition = np.zeros((node_count, node_count), dtype=np.float64)
    for row in range(node_count):
        if out_degree[row] > 0:
            transition[row] = adjacency[row] / out_degree[row]
    mass = restart.copy()
    for _ in range(iterations):
        dangling_mass = mass[out_degree == 0].sum()
        mass = alpha * (transition.T @ mass + dangling_mass * restart) + (1.0 - alpha) * restart
    return mass


def test_star_graph_matches_analytic_distribution() -> None:
    # Center seeds three dangling leaves. Stationary solution:
    # center = 1 / (1 + alpha); each leaf = alpha / (3 * (1 + alpha)).
    snapshot = _snapshot(
        ["center", "leaf-a", "leaf-b", "leaf-c"],
        [("center", "leaf-a"), ("center", "leaf-b"), ("center", "leaf-c")],
    )
    ranked = personalized_pagerank(
        snapshot, ["center"], alpha=ALPHA, iterations=500, tol=1e-14, top_n=4
    )
    masses = _mass_dict(ranked)
    assert masses["center"] == pytest.approx(1.0 / (1.0 + ALPHA), abs=1e-9)
    for leaf in ("leaf-a", "leaf-b", "leaf-c"):
        assert masses[leaf] == pytest.approx(ALPHA / (3.0 * (1.0 + ALPHA)), abs=1e-9)
    _assert_masses_sum_to_one(ranked)


def test_matches_dense_reference_on_seeded_random_graph() -> None:
    rng = np.random.default_rng(42)
    node_count = 25
    adjacency = (rng.random((node_count, node_count)) < 0.12).astype(np.float64)
    np.fill_diagonal(adjacency, 0.0)
    node_ids = [f"node-{i:02d}" for i in range(node_count)]
    edges = [
        (node_ids[source], node_ids[target])
        for source in range(node_count)
        for target in range(node_count)
        if adjacency[source, target] > 0
    ]
    snapshot = _snapshot(node_ids, edges)
    seed_positions = [0, 3, 7]
    restart = np.zeros(node_count, dtype=np.float64)
    restart[seed_positions] = 1.0 / len(seed_positions)
    iterations = 60

    expected = _dense_reference_ppr(adjacency, restart, alpha=ALPHA, iterations=iterations)
    # tol=0.0 disables early exit so both implementations run the same count.
    ranked = personalized_pagerank(
        snapshot,
        [node_ids[i] for i in seed_positions],
        alpha=ALPHA,
        iterations=iterations,
        tol=0.0,
        top_n=node_count,
    )
    masses = _mass_dict(ranked)
    for position, node_id in enumerate(node_ids):
        assert masses.get(node_id, 0.0) == pytest.approx(expected[position], abs=1e-12)
    _assert_masses_sum_to_one(ranked)


def test_identical_inputs_produce_identical_output() -> None:
    node_ids = ["seed", "d", "c", "b", "a"]
    edges = [("seed", "d"), ("seed", "c"), ("seed", "b"), ("seed", "a")]
    first = personalized_pagerank(_snapshot(node_ids, edges), ["seed"], top_n=5)
    second = personalized_pagerank(_snapshot(node_ids, edges), ["seed"], top_n=5)
    assert first == second


def test_exact_ties_sorted_by_node_id() -> None:
    snapshot = _snapshot(["seed", "zzz", "aaa"], [("seed", "zzz"), ("seed", "aaa")])
    ranked = personalized_pagerank(snapshot, ["seed"], top_n=3)
    assert [node_id for node_id, _ in ranked] == ["seed", "aaa", "zzz"]
    assert ranked[1][1] == ranked[2][1]
    _assert_masses_sum_to_one(ranked)


def test_top_n_boundary_ties_resolved_by_node_id() -> None:
    # Four exactly tied children; top_n cuts through the tie group, so the
    # argpartition path must still keep the lexicographically smallest ids.
    snapshot = _snapshot(
        ["seed", "d", "c", "b", "a"],
        [("seed", "d"), ("seed", "c"), ("seed", "b"), ("seed", "a")],
    )
    ranked = personalized_pagerank(snapshot, ["seed"], top_n=3)
    assert [node_id for node_id, _ in ranked] == ["seed", "a", "b"]


def test_parallel_edges_increase_transition_mass() -> None:
    snapshot = _snapshot(
        ["seed", "favored", "other"],
        [("seed", "favored"), ("seed", "favored"), ("seed", "other")],
    )
    masses = _mass_dict(personalized_pagerank(snapshot, ["seed"], top_n=3))
    assert masses["favored"] > masses["other"] > 0.0


def test_dangling_nodes_conserve_total_mass() -> None:
    snapshot = _snapshot(
        ["seed", "dead-end-1", "dead-end-2"],
        [("seed", "dead-end-1"), ("seed", "dead-end-2")],
    )
    ranked = personalized_pagerank(snapshot, ["seed"], iterations=200, tol=1e-14, top_n=3)
    _assert_masses_sum_to_one(ranked)
    assert len(ranked) == 3


def test_absent_seeds_ignored_when_one_is_present() -> None:
    snapshot = _snapshot(["s", "a"], [("s", "a")])
    with_ghost = personalized_pagerank(snapshot, ["s", "ghost"], top_n=2)
    without_ghost = personalized_pagerank(snapshot, ["s"], top_n=2)
    assert with_ghost == without_ghost


def test_all_seeds_absent_raises() -> None:
    snapshot = _snapshot(["s", "a"], [("s", "a")])
    with pytest.raises(ValueError, match="none of the seeds"):
        personalized_pagerank(snapshot, ["ghost", "phantom"], top_n=2)


def test_empty_seeds_raises() -> None:
    snapshot = _snapshot(["s"], [])
    with pytest.raises(ValueError, match="at least one seed"):
        personalized_pagerank(snapshot, [], top_n=1)


def test_empty_graph_returns_empty() -> None:
    snapshot = _snapshot([], [])
    assert personalized_pagerank(snapshot, ["anything"], top_n=5) == []


def test_single_node_graph_holds_all_mass() -> None:
    snapshot = _snapshot(["only"], [])
    ranked = personalized_pagerank(snapshot, ["only"], top_n=5)
    assert ranked == [("only", pytest.approx(1.0, abs=1e-12))]


def test_top_n_exceeding_node_count_returns_all_positive_nodes() -> None:
    snapshot = _snapshot(["s", "a", "b"], [("s", "a"), ("a", "b"), ("b", "s")])
    ranked = personalized_pagerank(snapshot, ["s"], top_n=100)
    assert len(ranked) == 3
    _assert_masses_sum_to_one(ranked)


def test_top_n_zero_returns_empty() -> None:
    snapshot = _snapshot(["s", "a"], [("s", "a")])
    assert personalized_pagerank(snapshot, ["s"], top_n=0) == []


def test_disconnected_component_gets_zero_mass() -> None:
    snapshot = _snapshot(
        ["s", "a", "x", "y"],
        [("s", "a"), ("x", "y"), ("y", "x")],
    )
    ranked = personalized_pagerank(snapshot, ["s"], iterations=200, tol=1e-14, top_n=4)
    returned = {node_id for node_id, _ in ranked}
    assert returned == {"s", "a"}
    _assert_masses_sum_to_one(ranked)


def test_tol_early_exit_is_observable() -> None:
    snapshot = _snapshot(["s", "a", "b"], [("s", "a"), ("a", "b")])
    # A huge tol exits after the first iteration (L1 delta is at most 2),
    # which must match running exactly one iteration with early exit disabled.
    early_exit = personalized_pagerank(snapshot, ["s"], iterations=500, tol=10.0, top_n=3)
    one_iteration = personalized_pagerank(snapshot, ["s"], iterations=1, tol=0.0, top_n=3)
    assert early_exit == one_iteration


def test_iteration_bound_is_respected() -> None:
    snapshot = _snapshot(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "d")])
    one = _mass_dict(personalized_pagerank(snapshot, ["a"], iterations=1, tol=0.0, top_n=4))
    many = _mass_dict(personalized_pagerank(snapshot, ["a"], iterations=50, tol=0.0, top_n=4))
    assert "d" not in one  # three hops away: unreachable in a single iteration
    assert many["d"] > 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"iterations": 0},
        {"tol": -1e-9},
        {"top_n": -1},
    ],
)
def test_invalid_parameters_raise(overrides: dict[str, Any]) -> None:
    snapshot = _snapshot(["s", "a"], [("s", "a")])
    kwargs: dict[str, Any] = {"alpha": ALPHA, "iterations": 5, "tol": 0.0, "top_n": 2}
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        personalized_pagerank(snapshot, ["s"], **kwargs)


def test_from_edges_unknown_endpoint_raises() -> None:
    with pytest.raises(ValueError, match="not in node_ids"):
        _snapshot(["a"], [("a", "ghost")])


def test_duplicate_node_ids_raise() -> None:
    with pytest.raises(ValueError, match="unique"):
        _snapshot(["a", "a"], [])


def test_snapshot_rejects_inconsistent_csr_arrays() -> None:
    with pytest.raises(ValueError, match="indptr"):
        AdjacencySnapshot(
            node_ids=("a", "b"),
            indptr=np.array([0, 1], dtype=np.int64),  # wrong length
            indices=np.array([1], dtype=np.int64),
            signature="sig",
        )
    with pytest.raises(ValueError, match="indices"):
        AdjacencySnapshot(
            node_ids=("a", "b"),
            indptr=np.array([0, 1, 1], dtype=np.int64),
            indices=np.array([5], dtype=np.int64),  # out of range
            signature="sig",
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        AdjacencySnapshot(
            node_ids=("a", "b"),
            indptr=np.array([0, 1, 1], dtype=np.int64),
            indices=np.array([[1]], dtype=np.int64),  # two-dimensional
            signature="sig",
        )
    with pytest.raises(ValueError, match="edge count"):
        AdjacencySnapshot(
            node_ids=("a", "b"),
            indptr=np.array([0, 1, 2], dtype=np.int64),  # claims 2 edges
            indices=np.array([1], dtype=np.int64),
            signature="sig",
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        AdjacencySnapshot(
            node_ids=("a", "b"),
            indptr=np.array([0, 2, 1], dtype=np.int64),
            indices=np.array([0], dtype=np.int64),
            signature="sig",
        )
    with pytest.raises(ValueError, match="no edges"):
        AdjacencySnapshot(
            node_ids=(),
            indptr=np.array([0], dtype=np.int64),
            indices=np.array([0], dtype=np.int64),
            signature="sig",
        )


def test_snapshot_counts_and_signature() -> None:
    snapshot = _snapshot(["a", "b"], [("a", "b")], signature="log-sig-7")
    assert snapshot.node_count == 2
    assert snapshot.edge_count == 1
    assert snapshot.signature == "log-sig-7"


def test_blend_weight_zero_passes_base_through() -> None:
    base = {"a": 0.9, "b": 0.2}
    walk = {"b": 0.05, "c": 0.10}
    blended = blend_walk_scores(base, walk, weight=0.0)
    assert blended["a"] == pytest.approx(0.9)
    assert blended["b"] == pytest.approx(0.2)
    assert blended["c"] == pytest.approx(0.0)


def test_blend_weight_one_returns_pure_normalized_walk() -> None:
    base = {"a": 0.9}
    walk = {"b": 0.05, "c": 0.10}
    blended = blend_walk_scores(base, walk, weight=1.0)
    assert blended["c"] == pytest.approx(1.0)  # max-normalized
    assert blended["b"] == pytest.approx(0.5)
    assert blended["a"] == pytest.approx(0.0)


def test_blend_intermediate_weight_is_linear() -> None:
    base = {"a": 0.8, "b": 0.4}
    walk = {"a": 0.10, "b": 0.05}
    blended = blend_walk_scores(base, walk, weight=0.25)
    assert blended["a"] == pytest.approx(0.75 * 0.8 + 0.25 * 1.0)
    assert blended["b"] == pytest.approx(0.75 * 0.4 + 0.25 * 0.5)


def test_blend_handles_missing_keys_on_either_side() -> None:
    blended = blend_walk_scores({"only-base": 0.6}, {"only-walk": 0.3}, weight=0.5)
    assert blended == {
        "only-base": pytest.approx(0.3),
        "only-walk": pytest.approx(0.5),
    }


def test_blend_empty_or_zero_walk_mass_contributes_nothing() -> None:
    base = {"a": 0.8}
    assert blend_walk_scores(base, {}, weight=0.5)["a"] == pytest.approx(0.4)
    assert blend_walk_scores(base, {"a": 0.0}, weight=0.5)["a"] == pytest.approx(0.4)


def test_blend_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="weight"):
        blend_walk_scores({}, {}, weight=1.5)
    with pytest.raises(ValueError, match="non-negative"):
        blend_walk_scores({}, {"a": -0.1}, weight=0.5)


class _InMemoryAdjacencyProvider:
    """In-memory AdjacencyProvider fake standing in for backend stores."""

    def __init__(self, snapshots: dict[str, AdjacencySnapshot]) -> None:
        self._snapshots = snapshots

    async def fetch_adjacency(self, session_id: str = "default") -> AdjacencySnapshot:
        return self._snapshots[session_id]


async def test_in_memory_provider_satisfies_contract() -> None:
    snapshot = _snapshot(["s", "a"], [("s", "a")], signature="log-sig-1")
    provider: AdjacencyProvider = _InMemoryAdjacencyProvider({"session-1": snapshot})
    fetched = await provider.fetch_adjacency("session-1")
    assert fetched.signature == "log-sig-1"
    ranked = personalized_pagerank(fetched, ["s"], top_n=2)
    assert [node_id for node_id, _ in ranked] == ["s", "a"]
    _assert_masses_sum_to_one(ranked)
