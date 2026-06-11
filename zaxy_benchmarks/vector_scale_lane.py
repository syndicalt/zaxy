"""Internal vector-scale lane: exact vs HNSW vs int8-quantized retrieval.

This lane measures the 2.3 embedded vector index modes directly against
:class:`zaxy.embedded_graph_store.EmbeddedGraphStore` (the store search
surface, not a full fabric, so 10^4-10^5 corpora build in seconds):

- **exact**: the default float64 dense-matrix path (ground truth; its
  recall@k is 1.0 by definition).
- **ann**: the Kuzu-native HNSW path, engaged by lowering
  ``vector_ann_threshold`` below the corpus size.
- **quantized**: the opt-in ``vector_quantization="int8"`` path (int8 dot
  products over oversampled candidates, exact float64 rerank).

Two recall metrics are reported for every mode, always together:

- ``recall_at_k_strict``: identity recall against the exact store's returned
  top-k names — the original lane metric, unchanged in semantics. It is kept
  unconditionally in every result block for continuity with all pre-2.2
  evidence.
- ``recall_at_k_tie_aware``: a retrieved vector counts as a hit when its
  exact float64 cosine score is >= the k-th true score for that query — the
  standard ann-benchmarks treatment of tied boundaries.

Exit criteria evaluate the tie-aware metric. Rationale (measured, see
``docs/research/artifacts/ann-2026-06/diag-ties-gaussian.json``): at
dimension 1536 the hash-embedding corpus has a *median of 210 corpus vectors
exactly tied* with the true top-10, and the float64 score gap between rank 10
and rank 40 is 0.0. Strict identity recall is ill-posed there — any top-10
drawn from the tied set is equally correct, and an exact float32 brute-force
scan caps at 0.5344 strict recall against float64 ground truth
(``diag-d1536-f32-ceiling.json``). Tie-aware recall scores every member of
the tied set as correct, which is *stricter* in spirit: it never rewards a
wrong vector, it only stops punishing equally-right ones. The strict number
stays in the output so the divergence is always visible.

Two corpus distributions are supported via ``distribution``:

- ``hash`` (default, preserving comparability with all prior lane evidence):
  synthetic texts embedded with the deterministic
  :class:`~zaxy.embedding.HashEmbeddingProvider`.
- ``gaussian``: seeded standard-normal vectors, unit-normalized — the
  realistic-distribution control. Hash-embedding value distributions are
  tie-dense and adversarial at high dimension (the 210-way ties above);
  real embedding models produce continuous distributions for which Gaussian
  is the standard proxy. Vectors come from a fixed-seed
  ``numpy.random.default_rng`` (PCG64) stream, so the corpus is exactly
  reproducible. The high-dim (1536) gate corpus per the 2.2 ANN plan is
  ``gaussian``.

Corpus content, exact/quantized recall (both metrics), group types, resident
index bytes, and the byte-budget block are exactly reproducible across runs
and live in the ``deterministic`` block. Query latency and on-disk ANN bytes
are machine-dependent and live in the ``measurements`` block, which is
explicitly excluded from determinism comparisons. **ANN recall is also
reported under ``measurements``**: the Kuzu-native HNSW graph construction is
not run-to-run reproducible (rebuilding the same corpus yields slightly
different recall), so per-run ANN recall is honest measurement, not a
deterministic claim — this is a real property of the production ANN path,
observed by this lane.

Each size also reports a ``byte_budget`` block (``exact_matrix_bytes`` =
count x dim x 8 versus ``VECTOR_INDEX_CACHE_MAX_BYTES``) so threshold
decisions can be made on the memory axis, where exact search actually stops
being viable at high dimension. This is analysis only; it gates nothing.

Corpus construction injects entities at the store's vector-index seam — the
same construction pattern as ``tests/test_embedded_graph_store.py`` — so the
measured search path (index build, HNSW shadow sync, quantized rerank) is the
real production path while corpus ingestion stays fast enough to scale.

The default sizes are ``(1_000, 10_000)``; the 10^5 point is exposed via the
``sizes`` parameter. The roadmap exit criterion is defined at 10^5, so runs
at smaller sizes report ``not_evaluated_at_target_scale`` rather than a pass.

Every result is labeled ``"validation": "internal"``: synthetic corpora,
synthetic vector distributions, environment-dependent timings — not a public
benchmark claim.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from zaxy.embedded_graph_store import EmbeddedGraphStore
    from zaxy.graph import GraphEntity

VECTOR_SCALE_LANE_VERSION = "vector-scale-lane-v2"
VALIDATION_LABEL = "internal"

DEFAULT_SIZES: tuple[int, ...] = (1_000, 10_000)
TARGET_SCALE = 100_000
RECALL_FLOOR = 0.95

DEFAULT_DIMENSION = 64
DEFAULT_ANN_THRESHOLD = 256
DEFAULT_QUERY_COUNT = 32
DEFAULT_TOP_K = 10
DEFAULT_LATENCY_PASSES = 3
DEFAULT_DISTRIBUTION = "hash"
DISTRIBUTIONS: tuple[str, ...] = ("hash", "gaussian")

#: Fixed PCG64 seeds for the gaussian (realistic-distribution) corpus and
#: query streams. The streams are deterministic for a given NumPy generator
#: implementation, so gaussian corpora are exactly reproducible across runs.
GAUSSIAN_CORPUS_SEED = 20260611
GAUSSIAN_QUERY_SEED = 20260612

_LANE_SESSION_ID = "scale-lane"
MODE_NAMES: tuple[str, ...] = ("exact", "ann", "quantized")


def corpus_texts(count: int) -> list[str]:
    """Deterministic synthetic corpus: shared topic tokens + unique tokens.

    Each record carries enough distinct tokens that hash-embedding collisions
    (which would create exact similarity ties and make recall ill-defined)
    are negligible at 10^5 records *at low dimension*. At high dimension the
    hash distribution is tie-dense regardless (median 210-way exact top-10
    ties measured at dim 1536), which is why the lane reports tie-aware
    recall and offers the gaussian distribution as the realistic control.
    """
    return [
        f"record {index} topic {index % 50} subject s{index * 2654435761 % 2**31} "
        f"facet f{index * 40503 % 99991} detail d{index * 69069 % 999983} marker m{index % 977}"
        for index in range(count)
    ]


def query_texts(count: int) -> list[str]:
    """Deterministic query set sharing topic/marker tokens with the corpus."""
    return [
        f"probe {index} topic {index % 50} marker m{index % 977} lookup l{index * 7919 % 104729}"
        for index in range(count)
    ]


def gaussian_unit_vectors(count: int, dimension: int, seed: int) -> npt.NDArray[np.float64]:
    """Seeded standard-normal vectors, unit-normalized (deterministic)."""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((count, dimension))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / norms, dtype=np.float64)


def distribution_version_tag(distribution: str, dimension: int) -> str:
    """Embedding-version tag stamped onto lane vectors for one distribution."""
    if distribution == "gaussian":
        return f"gaussian-seed{GAUSSIAN_CORPUS_SEED}-dim{dimension}"
    from zaxy.embedding import HashEmbeddingProvider

    return HashEmbeddingProvider(dimension=dimension).version_tag


def _build_entities(
    texts: Sequence[str],
    vectors: Sequence[list[float]],
    version_tag: str,
) -> list[GraphEntity]:
    from zaxy.graph import GraphEntity

    return [
        GraphEntity(
            name=f"record-{index}",
            entity_type="record",
            valid_from="2026-06-10T00:00:00Z",
            valid_to=None,
            properties={
                "embedding": vectors[index],
                "embedding_version": version_tag,
                "summary": texts[index],
            },
            session_id=_LANE_SESSION_ID,
        )
        for index in range(len(texts))
    ]


def _corpus_sha256(texts: Sequence[str], matrix: npt.NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1f")
    digest.update(np.ascontiguousarray(matrix).tobytes())
    return digest.hexdigest()


def _percentile_ms(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = int(round(fraction * (len(ordered) - 1)))
    return round(ordered[position] * 1000.0, 3)


def _group_type_name(store: EmbeddedGraphStore, dimension: int, version_tag: str) -> str:
    from zaxy import embedded_graph_store as store_module

    group = store._vector_index(_LANE_SESSION_ID, None).groups[(dimension, version_tag)]
    if isinstance(group, store_module._AnnVectorGroup):
        return "ann"
    if isinstance(group, store_module._QuantizedVectorGroup):
        return "quantized"
    return "dense"


def _resident_index_bytes(store: EmbeddedGraphStore) -> int:
    return int(store._vector_index(_LANE_SESSION_ID, None).matrix_bytes)


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


async def _measure_mode(
    store: EmbeddedGraphStore,
    query_vectors: Sequence[list[float]],
    *,
    top_k: int,
    latency_passes: int,
) -> tuple[list[list[str]], dict[str, Any]]:
    """Return per-query result-name lists and raw timing samples for one store."""
    build_start = time.perf_counter()
    await store.search_vector(query_vectors[0], limit=top_k, session_id=_LANE_SESSION_ID)
    first_query_seconds = time.perf_counter() - build_start

    results: list[list[str]] = []
    for vector in query_vectors:
        hits = await store.search_vector(vector, limit=top_k, session_id=_LANE_SESSION_ID)
        results.append([hit.entity.name for hit in hits])

    samples: list[float] = []
    for _ in range(latency_passes):
        for vector in query_vectors:
            start = time.perf_counter()
            await store.search_vector(vector, limit=top_k, session_id=_LANE_SESSION_ID)
            samples.append(time.perf_counter() - start)
    timing = {
        "first_query_ms": round(first_query_seconds * 1000.0, 3),
        "p50_ms": _percentile_ms(samples, 0.50),
        "p95_ms": _percentile_ms(samples, 0.95),
        "samples": len(samples),
    }
    return results, timing


def _strict_recall_at_k(ground_truth: list[list[str]], candidate: list[list[str]]) -> float:
    """Identity recall@k against the exact store's returned names (legacy metric)."""
    recalls = [
        len(set(expected) & set(actual)) / max(1, len(expected))
        for expected, actual in zip(ground_truth, candidate, strict=True)
    ]
    return round(sum(recalls) / len(recalls), 4)


def exact_score_matrix(
    corpus_matrix: npt.NDArray[np.float64],
    query_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Exact float64 cosine score matrix, shape (query_count, corpus_count).

    Rows and queries are unit-normalized exactly as the store's dense path
    normalizes them (per-row scalar norm division), and each query scores via
    one matrix-vector product — the same operation the exact path performs —
    so lane ground-truth scores are bit-identical to production scores.
    """
    corpus_norms = np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
    safe_norms = np.where(corpus_norms > 0.0, corpus_norms, 1.0)
    unit_corpus = np.asarray(corpus_matrix / safe_norms, dtype=np.float64)
    rows: list[npt.NDArray[np.float64]] = []
    for query in query_matrix:
        norm = float(np.linalg.norm(query))
        unit_query = query / norm if norm > 0.0 else query
        rows.append(unit_corpus @ unit_query)
    return np.stack(rows)


def tie_aware_recall_at_k(
    score_matrix: npt.NDArray[np.float64],
    retrieved_rows: Sequence[Sequence[int]],
    top_k: int,
) -> float:
    """Mean tie-aware recall@k over the query set.

    A retrieved corpus row counts as a hit when its exact float64 score is
    >= the k-th true score for that query (standard ann-benchmarks tie
    handling). The comparison is exact — no epsilon — because both sides are
    drawn from the same score matrix.
    """
    query_count, corpus_count = score_matrix.shape
    if len(retrieved_rows) != query_count:
        raise ValueError("retrieved_rows must match the score matrix query count")
    k = min(top_k, corpus_count)
    if k < 1:
        raise ValueError("top_k and corpus size must be at least 1")
    recalls: list[float] = []
    for query_index, rows in enumerate(retrieved_rows):
        scores = score_matrix[query_index]
        threshold = float(np.partition(scores, corpus_count - k)[corpus_count - k])
        hits = sum(1 for row in rows if float(scores[row]) >= threshold)
        recalls.append(hits / k)
    return round(sum(recalls) / len(recalls), 4)


def _entity_rows(names: Sequence[str]) -> list[int]:
    """Map lane entity names (``record-{row}``) back to corpus row indexes."""
    return [int(name.rsplit("-", 1)[1]) for name in names]


def _byte_budget(size: int, dimension: int) -> dict[str, Any]:
    """Exact-matrix resident bytes versus the store's vector-cache budget."""
    from zaxy.embedded_graph_store import VECTOR_INDEX_CACHE_MAX_BYTES

    exact_matrix_bytes = size * dimension * 8
    return {
        "exact_matrix_bytes": exact_matrix_bytes,
        "vector_index_cache_max_bytes": VECTOR_INDEX_CACHE_MAX_BYTES,
        "budget_fraction": round(exact_matrix_bytes / VECTOR_INDEX_CACHE_MAX_BYTES, 6),
        "exceeds_budget": exact_matrix_bytes > VECTOR_INDEX_CACHE_MAX_BYTES,
    }


def _lane_vectors(
    distribution: str,
    size: int,
    dimension: int,
    query_count: int,
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64], str]:
    """Build (texts, corpus matrix, query matrix, version tag) for one size."""
    texts = corpus_texts(size)
    if distribution == "gaussian":
        matrix = gaussian_unit_vectors(size, dimension, GAUSSIAN_CORPUS_SEED)
        queries = gaussian_unit_vectors(query_count, dimension, GAUSSIAN_QUERY_SEED)
    else:
        from zaxy.embedding import HashEmbeddingProvider

        provider = HashEmbeddingProvider(dimension=dimension)
        matrix = np.asarray([provider.embed(text) for text in texts], dtype=np.float64)
        queries = np.asarray(
            [provider.embed(text) for text in query_texts(query_count)], dtype=np.float64
        )
    return texts, matrix, queries, distribution_version_tag(distribution, dimension)


async def _run_size(
    workdir: Path,
    size: int,
    *,
    dimension: int,
    distribution: str,
    ann_threshold: int,
    query_count: int,
    top_k: int,
    latency_passes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all three modes at one corpus size; returns (deterministic, measurements)."""
    from zaxy.embedded_graph_store import EmbeddedGraphStore

    texts, matrix, query_matrix, version_tag = _lane_vectors(
        distribution, size, dimension, query_count
    )
    vectors = [list(map(float, row)) for row in matrix]
    entities = _build_entities(texts, vectors, version_tag)
    query_vectors = [list(map(float, row)) for row in query_matrix]
    score_matrix = exact_score_matrix(matrix, query_matrix)

    exact_store = EmbeddedGraphStore(
        workdir / f"exact-{size}.kuzu",
        active_embedding_version=version_tag,
        vector_ann_threshold=size + 1,
        vector_quantization="none",
    )
    exact_store._current_entity_index_cache[_LANE_SESSION_ID] = entities

    ann_path = workdir / f"ann-{size}.kuzu"
    ann_store = EmbeddedGraphStore(
        ann_path,
        active_embedding_version=version_tag,
        vector_ann_threshold=ann_threshold,
        vector_quantization="none",
    )
    await ann_store.connect()
    await ann_store.init_schema()
    ann_store._current_entity_index_cache[_LANE_SESSION_ID] = entities

    quantized_store = EmbeddedGraphStore(
        workdir / f"quantized-{size}.kuzu",
        active_embedding_version=version_tag,
        vector_ann_threshold=size + 1,
        vector_quantization="int8",
    )
    quantized_store._current_entity_index_cache[_LANE_SESSION_ID] = entities

    deterministic_modes: dict[str, Any] = {}
    timing_modes: dict[str, Any] = {}
    try:
        exact_results, exact_timing = await _measure_mode(
            exact_store, query_vectors, top_k=top_k, latency_passes=latency_passes
        )
        strict_recall: dict[str, float] = {}
        tie_aware_recall: dict[str, float] = {}
        for mode, store in (("ann", ann_store), ("quantized", quantized_store)):
            results, timing = await _measure_mode(
                store, query_vectors, top_k=top_k, latency_passes=latency_passes
            )
            group_type = _group_type_name(store, dimension, version_tag)
            strict_recall[mode] = _strict_recall_at_k(exact_results, results)
            tie_aware_recall[mode] = tie_aware_recall_at_k(
                score_matrix, [_entity_rows(names) for names in results], top_k
            )
            deterministic_modes[mode] = {
                "group_type": group_type,
                "engaged": group_type == mode,
                "resident_index_bytes": _resident_index_bytes(store),
            }
            timing_modes[mode] = timing
        # Quantized recall (int8 oversample + exact float64 rerank) is exactly
        # reproducible; HNSW recall is not (Kuzu's graph construction varies
        # across rebuilds of the same corpus), so it stays under measurements.
        deterministic_modes["quantized"]["recall_at_k_strict"] = strict_recall["quantized"]
        deterministic_modes["quantized"]["recall_at_k_tie_aware"] = tie_aware_recall["quantized"]
        exact_bytes = _resident_index_bytes(exact_store)
        deterministic_modes["exact"] = {
            "group_type": _group_type_name(exact_store, dimension, version_tag),
            "engaged": True,
            "recall_at_k_strict": 1.0,
            # Computed against the lane's independent float64 score matrix as
            # a cross-check that lane ground truth matches the store exactly.
            "recall_at_k_tie_aware": tie_aware_recall_at_k(
                score_matrix, [_entity_rows(names) for names in exact_results], top_k
            ),
            "recall_note": "exact mode is the ground truth; strict recall@k is 1.0 by definition",
            "resident_index_bytes": exact_bytes,
        }
        timing_modes["exact"] = exact_timing
        for mode in ("ann", "quantized"):
            deterministic_modes[mode]["bytes_vs_exact_ratio"] = (
                round(deterministic_modes[mode]["resident_index_bytes"] / exact_bytes, 4)
                if exact_bytes
                else None
            )
    finally:
        await ann_store.close()

    deterministic = {
        "corpus_sha256": _corpus_sha256(texts, matrix),
        "vector_count": size,
        "distribution": distribution,
        "byte_budget": _byte_budget(size, dimension),
        "modes": deterministic_modes,
    }
    measurements = {
        "latency_ms": timing_modes,
        "ann_recall_at_k_strict": strict_recall["ann"],
        "ann_recall_at_k_tie_aware": tie_aware_recall["ann"],
        "ann_backing_store_bytes": _directory_bytes(ann_path),
    }
    return deterministic, measurements


def run_vector_scale_lane(
    workdir: Path,
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    dimension: int = DEFAULT_DIMENSION,
    distribution: str = DEFAULT_DISTRIBUTION,
    ann_threshold: int = DEFAULT_ANN_THRESHOLD,
    query_count: int = DEFAULT_QUERY_COUNT,
    top_k: int = DEFAULT_TOP_K,
    latency_passes: int = DEFAULT_LATENCY_PASSES,
) -> dict[str, Any]:
    """Run the vector-scale lane in ``workdir`` and return one labeled report.

    ``sizes`` are corpus sizes to run (ascending order is enforced); the
    roadmap exit criterion is evaluated at the largest size actually run and
    reported as ``not_evaluated_at_target_scale`` below ``TARGET_SCALE``.
    ``distribution`` selects the corpus vector distribution: ``hash``
    (default, comparable with all prior lane evidence) or ``gaussian`` (the
    seeded realistic-distribution control; the high-dim gate corpus).
    """
    if not sizes:
        raise ValueError("sizes must include at least one corpus size")
    if any(size < 1 for size in sizes):
        raise ValueError("sizes must be positive")
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"distribution must be one of {', '.join(DISTRIBUTIONS)}; got {distribution!r}"
        )
    if query_count < 1:
        raise ValueError("query_count must be at least 1")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if latency_passes < 1:
        raise ValueError("latency_passes must be at least 1")
    return asyncio.run(
        _run_vector_scale_lane_async(
            workdir,
            sizes=tuple(sorted(set(sizes))),
            dimension=dimension,
            distribution=distribution,
            ann_threshold=ann_threshold,
            query_count=query_count,
            top_k=top_k,
            latency_passes=latency_passes,
        )
    )


async def _run_vector_scale_lane_async(
    workdir: Path,
    *,
    sizes: tuple[int, ...],
    dimension: int,
    distribution: str,
    ann_threshold: int,
    query_count: int,
    top_k: int,
    latency_passes: int,
) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    deterministic_sizes: dict[str, Any] = {}
    measurement_sizes: dict[str, Any] = {}
    for size in sizes:
        deterministic, measurements = await _run_size(
            workdir,
            size,
            dimension=dimension,
            distribution=distribution,
            ann_threshold=ann_threshold,
            query_count=query_count,
            top_k=top_k,
            latency_passes=latency_passes,
        )
        deterministic_sizes[str(size)] = deterministic
        measurement_sizes[str(size)] = measurements

    largest = str(sizes[-1])
    largest_modes = deterministic_sizes[largest]["modes"]
    largest_timing = measurement_sizes[largest]["latency_ms"]
    exact_bytes = largest_modes["exact"]["resident_index_bytes"]
    exact_p50 = largest_timing["exact"]["p50_ms"]
    mode_verdicts: dict[str, Any] = {}
    for mode in ("ann", "quantized"):
        mode_block = largest_modes[mode]
        if mode == "quantized":
            strict = mode_block["recall_at_k_strict"]
            tie_aware = mode_block["recall_at_k_tie_aware"]
        else:
            strict = measurement_sizes[largest]["ann_recall_at_k_strict"]
            tie_aware = measurement_sizes[largest]["ann_recall_at_k_tie_aware"]
        verdict = {
            "engaged": mode_block["engaged"],
            "recall_at_k_strict": strict,
            "recall_at_k_tie_aware": tie_aware,
            "recall_metric": "tie_aware",
            "recall_pass": tie_aware >= RECALL_FLOOR,
            "bytes_improved": mode_block["resident_index_bytes"] < exact_bytes,
            "latency_improved_p50": largest_timing[mode]["p50_ms"] < exact_p50,
        }
        verdict["all_criteria_met"] = bool(
            verdict["engaged"]
            and verdict["recall_pass"]
            and verdict["bytes_improved"]
            and verdict["latency_improved_p50"]
        )
        mode_verdicts[mode] = verdict

    evaluated_at_target_scale = sizes[-1] >= TARGET_SCALE
    if not evaluated_at_target_scale:
        status = "not_evaluated_at_target_scale"
    elif any(verdict["all_criteria_met"] for verdict in mode_verdicts.values()):
        status = "pass"
    else:
        status = "fail"

    return {
        "lane": "vector_scale",
        "version": VECTOR_SCALE_LANE_VERSION,
        "validation": VALIDATION_LABEL,
        "measurement": (
            "strict and tie-aware recall@k vs exact float64 ground truth, p50/p95 "
            "query latency, resident index bytes, and byte-budget fractions for "
            "the embedded store's exact, Kuzu-HNSW, and int8-quantized vector "
            "modes over deterministic synthetic corpora (hash-embedded or seeded "
            "gaussian). Timings are environment-dependent."
        ),
        "config": {
            "sizes": list(sizes),
            "dimension": dimension,
            "distribution": distribution,
            "ann_threshold": ann_threshold,
            "query_count": query_count,
            "top_k": top_k,
            "latency_passes": latency_passes,
            "embedding_version_tag": distribution_version_tag(distribution, dimension),
            "recall_floor": RECALL_FLOOR,
            "target_scale": TARGET_SCALE,
        },
        "deterministic": {
            "note": (
                "Corpus hashes, exact/quantized recall (strict and tie-aware), "
                "group types, resident bytes, and byte budgets are exactly "
                "reproducible across runs."
            ),
            "sizes": deterministic_sizes,
        },
        "measurements": {
            "note": (
                "Environment- or run-dependent values; excluded from determinism "
                "comparisons. ann recall (both metrics) lives here because "
                "Kuzu's HNSW graph construction is not run-to-run reproducible: "
                "rebuilding the same corpus yields slightly different recall."
            ),
            "sizes": measurement_sizes,
        },
        "exit_criteria": {
            "evaluated_at_size": sizes[-1],
            "evaluated_at_target_scale": evaluated_at_target_scale,
            "modes": mode_verdicts,
            "note": (
                "recall_pass evaluates recall_at_k_tie_aware (hit = exact "
                "float64 score >= the k-th true score, standard ann-benchmarks "
                "tie handling): at dim 1536 the hash corpus has a measured "
                "median of 210 corpus vectors exactly tied with the true "
                "top-10, so strict identity recall is ill-posed there. Strict "
                "recall is reported unconditionally alongside it. Quantized "
                "recall and bytes_improved are deterministic; ann recall "
                "varies slightly per run (HNSW rebuild nondeterminism) and "
                "latency_improved_p50 depends on the host machine. The roadmap "
                "criterion is defined at 10^5 vectors; smaller runs report "
                "not_evaluated_at_target_scale."
            ),
            "status": status,
        },
    }
