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

Corpora are synthetic texts embedded with the deterministic
:class:`~zaxy.embedding.HashEmbeddingProvider`; corpus content, exact and
quantized recall@k, group types, and resident index bytes are exactly
reproducible across runs and live in the ``deterministic`` block. Query
latency and on-disk ANN bytes are machine-dependent and live in the
``measurements`` block, which is explicitly excluded from determinism
comparisons. **ANN recall is also reported under ``measurements``**: the
Kuzu-native HNSW graph construction is not run-to-run reproducible (rebuilding
the same corpus yields slightly different recall), so per-run ANN recall is
honest measurement, not a deterministic claim — this is a real property of
the production ANN path, observed by this lane.

Corpus construction injects entities at the store's vector-index seam — the
same construction pattern as ``tests/test_embedded_graph_store.py`` — so the
measured search path (index build, HNSW shadow sync, quantized rerank) is the
real production path while corpus ingestion stays fast enough to scale.

The default sizes are ``(1_000, 10_000)``. The 10^5 point is exposed via the
``sizes`` parameter but excluded by default: on the reference development
machine the HNSW shadow-table sync alone takes roughly a minute at 10^4
vectors (insert-bound), so 10^5 cannot complete "in about a minute". The
roadmap exit criterion is defined at 10^5, so runs at smaller sizes report
``not_evaluated_at_target_scale`` rather than a pass.

Every result is labeled ``"validation": "internal"``: synthetic corpora, hash
embeddings, environment-dependent timings — not a public benchmark claim.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from zaxy.embedded_graph_store import EmbeddedGraphStore
    from zaxy.graph import GraphEntity

VECTOR_SCALE_LANE_VERSION = "vector-scale-lane-v1"
VALIDATION_LABEL = "internal"

#: 10^5 is excluded by default: the HNSW shadow sync is insert-bound and takes
#: ~1 minute at 10^4 on the reference machine, so 10^5 needs ~10 minutes.
DEFAULT_SIZES: tuple[int, ...] = (1_000, 10_000)
TARGET_SCALE = 100_000
RECALL_FLOOR = 0.95

DEFAULT_DIMENSION = 64
DEFAULT_ANN_THRESHOLD = 256
DEFAULT_QUERY_COUNT = 32
DEFAULT_TOP_K = 10
DEFAULT_LATENCY_PASSES = 3

_LANE_SESSION_ID = "scale-lane"
MODE_NAMES: tuple[str, ...] = ("exact", "ann", "quantized")


def corpus_texts(count: int) -> list[str]:
    """Deterministic synthetic corpus: shared topic tokens + unique tokens.

    Each record carries enough distinct tokens that hash-embedding collisions
    (which would create exact similarity ties and make recall ill-defined)
    are negligible at 10^5 records.
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


def _corpus_sha256(texts: Sequence[str], matrix: np.ndarray) -> str:
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


def _recall_at_k(ground_truth: list[list[str]], candidate: list[list[str]]) -> float:
    recalls = [
        len(set(expected) & set(actual)) / max(1, len(expected))
        for expected, actual in zip(ground_truth, candidate, strict=True)
    ]
    return round(sum(recalls) / len(recalls), 4)


async def _run_size(
    workdir: Path,
    size: int,
    *,
    dimension: int,
    ann_threshold: int,
    query_count: int,
    top_k: int,
    latency_passes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run all three modes at one corpus size; returns (deterministic, measurements)."""
    from zaxy.embedded_graph_store import EmbeddedGraphStore
    from zaxy.embedding import HashEmbeddingProvider

    provider = HashEmbeddingProvider(dimension=dimension)
    version_tag = provider.version_tag
    texts = corpus_texts(size)
    vectors = [provider.embed(text) for text in texts]
    matrix = np.asarray(vectors, dtype=np.float64)
    entities = _build_entities(texts, vectors, version_tag)
    query_vectors = [provider.embed(text) for text in query_texts(query_count)]

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
        measured_recall: dict[str, float] = {}
        for mode, store in (("ann", ann_store), ("quantized", quantized_store)):
            results, timing = await _measure_mode(
                store, query_vectors, top_k=top_k, latency_passes=latency_passes
            )
            group_type = _group_type_name(store, dimension, version_tag)
            measured_recall[mode] = _recall_at_k(exact_results, results)
            deterministic_modes[mode] = {
                "group_type": group_type,
                "engaged": group_type == mode,
                "resident_index_bytes": _resident_index_bytes(store),
            }
            timing_modes[mode] = timing
        # Quantized recall (int8 oversample + exact float64 rerank) is exactly
        # reproducible; HNSW recall is not (Kuzu's graph construction varies
        # across rebuilds of the same corpus), so it stays under measurements.
        deterministic_modes["quantized"]["recall_at_k"] = measured_recall["quantized"]
        exact_bytes = _resident_index_bytes(exact_store)
        deterministic_modes["exact"] = {
            "group_type": _group_type_name(exact_store, dimension, version_tag),
            "engaged": True,
            "recall_at_k": 1.0,
            "recall_note": "exact mode is the ground truth; recall@k is 1.0 by definition",
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
        "modes": deterministic_modes,
    }
    measurements = {
        "latency_ms": timing_modes,
        "ann_recall_at_k": measured_recall["ann"],
        "ann_backing_store_bytes": _directory_bytes(ann_path),
    }
    return deterministic, measurements


def run_vector_scale_lane(
    workdir: Path,
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    dimension: int = DEFAULT_DIMENSION,
    ann_threshold: int = DEFAULT_ANN_THRESHOLD,
    query_count: int = DEFAULT_QUERY_COUNT,
    top_k: int = DEFAULT_TOP_K,
    latency_passes: int = DEFAULT_LATENCY_PASSES,
) -> dict[str, Any]:
    """Run the vector-scale lane in ``workdir`` and return one labeled report.

    ``sizes`` are corpus sizes to run (ascending order is enforced); the
    roadmap exit criterion is evaluated at the largest size actually run and
    reported as ``not_evaluated_at_target_scale`` below ``TARGET_SCALE``.
    """
    if not sizes:
        raise ValueError("sizes must include at least one corpus size")
    if any(size < 1 for size in sizes):
        raise ValueError("sizes must be positive")
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
    ann_threshold: int,
    query_count: int,
    top_k: int,
    latency_passes: int,
) -> dict[str, Any]:
    from zaxy.embedding import HashEmbeddingProvider

    workdir.mkdir(parents=True, exist_ok=True)
    deterministic_sizes: dict[str, Any] = {}
    measurement_sizes: dict[str, Any] = {}
    for size in sizes:
        deterministic, measurements = await _run_size(
            workdir,
            size,
            dimension=dimension,
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
        recall = (
            mode_block["recall_at_k"]
            if mode == "quantized"
            else measurement_sizes[largest]["ann_recall_at_k"]
        )
        verdict = {
            "engaged": mode_block["engaged"],
            "recall_at_k": recall,
            "recall_pass": recall >= RECALL_FLOOR,
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
            "recall@k vs exact float64 ground truth, p50/p95 query latency, and "
            "resident index bytes for the embedded store's exact, Kuzu-HNSW, and "
            "int8-quantized vector modes over deterministic hash-embedded "
            "synthetic corpora. Timings are environment-dependent."
        ),
        "config": {
            "sizes": list(sizes),
            "dimension": dimension,
            "ann_threshold": ann_threshold,
            "query_count": query_count,
            "top_k": top_k,
            "latency_passes": latency_passes,
            "embedding_version_tag": HashEmbeddingProvider(dimension=dimension).version_tag,
            "recall_floor": RECALL_FLOOR,
            "target_scale": TARGET_SCALE,
        },
        "deterministic": {
            "note": (
                "Corpus hashes, exact/quantized recall, group types, and resident "
                "bytes are exactly reproducible across runs."
            ),
            "sizes": deterministic_sizes,
        },
        "measurements": {
            "note": (
                "Environment- or run-dependent values; excluded from determinism "
                "comparisons. ann_recall_at_k lives here because Kuzu's HNSW "
                "graph construction is not run-to-run reproducible: rebuilding "
                "the same corpus yields slightly different recall."
            ),
            "sizes": measurement_sizes,
        },
        "exit_criteria": {
            "evaluated_at_size": sizes[-1],
            "evaluated_at_target_scale": evaluated_at_target_scale,
            "modes": mode_verdicts,
            "note": (
                "quantized recall_pass and bytes_improved are deterministic; "
                "ann recall_pass varies slightly per run (HNSW rebuild "
                "nondeterminism) and latency_improved_p50 depends on the host "
                "machine. The roadmap criterion is defined at 10^5 vectors; "
                "smaller runs report not_evaluated_at_target_scale."
            ),
            "status": status,
        },
    }
