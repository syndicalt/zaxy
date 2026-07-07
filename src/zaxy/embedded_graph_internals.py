"""Internal indexes and helpers for the embedded LadybugDB graph store."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import queue
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from zaxy.graph import (
        GraphEntity,
        SearchResult,
    )


logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


_KEYWORD_STOP_WORDS = frozenset(
    {
        "am",
        "and",
        "are",
        "at",
        "did",
        "do",
        "does",
        "first",
        "for",
        "had",
        "have",
        "how",
        "in",
        "it",
        "me",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


@dataclass(frozen=True)
class _KeywordIndex:
    entities: list[GraphEntity]
    term_counts: list[Counter[str]]
    term_entity_ids: dict[str, tuple[int, ...]]
    term_idf: dict[str, float]
    document_length_norms: list[float]


VECTOR_INDEX_CACHE_MAX_ENTRIES = 8


LEGACY_EMBEDDING_VERSION = "legacy"


VECTOR_SEARCH_OVERSAMPLE = 4


_ANN_SHADOW_TABLE_PREFIX = "ZaxyVectorAnnShadow"


_ANN_INSERT_BATCH_SIZE = 1024


_ANN_DELTA_REBUILD_FRACTION = 0.1


_QUERY_PARAMETER_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


_INCOMPATIBLE_STORAGE_MARKER = "not a valid Lbug database file"


_PRE_LADYBUG_BACKUP_SUFFIX = ".pre-ladybug.bak"


@dataclass(frozen=True)
class _VectorGroup:
    """Unit-normalized embedding matrix for one (dimension, version) group."""

    matrix: npt.NDArray[np.float64]
    entity_indexes: list[int]

    @property
    def matrix_bytes(self) -> int:
        return int(self.matrix.nbytes)


@dataclass(frozen=True)
class _QuantizedVectorGroup:
    """Int8-quantized unit vectors with per-vector scale factors.

    Original float vectors are not duplicated here: the exact rerank reads them
    from the ``embedding`` property of the entity objects the index already
    references, so quantization saves the full float64 matrix.
    """

    matrix: npt.NDArray[np.int8]
    scales: npt.NDArray[np.float64]
    entity_indexes: list[int]

    @property
    def matrix_bytes(self) -> int:
        return int(self.matrix.nbytes + self.scales.nbytes)


@dataclass(frozen=True)
class _AnnVectorGroup:
    """Engine-native HNSW-backed group; vectors are resident in the database."""

    table_name: str
    index_name: str
    dimension: int
    version: str
    session_id: str
    vector_count: int

    @property
    def matrix_bytes(self) -> int:
        return 0


@dataclass(frozen=True)
class _AnnGenerationState:
    """Resident shadow generation for one (session, version, dimension) scope.

    ``content_digest`` hashes the (entity_row, float32 vector) sequence the
    generation table holds, so a rebuild can prove the new corpus is an
    unchanged extension before riding the incremental insert path. The swap
    to a fresh generation replaces this record atomically (single assignment)
    before superseded generations are dropped.
    """

    table_name: str
    index_name: str
    generation: int
    vector_count: int
    content_digest: str


_AnyVectorGroup = _VectorGroup | _QuantizedVectorGroup | _AnnVectorGroup


@dataclass(frozen=True)
class _VectorIndex:
    entities: list[GraphEntity]
    groups: dict[tuple[int, str], _AnyVectorGroup]

    @property
    def matrix_bytes(self) -> int:
        return sum(group.matrix_bytes for group in self.groups.values())


@dataclass(frozen=True)
class _TraversalIndex:
    adjacency: dict[str, list[tuple[str, GraphEntity, str]]]
    keys_by_name: dict[str, set[str]]


def _adjacency_signature(
    session_id: str,
    node_ids: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> str:
    """Hash the snapshot content so the signature changes exactly with the graph.

    Nodes and directed edges are hashed in their (already deterministic)
    sorted order with type/field separators so distinct graphs cannot collide
    by concatenation. Content hashing is the embedded analogue of the
    log-signature pattern: cached walk results keyed on this signature stay
    valid until the projected graph itself changes.
    """
    hasher = hashlib.sha256()
    hasher.update(session_id.encode("utf-8"))
    for node_id in node_ids:
        hasher.update(b"\x00n")
        hasher.update(node_id.encode("utf-8"))
    for source_key, target_key in edges:
        hasher.update(b"\x00e")
        hasher.update(source_key.encode("utf-8"))
        hasher.update(b"\x00>")
        hasher.update(target_key.encode("utf-8"))
    return f"adjacency:sha256:{hasher.hexdigest()}"


def _node_key(session_id: str, entity_type: str, name: str, source_event_seq: int) -> str:
    return f"{session_id}\x1f{entity_type}\x1f{name}\x1f{source_event_seq}"


def _event_key(session_id: str, seq: int) -> str:
    return f"{session_id}\x1f{seq}"


def _entity_properties_json(entity: Any) -> str:
    properties = dict(entity.properties or {})
    if entity.embedding is not None:
        properties["embedding"] = entity.embedding
    return json.dumps(properties, sort_keys=True)


def _first_count(rows: list[list[Any]]) -> int:
    return int(rows[0][0]) if rows else 0


def _is_missing_projection_table_error(exc: RuntimeError) -> bool:
    message = str(exc)
    missing_tables = ("Table Entity does not exist", "Table Event does not exist", "Table NEXT_EVENT does not exist")
    return any(table in message for table in missing_tables)


def _engine_string_literal(value: str) -> str:
    """Render a python string as an engine string literal, byte-faithfully."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _armor_json_shaped_string_parameters(
    query: str,
    parameters: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Inline JSON-shaped string parameters as escaped string literals.

    LadybugDB 0.17.1's parameter binding silently converts a bound string
    whose first character is ``{`` or ``[`` — and that parses as JSON — into
    a STRUCT/LIST value, so the engine stores its own re-rendering instead of
    the original bytes (``'{"a": 1}'`` comes back as ``'{a: 1}'``; verified
    write-side via ``size()``). That corrupts ``properties_json``,
    ``evidence_json``, and any user text of that shape. Until the fork
    offers a raw-string binding, every string parameter the sniffer could
    touch is inlined as an escaped string literal instead — the same
    technique the binding library itself uses for its ``to_json()``
    rewriting — which round-trips byte-exactly (verified for backslashes,
    quotes, raw newlines, unicode, and nested JSON payloads).
    """
    sniffable = {
        key: value
        for key, value in parameters.items()
        if isinstance(value, str) and value[:1] in ("{", "[")
    }
    if not sniffable:
        return query, parameters
    remaining = dict(parameters)
    for key, value in sniffable.items():
        literal = _engine_string_literal(value)

        def _as_literal(_match: re.Match[str], _literal: str = literal) -> str:
            # A callable replacement keeps backslashes in the literal verbatim.
            return _literal

        query = re.sub(rf"\${re.escape(key)}\b", _as_literal, query)
        del remaining[key]
    return query, remaining


def _is_incompatible_storage_error(exc: RuntimeError) -> bool:
    """Match LadybugDB's refusal of a database file it cannot read.

    LadybugDB 0.17.1 raises ``RuntimeError: Runtime exception: Unable to open
    database. The file is not a valid Lbug database file!`` for a Kuzu-format
    (or otherwise foreign) file — verified against a real kuzu-0.11.3
    artifact. Genuinely corrupted files match too; moving aside and rebuilding
    from the Eventloom log is the correct remediation for both.
    """
    return _INCOMPATIBLE_STORAGE_MARKER in str(exc)


# Markers for a structurally damaged store (vs an unreadable-format one). The
# common case is a dirty/uncheckpointed WAL left by an uncleanly-killed owner,
# which surfaces as a LadybugDB C++ assertion on WAL replay. The store is
# derived state, so the correct remediation is identical to the incompatible
# case: move the artifact aside (never delete) and rebuild from the Eventloom
# log. The lock-contention error is deliberately NOT matched here — that is
# handled by graph-degraded fallback, not by moving the store aside.
_CORRUPT_STORE_MARKERS = (
    "UNREACHABLE_CODE",
    "wal_record",
    "Assertion failed",
)


def _is_corrupt_store_error(exc: RuntimeError) -> bool:
    """Match a structurally corrupt embedded store (e.g. dirty/broken WAL)."""
    message = str(exc)
    if "Could not set lock" in message:
        return False
    return any(marker in message for marker in _CORRUPT_STORE_MARKERS)


class EmbeddedProjectionLockedError(RuntimeError):
    """The embedded projection's exclusive write lock is held by another process.

    Raised in lieu of blocking forever when the LadybugDB store cannot acquire
    its single-writer lock within the configured timeout, or when the engine
    itself reports the lock is held. Carries enough context for the MCP/CLI
    layers to reap a verified stale owner, retry once, or degrade to the
    graph-degraded (null) projection backend instead of hanging the tool call.
    """

    def __init__(
        self,
        *,
        reason: str,
        operation: str,
        timeout_seconds: float | None = None,
    ) -> None:
        self.reason = reason
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        detail = (
            f"embedded projection is locked by another instance ({reason}); "
            f"failed during {operation}"
        )
        if timeout_seconds is not None:
            detail += f" after {timeout_seconds:g}s"
        super().__init__(detail)


def is_embedded_projection_lock_error(exc: BaseException) -> bool:
    """Return whether ``exc`` indicates the embedded projection lock is held.

    Matches the typed :class:`EmbeddedProjectionLockedError` and the underlying
    engine string (``"Could not set lock on file"`` on a ``.kuzu`` path) so the
    MCP/CLI degrade paths treat the bounded-timeout and engine-raise cases
    identically.
    """
    if isinstance(exc, EmbeddedProjectionLockedError):
        return True
    return "Could not set lock on file" in str(exc) and ".kuzu" in str(exc)


# Abandoned lock-op daemons cannot be killed (a blocked C extension call is
# uninterruptible), but they MUST be observable: each still holds whatever
# native resources the engine call acquired, and unbounded accumulation under
# repeated contention is itself a failure signal. The registry keeps a strong
# reference to each abandoned worker (so accounting survives thread finish
# ordering) and drops finished ones on every new abandonment.
_abandoned_lock_ops_lock = threading.Lock()
_abandoned_lock_ops: list[threading.Thread] = []
_abandoned_lock_ops_total = 0


def abandoned_lock_op_stats() -> dict[str, int]:
    """Return abandoned lock-op daemon accounting: still-live and lifetime totals."""
    with _abandoned_lock_ops_lock:
        live = sum(1 for worker in _abandoned_lock_ops if worker.is_alive())
        return {"live": live, "total": _abandoned_lock_ops_total}


def _record_abandoned_lock_op(worker: threading.Thread, operation: str) -> None:
    global _abandoned_lock_ops_total
    with _abandoned_lock_ops_lock:
        _abandoned_lock_ops[:] = [w for w in _abandoned_lock_ops if w.is_alive()]
        _abandoned_lock_ops.append(worker)
        _abandoned_lock_ops_total += 1
        live = len(_abandoned_lock_ops)
        total = _abandoned_lock_ops_total
    logger.warning(
        "embedded lock-op daemon abandoned after timeout during %s "
        "(%d still blocked, %d abandoned since process start); the thread cannot "
        "be interrupted while the engine call blocks and will exit with it",
        operation,
        live,
        total,
    )


async def await_blocking_with_timeout(
    func: Callable[[], Any],
    *,
    timeout: float,
    operation: str,
) -> Any:
    """Run a blocking call on a daemon thread, failing fast on a lock timeout.

    ``asyncio.wait_for`` cannot interrupt a blocking C extension call, and
    ``asyncio.to_thread`` runs on the default executor whose pool threads are
    non-daemon — a permanently blocked member would hang process shutdown. The
    blocking call is therefore executed on an explicit *daemon* thread whose
    result is awaited through a queue with a deadline. On timeout the daemon is
    abandoned (it cannot block exit) but recorded in the abandoned-lock-op
    registry so accumulation is observable, and
    :class:`EmbeddedProjectionLockedError` is raised so callers can reap a
    stale owner, retry, or degrade instead of hanging the event loop
    indefinitely. On completion the worker is joined so the common path leaks
    nothing.
    """
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def _runner() -> None:
        try:
            result_queue.put(("ok", func()))
        except BaseException as exc:  # noqa: BLE001 - propagate every engine error
            result_queue.put(("err", exc))

    worker = threading.Thread(target=_runner, name="zaxy-embedded-lock-op", daemon=True)
    worker.start()
    loop = asyncio.get_event_loop()
    try:
        outcome = await loop.run_in_executor(
            None, result_queue.get, True, timeout
        )
    except queue.Empty:
        _record_abandoned_lock_op(worker, operation)
        raise EmbeddedProjectionLockedError(
            reason="acquisition-timeout",
            operation=operation,
            timeout_seconds=timeout,
        ) from None
    # The queue item is only put after func() returns/raises, so the worker is
    # finishing; a bounded join reclaims it without risking a stall.
    worker.join(timeout=1.0)
    kind, value = outcome
    if kind == "ok":
        return value
    if is_embedded_projection_lock_error(value):
        raise EmbeddedProjectionLockedError(
            reason="engine-reported-held", operation=operation
        ) from value
    raise value


def pre_ladybug_backup_paths(path: Path) -> list[Path]:
    """Return existing pre-LadybugDB backup artifacts for one projection path.

    Covers the primary ``<name>.pre-ladybug.bak`` artifact plus its ``.wal``
    sibling and any numbered variants from repeated migrations.
    """
    if not path.parent.exists():
        return []
    return sorted(path.parent.glob(f"{path.name}{_PRE_LADYBUG_BACKUP_SUFFIX}*"))


def _claim_backup_path(path: Path) -> Path:
    """Atomically claim a free backup name with ``O_CREAT | O_EXCL``.

    A bare exists()-then-replace scan is a TOCTOU: two processes self-healing
    the same projection can pick the same ``.bak`` name, and ``Path.replace``
    overwrites without re-checking — silently clobbering the other's backup.
    Creating the placeholder exclusively makes name selection race-free: the
    kernel guarantees only one claimant wins each candidate. The claimed
    placeholder is immediately overwritten by the caller's atomic ``replace``.
    The ``.wal`` sibling name is derived from the claimed primary, so any
    process following this protocol can never collide on it either.
    """
    backup_path = path.with_name(path.name + _PRE_LADYBUG_BACKUP_SUFFIX)
    counter = 1
    while True:
        wal_sibling = backup_path.with_name(backup_path.name + ".wal")
        if not wal_sibling.exists():
            try:
                fd = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
                return backup_path
        backup_path = path.with_name(f"{path.name}{_PRE_LADYBUG_BACKUP_SUFFIX}.{counter}")
        counter += 1


def _move_incompatible_store_aside(path: Path) -> Path:
    """Move an unreadable projection artifact (and its WAL) to a .bak path.

    User data is never deleted: the backup name is claimed exclusively (see
    :func:`_claim_backup_path`), so neither a previous migration's backup nor
    a concurrent self-heal's backup can be overwritten.
    """
    backup_path = _claim_backup_path(path)
    path.replace(backup_path)
    wal_path = path.with_name(path.name + ".wal")
    if wal_path.exists():
        wal_path.replace(backup_path.with_name(backup_path.name + ".wal"))
    return backup_path


def _row_to_entity(row: list[Any]) -> GraphEntity:
    from zaxy.graph import GraphEntity

    properties = json.loads(row[5] or "{}")
    if row[4] is not None:
        properties.setdefault("summary", row[4])
    if len(row) > 7 and row[7] is not None:
        properties["source_event_seq"] = int(row[7])
    if len(row) > 8 and row[8] is not None:
        properties["source_event_hash"] = str(row[8])
    return GraphEntity(
        name=row[0],
        entity_type=row[1],
        valid_from=row[2],
        valid_to=row[3],
        properties=properties,
        session_id=row[6],
    )


def _entity_with_path_metadata(entity: GraphEntity, *, relation_types: list[str]) -> GraphEntity:
    from zaxy.graph import GraphEntity

    return GraphEntity(
        name=entity.name,
        entity_type=entity.entity_type,
        valid_from=entity.valid_from,
        valid_to=entity.valid_to,
        properties={
            **entity.properties,
            "_path_relation_types": relation_types,
            "_path_length": len(relation_types),
        },
        session_id=entity.session_id,
    )


def _causal_edge_metadata_from_row(
    row: Any,
    *,
    source_entity: GraphEntity,
    target_entity: GraphEntity,
) -> dict[str, Any] | None:
    evidence = _json_dict(row[25])
    source_event_seq = row[23]
    source_event_hash = str(row[24] or "")
    relation_type = str(row[10])
    confidence = _optional_float(row[21])
    cited_seq = _optional_int(row[23])
    if confidence is None or cited_seq is None or not source_event_hash or not evidence:
        return None
    citation = _edge_citation(source_entity.session_id, source_event_seq, source_event_hash)
    return {
        "causal_source_name": source_entity.name,
        "causal_source_type": source_entity.entity_type,
        "causal_target_name": target_entity.name,
        "causal_target_type": target_entity.entity_type,
        "relation_type": relation_type,
        "graph_relation_type": relation_type,
        "causal_relation_type": evidence.get("causal_relation_type") or relation_type.removeprefix("causal_"),
        "confidence": confidence,
        "inference_method": str(row[22] or "unknown"),
        "citation": citation,
        "review_status": evidence.get("review_status") or "proposed",
        "authority_status": evidence.get("authority_status") or "non_authoritative",
        "source_event_seq": cited_seq,
        "source_event_hash": source_event_hash or None,
        "evidence": evidence,
        "session_id": source_entity.session_id,
    }


def _entity_with_causal_metadata(
    entity: GraphEntity,
    *,
    edge_metadata: dict[str, Any],
    path_relation_types: list[str],
    path_citations: list[str],
) -> GraphEntity:
    from zaxy.graph import GraphEntity

    return GraphEntity(
        name=entity.name,
        entity_type=entity.entity_type,
        valid_from=entity.valid_from,
        valid_to=entity.valid_to,
        properties={
            **entity.properties,
            **edge_metadata,
            "_path_relation_types": path_relation_types,
            "_path_citations": path_citations,
            "_path_length": len(path_relation_types),
        },
        session_id=entity.session_id,
    )


def _edge_citation(session_id: str, source_event_seq: Any, source_event_hash: str) -> str:
    if source_event_seq is not None and source_event_hash:
        return f"eventloom://{session_id}/events/{source_event_seq}#{source_event_hash[:12]}"
    return "eventloom://unknown/events/unknown#unknown"


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entity_keyword_text(entity: GraphEntity) -> str:
    values = [entity.name, entity.entity_type]
    for key, value in entity.properties.items():
        if key == "embedding" or key.startswith("_"):
            continue
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, int | float | bool):
            values.append(str(value))
    return " ".join(values)


def _terms(text: str) -> list[str]:
    return [term for term in _TOKEN_RE.findall(text.casefold()) if len(term) > 1]


def _keyword_query_terms(text: str) -> list[str]:
    return [term for term in _terms(text) if term not in _KEYWORD_STOP_WORDS]


def _keyword_index_from_entities(entities: list[GraphEntity]) -> _KeywordIndex:
    term_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    document_lengths: list[int] = []
    total_document_length = 0
    for entity in entities:
        terms = _terms(_entity_keyword_text(entity))
        counts = Counter(terms)
        term_counts.append(counts)
        length = len(terms)
        document_lengths.append(length)
        total_document_length += length
        document_frequency.update(counts.keys())
    average_length = total_document_length / len(term_counts) if term_counts else 0.0
    return _KeywordIndex(
        entities=entities,
        term_counts=term_counts,
        term_entity_ids=_term_entity_ids(term_counts),
        term_idf=_term_idf(document_frequency, len(term_counts)),
        document_length_norms=_document_length_norms(document_lengths, average_length),
    )


def _keyword_candidate_terms(
    index: _KeywordIndex,
    query_terms: list[str],
    *,
    max_candidates: int = 1000,
    min_terms: int = 4,
) -> list[str]:
    unique_terms = list(dict.fromkeys(query_terms))
    if not unique_terms:
        return []
    if _candidate_union_size_at_most(index, unique_terms, max_candidates):
        return unique_terms

    selected: list[str] = []
    selected_candidates: set[int] = {*()}
    sorted_terms = sorted(
        unique_terms,
        key=lambda term: (
            len(index.term_entity_ids.get(term, ())),
            -index.term_idf.get(term, 0.0),
        ),
    )
    for term in sorted_terms:
        postings = index.term_entity_ids.get(term, ())
        if not postings:
            continue
        remaining_capacity = max_candidates - len(selected_candidates)
        new_candidate_count = _new_candidate_count_until_overflow(
            postings,
            selected_candidates,
            max_new_candidates=remaining_capacity,
        )
        if len(selected) < min_terms or len(selected_candidates) + new_candidate_count <= max_candidates:
            selected.append(term)
            selected_candidates.update(postings)
    selected_lookup = {*selected}
    return [term for term in unique_terms if term in selected_lookup]


def _candidate_union_size_at_most(index: _KeywordIndex, terms: list[str], max_candidates: int) -> bool:
    candidates: set[int] = {*()}
    for term in terms:
        for entity_index in index.term_entity_ids.get(term, ()):
            candidates.add(entity_index)
            if len(candidates) > max_candidates:
                return False
    return True


def _new_candidate_count_until_overflow(
    postings: Sequence[int],
    selected_candidates: set[int],
    *,
    max_new_candidates: int,
) -> int:
    new_candidate_count = 0
    for entity_index in postings:
        if entity_index in selected_candidates:
            continue
        new_candidate_count += 1
        if new_candidate_count > max_new_candidates:
            return new_candidate_count
    return new_candidate_count


def _term_entity_ids(term_counts: list[Counter[str]]) -> dict[str, tuple[int, ...]]:
    postings: dict[str, list[int]] = {}
    for index, counts in enumerate(term_counts):
        for term in counts:
            postings.setdefault(term, []).append(index)
    return {term: tuple(indices) for term, indices in postings.items()}


def _term_idf(document_frequency: Counter[str], document_count: int) -> dict[str, float]:
    if document_count <= 0:
        return {}
    return {
        term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }


def _document_length_norms(
    document_lengths: list[int],
    average_length: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if average_length <= 0:
        return [0.0 for _ in document_lengths]
    return [k1 * (1 - b + b * document_length / average_length) for document_length in document_lengths]


def _bm25_score_from_precomputed(
    query_terms: Sequence[str],
    term_counts: Counter[str],
    *,
    document_length_norm: float,
    term_idf: dict[str, float],
    k1: float = 1.5,
) -> float:
    if document_length_norm <= 0:
        return 0.0
    score = 0.0
    for term in query_terms:
        term_frequency = term_counts[term]
        if term_frequency <= 0:
            continue
        denominator = term_frequency + document_length_norm
        score += term_idf.get(term, 0.0) * (term_frequency * (k1 + 1)) / denominator
    return score


def _ann_scope_digest(session_id: str, version: str) -> str:
    """Identifier-safe digest naming one (session, version) shadow scope."""
    return hashlib.sha256(f"{session_id}\x1f{version}".encode()).hexdigest()[:16]


def _ann_content_digest(
    entity_rows: npt.NDArray[np.int64],
    matrix: npt.NDArray[np.float32],
) -> str:
    """Hash a shadow generation's (entity_row, float32 vector) content.

    Hashed in the exact float32 representation the shadow table stores, so
    the digest of a leading slice of a new corpus proves the resident rows
    (values *and* entity-row mapping) are unchanged before the delta policy
    chooses the incremental insert path over a generation swap.
    """
    hasher = hashlib.sha256()
    hasher.update(np.int64(entity_rows.shape[0]).tobytes())
    hasher.update(np.ascontiguousarray(entity_rows).tobytes())
    hasher.update(np.ascontiguousarray(matrix).tobytes())
    return hasher.hexdigest()


def _quantized_candidate_entity_indexes(
    group: _QuantizedVectorGroup,
    unit_query: npt.NDArray[np.float64],
    *,
    limit: int,
) -> list[int]:
    """Select oversampled candidate entities with int8 dot products."""
    query_max = float(np.abs(unit_query).max())
    if query_max == 0.0:
        return []
    quantized_query = np.clip(np.rint(unit_query * (127.0 / query_max)), -127, 127).astype(np.int8)
    integer_scores = group.matrix.astype(np.int32) @ quantized_query.astype(np.int32)
    approximate_scores = integer_scores.astype(np.float64) * group.scales
    candidate_count = min(approximate_scores.size, limit * VECTOR_SEARCH_OVERSAMPLE)
    if candidate_count <= 0:
        return []
    if candidate_count < approximate_scores.size:
        candidate_rows = np.argpartition(-approximate_scores, candidate_count - 1)[:candidate_count]
    else:
        candidate_rows = np.arange(approximate_scores.size)
    return sorted(group.entity_indexes[int(row)] for row in candidate_rows)


def _exact_rerank_results(
    candidate_entity_indexes: Sequence[int],
    unit_query: npt.NDArray[np.float64],
    *,
    limit: int,
    entities: list[GraphEntity],
) -> list[SearchResult]:
    """Rerank approximate candidates with exact float64 scores.

    Shared by the ANN and int8 selectors: scores read the float64 vectors
    already resident on the indexed entities, so candidate selection never
    decides the final ordering. Candidates arrive in ascending entity order,
    which keeps the stable descending sort's tie behavior aligned with the
    dense path. Results stay ``exact=False`` because the candidate set is
    approximate.

    Candidate embeddings were validated when the vector index admitted them
    into a group, so the hot path converts them in one bulk pass (the
    per-candidate conversion loop measurably dominated high-dimension query
    latency); if a property has been mutated since admission, the rerank
    falls back to per-candidate validation and skips the bad rows.
    """
    from zaxy.graph import SearchResult

    if not candidate_entity_indexes:
        return []
    kept = list(candidate_entity_indexes)
    try:
        matrix = np.asarray(
            [entities[entity_index].properties["embedding"] for entity_index in kept],
            dtype=np.float64,
        )
        if matrix.ndim != 2:
            raise ValueError("candidate embeddings are not uniform vectors")
    except (KeyError, TypeError, ValueError):
        kept = []
        vectors: list[list[float]] = []
        for entity_index in candidate_entity_indexes:
            vector = _embedding_vector(entities[entity_index].properties.get("embedding"))
            if vector is None:
                continue
            kept.append(entity_index)
            vectors.append(vector)
        if not kept:
            return []
        matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    scores = np.zeros(len(kept), dtype=np.float64)
    nonzero_rows = norms > 0.0
    scores[nonzero_rows] = (matrix[nonzero_rows] / norms[nonzero_rows, np.newaxis]) @ unit_query
    positive_rows = np.flatnonzero(scores > 0.0)
    if positive_rows.size == 0:
        return []
    ordered_rows = positive_rows[np.argsort(-scores[positive_rows], kind="stable")]
    return [
        SearchResult(
            entity=entities[kept[int(row)]],
            score=float(scores[row]),
            source="vector",
            raw_score=float(scores[row]),
            exact=False,
        )
        for row in ordered_rows[:limit]
    ]


def _dense_vector_results(
    group: _VectorGroup,
    unit_query: npt.NDArray[np.float64],
    *,
    limit: int,
    entities: list[GraphEntity],
) -> list[SearchResult]:
    """Exact dense-matrix scoring; the pre-versioning behavior, unchanged."""
    from zaxy.graph import SearchResult

    scores = group.matrix @ unit_query
    positive_rows = np.flatnonzero(scores > 0.0)
    if positive_rows.size == 0:
        return []
    # Stable sort keeps first-projected entities ahead on score ties,
    # matching the previous heapq.nlargest behavior.
    ordered_rows = positive_rows[np.argsort(-scores[positive_rows], kind="stable")]
    results: list[SearchResult] = []
    for row in ordered_rows[:limit]:
        score = float(scores[row])
        results.append(
            SearchResult(
                entity=entities[group.entity_indexes[int(row)]],
                score=score,
                source="vector",
                raw_score=score,
                exact=True,
            )
        )
    return results


def _quantize_unit_matrix(
    matrix: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float64]]:
    """Quantize unit-vector rows to int8 with per-vector scale factors."""
    row_max = np.abs(matrix).max(axis=1)
    # Unit vectors always have a nonzero component; guard anyway so a zero row
    # quantizes to zeros instead of dividing by zero.
    safe_row_max = np.where(row_max == 0.0, 1.0, row_max)
    quantized = np.clip(
        np.rint(matrix * (127.0 / safe_row_max)[:, np.newaxis]),
        -127,
        127,
    ).astype(np.int8)
    scales = row_max / 127.0
    return quantized, scales


def _embedding_version(properties: dict[str, Any]) -> str:
    """Return the stored embedding version tag; absent means legacy."""
    version = properties.get("embedding_version")
    if isinstance(version, str) and version:
        return version
    return LEGACY_EMBEDDING_VERSION


def _embedding_vector(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    vector = []
    for item in value:
        if not isinstance(item, int | float):
            return None
        vector.append(float(item))
    return vector


def _json_dict(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _properties_reference_source(properties: dict[str, Any], source_path: str) -> bool:
    for key in ("source_path", "target_path", "test_path", "covered_path"):
        if properties.get(key) == source_path:
            return True
    return False
