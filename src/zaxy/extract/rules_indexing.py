"""Rule extractors: document, code, source, and projection indexing events."""

from __future__ import annotations

from zaxy.event import Event
from zaxy.extract.core import (
    ExtractedEdge,
    ExtractedEntity,
    ExtractionResult,
    _document_session_context,
    _event_ref,
    _longmemeval_document_properties,
    _merge_properties,
    _neutral_audit_projection,
    _optional_text,
    _positive_int,
    _refresh_transform_properties,
    _retrieval_salience_properties,
    _source_sha256_property,
    register,
)
from zaxy.neutral import (
    neutral_document_record,
)


@register("document.indexed")
def _extract_document_indexed(event: Event) -> ExtractionResult:
    """Extract a cited document chunk from filesystem ingestion."""
    path = _optional_text(event.payload.get("path")) or "document"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    end_line = _positive_int(event.payload.get("end_line"), default=start_line)
    content = _optional_text(event.payload.get("content")) or ""
    sha256 = _optional_text(event.payload.get("sha256"))
    document_name = f"{path}:{start_line}-{end_line}"
    entity = ExtractedEntity(
        name=document_name,
        entity_type="document",
        observed_at=event.timestamp,
        summary=content,
        properties=_merge_properties(
            {
                "source_path": path,
                "source_start_line": start_line,
                "source_end_line": end_line,
                "source_sha256": sha256,
                **_refresh_transform_properties(event.payload),
            },
            _longmemeval_document_properties(event.payload),
            _retrieval_salience_properties(event.payload),
        )
        or {},
    )
    neutral = neutral_document_record(
        actor=event.actor,
        timestamp=event.timestamp,
        path=path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        source_event_ref=_event_ref(event),
        permission_scope=_optional_text(event.payload.get("permission_scope")),
        uncertainty=_optional_text(event.payload.get("uncertainty")),
        candidate_claim=_optional_text(event.payload.get("candidate_claim")),
    )
    neutral_entity = ExtractedEntity(
        name=neutral.substrate_id,
        entity_type="neutral_substrate",
        observed_at=event.timestamp,
        summary=neutral.quote,
        properties=neutral.to_properties(),
    )
    neutral_edge = ExtractedEdge(
        source=neutral.substrate_id,
        target=document_name,
        relation_type="neutral_substrate_cites_source",
        valid_from=event.timestamp,
    )
    audit_entity, audit_edge = _neutral_audit_projection(event, neutral.substrate_id)
    entities, edges = _document_session_context(
        event,
        document_name=document_name,
    )
    return ExtractionResult(
        entities=[entity, neutral_entity, *([audit_entity] if audit_entity is not None else []), *entities],
        edges=[neutral_edge, *([audit_edge] if audit_edge is not None else []), *edges],
        source_event_seq=event.seq,
    )


@register("code.file.indexed")
def _extract_code_file_indexed(event: Event) -> ExtractionResult:
    """Extract a code file inventory node from codebase indexing."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    sha256 = _optional_text(event.payload.get("sha256"))
    byte_count = _positive_int(event.payload.get("bytes"), default=0)
    line_count = _positive_int(event.payload.get("lines"), default=0)
    actor = ExtractedEntity(
        name=event.actor,
        entity_type="actor",
        observed_at=event.timestamp,
    )
    code_file = ExtractedEntity(
        name=path,
        entity_type="code_file",
        observed_at=event.timestamp,
        summary=f"{language} source file with {line_count} lines",
        properties={
            "source_path": path,
            "language": language,
            "source_sha256": sha256,
            "bytes": byte_count,
            "lines": line_count,
            **_refresh_transform_properties(event.payload),
        },
    )
    edge = ExtractedEdge(
        source=actor.name,
        target=code_file.name,
        relation_type="indexed_code_file",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[actor, code_file],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("source.discovered")
@register("source.changed")
@register("source.unchanged")
@register("source.deleted")
def _extract_source_refresh_event(event: Event) -> ExtractionResult:
    """Extract source freshness metadata from context refresh events."""
    path = _optional_text(event.payload.get("path")) or "source"
    source_kind = _optional_text(event.payload.get("source_kind")) or "unknown"
    sha256 = _optional_text(event.payload.get("sha256"))
    previous_sha256 = _optional_text(event.payload.get("previous_sha256"))
    byte_count = _positive_int(event.payload.get("bytes"), default=0)
    status = event.type.removeprefix("source.")
    refresh_properties = _refresh_transform_properties(event.payload)
    if refresh_reason := _optional_text(event.payload.get("refresh_reason")):
        refresh_properties["refresh_reason"] = refresh_reason
    entity = ExtractedEntity(
        name=path,
        entity_type="source",
        observed_at=event.timestamp,
        summary=f"{source_kind} source {path} {status}",
        properties=_merge_properties(
            {
                "source_path": path,
                "source_kind": source_kind,
                "source_sha256": sha256,
                "previous_sha256": previous_sha256,
                "bytes": byte_count,
                "refresh_status": status,
                **refresh_properties,
            },
            {},
        )
        or {},
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)


@register("projection.updated")
@register("projection.retired")
def _extract_projection_refresh_event(event: Event) -> ExtractionResult:
    """Extract projection lifecycle metadata from context refresh events."""
    path = _optional_text(event.payload.get("path")) or "source"
    source_kind = _optional_text(event.payload.get("source_kind")) or "unknown"
    projection = _optional_text(event.payload.get("projection")) or "memory"
    status = event.type.removeprefix("projection.")
    projection_properties = _refresh_transform_properties(event.payload)
    if source_sha256 := _optional_text(event.payload.get("source_sha256")):
        projection_properties["source_sha256"] = source_sha256
    source = ExtractedEntity(
        name=path,
        entity_type="source",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "source_kind": source_kind,
        },
    )
    projection_entity = ExtractedEntity(
        name=f"projection:{source_kind}:{path}",
        entity_type="projection",
        observed_at=event.timestamp,
        summary=f"{projection} projection {status} for {path}",
        properties={
            "source_path": path,
            "source_kind": source_kind,
            "projection": projection,
            "projection_status": status,
            "source_event": _optional_text(event.payload.get("source_event")),
            "reason": _optional_text(event.payload.get("reason")),
            **projection_properties,
        },
    )
    edge = ExtractedEdge(
        source=source.name,
        target=projection_entity.name,
        relation_type=f"projection_{status}",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[source, projection_entity],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.symbol.indexed")
def _extract_code_symbol_indexed(event: Event) -> ExtractionResult:
    """Extract a code symbol and connect it to the defining file."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    name = _optional_text(event.payload.get("name")) or "symbol"
    qualified_name = _optional_text(event.payload.get("qualified_name")) or name
    kind = _optional_text(event.payload.get("kind")) or "symbol"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    end_line = _positive_int(event.payload.get("end_line"), default=start_line)
    code_file = ExtractedEntity(
        name=path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "language": language,
        },
    )
    symbol = ExtractedEntity(
        name=f"{path}::{qualified_name}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        summary=f"{language} {kind} {qualified_name} defined in {path}:{start_line}-{end_line}",
        properties={
            "source_path": path,
            "language": language,
            "symbol_name": name,
            "qualified_name": qualified_name,
            "symbol_kind": kind,
            "source_start_line": start_line,
            "source_end_line": end_line,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    edge = ExtractedEdge(
        source=code_file.name,
        target=symbol.name,
        relation_type="defines_symbol",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[code_file, symbol],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.import.indexed")
def _extract_code_import_indexed(event: Event) -> ExtractionResult:
    """Extract a code import and connect it to the importing file."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    module = _optional_text(event.payload.get("module")) or "unknown"
    name = _optional_text(event.payload.get("name")) or module
    kind = _optional_text(event.payload.get("kind")) or "import"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    code_file = ExtractedEntity(
        name=path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "language": language,
        },
    )
    imported = ExtractedEntity(
        name=f"import:{module}:{name}",
        entity_type="code_import",
        observed_at=event.timestamp,
        summary=f"{language} {kind} {name} from {module} in {path}:{start_line}",
        properties={
            "source_path": path,
            "language": language,
            "module": module,
            "import_name": name,
            "import_kind": kind,
            "source_start_line": start_line,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    edge = ExtractedEdge(
        source=code_file.name,
        target=imported.name,
        relation_type="imports",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[code_file, imported],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.dependency.indexed")
def _extract_code_dependency_indexed(event: Event) -> ExtractionResult:
    """Extract a resolved local code dependency between files."""
    source_path = _optional_text(event.payload.get("source_path")) or "source-code-file"
    target_path = _optional_text(event.payload.get("target_path")) or "target-code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    module = _optional_text(event.payload.get("module")) or "unknown"
    import_name = _optional_text(event.payload.get("import_name")) or module
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    resolution = _optional_text(event.payload.get("resolution")) or "unknown"
    source_file = ExtractedEntity(
        name=source_path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": source_path,
            "language": language,
        },
    )
    target_file = ExtractedEntity(
        name=target_path,
        entity_type="code_file",
        observed_at=event.timestamp,
        properties={
            "source_path": target_path,
            "language": language,
        },
    )
    edge = ExtractedEdge(
        source=source_path,
        target=target_path,
        relation_type="depends_on_file",
        valid_from=event.timestamp,
    )
    dependency = ExtractedEntity(
        name=f"{source_path}->{target_path}:{start_line}",
        entity_type="code_dependency",
        observed_at=event.timestamp,
        summary=f"{source_path} imports {import_name} from {module} via {target_path}:{start_line}",
        properties={
            "source_path": source_path,
            "target_path": target_path,
            "language": language,
            "module": module,
            "import_name": import_name,
            "source_start_line": start_line,
            "resolution": resolution,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    return ExtractionResult(
        entities=[source_file, target_file, dependency],
        edges=[edge],
        source_event_seq=event.seq,
    )


@register("code.call.indexed")
def _extract_code_call_indexed(event: Event) -> ExtractionResult:
    """Extract a code call-site and resolved call edge when available."""
    path = _optional_text(event.payload.get("path")) or "code-file"
    language = _optional_text(event.payload.get("language")) or "unknown"
    caller = _optional_text(event.payload.get("caller")) or "caller"
    callee = _optional_text(event.payload.get("callee")) or "callee"
    callee_qualified_name = _optional_text(event.payload.get("callee_qualified_name")) or callee
    target_path = _optional_text(event.payload.get("target_path"))
    target_qualified_name = _optional_text(event.payload.get("target_qualified_name"))
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    resolution = _optional_text(event.payload.get("resolution")) or "unresolved"
    caller_symbol = ExtractedEntity(
        name=f"{path}::{caller}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        properties={
            "source_path": path,
            "language": language,
            "qualified_name": caller,
        },
    )
    call = ExtractedEntity(
        name=f"{path}::{caller}->{callee_qualified_name}:{start_line}",
        entity_type="code_call",
        observed_at=event.timestamp,
        summary=f"{caller} calls {callee_qualified_name} in {path}:{start_line}",
        properties={
            "source_path": path,
            "language": language,
            "caller": caller,
            "callee": callee,
            "callee_qualified_name": callee_qualified_name,
            "target_path": target_path,
            "target_qualified_name": target_qualified_name,
            "source_start_line": start_line,
            "resolution": resolution,
            **_source_sha256_property(event.payload),
            **_refresh_transform_properties(event.payload),
        },
    )
    entities = [caller_symbol, call]
    edges: list[ExtractedEdge] = []
    if target_path and target_qualified_name:
        target_symbol = ExtractedEntity(
            name=f"{target_path}::{target_qualified_name}",
            entity_type="code_symbol",
            observed_at=event.timestamp,
            properties={
                "source_path": target_path,
                "language": language,
                "qualified_name": target_qualified_name,
            },
        )
        entities.append(target_symbol)
        edges.append(
            ExtractedEdge(
                source=caller_symbol.name,
                target=target_symbol.name,
                relation_type="calls_symbol",
                valid_from=event.timestamp,
            )
        )
    return ExtractionResult(
        entities=entities,
        edges=edges,
        source_event_seq=event.seq,
    )


@register("code.coverage.indexed")
def _extract_code_coverage_indexed(event: Event) -> ExtractionResult:
    """Extract a test-to-production symbol coverage link."""
    test_path = _optional_text(event.payload.get("test_path")) or "test-code-file"
    test_name = _optional_text(event.payload.get("test_name")) or "test"
    test_qualified_name = _optional_text(event.payload.get("test_qualified_name")) or test_name
    target_path = _optional_text(event.payload.get("target_path")) or "target-code-file"
    target_name = _optional_text(event.payload.get("target_name")) or "target"
    target_qualified_name = _optional_text(event.payload.get("target_qualified_name")) or target_name
    language = _optional_text(event.payload.get("language")) or "unknown"
    start_line = _positive_int(event.payload.get("start_line"), default=1)
    resolution = _optional_text(event.payload.get("resolution")) or "unknown"
    test_symbol = ExtractedEntity(
        name=f"{test_path}::{test_qualified_name}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        properties={
            "source_path": test_path,
            "language": language,
            "symbol_name": test_name,
            "qualified_name": test_qualified_name,
            "symbol_kind": "test",
        },
    )
    target_symbol = ExtractedEntity(
        name=f"{target_path}::{target_qualified_name}",
        entity_type="code_symbol",
        observed_at=event.timestamp,
        properties={
            "source_path": target_path,
            "language": language,
            "symbol_name": target_name,
            "qualified_name": target_qualified_name,
        },
    )
    coverage = ExtractedEntity(
        name=f"{test_symbol.name}=>{target_symbol.name}:{start_line}",
        entity_type="code_coverage",
        observed_at=event.timestamp,
        summary=f"{test_qualified_name} tests {target_qualified_name} at {test_path}:{start_line}",
        properties={
            "test_path": test_path,
            "test_name": test_name,
            "test_qualified_name": test_qualified_name,
            "target_path": target_path,
            "target_name": target_name,
            "target_qualified_name": target_qualified_name,
            "language": language,
            "source_start_line": start_line,
            "resolution": resolution,
        },
    )
    edge = ExtractedEdge(
        source=test_symbol.name,
        target=target_symbol.name,
        relation_type="tests_symbol",
        valid_from=event.timestamp,
    )
    return ExtractionResult(
        entities=[test_symbol, target_symbol, coverage],
        edges=[edge],
        source_event_seq=event.seq,
    )
