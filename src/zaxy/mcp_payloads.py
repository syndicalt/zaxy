"""Pure payload/argument codecs for the MCP tool surface.

Small module-level functions that convert between MCP tool payload dicts and
the shared core contracts (contexts, checkouts, selectors), plus argument
validators shared by the tool handlers.

Extracted from :mod:`zaxy.mcp_server`, which re-imports every name here so
handler call sites and `from zaxy.mcp_server import ...` keep working.
"""

from __future__ import annotations

from typing import Any

from zaxy.context import Context
from zaxy.core import ContextAssembly, MemoryCheckout
from zaxy.export_view import ExportSelector
from zaxy.mcp_tool_specs import REASONING_PHASES
from zaxy.purpose import purpose_profile
from zaxy.security import validate_session_id
from zaxy.working_set import format_working_set


def _checkout_activity_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    token_efficiency = payload.get("token_efficiency")
    if isinstance(token_efficiency, dict):
        return {"token_efficiency": token_efficiency}
    return {}


def _activity_event_citation(event: Any) -> str | None:
    """Return the stable citation of a sealed memory-activity marker event."""
    thread = getattr(event, "thread", None)
    seq = getattr(event, "seq", None)
    event_hash = getattr(event, "hash", None)
    if (
        not isinstance(thread, str)
        or not isinstance(seq, int)
        or isinstance(seq, bool)
        or not isinstance(event_hash, str)
    ):
        return None
    return f"eventloom://{thread}/events/{seq}#{event_hash[:12]}"


def _context_assembly_from_payload(
    payload: dict[str, Any],
    *,
    replay_events: list[Any] | None = None,
) -> ContextAssembly:
    """Convert an MCP context payload into the shared core assembly contract."""
    contexts = [
        _context_from_payload(context)
        for context in payload.get("contexts", [])
        if isinstance(context, dict)
    ]
    warnings = payload.get("warnings")
    assembly_policy = payload.get("assembly_policy")
    counts = payload.get("context_counts")
    working_set = payload.get("working_set")
    return ContextAssembly(
        session_id=str(payload.get("session_id") or "default"),
        prompt=str(payload.get("prompt") or ""),
        contexts=contexts,
        replay_event_count=int(payload.get("replay_event_count") or 0),
        compacted=payload.get("compacted") is True,
        warnings=list(warnings) if isinstance(warnings, list) else [],
        assembly_policy=assembly_policy if isinstance(assembly_policy, dict) else {},
        context_counts=counts if isinstance(counts, dict) else {},
        working_set=working_set if isinstance(working_set, dict) else {},
        replay_events=list(replay_events) if replay_events else [],
    )


def _memory_checkout_from_payload(payload: dict[str, Any]) -> MemoryCheckout:
    """Convert an MCP checkout payload into the shared core checkout contract."""
    return MemoryCheckout(
        session_id=validate_session_id(str(payload.get("session_id") or "default")),
        query=str(payload.get("query") or ""),
        prompt=str(payload.get("prompt") or ""),
        working_set=_dict_payload(payload.get("working_set")),
        ref=_optional_dict_payload(payload.get("ref")),
        current_facts=_dict_list_payload(payload.get("current_facts")),
        evidence=_dict_list_payload(payload.get("evidence")),
        provenance=_dict_list_payload(payload.get("provenance")),
        retention=_dict_payload(payload.get("retention")),
        warnings=_string_payload_list(payload.get("warnings")),
        guidance=_dict_payload(payload.get("guidance")),
        quality=_dict_payload(payload.get("quality")),
        diagnostics=_dict_payload(payload.get("diagnostics")),
        context_counts=_int_dict_payload(payload.get("context_counts")),
        replay_event_count=_int_payload(payload.get("replay_event_count")),
        compacted=payload.get("compacted") is True,
        assembly_policy=_dict_payload(payload.get("assembly_policy")),
        purpose=_dict_payload(payload.get("purpose")),
    )


def _require_synthesis_row_in_checkout(checkout: MemoryCheckout, row: dict[str, Any]) -> None:
    """Ensure MCP row feedback refers to a row carried by this checkout."""
    identity = _synthesis_row_identity(row)
    if not any(identity.values()):
        raise ValueError("row must include fact_id, source_group, or citation")
    diagnostics = checkout.diagnostics if isinstance(checkout.diagnostics, dict) else {}
    synthesis = diagnostics.get("synthesis")
    if not isinstance(synthesis, dict):
        raise ValueError("checkout must include diagnostics.synthesis.ledger_rows")
    ledger_rows = synthesis.get("ledger_rows")
    if not isinstance(ledger_rows, list) or not ledger_rows:
        raise ValueError("checkout must include diagnostics.synthesis.ledger_rows")
    for ledger_row in ledger_rows:
        if isinstance(ledger_row, dict) and _synthesis_row_identity(ledger_row) == identity:
            return
    raise ValueError("row must match diagnostics.synthesis.ledger_rows")


def _synthesis_row_identity(row: dict[str, Any]) -> dict[str, str | None]:
    return {
        "fact_id": _optional_text(row.get("fact_id")),
        "source_group": _optional_text(row.get("source_group")),
        "citation": _optional_text(row.get("citation")),
    }


def _context_from_payload(payload: dict[str, Any]) -> Context:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return Context(
        content=str(payload.get("content") or ""),
        source=str(payload.get("source") or "unknown"),
        score=float(payload.get("score") or 0.0),
        valid_from=payload.get("valid_from")
        if isinstance(payload.get("valid_from"), str)
        else None,
        valid_to=payload.get("valid_to") if isinstance(payload.get("valid_to"), str) else None,
        metadata=metadata,
    )


def _contexts_as_of_seq(contexts: list[Context], as_of_seq: int) -> list[Context]:
    filtered = []
    for context in contexts:
        citation = _result_citation(context)
        seq, _event_hash = _citation_event_identity(citation)
        if seq is None or seq <= as_of_seq:
            filtered.append(context)
    return filtered


def _citation_event_identity(citation: str | None) -> tuple[int | None, str | None]:
    if not citation:
        return None, None
    event_seq: int | None = None
    event_hash: str | None = None
    if "/events/" in citation:
        tail = citation.split("/events/", 1)[1]
        seq_text = tail.split("#", 1)[0].split("/", 1)[0]
        if seq_text.isdigit():
            event_seq = int(seq_text)
    if "#" in citation:
        fragment = citation.rsplit("#", 1)[1]
        event_hash = fragment or None
    return event_seq, event_hash


def _format_prompt(events: list[Any], results: list[Any], *, working_set: Any | None = None) -> str:
    lines = []
    if working_set is not None:
        lines.extend([format_working_set(working_set), ""])
    lines.append("# Recent Events")
    for event in events:
        lines.append(f"[{event.seq}] {event.type} by {event.actor}")
        content = _event_content(event)
        if content:
            lines.append(content)
    lines.append("")
    lines.append("# Retrieved Context")
    for result in results:
        citation_value = _result_citation(result)
        citation = f" ({citation_value})" if citation_value else ""
        lines.append(f"- {result.content}{citation}")
    return "\n".join(lines).strip()


def _event_content(event: Any) -> str:
    payload = getattr(event, "payload", {})
    if not isinstance(payload, dict):
        return ""
    parts = [
        str(payload[key])
        for key in ("title", "summary", "content", "text", "decision", "task")
        if payload.get(key)
    ]
    return " ".join(parts)


def _context_payload(result: Any) -> dict[str, Any]:
    metadata = getattr(result, "metadata", None) or {}
    return {
        "content": result.content,
        "source": result.source,
        "score": result.score,
        "valid_from": result.valid_from,
        "valid_to": result.valid_to,
        "citation": _result_citation(result),
        "score_explanation": metadata.get("score_explanation")
        or getattr(result, "score_explanation", None),
        "metadata": metadata,
    }


def _context_assembly_payload(assembly: ContextAssembly) -> dict[str, Any]:
    """Serialize a fabric ContextAssembly into the context-tool output payload."""
    return {
        "session_id": assembly.session_id,
        "prompt": assembly.prompt,
        "contexts": [_context_payload(context) for context in assembly.contexts],
        "replay_event_count": assembly.replay_event_count,
        "compacted": assembly.compacted,
        "assembly_policy": assembly.assembly_policy,
        "context_counts": assembly.context_counts,
        "working_set": assembly.working_set,
    }


def _query_context_payload(context: Context) -> dict[str, Any]:
    """Flatten a Context into the memory_query output contract.

    Lifts citation/score_explanation out of ``metadata`` to top level so the
    shared fabric query path preserves the historical memory_query result shape.
    """
    metadata = context.metadata or {}
    return {
        "content": context.content,
        "source": context.source,
        "score": context.score,
        "valid_from": context.valid_from,
        "valid_to": context.valid_to,
        "citation": _result_citation(context),
        "score_explanation": metadata.get("score_explanation"),
    }


def _context_from_query_result(result: Any) -> Context:
    metadata: dict[str, Any] = {}
    citation = getattr(result, "citation", None)
    if citation:
        metadata["citation"] = citation
    score_explanation = getattr(result, "score_explanation", None)
    if score_explanation:
        metadata["score_explanation"] = score_explanation
    entity_name = getattr(result, "entity_name", None)
    if isinstance(entity_name, str) and entity_name:
        metadata["entity_name"] = entity_name
    entity_type = getattr(result, "entity_type", None)
    if isinstance(entity_type, str) and entity_type:
        metadata["entity_type"] = entity_type
    return Context(
        content=result.content,
        source=result.source,
        score=result.score,
        valid_from=result.valid_from,
        valid_to=result.valid_to,
        metadata=metadata or None,
    )


def _result_citation(result: Any) -> str | None:
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        metadata_citation = metadata.get("citation")
        if isinstance(metadata_citation, str):
            return metadata_citation
    citation = getattr(result, "citation", None)
    return citation if isinstance(citation, str) else None


def _normalize_feedback(feedback: object) -> str:
    normalized = str(feedback).casefold().strip()
    if normalized not in {"used", "helpful", "irrelevant"}:
        raise ValueError("feedback must be one of: used, helpful, irrelevant")
    return normalized


def _dict_payload(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_dict_payload(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _dict_list_payload(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_payload_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _int_dict_payload(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _int_payload(value: object) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _skill_event_type(action: object) -> str:
    normalized = str(action).casefold().strip()
    allowed = {
        "proposed",
        "validated",
        "revised",
        "deprecated",
        "contradicted",
        "applied",
        "outcome_recorded",
    }
    if normalized not in allowed:
        raise ValueError("skill action must be one of: " + ", ".join(sorted(allowed)))
    return f"skill.{normalized}"


def _required_text(value: object, field: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{field} is required")
    return text


def _required_strict_text(value: object, field: str) -> str:
    text = _optional_strict_text(value, field)
    if text is None:
        raise ValueError(f"{field} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fok_probe_text(query: str, cues: object) -> str:
    """Combine the query with optional cue field values for the FoK probe.

    Cues are an object of string fields (mission, workspace, tool, phase,
    ...); their values are probed against the session index as additional
    query terms, so a cue naming a known entity raises the verdict and an
    unknown cue honestly dilutes it.
    """
    if cues is None:
        return query
    if not isinstance(cues, dict):
        raise ValueError("cues must be an object of string fields")
    values: list[str] = []
    for key, value in cues.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"cues[{key!r}] must be a non-empty string")
        values.append(value.strip())
    if not values:
        return query
    return " ".join([query, *values])


def _optional_max_tokens(value: object) -> int | None:
    """Validate an optional non-negative integer prompt token budget."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_tokens must be an integer")
    if value < 0:
        raise ValueError("max_tokens must be >= 0")
    return value


def _optional_export_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _export_selector_from_arguments(arguments: dict[str, Any]) -> ExportSelector:
    """Map memory_export tool arguments to an ExportSelector (which validates)."""
    kwargs: dict[str, Any] = {}
    if arguments.get("grains") is not None:
        kwargs["grains"] = frozenset(_optional_text_list(arguments.get("grains")))
    if arguments.get("kinds") is not None:
        kwargs["kinds"] = frozenset(_optional_text_list(arguments.get("kinds")))
    if arguments.get("exclude_sensitivities") is not None:
        kwargs["exclude_sensitivities"] = frozenset(
            _optional_text_list(arguments.get("exclude_sensitivities"))
        )
    for field_name in ("since_seq", "max_seq", "limit"):
        resolved = _optional_export_int(arguments.get(field_name), field_name)
        if resolved is not None:
            kwargs[field_name] = resolved
    query_limit = _optional_export_int(arguments.get("query_limit"), "query_limit")
    if query_limit is not None:
        kwargs["query_limit"] = query_limit
    for field_name in ("since_time", "until_time", "query"):
        resolved_text = _optional_text(arguments.get(field_name))
        if resolved_text is not None:
            kwargs[field_name] = resolved_text
    return ExportSelector(**kwargs)


def _optional_strict_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    return text or None


def _validate_consolidation_window_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("window_size must be an integer")
    if value < 1 or value > 200:
        raise ValueError("window_size must be between 1 and 200")
    return value


def _validate_reasoning_phase(value: object, *, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("phase must be a string")
    phase = value.strip()
    if phase not in REASONING_PHASES:
        raise ValueError("phase must be one of: " + ", ".join(REASONING_PHASES))
    return phase


def _validate_reasoning_confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("confidence must be a number")
    confidence = float(value)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


def _validate_reasoning_source_events(value: object) -> list[dict[str, int | str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("source_events must be a non-empty array")
    source_events: list[dict[str, int | str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"source_events[{index}] must be an object")
        seq = item.get("seq")
        event_hash = item.get("hash")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError(f"source_events[{index}].seq must be a positive integer")
        if not isinstance(event_hash, str) or len(event_hash) != 64:
            raise ValueError(f"source_events[{index}].hash must be 64 lowercase hex characters")
        if any(char not in "0123456789abcdef" for char in event_hash):
            raise ValueError(f"source_events[{index}].hash must be 64 lowercase hex characters")
        source_events.append({"seq": seq, "hash": event_hash})
    return source_events


def _purpose_payload(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str | dict):
        return purpose_profile(value).to_dict()
    raise ValueError("purpose must be a profile name or object")


def _optional_text_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    texts: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text is not None:
            texts.append(text)
    return texts


def _approval_decisions(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("decisions must be an array")
    decisions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each approval decision must be an object")
        decisions.append(item)
    return decisions


def _fleet_source_events(value: object) -> list[dict[str, Any]]:
    """Validate fleet_promote source_events into cited ``{seq, hash}`` refs."""
    if not isinstance(value, list) or not value:
        raise ValueError("source_events must be a non-empty array of {seq, hash} objects")
    refs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each source_events entry must be an object")
        seq = item.get("seq")
        event_hash = item.get("hash")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 1:
            raise ValueError("source_events entry seq must be a positive integer")
        if not isinstance(event_hash, str) or not event_hash:
            raise ValueError("source_events entry hash must be a non-empty string")
        refs.append({"seq": seq, "hash": event_hash})
    return refs


def _coordination_result_payload(result: Any, event_type: str) -> dict[str, Any]:
    """Return stable JSON for coordination write results."""
    payload = {
        "event_type": event_type,
        "seq": result.event.seq,
        "hash": result.event.hash,
        "mission_id": result.mission_id,
        "worker_id": result.worker_id,
        "finding_id": result.finding_id,
        "handoff_id": result.handoff_id,
        "summary": result.summary,
        "evidence": result.evidence,
        "next_steps": result.event.payload.get("next_steps"),
        "risks": result.event.payload.get("risks"),
    }
    return {key: value for key, value in payload.items() if value is not None and value != []}
