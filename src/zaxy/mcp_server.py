"""MCP server exposing memory tools.

Provides stdio and SSE transport for agent frameworks to interact
with Zaxy via the Model Context Protocol.

Tools exposed:
- memory_append: Append a typed event to the log.
- memory_query: Query the temporal knowledge graph.
- memory_causal_successors: Read causal effects from the graph.
- memory_causal_predecessors: Read causal causes from the graph.
- memory_consolidation_candidate: Append a cited consolidation candidate.
- memory_consolidation_propose_from_log: Create review-pending candidates from log segments.
- memory_consolidation_status: Read review-gated consolidation status.
- memory_consolidation_review: Append a consolidation review.
- memory_explain_outcome: Explain an outcome with cited reasoning context.
- memory_propose_belief_update: Append a review-pending belief proposal through MemoryFabric.
- memory_claim_confidence: Score a claim against cited memory evidence.
- memory_similar_procedures: Retrieve similar procedures for reasoning reuse.
- memory_feedback: Record retrieval feedback for a graph entity.
- memory_synthesis_artifact: Persist checkout synthesis answer candidates and feedback.
- memory_synthesis_evidence: Record feedback for one synthesis ledger row.
- memory_replay: Replay events from a session.
- memory_invalidate: Mark a fact as superseded.
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import inspect
import json
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import anyio
import jwt
from mcp import types
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import TextContent, Tool

from zaxy.capabilities import build_memory_bootstrap, build_memory_capabilities
from zaxy.causal import causal_query_result_from_projection, causal_relation_to_graph_relation
from zaxy.config import get_settings
from zaxy.consolidation import (
    build_consolidation_candidate_event,
    build_consolidation_review_event,
)
from zaxy.context import Context, ContextAssemblyPolicy, context_counts
from zaxy.core import (
    ContextAssembly,
    MemoryCheckout,
    MemoryFabric,
    build_memory_checkout,
)
from zaxy.extract import extract
from zaxy.lifecycle import (
    build_session_ended_event,
    build_subagent_completed_event,
    build_tool_call_completed_event,
)
from zaxy.log import get_logger, setup_logging
from zaxy.mcp_runtime import EmbeddedMcpOwnerClaim, EmbeddedMcpRuntimeCoordinator
from zaxy.memory_persistence import record_memory_activity
from zaxy.metrics import get_metrics
from zaxy.pagination import encode_query_cursor, validate_query_cursor
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store
from zaxy.purpose import purpose_profile
from zaxy.query import QueryRouter, build_retention_policy
from zaxy.refs import MemoryRef, MemoryRefStore
from zaxy.remote_security import AuditEventExporter, RemoteAuditEvent, SessionRateLimiter
from zaxy.runtime import LocalEmbeddedGraphRuntime, LocalNeo4jRuntime, LocalPgGraphRuntime
from zaxy.security import (
    MAX_QUERY_LIMIT,
    MAX_REPLAY_EVENTS,
    validate_event_text,
    validate_from_seq,
    validate_limit,
    validate_payload,
    validate_query,
    validate_session_id,
    validate_traversal_depth,
)
from zaxy.session import SessionManager
from zaxy.synthesis_artifact import (
    build_synthesis_artifact,
    build_synthesis_candidate_event_payload,
    build_synthesis_evidence_event_payload,
    normalize_synthesis_outcome,
    synthesis_outcome_event_type,
)
from zaxy.trace import MemoryTracer
from zaxy.verbatim import VerbatimIndex
from zaxy.working_set import build_working_set, format_working_set
from zaxy.workspace import (
    WorkspaceProfile,
    build_session_genesis_event,
    build_workspace_instruction_event,
    existing_session_genesis_profile,
    existing_workspace_instructions_signature,
    mark_workspace_instruction_event_updated,
    workspace_profile_from_payload,
)

app = Server("zaxy-memory")
remote_session_scope: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "remote_session_scope",
    default=None,
)
REASONING_PHASES = ["planning", "execution", "review", "reflection"]


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

TOOLS = [
    Tool(
        name="memory_append",
        description="Append a typed event to the agent's persistent memory log.",
        inputSchema={
            "type": "object",
            "required": ["event_type", "actor", "payload"],
            "properties": {
                "event_type": {"type": "string", "description": "Event type, e.g. 'goal.created'"},
                "actor": {"type": "string", "description": "Actor that emitted the event"},
                "payload": {"type": "object", "description": "Structured payload"},
                "thread": {"type": "string", "description": "Logical thread / session ID (legacy, use session_id)"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_query",
        description="Query the temporal knowledge graph for relevant context.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "temporal_filter": {"type": "string", "description": "ISO-8601 point-in-time filter"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
                "session_id": {"type": "string", "description": "Session ID for scoped retrieval"},
                "cursor": {"type": "string", "description": "Opaque cursor from a prior paged memory_query call"},
                "paged": {"type": "boolean", "description": "Return contexts with pagination metadata"},
                "session_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Local-only explicit cross-session query scope",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_causal_successors",
        description="Read directed causal effects of an entity from graph-backed memory.",
        inputSchema={
            "type": "object",
            "required": ["entity_name"],
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name to start causal traversal from"},
                "relation_type": {"type": "string", "description": "Optional causal relation taxonomy label"},
                "depth": {"type": "integer", "description": "Traversal depth", "default": 2, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped causal retrieval"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_causal_predecessors",
        description="Read directed causal causes of an entity from graph-backed memory.",
        inputSchema={
            "type": "object",
            "required": ["entity_name"],
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name to start causal traversal from"},
                "relation_type": {"type": "string", "description": "Optional causal relation taxonomy label"},
                "depth": {"type": "integer", "description": "Traversal depth", "default": 2, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped causal retrieval"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_consolidation_candidate",
        description="Append a cited, review-pending consolidation candidate event.",
        inputSchema={
            "type": "object",
            "required": [
                "candidate_type",
                "title",
                "summary",
                "source_events",
                "confidence",
                "method",
            ],
            "properties": {
                "candidate_type": {
                    "type": "string",
                    "enum": ["episode", "claim", "procedure"],
                    "description": "Consolidation candidate type",
                },
                "title": {"type": "string", "description": "Candidate title"},
                "summary": {"type": "string", "description": "Candidate summary"},
                "source_events": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["seq", "hash"],
                        "properties": {
                            "seq": {"type": "integer", "minimum": 1},
                            "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                        "additionalProperties": False,
                    },
                    "description": "Cited Eventloom source events",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Candidate confidence from 0.0 to 1.0",
                },
                "method": {"type": "string", "description": "Consolidation method identifier"},
                "purpose": {"type": "string", "description": "Optional consolidation purpose"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
                "actor": {
                    "type": "string",
                    "description": "Actor recording the candidate",
                    "default": "zaxy-consolidation",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_consolidation_propose_from_log",
        description="Create non-authoritative consolidation candidates from Eventloom log segments.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to replay for proposal windows"},
                "actor": {
                    "type": "string",
                    "description": "Actor recording generated candidates",
                    "default": "zaxy-consolidation",
                },
                "purpose": {"type": "string", "description": "Optional consolidation purpose"},
                "window_size": {
                    "type": "integer",
                    "description": "Number of source events per proposal window",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_consolidation_status",
        description="Read review-gated consolidation candidate status.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to inspect"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_consolidation_review",
        description="Append a consolidation candidate review event without promoting authority.",
        inputSchema={
            "type": "object",
            "required": ["candidate_id", "status", "rationale"],
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "pattern": "^consolidation:(episode|claim|procedure):[0-9a-f]{24}$",
                    "description": "Consolidation candidate ID",
                },
                "status": {
                    "type": "string",
                    "enum": ["accepted", "rejected", "deferred", "conflicted"],
                    "description": "Review lifecycle status",
                },
                "rationale": {"type": "string", "description": "Review rationale"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
                "actor": {
                    "type": "string",
                    "description": "Actor recording the review",
                    "default": "zaxy-reviewer",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_explain_outcome",
        description="Explain an outcome from cited causal and checkout memory context.",
        inputSchema={
            "type": "object",
            "required": ["outcome"],
            "properties": {
                "outcome": {"type": "string", "description": "Outcome to explain"},
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "default": "planning",
                    "description": "Reasoning phase for purpose-conditioned retrieval",
                },
                "depth": {"type": "integer", "description": "Causal traversal depth", "default": 2, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_propose_belief_update",
        description="Append a cited, review-pending belief update proposal without promoting authority.",
        inputSchema={
            "type": "object",
            "required": ["claim", "rationale", "confidence", "source_events"],
            "properties": {
                "claim": {"type": "string", "description": "Claim to propose for review"},
                "rationale": {"type": "string", "description": "Cited rationale for the proposal"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Proposal confidence from 0.0 to 1.0",
                },
                "source_events": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["seq", "hash"],
                        "properties": {
                            "seq": {"type": "integer", "minimum": 1},
                            "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                        "additionalProperties": False,
                    },
                    "description": "Cited Eventloom source events supporting the proposal",
                },
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "default": "reflection",
                    "description": "Reasoning phase for purpose-conditioned proposal recording",
                },
                "actor": {
                    "type": "string",
                    "description": "Actor recording the proposal",
                    "default": "zaxy-reasoning",
                },
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_claim_confidence",
        description="Score claim confidence from cited support and conflict evidence.",
        inputSchema={
            "type": "object",
            "required": ["claim"],
            "properties": {
                "claim": {"type": "string", "description": "Claim to score against memory evidence"},
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "default": "review",
                    "description": "Reasoning phase for purpose-conditioned evidence scoring",
                },
                "limit": {"type": "integer", "description": "Max evidence items", "default": 5, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_similar_procedures",
        description="Retrieve similar procedure candidates from Skill Memory and consolidation memory.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Procedure retrieval query"},
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "default": "planning",
                    "description": "Reasoning phase for purpose-conditioned procedure retrieval",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max procedure candidates",
                    "default": 5,
                    "minimum": 1,
                },
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_record_known_unknown",
        description="Record a cited, open, non-authoritative known unknown.",
        inputSchema={
            "type": "object",
            "required": ["question", "reason", "source_events", "claim_key"],
            "properties": {
                "question": {"type": "string", "description": "Known-unknown question to track"},
                "reason": {"type": "string", "description": "Reason this uncertainty was recorded"},
                "source_events": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["seq", "hash"],
                        "properties": {
                            "seq": {"type": "integer", "minimum": 1},
                            "hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                        "additionalProperties": False,
                    },
                    "description": "Cited Eventloom source events supporting the known unknown",
                },
                "claim_key": {"type": "string", "description": "Stable claim or uncertainty key"},
                "gap_type": {"type": "string", "description": "Uncertainty gap type", "default": "missing_evidence"},
                "reverify_query": {"type": "string", "description": "Suggested query for re-verification"},
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "default": "review",
                    "description": "Reasoning phase for purpose-conditioned uncertainty recording",
                },
                "actor": {
                    "type": "string",
                    "description": "Actor recording the known unknown",
                    "default": "zaxy-reasoning",
                },
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_known_unknowns",
        description="List replay-derived known unknowns for a session.",
        inputSchema={
            "type": "object",
            "required": [],
            "properties": {
                "status": {"type": "string", "description": "Known-unknown status filter or all", "default": "open"},
                "limit": {"type": "integer", "description": "Max known unknowns", "default": 10, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_confidence_trajectory",
        description="List append-only confidence assessments for a claim.",
        inputSchema={
            "type": "object",
            "required": ["claim"],
            "properties": {
                "claim": {"type": "string", "description": "Claim or claim key to inspect"},
                "limit": {"type": "integer", "description": "Max trajectory points", "default": 10, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_reverification_needs",
        description="List open unknowns, unresolved conflicts, and low-confidence claims needing re-verification.",
        inputSchema={
            "type": "object",
            "required": [],
            "properties": {
                "query": {"type": "string", "description": "Optional query filter"},
                "limit": {"type": "integer", "description": "Max re-verification needs", "default": 10, "minimum": 1},
                "min_confidence": {
                    "type": "number",
                    "description": "Confidence threshold",
                    "default": 0.7,
                    "minimum": 0,
                    "maximum": 1,
                },
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_plan_from_procedures",
        description="Build a non-authoritative planning packet from applicable procedures.",
        inputSchema={
            "type": "object",
            "required": ["goal"],
            "properties": {
                "goal": {"type": "string", "description": "Planning goal"},
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "default": "planning",
                    "description": "Reasoning phase for purpose-conditioned procedural planning",
                },
                "limit": {"type": "integer", "description": "Max plan steps/source procedures", "default": 5, "minimum": 1},
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_verbatim",
        description="Retrieve exact Eventloom source chunks with citations.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "session_id": {"type": "string", "description": "Session ID for source recall"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_feedback",
        description="After using retrieved context, record whether a memory item was useful, stale, corrected, or reinforced.",
        inputSchema={
            "type": "object",
            "required": ["entity_name", "entity_type", "feedback"],
            "properties": {
                "entity_name": {"type": "string", "description": "Retrieved graph entity name"},
                "entity_type": {"type": "string", "description": "Retrieved graph entity type"},
                "feedback": {
                    "type": "string",
                    "enum": ["used", "helpful", "irrelevant"],
                    "description": "Retrieval outcome to record",
                },
                "actor": {"type": "string", "description": "Actor recording feedback", "default": "zaxy"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
                "query": {"type": "string", "description": "Query that returned the context"},
                "source": {"type": "string", "description": "Retrieval source", "default": "mcp"},
                "score": {"type": "number", "description": "Original retrieval score"},
                "citation": {"type": "string", "description": "Eventloom citation for the retrieved context"},
                "reason": {"type": "string", "description": "Short rationale for the feedback"},
                "purpose": {
                    "oneOf": [{"type": "string"}, {"type": "object"}],
                    "description": "Optional purpose profile or preset that made this memory useful",
                },
                "outcome": {
                    "type": "string",
                    "description": "Optional action outcome, e.g. supported_handoff or avoided_failed_path",
                },
                "importance": {
                    "type": "number",
                    "description": "Optional 0..1 reinforcement importance for positive feedback",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_synthesis_artifact",
        description="Persist Memory Checkout answer candidates as synthesis artifacts and optional outcome feedback.",
        inputSchema={
            "type": "object",
            "required": ["checkout"],
            "properties": {
                "checkout": {
                    "type": "object",
                    "description": "Memory Checkout response payload containing diagnostics.synthesis.answer_candidates",
                },
                "candidate": {
                    "type": "object",
                    "description": "Optional selected answer candidate from checkout diagnostics",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["used", "helpful", "rejected", "corrected", "excluded"],
                    "description": "Optional outcome to record for the selected answer candidate",
                },
                "reason": {"type": "string", "description": "Short rationale for candidate outcome feedback"},
                "actor": {"type": "string", "description": "Actor recording the artifact", "default": "zaxy"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_synthesis_evidence",
        description="Record feedback for one Memory Checkout synthesis ledger row.",
        inputSchema={
            "type": "object",
            "required": ["checkout", "row", "outcome"],
            "properties": {
                "checkout": {
                    "type": "object",
                    "description": "Memory Checkout response payload containing diagnostics.synthesis.ledger_rows",
                },
                "row": {
                    "type": "object",
                    "description": "One ledger row from diagnostics.synthesis.ledger_rows",
                },
                "candidate": {
                    "type": "object",
                    "description": "Optional answer candidate associated with the ledger row",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["used", "helpful", "excluded"],
                    "description": "Evidence-row outcome to record",
                },
                "reason": {"type": "string", "description": "Short rationale for evidence-row feedback"},
                "actor": {"type": "string", "description": "Actor recording the evidence feedback", "default": "zaxy"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_skill",
        description="Append a typed skill lifecycle event and project it into Skill Memory.",
        inputSchema={
            "type": "object",
            "required": ["action", "skill_id"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "proposed",
                        "validated",
                        "revised",
                        "deprecated",
                        "contradicted",
                        "applied",
                        "outcome_recorded",
                    ],
                    "description": "Skill lifecycle action to record.",
                },
                "skill_id": {"type": "string", "description": "Stable skill identifier"},
                "version": {"type": "string", "description": "Skill version", "default": "1"},
                "name": {"type": "string", "description": "Human-readable skill name"},
                "summary": {"type": "string", "description": "Short skill summary"},
                "procedure": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered procedural steps",
                },
                "applicability": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Situations where the skill applies",
                },
                "failure_modes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Known failure modes or invalidating conditions for the skill",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Eventloom citations supporting the skill",
                },
                "task": {"type": "string", "description": "Task where the skill was applied"},
                "success_score": {"type": "number", "description": "Outcome score from 0 to 1"},
                "feedback": {"type": "string", "description": "Outcome feedback label"},
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Outcome evidence such as commands or citations",
                },
                "reason": {"type": "string", "description": "Reason for status changes"},
                "rollback": {"type": "string", "description": "Rollback guidance for deprecated or contradicted procedures"},
                "contradiction_reason": {"type": "string", "description": "Reason a procedure was contradicted"},
                "supersedes_version": {"type": "string", "description": "Version replaced by this event"},
                "actor": {"type": "string", "description": "Actor recording the skill event", "default": "zaxy"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_replay",
        description="Replay events from a session starting at a sequence number.",
        inputSchema={
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "Session / thread ID"},
                "from_seq": {"type": "integer", "description": "Start sequence", "default": 1},
                "admin_token": {"type": "string", "description": "Admin token if configured"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_invalidate",
        description="Mark a fact as invalid at a given time (bi-temporal update).",
        inputSchema={
            "type": "object",
            "required": ["entity_name", "entity_type", "invalid_at"],
            "properties": {
                "entity_name": {"type": "string"},
                "entity_type": {"type": "string"},
                "invalid_at": {"type": "string", "description": "ISO-8601 timestamp"},
                "admin_token": {"type": "string", "description": "Admin token if configured"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_capabilities",
        description="Describe Zaxy's active memory capabilities and ambient usage loop for this session.",
        inputSchema={
            "type": "object",
            "required": [],
            "properties": {
                "session_id": {"type": "string"},
                "current_task": {"type": "string", "description": "Current task or question to seed checkout guidance"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_bootstrap",
        description="At session start, return compact Zaxy memory guidance and the recommended first checkout call.",
        inputSchema={
            "type": "object",
            "required": [],
            "properties": {
                "session_id": {"type": "string"},
                "current_task": {"type": "string", "description": "Current task or question to seed checkout guidance"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_checkout",
        description="Before substantial work, checkout current, cited, prompt-ready memory state for a session.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
                "ref": {"type": "string", "description": "Memory ref to checkout, e.g. HEAD or refs/heads/main"},
                "replay_from_seq": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
                "purpose": {
                    "oneOf": [{"type": "string"}, {"type": "object"}],
                    "description": "Purpose profile name or object used to condition checkout guidance.",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="context_assemble",
        description="Assemble replay plus ranked retrieval into a prompt-ready context bundle.",
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
                "replay_from_seq": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="context_after_turn",
        description="Persist a completed turn and return compact context for the next turn.",
        inputSchema={
            "type": "object",
            "required": ["role", "content"],
            "properties": {
                "role": {"type": "string"},
                "content": {"type": "string"},
                "query": {"type": "string"},
                "source": {"type": "string", "default": "mcp"},
                "session_id": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="subagent_cleanup",
        description="Finalize a subagent session and return its handoff bundle.",
        inputSchema={
            "type": "object",
            "required": ["parent_session_id", "subagent_session_id", "summary"],
            "properties": {
                "parent_session_id": {"type": "string"},
                "subagent_session_id": {"type": "string"},
                "summary": {"type": "string"},
                "query": {"type": "string", "default": "subagent handoff"},
                "limit": {"type": "integer", "default": 10},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_start",
        description="Start a parent coordination mission session.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "objective"],
            "properties": {
                "mission_id": {"type": "string"},
                "objective": {"type": "string"},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_worker_create",
        description="Register a worker session under a parent mission.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "worker_id"],
            "properties": {
                "mission_id": {"type": "string"},
                "worker_id": {"type": "string"},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_assign",
        description="Assign scoped work to a coordination worker.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "worker_id", "assignment"],
            "properties": {
                "mission_id": {"type": "string"},
                "worker_id": {"type": "string"},
                "assignment": {"type": "string"},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_report_finding",
        description="Record a worker-local coordination finding with evidence; it is not trusted parent state until reviewed and promoted.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "worker_id", "summary"],
            "properties": {
                "mission_id": {"type": "string"},
                "worker_id": {"type": "string"},
                "summary": {"type": "string"},
                "actor": {"type": "string", "default": "worker"},
                "evidence": {"type": "array", "items": {"type": "object"}},
                "confidence": {"type": "number"},
                "claim_key": {"type": "string"},
                "claim_value": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_merge_brief",
        description="Return a replay-backed coordination brief for a mission.",
        inputSchema={
            "type": "object",
            "required": ["mission_id"],
            "properties": {"mission_id": {"type": "string"}},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_checkout",
        description="Return accepted coordination state for prompt injection, with optional diagnostics for pending or conflicted findings.",
        inputSchema={
            "type": "object",
            "required": ["mission_id"],
            "properties": {
                "mission_id": {"type": "string"},
                "include_diagnostics": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_performance_ledger",
        description="Return worker-level coordination outcome metrics for a mission.",
        inputSchema={
            "type": "object",
            "required": ["mission_id"],
            "properties": {"mission_id": {"type": "string"}},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_approval_packet",
        description="Return a portable pending/conflicted finding packet for a remote reviewer.",
        inputSchema={
            "type": "object",
            "required": ["mission_id"],
            "properties": {"mission_id": {"type": "string"}},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_apply_approval",
        description="Apply remote approval decisions to a coordination mission.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "decisions"],
            "properties": {
                "mission_id": {"type": "string"},
                "decisions": {"type": "array", "items": {"type": "object"}},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_review_finding",
        description="Review a worker finding as accepted, rejected, deferred, or conflicted.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "finding_id", "status"],
            "properties": {
                "mission_id": {"type": "string"},
                "finding_id": {"type": "string"},
                "status": {"type": "string", "enum": ["accepted", "rejected", "deferred", "conflicted"]},
                "rationale": {"type": "string"},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_promote",
        description="Promote an accepted finding into the parent mission history.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "finding_id"],
            "properties": {
                "mission_id": {"type": "string"},
                "finding_id": {"type": "string"},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_handoff",
        description="Create a final coordination handoff event for a mission.",
        inputSchema={
            "type": "object",
            "required": ["mission_id", "summary"],
            "properties": {
                "mission_id": {"type": "string"},
                "summary": {"type": "string"},
                "next_steps": {"type": "array", "items": {"type": "string"}},
                "risks": {"type": "array", "items": {"type": "string"}},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_record_synthesis_artifact",
        description=(
            "Persist a Coordinate synthesis proof packet for a mission-scoped Memory Checkout. "
            "Handoff-scoped packets require handoff_id and return handoff_event_ref with the cited handoff seq/hash."
        ),
        inputSchema={
            "type": "object",
            "required": ["mission_id", "checkout"],
            "properties": {
                "mission_id": {"type": "string"},
                "checkout": {
                    "type": "object",
                    "description": "Memory Checkout payload scoped to the same mission_id",
                },
                "decision_scope": {"type": "string", "default": "brief"},
                "handoff_id": {
                    "type": "string",
                    "description": "Required when decision_scope is handoff; binds proof to a handoff event.",
                },
                "candidate": {
                    "type": "object",
                    "description": "Optional selected answer candidate from checkout diagnostics",
                },
                "outcome": {
                    "type": "string",
                    "enum": ["used", "helpful", "rejected", "corrected", "excluded"],
                },
                "reason": {"type": "string"},
                "actor": {"type": "string", "default": "coordinator"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="coordination_proof_trace",
        description="Replay a Coordinate proof packet chain by artifact_id, handoff_id, or proof_seq.",
        inputSchema={
            "type": "object",
            "required": ["mission_id"],
            "properties": {
                "mission_id": {"type": "string"},
                "artifact_id": {"type": "string"},
                "handoff_id": {"type": "string"},
                "proof_seq": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    ),
]


# ------------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------------

class ZaxyMCPServer:
    """MCP server wiring for Zaxy memory operations.

    Args:
        eventloom_path: Directory containing .jsonl logs.
        neo4j_uri: Bolt URI for Neo4j.
        neo4j_user: Neo4j username.
        neo4j_password: Neo4j password.
    """

    def __init__(
        self,
        eventloom_path: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
        projection_backend: str | None = None,
        pggraph_dsn: str | None = None,
        embedded_graph_path: str | Path | None = None,
        latticedb_path: str | Path | None = None,
        workspace_root: str | Path | None = None,
        default_session_id: str | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        backend = projection_backend or settings.projection_backend
        self._projection_backend = backend
        self._admin_token = settings.mcp_admin_token
        self._default_session_id = validate_session_id(default_session_id or settings.eventloom_thread)
        self._lifecycle_capture_enabled = settings.mcp_lifecycle_capture_enabled
        self._workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self._initialized_workspaces: dict[tuple[str, str], WorkspaceProfile] = {}
        self._initialized_instruction_signatures: dict[tuple[str, str], str] = {}
        self._eventloom_path = eventloom_path or settings.eventloom_path
        self.session_manager = SessionManager(base_path=self._eventloom_path)
        self.refs = MemoryRefStore(self._eventloom_path)
        self._neo4j_uri = neo4j_uri or settings.neo4j_uri
        self._neo4j_user = neo4j_user or settings.neo4j_user
        self._neo4j_password = neo4j_password or settings.neo4j_password
        self._neo4j_ca_cert = settings.neo4j_ca_cert
        self._neo4j_trust_all = settings.neo4j_trust_all
        self._pggraph_dsn = pggraph_dsn or settings.pggraph_dsn
        resolved_embedded_graph_path = (
            Path(embedded_graph_path)
            if embedded_graph_path is not None
            else Path(self._eventloom_path) / "projections" / "embedded.kuzu"
            if eventloom_path is not None
            else Path(settings.embedded_graph_path)
        )
        self._embedded_graph_path = resolved_embedded_graph_path
        self._latticedb_path = Path(latticedb_path or settings.latticedb_path)
        self.local_projection_runtime = self._build_local_projection_runtime(
            settings,
            projection_backend=backend,
            pggraph_dsn=pggraph_dsn,
            embedded_graph_path=resolved_embedded_graph_path,
        )
        self.local_neo4j = (
            self.local_projection_runtime
            if backend.casefold().strip() != "pggraph"
            else None
        )
        self.graph = build_projection_store(
            ProjectionBackendConfig(
                backend=backend,
                neo4j_uri=self._neo4j_uri,
                neo4j_user=self._neo4j_user,
                neo4j_password=self._neo4j_password,
                neo4j_ca_cert=settings.neo4j_ca_cert,
                neo4j_trust_all=settings.neo4j_trust_all,
                pggraph_dsn=self._pggraph_dsn,
                embedded_graph_path=resolved_embedded_graph_path,
                latticedb_path=self._latticedb_path,
                embedding_dimension=settings.embedding_dimension,
            )
        )
        self.tracer = MemoryTracer(
            base_url=settings.pathlight_url,
            project_id=settings.pathlight_project_id,
            disabled=not settings.pathlight_enabled,
        )
        self._retention_policy = build_retention_policy(settings)
        self.context_assembly_policy = ContextAssemblyPolicy(
            verbatim_enabled=settings.context_verbatim_enabled,
            verbatim_slots=settings.context_verbatim_slots,
        )

    def _build_local_projection_runtime(
        self,
        settings: Any,
        *,
        projection_backend: str | None = None,
        pggraph_dsn: str | None = None,
        embedded_graph_path: str | Path | None = None,
    ) -> LocalNeo4jRuntime | LocalPgGraphRuntime | LocalEmbeddedGraphRuntime:
        backend = (projection_backend or settings.projection_backend).casefold().strip()
        if backend == "pggraph":
            return LocalPgGraphRuntime(
                dsn=pggraph_dsn or settings.pggraph_dsn,
                enabled=settings.pggraph_auto_start and settings.zaxy_env.lower() != "production",
                image=settings.pggraph_auto_start_image,
                container_name=settings.pggraph_auto_start_container,
                pggraph_repo=settings.pggraph_repo,
            )
        if backend == "embedded":
            return LocalEmbeddedGraphRuntime(path=embedded_graph_path or settings.embedded_graph_path)
        return LocalNeo4jRuntime(
            uri=self._neo4j_uri,
            user=self._neo4j_user,
            password=self._neo4j_password,
            enabled=settings.neo4j_auto_start and settings.zaxy_env.lower() != "production",
            image=settings.neo4j_auto_start_image,
            container_name=settings.neo4j_auto_start_container,
        )

    async def setup(self) -> None:
        """Connect to the selected projection backend and initialize schema."""
        self.local_projection_runtime.ensure_available()
        await self.graph.connect()
        await self.graph.init_schema()
        await self.tracer.connect()
        await self.ensure_session_initialized(
            self._workspace_root,
            session_id=self._default_session_id,
        )

    async def teardown(self) -> None:
        """Close connections."""
        await self.capture_session_ended(reason="teardown", status="succeeded")
        await self.graph.close()
        await self.tracer.close()

    async def _append_lifecycle_event(
        self,
        event_input: dict[str, Any],
        *,
        session_id: str,
    ) -> None:
        sid = validate_session_id(session_id)
        eventlog = self.session_manager.get(sid).eventlog
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=validate_payload(event_input["payload"]),
            thread=sid,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=sid)
        await self.tracer.trace_append(event_input["event_type"], event_input["actor"], event.seq)

    async def capture_tool_call_completed(
        self,
        *,
        tool_name: str,
        status: str,
        session_id: str,
        arguments: dict[str, Any],
        result_summary: str | None = None,
    ) -> None:
        """Append and project a redacted tool-call lifecycle event."""
        sid = validate_session_id(session_id)
        event_input = build_tool_call_completed_event(
            tool_name=tool_name,
            status=status,
            session_id=sid,
            arguments=arguments,
            result_summary=result_summary,
        )
        await self._append_lifecycle_event(event_input, session_id=sid)

    async def capture_session_ended(
        self,
        *,
        reason: str,
        status: str,
    ) -> None:
        """Append and project a session-end lifecycle event."""
        if not self._lifecycle_capture_enabled:
            return
        event_input = build_session_ended_event(
            session_id=self._default_session_id,
            reason=reason,
            status=status,
        )
        with suppress(Exception):
            await self._append_lifecycle_event(event_input, session_id=self._default_session_id)

    async def ensure_session_initialized(
        self,
        path: str | Path,
        *,
        session_id: str,
    ) -> WorkspaceProfile:
        """Idempotently append and project a workspace genesis event."""
        sid = validate_session_id(session_id)
        root = str(Path(path).resolve())
        key = (root, sid)
        cached = self._initialized_workspaces.get(key)
        if cached is not None:
            await self._ensure_workspace_instructions(root, session_id=sid)
            return cached

        eventlog = self.session_manager.get(sid).eventlog
        profile = existing_session_genesis_profile(
            eventlog.read_all(),
            root=root,
            session_id=sid,
        )
        if profile is not None:
            self._initialized_workspaces[key] = profile
            await self._ensure_workspace_instructions(root, session_id=sid)
            return profile

        event_input = build_session_genesis_event(root, session_id=sid)
        payload = validate_payload(event_input["payload"])
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=payload,
            thread=sid,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=sid)
        await self.tracer.trace_append(event_input["event_type"], event_input["actor"], event.seq)
        profile = workspace_profile_from_payload(payload)
        self._initialized_workspaces[key] = profile
        await self._ensure_workspace_instructions(root, session_id=sid)
        return profile

    async def _ensure_workspace_instructions(
        self,
        path: str | Path,
        *,
        session_id: str,
    ) -> None:
        """Idempotently append and project discovered workspace instruction summaries."""
        root = str(Path(path).resolve())
        key = (root, session_id)
        event_input = build_workspace_instruction_event(root, session_id=session_id)
        if event_input is None:
            return
        signature = str(event_input["payload"]["signature"])
        if self._initialized_instruction_signatures.get(key) == signature:
            return

        eventlog = self.session_manager.get(session_id).eventlog
        existing_signature = existing_workspace_instructions_signature(
            eventlog.read_all(),
            root=root,
            session_id=session_id,
        )
        if existing_signature == signature:
            self._initialized_instruction_signatures[key] = signature
            return
        if existing_signature is not None:
            event_input = mark_workspace_instruction_event_updated(
                event_input,
                previous_signature=existing_signature,
            )

        payload = validate_payload(event_input["payload"])
        event = eventlog.append(
            event_input["event_type"],
            actor=event_input["actor"],
            payload=payload,
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_input["event_type"], event_input["actor"], event.seq)
        self._initialized_instruction_signatures[key] = signature

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def handle_memory_append(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_append tool call."""
        event_type = validate_event_text(arguments.get("event_type"), "event_type")
        actor = validate_event_text(arguments.get("actor"), "actor")
        payload = validate_payload(arguments.get("payload", {}))
        session_id = self._session_id_from_arguments(
            arguments,
            default=self._default_session_id,
        )

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(event_type, actor=actor, payload=payload, thread=session_id)

        # Project to graph
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)

        return [TextContent(type="text", text=json.dumps({"seq": event.seq, "hash": event.hash}))]

    async def handle_memory_causal_successors(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_causal_successors tool calls."""
        return await self._handle_memory_causal_neighbors(arguments, direction="successors")

    async def handle_memory_causal_predecessors(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_causal_predecessors tool calls."""
        return await self._handle_memory_causal_neighbors(arguments, direction="predecessors")

    async def _handle_memory_causal_neighbors(
        self,
        arguments: dict[str, Any],
        *,
        direction: Literal["successors", "predecessors"],
    ) -> list[TextContent]:
        entity_name = validate_query(cast(str, arguments.get("entity_name")))
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        raw_depth = arguments.get("depth", 2)
        if isinstance(raw_depth, bool):
            raise ValueError("depth must be an integer")
        depth = validate_traversal_depth(raw_depth)
        relation_type = _optional_text(arguments.get("relation_type"))
        graph_relation_type = (
            causal_relation_to_graph_relation(relation_type) if relation_type is not None else None
        )

        neighbors = await self.graph.search_causal_neighbors(
            entity_name,
            direction=direction,
            relation_type=graph_relation_type,
            depth=depth,
            temporal_point=None,
            session_id=session_id,
        )
        results = [
            result.to_dict()
            for entity in neighbors
            if (result := causal_query_result_from_projection(entity, direction=direction)) is not None
        ]
        return [TextContent(type="text", text=json.dumps({"results": results}, indent=2))]

    async def handle_memory_consolidation_candidate(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_consolidation_candidate tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        actor = _optional_strict_text(arguments.get("actor"), "actor") or "zaxy-consolidation"
        event_input = build_consolidation_candidate_event(
            actor=actor,
            session_id=session_id,
            candidate_type=_required_strict_text(arguments.get("candidate_type"), "candidate_type"),
            title=_required_strict_text(arguments.get("title"), "title"),
            summary=_required_strict_text(arguments.get("summary"), "summary"),
            source_events=arguments.get("source_events", []),
            confidence=_validate_reasoning_confidence(arguments.get("confidence")),
            method=_required_strict_text(arguments.get("method"), "method"),
            purpose=_optional_strict_text(arguments.get("purpose"), "purpose"),
        )
        event = await self._append_project_and_trace_event(event_input, session_id=session_id)
        return [TextContent(type="text", text=json.dumps({"seq": event.seq, "hash": event.hash}))]

    async def handle_memory_consolidation_propose_from_log(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_consolidation_propose_from_log tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        actor = _optional_strict_text(arguments.get("actor"), "actor") or "zaxy-consolidation"
        purpose = _optional_strict_text(arguments.get("purpose"), "purpose")
        window_size = _validate_consolidation_window_size(arguments.get("window_size", 5))
        fabric = self._memory_fabric()
        try:
            await fabric.connect()
            result = await fabric.propose_consolidation_candidates(
                session_id=session_id,
                actor=actor,
                purpose=purpose,
                window_size=window_size,
            )
        finally:
            with suppress(Exception):
                await fabric.close()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_consolidation_status(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_consolidation_status tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        fabric = self._memory_fabric()
        try:
            await fabric.connect()
            result = await fabric.consolidation_status(session_id=session_id)
        finally:
            with suppress(Exception):
                await fabric.close()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_consolidation_review(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_consolidation_review tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        actor = _optional_strict_text(arguments.get("actor"), "actor") or "zaxy-reviewer"
        event_input = build_consolidation_review_event(
            actor=actor,
            session_id=session_id,
            candidate_id=_required_strict_text(arguments.get("candidate_id"), "candidate_id"),
            status=_required_strict_text(arguments.get("status"), "status"),
            rationale=_required_strict_text(arguments.get("rationale"), "rationale"),
        )
        event = await self._append_project_and_trace_event(event_input, session_id=session_id)
        return [TextContent(type="text", text=json.dumps({"seq": event.seq, "hash": event.hash}))]

    async def handle_memory_explain_outcome(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_explain_outcome tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "explain_outcome",
            _required_strict_text(arguments.get("outcome"), "outcome"),
            phase=_validate_reasoning_phase(arguments.get("phase"), default="planning"),
            session_id=session_id,
            depth=validate_traversal_depth(arguments.get("depth", 2)),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_propose_belief_update(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_propose_belief_update tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        actor = _optional_strict_text(arguments.get("actor"), "actor") or "zaxy-reasoning"
        result = await self._call_reasoning_fabric(
            "propose_belief_update",
            _required_strict_text(arguments.get("claim"), "claim"),
            rationale=_required_strict_text(arguments.get("rationale"), "rationale"),
            confidence=_validate_reasoning_confidence(arguments.get("confidence")),
            source_events=_validate_reasoning_source_events(arguments.get("source_events")),
            phase=_validate_reasoning_phase(arguments.get("phase"), default="reflection"),
            session_id=session_id,
            actor=actor,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_claim_confidence(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_claim_confidence tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "get_claim_confidence",
            _required_strict_text(arguments.get("claim"), "claim"),
            phase=_validate_reasoning_phase(arguments.get("phase"), default="review"),
            session_id=session_id,
            limit=validate_limit(arguments.get("limit"), default=5),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_similar_procedures(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_similar_procedures tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "retrieve_similar_procedures",
            _required_strict_text(arguments.get("query"), "query"),
            phase=_validate_reasoning_phase(arguments.get("phase"), default="planning"),
            session_id=session_id,
            limit=validate_limit(arguments.get("limit"), default=5),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_record_known_unknown(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_record_known_unknown tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        actor = _optional_strict_text(arguments.get("actor"), "actor") or "zaxy-reasoning"
        result = await self._call_reasoning_fabric(
            "record_known_unknown",
            _required_strict_text(arguments.get("question"), "question"),
            reason=_required_strict_text(arguments.get("reason"), "reason"),
            source_events=_validate_reasoning_source_events(arguments.get("source_events")),
            claim_key=_required_strict_text(arguments.get("claim_key"), "claim_key"),
            gap_type=_optional_strict_text(arguments.get("gap_type"), "gap_type") or "missing_evidence",
            reverify_query=_optional_strict_text(arguments.get("reverify_query"), "reverify_query"),
            phase=_validate_reasoning_phase(arguments.get("phase"), default="review"),
            session_id=session_id,
            actor=actor,
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_known_unknowns(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_known_unknowns tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "list_known_unknowns",
            session_id=session_id,
            status=_optional_strict_text(arguments.get("status"), "status") or "open",
            limit=validate_limit(arguments.get("limit"), default=10),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_confidence_trajectory(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_confidence_trajectory tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "list_confidence_trajectory",
            _required_strict_text(arguments.get("claim"), "claim"),
            session_id=session_id,
            limit=validate_limit(arguments.get("limit"), default=10),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_reverification_needs(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_reverification_needs tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "list_reverification_needs",
            query=_optional_strict_text(arguments.get("query"), "query"),
            session_id=session_id,
            limit=validate_limit(arguments.get("limit"), default=10),
            min_confidence=_validate_reasoning_confidence(arguments.get("min_confidence", 0.7)),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def handle_memory_plan_from_procedures(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_plan_from_procedures tool calls."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        result = await self._call_reasoning_fabric(
            "plan_from_procedures",
            _required_strict_text(arguments.get("goal"), "goal"),
            phase=_validate_reasoning_phase(arguments.get("phase"), default="planning"),
            session_id=session_id,
            limit=validate_limit(arguments.get("limit"), default=5),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _call_reasoning_fabric(self, method_name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        fabric = self._memory_fabric()
        try:
            await fabric.connect()
            result = await getattr(fabric, method_name)(*args, **kwargs)
            return dict(result)
        finally:
            with suppress(Exception):
                await fabric.close()

    async def _append_project_and_trace_event(self, event_input: dict[str, Any], *, session_id: str) -> Any:
        event_type = validate_event_text(event_input["event_type"], "event_type")
        actor = validate_event_text(event_input["actor"], "actor")
        payload = validate_payload(event_input["payload"])
        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(event_type, actor=actor, payload=payload, thread=session_id)
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)
        return event

    def _memory_fabric(self) -> MemoryFabric:
        return MemoryFabric(
            eventloom_path=self._eventloom_path,
            neo4j_uri=self._neo4j_uri,
            neo4j_user=self._neo4j_user,
            neo4j_password=self._neo4j_password,
            neo4j_ca_cert=self._neo4j_ca_cert,
            neo4j_trust_all=self._neo4j_trust_all,
            projection_backend=self._projection_backend,
            pggraph_dsn=self._pggraph_dsn,
            embedded_graph_path=self._embedded_graph_path,
            latticedb_path=self._latticedb_path,
        )

    async def handle_coordination_start(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_start tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        objective = _required_text(arguments.get("objective"), "objective")
        result = self._coordination_manager().start_mission(mission_id, objective=objective, actor=actor)
        await self._project_coordination_result(result.event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, "coordination.mission.created")))]

    async def handle_coordination_worker_create(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_worker_create tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        worker_id = validate_session_id(_required_text(arguments.get("worker_id"), "worker_id"))
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        result = self._coordination_manager().create_worker(mission_id, worker_id, actor=actor)
        await self._project_coordination_result(result.event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, "coordination.worker.created")))]

    async def handle_coordination_assign(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_assign tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        worker_id = validate_session_id(_required_text(arguments.get("worker_id"), "worker_id"))
        assignment = _required_text(arguments.get("assignment"), "assignment")
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        result = self._coordination_manager().assign(mission_id, worker_id, assignment, actor=actor)
        await self._project_coordination_result(result.event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, "coordination.assignment.created")))]

    async def handle_coordination_report_finding(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_report_finding tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        worker_id = validate_session_id(_required_text(arguments.get("worker_id"), "worker_id"))
        summary = _required_text(arguments.get("summary"), "summary")
        actor = _optional_text(arguments.get("actor")) or "worker"
        raw_evidence = arguments.get("evidence")
        evidence_items = raw_evidence if isinstance(raw_evidence, list) else []
        result = self._coordination_manager().report_finding(
            mission_id,
            worker_id,
            summary=summary,
            actor=actor,
            evidence=[item for item in evidence_items if isinstance(item, dict)],
            confidence=arguments.get("confidence") if arguments.get("confidence") is not None else None,
            claim_key=_optional_text(arguments.get("claim_key")),
            claim_value=_optional_text(arguments.get("claim_value")),
        )
        await self._project_coordination_result(result.event, session_id=worker_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, "coordination.finding.reported")))]

    async def handle_coordination_merge_brief(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_merge_brief tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        brief = self._coordination_manager().brief(mission_id)
        return [TextContent(type="text", text=json.dumps(brief.to_dict(), indent=2, sort_keys=True))]

    async def handle_coordination_checkout(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_checkout tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        include_diagnostics = bool(arguments.get("include_diagnostics", False))
        checkout = self._coordination_manager().checkout(mission_id, include_diagnostics=include_diagnostics)
        return [TextContent(type="text", text=json.dumps(checkout.to_dict(), indent=2, sort_keys=True))]

    async def handle_coordination_performance_ledger(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_performance_ledger tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        ledger = self._coordination_manager().performance_ledger(mission_id)
        return [TextContent(type="text", text=json.dumps(ledger.to_dict(), indent=2, sort_keys=True))]

    async def handle_coordination_approval_packet(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_approval_packet tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        packet = self._coordination_manager().approval_packet(mission_id)
        return [TextContent(type="text", text=json.dumps(packet.to_dict(), indent=2, sort_keys=True))]

    async def handle_coordination_apply_approval(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_apply_approval tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        decisions = _approval_decisions(arguments.get("decisions"))
        result = self._coordination_manager().apply_approval_decisions(
            mission_id,
            decisions,
            actor=actor,
        )
        for event in result.events:
            await self._project_coordination_result(event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(result.to_dict(), indent=2, sort_keys=True))]

    async def handle_coordination_review_finding(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_review_finding tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        finding_id = _required_text(arguments.get("finding_id"), "finding_id")
        status = _required_text(arguments.get("status"), "status")
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        result = self._coordination_manager().review_finding(
            mission_id,
            finding_id,
            status=status,
            actor=actor,
            rationale=_optional_text(arguments.get("rationale")),
        )
        await self._project_coordination_result(result.event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, "coordination.finding.reviewed")))]

    async def handle_coordination_promote(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_promote tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        finding_id = _required_text(arguments.get("finding_id"), "finding_id")
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        result = self._coordination_manager().promote_finding(mission_id, finding_id, actor=actor)
        await self._project_coordination_result(result.event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, "coordination.finding.promoted")))]

    async def handle_coordination_handoff(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_handoff tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        result = self._coordination_manager().create_handoff(
            mission_id,
            summary=_required_text(arguments.get("summary"), "summary"),
            next_steps=_optional_text_list(arguments.get("next_steps")),
            risks=_optional_text_list(arguments.get("risks")),
            actor=actor,
        )
        await self._project_coordination_result(result.event, session_id=mission_id)
        return [TextContent(type="text", text=json.dumps(_coordination_result_payload(result, result.event.type)))]

    async def handle_coordination_record_synthesis_artifact(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_record_synthesis_artifact tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        checkout_payload = arguments.get("checkout")
        if not isinstance(checkout_payload, dict):
            raise ValueError("checkout must be a Memory Checkout object")
        checkout = _memory_checkout_from_payload(checkout_payload)
        if checkout.session_id != mission_id:
            raise ValueError("Coordinate synthesis checkout session_id must match mission_id")
        actor = _optional_text(arguments.get("actor")) or "coordinator"
        decision_scope = _optional_text(arguments.get("decision_scope")) or "brief"
        handoff_id = _optional_text(arguments.get("handoff_id"))
        has_candidate = arguments.get("candidate") is not None
        has_outcome = arguments.get("outcome") is not None
        if has_candidate != has_outcome:
            raise ValueError("candidate and outcome must be supplied together")

        candidate_payload: dict[str, Any] | None = None
        candidate_event_type: str | None = None
        normalized_outcome: str | None = None
        if has_candidate and has_outcome:
            candidate = arguments.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("candidate must be an object")
            raw_outcome = arguments.get("outcome")
            if not isinstance(raw_outcome, str):
                raise ValueError("outcome must be supplied with candidate")
            normalized_outcome = normalize_synthesis_outcome(raw_outcome)
            candidate_event_type = synthesis_outcome_event_type(normalized_outcome)
            candidate_payload = build_synthesis_candidate_event_payload(
                checkout=checkout,
                candidate=candidate,
                outcome=normalized_outcome,
                reason=_optional_text(arguments.get("reason")),
            )

        artifact_payload = build_synthesis_artifact(checkout)
        proof_packet = self._coordination_manager().proof_packet(
            mission_id,
            artifact_payload,
            decision_scope=decision_scope,
            handoff_id=handoff_id,
        )
        proof_payload = validate_payload(proof_packet.to_dict())
        eventlog = self.session_manager.get(mission_id).eventlog
        artifact_event = eventlog.append(
            "memory.synthesis.artifact.created",
            actor=actor,
            payload=validate_payload(artifact_payload),
            thread=mission_id,
        )
        await self._project_coordination_result(artifact_event, session_id=mission_id)

        candidate_event = None
        if candidate_payload is not None and candidate_event_type is not None:
            candidate_event = eventlog.append(
                candidate_event_type,
                actor=actor,
                payload=validate_payload(candidate_payload),
                thread=mission_id,
            )
            await self._project_coordination_result(candidate_event, session_id=mission_id)

        proof_event = eventlog.append(
            "coordination.proof_packet.created",
            actor=actor,
            payload=proof_payload,
            thread=mission_id,
        )
        await self._project_coordination_result(proof_event, session_id=mission_id)
        response = {
            "artifact_id": artifact_payload["artifact_id"],
            "artifact_event": {
                "seq": artifact_event.seq,
                "hash": artifact_event.hash,
                "event_type": artifact_event.type,
            },
            "candidate_event": (
                {
                    "seq": candidate_event.seq,
                    "hash": candidate_event.hash,
                    "event_type": candidate_event_type,
                    "outcome": normalized_outcome,
                }
                if candidate_event is not None
                else None
            ),
            "proof_event": {
                "seq": proof_event.seq,
                "hash": proof_event.hash,
                "event_type": proof_event.type,
            },
            "proof_packet": proof_payload,
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def handle_coordination_proof_trace(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle coordination_proof_trace tool calls."""
        mission_id = validate_session_id(_required_text(arguments.get("mission_id"), "mission_id"))
        self._enforce_coordination_mission_scope(mission_id)
        proof_seq = arguments.get("proof_seq")
        if proof_seq is not None and (not isinstance(proof_seq, int) or proof_seq < 1):
            raise ValueError("proof_seq must be a positive integer")
        trace = self._coordination_manager().proof_trace(
            mission_id,
            artifact_id=_optional_text(arguments.get("artifact_id")),
            handoff_id=_optional_text(arguments.get("handoff_id")),
            proof_seq=proof_seq,
        )
        return [TextContent(type="text", text=json.dumps(trace.to_dict(), indent=2))]

    def _coordination_manager(self) -> Any:
        """Return a coordination manager bound to this server's session manager."""
        from zaxy.coordination import CoordinationManager
        from zaxy.coordination_semantic import build_semantic_conflict_detector

        manager = CoordinationManager(
            eventloom_path=self._eventloom_path,
            semantic_conflict_detector=build_semantic_conflict_detector(self._settings),
        )
        manager.session_manager = self.session_manager
        return manager

    def _enforce_coordination_mission_scope(self, mission_id: str) -> None:
        """Prevent remote clients from writing outside their authenticated mission."""
        if remote_session_scope.get() is not None:
            self._session_id_from_arguments({"session_id": mission_id})

    async def _project_coordination_result(self, event: Any, *, session_id: str) -> None:
        """Project a coordination event through the standard graph path."""
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event.type, event.actor, event.seq)

    async def handle_memory_query(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_query tool call."""
        query = validate_query(arguments["query"])
        temporal = arguments.get("temporal_filter")
        limit = validate_limit(arguments.get("limit"), default=10)
        cursor = _optional_text(arguments.get("cursor"))
        paged = bool(arguments.get("paged")) or cursor is not None
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        if arguments.get("session_ids") is not None:
            if cursor:
                raise ValueError("cursor cannot be combined with session_ids")
            output = await self._handle_cross_session_memory_query(
                query=query,
                temporal=temporal,
                limit=limit,
                session_ids=arguments["session_ids"],
            )
            return [TextContent(type="text", text=json.dumps(output, indent=2))]

        offset = 0
        if cursor:
            decoded = validate_query_cursor(
                cursor,
                query=query,
                session_id=session_id,
                temporal_point=temporal,
            )
            offset = decoded.offset
        fetch_limit = min(offset + limit + 1, MAX_QUERY_LIMIT)

        router = QueryRouter(
            self.graph,
            session_id=session_id,
            retention_policy=self._retention_policy,
        )
        results = await router.query(query, temporal_point=temporal, limit=fetch_limit)
        await self.tracer.trace_query(query, len(results), 0.0, temporal)
        page_results = results[offset : offset + limit]
        has_more = len(results) > offset + limit
        next_cursor = None
        if has_more:
            next_cursor = encode_query_cursor(
                query=query,
                session_id=session_id,
                temporal_point=temporal,
                offset=offset + limit,
            )

        output = [
            {
                "content": r.content,
                "source": r.source,
                "score": r.score,
                "valid_from": r.valid_from,
                "valid_to": r.valid_to,
                "citation": r.citation,
                "score_explanation": r.score_explanation,
            }
            for r in page_results
        ]
        if paged:
            page_output = {
                "contexts": output,
                "next_cursor": next_cursor,
                "cursor": cursor,
                "has_more": has_more,
                "offset": offset,
                "session_id": session_id,
            }
            return [TextContent(type="text", text=json.dumps(page_output, indent=2))]
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def _handle_cross_session_memory_query(
        self,
        *,
        query: str,
        temporal: str | None,
        limit: int,
        session_ids: Any,
    ) -> list[dict[str, Any]]:
        """Run an explicit local cross-session query and annotate each hit."""
        if remote_session_scope.get() is not None:
            raise PermissionError("session scope does not permit cross-session query")
        if not isinstance(session_ids, list) or not session_ids:
            raise ValueError("session_ids must be a non-empty list")
        sessions = [validate_session_id(session_id) for session_id in session_ids]
        merged: list[dict[str, Any]] = []
        for scoped_session in sessions:
            router = QueryRouter(
                self.graph,
                session_id=scoped_session,
                retention_policy=self._retention_policy,
            )
            results = await router.query(query, temporal_point=temporal, limit=limit)
            for result in results:
                merged.append(
                    {
                        "content": result.content,
                        "source": result.source,
                        "score": result.score,
                        "valid_from": result.valid_from,
                        "valid_to": result.valid_to,
                        "citation": result.citation,
                        "score_explanation": result.score_explanation,
                        "session_id": scoped_session,
                    }
                )
        merged.sort(key=lambda row: row["score"], reverse=True)
        return merged[:limit]

    async def handle_memory_verbatim(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_verbatim source-recall tool call."""
        query = validate_query(arguments["query"])
        limit = validate_limit(arguments.get("limit"), default=10)
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)

        eventlog = self.session_manager.get(session_id).eventlog
        hits = VerbatimIndex.from_event_logs([eventlog]).query(query, limit=limit)
        output = [
            {
                "content": hit.content,
                "source": "verbatim",
                "score": hit.score,
                "citation": hit.citation,
                "source_kind": hit.source_kind,
                "metadata": hit.metadata,
            }
            for hit in hits
        ]
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_feedback(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_feedback tool call."""
        feedback = _normalize_feedback(arguments["feedback"])
        entity_name = _required_text(arguments["entity_name"], "entity_name")
        entity_type = _required_text(arguments["entity_type"], "entity_type")
        actor = _optional_text(arguments.get("actor")) or "zaxy"
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)

        payload: dict[str, Any] = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "query": _optional_text(arguments.get("query")),
            "source": _optional_text(arguments.get("source")) or "mcp",
            "score": arguments.get("score"),
            "citation": _optional_text(arguments.get("citation")),
            "reason": _optional_text(arguments.get("reason")),
        }
        purpose_payload = _purpose_payload(arguments.get("purpose"))
        if purpose_payload:
            payload["purpose"] = purpose_payload
        outcome = _optional_text(arguments.get("outcome"))
        if outcome:
            payload["outcome"] = outcome
        event_type = "memory.feedback"
        if feedback in {"used", "helpful"}:
            event_type = "memory.reinforced"
            importance = arguments.get("importance")
            if importance is not None:
                payload["importance"] = max(0.0, min(1.0, float(importance)))
        else:
            payload["feedback"] = feedback
        payload = {key: value for key, value in payload.items() if value is not None}

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(
            event_type,
            actor=actor,
            payload=validate_payload(payload),
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)

        return [
            TextContent(
                type="text",
                text=json.dumps({"seq": event.seq, "hash": event.hash, "event_type": event_type}),
            )
        ]

    async def handle_memory_synthesis_artifact(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_synthesis_artifact tool call."""
        checkout_payload = arguments.get("checkout")
        if not isinstance(checkout_payload, dict):
            raise ValueError("checkout must be a Memory Checkout object")
        actor = _optional_text(arguments.get("actor")) or "zaxy"
        checkout = _memory_checkout_from_payload(checkout_payload)
        session_id = self._session_id_from_arguments(
            {"session_id": checkout.session_id},
            default=self._default_session_id,
        )
        has_candidate = arguments.get("candidate") is not None
        has_outcome = arguments.get("outcome") is not None
        if has_candidate != has_outcome:
            raise ValueError("candidate and outcome must be supplied together")

        candidate_event = None
        candidate_event_type: str | None = None
        normalized_outcome: str | None = None
        candidate_event_payload: dict[str, Any] | None = None
        if has_candidate and has_outcome:
            candidate = arguments.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("candidate must be an object")
            raw_outcome = arguments.get("outcome")
            if not isinstance(raw_outcome, str):
                raise ValueError("outcome must be supplied with candidate")
            normalized_outcome = normalize_synthesis_outcome(raw_outcome)
            candidate_event_payload = build_synthesis_candidate_event_payload(
                checkout=checkout,
                candidate=candidate,
                outcome=normalized_outcome,
                reason=_optional_text(arguments.get("reason")),
            )
            candidate_event_type = synthesis_outcome_event_type(normalized_outcome)

        artifact_payload = build_synthesis_artifact(checkout)
        eventlog = self.session_manager.get(session_id).eventlog
        artifact_event = eventlog.append(
            "memory.synthesis.artifact.created",
            actor=actor,
            payload=validate_payload(artifact_payload),
            thread=session_id,
        )
        extraction = extract(artifact_event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append("memory.synthesis.artifact.created", actor, artifact_event.seq)

        if candidate_event_payload is not None and candidate_event_type is not None:
            candidate_event = eventlog.append(
                candidate_event_type,
                actor=actor,
                payload=validate_payload(candidate_event_payload),
                thread=session_id,
            )
            extraction = extract(candidate_event)
            await self.graph.upsert_extraction(extraction, session_id=session_id)
            await self.tracer.trace_append(candidate_event_type, actor, candidate_event.seq)

        response = {
            "artifact_id": artifact_payload["artifact_id"],
            "artifact_event": {
                "seq": artifact_event.seq,
                "hash": artifact_event.hash,
                "event_type": "memory.synthesis.artifact.created",
            },
            "candidate_event": (
                {
                    "seq": candidate_event.seq,
                    "hash": candidate_event.hash,
                    "event_type": candidate_event_type,
                    "outcome": normalized_outcome,
                }
                if candidate_event is not None
                else None
            ),
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def handle_memory_synthesis_evidence(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_synthesis_evidence tool calls."""
        checkout_payload = arguments.get("checkout")
        if not isinstance(checkout_payload, dict):
            raise ValueError("checkout must be a Memory Checkout object")
        row = arguments.get("row")
        if not isinstance(row, dict):
            raise ValueError("row must be a synthesis ledger row object")
        candidate = arguments.get("candidate")
        if candidate is not None and not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        raw_outcome = arguments.get("outcome")
        if not isinstance(raw_outcome, str):
            raise ValueError("outcome must be one of: used, helpful, excluded")
        normalized_outcome = normalize_synthesis_outcome(raw_outcome)
        if normalized_outcome not in {"used", "excluded"}:
            raise ValueError("outcome must be one of: used, helpful, excluded")
        actor = _optional_text(arguments.get("actor")) or "zaxy"
        checkout = _memory_checkout_from_payload(checkout_payload)
        _require_synthesis_row_in_checkout(checkout, row)
        session_id = self._session_id_from_arguments(
            {"session_id": checkout.session_id},
            default=self._default_session_id,
        )
        payload = build_synthesis_evidence_event_payload(
            checkout=checkout,
            row=row,
            outcome=normalized_outcome,
            candidate=candidate,
            reason=_optional_text(arguments.get("reason")),
        )
        event_type = "memory.evidence.reinforced" if normalized_outcome == "used" else "memory.evidence.excluded"
        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(
            event_type,
            actor=actor,
            payload=validate_payload(payload),
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)
        response = {
            "seq": event.seq,
            "hash": event.hash,
            "event_type": event_type,
            "outcome": normalized_outcome,
            "source_group": payload.get("source_group"),
            "fact_id": payload.get("fact_id"),
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def handle_memory_skill(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_skill lifecycle helper calls."""
        event_type = _skill_event_type(arguments.get("action"))
        skill_id = _required_text(arguments.get("skill_id"), "skill_id")
        actor = _optional_text(arguments.get("actor")) or "zaxy"
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        payload: dict[str, Any] = {"skill_id": skill_id}
        for key in (
            "version",
            "name",
            "summary",
            "task",
            "feedback",
            "reason",
            "rollback",
            "contradiction_reason",
            "supersedes_version",
        ):
            value = _optional_text(arguments.get(key))
            if value is not None:
                payload[key] = value
        for key in ("procedure", "applicability", "citations", "evidence", "failure_modes"):
            values = _optional_text_list(arguments.get(key))
            if values:
                payload[key] = values
        success_score = arguments.get("success_score")
        if success_score is not None:
            payload["success_score"] = max(0.0, min(1.0, float(success_score)))

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(
            event_type,
            actor=actor,
            payload=validate_payload(payload),
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append(event_type, actor, event.seq)

        return [
            TextContent(
                type="text",
                text=json.dumps({"seq": event.seq, "hash": event.hash, "event_type": event_type}),
            )
        ]

    async def handle_memory_replay(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_replay tool call."""
        self._require_admin(arguments)
        session_id = self._session_id_from_arguments(arguments)
        from_seq = validate_from_seq(arguments.get("from_seq"))

        replay = self.session_manager.replay(session_id, from_seq=from_seq)
        events = replay.events[:MAX_REPLAY_EVENTS]

        output = {
            "integrity": replay.integrity.model_dump(),
            "events": [e.model_dump() for e in events],
            "truncated": len(replay.events) > len(events),
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_invalidate(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_invalidate tool call."""
        self._require_admin(arguments)
        name = arguments["entity_name"]
        entity_type = arguments["entity_type"]
        invalid_at = arguments["invalid_at"]

        await self.graph.invalidate_entity(
            name,
            entity_type,
            invalid_at,
            session_id=self._session_id_from_arguments(arguments, default=self._default_session_id),
        )
        return [TextContent(type="text", text=json.dumps({"status": "invalidated"}))]

    async def handle_memory_capabilities(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_capabilities tool call."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        manifest = build_memory_capabilities(
            eventloom_path=self._eventloom_path,
            session_id=session_id,
            workspace_root=self._workspace_root,
            current_task=_optional_text(arguments.get("current_task")),
        )
        return [TextContent(type="text", text=json.dumps(manifest, indent=2))]

    async def handle_memory_bootstrap(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_bootstrap tool call."""
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        bootstrap = build_memory_bootstrap(
            eventloom_path=self._eventloom_path,
            session_id=session_id,
            workspace_root=self._workspace_root,
            current_task=_optional_text(arguments.get("current_task")),
        )
        record_memory_activity(
            self._eventloom_path,
            session_id=session_id,
            activity="bootstrap",
            source="mcp",
            query=_optional_text(arguments.get("current_task")),
        )
        return [TextContent(type="text", text=json.dumps(bootstrap, indent=2))]

    async def handle_context_assemble(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle context_assemble tool call."""
        query = validate_query(arguments["query"])
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        replay_from_seq = validate_from_seq(arguments.get("replay_from_seq"))
        limit = validate_limit(arguments.get("limit"), default=10)
        max_recent_events = validate_limit(arguments.get("max_recent_events"), default=20)

        output = await self._assemble_context_payload(
            query=query,
            session_id=session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
        )
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_memory_checkout(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle memory_checkout tool call."""
        query = validate_query(arguments["query"])
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        replay_from_seq = validate_from_seq(arguments.get("replay_from_seq"))
        limit = validate_limit(arguments.get("limit"), default=10)
        max_recent_events = validate_limit(arguments.get("max_recent_events"), default=20)
        ref = _optional_text(arguments.get("ref"))
        resolved_ref = self._resolve_checkout_ref(ref, session_id=session_id)
        checkout_session_id = resolved_ref.session_id if resolved_ref is not None else session_id

        assembly = await self._assemble_context_payload(
            query=query,
            session_id=checkout_session_id,
            replay_from_seq=replay_from_seq,
            limit=limit,
            max_recent_events=max_recent_events,
            as_of_seq=resolved_ref.target_seq if resolved_ref is not None else None,
        )
        output = build_memory_checkout(
            query=query,
            assembly=_context_assembly_from_payload(assembly),
            ref=resolved_ref,
            purpose=arguments.get("purpose"),
        ).to_dict()
        record_memory_activity(
            self._eventloom_path,
            session_id=session_id,
            activity="checkout",
            source="mcp",
            query=query,
            metadata=_checkout_activity_metadata(output),
        )
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_context_after_turn(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle context_after_turn tool call."""
        role = arguments["role"]
        content = arguments["content"]
        session_id = self._session_id_from_arguments(arguments, default=self._default_session_id)
        source = arguments.get("source", "mcp")
        query = validate_query(arguments.get("query") or content)
        limit = validate_limit(arguments.get("limit"), default=10)
        max_recent_events = validate_limit(arguments.get("max_recent_events"), default=20)

        eventlog = self.session_manager.get(session_id).eventlog
        event = eventlog.append(
            "transcript.turn",
            actor=role,
            payload={"role": role, "content": content, "source": source},
            thread=session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=session_id)
        await self.tracer.trace_append("transcript.turn", role, event.seq)

        output = await self._assemble_context_payload(
            query=query,
            session_id=session_id,
            replay_from_seq=1,
            limit=limit,
            max_recent_events=max_recent_events,
        )
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def handle_subagent_cleanup(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle subagent_cleanup tool call."""
        parent_session_id = validate_session_id(arguments["parent_session_id"])
        subagent_session_id = validate_session_id(arguments["subagent_session_id"])
        summary_text = str(arguments["summary"])
        query = validate_query(arguments.get("query") or "subagent handoff")
        limit = validate_limit(arguments.get("limit"), default=10)

        if remote_session_scope.get() is not None:
            self._session_id_from_arguments({"session_id": subagent_session_id})

        eventlog = self.session_manager.get(subagent_session_id).eventlog
        event = eventlog.append(
            "subagent.cleaned",
            actor="zaxy",
            payload={
                "parent_session_id": parent_session_id,
                "subagent_session_id": subagent_session_id,
                "summary": summary_text,
            },
            thread=subagent_session_id,
        )
        extraction = extract(event)
        await self.graph.upsert_extraction(extraction, session_id=subagent_session_id)
        await self.tracer.trace_append("subagent.cleaned", "zaxy", event.seq)
        lifecycle = build_subagent_completed_event(
            parent_session_id=parent_session_id,
            subagent_session_id=subagent_session_id,
            status="succeeded",
            summary=summary_text,
        )
        await self._append_lifecycle_event(lifecycle, session_id=subagent_session_id)

        assembly = await self._assemble_context_payload(
            query=query,
            session_id=subagent_session_id,
            replay_from_seq=1,
            limit=limit,
            max_recent_events=20,
        )
        replay = self.session_manager.replay(subagent_session_id, from_seq=1)
        output = {
            **assembly,
            "summary": self.session_manager.handoff_summary(subagent_session_id),
            "integrity_ok": bool(getattr(getattr(replay, "integrity", None), "ok", False)),
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def _assemble_context_payload(
        self,
        *,
        query: str,
        session_id: str,
        replay_from_seq: int,
        limit: int,
        max_recent_events: int,
        as_of_seq: int | None = None,
    ) -> dict[str, Any]:
        replay = self.session_manager.replay(
            session_id,
            from_seq=replay_from_seq,
            verify_integrity=False,
        )
        events = list(replay.events)
        if as_of_seq is not None:
            events = [event for event in events if event.seq <= as_of_seq]
        compacted = len(events) > max_recent_events
        recent_events = events[-max_recent_events:] if compacted else events
        router = QueryRouter(
            self.graph,
            session_id=session_id,
            retention_policy=self._retention_policy,
        )
        results = await router.query(query, limit=limit)
        graph_contexts = [_context_from_query_result(result) for result in results]
        verbatim_hits = (
            VerbatimIndex.from_event_logs(
                [self.session_manager.get(session_id).eventlog]
            ).query(query, limit=limit)
            if self.context_assembly_policy.should_query_verbatim(limit=limit)
            else []
        )
        verbatim_contexts = [
            Context(
                content=hit.content,
                source="verbatim",
                score=hit.score,
                metadata={
                    "citation": hit.citation,
                    "source_kind": hit.source_kind,
                    **hit.metadata,
                },
            )
            for hit in verbatim_hits
        ]
        contexts = self.context_assembly_policy.assemble(
            graph_contexts,
            verbatim_contexts,
            limit=limit,
        )
        if as_of_seq is not None:
            contexts = _contexts_as_of_seq(contexts, as_of_seq)
        working_set = build_working_set(recent_events, contexts)
        await self.tracer.trace_query(query, len(results), 0.0, None)
        return {
            "session_id": session_id,
            "prompt": _format_prompt(recent_events, contexts, working_set=working_set),
            "contexts": [_context_payload(context) for context in contexts],
            "replay_event_count": len(recent_events),
            "compacted": compacted,
            "assembly_policy": self.context_assembly_policy.describe(),
            "context_counts": context_counts(contexts, replay_count=len(recent_events)),
            "working_set": working_set.to_dict(),
        }

    def _require_admin(self, arguments: dict[str, Any]) -> None:
        """Require an admin token for destructive or bulk-read tools when configured."""
        if not self._admin_token:
            return
        supplied = str(arguments.get("admin_token") or "")
        if not hmac.compare_digest(supplied, self._admin_token):
            raise PermissionError("admin_token is required for this tool")

    def _session_id_from_arguments(
        self,
        arguments: dict[str, Any],
        default: str | None = None,
    ) -> str:
        """Resolve and enforce a tool session ID.

        Remote SSE requests run with a context-scoped session. They may omit a
        session argument, but they cannot cross into another session.
        """
        explicit = arguments.get("session_id") or arguments.get("thread")
        requested = explicit or default
        scoped = remote_session_scope.get()
        if scoped is not None:
            if explicit is None:
                return scoped
            safe_requested = validate_session_id(explicit)
            if safe_requested != scoped:
                raise PermissionError("session scope does not permit this session_id")
            return safe_requested
        if requested is None:
            raise ValueError("session_id is required")
        return validate_session_id(requested)

    def _enforce_optional_session(self, arguments: dict[str, Any]) -> None:
        """Reject cross-session optional tool arguments under remote scope."""
        if (
            remote_session_scope.get() is not None
            and ("session_id" in arguments or "thread" in arguments)
        ):
            self._session_id_from_arguments(arguments)

    def _resolve_checkout_ref(self, ref: str | None, *, session_id: str) -> MemoryRef | None:
        if ref is None:
            return None
        if ref == "HEAD":
            latest = self.session_manager.get(session_id).eventlog.last_event()
            if latest is None:
                return None
            return MemoryRef(
                name="HEAD",
                session_id=session_id,
                target_seq=latest.seq,
                target_hash=latest.hash,
                ref_type="head",
                updated_at=latest.timestamp,
            )
        resolved = self.refs.resolve(ref)
        if resolved is None:
            raise ValueError(f"Unknown memory ref: {ref}")
        return resolved


class JWTDecoder(Protocol):
    def __call__(self, token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        """Decode and validate a JWT."""


class JWKSClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any:
        """Return the signing key for a JWT."""


class MCPTransportAuth:
    """Authenticate and scope remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        token: str | None,
        session_header: str = "x-zaxy-session-id",
        oidc_issuer: str | None = None,
        oidc_audience: str | None = None,
        oidc_jwks_url: str | None = None,
        oidc_required_scope: str = "zaxy:mcp",
        oidc_session_claim: str = "zaxy_session",
        jwt_client: JWKSClient | None = None,
        jwt_decoder: JWTDecoder | None = None,
    ) -> None:
        self._token = token
        self._session_header = session_header.casefold()
        self._oidc_issuer = oidc_issuer
        self._oidc_audience = oidc_audience
        self._oidc_jwks_url = oidc_jwks_url
        self._oidc_required_scope = oidc_required_scope
        self._oidc_session_claim = oidc_session_claim
        self._jwt_client = jwt_client
        self._jwt_decoder = jwt_decoder or jwt.decode

    def authorize(self, headers: Mapping[str, str]) -> str:
        """Validate request headers and return the remote session scope."""
        normalized = {key.casefold(): value for key, value in headers.items()}
        if self._oidc_enabled:
            return self._authorize_oidc(normalized)
        if self._token is not None:
            header = normalized.get("authorization")
            if not header or not header.startswith("Bearer "):
                raise PermissionError("Authorization bearer token is required")
            supplied = header.removeprefix("Bearer ").strip()
            if not hmac.compare_digest(supplied, self._token):
                raise PermissionError("Authorization bearer token is invalid")
            session_id = normalized.get(self._session_header)
            if not session_id:
                raise PermissionError("remote session header is required")
            return validate_session_id(session_id)
        return validate_session_id(normalized.get(self._session_header, "default"))

    @property
    def _oidc_enabled(self) -> bool:
        return bool(self._oidc_issuer and self._oidc_audience and self._oidc_jwks_url)

    def _authorize_oidc(self, headers: Mapping[str, str]) -> str:
        header = headers.get("authorization")
        if not header or not header.startswith("Bearer "):
            raise PermissionError("Authorization bearer token is required")
        token = header.removeprefix("Bearer ").strip()
        if not token:
            raise PermissionError("Authorization bearer token is required")
        try:
            jwks_client = self._jwt_client or jwt.PyJWKClient(str(self._oidc_jwks_url))
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            claims = self._jwt_decoder(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=self._oidc_audience,
                issuer=self._oidc_issuer,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except Exception as exc:
            raise PermissionError("Authorization bearer token is invalid") from exc

        scopes = _claim_values(claims.get("scope")) | _claim_values(claims.get("scp"))
        if self._oidc_required_scope and self._oidc_required_scope not in scopes:
            raise PermissionError("Authorization bearer token missing required scope")

        session_claim = claims.get(self._oidc_session_claim)
        if not isinstance(session_claim, str) or not session_claim:
            raise PermissionError("Authorization bearer token missing session claim")
        return validate_session_id(session_claim)


class RemoteRateLimitError(PermissionError):
    """Raised when a remote session exceeds its request rate limit."""

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__("remote MCP rate limit exceeded")
        self.retry_after_seconds = retry_after_seconds


class RemoteRequestGuard:
    """Authorize, rate-limit, and audit remote MCP/SSE HTTP requests."""

    def __init__(
        self,
        *,
        auth: MCPTransportAuth,
        rate_limit_enabled: bool,
        rate_limit_requests: int,
        rate_limit_window_seconds: int,
        audit_enabled: bool,
        audit_path: Path | str,
    ) -> None:
        self._auth = auth
        self._limiter = SessionRateLimiter(
            enabled=rate_limit_enabled,
            max_requests=rate_limit_requests,
            window_seconds=rate_limit_window_seconds,
        )
        self._audit = AuditEventExporter(path=Path(audit_path), enabled=audit_enabled)

    def authorize(
        self,
        headers: Mapping[str, str],
        *,
        route: str,
        method: str,
        client_host: str | None,
    ) -> str:
        """Return authorized session ID or raise an auth/rate-limit error."""
        try:
            session_id = self._auth.authorize(headers)
        except (PermissionError, ValueError) as exc:
            self._write_audit(
                session_id=None,
                route=route,
                method=method,
                outcome="denied_auth",
                reason=str(exc),
                client_host=client_host,
            )
            raise

        decision = self._limiter.check(session_id)
        if not decision.allowed:
            get_metrics().record_rate_limit_denial(session_id)
            self._write_audit(
                session_id=session_id,
                route=route,
                method=method,
                outcome="denied_rate_limit",
                reason="rate limit exceeded",
                client_host=client_host,
            )
            raise RemoteRateLimitError(decision.retry_after_seconds)

        self._write_audit(
            session_id=session_id,
            route=route,
            method=method,
            outcome="allowed",
            reason=None,
            client_host=client_host,
        )
        return session_id

    def _write_audit(
        self,
        *,
        session_id: str | None,
        route: str,
        method: str,
        outcome: str,
        reason: str | None,
        client_host: str | None,
    ) -> None:
        self._audit.write(
            RemoteAuditEvent(
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                session_id=session_id,
                route=route,
                method=method,
                outcome=outcome,  # type: ignore[arg-type]
                reason=reason,
                client_host=client_host,
            )
        )


def _checkout_activity_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    token_efficiency = payload.get("token_efficiency")
    if isinstance(token_efficiency, dict):
        return {"token_efficiency": token_efficiency}
    return {}


def _context_assembly_from_payload(payload: dict[str, Any]) -> ContextAssembly:
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
        valid_from=payload.get("valid_from") if isinstance(payload.get("valid_from"), str) else None,
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


def _claim_values(value: Any) -> set[str]:
    if isinstance(value, str):
        return {part for part in value.split() if part}
    if isinstance(value, list):
        return {str(part) for part in value if part}
    return set()


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


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

server: ZaxyMCPServer | None = None


async def _dispatch_tool_call(
    active_server: ZaxyMCPServer,
    name: str,
    arguments: dict[str, Any],
) -> list[TextContent]:
    if name == "memory_append":
        return await active_server.handle_memory_append(arguments)
    if name == "memory_query":
        return await active_server.handle_memory_query(arguments)
    if name == "memory_causal_successors":
        return await active_server.handle_memory_causal_successors(arguments)
    if name == "memory_causal_predecessors":
        return await active_server.handle_memory_causal_predecessors(arguments)
    if name == "memory_consolidation_candidate":
        return await active_server.handle_memory_consolidation_candidate(arguments)
    if name == "memory_consolidation_propose_from_log":
        return await active_server.handle_memory_consolidation_propose_from_log(arguments)
    if name == "memory_consolidation_status":
        return await active_server.handle_memory_consolidation_status(arguments)
    if name == "memory_consolidation_review":
        return await active_server.handle_memory_consolidation_review(arguments)
    if name == "memory_explain_outcome":
        return await active_server.handle_memory_explain_outcome(arguments)
    if name == "memory_propose_belief_update":
        return await active_server.handle_memory_propose_belief_update(arguments)
    if name == "memory_claim_confidence":
        return await active_server.handle_memory_claim_confidence(arguments)
    if name == "memory_similar_procedures":
        return await active_server.handle_memory_similar_procedures(arguments)
    if name == "memory_record_known_unknown":
        return await active_server.handle_memory_record_known_unknown(arguments)
    if name == "memory_known_unknowns":
        return await active_server.handle_memory_known_unknowns(arguments)
    if name == "memory_confidence_trajectory":
        return await active_server.handle_memory_confidence_trajectory(arguments)
    if name == "memory_reverification_needs":
        return await active_server.handle_memory_reverification_needs(arguments)
    if name == "memory_plan_from_procedures":
        return await active_server.handle_memory_plan_from_procedures(arguments)
    if name == "memory_verbatim":
        return await active_server.handle_memory_verbatim(arguments)
    if name == "memory_feedback":
        return await active_server.handle_memory_feedback(arguments)
    if name == "memory_synthesis_artifact":
        return await active_server.handle_memory_synthesis_artifact(arguments)
    if name == "memory_synthesis_evidence":
        return await active_server.handle_memory_synthesis_evidence(arguments)
    if name == "memory_skill":
        return await active_server.handle_memory_skill(arguments)
    if name == "memory_replay":
        return await active_server.handle_memory_replay(arguments)
    if name == "memory_invalidate":
        return await active_server.handle_memory_invalidate(arguments)
    if name == "memory_capabilities":
        return await active_server.handle_memory_capabilities(arguments)
    if name == "memory_bootstrap":
        return await active_server.handle_memory_bootstrap(arguments)
    if name == "memory_checkout":
        return await active_server.handle_memory_checkout(arguments)
    if name == "context_assemble":
        return await active_server.handle_context_assemble(arguments)
    if name == "context_after_turn":
        return await active_server.handle_context_after_turn(arguments)
    if name == "subagent_cleanup":
        return await active_server.handle_subagent_cleanup(arguments)
    if name == "coordination_start":
        return await active_server.handle_coordination_start(arguments)
    if name == "coordination_worker_create":
        return await active_server.handle_coordination_worker_create(arguments)
    if name == "coordination_assign":
        return await active_server.handle_coordination_assign(arguments)
    if name == "coordination_report_finding":
        return await active_server.handle_coordination_report_finding(arguments)
    if name == "coordination_merge_brief":
        return await active_server.handle_coordination_merge_brief(arguments)
    if name == "coordination_checkout":
        return await active_server.handle_coordination_checkout(arguments)
    if name == "coordination_performance_ledger":
        return await active_server.handle_coordination_performance_ledger(arguments)
    if name == "coordination_approval_packet":
        return await active_server.handle_coordination_approval_packet(arguments)
    if name == "coordination_apply_approval":
        return await active_server.handle_coordination_apply_approval(arguments)
    if name == "coordination_review_finding":
        return await active_server.handle_coordination_review_finding(arguments)
    if name == "coordination_promote":
        return await active_server.handle_coordination_promote(arguments)
    if name == "coordination_handoff":
        return await active_server.handle_coordination_handoff(arguments)
    if name == "coordination_record_synthesis_artifact":
        return await active_server.handle_coordination_record_synthesis_artifact(arguments)
    if name == "coordination_proof_trace":
        return await active_server.handle_coordination_proof_trace(arguments)
    raise ValueError(f"Unknown tool: {name}")


def _capture_session_id(active_server: ZaxyMCPServer, arguments: dict[str, Any]) -> str:
    fallback = _default_capture_session_id(active_server)
    try:
        resolved = active_server._session_id_from_arguments(
            arguments,
            default=fallback,
        )
        if inspect.isawaitable(resolved):
            if inspect.iscoroutine(resolved):
                resolved.close()
            return fallback
        if isinstance(resolved, str):
            return resolved
        return fallback
    except Exception:
        return fallback


def _default_capture_session_id(active_server: ZaxyMCPServer) -> str:
    default = vars(active_server).get("_default_session_id", "default")
    return default if isinstance(default, str) else "default"


async def _capture_tool_call_best_effort(
    active_server: ZaxyMCPServer,
    *,
    name: str,
    arguments: dict[str, Any],
    session_id: str,
    status: str,
    result_summary: str | None,
) -> None:
    if vars(active_server).get("_lifecycle_capture_enabled", True) is False:
        return
    with suppress(Exception):
        await active_server.capture_tool_call_completed(
            tool_name=name,
            status=status,
            session_id=session_id,
            arguments=arguments,
            result_summary=result_summary,
        )


def _tool_result_summary(result: list[TextContent]) -> str:
    if not result:
        return "0 content items"
    first = result[0]
    text = getattr(first, "text", "")
    if text:
        return str(text)[:240]
    return f"{len(result)} content item{'s' if len(result) != 1 else ''}"


def _mcp_error_payload(exc: Exception) -> dict[str, dict[str, str]]:
    """Return the stable v0.6 MCP error payload."""
    message = str(exc)
    if isinstance(exc, ValueError) and message.startswith("Unknown tool:"):
        code = "unknown_tool"
        remediation = "Call list_tools and retry with one of the advertised tool names."
    elif isinstance(exc, ValueError):
        code = "invalid_request"
        remediation = "Check the tool input schema, fix the request payload, and retry."
    else:
        code = "internal_error"
        remediation = "Retry later or inspect Zaxy logs and doctor output for service health."
    return {
        "error": {
            "code": code,
            "message": message,
            "remediation": remediation,
        }
    }


def _mcp_error_result(exc: Exception) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(_mcp_error_payload(exc), indent=2, sort_keys=True))]


def _mcp_error_summary(exc: Exception) -> str:
    error = _mcp_error_payload(exc)["error"]
    return f"{error['code']}: {error['message']}"


@asynccontextmanager
async def _socket_mcp_transport(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> AsyncIterator[tuple[Any, Any]]:
    """Adapt a Unix socket line stream to MCP's in-memory transport."""
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    async def socket_reader() -> None:
        try:
            async with read_stream_writer:
                while line := await reader.readline():
                    try:
                        message = types.JSONRPCMessage.model_validate_json(line.decode("utf-8"))
                    except Exception as exc:
                        await read_stream_writer.send(exc)
                        continue
                    await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()

    async def socket_writer() -> None:
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    writer.write(payload.encode("utf-8") + b"\n")
                    await writer.drain()
        except anyio.ClosedResourceError:  # pragma: no cover
            await anyio.lowlevel.checkpoint()
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async with anyio.create_task_group() as tg:
        tg.start_soon(socket_reader)
        tg.start_soon(socket_writer)
        yield read_stream, write_stream


async def _run_socket_mcp_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    async with _socket_mcp_transport(reader, writer) as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def _start_owner_socket_server(owner_claim: EmbeddedMcpOwnerClaim) -> asyncio.AbstractServer:
    with suppress(FileNotFoundError):
        owner_claim.paths.socket_path.unlink()
    return await asyncio.start_unix_server(
        _run_socket_mcp_client,
        path=str(owner_claim.paths.socket_path),
    )


async def proxy_main(coordinator: EmbeddedMcpRuntimeCoordinator) -> None:
    """Proxy this stdio MCP process to the workspace embedded owner."""
    setup_logging()
    logger = get_logger("mcp_server")
    record = coordinator.wait_for_owner_record()
    socket_path = str(record["socket_path"])
    reader, writer = await asyncio.open_unix_connection(socket_path)
    logger.info("zaxy_mcp_proxy_connected socket=%s", socket_path)
    eof_sent = False

    async def stdin_to_socket() -> None:
        nonlocal eof_sent
        while True:
            line = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not line:
                writer.write_eof()
                eof_sent = True
                await writer.drain()
                break
            writer.write(line)
            await writer.drain()

    async def socket_to_stdout() -> None:
        while line := await reader.readline():
            await asyncio.to_thread(sys.stdout.buffer.write, line)
            await asyncio.to_thread(sys.stdout.buffer.flush)

    tasks = [asyncio.create_task(stdin_to_socket()), asyncio.create_task(socket_to_stdout())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        for task in pending:
            task.cancel()
    finally:
        if not eof_sent:
            with suppress(Exception):
                writer.write_eof()
                eof_sent = True
                await writer.drain()
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()


async def main(owner_claim: EmbeddedMcpOwnerClaim | None = None) -> None:
    """Run the MCP stdio server with graceful shutdown."""
    setup_logging()
    logger = get_logger("mcp_server")

    # Allow external configuration (e.g. from CLI) via module-level override
    active_server = server or ZaxyMCPServer()

    try:
        await active_server.setup()
    except Exception:
        if owner_claim is not None:
            owner_claim.close()
        raise
    owner_socket_server: asyncio.AbstractServer | None = None
    if owner_claim is not None:
        owner_socket_server = await _start_owner_socket_server(owner_claim)
        owner_claim.write_ready_record(
            workspace_root=active_server._workspace_root,
            projection_backend="embedded",
            graph_path=active_server._embedded_graph_path,
        )
    logger.info("zaxy_mcp_server_ready server=%s", get_settings().server_name)

    # Graceful shutdown on SIGTERM/SIGINT
    import signal

    shutdown_event: asyncio.Event = asyncio.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("zaxy_mcp_server_signal_received signal=%s", sig_name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    @app.list_tools()  # type: ignore[untyped-decorator, no-untyped-call]
    async def list_tools() -> list[Tool]:
        return TOOLS

    @app.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        session_id = _capture_session_id(active_server, arguments)
        try:
            result = await _dispatch_tool_call(active_server, name, arguments)
        except Exception as exc:
            await _capture_tool_call_best_effort(
                active_server,
                name=name,
                arguments=arguments,
                session_id=session_id,
                status="failed",
                result_summary=_mcp_error_summary(exc),
            )
            return _mcp_error_result(exc)
        await _capture_tool_call_best_effort(
            active_server,
            name=name,
            arguments=arguments,
            session_id=session_id,
            status="succeeded",
            result_summary=_tool_result_summary(result),
        )
        return result

    try:
        async with stdio_server() as (read_stream, write_stream):
            run_task = asyncio.create_task(
                app.run(read_stream, write_stream, app.create_initialization_options())
            )
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            done, pending = await asyncio.wait(
                [run_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
    finally:
        logger.info("zaxy_mcp_server_shutting_down")
        if owner_socket_server is not None:
            owner_socket_server.close()
            await owner_socket_server.wait_closed()
        await active_server.teardown()
        if owner_claim is not None:
            owner_claim.close()
        logger.info("zaxy_mcp_server_stopped")


async def main_sse(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Run the MCP server with SSE transport over HTTP.

    This allows the server to run as a background daemon, accessible
    via HTTP/SSE endpoints for MCP clients that support remote servers.
    """
    setup_logging()
    logger = get_logger("mcp_server")
    settings = get_settings()

    active_server = server or ZaxyMCPServer()

    await active_server.setup()
    logger.info("zaxy_mcp_server_ready transport=sse host=%s port=%s", host, port)

    import signal

    shutdown_event: asyncio.Event = asyncio.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("zaxy_mcp_server_signal_received signal=%s", sig_name)
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _on_signal)

    # SSE transport setup
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    sse_transport = SseServerTransport("/messages/")
    transport_auth = MCPTransportAuth(
        token=settings.mcp_remote_auth_token,
        session_header=settings.mcp_remote_session_header,
        oidc_issuer=settings.mcp_oidc_issuer,
        oidc_audience=settings.mcp_oidc_audience,
        oidc_jwks_url=settings.mcp_oidc_jwks_url,
        oidc_required_scope=settings.mcp_oidc_required_scope,
        oidc_session_claim=settings.mcp_oidc_session_claim,
    )
    request_guard = RemoteRequestGuard(
        auth=transport_auth,
        rate_limit_enabled=settings.mcp_rate_limit_enabled,
        rate_limit_requests=settings.mcp_rate_limit_requests,
        rate_limit_window_seconds=settings.mcp_rate_limit_window_seconds,
        audit_enabled=settings.mcp_audit_enabled,
        audit_path=settings.mcp_audit_path,
    )

    async def _sse_handler(request: Any) -> Any:
        try:
            session_id = request_guard.authorize(
                request.headers,
                route="/sse",
                method=request.method,
                client_host=getattr(request.client, "host", None),
            )
        except RemoteRateLimitError as exc:
            return PlainTextResponse(
                str(exc),
                status_code=429,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except (PermissionError, ValueError) as exc:
            return PlainTextResponse(str(exc), status_code=401)
        token = remote_session_scope.set(session_id)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            try:
                await app.run(read_stream, write_stream, app.create_initialization_options())
            finally:
                remote_session_scope.reset(token)

    async def _messages_handler(request: Any) -> Any:
        try:
            session_id = request_guard.authorize(
                request.headers,
                route="/messages/",
                method=request.method,
                client_host=getattr(request.client, "host", None),
            )
        except RemoteRateLimitError as exc:
            return PlainTextResponse(
                str(exc),
                status_code=429,
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except (PermissionError, ValueError) as exc:
            return PlainTextResponse(str(exc), status_code=401)
        token = remote_session_scope.set(session_id)
        try:
            return await sse_transport.handle_post_message(
                request.scope, request.receive, request._send
            )
        finally:
            remote_session_scope.reset(token)

    starlette_app = Starlette(
        debug=False,
        routes=[
            Route("/sse", _sse_handler),
            Route("/messages/", _messages_handler, methods=["POST"]),
        ],
    )

    import uvicorn

    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(config)

    serve_task = asyncio.create_task(uvicorn_server.serve())
    shutdown_task = asyncio.create_task(shutdown_event.wait())

    done, pending = await asyncio.wait(
        [serve_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()

    logger.info("zaxy_mcp_server_shutting_down")
    await active_server.teardown()
    logger.info("zaxy_mcp_server_stopped")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
