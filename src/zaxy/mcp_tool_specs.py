"""MCP tool specifications.

The tool schema list and its operation constants live here so the large,
mostly-declarative data block is separate from the server logic. Imported
back into zaxy.mcp_server, so `from zaxy.mcp_server import TOOLS` is unchanged.
"""

from __future__ import annotations

from typing import Any

from mcp.types import Tool

REASONING_PHASES = ["planning", "execution", "review", "reflection"]

# Umbrella tool operation tables: operation -> (legacy handler name, truly
# required arguments). Umbrella tools are additive dispatch shims; every
# legacy single-purpose tool stays available and unchanged.
MEMORY_CONSOLIDATION_OPERATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "candidate": (
        "handle_memory_consolidation_candidate",
        ("candidate_type", "title", "summary", "source_events", "confidence", "method"),
    ),
    "propose_from_log": ("handle_memory_consolidation_propose_from_log", ()),
    "status": ("handle_memory_consolidation_status", ()),
    "review": ("handle_memory_consolidation_review", ("candidate_id", "status", "rationale")),
}
MEMORY_CONFIDENCE_OPERATIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "claim": ("handle_memory_claim_confidence", ("claim",)),
    "trajectory": ("handle_memory_confidence_trajectory", ("claim",)),
    "reverification": ("handle_memory_reverification_needs", ()),
    "known_unknowns": ("handle_memory_known_unknowns", ()),
    "record_known_unknown": (
        "handle_memory_record_known_unknown",
        ("question", "reason", "source_events", "claim_key"),
    ),
}


def _umbrella_required_clauses(
    operations: dict[str, tuple[str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    """Build JSON Schema if/then clauses marking per-operation required arguments."""
    return [
        {
            "if": {"properties": {"operation": {"const": operation}}},
            "then": {"required": list(required)},
        }
        for operation, (_, required) in operations.items()
        if required
    ]


# ------------------------------------------------------------------
# Tool definitions
# ------------------------------------------------------------------

TOOLS = [
    Tool(
        name="memory_checkout",
        description=(
            "The front door to Zaxy memory: call this first, before substantial work, to "
            "checkout current, cited, prompt-ready memory state for a session. Start here; "
            "every other memory tool is plumbing or power use, discoverable through "
            "memory_capabilities."
        ),
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
                "max_tokens": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Optional prompt token budget; sections are greedily packed and "
                        "elisions are reported in diagnostics."
                    ),
                },
                "purpose": {
                    "oneOf": [{"type": "string"}, {"type": "object"}],
                    "description": "Purpose profile name or object used to condition checkout guidance.",
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_append",
        description=(
            "Append a typed event to the agent's persistent memory log. Appended state "
            "becomes retrievable through memory_checkout, the front door for reading "
            "memory back."
        ),
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
        description=(
            "Query the temporal knowledge graph for relevant context. Power use behind "
            "the memory_checkout front door: reach for it when you need targeted hits, "
            "temporal filters, or pagination rather than a prompt-ready packet."
        ),
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
        name="context_assemble",
        description=(
            "Assemble replay plus ranked retrieval into a prompt-ready context bundle. "
            "memory_checkout, the front door, wraps this assembly with current facts, "
            "citations, and a trust contract."
        ),
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "session_id": {"type": "string"},
                "replay_from_seq": {"type": "integer", "default": 1},
                "limit": {"type": "integer", "default": 10},
                "max_recent_events": {"type": "integer", "default": 20},
                "max_tokens": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Optional prompt token budget; sections are greedily packed and "
                        "elisions are reported in the budget payload."
                    ),
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_feedback",
        description=(
            "After using retrieved context, record whether a memory item was useful, stale, "
            "corrected, or reinforced. This closes the loop on context surfaced by "
            "memory_checkout, the front door."
        ),
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
        name="memory_invalidate",
        description=(
            "Mark a fact as invalid at a given time (bi-temporal update). History is "
            "preserved; the correction surfaces in later memory_checkout (front door) "
            "results."
        ),
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
        description=(
            "Discover Zaxy's memory surface: active capabilities, the ambient usage loop, "
            "the active tool profile, and which tools remain callable beyond the listed set. "
            "memory_checkout is the front door; call this at session start or whenever tool "
            "awareness is unclear."
        ),
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
        name="memory_feeling_of_knowing",
        description=(
            "Experimental metamemory pre-check: predict whether memory_checkout would likely "
            "return something for a query, in roughly a millisecond, from in-memory session "
            "state only (no embedding call, no graph query). Returns a non-authoritative "
            "verdict (likely | possible | unlikely) with its signal breakdown and raw score. "
            "It is a cheap prediction about checkout, never a memory answer or evidence, and "
            "its calibration against real checkout outcomes is still being measured — when in "
            "doubt, call memory_checkout."
        ),
        inputSchema={
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query you are considering sending to memory_checkout",
                },
                "session_id": {"type": "string", "description": "Session ID for scoped prediction"},
                "cues": {
                    "type": "object",
                    "description": (
                        "Optional encoding-specificity cue fields (e.g. mission, workspace, "
                        "tool, phase); cue values are probed against session memory alongside "
                        "the query terms."
                    ),
                    "additionalProperties": {"type": "string"},
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
        name="memory_export",
        description=(
            "Export a session's memory as a portable, cited bundle any product can consume. "
            "Pull-style: the caller selects what to export via the selector; entries carry "
            "sealed Eventloom citations. Returns a signed bundle when the server has an export "
            "signing key configured and sign=true, otherwise an unsigned canonical bundle."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session to export"},
                "grains": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["event", "semantic"]},
                    "description": "Which grains to include (default both)",
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Restrict to these event types (gates both grains)",
                },
                "since_seq": {"type": "integer", "description": "Exclusive delta cursor: seq > since_seq"},
                "max_seq": {"type": "integer", "description": "Inclusive upper bound on seq"},
                "since_time": {"type": "string", "description": "Inclusive ISO-8601 lower time bound"},
                "until_time": {"type": "string", "description": "Inclusive ISO-8601 upper time bound"},
                "query": {"type": "string", "description": "Lexical pre-filter via the verbatim index"},
                "query_limit": {"type": "integer", "description": "Top-N for the query pre-filter", "default": 50},
                "exclude_sensitivities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Redaction policy: drop entries whose event sensitivity tier is listed",
                },
                "limit": {"type": "integer", "description": "Cap to the most recent N matching events"},
                "sign": {
                    "type": "boolean",
                    "default": False,
                    "description": "Sign with the server-configured export key (errors if none is configured)",
                },
                "disclose": {
                    "type": "object",
                    "description": (
                        "Return a verifiable partial disclosure: sign the full bundle over the "
                        "main selector, then reveal only entries matching this sub-selector "
                        "(with Merkle inclusion proofs). Requires the server export key."
                    ),
                    "properties": {
                        "grains": {"type": "array", "items": {"type": "string", "enum": ["event", "semantic"]}},
                        "kinds": {"type": "array", "items": {"type": "string"}, "description": "Match the entry kind (event type, or entity:/edge: kind)"},
                        "since_seq": {"type": "integer"},
                        "max_seq": {"type": "integer"},
                        "since_time": {"type": "string"},
                        "until_time": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "admin_token": {"type": "string", "description": "Admin token when admin gating is configured"},
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
    Tool(
        name="memory_consolidation",
        description=(
            "Run one consolidation lifecycle operation: candidate (append a cited candidate), "
            "propose_from_log (generate candidates from Eventloom segments), status (read "
            "review-gated counts), or review (append a review). Additive umbrella over the "
            "memory_consolidation_* tools; remaining arguments pass through unchanged. "
            "memory_checkout stays the front door for reading memory state."
        ),
        inputSchema={
            "type": "object",
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(MEMORY_CONSOLIDATION_OPERATIONS),
                    "description": "Consolidation lifecycle operation to run",
                },
                "candidate_type": {
                    "type": "string",
                    "enum": ["episode", "claim", "procedure"],
                    "description": "Consolidation candidate type (candidate)",
                },
                "title": {"type": "string", "description": "Candidate title (candidate)"},
                "summary": {"type": "string", "description": "Candidate summary (candidate)"},
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
                    "description": "Cited Eventloom source events (candidate)",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Candidate confidence from 0.0 to 1.0 (candidate)",
                },
                "method": {"type": "string", "description": "Consolidation method identifier (candidate)"},
                "window_size": {
                    "type": "integer",
                    "description": "Number of source events per proposal window (propose_from_log)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 200,
                },
                "candidate_id": {
                    "type": "string",
                    "pattern": "^consolidation:(episode|claim|procedure):[0-9a-f]{24}$",
                    "description": "Consolidation candidate ID (review)",
                },
                "status": {
                    "type": "string",
                    "enum": ["accepted", "rejected", "deferred", "conflicted"],
                    "description": "Review lifecycle status (review)",
                },
                "rationale": {"type": "string", "description": "Review rationale (review)"},
                "purpose": {"type": "string", "description": "Optional consolidation purpose"},
                "session_id": {"type": "string", "description": "Session ID for multi-agent sharding"},
                "actor": {
                    "type": "string",
                    "description": (
                        "Actor recording the event; defaults to zaxy-consolidation for "
                        "candidate/propose_from_log and zaxy-reviewer for review"
                    ),
                },
            },
            "allOf": _umbrella_required_clauses(MEMORY_CONSOLIDATION_OPERATIONS),
            "additionalProperties": False,
        },
    ),
    Tool(
        name="memory_confidence",
        description=(
            "Run one confidence/metacognition operation: claim (score claim confidence), "
            "trajectory (list confidence assessments for a claim), reverification (list "
            "re-verification needs), known_unknowns (list known unknowns), or "
            "record_known_unknown (record a cited known unknown). Additive umbrella over the "
            "single-purpose confidence tools; remaining arguments pass through unchanged. "
            "memory_checkout stays the front door for reading memory state."
        ),
        inputSchema={
            "type": "object",
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": list(MEMORY_CONFIDENCE_OPERATIONS),
                    "description": "Confidence/metacognition operation to run",
                },
                "claim": {"type": "string", "description": "Claim to score or inspect (claim, trajectory)"},
                "question": {"type": "string", "description": "Known-unknown question to track (record_known_unknown)"},
                "reason": {"type": "string", "description": "Reason this uncertainty was recorded (record_known_unknown)"},
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
                    "description": "Cited Eventloom source events (record_known_unknown)",
                },
                "claim_key": {"type": "string", "description": "Stable claim or uncertainty key (record_known_unknown)"},
                "gap_type": {
                    "type": "string",
                    "description": "Uncertainty gap type (record_known_unknown)",
                    "default": "missing_evidence",
                },
                "reverify_query": {
                    "type": "string",
                    "description": "Suggested query for re-verification (record_known_unknown)",
                },
                "query": {"type": "string", "description": "Optional query filter (reverification)"},
                "status": {
                    "type": "string",
                    "description": "Known-unknown status filter or all (known_unknowns)",
                    "default": "open",
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Confidence threshold (reverification)",
                    "default": 0.7,
                    "minimum": 0,
                    "maximum": 1,
                },
                "limit": {"type": "integer", "description": "Max results", "minimum": 1},
                "phase": {
                    "type": "string",
                    "enum": REASONING_PHASES,
                    "description": "Reasoning phase for purpose-conditioned retrieval (claim, record_known_unknown)",
                },
                "actor": {
                    "type": "string",
                    "description": "Actor recording the event (record_known_unknown)",
                    "default": "zaxy-reasoning",
                },
                "session_id": {"type": "string", "description": "Session ID for scoped reasoning"},
            },
            "allOf": _umbrella_required_clauses(MEMORY_CONFIDENCE_OPERATIONS),
            "additionalProperties": False,
        },
    ),
]
