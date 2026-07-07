"""Central configuration for Zaxy.

All settings are loaded from environment variables with sensible defaults.
In Docker, values are injected via compose environment or Docker secrets.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from zaxy.security import eventlog_path


class Settings(BaseSettings):
    """Production-ready configuration with env var support.

    Priority (highest first):
    1. Environment variables
    2. `.env` file
    3. Default values below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow extra env vars without error
    )

    # ------------------------------------------------------------------
    # Eventloom
    # ------------------------------------------------------------------
    eventloom_path: str = Field(
        default=".eventloom",
        description="Directory for append-only JSONL event logs",
    )
    eventloom_thread: str = Field(
        default="default",
        description="Default thread/session identifier",
    )
    zaxy_domain: str | None = Field(
        default=None,
        description="Project/domain identifier used to derive safe default sessions",
    )

    # ------------------------------------------------------------------
    # Neo4j
    # ------------------------------------------------------------------
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt URI",
    )
    neo4j_user: str = Field(
        default="neo4j",
        description="Neo4j username",
    )
    neo4j_password: str = Field(
        default="testpassword",
        description="Neo4j password (override in production)",
    )
    neo4j_password_file: str | None = Field(
        default=None,
        description="Path to a file containing the Neo4j password",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j database name",
    )
    neo4j_auto_start: bool = Field(
        default=False,
        description="Automatically start local Neo4j when explicitly using the Neo4j backend",
    )
    neo4j_auto_start_image: str = Field(
        default="neo4j:5.26-community",
        description="Docker image used when automatically starting local Neo4j",
    )
    neo4j_auto_start_container: str = Field(
        default="zaxy-neo4j",
        description="Docker container name used for automatically started local Neo4j",
    )

    # ------------------------------------------------------------------
    # TLS / Security
    # ------------------------------------------------------------------
    neo4j_ca_cert: str | None = Field(
        default=None,
        description="Path to CA certificate for Neo4j TLS (bolt+s://)",
    )
    neo4j_trust_all: bool = Field(
        default=False,
        description="Trust all certificates (dev only, insecure)",
    )

    # ------------------------------------------------------------------
    # Projection backend
    # ------------------------------------------------------------------
    projection_backend: str = Field(
        default="embedded",
        description="Projection backend: embedded, neo4j, pggraph, or latticedb",
    )
    # The .kuzu extension is cosmetic/historic: changing the default would
    # orphan every configured path, and LadybugDB (the maintained Kuzu fork
    # that backs the embedded store since 2.3) does not care about extensions.
    embedded_graph_path: str = Field(
        default=".eventloom/projections/embedded.kuzu",
        description="Repo-local embedded graph projection path",
    )
    latticedb_path: str = Field(
        default=".eventloom/projections/memory.latticedb",
        description="Repo-local LatticeDB projection path",
    )
    pggraph_dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/zaxy",
        description="Experimental pgGraph PostgreSQL DSN",
    )
    pggraph_auto_start: bool = Field(
        default=False,
        description="Automatically start local pgGraph/PostgreSQL for development when selected",
    )
    pggraph_auto_start_image: str = Field(
        default="pgvector/pgvector:pg17",
        description="Docker image used for local pgGraph/PostgreSQL bootstrap",
    )
    pggraph_auto_start_container: str = Field(
        default="zaxy-pggraph",
        description="Docker container name used for local pgGraph/PostgreSQL bootstrap",
    )
    pggraph_repo: str | None = Field(
        default=None,
        description="Local pgGraph checkout containing scripts/quickstart.sh for extension bootstrap",
    )

    # ------------------------------------------------------------------
    # External plugins (Zaxy 3 / I6)
    # ------------------------------------------------------------------
    plugins: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias=AliasChoices("plugins", "ZAXY_PLUGINS"),
        description=(
            "External Zaxy plugin import specs as 'module:attr' strings, loaded in "
            "addition to installed 'zaxy.plugins' entry points. Set ZAXY_PLUGINS to a "
            "comma-separated list (e.g. 'pkg_a:PLUGIN,pkg_b:PLUGIN')."
        ),
    )

    # ------------------------------------------------------------------
    # Memory evolution governance (Zaxy 3 / I4)
    # ------------------------------------------------------------------
    evolution_autonomy_default: str = Field(
        default="auto_with_rollback",
        description=(
            "Default autonomy tier for governed memory evolution: auto_with_rollback "
            "(auto-apply above threshold, reversible within the rollback window; default), "
            "propose_only (never auto-promote), or require_review"
        ),
    )
    evolution_rollback_window_seconds: int = Field(
        default=86400,
        description="Rollback window (seconds) for auto-applied evolutions under auto_with_rollback",
    )
    evolution_op_autonomy: str | None = Field(
        default=None,
        description=(
            "Optional per-op autonomy overrides as 'op=tier,op=tier' "
            "(e.g. 'forget=propose_only') to tighten guardrails for specific ops"
        ),
    )
    crystallization_enabled: bool = Field(
        default=False,
        description=(
            "Enable the governed sleep-time crystallization runner "
            "(operator/cron-triggered; off by default)"
        ),
    )
    forgetting_enabled: bool = Field(
        default=False,
        description=(
            "Enable verified forgetting via cryptographic erasure (I5b): forgettable "
            "payloads are sealed as ciphertext and the data-encryption key lives in an "
            "out-of-log erasure vault. Opt-in, off by default; when off the plaintext "
            "append path is byte-identical."
        ),
    )
    forgetting_kek_path: str | None = Field(
        default=None,
        description=(
            "Path to the key-encryption key (KEK) that wraps forgettable DEKs. Defaults "
            "to '<eventloom_dir>/__erasure_kek__.key' (a dev key auto-generated 0600 on "
            "first use). In production point this at a KMS/secret-managed key file; the "
            "KEK and wrapped DEKs are NEVER written to the append-only log."
        ),
    )
    fleet_enabled: bool = Field(
        default=False,
        description=(
            "Enable the governed fleet memory plane (cross-agent/cross-session "
            "propagation, opt-in, off by default)"
        ),
    )
    fleet_default_trust_tier: str = Field(
        default="member",
        description=(
            "Default trust tier for newly enrolled fleet agents: untrusted, "
            "member, trusted, or steward (validated against TRUST_TIERS at use site)"
        ),
    )
    long_horizon_enabled: bool = Field(
        default=False,
        description=(
            "Enable the two-tier (episodic recent + consolidated remote) Memory "
            "Checkout assembly for very long sessions; opt-in, off by default. "
            "When off, checkout is byte-identical to the single-tier contract."
        ),
    )
    long_horizon_recent_window: int = Field(
        default=50,
        gt=0,
        description=(
            "Recent events kept at full detail in the episodic tier; older "
            "history beyond this count is represented by the cited, "
            "non-authoritative consolidated tier (accepted/active consolidation "
            "candidates). Events-count window (the recall pipeline is seq-indexed)."
        ),
    )

    # ------------------------------------------------------------------
    # Pathlight
    # ------------------------------------------------------------------
    pathlight_url: str = Field(
        default="http://localhost:4100",
        description="Pathlight collector URL",
    )
    pathlight_enabled: bool = Field(
        default=False,
        description="Enable Pathlight trace emission and health checks",
    )
    pathlight_project_id: str | None = Field(
        default=None,
        description="Pathlight project identifier",
    )
    pathlight_access_token: str | None = Field(
        default=None,
        description="Optional Pathlight access token",
    )
    pathlight_access_token_file: str | None = Field(
        default=None,
        description="Path to a file containing the Pathlight access token",
    )
    trace_raw_queries: bool = Field(
        default=False,
        description="Emit raw query text to Pathlight traces (off by default)",
    )

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    server_name: str = Field(
        default="zaxy-memory",
        description="MCP server name",
    )
    zaxy_env: str = Field(
        default="development",
        description="Runtime environment: development, test, or production",
    )
    mcp_admin_token: str | None = Field(
        default=None,
        description="Optional admin token required for replay/invalidate tools",
    )
    mcp_admin_token_file: str | None = Field(
        default=None,
        description="Path to a file containing the MCP admin token",
    )
    mcp_remote_auth_token: str | None = Field(
        default=None,
        description="Bearer token required for remote MCP/SSE transport",
    )
    mcp_remote_auth_token_file: str | None = Field(
        default=None,
        description="Path to a file containing the remote MCP/SSE bearer token",
    )
    mcp_remote_session_header: str = Field(
        default="x-zaxy-session-id",
        description="HTTP header that scopes remote MCP/SSE clients to a session",
    )
    mcp_export_signing_private_key_file: str | None = Field(
        default=None,
        description="PKCS8 PEM private key file used to sign memory_export bundles (opt-in)",
    )
    mcp_export_signing_public_key_file: str | None = Field(
        default=None,
        description="Hex public key file paired with the export signing private key",
    )
    mcp_export_signing_algorithm: str = Field(
        default="ml-dsa-65",
        description="Signature algorithm of the configured export signing key",
    )
    mcp_oidc_issuer: str | None = Field(
        default=None,
        description="OIDC issuer URL for remote MCP/SSE JWT validation",
    )
    mcp_oidc_audience: str | None = Field(
        default=None,
        description="Expected OIDC audience for remote MCP/SSE JWT validation",
    )
    mcp_oidc_jwks_url: str | None = Field(
        default=None,
        description="JWKS URL for remote MCP/SSE JWT signature validation",
    )
    mcp_oidc_required_scope: str = Field(
        default="zaxy:mcp",
        description="Required OAuth scope for remote MCP/SSE access",
    )
    mcp_oidc_session_claim: str = Field(
        default="zaxy_session",
        description="JWT claim containing the Zaxy session/tenant scope",
    )
    mcp_oidc_client_secret: str | None = Field(
        default=None,
        description="Optional OIDC client secret for future token introspection flows",
    )
    mcp_oidc_client_secret_file: str | None = Field(
        default=None,
        description="Path to a file containing the OIDC client secret",
    )
    mcp_rate_limit_enabled: bool = Field(
        default=True,
        description="Enable per-session remote MCP/SSE request rate limiting",
    )
    mcp_rate_limit_requests: int = Field(
        default=120,
        description="Maximum remote MCP/SSE requests per rate-limit window",
    )
    mcp_rate_limit_window_seconds: int = Field(
        default=60,
        description="Remote MCP/SSE rate-limit window in seconds",
    )
    mcp_audit_enabled: bool = Field(
        default=False,
        description="Export remote MCP/SSE request audit JSONL records",
    )
    mcp_audit_path: str = Field(
        default=".eventloom/remote_audit.jsonl",
        description="Path for remote MCP/SSE request audit JSONL export",
    )
    mcp_lifecycle_capture_enabled: bool = Field(
        default=True,
        description="Append redacted lifecycle events for MCP tool calls",
    )
    # Default flipped from "full" to "core" in 2.1.0 backed by the internal
    # tool-adoption lane (listing surface 8,165 -> 1,344 estimated tokens, an
    # 83.5% reduction, with the front door listed first). Profiles change
    # listing only — dispatch is never filtered, so every tool stays callable
    # by name. Set MCP_TOOL_PROFILE=full to restore the previous listing.
    mcp_tool_profile: Literal["core", "full"] = Field(
        default="core",
        description="MCP tool listing profile: core lists the front-door verb set, full lists every tool",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level",
    )
    log_format: str = Field(
        default="console",
        description="Log format: console or json",
    )

    # ------------------------------------------------------------------
    # Query router
    # ------------------------------------------------------------------
    # Default flipped from "local_fast" to "cognitive" in 2.1.0 backed by the
    # internal forgetting lane (cold-start parity exact, no-recall-loss 1.0,
    # pin/authority exemptions 1.0, ranking lift 1.0 vs 0.0). The cognitive
    # profile composes the same local_fast retrieval stack plus the
    # salience-ranking, cue-blending, and graph-walk flags. Set
    # RETRIEVAL_PROFILE=local_fast to restore the previous plain ranking.
    retrieval_profile: str = Field(
        default="cognitive",
        description=(
            "Named retrieval profile: cognitive, local_fast, local_sota, hosted_sota, or custom"
        ),
    )
    query_default_limit: int = Field(
        default=10,
        description="Default result limit for queries",
    )
    query_scoring_profile: str = Field(
        default="balanced",
        description="Query scoring profile: balanced, precision, recall, or temporal",
    )
    retention_policy: str = Field(
        default="none",
        description="Retrieval retention policy: none, filter_expired, or decay",
    )
    retention_decay_half_life_days: int = Field(
        default=30,
        description="Half-life in days for decay-aware retrieval scoring",
    )
    retention_expired_weight: float = Field(
        default=0.0,
        description="Score multiplier for expired results under decay policy",
    )
    context_verbatim_enabled: bool = Field(
        default=True,
        description="Include exact Eventloom source recall in assembled context",
    )
    context_verbatim_slots: int = Field(
        default=1,
        ge=0,
        description="Maximum assembled context slots reserved for verbatim source recall",
    )
    context_packet_memory_enabled: bool = Field(
        default=True,
        description="Include recent projected LLM packet memory in assembled context",
    )
    context_packet_memory_slots: int = Field(
        default=1,
        ge=0,
        description="Maximum assembled context slots reserved for recent packet memory",
    )
    salience_half_life_days: float = Field(
        default=30.0,
        gt=0,
        description="Half-life in days for salience recency decay in the reinforcement ledger",
    )
    salience_floor: float = Field(
        default=0.15,
        ge=0,
        description=(
            "Salience score below which non-exempt memories are attenuated out of "
            "default checkout ranking under the cognitive retrieval profile"
        ),
    )
    encoding_gate_enabled: bool = Field(
        default=False,
        description=(
            "Tag appended events with a novel/reinforcing/redundant encoding "
            "classification; events are always appended and hash-chained regardless"
        ),
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    embedding_enabled: bool = Field(
        default=True,
        description="Generate embeddings for vector search",
    )
    embedding_provider: str = Field(
        default="hash",
        description="Embedding provider: hash, openai, or local-http",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension; must match the Neo4j vector index",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for hosted embeddings",
    )
    openai_api_key_file: str | None = Field(
        default=None,
        description="Path to a file containing the OpenAI API key",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL",
    )
    embedding_http_url: str | None = Field(
        default=None,
        description="Local HTTP embedding endpoint URL",
    )
    embedding_http_model: str | None = Field(
        default=None,
        description="Optional model name for local HTTP embedding endpoints",
    )
    embedding_http_api_key: str | None = Field(
        default=None,
        description="Optional bearer token for local HTTP embedding endpoints",
    )
    embedding_sentence_transformer_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Local sentence-transformers model name for semantic embeddings",
    )
    # Default lowered from 1_000_000 in 2.2 (gate G4): after the 2.2 query-path
    # rerank and COPY-based builds, the vector-scale lane passed every ANN exit
    # criterion at exactly 10^5 vectors (dimension 64) on two consecutive runs —
    # recall@10 of 1.0 on both the strict and tie-aware metrics, ANN p50
    # at-or-better than the exact matrix in-run (24.17ms vs 24.20ms, then
    # 26.67ms vs 30.82ms), resident index bytes improved (0 vs 51.2MB), and
    # full COPY shadow builds of 92-98s. Evidence:
    # docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json and -r2.json.
    vector_ann_threshold: int = Field(
        default=100_000,
        ge=1,
        description=(
            "Per-scope vector count at or above which the embedded backend uses "
            "an engine-native (LadybugDB) HNSW index instead of the exact dense matrix, "
            "provided the scope's vector dimension is at or below "
            "vector_ann_max_dimension; independently of this count, the ANN "
            "path also engages when a scope's exact float64 matrix would "
            "exceed the vector index cache byte budget (see "
            "vector_ann_byte_budget_engagement), still subject to the same "
            "dimension ceiling"
        ),
    )
    # The 64 default is the measured envelope of the G4 evidence: the lane's
    # ALL-criteria double pass exists only at dimension 64 (10^5 vectors), and
    # the same lane shows the conclusion does not transfer upward — at
    # dimension 1536 / 50k gaussian vectors, HNSW recall@10 is 0.6 even at
    # efs 400 while the exact matrix answers in 22ms p50 despite sitting 2.4x
    # over the cache byte budget (the eviction design always keeps the newest
    # matrix resident, so a single large scope never thrashes; the budget
    # bounds multi-scope cache totals only). Evidence:
    # docs/research/artifacts/ann-2026-06/ann3-d64-100k-r1.json, -r2.json, and
    # ann3-d1536-50k-gauss-crossover.json.
    vector_ann_max_dimension: int = Field(
        default=64,
        ge=1,
        description=(
            "Maximum vector dimension at which the embedded backend's ANN "
            "(LadybugDB HNSW) path may engage; scopes with higher-dimensional "
            "vectors always use exact float64 (or explicitly opted-in int8) "
            "search regardless of corpus size. The default is the dimension "
            "the vector-scale lane proved ANN better at; raise it only with "
            "lane evidence for your dimension and distribution"
        ),
    )
    vector_ann_byte_budget_engagement: bool = Field(
        default=True,
        description=(
            "Engage the embedded ANN path whenever a scope's exact float64 "
            "matrix (count x dimension x 8 bytes) would exceed the 256 MiB "
            "vector index cache byte budget, regardless of "
            "vector_ann_threshold but still only at or below "
            "vector_ann_max_dimension. An explicit VECTOR_QUANTIZATION=int8 "
            "opt-in takes precedence below the count threshold. The byte "
            "budget bounds the cache total across scopes — the newest matrix "
            "always stays resident, so a single over-budget scope degrades to "
            "a cache of one rather than thrashing — which is why exact search "
            "above budget remains viable and is the measured high-dimension "
            "recommendation"
        ),
    )
    # Default raised from 200 in 2.2: the vector-scale lane's efs sweep on a
    # realistic (gaussian) distribution at dimension 1536 measured recall@10 of
    # 0.8531 at efs 200, 0.9875 at efs 400, and 1.0 at efs 800, with ~2ms of
    # added p50 per step. 400 is the evidence-backed high-dimension default;
    # 800 remains the maximum-recall recommendation.
    vector_ann_efs: int = Field(
        default=400,
        ge=1,
        description=(
            "Query-time HNSW candidate-list size (efs) for the embedded ANN "
            "vector path; the primary recall knob — higher values trade "
            "latency for recall. The effective value is never below the "
            "oversampled candidate count the query requests from the index"
        ),
    )
    vector_quantization: Literal["none", "int8"] = Field(
        default="none",
        description=(
            "Embedded vector index storage quantization: none (exact float64) or "
            "int8 (per-vector scales with float rerank of oversampled candidates)"
        ),
    )
    embedded_lock_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Maximum seconds to wait for the embedded projection's exclusive "
            "write lock during open and the startup write-lock probe. The "
            "embedded backend is single-writer; when a stale process holds the "
            "lock, acquisition fails fast with EmbeddedProjectionLockedError "
            "(triggering owner reap-and-retry then a graph-degraded fallback) "
            "instead of hanging the MCP server indefinitely"
        ),
    )

    embedded_store_bloat_min_bytes: int = Field(
        default=256 * 1024 * 1024,
        ge=0,
        description=(
            "Minimum on-disk size (bytes) before the embedded projection's "
            "pre-open bloat guard may quarantine it; 0 disables the guard. A "
            "pathologically bloated store (e.g. 397MB grown from ~500KB of "
            "event logs) hangs then crashes NATIVELY inside the engine open, "
            "which no exception handler can catch — so the guard must run "
            "before the open. The projection is derived state: quarantine "
            "moves it aside (never deletes) and it rebuilds from the log"
        ),
    )

    embedded_store_bloat_log_multiplier: float = Field(
        default=100.0,
        gt=0,
        description=(
            "Bloat threshold as a multiple of the sibling Eventloom JSONL "
            "bytes: a store both larger than embedded_store_bloat_min_bytes "
            "AND more than this multiple of its source logs is quarantined "
            "before open. Healthy projections run well under 10x their logs"
        ),
    )

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    reranker_provider: str = Field(
        default="none",
        description="Reranker provider: none, lexical, http, late-interaction-http, or openai",
    )
    reranker_url: str | None = Field(
        default=None,
        description="HTTP reranker endpoint URL",
    )
    reranker_api_key: str | None = Field(
        default=None,
        description="Optional bearer token for HTTP reranker endpoint",
    )
    reranker_api_key_file: str | None = Field(
        default=None,
        description="Path to a file containing the reranker bearer token",
    )
    openai_rerank_model: str = Field(
        default="gpt-5-mini",
        description="OpenAI-compatible chat model used for reranking",
    )

    # ------------------------------------------------------------------
    # Coordination
    # ------------------------------------------------------------------
    coordination_semantic_conflict_provider: str = Field(
        default="none",
        description="Coordination semantic conflict provider: none, lexical, or http",
    )
    coordination_semantic_conflict_url: str | None = Field(
        default=None,
        description="Hosted coordination semantic conflict adapter URL",
    )
    coordination_semantic_conflict_api_key: str | None = Field(
        default=None,
        description="Optional bearer token for hosted coordination semantic conflict adapter",
    )
    coordination_semantic_conflict_api_key_file: str | None = Field(
        default=None,
        description="Path to a file containing the hosted coordination semantic conflict bearer token",
    )
    coordination_semantic_conflict_min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum hosted semantic conflict confidence accepted into coordination state",
    )
    coordination_semantic_conflict_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Hosted semantic conflict adapter request timeout in seconds",
    )
    coordination_semantic_min_shared_subject_tokens: int = Field(
        default=2,
        ge=1,
        description="Minimum shared non-polarity subject tokens for lexical coordination semantic conflicts",
    )

    def model_post_init(self, __context: Any) -> None:
        """Load Docker/Kubernetes-style secret files after env parsing."""
        self._load_secret_file("NEO4J_PASSWORD", "neo4j_password", "neo4j_password_file")
        self._load_secret_file("MCP_ADMIN_TOKEN", "mcp_admin_token", "mcp_admin_token_file")
        self._load_secret_file(
            "MCP_REMOTE_AUTH_TOKEN",
            "mcp_remote_auth_token",
            "mcp_remote_auth_token_file",
        )
        self._load_secret_file(
            "MCP_OIDC_CLIENT_SECRET",
            "mcp_oidc_client_secret",
            "mcp_oidc_client_secret_file",
        )
        self._load_secret_file("OPENAI_API_KEY", "openai_api_key", "openai_api_key_file")
        self._load_secret_file("RERANKER_API_KEY", "reranker_api_key", "reranker_api_key_file")
        self._load_secret_file(
            "COORDINATION_SEMANTIC_CONFLICT_API_KEY",
            "coordination_semantic_conflict_api_key",
            "coordination_semantic_conflict_api_key_file",
        )
        self._load_secret_file(
            "PATHLIGHT_ACCESS_TOKEN",
            "pathlight_access_token",
            "pathlight_access_token_file",
        )

    def _load_secret_file(self, env_name: str, field_name: str, file_field_name: str) -> None:
        """Populate a sensitive field from ENV_NAME_FILE when direct env is absent."""
        file_env = f"{env_name}_FILE"
        file_path = os.getenv(file_env) or getattr(self, file_field_name)
        if not file_path:
            return
        current_value = getattr(self, field_name)
        default_value = type(self).model_fields[field_name].default
        if os.getenv(env_name) is not None or current_value not in (None, default_value):
            return
        try:
            value = Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"{file_env} could not be read: {file_path}") from exc
        object.__setattr__(self, field_name, value)

    @field_validator("plugins", mode="before")
    @classmethod
    def _parse_plugins(cls, value: object) -> list[str]:
        """Accept a comma-separated string (env) or a list of import specs."""
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list | tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        raise ValueError("plugins must be a string or a list of 'module:attr' strings")

    @model_validator(mode="after")
    def _validate_production_security(self) -> Settings:
        """Reject known-insecure production defaults."""
        if self.zaxy_env.lower() == "production":
            if self.projection_backend.casefold().strip() == "neo4j":
                if self.neo4j_password == "testpassword":
                    raise ValueError("NEO4J_PASSWORD must be overridden in production")
                if self.neo4j_uri.startswith("bolt://") and not self.neo4j_ca_cert:
                    raise ValueError("NEO4J_URI must use TLS or NEO4J_CA_CERT in production")
            if not self.mcp_admin_token:
                raise ValueError("MCP_ADMIN_TOKEN must be configured in production")
            if not self.remote_transport_auth_configured:
                raise ValueError(
                    "MCP_REMOTE_AUTH_TOKEN or complete MCP_OIDC_ISSUER/"
                    "MCP_OIDC_AUDIENCE/MCP_OIDC_JWKS_URL must be configured in production"
                )
        return self

    @property
    def remote_transport_auth_configured(self) -> bool:
        """Whether the remote MCP transport has authentication configured.

        True when either a static bearer token (``mcp_remote_auth_token``) or a
        complete OIDC configuration (issuer + audience + JWKS URL) is present.
        Used both by production-settings validation and by the CLI, which
        refuses to bind a non-loopback SSE host when this is False.
        """
        has_static = bool(self.mcp_remote_auth_token)
        has_oidc = all(
            (self.mcp_oidc_issuer, self.mcp_oidc_audience, self.mcp_oidc_jwks_url)
        )
        return has_static or has_oidc

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def eventloom_dir(self) -> Path:
        """Return the Eventloom directory as a Path."""
        return Path(self.eventloom_path)

    def eventloom_log(self, thread: str | None = None) -> Path:
        """Return the JSONL path for a given thread."""
        name = thread or self.eventloom_thread
        return eventlog_path(self.eventloom_dir, name)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Use this in application code to avoid re-parsing env vars.
    """
    return Settings()
