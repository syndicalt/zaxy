"""Tests for the public site and documentation coverage."""

from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_DOCS = [
    "docs/why-zaxy.md",
    "docs/getting-started.md",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/mcp.md",
    "docs/mcp-install-targets.md",
    "docs/eventloom.md",
    "docs/agent-events.md",
    "docs/hooks.md",
    "docs/codebase.md",
    "docs/graph-schema.md",
    "docs/retrieval.md",
    "docs/embeddings.md",
    "docs/security.md",
    "docs/operations.md",
    "docs/deployment.md",
    "docs/testing.md",
    "docs/benchmarks.md",
    "docs/benchmark-review.md",
    "docs/consolidation.md",
    "docs/api.md",
]


class LinkParser(HTMLParser):
    """Collect local links from an HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def test_public_site_has_product_positioning_and_required_sections() -> None:
    """The public site should explain the product, architecture, docs, and install path."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "<title>Zaxy - temporal memory for AI agents</title>" in html
    assert 'name="description"' in html
    assert 'property="og:image"' in html
    assert "Git for LLM memory" in html
    assert "Temporal memory for AI agents" in html
    assert "Eventloom log" in html
    assert "Neo4j temporal graph" in html
    assert "Memory Checkout" in html
    assert "Checkout diagnostics" in html
    assert "answerability" in html
    assert "required_action" in html
    assert "current_citation_count" in html
    assert "memory_capabilities" in html
    assert "deterministic capture" in html
    assert "local-codex" in html
    assert "NEXT_EVENT" in html
    assert "Pathlight" in html
    assert "memory_append" in html
    assert "memory_query" in html
    assert "scripts/release-check.sh --root ." in html
    assert "PyPI 0.3.1" in html
    assert "1167 tests" in html
    assert "92.13% coverage" in html

    for section_id in (
        "why",
        "architecture",
        "mcp",
        "retrieval",
        "backend",
        "dashboard",
        "security",
        "install",
        "docs",
    ):
        assert f'id="{section_id}"' in html


def test_mcp_docs_show_memory_checkout_consumption_contract() -> None:
    """MCP docs should show how models consume checkout quality and feedback fields."""
    text = Path("docs/mcp.md").read_text(encoding="utf-8")

    assert '"answerability": "answer_from_memory"' in text
    assert '"current_citation_count": 1' in text
    assert '"required_action": null' in text
    assert '"tool": "memory_feedback"' in text
    assert '"feedback": "used"' in text
    assert '"answerability": "refresh_recommended"' in text


def test_docs_describe_skill_memory_contract_and_guardrail() -> None:
    """Docs should cover Skill Memory event, MCP, checkout, and benchmark contracts."""
    agent_events = Path("docs/agent-events.md").read_text(encoding="utf-8")
    mcp = Path("docs/mcp.md").read_text(encoding="utf-8")
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")

    assert "skill.proposed" in agent_events
    assert "skill.outcome_recorded" in agent_events
    assert "SkillVersion" in agent_events
    assert "memory_skill(action, skill_id, ...)" in mcp
    assert "Applicable Skills" in mcp
    assert "Skill Analytics" in mcp
    assert "rollback" in agent_events
    assert "contradiction analytics" in benchmarks
    assert "Skill Memory changes must pass the full 500-question guardrail" in benchmarks


def test_pggraph_backend_roadmap_records_contract_first_state() -> None:
    """Docs should keep pgGraph behind the projection contract and explicit backend selector."""
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    spec = Path("docs/superpowers/specs/2026-05-17-skill-memory-pggraph-evaluation-design.md").read_text(
        encoding="utf-8",
    )

    assert "Skill Memory procedural world-model layer" in agents
    assert "Projection backend contract and Neo4j factory" in agents
    assert "Experimental pgGraph adapter behind `PROJECTION_BACKEND=pggraph`" in agents
    assert (
        "pgGraph adapter supports projection, exact search, keyword search, vector search, invalidation, and traversal"
        in spec
    )
    assert "pgGraph vector search uses pgvector ranking" in spec
    assert "PGGRAPH_INTEGRATION_DSN" in spec
    assert 'pip install "zaxy-memory[pggraph]"' in spec
    assert "version 0.1.0, PostgreSQL 13-18 support, and alpha status" in spec
    assert "Projection backend changes must pass the full 500-question guardrail" in benchmarks
    assert "vector search uses pgvector" in benchmarks
    assert "longmemeval-100-pggraph-comparison" in benchmarks
    assert "longmemeval-100-neo4j-comparison" in benchmarks
    assert "longmemeval-500-pggraph-comparison" in benchmarks
    assert "longmemeval-500-neo4j-current-checkout" in benchmarks
    assert "pgGraph checkout" in benchmarks
    assert "0.910" in benchmarks
    assert "0.714" in benchmarks
    assert "0.632" in benchmarks
    assert "0.958" in benchmarks
    assert "same-harness Neo4j checkout control" in benchmarks
    assert "Neo4j checkout" in benchmarks
    assert "0.930" in benchmarks
    assert "0.626" in benchmarks
    assert "no longer shows a pgGraph-specific quality" in benchmarks
    assert "regression" in benchmarks
    assert "zaxy reproject" in spec
    assert "zaxy init --projection-backend pggraph --pggraph-repo" in spec
    assert "PGGRAPH_REPO" in spec
    assert "--projection-backend pggraph" in spec
    assert "--reset-projection" in spec
    assert "failure recovery" in spec
    assert "zaxy memory status --graph --projection-backend pggraph" in spec
    assert "zaxy memory inferred-status --projection-backend pggraph" in spec


def test_zero_friction_runtime_roadmap_sets_frontier_bar() -> None:
    """Roadmap should target frontier memory quality, not a convenience-only backend swap."""
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    architecture = Path("docs/architecture.md").read_text(encoding="utf-8")
    roadmap = Path("docs/zero-friction-runtime-roadmap.md").read_text(encoding="utf-8")

    assert "frontier-grade memory" in roadmap
    assert "zaxy init" in roadmap
    assert "Bare `zaxy init`" in roadmap
    assert "PROJECTION_BACKEND=embedded" in roadmap
    assert "Kuzu" in roadmap
    assert "LatticeDB" in roadmap
    assert "Memory Activation Layer" in roadmap
    assert "activation efficiency" in roadmap
    assert "ruthless token discipline" in roadmap
    assert "invent new techniques" in roadmap
    assert "same-harness benchmark output compares embedded, LatticeDB, Neo4j" in roadmap
    assert "LatticeDB candidate gate status: failed current active-backend gate" in roadmap
    assert "Embedded graph-traversal gate status: passed focused 10-subject smoke" in roadmap
    assert "40-question LongMemEval-compatible embedded shootout" in roadmap
    assert "100 dashboard nodes and 100 dashboard edges" in roadmap
    assert "roughly 120-second projection/rebuild" in roadmap
    assert "cost to roughly 10 seconds" in roadmap
    assert "100-query scale evidence status: answer-ready quality passed" in roadmap
    assert "embedded\nscale guardrail now passes" in roadmap
    assert "answer-ready contract scored `Answer@5=1.0`" in " ".join(roadmap.split())
    assert "`Recall@5=1.0`, first checkout" in " ".join(roadmap.split())
    assert "raw retrieve improved to `Recall@5=0.99`" in " ".join(roadmap.split())
    assert "BM25 scored `Answer@5=0.52`" in " ".join(roadmap.split())
    assert "dashboard parity is wired" in roadmap
    assert "Latest reminder" in roadmap
    assert "memory_activation.latest_reminder" in roadmap
    assert "write_instructions.memory_activation" in roadmap
    assert "required tool before roadmap, implementation, release, review, resume" in roadmap
    assert "high-context sessions" in roadmap
    assert "fresh checkout before work starts" in roadmap
    assert "Embedded infra check status: wired" in roadmap
    assert "zaxy status --projection-backend embedded" in roadmap
    assert "zaxy memory status --graph --projection-backend embedded --embedded-graph-path" in roadmap
    assert "zaxy memory inferred-status --session-id <session>" in roadmap
    assert "zaxy reproject .eventloom/<session>.jsonl --session-id <session>" in roadmap
    assert "MCP embedded runtime status: wired" in roadmap
    assert "local-embedded-codex" in roadmap
    assert "zero-friction-runtime-roadmap.md" in architecture
    assert "Build the zero-friction embedded graph runtime path" in agents
    assert "Memory Activation Layer" in agents


def test_embedded_runtime_docs_do_not_frame_current_kuzu_as_prototype() -> None:
    """Embedded/Kuzu docs should describe the promoted runtime as first-class."""
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "AGENTS.md",
            "docs/zero-friction-runtime-roadmap.md",
            "docs/superpowers/plans/2026-05-20-zero-friction-embedded-graph-runtime.md",
        )
    )

    assert "embedded runtime: passed and promoted to default" in combined
    assert "Build the zero-friction embedded graph runtime path" in combined
    assert "embedded Kuzu is the default projection backend" in combined
    assert "embedded graph runtime prototype" not in combined
    assert "embedded prototype:" not in combined
    assert "prototype embedded graph runtime" not in combined
    assert "Prototype Gates" not in combined
    assert "Prototype status:" not in combined
    assert "document prototype status and gates" not in combined
    assert "eligible for broader benchmark" not in combined
    assert "Kuzu optional dependency" not in combined
    assert "optional dependency group `embedded" not in combined
    assert "A prototype that only works outside Zaxy is not enough" not in combined
    assert "The prototype should report" not in combined


def test_docs_show_embedded_local_profile_option() -> None:
    """Local profile docs should expose the embedded no-sidecar profile path."""
    getting_started = Path("docs/getting-started.md").read_text(encoding="utf-8")
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")
    embeddings = Path("docs/embeddings.md").read_text(encoding="utf-8")

    assert "zaxy local-profile --projection-backend embedded --output .env.local" in getting_started
    assert "zaxy local-profile --projection-backend embedded --output .env.local" in runbook
    assert "zaxy local-profile --projection-backend embedded" in embeddings


def test_backend_shootout_contract_is_documented_and_scripted() -> None:
    """Backend shootout should define the embedded-vs-sidecar measurement contract."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    normalized_benchmarks = " ".join(benchmarks.split())
    script = Path("scripts/backend-shootout.py").read_text(encoding="utf-8")
    guardrail = Path("scripts/check-backend-shootout.py").read_text(encoding="utf-8")

    for phrase in (
        "Backend Shootout",
        "embedded",
        "LatticeDB",
        "Neo4j",
        "pgGraph",
        "BM25",
        "cold bootstrap time",
        "first useful init time",
        "first checkout time",
        "append-to-projection",
        "checkout p95",
        "checkout p99",
        "returned tokens",
        "injected tokens",
        "citation coverage",
        "resident memory",
        "on-disk footprint",
        "recovery time",
        "report schema version",
        "source fingerprints",
        "workload fingerprints",
        "event/query counts",
    ):
        assert phrase in benchmarks
    assert "--eventloom-path" in script
    assert "latticedb" in script
    assert "--session-id" in script
    assert "--backends" in script
    assert "--queries-file" in script
    assert "--output" in script
    assert "Defaults run embedded plus BM25 without optional sidecar infrastructure." in script
    assert "check-backend-shootout.py" in benchmarks
    assert "--require-report-metadata" in benchmarks
    assert "--require-report-metadata" in guardrail
    assert "--require-markdown-report" in benchmarks
    assert "--require-markdown-report" in guardrail
    assert "--verify-report-fingerprints" in benchmarks
    assert "--verify-report-fingerprints" in guardrail
    assert "--require-labeled-metrics" in benchmarks
    assert "--require-dashboard-source embedded=embedded" in benchmarks
    assert "--min-quality-per-1k-injected-tokens embedded=1.0" in benchmarks
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=1.0" in benchmarks
    assert "--min-projection-events-per-second embedded=40" in benchmarks
    assert "--max-first-useful-init-ms embedded=15000" in benchmarks
    assert "--max-resident-memory-delta-bytes embedded=768000000" in benchmarks
    assert "--max-on-disk-footprint-bytes embedded=256000000" in benchmarks
    assert "--max-dashboard-graph-load-ms embedded=250" in benchmarks
    assert "--max-checkout-p99-ms embedded=25" in benchmarks
    assert "--max-exact-p99-ms embedded=10" in benchmarks
    assert "--max-keyword-p99-ms embedded=5" in benchmarks
    assert "--max-vector-p99-ms embedded=5" in benchmarks
    assert "--max-traversal-p99-ms embedded=5" in benchmarks
    assert "--max-dashboard-graph-load-ms embedded=500" in benchmarks
    assert "--max-rebuild-recovery-ms embedded=15000" in benchmarks
    assert "--max-checkout-p95-ms embedded=100" in benchmarks
    assert "--min-quality-per-1k-returned-tokens embedded=0.10" in benchmarks
    assert "--min-answer-at-5-per-1k-returned-tokens embedded=0.10" in benchmarks
    assert "--min-quality-per-1k-injected-tokens embedded=0.10" in benchmarks
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=0.10" in benchmarks
    assert "--max-exact-p95-ms embedded=15" in benchmarks
    assert "--max-keyword-p95-ms embedded=75" in benchmarks
    assert "--max-vector-p95-ms embedded=25" in benchmarks
    assert "--max-traversal-p95-ms embedded=10" in benchmarks
    assert "--min-recall-at-5 0.90" in benchmarks
    assert "--max-first-useful-init-ms embedded=45000" in benchmarks
    assert "--max-resident-memory-delta-bytes embedded=1700000000" in benchmarks
    assert "--max-on-disk-footprint-bytes embedded=512000000" in benchmarks
    assert "--max-rebuild-recovery-ms embedded=45000" in benchmarks
    assert "--max-checkout-p95-ms embedded=200" in benchmarks
    assert "--min-quality-per-1k-returned-tokens embedded=0.15" in benchmarks
    assert "--min-answer-at-5-per-1k-returned-tokens embedded=0.15" in benchmarks
    assert "--min-quality-per-1k-injected-tokens embedded=0.15" in benchmarks
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=0.15" in benchmarks
    assert "--max-keyword-p95-ms embedded=20" in benchmarks
    assert "--max-checkout-p99-ms embedded=250" in benchmarks
    assert "--max-exact-p99-ms embedded=12" in benchmarks
    assert "--max-keyword-p99-ms embedded=15" in benchmarks
    assert "--max-vector-p99-ms embedded=20" in benchmarks
    assert "Backend shootout guardrail passed" in guardrail
    assert "min-projection-events-per-second" in guardrail
    assert "max-rebuild-recovery-ms" in guardrail
    assert "min-quality-per-1k-returned-tokens" in guardrail
    assert "min-answer-at-5-per-1k-returned-tokens" in guardrail
    assert "min-quality-per-1k-injected-tokens" in guardrail
    assert "min-answer-at-5-per-1k-injected-tokens" in guardrail
    assert "max-keyword-p95-ms" in guardrail
    assert "max-keyword-p99-ms" in guardrail
    report = json.loads(Path("reports/backend-shootout/backend-shootout.json").read_text(encoding="utf-8"))
    if any(row["status"] != "ok" for row in report["summaries"]):
        assert "with all rows passing" not in benchmarks
        assert "error rows" in benchmarks
    assert "parked candidate" in benchmarks
    assert "backend-shootout-graph-traversal-embedded-after-carry-forward" in benchmarks
    assert "longmemeval-40-backend-shootout.json" in benchmarks
    assert "longmemeval-100-backend-shootout.json" in benchmarks
    assert "medium-scale backend evidence" in benchmarks
    assert "Answer@5=0.575" in benchmarks
    assert "100 nodes and 100 edges" in benchmarks
    assert "cold bootstrap `225.93ms`" in benchmarks
    assert "`10.55ms`" in benchmarks
    assert "append-to-projection p95 `24.674ms`" in benchmarks
    assert "resident memory delta" in benchmarks
    assert "on-disk footprint" in benchmarks
    assert "`9347.717ms`" in benchmarks
    assert "projection throughput `57.007` events/sec" in normalized_benchmarks
    assert "vector retrieval enabled" in benchmarks
    assert "answer-ready synthesis now closes the answer-surface gap" in benchmarks
    assert "lane p95s of exact `0.007ms`, keyword `3.285ms`" in normalized_benchmarks
    assert "roughly 10" in benchmarks
    assert "100-query scale evidence" in benchmarks
    assert "The `answer_ready` row scored `Answer@5=0.99` and `Recall@5=1.0`" in normalized_benchmarks
    assert "cold bootstrap `421.649ms`" in normalized_benchmarks
    assert "first useful init `29620.186ms`" in normalized_benchmarks
    assert "append-to-projection p95 `26.931ms`" in normalized_benchmarks
    assert "Answer@5 per 1k injected tokens `0.2889`" in normalized_benchmarks
    assert "projection throughput `53.393` events/sec" in normalized_benchmarks
    assert "first answer-ready checkout does not pay" in normalized_benchmarks
    assert "BM25 scored `Answer@5=0.52`" in benchmarks
    assert "retrieve path now clears a stricter `Recall@5=0.90` release floor" in normalized_benchmarks


def test_install_docs_offer_zero_surprise_first_run_path() -> None:
    """Install docs should make local setup verifiable without guessing where state went."""
    readme = Path("README.md").read_text(encoding="utf-8")
    getting_started = Path("docs/getting-started.md").read_text(encoding="utf-8")
    site = Path("site/index.html").read_text(encoding="utf-8")

    combined = "\n".join([readme, getting_started, site])
    assert "Five-minute local smoke test" in combined
    assert "pipx install zaxy-memory" in combined
    assert "zaxy init" in combined
    assert "Bare `zaxy init` now expands to the local embedded Codex path" in combined
    assert "zaxy init --capture start" in combined
    assert "zaxy init . --domain my-project --preset local-codex --capture start --infra check" not in combined
    assert "zaxy init . --domain my-project --preset local-embedded-codex --capture start" not in combined
    assert "no external graph service" in combined
    assert ".eventloom/" in combined
    assert "zaxy memory log --eventloom-path .eventloom --limit 5" in combined
    assert "zaxy memory bootstrap --eventloom-path .eventloom" in combined
    assert "zaxy doctor --eventloom-path .eventloom" in combined
    assert "MCP config" in combined


def test_public_site_reflects_current_onboarding_and_runtime_surfaces() -> None:
    """The public site should explain the current install, backend, and dashboard surfaces."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "What happens when you run init" in html
    assert "writes `.env.local`" in html
    assert "records session genesis and heartbeat" in html
    assert "prints the MCP command or config path" in html
    assert "Embedded Kuzu is the default production graph projection" in html
    assert "pgGraph is experimental" in html
    assert "Bare zaxy init" in html
    assert "selects the embedded projection" in html
    assert "local-embedded-codex" not in html
    assert "PROJECTION_BACKEND=pggraph" in html
    assert "zaxy reproject --projection-backend pggraph --reset-projection" in html
    assert "Read-only local dashboard" in html
    assert "zaxy dashboard --host 127.0.0.1 --port 8765" in html
    assert "--projection-backend pggraph --pggraph-dsn" in html
    assert "Eventloom sessions" in html
    assert "graph projection" in html
    assert "Checkout diagnostics" in html


def test_public_site_benchmark_claims_use_current_full_set_guardrails() -> None:
    """The public site should lead with current reproducible floors, not stale headline-only claims."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "Full 500-question LongMemEval-compatible guardrail" in html
    assert "0.724" in html
    assert "0.628" in html
    assert "0.972" in html
    assert "p95" in html
    assert "1472.11 ms" in html
    assert "100-question headline remains archived evidence" in html
    assert "0.970" in html
    assert "0.950" in html
    assert "pgGraph remains experimental" in html
    assert "0.958" in html
    assert "0.714" in html
    assert "PyPI 0.2.1" not in html
    assert "1005 tests" not in html
    assert "92.04% coverage" not in html


def test_why_zaxy_doc_explains_markdown_vector_tradeoffs() -> None:
    """The docs should explain why Zaxy is heavier than markdown or vector memory."""
    text = Path("docs/why-zaxy.md").read_text(encoding="utf-8").casefold()

    assert "markdown" in text
    assert "vector" in text
    assert "temporal" in text
    assert "provenance" in text
    assert "multi-hop" in text
    assert "pipx install zaxy-memory" in text
    assert "zaxy init" in text
    assert "--preset local-codex" not in text
    assert "--session-id my-project-default" not in text
    assert "neo4j" in text
    assert "pggraph" in text


def test_framework_integration_docs_record_next_hardening_target() -> None:
    """Docs should pin the native-preview learning into a maintained UX target."""
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    integrations = Path("docs/integrations.md").read_text(encoding="utf-8")
    competitive = Path("docs/competitive-positioning.md").read_text(encoding="utf-8")

    assert "Common native-preview adapter contract" in agents
    assert "Use LangGraph and CrewAI native-preview usage" not in agents
    assert "zaxy integrations --recommendation --json" in integrations
    assert "common-native-preview-contract" in integrations
    assert "model-facing UX hardening" in integrations
    assert "AutoGen remains template-only" in integrations
    assert "common native-preview payload contract" in competitive


def test_docs_describe_memory_persistence_hardening() -> None:
    """Docs should describe reminder policy, hooks, dashboard, and middleware."""
    mcp = Path("docs/mcp.md").read_text(encoding="utf-8")
    hooks = Path("docs/hooks.md").read_text(encoding="utf-8")
    integrations = Path("docs/integrations.md").read_text(encoding="utf-8")
    site = Path("site/index.html").read_text(encoding="utf-8")

    assert "memory.reminder.suggested" in mcp
    assert "memory.reminder.suggested" in hooks
    assert "create_langgraph_memory_checkout_node" in integrations
    assert "create_crewai_memory_checkout_step" in integrations
    assert "zaxy_autogen_context" in integrations
    assert "Memory Checkout before replying" in integrations
    assert "Last checkout" in site


def test_full_set_guardrail_docs_distinguish_legacy_and_same_harness_floors() -> None:
    """Docs should not mix the legacy limit=10 floor with current backend-eval floors."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    testing = Path("docs/testing.md").read_text(encoding="utf-8")
    competitive = Path("docs/competitive-positioning.md").read_text(encoding="utf-8")
    retrieval = Path("docs/retrieval.md").read_text(encoding="utf-8")
    combined = "\n".join([benchmarks, testing, competitive, retrieval])

    assert "Legacy limit=10 full-set floor" in benchmarks
    assert "Current same-harness backend-evaluation floor" in benchmarks
    assert "0dc36a139bb9a4fdc7c6cd34400737a58a1eb7410517341f015e9fbfc76ed854" in combined
    assert "longmemeval-500-neo4j-current-checkout/live-benchmark.json" in testing
    assert "--min-mean-score 0.714" in testing
    assert "--min-answer-recall-at-5 0.626" in testing
    assert "--min-recall-at-5 0.958" in testing
    assert "legacy `limit=10`" in competitive
    assert "current same-harness `limit=5`" in competitive
    assert "current backend-evaluation floor" in retrieval


def test_memory_checkout_docs_and_site_match_golden_contract_fixture() -> None:
    """Published checkout examples should stay aligned with the canonical tool contract."""
    fixture = json.loads(Path("docs/examples/memory-checkout-contract.json").read_text(encoding="utf-8"))
    answerable = fixture["answerable"]
    refresh = fixture["refresh_recommended"]
    docs = Path("docs/mcp.md").read_text(encoding="utf-8")
    site = Path("site/index.html").read_text(encoding="utf-8")

    assert "docs/examples/memory-checkout-contract.json" in docs
    assert answerable["quality"]["answerability"] == "answer_from_memory"
    assert answerable["diagnostics"]["current_citation_count"] == 1
    assert answerable["quality"]["required_action"] is None
    assert answerable["guidance"]["feedback"]["tool"] == "memory_feedback"
    assert answerable["guidance"]["feedback"]["payloads"][0]["feedback"] == "used"
    assert refresh["quality"]["answerability"] == "refresh_recommended"
    assert refresh["quality"]["required_action"]["tool"] == "memory_checkout"

    shared_contract_fields = (
        f'"answerability": "{answerable["quality"]["answerability"]}"',
        f'"current_citation_count": {answerable["diagnostics"]["current_citation_count"]}',
        f'"tool": "{answerable["guidance"]["feedback"]["tool"]}"',
    )
    docs_only_contract_fields = (
        f'"answerability": "{refresh["quality"]["answerability"]}"',
    )
    for expected in shared_contract_fields:
        assert expected in docs
        assert expected in site
    for expected in docs_only_contract_fields:
        assert expected in docs


def test_public_site_benchmark_claim_is_scoped_to_fixture() -> None:
    """Benchmark copy should not overclaim against broad markdown/vector systems."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "Benchmark evidence" in html
    assert "LongMemEval-compatible" in html
    assert "BM25 baseline" in html
    assert "MemPalace" in html
    assert "Mem0" in html
    assert "Agent Memory" in html
    assert "text-embedding-3-small" in html
    assert "1.000" in html
    assert "+0.480" in html
    assert "0.950" in html
    assert "0.972" in html
    assert "0.840" in html
    assert "0.770" in html
    assert "650 paired queries" in html
    assert "external disclosures" in html
    assert "not same-harness results" in html
    assert 'class="benchmark-metrics"' in html
    assert 'class="benchmark-card benchmark-card-primary"' in html
    assert 'class="benchmark-comparison"' in html
    assert 'class="evidence-badge evidence-badge-local"' in html
    assert 'class="evidence-badge evidence-badge-external"' in html
    assert 'class="benchmark-links"' in html
    assert "reports/benchmarks/live-benchmark.md" in html
    assert "reports/benchmarks/longmemeval-100-comparison/live-benchmark.md" in html
    assert "docs/benchmarks.html" in html
    assert "docs/benchmark-review.html" in html
    assert "production-grade vector RAG" not in html
    assert "destroyed" not in html.casefold()


def test_benchmark_docs_disclose_harness_external_claims_and_sources() -> None:
    """Benchmark docs should separate Zaxy-run evidence from competitor disclosures."""
    text = Path("docs/benchmarks.md").read_text(encoding="utf-8")

    assert "LongMemEval-compatible" in text
    assert "0.970" in text
    assert "1.000" in text
    assert "0.540" in text
    assert "0.840" in text
    assert "BM25" in text
    assert "same-harness" in text
    assert "Approx tokens" in text
    assert "external disclosures" in text
    assert "MemPalace" in text
    assert "96.6%" in text
    assert "98.4%" in text
    assert "Agent Memory" in text
    assert "95.2%" in text
    assert "Mem0" in text
    assert "+26% Accuracy" in text
    assert "https://www.agent-memory.dev/" in text
    assert "https://github.com/MemPalace/mempalace/blob/develop/benchmarks/BENCHMARKS.md" in text
    assert "https://github.com/mem0ai/mem0/blob/main/LLM.md" in text
    assert "../reports/benchmarks/live-benchmark.md" in text
    assert "../reports/benchmarks/longmemeval-100-comparison/live-benchmark.md" in text
    assert "not same-harness results" in text


def test_benchmark_docs_record_competitor_adapter_feasibility() -> None:
    """Docs should say which competitor adapters are buildable and which remain disclosures."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    positioning = Path("docs/competitive-positioning.md").read_text(encoding="utf-8")
    combined = f"{benchmarks}\n{positioning}"

    assert "Same-Harness Adapter Feasibility" in combined
    assert "MemPalace" in combined
    assert "adapter candidate" in combined
    assert "benchmarks/longmemeval_bench.py" in combined
    assert "Mem0" in combined
    assert "benchmark harness candidate" in combined
    assert "requires Docker" in combined
    assert "Agent Memory" in combined
    assert "external disclosure only" in combined
    assert "No same-harness adapter should be published without" in combined


def test_public_longmemeval_reports_keep_bm25_tradeoff_baseline() -> None:
    """Public LongMemEval reports should include BM25 latency and token tradeoffs."""
    report_paths = [
        path
        for path in Path("reports/benchmarks").glob("*/live-benchmark.json")
        if "longmemeval" in path.as_posix()
    ]
    report_paths.append(Path("reports/benchmarks/live-benchmark.json"))

    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        workload = payload.get("workload") or {}
        if "longmemeval" not in str(workload.get("version", "")).casefold():
            continue
        summaries = {
            str(summary.get("backend")): summary
            for summary in payload.get("summaries", [])
            if isinstance(summary, dict)
        }
        assert "bm25" in summaries, f"{report_path} is missing the BM25 baseline"
        bm25 = summaries["bm25"]
        assert isinstance(bm25.get("latency_ms_p95"), int | float)
        assert isinstance(bm25.get("mean_returned_bytes"), int | float)
        assert isinstance(bm25.get("mean_approx_tokens"), int | float)

        markdown_path = report_path.with_suffix(".md")
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "| bm25 |" in markdown.casefold()
        assert "p95 ms" in markdown
        assert "Returned bytes" in markdown
        assert "Approx tokens" in markdown


def test_public_site_links_to_all_core_docs() -> None:
    """The public site should expose the complete documentation set."""
    html = Path("site/index.html").read_text(encoding="utf-8")
    parser = LinkParser()
    parser.feed(html)

    for doc in REQUIRED_DOCS:
        rendered_doc = str(Path(doc).with_suffix(".html"))
        assert rendered_doc in parser.links
        assert doc not in parser.links
    assert "../docs/architecture.md" not in parser.links


def test_docs_describe_incremental_context_refresh_and_backend_reconciliation() -> None:
    """Docs should explain source refresh and stale projection retirement."""
    getting_started = Path("docs/getting-started.md").read_text(encoding="utf-8")
    codebase = Path("docs/codebase.md").read_text(encoding="utf-8")
    eventloom = Path("docs/eventloom.md").read_text(encoding="utf-8")

    assert "zaxy refresh-context" in getting_started
    assert "--projection-backend pggraph" in getting_started
    assert "retire stale projection rows" in codebase.replace("\n", " ")
    assert "source.changed" in eventloom
    assert "projection.retired" in eventloom


def test_operator_docs_describe_backend_release_gates() -> None:
    """Operator-facing release docs should include backend shootout and scale gates."""
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")
    operations = Path("docs/operations.md").read_text(encoding="utf-8")
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")
    getting_started = Path("docs/getting-started.md").read_text(encoding="utf-8")

    for text in (deployment, operations, runbook, getting_started):
        normalized = " ".join(text.split())
        assert "backend shootout" in normalized
        assert "100-query" in normalized
        assert "injected-token" in normalized


def test_public_site_docs_are_rendered_html_not_raw_markdown() -> None:
    """Published docs links should stay on the site as rendered HTML pages."""
    for doc in REQUIRED_DOCS:
        rendered_path = Path("site") / Path(doc).with_suffix(".html")
        html = rendered_path.read_text(encoding="utf-8")
        title = Path(doc).read_text(encoding="utf-8").splitlines()[0].lstrip("# ")

        assert rendered_path.exists(), doc
        assert "<!doctype html>" in html
        assert "<h1" in html
        assert title in html
        assert f'href="{Path(doc).name}"' not in html


def test_site_docs_generator_keeps_rendered_pages_current() -> None:
    """Generated docs should match the checked-in markdown sources."""
    result = subprocess.run(
        ["python", "scripts/build-site-docs.py", "--check"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_site_local_links_resolve() -> None:
    """All local public-site links should resolve to committed files or page anchors."""
    site_path = Path("site/index.html")
    html = site_path.read_text(encoding="utf-8")
    parser = LinkParser()
    parser.feed(html)
    anchors = set(re.findall(r'id="([^"]+)"', html))

    for href in parser.links:
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        path_part, _, anchor = href.partition("#")
        if path_part:
            if (path_part.startswith("docs/") and path_part.endswith(".html")) or path_part == "README.html":
                target = (site_path.parent / path_part).resolve()
            elif path_part.startswith(("docs/", "README.md", "reports/")):
                target = Path(path_part).resolve()
            else:
                target = (site_path.parent / path_part).resolve()
            assert target.exists(), href
        if anchor:
            assert anchor in anchors or path_part, href


def test_required_docs_are_substantial_and_cross_linked() -> None:
    """Core docs should be long enough to be useful and link back to related topics."""
    for doc in REQUIRED_DOCS:
        path = Path(doc)
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# "), doc
        assert len(text.split()) >= 250, doc
        assert "README.md" in text or "runbook.md" in text or "site/index.html" in text, doc


def test_agents_roadmap_records_dual_clean_repo_uat() -> None:
    """Roadmap should track Codex and Claude Code clean-workspace UAT coverage."""
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "Dual clean-repo Codex and Claude Code UAT" in agents
    assert "fresh Codex and Claude Code workspaces" not in agents


def test_docs_reference_public_benchmark_guardrail_script() -> None:
    """Docs and roadmap should expose the clean-checkout benchmark guardrail command."""
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    testing = Path("docs/testing.md").read_text(encoding="utf-8")

    assert "Public benchmark guardrail script" in agents
    assert "scripts/benchmark-guardrails.sh" in benchmarks
    assert "scripts/benchmark-guardrails.sh" in testing
    assert "cached LongMemEval dataset" in testing


def test_hooks_docs_explain_capture_readiness() -> None:
    """Hook docs should explain the automatic-capture readiness signal."""
    text = Path("docs/hooks.md").read_text(encoding="utf-8")

    assert "capture readiness" in text
    assert "capture_health" in text
    assert "active_observation_types" in text
    assert "missing_observation_types" in text
    assert "zaxy hook-status --json" in text


def test_docs_validation_script_checks_site_and_markdown_links(tmp_path: Path) -> None:
    """The docs validation gate should fail fast when a local doc link is broken."""
    root = tmp_path / "project"
    docs = root / "docs"
    site = root / "site"
    docs.mkdir(parents=True)
    site.mkdir()
    (docs / "ok.md").write_text("# OK\n\nSee [missing](missing.md).\n", encoding="utf-8")
    (site / "index.html").write_text(
        '<!doctype html><html><body><a href="../docs/ok.md">OK</a></body></html>',
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", "scripts/validate-docs.sh", "--root", str(root)],
        cwd=Path.cwd(),
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "missing.md" in result.stderr


def test_release_gate_runs_docs_validation() -> None:
    """The go-live gate should include docs/site and backend shootout validation."""
    script = Path("scripts/release-check.sh").read_text(encoding="utf-8")
    testing = Path("docs/testing.md").read_text(encoding="utf-8")

    assert 'DOCS_CMD="scripts/validate-docs.sh"' in script
    assert '"${DOCS_CMD}" --root "${ROOT}"' in script
    assert "BACKEND_SHOOTOUT_CMD" in script
    assert "BACKEND_PERFORMANCE_CMD" in script
    assert "BACKEND_SCALE_CMD" in script
    assert "scripts/check-backend-shootout.py" in script
    assert 'bash -c "${BACKEND_SHOOTOUT_CMD}"' in script
    assert 'bash -c "${BACKEND_PERFORMANCE_CMD}"' in script
    assert 'bash -c "${BACKEND_SCALE_CMD}"' in script
    assert (
        "backend-shootout.json --require-report-metadata --require-markdown-report --require-query-results "
        "--require-git-tracked-inputs --verify-report-fingerprints --require-backends embedded,bm25"
    ) in script
    assert script.count("--require-query-results") >= 3
    assert script.count("--require-git-tracked-inputs") >= 3
    assert "--forbid-backends neo4j,pggraph,latticedb" in script
    assert "--forbid-backends latticedb" not in script
    assert "--min-quality-per-1k-injected-tokens embedded=1.0" in script
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=1.0" in script
    assert "longmemeval-100-backend-shootout.json" in script
    assert "--max-checkout-p95-ms embedded=200" in script
    assert "--min-quality-per-1k-injected-tokens embedded=0.15" in script
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=0.15" in script
    assert "backend shootout" in testing
    assert "medium-scale embedded runtime evidence" in testing
    assert "100-query embedded scale evidence" in testing
    assert "--require-query-results" in testing
    assert "--require-git-tracked-inputs" in testing
    assert "--max-cold-bootstrap-ms embedded=250" in testing
    assert "activation efficiency" in testing
    assert "--min-activation-rate" in testing
    assert "--max-checkout-prompt-tokens" in testing
    assert "--min-checkout-facts-per-1k-tokens" in testing
    assert "scripts/beta-uat.sh" in testing
    assert "bare embedded init" in testing
    assert "PROJECTION_BACKEND=embedded" in testing
    assert "NEO4J_AUTO_START=false" in testing
    assert "zaxy memory status --eventloom-path .eventloom --graph" in testing
    assert "zaxy memory inferred-status" in testing
    assert "zaxy reproject" in testing
    assert "scripts/check-backend-shootout.py" in testing
    assert "--min-quality-per-1k-returned-tokens embedded=0.10" in script
    assert "--min-answer-at-5-per-1k-returned-tokens embedded=0.10" in script
    assert "--min-quality-per-1k-injected-tokens embedded=0.10" in script
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=0.10" in script
    assert "--max-cold-bootstrap-ms embedded=600" in script
    assert "--max-first-checkout-ms embedded=50" in script
    assert "--max-append-to-projection-p95-ms embedded=35" in script
    assert "--max-resident-memory-delta-bytes embedded=768000000" in script
    assert "--max-on-disk-footprint-bytes embedded=256000000" in script
    assert "--max-dashboard-graph-load-ms embedded=250" in script
    assert "--max-checkout-p99-ms embedded=25" in script
    assert "--max-exact-p99-ms embedded=10" in script
    assert "--max-keyword-p99-ms embedded=5" in script
    assert "--max-vector-p99-ms embedded=5" in script
    assert "--max-traversal-p99-ms embedded=5" in script
    assert "--max-dashboard-graph-load-ms embedded=500" in script
    assert "--max-keyword-p95-ms embedded=75" in script
    assert "--max-keyword-p99-ms embedded=40" in script
    assert "--max-vector-p99-ms embedded=35" in script
    assert "--max-traversal-p95-ms embedded=10" in script
    assert "--max-traversal-p99-ms embedded=10" in script
    assert "--min-quality-per-1k-returned-tokens embedded=0.10" in testing
    assert "--min-answer-at-5-per-1k-returned-tokens embedded=0.10" in testing
    assert "--min-quality-per-1k-injected-tokens embedded=0.10" in testing
    assert "--min-answer-at-5-per-1k-injected-tokens embedded=0.10" in testing
    assert "--min-quality-per-1k-injected-tokens embedded=0.15" in testing
    assert "--max-cold-bootstrap-ms embedded=600" in testing
    assert "--max-first-checkout-ms embedded=150" in testing
    assert "--max-append-to-projection-p95-ms embedded=40" in testing
    assert "--max-resident-memory-delta-bytes embedded=1700000000" in testing
    assert "--max-on-disk-footprint-bytes embedded=512000000" in testing
    assert "--max-dashboard-graph-load-ms embedded=500" in testing
    assert "--max-checkout-p95-ms embedded=200" in testing
    assert "--max-checkout-p99-ms embedded=250" in testing
    assert "--max-keyword-p95-ms embedded=20" in testing
    assert "--max-keyword-p99-ms embedded=15" in testing
    assert "--max-vector-p99-ms embedded=35" in testing


def test_embedded_runtime_docs_publish_read_index_warmup_contract() -> None:
    """Source docs and rendered site should preserve embedded read-index warmup behavior."""
    roadmap = Path("docs/zero-friction-runtime-roadmap.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/zero-friction-runtime-roadmap.html").read_text(encoding="utf-8")

    for text in (roadmap, rendered):
        normalized = " ".join(text.split())
        assert "Embedded read-index warmup is now part of the runtime path" in text
        assert "MemoryFabric.connect()" in text
        assert "current-entity, keyword, vector, and traversal indexes" in text
        assert "Eventloom verbatim source index" in text
        assert "retrieve()" in text
        assert "additional requested projection session" in text
        assert "at most once" in normalized


def test_operations_docs_center_embedded_default_and_optional_sidecars() -> None:
    """Operator docs should not describe Neo4j as mandatory default infrastructure."""
    runbook = Path("docs/runbook.md").read_text(encoding="utf-8")
    operations = Path("docs/operations.md").read_text(encoding="utf-8")
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")

    combined = "\n".join((runbook, operations, deployment))

    assert "Embedded Kuzu projection" in runbook
    assert "zaxy init" in runbook
    assert "PROJECTION_BACKEND=embedded" in runbook
    assert "optional sidecar" in combined
    assert "zaxy-memory[neo4j]" in combined
    assert "zaxy-memory[pathlight]" in combined
    assert "Neo4j** (core)" not in runbook
    assert "docker compose up -d neo4j" not in runbook
    assert "keep Neo4j healthy" not in operations
    assert "Neo4j must be reachable" not in deployment
    assert "Add hot caches only after benchmark evidence shows" not in runbook
    assert "Keep embedded read-index warmup and hot caches benchmark-gated" in runbook


def test_readme_and_api_docs_name_optional_infra_extras() -> None:
    """Docs should tell users which extras enable optional infrastructure."""
    readme = Path("README.md").read_text(encoding="utf-8")
    api = Path("docs/api.md").read_text(encoding="utf-8")

    combined = "\n".join((readme, api))
    assert "zaxy-memory[neo4j]" in combined
    assert "zaxy-memory[pathlight]" in combined
    assert "plain install uses embedded Kuzu" in combined


def test_readme_integration_compose_uses_explicit_profile() -> None:
    """README integration commands should match profile-gated sidecar compose."""
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "docker compose --profile integration up -d neo4j-test neo4j-tls" in readme
    assert "docker compose up -d neo4j-test" not in readme
    assert "docker compose up -d neo4j-tls" not in readme
    assert '`./scripts/setup.sh`,\nand `zaxy status`' in readme
    assert '`./scripts/setup.sh`,\nand `docker compose up -d`' not in readme


def test_benchmark_docs_describe_sidecar_free_default_shootout() -> None:
    """Backend shootout docs should not imply sidecars run by default."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")

    assert "The default active backend set is `embedded` and `bm25`." in benchmarks
    assert "explicit backend set" in benchmarks
    assert "The default active backend set is `embedded`, `neo4j`, `pggraph`, and `bm25`." not in benchmarks


def test_core_docs_use_backend_neutral_projection_language() -> None:
    """Core docs should describe embedded as default and Neo4j as optional."""
    docs = {
        "mcp": Path("docs/mcp.md").read_text(encoding="utf-8"),
        "embeddings": Path("docs/embeddings.md").read_text(encoding="utf-8"),
        "eventloom": Path("docs/eventloom.md").read_text(encoding="utf-8"),
        "retrieval": Path("docs/retrieval.md").read_text(encoding="utf-8"),
        "security": Path("docs/security.md").read_text(encoding="utf-8"),
        "graph_schema": Path("docs/graph-schema.md").read_text(encoding="utf-8"),
        "api": Path("docs/api.md").read_text(encoding="utf-8"),
    }
    combined = "\n".join(docs.values())

    assert "upserts the selected graph projection" in docs["mcp"]
    assert "The graph projection is Zaxy's structured reasoning layer" in docs["graph_schema"]
    assert "default implementation is embedded Kuzu" in docs["graph_schema"]
    assert "selected projection backend vector search" in docs["embeddings"]
    assert "optional Neo4j sidecar" in combined
    assert "set `NEO4J_AUTO_START=true`" in combined

    stale_phrases = [
        "upserts the Neo4j projection",
        "Neo4j vector search",
        "Neo4j is Zaxy's structured reasoning layer",
        "shared Neo4j database",
        "patching Neo4j directly",
        "rebuilds Neo4j projections",
        "Neo4j facts remain replayable",
        "Neo4j cannot be reached",
        "must match the Neo4j vector index",
        "Set `NEO4J_AUTO_START=false`",
    ]
    for phrase in stale_phrases:
        assert phrase not in combined


def test_optional_neo4j_index_script_is_idempotent_and_sidecar_scoped() -> None:
    """Manual Neo4j index helper should be safe to rerun and clearly optional."""
    cypher = Path("scripts/setup_neo4j_indexes.cypher").read_text(encoding="utf-8")
    graph_schema = Path("docs/graph-schema.md").read_text(encoding="utf-8")

    create_lines = [line for line in cypher.splitlines() if line.startswith("CREATE ")]
    assert create_lines
    assert all("IF NOT EXISTS" in line for line in create_lines)
    assert "optional Neo4j sidecar" in graph_schema
    assert "embedded Kuzu" in graph_schema


def test_runtime_docstrings_do_not_claim_neo4j_is_default_projection() -> None:
    """Runtime-facing module docs should match the embedded-first projection model."""
    runtime_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("src/zaxy/core.py", "src/zaxy/embedded_graph_store.py", "src/zaxy/pggraph_store.py")
    )

    assert "selected projection graph" in runtime_text
    assert "Connect to projection backend and tracer" in runtime_text
    assert "knowledge graph (Neo4j)" not in runtime_text
    assert "Connect to Neo4j and Pathlight" not in runtime_text
    assert "Neo4j remains the sidecar control backend until" not in runtime_text
    assert "Kuzu-backed embedded projection store." in runtime_text
    assert "projection store shell" not in runtime_text
    assert "Methods fail clearly until the Kuzu implementation lands" not in runtime_text


def test_readme_does_not_duplicate_production_secrets_section() -> None:
    """The README should stay concise and avoid repeated operational sections."""
    readme = Path("README.md").read_text(encoding="utf-8")

    assert readme.count("## Production Secrets") == 1


def test_rendered_planning_docs_do_not_claim_neo4j_is_current_default() -> None:
    """Rendered planning docs should not contradict the embedded-first default."""
    planning_docs = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (Path("docs/superpowers/plans"), Path("docs/superpowers/specs"))
        for path in root.glob("*.md")
    )

    assert "Current status: embedded Kuzu is the default projection backend" in planning_docs
    assert "Neo4j remains the default" not in planning_docs
    assert "Neo4j remains the default." not in planning_docs
    assert "Neo4j remains the default while" not in planning_docs
    assert "Neo4j remaining the default" not in planning_docs
    assert "default backend remains Neo4j" not in planning_docs
    assert "keeps `PROJECTION_BACKEND=neo4j` as the default" not in planning_docs
    assert "Neo4j remains the only production and published benchmark backend" not in planning_docs
    assert "preserves Neo4j as default" not in planning_docs
    assert "Neo4j is the production default" not in planning_docs
    assert "Keep PROJECTION_BACKEND=neo4j until" not in planning_docs


def test_github_pages_workflow_publishes_site_directory() -> None:
    """GitHub Pages should deploy the static site and docs under the project URL."""
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert "branches: [master]" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "cp -R site/. _site/" in workflow
    assert "python scripts/build-site-docs.py --check" in workflow
    assert "cp -R docs _site/docs" not in workflow
    assert "cp README.md _site/README.md" not in workflow
    assert "cp README.md _site/README.html" not in workflow
    assert "cp -R reports _site/reports" in workflow
    assert "cp -R reports _site/site/reports" not in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "path: _site" in workflow
    assert "actions/deploy-pages@v4" in workflow
