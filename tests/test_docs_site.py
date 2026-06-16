"""Tests for the public site and documentation coverage."""

from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_DOCS = [
    "docs/why-zaxy.md",
    "docs/announcements/zaxy-coordinate.md",
    "docs/announcements/zaxy-v1.0.md",
    "docs/coordinate-roadmap.md",
    "docs/getting-started.md",
    "docs/mcp-quickstart.md",
    "docs/coordinate-quickstart.md",
    "docs/first-run-validation.md",
    "docs/external-validation.md",
    "docs/architecture.md",
    "docs/integrations.md",
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
    "docs/consolidation.md",
    "docs/api.md",
    "docs/api-inventory.md",
    "docs/migration.md",
    "docs/stability-commitment.md",
    "docs/packet-analyzer.md",
    "docs/workspace-genesis.md",
    "docs/runbook.md",
]

ARCHIVED_DOCS = [
    "docs/archive/benchmark-contributions.md",
    "docs/archive/benchmark-review.md",
    "docs/archive/competitive-positioning.md",
    "docs/archive/experimental-associative-memory.md",
    "docs/archive/memory-is-purpose-zaxy-analysis.md",
    "docs/archive/release-validation-checklist.md",
    "docs/archive/synthesis-context-research.md",
    "docs/archive/v09-gate-audit.md",
    "docs/archive/v1-roadmap.md",
    "docs/archive/v10-gate-audit.md",
    "docs/archive/zero-friction-runtime-roadmap.md",
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

    assert "<title>Zaxy - event-sourced memory for agent work</title>" in html
    assert 'name="description"' in html
    assert 'property="og:image"' in html
    assert "https://docs.zaxy.io/assets/zaxy-v2.3-header.png" in html
    for path in ("docs/assets/zaxy-v2.3-header.png", "site/assets/zaxy-v2.3-header.png"):
        assert Path(path).exists(), path
        assert Path(path).stat().st_size > 0, path
    assert "Event-sourced memory for agent work" in html
    assert "salience-based forgetting" in html
    assert "Cognitive memory" in html
    assert "purpose-conditioned checkout" in html
    assert "Worker sessions" in html
    assert "Parent mission" in html
    assert "Approval packets" in html
    assert "Memory Checkout" in html
    assert "coordination_checkout" in html
    assert "coordination_approval_packet" in html
    assert "CoordinationAdapter" in html
    assert "Headline 500" in html
    assert "LongMemEval-compatible checkout diagnostic" in html
    assert "Eventloom source of truth" in html
    assert "Pathlight" in html
    assert "embedded LadybugDB" in html
    assert "PyPI 2.4.4" in html
    assert "Harvey LAB 10/10 tasks" in html
    assert "Harvey LAB mean 0.788" in html
    assert "Headline 500 R@5 1.000" in html
    assert "Harvey LAB external signal" in html
    assert "headline 500 checkout evidence" in html
    assert "external verification requested" in html
    assert "Checkout diagnostic, not official LME" in html
    assert "Research paper" in html
    assert "docs/research/zaxy-memory-fabric-paper.html" in html

    for section_id in (
        "coordinate",
        "purpose",
        "architecture",
        "mcp",
        "benchmarks",
        "install",
        "docs",
    ):
        assert f'id="{section_id}"' in html


def test_memory_purpose_positioning_does_not_overclaim_company_brain() -> None:
    """Broader purpose profiles must stay framed as project-local agent work memory."""
    text = Path("docs/archive/memory-is-purpose-zaxy-analysis.md").read_text(encoding="utf-8")

    assert 'Zaxy should not yet claim the full Company Brain category.' in text
    assert "purpose-conditioned memory layer for agent work" in text
    assert "enterprise connectors before the policy model is proven." in text
    assert "Public positioning still says \"agent work memory\"" in text


def test_v05_docs_render_to_static_site() -> None:
    """v0.5 roadmap and quickstarts should be published to the static site."""
    for path in (
        "site/docs/archive/v1-roadmap.html",
        "site/docs/announcements/zaxy-v1.0-x-article.html",
        "site/docs/announcements/zaxy-v1.1-x-article.html",
        "site/docs/media/zaxy-collaborate-demo.html",
        "site/docs/mcp-quickstart.html",
        "site/docs/coordinate-quickstart.html",
        "site/docs/first-run-validation.html",
    ):
        assert Path(path).exists()


def test_v10_release_media_assets_are_published() -> None:
    """The v1.0 release should publish the header and scripted demo artifacts."""
    announcement = Path("docs/announcements/zaxy-v1.0.md").read_text(encoding="utf-8")
    x_article = Path("docs/announcements/zaxy-v1.0-x-article.md").read_text(encoding="utf-8")
    demo = Path("docs/media/zaxy-collaborate-demo.md").read_text(encoding="utf-8")

    for path in (
        "docs/assets/zaxy-v1-header.png",
        "site/assets/zaxy-v1-header.png",
        "docs/media/zaxy-collaborate-demo.mp4",
        "docs/media/zaxy-collaborate-demo.gif",
    ):
        assert Path(path).exists(), path
        assert Path(path).stat().st_size > 0, path
    assert "Zaxy 1.0 release header" in announcement
    assert "External verification request" in x_article
    assert "zaxy doctor --beta-readiness" in x_article
    assert "zaxy-collaborate-demo.mp4" in demo


def test_v11_release_article_and_graph_image_are_published() -> None:
    """The v1.1 release should publish the article, rendered page, and social image."""
    x_article = Path("docs/announcements/zaxy-v1.1-x-article.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/announcements/zaxy-v1.1-x-article.html").read_text(encoding="utf-8")
    homepage = Path("site/index.html").read_text(encoding="utf-8")

    for path in (
        "docs/assets/zaxy-v1.1-header.png",
        "site/assets/zaxy-v1.1-header.png",
    ):
        assert Path(path).exists(), path
        assert Path(path).stat().st_size > 0, path

    assert "accepted-state recovery for agent memory" in x_article
    assert "StateRecoveryBench, MemoryFabric checkout lane" in x_article
    assert "citation coverage" in x_article
    assert "../../assets/zaxy-v1.1-header.png" in rendered
    # The homepage social image moved to the v2.1 header (pinned in
    # test_public_site_has_product_positioning_and_required_sections); the
    # v1.1 article page keeps its own header asset above.
    assert "zaxy-v1.1-header.png" not in homepage


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
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "skill.proposed" in agent_events
    assert "skill.outcome_recorded" in agent_events
    assert "SkillVersion" in agent_events
    assert "memory_skill(action, skill_id, ...)" in mcp
    assert "Applicable Skills" in mcp
    assert "Skill Analytics" in mcp
    assert "rollback" in agent_events
    assert "contradiction diagnostics" in agents
    assert "Skill Memory procedural world-model layer" in agents


def test_benchmark_docs_pin_full_set_quality_reports_and_caveats() -> None:
    """Archived full-set iteration reports should not remain active public claims."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/benchmarks.html").read_text(encoding="utf-8")
    archived_root = Path("reports/archive/benchmarks")

    for run_name in (
        "longmemeval-500-current25-zaxyonly-answer-promotion-embedded-isolated-20260603",
        "longmemeval-500-current24-zaxyonly-numeric-state-embedded-isolated-20260603",
        "longmemeval-500-current22-zaxyonly-page-future-age-neo4j-20260603",
    ):
        assert run_name not in benchmarks
        assert run_name not in rendered
        assert (archived_root / run_name).exists(), run_name

    assert "longmemeval-500-publish-20260607" in benchmarks
    assert "Older backend shootouts, partial slices, experimental LongMemEval iterations" in benchmarks


def test_docs_publish_coordination_competitor_claim_gate() -> None:
    """Active CoordinationBench evidence should keep competitor claims disclosure-only."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    roadmap = Path("docs/coordinate-roadmap.md").read_text(encoding="utf-8")
    report = json.loads(
        Path("reports/benchmarks/coordination-real-v1/coordination-benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    report_md = Path(
        "reports/benchmarks/coordination-real-v1/coordination-benchmark.md"
    ).read_text(encoding="utf-8")

    assert "coordination-real-v1" in benchmarks
    assert "project-defined internal guardrail" in benchmarks
    assert "cannot silently become a public claim" in benchmarks
    assert "Quarq and Semantic Reach/Hybi remain small-project disclosure rows" in roadmap
    assert report["competitor_claim_gate"]["status"] == "blocked"
    assert set(report["competitor_claim_gate"]["blocked_adapters"]) == {"quarq", "hybi"}
    assert report["competitor_adapters"]["quarq"]["claim_status"] == "disclosure_only"
    assert report["competitor_adapters"]["hybi"]["claim_status"] == "disclosure_only"
    assert "## Competitor Claim Gate" in report_md
    assert "Quarq" in report_md
    assert "Semantic Reach / HyperBinder / Hybi" in report_md


def test_pggraph_backend_roadmap_records_contract_first_state() -> None:
    """Docs should keep pgGraph behind the projection contract and explicit backend selector."""
    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    archived_positioning = Path("docs/archive/competitive-positioning.md").read_text(encoding="utf-8")
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
    assert "pgGraph collaboration track" in agents
    assert "LatticeDB evaluation" in agents
    assert "longmemeval-500-pggraph-comparison" not in benchmarks
    assert "development history, not" in benchmarks
    assert "pgGraph" in archived_positioning
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
    roadmap = Path("docs/archive/zero-friction-runtime-roadmap.md").read_text(encoding="utf-8")

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
    assert "[zero-friction-runtime-roadmap.md](archive/zero-friction-runtime-roadmap.md)" in architecture
    assert "embedded LadybugDB" in architecture
    assert "Build the zero-friction embedded graph runtime path" in agents
    assert "Memory Activation Layer" in agents


def test_embedded_runtime_docs_do_not_frame_current_kuzu_as_prototype() -> None:
    """Embedded/Kuzu docs should describe the promoted runtime as first-class."""
    combined = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in (
            "AGENTS.md",
            "docs/archive/zero-friction-runtime-roadmap.md",
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
    """Backend shootout scripts should keep detailed backend measurement contracts."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    normalized_benchmarks = " ".join(benchmarks.split())
    script = Path("scripts/backend-shootout.py").read_text(encoding="utf-8")
    guardrail = Path("scripts/check-backend-shootout.py").read_text(encoding="utf-8")
    archived = Path("docs/archive/zero-friction-runtime-roadmap.md").read_text(encoding="utf-8")

    for phrase in (
        "BM25",
    ):
        assert phrase in benchmarks

    for phrase in (
        "LatticeDB",
        "Neo4j",
        "pgGraph",
        "cold_bootstrap_ms",
        "first_useful_init_ms",
        "first_checkout_ms",
        "append_to_projection_p95_ms",
        "checkout_p95_ms",
        "checkout_p99_ms",
        "returned tokens",
        "injected tokens",
        "citation coverage",
        "resident_memory_delta_bytes",
        "on_disk_footprint_bytes",
        "rebuild_recovery_ms",
        "report_schema_version",
        "source_fingerprints",
        "workload_fingerprints",
        "event_count",
    ):
        assert phrase in script or phrase in guardrail or phrase in archived
    assert "--eventloom-path" in script
    assert "latticedb" in script
    assert "--session-id" in script
    assert "--backends" in script
    assert "--queries-file" in script
    assert "--output" in script
    assert "Defaults run embedded plus BM25 without optional sidecar infrastructure." in script
    assert "Validate backend shootout reports for release-gate use." in guardrail
    assert "--require-report-metadata" in guardrail
    assert "--require-report-metadata" in guardrail
    assert "--require-markdown-report" in guardrail
    assert "--verify-report-fingerprints" in guardrail
    assert "LongMemEval-compatible checkout" in benchmarks
    assert "archived under" in benchmarks
    assert "Backend shootout guardrail passed" in guardrail
    assert "min-projection-events-per-second" in guardrail
    assert "max-rebuild-recovery-ms" in guardrail
    assert "min-quality-per-1k-returned-tokens" in guardrail
    assert "min-answer-at-5-per-1k-returned-tokens" in guardrail
    assert "min-quality-per-1k-injected-tokens" in guardrail
    assert "min-answer-at-5-per-1k-injected-tokens" in guardrail
    assert "max-keyword-p95-ms" in guardrail
    assert "max-keyword-p99-ms" in guardrail
    assert "backend-shootout" not in normalized_benchmarks


def test_install_docs_offer_zero_surprise_first_run_path() -> None:
    """Install docs should make local setup verifiable without guessing where state went."""
    readme = Path("README.md").read_text(encoding="utf-8")
    getting_started = Path("docs/getting-started.md").read_text(encoding="utf-8")
    site = Path("site/index.html").read_text(encoding="utf-8")

    combined = "\n".join([readme, getting_started, site])
    assert "Install, init, verify" in combined
    assert "pipx install zaxy-memory" in combined
    assert "zaxy init" in combined
    assert "Bare `zaxy init` sets up the local embedded graph posture" in combined
    assert "zaxy init --codex-mcp-install auto" in combined
    assert "review that config before replacing it" in combined
    assert "silently replace" in combined
    assert "`codex mcp add` command" in combined
    assert "zaxy init --codex-mcp-install user" in combined
    assert "zaxy init --verbose" in combined
    assert "setup.pending" in combined
    assert "readiness.status" in combined
    assert "readiness.actions" in combined
    assert "readiness.action_items" in combined
    assert "compact-output tips" in combined
    assert "Activation actions use those" in combined
    assert "explicit Eventloom/workspace paths" in combined
    assert "non-command review tasks" in combined
    assert "readiness.blocking_diagnostics" in combined
    assert "readiness.non_blocking_diagnostics" in combined
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
    assert "embedded LadybugDB" in html
    assert "advanced integration tracks" in html
    assert "zaxy init" in html
    assert "local-embedded-codex" not in html
    assert "dashboard --enable-coordinate-review" in html
    assert "read-only remains the default" in html
    assert "graph posture" in html
    assert "Last checkout" in html


def test_public_site_benchmark_claims_use_current_full_set_guardrails() -> None:
    """The public site should lead with current reproducible floors, not stale headline-only claims."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "LongMemEval-compatible checkout diagnostic" in html
    assert "Harvey LAB external memory-ablation" in html
    assert "0.788 mean criterion pass rate" in html
    assert "+0.184 vs regular/no-memory" in html
    assert "docs/benchmarks.html#harvey-lab" in html
    assert "docs/benchmarks.html#headline-500" in html
    assert "mean 0.956" in html
    assert "Answer@5 0.910" in html
    assert "Evidence first, claims second" in html
    assert "archived as development history" in html
    assert "PyPI 0.2.1" not in html
    assert "1005 tests" not in html
    assert "91.96% coverage" not in html


def test_research_paper_is_rendered_and_linked_from_public_site() -> None:
    """The research paper should be published as rendered static-site HTML."""
    homepage = Path("site/index.html").read_text(encoding="utf-8")
    paper = Path("site/docs/research/zaxy-memory-fabric-paper.html").read_text(encoding="utf-8")

    assert 'href="docs/research/zaxy-memory-fabric-paper.html"' in homepage
    assert "Zaxy: Event-Sourced Memory Fabric" in paper
    assert "Harvey LAB memory-ablation suite" in paper
    assert "mean criterion pass rate 0.788" in paper
    assert "MathJax" in paper
    assert 'class="math-block"' in paper
    assert r"\mathcal{E}_s = \langle e_1, e_2, \ldots, e_n \rangle" in paper
    assert r"e_i = \bigl(i, t_i, \tau_i, a_i, p_i, h_i, h_{i-1}\bigr)" in paper


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
    competitive = Path("docs/archive/competitive-positioning.md").read_text(encoding="utf-8")

    assert "Common native-preview adapter contract" in agents
    assert "Use LangGraph and CrewAI native-preview usage" not in agents
    assert "zaxy integrations --recommendation --json" in integrations
    assert "common-native-preview-contract" in integrations
    assert "model-facing UX hardening" in integrations
    assert "AutoGen remains template-only" in integrations
    assert "shared native payload contract" in competitive


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


def test_docs_publish_langgraph_v06_native_contract() -> None:
    """Integration docs and rendered site should publish the LangGraph beta metadata contract."""
    integrations = Path("docs/integrations.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/integrations.html").read_text(encoding="utf-8")
    roadmap = Path("docs/archive/v1-roadmap.md").read_text(encoding="utf-8")

    for text in (integrations, rendered):
        assert "zaxy.native.v0.6" in text
        assert "checkout_failed" in text
        assert "It does not inject stale context after a checkout failure" in text
    assert "langgraph_example" in roadmap
    assert "zaxy doctor --release-smoke" in roadmap


def test_docs_publish_v06_native_integration_contract_snapshot() -> None:
    """v0.6 should publish the shared native lifecycle contract outside MCP."""
    fixture = json.loads(Path("docs/examples/native-integration-contract.json").read_text(encoding="utf-8"))
    integrations = Path("docs/integrations.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/integrations.html").read_text(encoding="utf-8")

    assert fixture["contract"] == "zaxy.native.v0.6"
    assert fixture["metadata_location"] == "payload['zaxy']"
    assert fixture["lifecycle"] == {
        "before_model_or_task": "memory_checkout",
        "after_model_or_task": "capture_assistant_or_task_output",
        "after_tool_call": "capture_redacted_observation",
        "after_context_use": "record_feedback",
    }
    assert fixture["required_payload_keys"] == [
        "contract",
        "framework",
        "operation",
        "source",
        "kind",
        "status",
        "session_id",
        "query",
        "current_fact_count",
        "warning_count",
        "diagnostics",
        "quality",
        "feedback",
        "error",
    ]
    assert fixture["failure"]["status"] == "error"
    assert fixture["failure"]["error_code"] == "checkout_failed"
    assert "docs/examples/native-integration-contract.json" in integrations
    assert "before model/task call" in integrations
    assert "after context use" in integrations
    assert "native-integration-contract.json" in rendered


def test_docs_publish_crewai_v06_native_contract() -> None:
    """Integration docs should say CrewAI uses the shared native checkout contract."""
    integrations = Path("docs/integrations.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/integrations.html").read_text(encoding="utf-8")

    assert "CrewAI checkout uses the same `zaxy.native.v0.6` metadata contract" in integrations
    assert "`crew`, `agent`, and `task_id`" in integrations
    assert "empty `memory` and `contexts`" in integrations
    assert "CrewAI checkout uses the same" in rendered
    assert "zaxy.native.v0.6" in rendered
    assert "checkout_failed" in rendered


def test_full_set_guardrail_docs_distinguish_legacy_and_same_harness_floors() -> None:
    """Docs should not mix the legacy limit=10 floor with current backend-eval floors."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    testing = Path("docs/testing.md").read_text(encoding="utf-8")
    competitive = Path("docs/archive/competitive-positioning.md").read_text(encoding="utf-8")

    assert "LongMemEval-compatible checkout" in benchmarks
    assert "Older backend shootouts" in benchmarks
    assert "legacy `limit=10`" in competitive
    assert "current same-harness `limit=5`" in competitive
    assert "current74 checkout" in competitive
    assert "current public benchmark evidence is intentionally narrow" in testing
    assert "longmemeval-500-publish-20260607/live-benchmark.md" in benchmarks
    assert "legacy `limit=10`" in competitive
    assert "current same-harness `limit=5`" in competitive
    assert "LongMemEval-compatible checkout" in benchmarks


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
    assert "Memory Checkout" in site
    assert "answerability" in site
    assert "current_citation_count" in site
    assert "memory_feedback" in site
    for expected in docs_only_contract_fields:
        assert expected in docs


def test_mcp_docs_publish_tool_contract_snapshot() -> None:
    """v0.6 MCP docs should expose the canonical public tool surface snapshot."""
    fixture = json.loads(Path("docs/examples/mcp-tool-contract.json").read_text(encoding="utf-8"))
    docs = Path("docs/mcp.md").read_text(encoding="utf-8")

    assert fixture["tool_count"] == len(fixture["tools"]) == 48
    assert "docs/examples/mcp-tool-contract.json" in docs
    assert "MCP tool contract snapshot" in docs
    assert {tool["name"] for tool in fixture["tools"]} >= {
        "memory_bootstrap",
        "memory_checkout",
        "memory_feedback",
        "memory_synthesis_artifact",
        "memory_synthesis_evidence",
        "coordination_report_finding",
        "coordination_checkout",
        "coordination_record_synthesis_artifact",
        "coordination_proof_trace",
    }


def test_mcp_docs_publish_structured_error_contract() -> None:
    """v0.6 MCP docs should explain stable client-facing error payloads."""
    docs = Path("docs/mcp.md").read_text(encoding="utf-8")

    assert "Structured Error Payloads" in docs
    assert '"code": "unknown_tool"' in docs
    assert '"message": "Unknown tool: unknown_tool"' in docs
    assert '"remediation": "Call list_tools and retry with one of the advertised tool names."' in docs
    assert "invalid_request" in docs
    assert "internal_error" in docs


def test_mcp_quickstart_documents_v06_recommended_client_routes() -> None:
    """v0.6 MCP quickstart should give one local route for each named client class."""
    quickstart = Path("docs/mcp-quickstart.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/mcp-quickstart.html").read_text(encoding="utf-8")

    expected = {
        "Codex": "zaxy init",
        "Claude Code": "zaxy ide-config claude-code --install --workspace . --eventloom-path .eventloom",
        "Claude Desktop": "zaxy ide-config claude-desktop --eventloom-path .eventloom",
        "Cursor": "zaxy ide-config cursor --install --workspace . --eventloom-path .eventloom",
        "Generic MCP": "zaxy serve --transport stdio",
    }
    for label, command in expected.items():
        assert label in quickstart
        assert command in quickstart
        assert label in rendered
        assert command in rendered
    for command in (
        "zaxy init --codex-mcp-install auto",
        "zaxy init --codex-mcp-install user",
        "zaxy init --codex-mcp-install command",
    ):
        assert command in quickstart
        assert command in rendered


def test_mcp_docs_publish_representative_response_snapshots() -> None:
    """v0.6 MCP docs should expose representative response snapshots."""
    snapshots = json.loads(Path("docs/examples/mcp-response-snapshots.json").read_text(encoding="utf-8"))
    docs = Path("docs/mcp.md").read_text(encoding="utf-8")

    assert "docs/examples/mcp-response-snapshots.json" in docs
    assert "Representative Response Snapshots" in docs
    assert set(snapshots) == {
        "memory_bootstrap",
        "memory_checkout",
        "memory_query",
        "memory_verbatim",
        "context_assemble",
        "memory_feedback",
        "memory_synthesis_artifact",
        "memory_synthesis_evidence",
        "coordination_checkout",
    }
    assert snapshots["memory_bootstrap"]["recommended_next_tool"] == "memory_checkout"
    assert snapshots["memory_checkout"]["quality"]["answerability"] == "answer_from_memory"
    assert snapshots["memory_query"]["first_result"]["source"] == "exact"
    assert snapshots["memory_verbatim"]["first_result"]["source"] == "verbatim"
    assert snapshots["context_assemble"]["context_count"] == 1
    assert snapshots["memory_feedback"]["event_type"] == "memory.reinforced"
    assert snapshots["memory_synthesis_artifact"]["candidate_event_type"] == "memory.synthesis.used"
    assert snapshots["memory_synthesis_evidence"]["event_type"] == "memory.evidence.excluded"
    assert snapshots["coordination_checkout"]["accepted_count"] == 1
    assert "prompt context assembly" in docs
    assert "graph retrieval" in docs
    assert "verbatim source recall" in docs
    assert "feedback reinforcement" in docs
    assert "synthesis artifact writes" in docs
    assert "synthesis evidence row feedback" in docs
    assert "accepted coordination checkout counts" in docs
    assert snapshots["memory_checkout"]["diagnostics"]["feedback_tool"] == "memory_feedback"


def test_changelog_records_memory_synthesis_artifact_contract() -> None:
    """Public synthesis artifact contract changes should be release-noted."""
    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")

    assert "memory_synthesis_artifact" in changelog
    assert "memory_synthesis_evidence" in changelog
    assert "ledger rows" in changelog


def test_public_site_benchmark_claim_is_scoped_to_fixture() -> None:
    """Benchmark copy should not overclaim against broad markdown/vector systems."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "Benchmark evidence" in html
    assert "LongMemEval-compatible checkout 500" in html
    assert "Checkout diagnostic, not official LME" in html
    assert "1.000" in html
    assert "0.956" in html
    assert "0.910" in html
    assert 'class="benchmark-metrics"' in html
    assert 'class="benchmark-card benchmark-card-primary"' in html
    assert 'class="benchmark-comparison"' in html
    assert 'class="benchmark-links"' in html
    assert "docs/external-validation.html" in html
    assert "docs/benchmarks.html" in html
    assert "docs/coordinate-roadmap.html" in html
    assert "docs/benchmark-review.html" not in html
    assert "production-grade vector RAG" not in html
    assert "destroyed" not in html.casefold()


def test_benchmark_docs_disclose_harness_external_claims_and_sources() -> None:
    """Benchmark docs should separate Zaxy-run evidence from competitor disclosures."""
    text = Path("docs/benchmarks.md").read_text(encoding="utf-8")

    assert "LongMemEval-compatible" in text
    assert "0.956" in text
    assert "0.910" in text
    assert "1.000" in text
    assert "0.520" in text
    assert "0.770" in text
    assert "BM25" in text
    assert "same-harness" in text
    assert "Approx tokens" in text
    assert "../reports/benchmarks/longmemeval-500-publish-20260607/live-benchmark.md" in text
    assert "../reports/benchmarks/longmemeval-500-publish-20260607/run-config.md" in text
    assert "Harvey LAB external" in text
    assert "0.788" in text
    assert "+0.184" in text
    assert "Archived partial runs" not in text
    assert "Do not describe the LongMemEval-compatible checkout run as an official" in text
    assert "Do not cite archived partial runs as current benchmark claims" in text


def test_benchmark_docs_record_competitor_adapter_feasibility() -> None:
    """Archived positioning should retain competitor adapter feasibility detail."""
    benchmarks = Path("docs/benchmarks.md").read_text(encoding="utf-8")
    positioning = Path("docs/archive/competitive-positioning.md").read_text(encoding="utf-8")
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
    assert "Zep/Graphiti" in combined
    assert "GBrain" in combined
    assert "same-harness adapter" in combined
    assert "pinned runner" in combined


def test_public_longmemeval_reports_keep_bm25_tradeoff_baseline() -> None:
    """Public LongMemEval reports should include BM25 latency and token tradeoffs."""
    report_paths = [
        path
        for path in Path("reports/benchmarks").glob("*/live-benchmark.json")
        if "longmemeval" in path.as_posix()
    ]

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
        if "bm25" not in summaries and (
            "zaxyonly" in report_path.as_posix()
            or "probe" in report_path.as_posix()
            or set(summaries) == {"zaxy-checkout"}
        ):
            continue
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
    cross_link_exempt = {"docs/packet-analyzer.md", "docs/runbook.md"}
    for doc in REQUIRED_DOCS:
        path = Path(doc)
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# "), doc
        assert len(text.split()) >= 250, doc
        if doc not in cross_link_exempt:
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
    assert "Before publishing a new full 500" in benchmarks
    assert "public benchmark evidence is intentionally narrow" in testing
    assert "headline 500-question LongMemEval-compatible checkout report" in testing


def test_hooks_docs_explain_capture_readiness() -> None:
    """Hook docs should explain the automatic-capture readiness signal."""
    text = Path("docs/hooks.md").read_text(encoding="utf-8")

    assert "capture readiness" in text
    assert "capture_health" in text
    assert "active_observation_types" in text
    assert "missing_observation_types" in text
    assert "zaxy hook-status --json" in text


def test_getting_started_documents_status_memory_activation() -> None:
    """Getting Started should show that top-level status reports memory activation."""
    text = Path("docs/getting-started.md").read_text(encoding="utf-8")

    assert "zaxy status --eventloom-path .eventloom" in text
    assert "memory_activation.remediations" in text
    assert "stale or missing checkout is actionable" in text


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

    assert 'DOCS_CMD="python scripts/build-site-docs.py --check && scripts/validate-docs.sh"' in script
    assert 'run_gate "docs validation" "${DOCS_CMD} --root \\"${ROOT}\\""' in script
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
    assert "public benchmark evidence is intentionally narrow" in testing
    assert "Older benchmark suites, backend shootouts" in testing
    assert "headline report" in testing
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
    assert "archived benchmark reports" in testing
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
    assert "reports/archive/" in testing


def test_embedded_runtime_docs_publish_read_index_warmup_contract() -> None:
    """Source docs and rendered site should preserve embedded read-index warmup behavior."""
    roadmap = Path("docs/archive/zero-friction-runtime-roadmap.md").read_text(encoding="utf-8")
    rendered = Path("site/docs/archive/zero-friction-runtime-roadmap.html").read_text(encoding="utf-8")

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

    assert "Embedded LadybugDB projection" in runbook
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
    assert "plain install uses embedded LadybugDB" in combined


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
    archived = Path("docs/archive/zero-friction-runtime-roadmap.md").read_text(encoding="utf-8")

    assert "public benchmark surface intentionally small" in benchmarks
    assert "production settings default" in archived
    assert "Neo4j and pgGraph remain explicit sidecar" in archived
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
    assert "default implementation is embedded LadybugDB" in docs["graph_schema"]
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
    assert "embedded LadybugDB" in graph_schema


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
    assert "LadybugDB-backed embedded projection store." in runtime_text
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
    assert '"reports/**"' in workflow
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


def test_export_contract_spec_pins_code_identifiers() -> None:
    """The export-contract spec must name the load-bearing identifiers from code,
    so the prose cannot silently drift from what Zaxy actually emits."""
    import warnings

    from zaxy.export_view import EXPORT_ENTRY_SCHEMA_VERSION, UNSIGNED_BUNDLE_VERSION

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from zaxy.portable.export import BUNDLE_VERSION

    spec = Path("docs/export-contract.md").read_text(encoding="utf-8")

    # Version strings (the things a consumer pins).
    assert EXPORT_ENTRY_SCHEMA_VERSION in spec
    assert UNSIGNED_BUNDLE_VERSION in spec
    assert BUNDLE_VERSION in spec

    # The pull surfaces (tool name + CLI command names).
    assert "memory_export" in spec
    for command in ("zaxy export", "export-disclose", "verify-export-subset"):
        assert command in spec

    # Referenced from the docs index.
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/export-contract.md" in readme
