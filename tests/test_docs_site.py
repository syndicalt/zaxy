"""Tests for the public site and documentation coverage."""

from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_DOCS = [
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

    for section_id in (
        "why",
        "architecture",
        "mcp",
        "retrieval",
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
    """Docs should keep pgGraph behind the projection contract and Neo4j default."""
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
    assert "0.990" in html
    assert "0.840" in html
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
    assert "docs/benchmarks.md" in html
    assert "docs/benchmark-review.md" in html
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
        assert doc in parser.links
    assert "../docs/architecture.md" not in parser.links


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
            if path_part.startswith(("docs/", "README.md", "reports/")):
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
    """The go-live gate should include docs/site validation."""
    script = Path("scripts/release-check.sh").read_text(encoding="utf-8")

    assert 'DOCS_CMD="scripts/validate-docs.sh"' in script
    assert '"${DOCS_CMD}" --root "${ROOT}"' in script


def test_github_pages_workflow_publishes_site_directory() -> None:
    """GitHub Pages should deploy the static site and docs under the project URL."""
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert "branches: [master]" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "cp -R site/. _site/" in workflow
    assert "cp -R docs _site/docs" in workflow
    assert "cp README.md _site/README.md" in workflow
    assert "cp -R reports _site/reports" in workflow
    assert "cp -R reports _site/site/reports" in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
    assert "path: _site" in workflow
    assert "actions/deploy-pages@v4" in workflow
