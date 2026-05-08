"""Tests for the public site and documentation coverage."""

from __future__ import annotations

import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path

REQUIRED_DOCS = [
    "docs/getting-started.md",
    "docs/architecture.md",
    "docs/configuration.md",
    "docs/mcp.md",
    "docs/eventloom.md",
    "docs/graph-schema.md",
    "docs/retrieval.md",
    "docs/embeddings.md",
    "docs/security.md",
    "docs/operations.md",
    "docs/deployment.md",
    "docs/testing.md",
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
    assert "Temporal memory for AI agents" in html
    assert "Eventloom log" in html
    assert "Neo4j temporal graph" in html
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


def test_public_site_benchmark_claim_is_scoped_to_fixture() -> None:
    """Benchmark copy should not overclaim against broad markdown/vector systems."""
    html = Path("site/index.html").read_text(encoding="utf-8")

    assert "Representative context benchmark" in html
    assert "text-embedding-3-small" in html
    assert "1.000" in html
    assert "+0.480" in html
    assert "650 paired queries" in html
    assert "not a universal benchmark" in html
    assert "reports/benchmarks/live-benchmark.md" in html
    assert "docs/benchmark-review.md" in html
    assert "production-grade vector RAG" not in html
    assert "destroyed" not in html.casefold()


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
