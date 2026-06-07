#!/usr/bin/env python3
"""Render repository Markdown documentation into static site HTML."""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT / "site"
DOCS_ROOT = ROOT / "docs"
GITHUB_BLOB_ROOT = "https://github.com/syndicalt/zaxy/blob/master"


@dataclass(frozen=True)
class RenderedPage:
    source: Path
    target: Path
    html: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated docs are stale")
    args = parser.parse_args()

    pages = build_pages()
    stale: list[Path] = []
    for page in pages:
        if args.check:
            if not page.target.exists() or page.target.read_text(encoding="utf-8") != page.html:
                stale.append(page.target)
            continue
        page.target.parent.mkdir(parents=True, exist_ok=True)
        page.target.write_text(page.html, encoding="utf-8")

    if stale:
        for path in stale:
            print(f"stale rendered doc: {path.relative_to(ROOT)}", file=sys.stderr)
        print("Run scripts/build-site-docs.py to refresh rendered documentation.", file=sys.stderr)
        return 1
    if not args.check:
        print(f"Rendered {len(pages)} documentation pages under {SITE_ROOT.relative_to(ROOT)}/")
    return 0


def build_pages() -> list[RenderedPage]:
    sources = [ROOT / "README.md", *sorted(DOCS_ROOT.rglob("*.md"))]
    return [render_page(source) for source in sources]


def render_page(source: Path) -> RenderedPage:
    markdown = source.read_text(encoding="utf-8")
    target = rendered_target(source)
    title = document_title(markdown, source)
    body = render_markdown(markdown, source=source)
    math_head = mathjax_head(markdown)
    rel_to_root = relative_prefix(target)
    index_link = relative_link(target, SITE_ROOT / "index.html")
    docs_link = relative_link(target, SITE_ROOT / "docs" / "getting-started.html")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)} - Zaxy Docs</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,600&family=Inter:wght@400;550;650;750;850&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="{rel_to_root}style.css" />{math_head}
</head>
<body class="doc-page">
  <nav class="nav" aria-label="Documentation">
    <a class="brand" href="{index_link}"><span class="brand-mark" aria-hidden="true"></span>Zaxy</a>
    <div class="nav-links">
      <a href="{index_link}">Overview</a>
      <a href="{rel_to_root}docs/coordinate-roadmap.html">Coordinate</a>
      <a href="{rel_to_root}docs/memory-is-purpose-zaxy-analysis.html">Purpose</a>
      <a href="{rel_to_root}docs/benchmarks.html">Benchmarks</a>
      <a href="{docs_link}">Start</a>
      <a href="https://github.com/syndicalt/zaxy" class="nav-cta">GitHub</a>
    </div>
  </nav>
  <main class="doc-layout">
    <article class="doc-content">
{body}
    </article>
  </main>
</body>
</html>
"""
    return RenderedPage(source=source, target=target, html=html_text)


def mathjax_head(markdown: str) -> str:
    if not any(delimiter in markdown for delimiter in (r"\[", "$$")):
        return ""
    return """
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>"""


def document_title(markdown: str, source: Path) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return source.stem.replace("-", " ").title()


def rendered_target(source: Path) -> Path:
    if source == ROOT / "README.md":
        return SITE_ROOT / "README.html"
    relative = source.relative_to(ROOT)
    return SITE_ROOT / relative.with_suffix(".html")


def relative_prefix(target: Path) -> str:
    depth = len(target.relative_to(SITE_ROOT).parents) - 1
    return "../" * depth


def relative_link(from_page: Path, repo_target: Path) -> str:
    rendered = rendered_target(repo_target) if repo_target.suffix == ".md" else repo_target
    if repo_target == ROOT / "README.md":
        rendered = SITE_ROOT / "README.html"
    if repo_target == SITE_ROOT / "index.html":
        rendered = SITE_ROOT / "index.html"
    return Path(os.path.relpath(rendered, from_page.parent)).as_posix()


def render_markdown(markdown: str, *, source: Path) -> str:
    lines = normalize_list_continuations(markdown.splitlines())
    out: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None
    blockquote: list[str] = []
    in_code = False
    code_language = ""
    code_lines: list[str] = []
    i = 0

    def close_paragraph() -> None:
        if paragraph:
            text = " ".join(part.strip() for part in paragraph).strip()
            out.append(f"      <p>{render_inline(text, source=source)}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_type
        if list_type is not None:
            out.append(f"      </{list_type}>")
            list_type = None

    def close_blockquote() -> None:
        if blockquote:
            text = " ".join(part.strip() for part in blockquote).strip()
            out.append(f"      <blockquote>{render_inline(text, source=source)}</blockquote>")
            blockquote.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if in_code:
            if stripped.startswith("```"):
                class_attr = f' class="language-{html.escape(code_language)}"' if code_language else ""
                code = html.escape("\n".join(code_lines))
                out.append(f"      <pre><code{class_attr}>{code}</code></pre>")
                in_code = False
                code_language = ""
                code_lines.clear()
            else:
                code_lines.append(line)
            i += 1
            continue
        if stripped.startswith("```"):
            close_paragraph()
            close_list()
            close_blockquote()
            in_code = True
            code_language = stripped.removeprefix("```").strip()
            i += 1
            continue
        if not stripped:
            close_paragraph()
            close_list()
            close_blockquote()
            i += 1
            continue
        math_block = maybe_math_block(lines, i)
        if math_block is not None:
            close_paragraph()
            close_list()
            close_blockquote()
            html_math, consumed = math_block
            out.append(html_math)
            i += consumed
            continue
        table = maybe_table(lines, i, source=source)
        if table is not None:
            close_paragraph()
            close_list()
            close_blockquote()
            html_table, consumed = table
            out.append(html_table)
            i += consumed
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            close_paragraph()
            close_list()
            close_blockquote()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            out.append(
                f'      <h{level} id="{slugify(text)}">{render_inline(text, source=source)}</h{level}>'
            )
            i += 1
            continue
        unordered = re.match(r"^[-*]\s+(.+)$", stripped)
        ordered = re.match(r"^\d+[.]\s+(.+)$", stripped)
        if unordered or ordered:
            close_paragraph()
            close_blockquote()
            desired = "ul" if unordered else "ol"
            if list_type != desired:
                close_list()
                list_type = desired
                out.append(f"      <{list_type}>")
            item = (unordered or ordered).group(1)
            out.append(f"        <li>{render_inline(item, source=source)}</li>")
            i += 1
            continue
        if stripped.startswith(">"):
            close_paragraph()
            close_list()
            blockquote.append(stripped.lstrip(">").strip())
            i += 1
            continue
        close_list()
        close_blockquote()
        paragraph.append(line)
        i += 1

    close_paragraph()
    close_list()
    close_blockquote()
    if in_code:
        code = html.escape("\n".join(code_lines))
        out.append(f"      <pre><code>{code}</code></pre>")
    return "\n".join(out)


def maybe_math_block(lines: list[str], index: int) -> tuple[str, int] | None:
    stripped = lines[index].strip()
    if stripped == r"\[":
        math_lines: list[str] = []
        i = index + 1
        while i < len(lines):
            if lines[i].strip() == r"\]":
                body = html.escape("\n".join(math_lines))
                return f'      <div class="math-block">\\[\n{body}\n\\]</div>', i - index + 1
            math_lines.append(lines[i])
            i += 1
        return None
    if stripped == "$$":
        math_lines = []
        i = index + 1
        while i < len(lines):
            if lines[i].strip() == "$$":
                body = html.escape("\n".join(math_lines))
                return f'      <div class="math-block">$$\n{body}\n$$</div>', i - index + 1
            math_lines.append(lines[i])
            i += 1
    return None


def normalize_list_continuations(lines: list[str]) -> list[str]:
    normalized: list[str] = []
    list_item = re.compile(r"^\s*(?:[-*]|\d+[.])\s+")
    for line in lines:
        if (
            normalized
            and line.startswith(("  ", "\t"))
            and line.strip()
            and list_item.match(normalized[-1])
            and not line.lstrip().startswith(("```", "-", "*"))
        ):
            normalized[-1] = f"{normalized[-1]} {line.strip()}"
            continue
        normalized.append(line)
    return normalized


def maybe_table(lines: list[str], index: int, *, source: Path) -> tuple[str, int] | None:
    if index + 1 >= len(lines):
        return None
    header = lines[index].strip()
    separator = lines[index + 1].strip()
    if "|" not in header or not re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", separator):
        return None
    rows = [header]
    i = index + 2
    while i < len(lines) and "|" in lines[i] and lines[i].strip():
        rows.append(lines[i].strip())
        i += 1
    header_cells = split_table_row(rows[0])
    body_rows = [split_table_row(row) for row in rows[1:]]
    html_rows = [
        "      <table>",
        "        <thead>",
        "          <tr>",
        *[f"            <th>{render_inline(cell, source=source)}</th>" for cell in header_cells],
        "          </tr>",
        "        </thead>",
        "        <tbody>",
    ]
    for row in body_rows:
        html_rows.append("          <tr>")
        html_rows.extend(f"            <td>{render_inline(cell, source=source)}</td>" for cell in row)
        html_rows.append("          </tr>")
    html_rows.extend(["        </tbody>", "      </table>"])
    return "\n".join(html_rows), len(rows) + 1


def split_table_row(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def render_inline(text: str, *, source: Path) -> str:
    code_spans: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        code_spans.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\u0000CODE{len(code_spans) - 1}\u0000"

    protected = re.sub(r"`([^`]+)`", stash_code, text)
    escaped = html.escape(protected)
    escaped = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda match: render_image(match, source=source),
        escaped,
    )
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: render_link(match, source=source),
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    for index, code in enumerate(code_spans):
        escaped = escaped.replace(html.escape(f"\u0000CODE{index}\u0000"), code)
    return escaped


def render_image(match: re.Match[str], *, source: Path) -> str:
    alt = match.group(1)
    href = html.unescape(match.group(2))
    rewritten = rewrite_link(href, source=source)
    return f'<img src="{html.escape(rewritten, quote=True)}" alt="{alt}" />'


def render_link(match: re.Match[str], *, source: Path) -> str:
    label = match.group(1)
    href = html.unescape(match.group(2))
    rewritten = rewrite_link(href, source=source)
    return f'<a href="{html.escape(rewritten, quote=True)}">{label}</a>'


def rewrite_link(href: str, *, source: Path) -> str:
    if href.startswith(("#", "http://", "https://", "mailto:", "data:")):
        return href
    split = urlsplit(href)
    path_part = split.path
    anchor = f"#{split.fragment}" if split.fragment else ""
    if not path_part:
        return href
    repo_target = (source.parent / path_part).resolve()
    try:
        repo_target.relative_to(ROOT)
    except ValueError:
        return href
    if repo_target == SITE_ROOT / "index.html":
        rendered = SITE_ROOT / "index.html"
    elif repo_target == ROOT / "README.md" or (
        repo_target.suffix == ".md" and is_relative_to(repo_target, DOCS_ROOT)
    ):
        rendered = rendered_target(repo_target)
    elif repo_target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"} and is_relative_to(
        repo_target,
        DOCS_ROOT / "assets",
    ):
        rendered = SITE_ROOT / "assets" / repo_target.name
    elif repo_target.suffix == ".md" and is_relative_to(repo_target, ROOT / "reports"):
        return href
    else:
        return github_source_link(repo_target, anchor=anchor)
    from_page = rendered_target(source)
    return Path(os.path.relpath(rendered, from_page.parent)).as_posix() + anchor


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def github_source_link(path: Path, *, anchor: str) -> str:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()
    return f"{GITHUB_BLOB_ROOT}/{relative}{anchor}"


def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.casefold().strip()
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


if __name__ == "__main__":
    raise SystemExit(main())
