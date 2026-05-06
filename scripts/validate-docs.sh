#!/usr/bin/env bash
# Validate the static public site and local Markdown/HTML documentation links.

set -euo pipefail

ROOT="$(pwd)"

usage() {
    cat <<USAGE
Usage: scripts/validate-docs.sh [--root PATH]

Checks that site/index.html exists and that local Markdown and HTML links resolve.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

python - "$ROOT" <<'PY'
from __future__ import annotations

import html.parser
import re
import sys
from pathlib import Path
from urllib.parse import unquote


root = Path(sys.argv[1]).resolve()
failures: list[str] = []


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = "href" if tag == "a" else "src" if tag in {"img", "script", "link"} else ""
        if not attr:
            return
        value = dict(attrs).get(attr)
        if value:
            self.links.append((attr, value))


def is_external(link: str) -> bool:
    return link.startswith(("http://", "https://", "mailto:", "data:", "#"))


def local_target(source: Path, link: str) -> tuple[Path, str]:
    path_part, _, anchor = link.partition("#")
    path_part = unquote(path_part)
    if not path_part:
        return source, anchor
    return (source.parent / path_part).resolve(), anchor


def anchors_for(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".html":
        return set(re.findall(r'id=["\']([^"\']+)["\']', text))
    anchors: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip().lower()
        anchor = re.sub(r"[^a-z0-9 -]", "", heading).replace(" ", "-")
        anchors.add(anchor)
    return anchors


def check_link(source: Path, link: str) -> None:
    if is_external(link):
        return
    target, anchor = local_target(source, link)
    try:
        target.relative_to(root)
    except ValueError:
        failures.append(f"{source.relative_to(root)} links outside repository: {link}")
        return
    if not target.exists():
        failures.append(f"{source.relative_to(root)} has broken link: {link}")
        return
    if anchor and anchor not in anchors_for(target):
        failures.append(f"{source.relative_to(root)} has broken anchor: {link}")


site_index = root / "site" / "index.html"
if not site_index.exists():
    failures.append("Missing site/index.html")

for path in sorted([*root.glob("docs/**/*.md"), *root.glob("site/**/*.html")]):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            check_link(path, match.group(1))
    elif path.suffix == ".html":
        parser = LinkParser()
        parser.feed(text)
        for _, link in parser.links:
            check_link(path, link)

if failures:
    for failure in failures:
        print(failure, file=sys.stderr)
    raise SystemExit(1)

print("Docs validation passed")
PY
