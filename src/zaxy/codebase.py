"""Codebase file inventory ingestion helpers."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

DEFAULT_MAX_BYTES = 512 * 1024
EXCLUDED_DIRS = {
    ".eventloom",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}
LANGUAGE_BY_SUFFIX = {
    ".css": "css",
    ".go": "go",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "javascript",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "shell",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".yaml": "yaml",
    ".yml": "yaml",
}


def collect_codebase_events(
    root: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[dict[str, Any]]:
    """Collect supported source files as code.file.indexed event inputs."""
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise ValueError(f"codebase root does not exist: {root_path}")
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    paths = [root_path] if root_path.is_file() else _iter_supported_files(root_path, max_bytes)
    events: list[dict[str, Any]] = []
    for path in paths:
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.casefold())
        if language is None:
            continue
        size = path.stat().st_size
        if size > max_bytes:
            continue
        content = path.read_bytes()
        rel_path = _relative_path(path, root_path)
        events.append(
            {
                "event_type": "code.file.indexed",
                "actor": "zaxy-codebase-indexer",
                "payload": {
                    "path": rel_path,
                    "language": language,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "bytes": len(content),
                    "lines": _line_count(content),
                },
            }
        )
        events.extend(_collect_symbol_events(path, rel_path, language, content))
    return events


def _collect_symbol_events(path: Path, rel_path: str, language: str, content: bytes) -> list[dict[str, Any]]:
    text = content.decode("utf-8", errors="replace")
    if language == "python":
        return _collect_python_symbol_events(rel_path, language, text)
    return _collect_pattern_symbol_events(rel_path, language, text)


def _collect_python_symbol_events(rel_path: str, language: str, text: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    events: list[dict[str, Any]] = []
    parent_by_node: dict[ast.AST, ast.AST | None] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_node[child] = parent

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                events.append(
                    _import_event(
                        rel_path,
                        language,
                        module=alias.name,
                        name=alias.asname or alias.name,
                        kind="import",
                        start_line=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                events.append(
                    _import_event(
                        rel_path,
                        language,
                        module=module,
                        name=alias.asname or alias.name,
                        kind="from_import",
                        start_line=node.lineno,
                    )
                )
        elif isinstance(node, ast.ClassDef):
            events.append(
                _symbol_event(
                    rel_path,
                    language,
                    name=node.name,
                    qualified_name=_python_qualified_name(node, parent_by_node),
                    kind="class",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                )
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            events.append(
                _symbol_event(
                    rel_path,
                    language,
                    name=node.name,
                    qualified_name=_python_qualified_name(node, parent_by_node),
                    kind="function",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                )
            )
    return events


def _python_qualified_name(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    parent_by_node: dict[ast.AST, ast.AST | None],
) -> str:
    parts = [node.name]
    parent = parent_by_node.get(node)
    while parent is not None:
        if isinstance(parent, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append(parent.name)
        parent = parent_by_node.get(parent)
    return ".".join(reversed(parts))


_IMPORT_RE = re.compile(r"""^\s*(?:import|export\s+.*?\s+from)\s+(.+?);?\s*$""")
_FROM_IMPORT_RE = re.compile(r"""^\s*import\s+(.+?)\s+from\s+['"]([^'"]+)['"];?\s*$""")
_REQUIRE_RE = re.compile(r"""require\(['"]([^'"]+)['"]\)""")
_CLASS_RE = re.compile(r"""^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)""")
_FUNCTION_RE = re.compile(
    r"""^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("""
)
_GO_IMPORT_RE = re.compile(r"""^\s*import\s+(?:\w+\s+)?["`]([^"`]+)["`]""")
_GO_TYPE_RE = re.compile(r"""^\s*type\s+([A-Za-z_]\w*)\s+""")
_GO_FUNC_RE = re.compile(r"""^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(""")
_RUST_USE_RE = re.compile(r"""^\s*(?:pub\s+)?use\s+([^;]+);""")
_RUST_SYMBOL_RE = re.compile(r"""^\s*(?:pub\s+)?(?:async\s+)?(fn|struct|enum|trait|mod)\s+([A-Za-z_]\w*)""")
_JAVA_IMPORT_RE = re.compile(r"""^\s*import\s+(?:static\s+)?([^;]+);""")
_JAVA_TYPE_RE = re.compile(r"""^\s*(?:public|private|protected|abstract|final|\s)*\s*(class|interface|enum)\s+([A-Za-z_]\w*)""")
_SHELL_SOURCE_RE = re.compile(r"""^\s*(?:source|\.)\s+(.+)$""")
_SHELL_FUNCTION_RE = re.compile(r"""^\s*(?:function\s+)?([A-Za-z_][\w-]*)\s*(?:\(\))?\s*\{""")


def _collect_pattern_symbol_events(rel_path: str, language: str, text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        if language in {"javascript", "typescript"}:
            events.extend(_js_ts_events(rel_path, language, line, line_number))
        elif language == "go":
            events.extend(_go_events(rel_path, language, line, line_number))
        elif language == "rust":
            events.extend(_rust_events(rel_path, language, line, line_number))
        elif language == "java":
            events.extend(_java_events(rel_path, language, line, line_number))
        elif language == "shell":
            events.extend(_shell_events(rel_path, language, line, line_number))
    return events


def _js_ts_events(rel_path: str, language: str, line: str, line_number: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if match := _FROM_IMPORT_RE.match(line):
        specifiers, module = match.groups()
        for name in _import_names(specifiers):
            events.append(_import_event(rel_path, language, module=module, name=name, kind="import", start_line=line_number))
    elif match := _REQUIRE_RE.search(line):
        module = match.group(1)
        events.append(_import_event(rel_path, language, module=module, name=module, kind="require", start_line=line_number))
    elif match := _IMPORT_RE.match(line):
        module = match.group(1).strip().strip("'\"")
        events.append(_import_event(rel_path, language, module=module, name=module, kind="import", start_line=line_number))

    if match := _CLASS_RE.match(line):
        events.append(_symbol_event(rel_path, language, name=match.group(1), qualified_name=match.group(1), kind="class", start_line=line_number, end_line=line_number))
    if match := _FUNCTION_RE.match(line):
        name = next(group for group in match.groups() if group)
        events.append(_symbol_event(rel_path, language, name=name, qualified_name=name, kind="function", start_line=line_number, end_line=line_number))
    return events


def _go_events(rel_path: str, language: str, line: str, line_number: int) -> list[dict[str, Any]]:
    if match := _GO_IMPORT_RE.match(line):
        module = match.group(1)
        return [_import_event(rel_path, language, module=module, name=module, kind="import", start_line=line_number)]
    if match := _GO_TYPE_RE.match(line):
        name = match.group(1)
        return [_symbol_event(rel_path, language, name=name, qualified_name=name, kind="type", start_line=line_number, end_line=line_number)]
    if match := _GO_FUNC_RE.match(line):
        name = match.group(1)
        return [_symbol_event(rel_path, language, name=name, qualified_name=name, kind="function", start_line=line_number, end_line=line_number)]
    return []


def _rust_events(rel_path: str, language: str, line: str, line_number: int) -> list[dict[str, Any]]:
    if match := _RUST_USE_RE.match(line):
        module = match.group(1).strip()
        return [_import_event(rel_path, language, module=module, name=module.rsplit("::", 1)[-1], kind="use", start_line=line_number)]
    if match := _RUST_SYMBOL_RE.match(line):
        kind, name = match.groups()
        return [_symbol_event(rel_path, language, name=name, qualified_name=name, kind=kind, start_line=line_number, end_line=line_number)]
    return []


def _java_events(rel_path: str, language: str, line: str, line_number: int) -> list[dict[str, Any]]:
    if match := _JAVA_IMPORT_RE.match(line):
        module = match.group(1)
        return [_import_event(rel_path, language, module=module, name=module.rsplit(".", 1)[-1], kind="import", start_line=line_number)]
    if match := _JAVA_TYPE_RE.match(line):
        kind, name = match.groups()
        return [_symbol_event(rel_path, language, name=name, qualified_name=name, kind=kind, start_line=line_number, end_line=line_number)]
    return []


def _shell_events(rel_path: str, language: str, line: str, line_number: int) -> list[dict[str, Any]]:
    if match := _SHELL_SOURCE_RE.match(line):
        module = match.group(1).strip().strip("\"'")
        return [_import_event(rel_path, language, module=module, name=module, kind="source", start_line=line_number)]
    if match := _SHELL_FUNCTION_RE.match(line):
        name = match.group(1)
        return [_symbol_event(rel_path, language, name=name, qualified_name=name, kind="function", start_line=line_number, end_line=line_number)]
    return []


def _import_names(specifiers: str) -> list[str]:
    names = specifiers.strip()
    if names.startswith("{") and names.endswith("}"):
        names = names[1:-1]
    return [
        part.strip().split(" as ", 1)[-1].strip()
        for part in names.split(",")
        if part.strip() and not part.strip().startswith("*")
    ]


def _symbol_event(
    rel_path: str,
    language: str,
    *,
    name: str,
    qualified_name: str,
    kind: str,
    start_line: int,
    end_line: int,
) -> dict[str, Any]:
    return {
        "event_type": "code.symbol.indexed",
        "actor": "zaxy-codebase-indexer",
        "payload": {
            "path": rel_path,
            "language": language,
            "name": name,
            "qualified_name": qualified_name,
            "kind": kind,
            "start_line": start_line,
            "end_line": end_line,
        },
    }


def _import_event(
    rel_path: str,
    language: str,
    *,
    module: str,
    name: str,
    kind: str,
    start_line: int,
) -> dict[str, Any]:
    return {
        "event_type": "code.import.indexed",
        "actor": "zaxy-codebase-indexer",
        "payload": {
            "path": rel_path,
            "language": language,
            "module": module,
            "name": name,
            "kind": kind,
            "start_line": start_line,
        },
    }


def _iter_supported_files(root: Path, max_bytes: int) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") or part in EXCLUDED_DIRS for part in relative_parts[:-1]):
            continue
        if not path.is_file():
            continue
        if path.suffix.casefold() not in LANGUAGE_BY_SUFFIX:
            continue
        if path.stat().st_size > max_bytes:
            continue
        files.append(path)
    return sorted(files)


def _relative_path(path: Path, root: Path) -> str:
    if root.is_file():
        return path.name
    return path.relative_to(root).as_posix()


def _line_count(content: bytes) -> int:
    if not content:
        return 0
    return len(content.decode("utf-8", errors="replace").splitlines())
