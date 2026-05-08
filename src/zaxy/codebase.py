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
    file_index = {_relative_path(path, root_path) for path in paths}
    code_events_by_path: dict[str, list[dict[str, Any]]] = {}
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
        code_events = _collect_symbol_events(path, rel_path, language, content)
        code_events_by_path[rel_path] = code_events
        events.extend(code_events)
        events.extend(_collect_dependency_events(rel_path, language, code_events, file_index))
        events.extend(_collect_coverage_events(rel_path, language, content, code_events, file_index))
    go_package_symbols = _go_package_symbols(code_events_by_path)
    for path in paths:
        language = LANGUAGE_BY_SUFFIX.get(path.suffix.casefold())
        if language is None:
            continue
        size = path.stat().st_size
        if size > max_bytes:
            continue
        rel_path = _relative_path(path, root_path)
        events.extend(
            _collect_call_events(
                rel_path,
                language,
                path.read_bytes(),
                code_events_by_path.get(rel_path, []),
                file_index,
                go_package_symbols,
            )
        )
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
    if match := _JAVA_METHOD_RE.match(line):
        name = match.group(1)
        return [_symbol_event(rel_path, language, name=name, qualified_name=name, kind="method", start_line=line_number, end_line=line_number)]
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


def _collect_dependency_events(
    rel_path: str,
    language: str,
    code_events: list[dict[str, Any]],
    file_index: set[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in code_events:
        if event["event_type"] != "code.import.indexed":
            continue
        payload = event["payload"]
        module = str(payload["module"])
        target_path, resolution = _resolve_dependency_target(rel_path, language, module, file_index)
        if target_path is None:
            continue
        events.append(
            {
                "event_type": "code.dependency.indexed",
                "actor": "zaxy-codebase-indexer",
                "payload": {
                    "source_path": rel_path,
                    "target_path": target_path,
                    "language": language,
                    "module": module,
                    "import_name": payload["name"],
                    "start_line": payload["start_line"],
                    "resolution": resolution,
                },
            }
        )
    return events


def _collect_call_events(
    rel_path: str,
    language: str,
    content: bytes,
    code_events: list[dict[str, Any]],
    file_index: set[str],
    go_package_symbols: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    if language in {"javascript", "typescript"}:
        text = content.decode("utf-8", errors="replace")
        return _collect_js_ts_call_events(rel_path, language, text, code_events, file_index)
    if language in {"go", "rust", "java"}:
        text = content.decode("utf-8", errors="replace")
        return _collect_pattern_call_events(rel_path, language, text, code_events, go_package_symbols or {})
    if language != "python":
        return []
    text = content.decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    same_file_symbols = {
        str(event["payload"]["qualified_name"]): event["payload"]
        for event in code_events
        if event["event_type"] == "code.symbol.indexed"
    }
    imported_symbols = _python_imported_symbol_targets(rel_path, code_events, file_index)
    return _collect_python_call_events(rel_path, language, tree, same_file_symbols, imported_symbols)


def _python_imported_symbol_targets(
    rel_path: str,
    code_events: list[dict[str, Any]],
    file_index: set[str],
) -> dict[str, tuple[str, str]]:
    targets: dict[str, tuple[str, str]] = {}
    for event in code_events:
        if event["event_type"] != "code.import.indexed":
            continue
        payload = event["payload"]
        module = str(payload["module"])
        target_path, _resolution = _resolve_python_dependency(rel_path, module, file_index)
        if target_path is None:
            continue
        name = str(payload["name"])
        targets[name] = (target_path, name)
    return targets


def _collect_python_call_events(
    rel_path: str,
    language: str,
    tree: ast.AST,
    same_file_symbols: dict[str, dict[str, Any]],
    imported_symbols: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    parent_by_node: dict[ast.AST, ast.AST | None] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_node[child] = parent

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        caller = _nearest_python_callable(node, parent_by_node)
        if caller is None:
            continue
        callee = _python_call_name(node.func)
        if callee is None:
            continue
        resolved = _resolve_python_call_target(rel_path, callee, same_file_symbols, imported_symbols)
        events.append(
            _call_event(
                rel_path,
                language,
                caller=caller,
                callee=callee.rsplit(".", 1)[-1],
                callee_qualified_name=callee,
                target_path=resolved[0],
                target_qualified_name=resolved[1],
                start_line=node.lineno,
                resolution=resolved[2],
            )
        )
    return events


_JS_TS_FUNCTION_START_RE = re.compile(
    r"""^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("""
)
_JS_TS_CALL_RE = re.compile(r"""\b([A-Za-z_$][\w$]*)\s*\(""")
_JS_TS_KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return"}


def _collect_js_ts_call_events(
    rel_path: str,
    language: str,
    text: str,
    code_events: list[dict[str, Any]],
    file_index: set[str],
) -> list[dict[str, Any]]:
    same_file_symbols = {
        str(event["payload"]["qualified_name"]): event["payload"]
        for event in code_events
        if event["event_type"] == "code.symbol.indexed"
    }
    imported_symbols = _js_ts_imported_symbol_targets(rel_path, code_events, file_index)
    events: list[dict[str, Any]] = []
    current_function: str | None = None
    brace_depth = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if current_function is None:
            match = _JS_TS_FUNCTION_START_RE.match(line)
            if not match:
                continue
            current_function = next(group for group in match.groups() if group)
            brace_depth = line.count("{") - line.count("}")
            continue

        for match in _JS_TS_CALL_RE.finditer(line):
            callee = match.group(1)
            if callee in _JS_TS_KEYWORDS:
                continue
            resolved = _resolve_js_ts_call_target(rel_path, callee, same_file_symbols, imported_symbols)
            events.append(
                _call_event(
                    rel_path,
                    language,
                    caller=current_function,
                    callee=callee,
                    callee_qualified_name=callee,
                    target_path=resolved[0],
                    target_qualified_name=resolved[1],
                    start_line=line_number,
                    resolution=resolved[2],
                )
            )
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            current_function = None
    return events


def _js_ts_imported_symbol_targets(
    rel_path: str,
    code_events: list[dict[str, Any]],
    file_index: set[str],
) -> dict[str, tuple[str, str]]:
    targets: dict[str, tuple[str, str]] = {}
    for event in code_events:
        if event["event_type"] != "code.import.indexed":
            continue
        payload = event["payload"]
        module = str(payload["module"])
        target_path, _resolution = _resolve_js_ts_dependency(rel_path, module, file_index)
        if target_path is None:
            continue
        name = str(payload["name"])
        targets[name] = (target_path, name)
    return targets


def _resolve_js_ts_call_target(
    rel_path: str,
    callee: str,
    same_file_symbols: dict[str, dict[str, Any]],
    imported_symbols: dict[str, tuple[str, str]],
) -> tuple[str | None, str | None, str]:
    if callee in same_file_symbols:
        return rel_path, callee, "same_file_symbol"
    if callee in imported_symbols:
        target_path, target_qualified_name = imported_symbols[callee]
        return target_path, target_qualified_name, "imported_symbol"
    return None, None, "unresolved"


_PATTERN_CALL_RE = re.compile(r"""\b([A-Za-z_]\w*)\s*\(""")
_PATTERN_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "func", "fn", "class", "new"}
_JAVA_METHOD_RE = re.compile(
    r"""^\s*(?:public|private|protected|static|final|abstract|\s)*\s*(?:[A-Za-z_][\w<>\[\]]+\s+)+([A-Za-z_]\w*)\s*\("""
)


def _collect_pattern_call_events(
    rel_path: str,
    language: str,
    text: str,
    code_events: list[dict[str, Any]],
    go_package_symbols: dict[str, dict[str, tuple[str, str]]],
) -> list[dict[str, Any]]:
    same_file_symbols = {
        str(event["payload"]["qualified_name"]): event["payload"]
        for event in code_events
        if event["event_type"] == "code.symbol.indexed"
    }
    events: list[dict[str, Any]] = []
    go_imports = _go_import_aliases(code_events) if language == "go" else {}
    current_function: str | None = None
    brace_depth = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if current_function is None:
            current_function = _pattern_function_name(language, line)
            if current_function is None:
                continue
            brace_depth = line.count("{") - line.count("}")
            if brace_depth <= 0:
                current_function = None
            continue

        for match in _PATTERN_CALL_RE.finditer(line):
            callee = match.group(1)
            if callee in _PATTERN_KEYWORDS or callee == current_function:
                continue
            qualified_callee = _qualified_pattern_call_name(line, match.start(), callee)
            resolved = _resolve_pattern_call_target(
                rel_path,
                qualified_callee,
                same_file_symbols,
                go_imports,
                go_package_symbols,
            )
            if resolved[2] == "unresolved":
                continue
            events.append(
                _call_event(
                    rel_path,
                    language,
                    caller=current_function,
                    callee=callee,
                    callee_qualified_name=qualified_callee,
                    target_path=resolved[0],
                    target_qualified_name=resolved[1],
                    start_line=line_number,
                    resolution=resolved[2],
                )
            )
        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            current_function = None
    return events


def _pattern_function_name(language: str, line: str) -> str | None:
    if language == "go":
        match = _GO_FUNC_RE.match(line)
        return match.group(1) if match else None
    if language == "rust":
        match = _RUST_SYMBOL_RE.match(line)
        if match and match.group(1) == "fn":
            return match.group(2)
        return None
    if language == "java":
        match = _JAVA_METHOD_RE.match(line)
        return match.group(1) if match else None
    return None


def _resolve_pattern_call_target(
    rel_path: str,
    callee: str,
    same_file_symbols: dict[str, dict[str, Any]],
    go_imports: dict[str, str],
    go_package_symbols: dict[str, dict[str, tuple[str, str]]],
) -> tuple[str | None, str | None, str]:
    if callee in same_file_symbols:
        return rel_path, callee, "same_file_symbol"
    if "." in callee:
        qualifier, symbol_name = callee.split(".", 1)
        imported_module = go_imports.get(qualifier)
        if imported_module:
            target = go_package_symbols.get(imported_module, {}).get(symbol_name)
            if target:
                return target[0], target[1], "imported_symbol"
    return None, None, "unresolved"


def _qualified_pattern_call_name(line: str, match_start: int, callee: str) -> str:
    prefix = line[:match_start].rstrip()
    qualifier_match = re.search(r"""([A-Za-z_]\w*)\.$""", prefix)
    if qualifier_match:
        return f"{qualifier_match.group(1)}.{callee}"
    return callee


def _go_import_aliases(code_events: list[dict[str, Any]]) -> dict[str, str]:
    imports: dict[str, str] = {}
    for event in code_events:
        if event["event_type"] != "code.import.indexed":
            continue
        payload = event["payload"]
        module = str(payload["module"])
        alias = str(payload["name"]).rsplit("/", 1)[-1]
        imports[alias] = module
    return imports


def _go_package_symbols(code_events_by_path: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, tuple[str, str]]]:
    packages: dict[str, dict[str, tuple[str, str]]] = {}
    for rel_path, code_events in code_events_by_path.items():
        if not rel_path.endswith(".go"):
            continue
        package_path = str(Path(rel_path).parent).replace(".", "")
        if package_path == "":
            continue
        for module in {package_path, f"example.com/project/{package_path}"}:
            package_symbols = packages.setdefault(module, {})
            for event in code_events:
                if event["event_type"] != "code.symbol.indexed":
                    continue
                payload = event["payload"]
                if payload["kind"] != "function":
                    continue
                name = str(payload["qualified_name"])
                package_symbols[name] = (rel_path, name)
    return packages


def _nearest_python_callable(node: ast.AST, parent_by_node: dict[ast.AST, ast.AST | None]) -> str | None:
    parent = parent_by_node.get(node)
    while parent is not None:
        if isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
            return _python_qualified_name(parent, parent_by_node)
        parent = parent_by_node.get(parent)
    return None


def _python_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _python_call_name(node.value)
        if base is None:
            return node.attr
        return f"{base}.{node.attr}"
    return None


def _resolve_python_call_target(
    rel_path: str,
    callee: str,
    same_file_symbols: dict[str, dict[str, Any]],
    imported_symbols: dict[str, tuple[str, str]],
) -> tuple[str | None, str | None, str]:
    if callee in same_file_symbols:
        return rel_path, callee, "same_file_symbol"
    short_name = callee.rsplit(".", 1)[-1]
    if short_name in same_file_symbols:
        return rel_path, str(same_file_symbols[short_name]["qualified_name"]), "same_file_symbol"
    if callee in imported_symbols:
        target_path, target_qualified_name = imported_symbols[callee]
        return target_path, target_qualified_name, "imported_symbol"
    if short_name in imported_symbols:
        target_path, target_qualified_name = imported_symbols[short_name]
        return target_path, target_qualified_name, "imported_symbol"
    return None, None, "unresolved"


def _collect_coverage_events(
    rel_path: str,
    language: str,
    content: bytes,
    code_events: list[dict[str, Any]],
    file_index: set[str],
) -> list[dict[str, Any]]:
    if language != "python" or not _is_python_test_file(rel_path):
        return []
    text = content.decode("utf-8", errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imported_symbols = _python_imported_symbol_targets(rel_path, code_events, file_index)
    if not imported_symbols:
        return []
    events: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or not node.name.startswith("test_"):
            continue
        test_qualified_name = _python_qualified_name(node, _parent_map(tree))
        seen_targets: set[tuple[str, str, int]] = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            callee = _python_call_name(child.func)
            if callee is None:
                continue
            symbol_name = callee.rsplit(".", 1)[-1]
            target = imported_symbols.get(callee) or imported_symbols.get(symbol_name)
            if target is None:
                continue
            target_path, target_qualified_name = target
            key = (target_path, target_qualified_name, child.lineno)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            events.append(
                _coverage_event(
                    rel_path,
                    language,
                    test_name=node.name,
                    test_qualified_name=test_qualified_name,
                    target_path=target_path,
                    target_name=symbol_name,
                    target_qualified_name=target_qualified_name,
                    start_line=child.lineno,
                    resolution="imported_symbol",
                )
            )
    return events


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST | None]:
    parent_by_node: dict[ast.AST, ast.AST | None] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parent_by_node[child] = parent
    return parent_by_node


def _is_python_test_file(rel_path: str) -> bool:
    path = Path(rel_path)
    return path.name.startswith("test_") or any(part == "tests" for part in path.parts)


def _resolve_dependency_target(
    rel_path: str,
    language: str,
    module: str,
    file_index: set[str],
) -> tuple[str | None, str | None]:
    if language == "python":
        return _resolve_python_dependency(rel_path, module, file_index)
    if language in {"javascript", "typescript"}:
        return _resolve_js_ts_dependency(rel_path, module, file_index)
    return None, None


def _resolve_python_dependency(rel_path: str, module: str, file_index: set[str]) -> tuple[str | None, str | None]:
    if module.startswith("."):
        target = _resolve_python_relative_dependency(rel_path, module, file_index)
        return target, "relative_file" if target else None
    target = _first_existing_python_module_path(module, file_index)
    return target, "module_file" if target else None


def _resolve_python_relative_dependency(rel_path: str, module: str, file_index: set[str]) -> str | None:
    level = len(module) - len(module.lstrip("."))
    remainder = module[level:]
    source_parent = Path(rel_path).parent
    base_parts = list(source_parent.parts)
    if level > 1:
        base_parts = base_parts[: max(0, len(base_parts) - (level - 1))]
    module_parts = [part for part in remainder.split(".") if part]
    candidates = _python_module_candidates("/".join([*base_parts, *module_parts]))
    return _first_in_index(candidates, file_index)


def _first_existing_python_module_path(module: str, file_index: set[str]) -> str | None:
    module_path = module.replace(".", "/")
    prefixes = ["", "src/"]
    candidates: list[str] = []
    for prefix in prefixes:
        candidates.extend(_python_module_candidates(f"{prefix}{module_path}"))
    return _first_in_index(candidates, file_index)


def _python_module_candidates(module_path: str) -> list[str]:
    clean_path = module_path.strip("/")
    if not clean_path:
        return []
    return [f"{clean_path}.py", f"{clean_path}/__init__.py"]


def _resolve_js_ts_dependency(rel_path: str, module: str, file_index: set[str]) -> tuple[str | None, str | None]:
    if not module.startswith("."):
        return None, None
    base = Path(rel_path).parent / module
    base_path = base.as_posix()
    candidates: list[str] = []
    for suffix in (".ts", ".tsx", ".js", ".jsx"):
        candidates.append(f"{base_path}{suffix}")
    for suffix in (".ts", ".tsx", ".js", ".jsx"):
        candidates.append(f"{base_path}/index{suffix}")
    target = _first_in_index(candidates, file_index)
    return target, "relative_file" if target else None


def _first_in_index(candidates: list[str], file_index: set[str]) -> str | None:
    for candidate in candidates:
        normalized = Path(candidate).as_posix()
        if normalized in file_index:
            return normalized
    return None


def _call_event(
    rel_path: str,
    language: str,
    *,
    caller: str,
    callee: str,
    callee_qualified_name: str,
    target_path: str | None,
    target_qualified_name: str | None,
    start_line: int,
    resolution: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": rel_path,
        "language": language,
        "caller": caller,
        "callee": callee,
        "callee_qualified_name": callee_qualified_name,
        "start_line": start_line,
        "resolution": resolution,
    }
    if target_path is not None:
        payload["target_path"] = target_path
    if target_qualified_name is not None:
        payload["target_qualified_name"] = target_qualified_name
    return {
        "event_type": "code.call.indexed",
        "actor": "zaxy-codebase-indexer",
        "payload": payload,
    }


def _coverage_event(
    rel_path: str,
    language: str,
    *,
    test_name: str,
    test_qualified_name: str,
    target_path: str,
    target_name: str,
    target_qualified_name: str,
    start_line: int,
    resolution: str,
) -> dict[str, Any]:
    return {
        "event_type": "code.coverage.indexed",
        "actor": "zaxy-codebase-indexer",
        "payload": {
            "test_path": rel_path,
            "test_name": test_name,
            "test_qualified_name": test_qualified_name,
            "target_path": target_path,
            "target_name": target_name,
            "target_qualified_name": target_qualified_name,
            "language": language,
            "start_line": start_line,
            "resolution": resolution,
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
