# Codebase Mapping

Codebase mapping starts with replayable file, symbol, import, dependency,
call-site, and coverage events. Zaxy walks a local repository or directory,
writes one
`code.file.indexed` event per supported source file, and by default adds
`code.symbol.indexed`, `code.import.indexed`, `code.dependency.indexed`, and
`code.call.indexed` events inferred from the file type. Python test files can
also emit `code.coverage.indexed` events when tests call imported production
symbols.

Run:

```bash
zaxy index-codebase . --session-id zaxy-default
```

Each file event payload contains:

- `path`: POSIX path relative to the indexed root
- `language`: inferred from file suffix
- `sha256`: content digest
- `bytes`: file size
- `lines`: text line count

The collector skips hidden directories, `.git`, `.eventloom`, Python and Node
caches, virtual environments, dependency directories, and build outputs. It
also skips files larger than the configured byte limit.

Symbol events contain `path`, `language`, `name`, `qualified_name`, `kind`,
`start_line`, and `end_line`. Import events contain `path`, `language`,
`module`, `name`, `kind`, and `start_line`. Zaxy uses Python's standard `ast`
module for `.py` files and conservative file-type-specific patterns for common
source languages such as JavaScript, TypeScript, Go, Rust, Java, and shell.
Malformed or unsupported source files still get file inventory events; symbol
extraction for that file is skipped instead of aborting indexing.

Dependency events resolve imports to local files when that can be done without
guessing. Python absolute imports are matched against root-level and `src/`
layout modules, Python relative imports are resolved from the importing file,
and JavaScript/TypeScript relative imports are resolved against sibling files or
`index` modules. Unresolved third-party imports remain `code.import.indexed`
events only.

Python call-site events use the standard `ast` module to record calls inside
functions. JavaScript, TypeScript, Go, Rust, and Java call-site events use
conservative line scanners for function or method blocks. Calls to same-file
symbols and locally imported symbols are resolved to target files and qualified
names when possible; unresolved calls still keep a cited `code.call.indexed`
event with caller, callee, and line metadata.

Go call-site mapping can resolve package-qualified calls such as `worker.Run()`
when the imported package path maps to a scanned local package directory. Rust
call-site mapping can resolve simple `use crate::module::symbol` imports to
scanned sibling module files. Java call-site mapping currently resolves
same-file calls only.

Python coverage events are conservative. Files under `tests/` or named
`test_*.py` are scanned for `test_*` functions. When a test calls an imported
local production symbol, Zaxy records the test symbol, target symbol, target
file, and line number. This is not a replacement for runtime coverage; it is a
static map of obvious test-to-code relationships.

The graph projection creates one `code_file` entity per indexed path. The
entity name is the relative path, the summary records language and line count,
and properties preserve the source path, language, SHA-256 hash, byte count, and
line count. The indexer actor is connected to each file with an
`indexed_code_file` relation. Symbol events project to `code_symbol` entities
connected from the file with `defines_symbol`; import events project to
`code_import` entities connected from the file with `imports`. Resolved local
dependency events connect source and target `code_file` entities with
`depends_on_file`. Resolved call-site events connect caller and callee
`code_symbol` entities with `calls_symbol` and keep a `code_call` citation
entity for provenance. Coverage events connect test and production
`code_symbol` entities with `tests_symbol` and keep a `code_coverage` citation
entity.

Use a project-scoped session when indexing. For example, a generated MCP config
for the Zaxy repository uses `EVENTLOOM_THREAD=zaxy-default`; indexing into that
same session keeps codebase inventory available to future `memory_query` and
`context_assemble` calls for this project. Do not index multiple unrelated
repositories into a raw `default` session. See [mcp.md](mcp.md) and
[eventloom.md](eventloom.md) for domain-separated defaults.

The codebase map is deliberately not a code search index. It does not include
file contents, snippets, private tokens, or full source text. If source-level
recall is needed, combine this feature with document ingestion for selected
files. Keeping this layer metadata-only reduces accidental sensitive content
capture while still creating a useful structural map of the repository.

Reindexing appends new events instead of mutating old ones. That preserves the
timeline: a file can be observed with one hash today and another hash tomorrow.
Replay can regenerate the graph projection from the event stream, and temporal
queries can distinguish what the codebase looked like at different points.

Related pages: [eventloom.md](eventloom.md), [retrieval.md](retrieval.md),
[graph-schema.md](graph-schema.md), and [runbook.md](runbook.md). The public
site summary is [site/index.html](../site/index.html).
