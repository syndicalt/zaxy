# Codebase Mapping

Codebase mapping starts with a replayable file inventory. Zaxy walks a local
repository or directory, writes one `code.file.indexed` event per supported
source file, and projects each event into a `code_file` graph entity.

Run:

```bash
zaxy index-codebase . --session-id zaxy-default
```

Each event payload contains:

- `path`: POSIX path relative to the indexed root
- `language`: inferred from file suffix
- `sha256`: content digest
- `bytes`: file size
- `lines`: text line count

The collector skips hidden directories, `.git`, `.eventloom`, Python and Node
caches, virtual environments, dependency directories, and build outputs. It
also skips files larger than the configured byte limit.

This first slice does not parse symbols, imports, call graphs, or test
relationships. Those should layer on top of the file inventory with additional
event types, keeping Eventloom as the source of truth and Neo4j as the
projection.

The graph projection creates one `code_file` entity per indexed path. The
entity name is the relative path, the summary records language and line count,
and properties preserve the source path, language, SHA-256 hash, byte count, and
line count. The indexer actor is connected to each file with an
`indexed_code_file` relation. This makes the first map intentionally simple:
agents can ask which files exist, which languages are present, which files are
large, and whether a file hash changed after a later indexing run.

Use a project-scoped session when indexing. For example, a generated MCP config
for the Zaxy repository uses `EVENTLOOM_THREAD=zaxy-default`; indexing into that
same session keeps codebase inventory available to future `memory_query` and
`context_assemble` calls for this project. Do not index multiple unrelated
repositories into a raw `default` session. See [mcp.md](mcp.md) and
[eventloom.md](eventloom.md) for domain-separated defaults.

The file inventory is deliberately not a code search index. It does not include
file contents, snippets, private tokens, or full source text. If source-level
recall is needed, combine this feature with document ingestion for selected
files or add a later symbol extractor that records names, line ranges, imports,
and test relationships as structured events. Keeping the first layer metadata
only reduces accidental sensitive content capture while still creating a useful
map of the repository.

Reindexing appends new events instead of mutating old ones. That preserves the
timeline: a file can be observed with one hash today and another hash tomorrow.
Replay can regenerate the graph projection from the event stream, and temporal
queries can distinguish what the codebase looked like at different points.

Related pages: [eventloom.md](eventloom.md), [retrieval.md](retrieval.md),
[graph-schema.md](graph-schema.md), and [runbook.md](runbook.md). The public
site summary is [site/index.html](../site/index.html).
