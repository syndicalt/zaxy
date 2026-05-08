# Codebase File Map Design

## Goal

Add the first codebase-mapping slice: durable file inventory events with graph projection.

## Scope

This slice indexes files only. It does not parse symbols, imports, call graphs, or test relationships. Those belong in later static-analysis slices.

## Architecture

Zaxy should preserve the existing Eventloom-first model:

1. walk a repository or directory;
2. emit `code.file.indexed` event inputs with relative path, language, sha256, byte count, and line count;
3. append those events through `MemoryFabric`, which writes Eventloom and projects graph facts through existing extraction;
4. expose a CLI command for local indexing.

## Event Shape

`code.file.indexed` payload:

- `path`: POSIX-style path relative to the indexed root
- `language`: inferred language label
- `sha256`: content digest
- `bytes`: file size in bytes
- `lines`: text line count

The event actor is `zaxy-codebase-indexer`.

## Exclusions

The collector skips hidden directories, `.git`, `.eventloom`, virtualenvs, caches, dependency directories, build outputs, and files larger than the configured byte limit.

## Graph Projection

The extractor creates a `code_file` entity named by relative path, with summary text containing language and line count. It preserves path, language, sha256, bytes, and lines as properties. It also creates an actor edge with relation type `indexed_code_file`.

## CLI

Add:

```bash
zaxy index-codebase PATH --session-id default
```

The command appends events through `MemoryFabric` and prints the indexed count.

## Tests

Tests cover collection, exclusions, language inference, extractor projection, MemoryFabric ingestion, and CLI behavior.
