# Local-First Retrieval Setup Design

## Goal

Make offline local retrieval setup a first-class Zaxy path by adding a CLI helper that emits and validates a deterministic local profile.

## Scope

The first slice targets a no-network baseline:

- `EMBEDDING_PROVIDER=hash`
- `RERANKER_PROVIDER=lexical`
- `EMBEDDING_ENABLED=true`
- `NEO4J_AUTO_START=true`
- no hosted API keys

The helper should not install model runtimes or manage third-party local services. Local HTTP reranker profiles can be added later once the deterministic path is polished.

## CLI Behavior

Add `zaxy local-profile`.

By default it prints an `.env`-style profile to stdout. With `--output PATH`, it writes the same profile to a file. It refuses to overwrite existing files unless `--force` is supplied.

With `--check`, the command validates the current offline-local profile by constructing the configured embedding provider and reranker. It reports actionable failures and exits non-zero through Typer errors when configuration is invalid.

## Components

- `src/zaxy/local_profile.py`: small pure functions for rendering profile text and checking local provider construction.
- `src/zaxy/__main__.py`: Typer command wiring.
- `docs/embeddings.md`, `docs/getting-started.md`, `docs/runbook.md`: document the offline profile path.
- `tests/test_local_profile.py`, `tests/test_cli.py`: unit and CLI coverage.

## Error Handling

The file writer refuses accidental overwrites. The checker surfaces provider construction errors as user-facing CLI errors instead of stack traces.

## Testing

Tests cover profile rendering, no-secret output, file overwrite protection, check success, and check failure for invalid local configuration.
