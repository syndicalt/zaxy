# Plugin API

Zaxy can be extended by separately-installed Python packages. A plugin registers
**rule-based extractors** (new event types projected into the graph) and
**projection-store backends** (alternative queryable indexes), without forking
Zaxy. Plugins load in-process by default, or in a fault-isolated subprocess when
opted in (see [Isolation model](#isolation-model)). They are fully optional: with
no plugins configured, Zaxy behaves exactly as it does today.

## The contract

A plugin is any object that satisfies the `zaxy.plugins.ZaxyPlugin` protocol:

```python
class ZaxyPlugin(Protocol):
    name: str
    version: str

    def register(self, api: PluginAPI) -> None: ...
```

Zaxy calls `register(api)` once, passing a `PluginAPI`:

```python
class PluginAPI:
    def register_extractor(
        self, event_type: str, fn: Callable[[Event], ExtractionResult]
    ) -> None: ...

    def register_projection_backend(
        self, name: str, factory: Callable[..., ProjectionStore]
    ) -> None: ...
```

- `register_extractor(event_type, fn)` installs a rule extractor for `event_type`.
  It is the imperative twin of the built-in `@zaxy.extract.register` decorator;
  the registered function is called by `zaxy.extract.extract` whenever an event of
  that type is seen. Last writer wins for a given event type.
- `register_projection_backend(name, factory)` records an external backend.
  When `PROJECTION_BACKEND` (or the `MemoryFabric(projection_backend=...)`
  argument) resolves to `name`, `zaxy.projection_backends.build_projection_store`
  calls `factory(config)` with the active `ProjectionBackendConfig` and uses the
  returned `ProjectionStore`.

## Discovery

Zaxy discovers plugins from two sources and de-duplicates them by name:

### 1. Entry points (installed packages)

An installed distribution declares the `zaxy.plugins` entry-point group:

```toml
# pyproject.toml of your plugin distribution
[project.entry-points."zaxy.plugins"]
example = "zaxy_example_plugin:PLUGIN"
```

After `pip install your-plugin`, Zaxy finds it automatically via
`importlib.metadata`. The entry-point value is a `"module:attr"` string pointing
at the plugin object.

### 2. The `ZAXY_PLUGINS` config

Without packaging, point Zaxy at a plugin object by import string. The
`Settings.plugins` field (env `ZAXY_PLUGINS`) is a comma-separated list of
`"module:attr"` specs:

```bash
export ZAXY_PLUGINS="my_pkg:PLUGIN,other_pkg:PLUGIN"
```

Both sources produce the same result; an entry-point plugin and a `ZAXY_PLUGINS`
entry that resolve to the same plugin object are registered at most once.

## When plugins load

Plugins load when a `MemoryFabric` is constructed (after settings are resolved,
before the projection store is built, so plugin backends are available). Loading
is:

- **Idempotent** — each plugin `name` is registered at most once per process;
  repeated loads do not re-run `register()`.
- **Isolated (per-plugin errors)** — if a plugin fails to import or raises inside
  `register()`, the error is logged (`zaxy.plugins` logger) and recorded, and
  loading continues with the next plugin. This covers *exceptions*; it does not
  cover a segfault or an infinite loop in an in-process plugin, which take the
  host down with them. Use the out-of-process mode for those.

You can also load plugins explicitly and inspect the result:

```python
from zaxy.config import get_settings
from zaxy.plugins import load_plugins

report = load_plugins(get_settings())
for result in report.results:
    print(result.name, result.version, result.source, result.status, result.error)
```

`PluginLoadReport.results` is a tuple of `PluginLoadResult(name, version, source,
status, error)`, where `source` is `"entry_point"`, `"config"`, or `"subprocess"`
and `status` is `"loaded"` or `"failed"`.

## CLI

List discovered plugins and their load status:

```bash
zaxy plugin list
zaxy plugin list --json
```

The JSON form emits `{"plugins": [{name, version, source, status, error}, ...]}`.

## Example plugin

A complete, installable reference plugin lives at
`examples/plugins/zaxy_example_plugin/`. It registers an extractor for the
`example.note` event type:

```python
# zaxy_example_plugin/__init__.py
from zaxy.extract import ExtractedEntity, ExtractionResult

def extract_example_note(event):
    text = (event.payload or {}).get("text")
    entity = ExtractedEntity(
        name=f"example-note:{event.seq}",
        entity_type="example_note",
        observed_at=event.timestamp,
        summary=str(text) if text else None,
    )
    return ExtractionResult(entities=[entity], edges=[], source_event_seq=event.seq)

class ExamplePlugin:
    name = "zaxy-example-plugin"
    version = "0.1.0"

    def register(self, api):
        api.register_extractor("example.note", extract_example_note)

PLUGIN = ExamplePlugin()
```

Load it without installing by setting `ZAXY_PLUGINS=zaxy_example_plugin:PLUGIN`
(with the package directory on `sys.path`), or install it (`pip install .`) to
expose the `zaxy.plugins` entry point.

## Reference plugin: the code-intelligence vertical

`examples/plugins/zaxy_codeintel_plugin/` packages Zaxy's own six-language code
intelligence as an external plugin — the API proven against a real vertical
rather than a toy. It registers all six `code.*` extractors (files, symbols,
imports, dependencies, call sites, test coverage) and re-exports the repository
walker.

It is a **shared core, not a fork**: the extractor functions still live in
`zaxy.extract.rules_indexing` and stay registered in-tree by their `@register`
decorators. The plugin re-registers those same function objects through
`PluginAPI` via the public `CODE_INTELLIGENCE_EXTRACTORS` map. Installing it
therefore does not change built-in behavior — registration is last-writer-wins
per event type, and it re-installs identical callables. Vendoring ~1,100 lines of
language parsing into the plugin would have created two sources of truth that
drift; the API proof is only honest if both paths run the same code.

## Isolation model

Zaxy offers two loading modes with genuinely different failure properties.

### In-process (default)

Everything above loads plugins **in-process** — the standard Python plugin
pattern. Per-plugin error isolation means an exception during import or
`register()` is caught and reported. But be clear about the limit: an
in-process plugin runs with Zaxy's full privileges, and a **hard crash
(segfault) or an infinite loop in it takes Zaxy down with it**, because it *is*
Zaxy's process. Only load in-process plugins you trust.

### Out-of-process (opt-in)

`ZAXY_PLUGINS_OUT_OF_PROCESS` (`Settings.plugins_out_of_process`) runs each
plugin in a supervised subprocess instead:

```bash
export ZAXY_PLUGINS_OUT_OF_PROCESS="my_pkg:PLUGIN"
export ZAXY_PLUGIN_TIMEOUT_SECONDS=10   # per-request deadline, default 10s
```

The host spawns `python -m zaxy.plugin_worker <module:attr>` per plugin and
registers **host-side stubs** for each event type the worker reports. The host
never imports the plugin module.

What this gives you, precisely:

- **Fault isolation.** A plugin that raises at import, or segfaults mid-call,
  kills only its own process. The host records a degraded operation
  (`plugin_out_of_process`), logs it, and the affected extraction returns an
  empty `ExtractionResult` instead of propagating.
- **Liveness.** Every request is bounded by `ZAXY_PLUGIN_TIMEOUT_SECONDS`. A
  worker that stops answering is killed and marked dead; later calls fail fast
  rather than respawning code that already proved unhealthy.

What it is **not**:

- **Not a security sandbox.** The child runs as the same user, with the same
  filesystem and network access as the host, and is handed event payloads.
  Untrusted plugin code still needs an OS-level sandbox (container, seccomp,
  separate user) layered on top.
- **Projection backends are not supported out-of-process.** A `ProjectionStore`
  is a live object with open connections, not data, so it cannot cross a process
  boundary. A remote plugin requesting one has the request reported as
  unsupported and logged — not silently dropped.

### IPC contract

Newline-delimited JSON, one object per line, over the worker's stdin/stdout:

```
host   -> worker  {"op": "describe"}
worker -> host    {"ok": true, "protocol": 1, "name": ..., "version": ...,
                   "event_types": [...], "unsupported_backends": [...]}

host   -> worker  {"op": "extract", "event_type": ..., "event": {...}}
worker -> host    {"ok": true, "result": {...}}

host   -> worker  {"op": "shutdown"}
```

Handled failures return `{"ok": false, "error": "..."}` rather than closing the
pipe, so the host distinguishes a *reported* plugin error from a crash (pipe
closed) from a hang (no line before the deadline).

Extraction results round-trip losslessly, including citations: entity
`properties` and edge `evidence` are arbitrary JSON and survive intact. The
worker also dups the real stdout and re-points `sys.stdout` at stderr before
importing plugin code, so a plugin that calls `print()` cannot corrupt the wire.
