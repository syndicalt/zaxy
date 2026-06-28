# Plugin API

Zaxy can be extended by separately-installed Python packages. A plugin registers
**rule-based extractors** (new event types projected into the graph) and
**projection-store backends** (alternative queryable indexes), without forking
Zaxy. Plugins are loaded in-process and are fully optional: with no plugins
configured, Zaxy behaves exactly as it does today.

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
- **Isolated** — if a plugin fails to import or raises inside `register()`, the
  error is logged (`zaxy.plugins` logger) and recorded, and loading continues
  with the next plugin. A broken third-party plugin can never crash Zaxy.

You can also load plugins explicitly and inspect the result:

```python
from zaxy.config import get_settings
from zaxy.plugins import load_plugins

report = load_plugins(get_settings())
for result in report.results:
    print(result.name, result.version, result.source, result.status, result.error)
```

`PluginLoadReport.results` is a tuple of `PluginLoadResult(name, version, source,
status, error)`, where `source` is `"entry_point"` or `"config"` and `status` is
`"loaded"` or `"failed"`.

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

## Isolation model and future work

This is the standard Python plugin pattern: plugins are **external installed
packages loaded in-process**. They run with the same privileges as Zaxy, so only
install plugins you trust. Per-plugin error isolation prevents a faulty plugin
from crashing Zaxy, but it does not sandbox a plugin's side effects.

True out-of-process / subprocess isolation (running each plugin in a separate
process with a constrained IPC surface) is a documented future extension; it is
intentionally **not** part of this API today.
