# zaxy-example-plugin

A minimal reference plugin for [Zaxy](https://github.com/syndicalt/zaxy) showing
the external plugin contract end to end. It registers a rule extractor for the
`example.note` event type.

## What it does

The package exposes a `PLUGIN` object that satisfies the `zaxy.plugins.ZaxyPlugin`
protocol (`name`, `version`, `register`). Its `register(api)` calls
`api.register_extractor("example.note", ...)`, so any `example.note` event flowing
through `zaxy.extract.extract` produces an `example_note` entity.

## How Zaxy discovers it

Two equivalent paths (see `docs/plugins.md` in the Zaxy repo):

1. **Entry point (installed package).** This `pyproject.toml` declares:

   ```toml
   [project.entry-points."zaxy.plugins"]
   example = "zaxy_example_plugin:PLUGIN"
   ```

   After `pip install .`, Zaxy loads it automatically.

2. **Config import string.** Without installing, point Zaxy at the module:

   ```bash
   export ZAXY_PLUGINS=zaxy_example_plugin:PLUGIN
   ```

## Try it

```bash
pip install .
zaxy plugin list            # shows: zaxy-example-plugin 0.1.0 [entry_point] loaded
```

```python
from zaxy.config import get_settings
from zaxy.plugins import load_plugins
from zaxy.extract import extract
from zaxy.event import Event

load_plugins(get_settings())
event = Event(
    seq=1,
    timestamp="2024-01-01T00:00:00Z",
    type="example.note",
    actor="demo",
    payload={"text": "hello"},
    hash="a" * 64,
)
result = extract(event)
print(result.entities[0].entity_type)  # -> "example_note"
```
