# zaxy-codeintel-plugin

Zaxy's code-intelligence layer packaged as the **reference external plugin** —
the plugin API proven against a real vertical instead of a toy extractor.

It registers six extractors:

| Event type | Produces |
| --- | --- |
| `code.file.indexed` | file inventory nodes |
| `code.symbol.indexed` | symbols, linked to the defining file |
| `code.import.indexed` | imports, linked to the importing file |
| `code.dependency.indexed` | resolved file-to-file dependencies |
| `code.call.indexed` | call sites and resolved call edges |
| `code.coverage.indexed` | test-to-production symbol coverage |

The repository walker that emits these events (`collect_codebase_events`,
6 languages: Python, JS/TS, Go, Rust, Java, shell) is re-exported here too.

## Shared core, not a fork

The extractor functions live in `zaxy.extract.rules_indexing` and remain
registered in-tree by their `@register` decorators. **Installing this plugin does
not change built-in behavior** — it re-registers the same function objects
through the external `PluginAPI`, which is last-writer-wins per event type.

Vendoring ~1,100 lines of language parsing into the plugin would have created two
sources of truth that drift apart. The exercise is to prove the *API* can carry a
real vertical; that proof is only honest if both paths run identical code. The
shared surface is the public `CODE_INTELLIGENCE_EXTRACTORS` map.

## Usage

Install (`pip install .`) to register via the `zaxy.plugins` entry point, or
without installing:

```bash
# in-process (default path)
export ZAXY_PLUGINS=zaxy_codeintel_plugin:PLUGIN

# subprocess-isolated (fault-isolated; see docs/plugins.md)
export ZAXY_PLUGINS_OUT_OF_PROCESS=zaxy_codeintel_plugin:PLUGIN
```
