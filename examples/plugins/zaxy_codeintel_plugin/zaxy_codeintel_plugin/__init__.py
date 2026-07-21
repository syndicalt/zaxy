"""Zaxy's code-intelligence layer, packaged as the reference external plugin.

This is the plugin API exercised against a real vertical rather than a toy: six
extractors covering files, symbols, imports, dependencies, call sites, and test
coverage across six languages, plus the repository walker that emits the events
they consume.

**Shared core, not a fork.** The extractor functions still live in
``zaxy.extract.rules_indexing`` and are still registered in-tree by their
``@register`` decorators — the built-in code-intelligence path is completely
unchanged. This package re-registers those same function objects through the
external :class:`~zaxy.plugins.PluginAPI`. Duplicating ~1,100 lines of language
parsing into the plugin would have created two sources of truth that drift; the
point of the exercise is to prove the *API* carries a real vertical, and it does
so honestly only if both paths run identical code.

Because registration is last-writer-wins per event type, installing this plugin
is idempotent against the built-ins: it re-installs the same callables.

Run it in-process::

    ZAXY_PLUGINS=zaxy_codeintel_plugin:PLUGIN

or subprocess-isolated::

    ZAXY_PLUGINS_OUT_OF_PROCESS=zaxy_codeintel_plugin:PLUGIN
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from zaxy.plugins import PluginAPI

__version__ = "0.1.0"

__all__ = ["PLUGIN", "CodeIntelligencePlugin", "collect_codebase_events", "event_types"]


def event_types() -> tuple[str, ...]:
    """Return the code-intelligence event types this plugin extracts."""
    from zaxy.extract.rules_indexing import CODE_INTELLIGENCE_EXTRACTORS

    return tuple(sorted(CODE_INTELLIGENCE_EXTRACTORS))


def collect_codebase_events(*args: object, **kwargs: object) -> object:
    """Walk a repository and emit code-intelligence events (re-exported)."""
    from zaxy.codebase import collect_codebase_events as _collect

    return _collect(*args, **kwargs)  # type: ignore[arg-type]


class CodeIntelligencePlugin:
    """Reference plugin installing Zaxy's six code-intelligence extractors."""

    name = "zaxy-codeintel-plugin"
    version = __version__

    def register(self, api: PluginAPI) -> None:
        """Install every code-intelligence extractor on the given :class:`PluginAPI`."""
        from zaxy.extract.rules_indexing import CODE_INTELLIGENCE_EXTRACTORS

        for event_type, fn in CODE_INTELLIGENCE_EXTRACTORS.items():
            api.register_extractor(event_type, fn)


# The object referenced by the entry point / ZAXY_PLUGINS import string.
PLUGIN = CodeIntelligencePlugin()
