"""External plugin API for Zaxy.

Separately-installed Python packages extend Zaxy by registering rule-based
extractors and projection-store backends. Plugins are discovered from two
sources:

1. ``importlib.metadata`` entry points in the ``zaxy.plugins`` group, declared
   by installed distributions, e.g.::

       [project.entry-points."zaxy.plugins"]
       example = "zaxy_example_plugin:PLUGIN"

2. The :attr:`zaxy.config.Settings.plugins` list (env ``ZAXY_PLUGINS``), a list
   of ``"module:attr"`` import strings resolved at load time.

Each plugin exposes ``name``/``version`` attributes and a ``register(api)``
method that receives a :class:`PluginAPI`. Loading is **isolated** per plugin:
a failure to import or register one plugin is logged and recorded, never raised,
so a broken third-party plugin can never crash Zaxy. Loading is **idempotent**:
the same plugin name is registered at most once per process.

This loads external *installed packages* in-process, the standard Python plugin
pattern. True out-of-process / subprocess isolation is a documented future
extension; see ``docs/plugins.md``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from zaxy.log import get_logger

if TYPE_CHECKING:
    from zaxy.config import Settings
    from zaxy.event import Event
    from zaxy.extract import ExtractionResult
    from zaxy.projection import ProjectionStore

logger = get_logger("plugins")

ENTRY_POINT_GROUP = "zaxy.plugins"

# Module-level registries shared across the process. External projection
# backends are keyed by their casefolded name; ``build_projection_store``
# consults this map for any name that is not a built-in backend.
_PROJECTION_BACKENDS: dict[str, Callable[..., ProjectionStore]] = {}

# Names (plugin ``.name`` values) already registered this process. Repeated
# ``load_plugins`` calls skip these so registration never double-fires.
_LOADED_PLUGINS: set[str] = set()


class ZaxyPlugin(Protocol):  # pragma: no cover
    """The contract an external Zaxy plugin object must satisfy."""

    name: str
    version: str

    def register(self, api: PluginAPI) -> None:
        """Install this plugin's capabilities via ``api``."""
        ...


class PluginAPI:
    """Capabilities handed to a plugin's :meth:`ZaxyPlugin.register` method.

    The API is intentionally small and stable: it delegates to the same
    in-process registries the built-ins use, so plugin-registered extractors and
    backends are indistinguishable from native ones at call time.
    """

    def register_extractor(
        self,
        event_type: str,
        fn: Callable[[Event], ExtractionResult],
    ) -> None:
        """Register a rule-based extractor for ``event_type``.

        Mirrors the ``@zaxy.extract.register`` decorator; last writer wins for a
        given event type.
        """
        from zaxy.extract.core import register_extractor as _register_extractor

        _register_extractor(event_type, fn)

    def register_projection_backend(
        self,
        name: str,
        factory: Callable[..., ProjectionStore],
    ) -> None:
        """Register an external projection-store backend under ``name``.

        ``factory`` is invoked by :func:`zaxy.projection_backends.build_projection_store`
        with the active ``ProjectionBackendConfig`` when ``PROJECTION_BACKEND``
        resolves to ``name``.
        """
        register_projection_backend(name, factory)


def register_projection_backend(
    name: str,
    factory: Callable[..., ProjectionStore],
) -> None:
    """Record an external projection-backend ``factory`` under ``name``."""
    _PROJECTION_BACKENDS[name.casefold().strip()] = factory


def get_projection_backend_factory(name: str) -> Callable[..., ProjectionStore] | None:
    """Return an external backend factory for ``name``, or ``None`` if unknown."""
    return _PROJECTION_BACKENDS.get(name.casefold().strip())


@dataclass(frozen=True)
class PluginSpec:
    """A discovered, not-yet-loaded plugin reference."""

    name: str
    """Discovery identifier: the entry-point name or the ``module:attr`` string."""
    source: str
    """Where the spec came from: ``"entry_point"`` or ``"config"``."""
    reference: str
    """Human-readable resolution target (``module:attr``)."""
    load: Callable[[], Any] = field(compare=False, repr=False)
    """Resolve and return the plugin object. May raise on bad imports."""


@dataclass(frozen=True)
class PluginLoadResult:
    """Outcome of attempting to load one plugin."""

    name: str
    version: str
    source: str
    status: str
    """``"loaded"`` or ``"failed"``."""
    error: str | None = None


@dataclass(frozen=True)
class PluginLoadReport:
    """Aggregate report for one :func:`load_plugins` call."""

    results: tuple[PluginLoadResult, ...] = ()

    @property
    def loaded(self) -> tuple[PluginLoadResult, ...]:
        """Plugins that registered successfully (or were already loaded)."""
        return tuple(r for r in self.results if r.status == "loaded")

    @property
    def failed(self) -> tuple[PluginLoadResult, ...]:
        """Plugins that failed to import or register."""
        return tuple(r for r in self.results if r.status == "failed")


def _config_loader(reference: str) -> Callable[[], Any]:
    """Return a thunk that resolves a ``module:attr`` import string."""

    def _load() -> Any:
        return _resolve_reference(reference)

    return _load


def _resolve_reference(reference: str) -> Any:
    """Resolve a ``"module:attr"`` import string to its object."""
    module_name, separator, attr_path = reference.partition(":")
    if not separator or not module_name.strip() or not attr_path.strip():
        raise ValueError(f"plugin spec {reference!r} must be in 'module:attr' form")
    obj: Any = importlib.import_module(module_name.strip())
    for part in attr_path.strip().split("."):
        obj = getattr(obj, part)
    return obj


def _entry_point_specs() -> list[PluginSpec]:
    """Return plugin specs from installed ``zaxy.plugins`` entry points."""
    try:
        entry_points = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - defensive; metadata is reliable
        logger.warning("Failed to read %r entry points: %s", ENTRY_POINT_GROUP, exc)
        return []
    return [
        PluginSpec(name=ep.name, source="entry_point", reference=ep.value, load=ep.load)
        for ep in entry_points
    ]


def discover_plugin_specs(settings: Settings) -> list[PluginSpec]:
    """Discover plugin specs from entry points and ``settings.plugins``.

    Entry points are listed first, then config import strings. Specs are
    de-duplicated by :attr:`PluginSpec.name`, preserving first occurrence.
    """
    specs: list[PluginSpec] = []
    seen: set[str] = set()
    for spec in _entry_point_specs():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        specs.append(spec)
    for raw in settings.plugins:
        reference = raw.strip()
        if not reference or reference in seen:
            continue
        seen.add(reference)
        specs.append(
            PluginSpec(
                name=reference,
                source="config",
                reference=reference,
                load=_config_loader(reference),
            )
        )
    return specs


def _plugin_str_attr(plugin: object, attr: str, *, default: str) -> str:
    """Return a non-empty string attribute from ``plugin`` or ``default``."""
    value = getattr(plugin, attr, None)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _load_spec(spec: PluginSpec, api: PluginAPI) -> PluginLoadResult:
    """Resolve and register one plugin spec, isolating ANY failure (never raises).

    A single guard wraps resolution, attribute access (``name``/``version``/
    ``register``), and ``register(api)`` so a side-effecting descriptor or a
    non-callable ``register`` can never propagate out of plugin loading.
    """
    name = spec.name
    version = ""
    try:
        plugin = spec.load()
        name = _plugin_str_attr(plugin, "name", default=spec.name)
        version = _plugin_str_attr(plugin, "version", default="")
        if name in _LOADED_PLUGINS:
            # Idempotent: already registered earlier this process; skip register().
            return PluginLoadResult(name=name, version=version, source=spec.source, status="loaded")
        register = plugin.register
        register(api)
    except Exception as exc:
        logger.warning(
            "Failed to load Zaxy plugin %r (source=%s): %s", name, spec.source, exc
        )
        return PluginLoadResult(
            name=name, version=version, source=spec.source, status="failed", error=str(exc)
        )

    _LOADED_PLUGINS.add(name)
    return PluginLoadResult(name=name, version=version, source=spec.source, status="loaded")


def load_plugins(settings: Settings) -> PluginLoadReport:
    """Discover and load all configured plugins, isolating per-plugin failures.

    Idempotent: plugins whose ``name`` was already registered this process are
    reported as ``loaded`` without re-running ``register()``.
    """
    api = PluginAPI()
    results = [_load_spec(spec, api) for spec in discover_plugin_specs(settings)]
    return PluginLoadReport(results=tuple(results))
