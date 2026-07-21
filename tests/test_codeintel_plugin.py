"""Tests for the code-intelligence reference plugin (examples/plugins/zaxy_codeintel_plugin).

The plugin shares the in-tree extractor implementations rather than vendoring
them, so these assert both halves of that arrangement: the plugin really does
install the whole vertical through the external PluginAPI, and the built-in
in-tree path is left byte-for-byte unchanged by installing it.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import zaxy.extract.core as extract_core
import zaxy.plugins as plugins
from zaxy.config import Settings
from zaxy.event import Event
from zaxy.extract import extract
from zaxy.extract.rules_indexing import CODE_INTELLIGENCE_EXTRACTORS
from zaxy.plugins import PluginAPI, load_plugins

PLUGIN_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "plugins" / "zaxy_codeintel_plugin"
)

EXPECTED_EVENT_TYPES = (
    "code.call.indexed",
    "code.coverage.indexed",
    "code.dependency.indexed",
    "code.file.indexed",
    "code.import.indexed",
    "code.symbol.indexed",
)


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Restore the extractor registry so plugin registration never leaks between tests."""
    rules = dict(extract_core._RULES)
    loaded = set(plugins._LOADED_PLUGINS)
    try:
        yield
    finally:
        extract_core._RULES.clear()
        extract_core._RULES.update(rules)
        plugins._LOADED_PLUGINS.clear()
        plugins._LOADED_PLUGINS.update(loaded)


@pytest.fixture
def plugin_on_path() -> Iterator[None]:
    """Make the reference plugin importable without installing it."""
    sys.path.insert(0, str(PLUGIN_DIR))
    try:
        yield
    finally:
        if str(PLUGIN_DIR) in sys.path:
            sys.path.remove(str(PLUGIN_DIR))
        sys.modules.pop("zaxy_codeintel_plugin", None)


def _file_indexed_event() -> Event:
    """Build a representative code.file.indexed event."""
    return Event(
        seq=1,
        timestamp="2024-01-01T00:00:00Z",
        type="code.file.indexed",
        actor="test",
        payload={"path": "src/mod.py", "language": "python", "line_count": 42},
        hash="a" * 64,
    )


def test_shared_core_map_covers_the_six_code_intelligence_event_types() -> None:
    """CODE_INTELLIGENCE_EXTRACTORS exposes exactly the six code.* extractors."""
    assert tuple(sorted(CODE_INTELLIGENCE_EXTRACTORS)) == EXPECTED_EVENT_TYPES


def test_shared_core_map_holds_the_same_callables_the_decorators_registered() -> None:
    """The public map is the in-tree registry's own functions, not copies."""
    for event_type, fn in CODE_INTELLIGENCE_EXTRACTORS.items():
        assert extract_core._RULES[event_type] is fn


def test_plugin_declares_the_full_vertical(plugin_on_path: None) -> None:
    """The reference plugin advertises all six code-intelligence event types."""
    import zaxy_codeintel_plugin

    assert zaxy_codeintel_plugin.event_types() == EXPECTED_EVENT_TYPES
    assert zaxy_codeintel_plugin.PLUGIN.name == "zaxy-codeintel-plugin"
    assert zaxy_codeintel_plugin.PLUGIN.version == zaxy_codeintel_plugin.__version__


def test_plugin_registers_every_extractor_through_the_public_api(plugin_on_path: None) -> None:
    """register() installs all six extractors via PluginAPI, not by importing internals."""
    import zaxy_codeintel_plugin

    installed: dict[str, object] = {}

    class _RecordingAPI(PluginAPI):
        def register_extractor(self, event_type: str, fn: object) -> None:  # type: ignore[override]
            installed[event_type] = fn

    zaxy_codeintel_plugin.PLUGIN.register(_RecordingAPI())
    assert tuple(sorted(installed)) == EXPECTED_EVENT_TYPES


def test_loading_the_plugin_leaves_the_in_tree_path_unchanged(plugin_on_path: None) -> None:
    """Installing the plugin re-registers identical callables: built-in behavior is preserved."""
    before = {event_type: extract_core._RULES[event_type] for event_type in EXPECTED_EVENT_TYPES}
    baseline = extract(_file_indexed_event())

    report = load_plugins(Settings(plugins=["zaxy_codeintel_plugin:PLUGIN"]))
    assert [r.status for r in report.results if r.name == "zaxy-codeintel-plugin"] == ["loaded"]

    after = {event_type: extract_core._RULES[event_type] for event_type in EXPECTED_EVENT_TYPES}
    assert after == before
    assert extract(_file_indexed_event()) == baseline


def test_plugin_extraction_matches_the_built_in_extraction(plugin_on_path: None) -> None:
    """The plugin-installed extractor yields the same graph as the in-tree path.

    ``extract`` additionally stamps source provenance (hash/type/thread) that the
    raw extractor does not, so the comparison is over the projected graph itself.
    """
    import zaxy_codeintel_plugin

    event = _file_indexed_event()
    builtin = extract(event)
    plugin_fn = CODE_INTELLIGENCE_EXTRACTORS["code.file.indexed"]
    projected = plugin_fn(event)

    assert projected.entities == builtin.entities
    assert projected.edges == builtin.edges
    assert projected.entities, "code.file.indexed must project at least one entity"
    assert "code.file.indexed" in zaxy_codeintel_plugin.event_types()


def test_plugin_reexports_the_repository_walker(plugin_on_path: None, tmp_path: Path) -> None:
    """collect_codebase_events is re-exported and walks a real repository."""
    import zaxy_codeintel_plugin

    (tmp_path / "mod.py").write_text("def hello():\n    return 1\n", encoding="utf-8")
    events = zaxy_codeintel_plugin.collect_codebase_events(tmp_path)

    assert isinstance(events, list)
    assert any(event["event_type"] == "code.file.indexed" for event in events)
    assert any(event["event_type"] == "code.symbol.indexed" for event in events)


def test_plugin_declares_the_entry_point_for_discovery() -> None:
    """The distribution declares a zaxy.plugins entry point so installing it is enough."""
    manifest = (PLUGIN_DIR / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.entry-points."zaxy.plugins"]' in manifest
    assert 'codeintel = "zaxy_codeintel_plugin:PLUGIN"' in manifest
