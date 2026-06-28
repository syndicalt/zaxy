"""Tests for the external plugin API (zaxy.plugins).

These genuinely exercise the contract: extractors and projection backends
registered by external plugins must fire through the real ``extract`` and
``build_projection_store`` paths, failures must be isolated, loading must be
idempotent, and default (no-plugin) behavior must be unchanged.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import zaxy.extract.core as extract_core
import zaxy.plugins as plugins
from zaxy.config import Settings, get_settings
from zaxy.event import Event
from zaxy.extract import extract
from zaxy.null_projection_store import NullProjectionStore
from zaxy.plugins import (
    ENTRY_POINT_GROUP,
    PluginAPI,
    discover_plugin_specs,
    load_plugins,
)
from zaxy.projection_backends import ProjectionBackendConfig, build_projection_store

EXAMPLE_PLUGIN_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "plugins" / "zaxy_example_plugin"
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_event(event_type: str, payload: dict, actor: str = "test") -> Event:
    """Build an Event with a dummy hash for extraction tests."""
    return Event(
        seq=1,
        timestamp="2024-01-01T00:00:00Z",
        type=event_type,
        actor=actor,
        payload=payload,
        hash="a" * 64,
    )


def _backend_config(backend: str) -> ProjectionBackendConfig:
    return ProjectionBackendConfig(
        backend=backend,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="testpassword",
        neo4j_ca_cert=None,
        neo4j_trust_all=False,
    )


# ------------------------------------------------------------------
# Test plugin objects (resolved via "module:attr" import strings)
# ------------------------------------------------------------------

def _good_extractor(event: Event):  # noqa: ANN202 - test helper
    from zaxy.extract import ExtractedEntity, ExtractionResult

    return ExtractionResult(
        entities=[
            ExtractedEntity(name="good", entity_type="good", observed_at=event.timestamp)
        ],
        edges=[],
        source_event_seq=event.seq,
    )


class _GoodPlugin:
    name = "good-plugin"
    version = "1.0"

    def __init__(self) -> None:
        self.register_calls = 0

    def register(self, api: PluginAPI) -> None:
        self.register_calls += 1
        api.register_extractor("good.event", _good_extractor)


class _BoomPlugin:
    name = "boom-plugin"
    version = "9.9"

    def register(self, api: PluginAPI) -> None:
        raise RuntimeError("boom failed to register")


class _CountingPlugin:
    name = "counting-plugin"
    version = "2.0"

    def __init__(self) -> None:
        self.register_calls = 0

    def register(self, api: PluginAPI) -> None:
        self.register_calls += 1
        api.register_extractor("counting.event", _good_extractor)


class _FakeStore:
    def __init__(self, config: ProjectionBackendConfig) -> None:
        self.config = config


class _BackendPlugin:
    name = "backend-plugin"
    version = "3.0"

    def register(self, api: PluginAPI) -> None:
        api.register_projection_backend("example_backend", _FakeStore)


_GOOD_PLUGIN = _GoodPlugin()
_BOOM_PLUGIN = _BoomPlugin()
_COUNTING_PLUGIN = _CountingPlugin()
_BACKEND_PLUGIN = _BackendPlugin()


class _SideEffectNamePlugin:
    """A plugin whose ``name`` access raises a non-AttributeError descriptor error."""

    version = "6.6"

    @property
    def name(self) -> str:
        raise RuntimeError("name descriptor exploded")

    def register(self, api: PluginAPI) -> None:  # pragma: no cover - never reached
        raise AssertionError("register must not run when name access fails")


_SIDE_EFFECT_PLUGIN = _SideEffectNamePlugin()


# ------------------------------------------------------------------
# Fixtures: isolate the process-global registries between tests
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    rules = dict(extract_core._RULES)
    backends = dict(plugins._PROJECTION_BACKENDS)
    loaded = set(plugins._LOADED_PLUGINS)
    _GOOD_PLUGIN.register_calls = 0
    _COUNTING_PLUGIN.register_calls = 0
    try:
        yield
    finally:
        extract_core._RULES.clear()
        extract_core._RULES.update(rules)
        plugins._PROJECTION_BACKENDS.clear()
        plugins._PROJECTION_BACKENDS.update(backends)
        plugins._LOADED_PLUGINS.clear()
        plugins._LOADED_PLUGINS.update(loaded)


@pytest.fixture
def example_on_path() -> Iterator[None]:
    sys.path.insert(0, str(EXAMPLE_PLUGIN_DIR))
    try:
        yield
    finally:
        if str(EXAMPLE_PLUGIN_DIR) in sys.path:
            sys.path.remove(str(EXAMPLE_PLUGIN_DIR))
        sys.modules.pop("zaxy_example_plugin", None)


# ------------------------------------------------------------------
# Config-path discovery + extractor actually fires
# ------------------------------------------------------------------

def test_config_plugin_loads_and_extractor_fires(example_on_path: None) -> None:
    settings = Settings(plugins=["zaxy_example_plugin:PLUGIN"])

    report = load_plugins(settings)

    by_name = {r.name: r for r in report.results}
    assert "zaxy-example-plugin" in by_name
    result_entry = by_name["zaxy-example-plugin"]
    assert result_entry.status == "loaded"
    assert result_entry.source == "config"
    assert result_entry.version == "0.1.0"
    assert result_entry.error is None

    # The plugin's extractor must now fire through the real extract() path.
    extraction = extract(_make_event("example.note", {"text": "hello world"}))
    assert len(extraction.entities) == 1
    entity = extraction.entities[0]
    assert entity.entity_type == "example_note"
    assert entity.name == "example-note:1"
    assert entity.summary == "hello world"


# ------------------------------------------------------------------
# Error isolation: a broken plugin is reported, never fatal, and does not
# block a subsequent good plugin.
# ------------------------------------------------------------------

def test_failed_plugin_is_isolated_and_does_not_block_others() -> None:
    settings = Settings(
        plugins=[f"{__name__}:_BOOM_PLUGIN", f"{__name__}:_GOOD_PLUGIN"]
    )

    report = load_plugins(settings)

    by_name = {r.name: r for r in report.results}
    assert by_name["boom-plugin"].status == "failed"
    assert "boom failed to register" in (by_name["boom-plugin"].error or "")
    # The good plugin still loaded despite the earlier failure.
    assert by_name["good-plugin"].status == "loaded"
    assert _GOOD_PLUGIN.register_calls == 1
    assert "good.event" in extract_core._RULES
    # The broken plugin did not register anything.
    assert "boom-plugin" not in plugins._LOADED_PLUGINS


def test_bad_import_string_is_reported_failed_not_raised() -> None:
    settings = Settings(plugins=["zaxy_no_such_module_xyz:PLUGIN"])

    report = load_plugins(settings)

    assert len(report.failed) == 1
    assert report.failed[0].status == "failed"
    assert report.failed[0].error


def test_side_effecting_attribute_access_is_isolated() -> None:
    """A plugin whose name/version/register descriptor raises must not escape load_plugins."""
    settings = Settings(
        plugins=[f"{__name__}:_SIDE_EFFECT_PLUGIN", f"{__name__}:_GOOD_PLUGIN"]
    )

    # Must NOT raise even though the first plugin's `name` access explodes.
    report = load_plugins(settings)

    failed = [r for r in report.results if r.status == "failed"]
    assert len(failed) == 1
    assert "name descriptor exploded" in (failed[0].error or "")
    # The good plugin still loaded despite the earlier exploding descriptor.
    by_name = {r.name: r for r in report.results}
    assert by_name["good-plugin"].status == "loaded"
    assert _GOOD_PLUGIN.register_calls == 1


# ------------------------------------------------------------------
# Idempotency: repeated load_plugins does not double-register.
# ------------------------------------------------------------------

def test_load_plugins_is_idempotent() -> None:
    settings = Settings(plugins=[f"{__name__}:_COUNTING_PLUGIN"])

    first = load_plugins(settings)
    second = load_plugins(settings)

    assert _COUNTING_PLUGIN.register_calls == 1
    assert [r.status for r in first.results] == ["loaded"]
    assert [r.status for r in second.results] == ["loaded"]
    assert "counting-plugin" in plugins._LOADED_PLUGINS


# ------------------------------------------------------------------
# Projection backend: a plugin-registered backend resolves; unknown raises.
# ------------------------------------------------------------------

def test_plugin_projection_backend_resolves_through_factory() -> None:
    settings = Settings(plugins=[f"{__name__}:_BACKEND_PLUGIN"])
    load_plugins(settings)

    config = _backend_config("example_backend")
    store = build_projection_store(config)

    assert isinstance(store, _FakeStore)
    assert store.config is config


def test_unknown_backend_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="projection backend must be one of"):
        build_projection_store(_backend_config("does-not-exist"))


# ------------------------------------------------------------------
# Entry-point discovery: discover_plugin_specs reads the zaxy.plugins group.
# ------------------------------------------------------------------

def test_discover_reads_entry_point_group(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_point = importlib_metadata.EntryPoint(
        name="example",
        value=f"{__name__}:_GOOD_PLUGIN",
        group=ENTRY_POINT_GROUP,
    )
    captured: dict[str, object] = {}

    def fake_entry_points(*, group: str | None = None):  # noqa: ANN202 - test stub
        captured["group"] = group
        return [entry_point] if group == ENTRY_POINT_GROUP else []

    monkeypatch.setattr(importlib_metadata, "entry_points", fake_entry_points)

    specs = discover_plugin_specs(Settings(plugins=[]))

    assert captured["group"] == ENTRY_POINT_GROUP
    ep_specs = [s for s in specs if s.source == "entry_point"]
    assert any(s.name == "example" for s in ep_specs)
    spec = next(s for s in ep_specs if s.name == "example")
    assert spec.load() is _GOOD_PLUGIN

    # And it loads through the entry-point path.
    report = load_plugins(Settings(plugins=[]))
    good = [r for r in report.results if r.name == "good-plugin"]
    assert good and good[0].status == "loaded"
    assert good[0].source == "entry_point"


# ------------------------------------------------------------------
# CLI: `zaxy plugin list --json` lists a configured plugin.
# ------------------------------------------------------------------

def test_cli_plugin_list_json(
    monkeypatch: pytest.MonkeyPatch, example_on_path: None
) -> None:
    from typer.testing import CliRunner

    from zaxy.__main__ import app

    monkeypatch.setenv("ZAXY_PLUGINS", "zaxy_example_plugin:PLUGIN")
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["plugin", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    by_name = {p["name"]: p for p in payload["plugins"]}
    assert "zaxy-example-plugin" in by_name
    assert by_name["zaxy-example-plugin"]["status"] == "loaded"
    assert by_name["zaxy-example-plugin"]["source"] == "config"
    assert by_name["zaxy-example-plugin"]["version"] == "0.1.0"


# ------------------------------------------------------------------
# Backward compatibility: with no plugins, behavior is unchanged.
# ------------------------------------------------------------------

def test_no_plugins_extract_and_backends_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZAXY_PLUGINS", raising=False)
    settings = Settings(plugins=[])

    # No config-sourced specs discovered.
    config_specs = [s for s in discover_plugin_specs(settings) if s.source == "config"]
    assert config_specs == []

    # Generic fallback extraction is unchanged.
    fallback = extract(_make_event("unknown.event", {"foo": "bar"}))
    assert fallback.entities[0].entity_type == "event"

    # Built-in projection backends still route as before.
    store = build_projection_store(_backend_config("null"))
    assert isinstance(store, NullProjectionStore)

    # load_plugins with empty config registers nothing from config.
    report = load_plugins(settings)
    assert all(r.source != "config" for r in report.results)


class _NoVersionPlugin:
    name = "noversion-plugin"
    # No ``version`` attribute -> _plugin_str_attr returns the default.

    def register(self, api: PluginAPI) -> None:
        api.register_extractor("noversion.event", _good_extractor)


_NO_VERSION_PLUGIN = _NoVersionPlugin()


def test_cli_plugin_list_human_no_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from zaxy.__main__ import app

    monkeypatch.delenv("ZAXY_PLUGINS", raising=False)
    monkeypatch.setattr(plugins, "_entry_point_specs", lambda: [])
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["plugin", "list"])

    assert result.exit_code == 0, result.output
    assert "No plugins discovered" in result.output


def test_cli_plugin_list_human_reports_loaded_and_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from zaxy.__main__ import app

    monkeypatch.setenv("ZAXY_PLUGINS", f"{__name__}:_GOOD_PLUGIN,{__name__}:_BOOM_PLUGIN")
    monkeypatch.setattr(plugins, "_entry_point_specs", lambda: [])
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["plugin", "list"])

    assert result.exit_code == 0, result.output
    assert "good-plugin 1.0 [config] loaded" in result.output
    assert "boom-plugin" in result.output
    assert "failed: boom failed to register" in result.output


def test_malformed_and_duplicate_config_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugins, "_entry_point_specs", lambda: [])
    settings = Settings(
        plugins=[
            "no-colon-here",
            "",
            f"{__name__}:_GOOD_PLUGIN",
            f"{__name__}:_GOOD_PLUGIN",
            f"{__name__}:_NO_VERSION_PLUGIN",
        ]
    )

    report = load_plugins(settings)

    by_name = {r.name: r for r in report.results}
    # Malformed 'module:attr' reference is reported failed, not raised.
    assert by_name["no-colon-here"].status == "failed"
    assert "module:attr" in (by_name["no-colon-here"].error or "")
    # Empty + duplicate references are skipped; the good plugin registers once.
    assert by_name["good-plugin"].status == "loaded"
    assert _GOOD_PLUGIN.register_calls == 1
    # A plugin with no version attribute renders the default empty version.
    assert by_name["noversion-plugin"].status == "loaded"
    assert by_name["noversion-plugin"].version == ""
    # The loaded-report property reflects only the successfully registered plugins.
    assert {r.name for r in report.loaded} == {"good-plugin", "noversion-plugin"}
