"""Tests for out-of-process plugin execution (zaxy.plugin_subprocess, zaxy.plugin_worker).

These exercise the real isolation claims end to end with real subprocesses: a
plugin that raises at import, a plugin that genuinely segfaults, and a plugin
that never returns. Each test asserts both halves of the guarantee — the host
survived AND the host reported the fault — because a crash test that passes
merely because the plugin never loaded proves nothing.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import zaxy.extract.core as extract_core
import zaxy.plugins as plugins
from zaxy.config import Settings
from zaxy.event import Event
from zaxy.extract import ExtractedEdge, ExtractedEntity, ExtractionResult, extract
from zaxy.plugin_ipc import (
    PROTOCOL_VERSION,
    decode_event,
    decode_extraction_result,
    encode_event,
    encode_extraction_result,
)
from zaxy.plugin_subprocess import (
    PluginWorker,
    PluginWorkerCrashedError,
    PluginWorkerError,
    PluginWorkerTimeoutError,
    active_workers,
    load_out_of_process_plugins,
    shutdown_workers,
)
from zaxy.plugins import PluginAPI, load_plugins

FIXTURE_DIR = Path(__file__).resolve().parent / "plugin_fixtures"


def _make_event(event_type: str, payload: dict | None = None, seq: int = 1) -> Event:
    """Build an Event with a dummy hash for extraction tests."""
    return Event(
        seq=seq,
        timestamp="2024-01-01T00:00:00Z",
        type=event_type,
        actor="test",
        thread="thread-a",
        payload=payload or {},
        hash="a" * 64,
    )


@pytest.fixture(autouse=True)
def _fixtures_importable_by_children(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Put the fixture plugins on PYTHONPATH so spawned workers can import them."""
    existing = os.environ.get("PYTHONPATH", "")
    combined = os.pathsep.join([str(FIXTURE_DIR), existing]) if existing else str(FIXTURE_DIR)
    monkeypatch.setenv("PYTHONPATH", combined)
    yield


@pytest.fixture(autouse=True)
def _restore_registries() -> Iterator[None]:
    """Restore extractor registries and reap workers so tests never leak state."""
    rules = dict(extract_core._RULES)
    loaded = set(plugins._LOADED_PLUGINS)
    try:
        yield
    finally:
        shutdown_workers()
        extract_core._RULES.clear()
        extract_core._RULES.update(rules)
        plugins._LOADED_PLUGINS.clear()
        plugins._LOADED_PLUGINS.update(loaded)


def _settings(references: list[str], timeout: float = 15.0) -> Settings:
    """Build Settings configuring the given out-of-process plugin references."""
    return Settings(plugins_out_of_process=references, plugin_timeout_seconds=timeout)


# ------------------------------------------------------------------
# IPC codec: lossless round trip
# ------------------------------------------------------------------

def test_encode_decode_event_round_trips_all_fields() -> None:
    """An Event survives the JSON wire encoding unchanged."""
    event = _make_event("remote.note", {"text": "hello", "nested": {"a": [1, 2]}}, seq=7)
    assert decode_event(encode_event(event)) == event


def test_encode_decode_extraction_result_round_trips_citations_losslessly() -> None:
    """Entity properties (citations), embeddings, and edge evidence survive the wire."""
    result = ExtractionResult(
        entities=[
            ExtractedEntity(
                name="n",
                entity_type="t",
                observed_at="2024-01-01T00:00:00Z",
                summary="sum",
                embedding=[0.5, -1.25],
                properties={"citations": ["a.py:1-2", "b.md:9"], "deep": {"x": [1, {"y": 2}]}},
            )
        ],
        edges=[
            ExtractedEdge(
                source="n",
                target="m",
                relation_type="cites",
                valid_from="2024-01-01T00:00:00Z",
                confidence=0.5,
                evidence={"citation": "a.py:1-2"},
            )
        ],
        source_event_seq=3,
        source_event_hash="h",
        source_event_prev_hash="p",
        source_event_type="remote.note",
        source_thread="thread-a",
    )
    assert decode_extraction_result(encode_extraction_result(result)) == result


def test_decode_extraction_result_rejects_non_object_payload() -> None:
    """A non-object extraction payload is rejected rather than silently accepted."""
    with pytest.raises(ValueError, match="JSON object"):
        decode_extraction_result(["not", "an", "object"])


def test_decode_event_rejects_non_object_payload() -> None:
    """A non-object event payload is rejected rather than silently accepted."""
    with pytest.raises(ValueError, match="JSON object"):
        decode_event("nope")


# ------------------------------------------------------------------
# Happy path: real subprocess, real extraction
# ------------------------------------------------------------------

def test_worker_describe_reports_protocol_name_and_event_types() -> None:
    """The handshake reports the plugin's identity and the event types it extracts."""
    worker = PluginWorker("fixture_good_plugin:PLUGIN", timeout=15.0)
    description = worker.describe()
    assert description.protocol == PROTOCOL_VERSION
    assert description.name == "remote-good"
    assert description.version == "9.9"
    assert description.event_types == ("remote.note",)
    worker.close()


def test_worker_extract_returns_citations_from_a_real_subprocess() -> None:
    """A remote extraction round-trips citations and embeddings out of a child process."""
    worker = PluginWorker("fixture_good_plugin:PLUGIN", timeout=15.0)
    result = worker.extract("remote.note", _make_event("remote.note", {"text": "hi"}, seq=4))
    entity = result.entities[0]
    assert entity.properties is not None
    assert entity.properties["citations"] == ["src/mod.py:10-14", "docs/spec.md:3"]
    assert entity.properties["nested"] == {"depth": [1, 2, {"leaf": True}]}
    assert entity.embedding == [0.25, -1.5, 3.0]
    assert result.edges[0].evidence == {"citation": "src/mod.py:10-14", "lines": [10, 14]}
    assert result.edges[0].confidence == 0.75
    assert result.source_event_seq == 4
    assert result.source_thread == "thread-a"
    worker.close()


def test_out_of_process_extractor_fires_through_the_real_extract_path() -> None:
    """An out-of-process plugin's extractor is reachable via zaxy.extract.extract."""
    api = PluginAPI()
    results = load_out_of_process_plugins(_settings(["fixture_good_plugin:PLUGIN"]), api)
    assert [r.status for r in results] == ["loaded"]
    assert results[0].source == "subprocess"

    extracted = extract(_make_event("remote.note", {"text": "through-extract"}))
    assert extracted.entities[0].entity_type == "remote_note"
    assert extracted.entities[0].properties is not None
    assert "src/mod.py:10-14" in extracted.entities[0].properties["citations"]


def test_host_never_imports_the_out_of_process_plugin_module() -> None:
    """The out-of-process path leaves the plugin module absent from the host process."""
    import sys

    sys.modules.pop("fixture_good_plugin", None)
    api = PluginAPI()
    load_out_of_process_plugins(_settings(["fixture_good_plugin:PLUGIN"]), api)
    extract(_make_event("remote.note", {"text": "x"}))
    assert "fixture_good_plugin" not in sys.modules


def test_plugin_stdout_noise_cannot_corrupt_the_protocol_stream() -> None:
    """A plugin that prints (even a forged frame) does not break the wire."""
    worker = PluginWorker("fixture_noisy_plugin:PLUGIN", timeout=15.0)
    assert worker.describe().name == "remote-noisy"
    result = worker.extract("remote.noisy", _make_event("remote.noisy"))
    assert result.source_event_seq == 1
    worker.close()


# ------------------------------------------------------------------
# Fault isolation: crash
# ------------------------------------------------------------------

def test_import_time_crash_is_reported_and_host_survives() -> None:
    """A plugin raising at import is reported as failed; the host keeps running."""
    host_pid = os.getpid()
    api = PluginAPI()
    results = load_out_of_process_plugins(_settings(["fixture_import_crash_plugin:PLUGIN"]), api)

    assert [r.status for r in results] == ["failed"]
    assert "exploded at import time" in (results[0].error or "")
    assert os.getpid() == host_pid
    # The host is still fully functional afterwards.
    assert extract(_make_event("remote.note")) is not None


def test_segfaulting_plugin_kills_only_its_own_process() -> None:
    """A genuine SIGSEGV in a plugin surfaces as an error; the host process survives."""
    host_pid = os.getpid()
    worker = PluginWorker("fixture_segfault_plugin:PLUGIN", timeout=15.0)
    assert worker.describe().event_types == ("remote.segfault",)

    with pytest.raises(PluginWorkerCrashedError) as excinfo:
        worker.extract("remote.segfault", _make_event("remote.segfault"))

    # -11 is SIGSEGV: the child really crashed rather than raising politely.
    assert "returncode=-11" in str(excinfo.value)
    assert worker.dead_reason is not None
    assert os.getpid() == host_pid


def test_segfaulting_extractor_degrades_to_empty_result_not_a_host_crash() -> None:
    """Through the extract path, a segfaulting plugin yields an empty result, not a crash."""
    host_pid = os.getpid()
    api = PluginAPI()
    load_out_of_process_plugins(_settings(["fixture_segfault_plugin:PLUGIN"]), api)

    result = extract(_make_event("remote.segfault", seq=12))
    assert result.entities == []
    assert result.edges == []
    assert result.source_event_seq == 12
    assert os.getpid() == host_pid


def test_dead_worker_fails_fast_instead_of_respawning() -> None:
    """Once a worker has died, later requests raise immediately rather than respawn."""
    worker = PluginWorker("fixture_segfault_plugin:PLUGIN", timeout=15.0)
    worker.describe()
    with pytest.raises(PluginWorkerCrashedError):
        worker.extract("remote.segfault", _make_event("remote.segfault"))

    started = time.monotonic()
    with pytest.raises(PluginWorkerCrashedError):
        worker.extract("remote.segfault", _make_event("remote.segfault"))
    assert time.monotonic() - started < 2.0


# ------------------------------------------------------------------
# Liveness: hang
# ------------------------------------------------------------------

def test_hanging_plugin_is_bounded_by_the_configured_timeout() -> None:
    """A plugin that never returns is killed at the deadline instead of hanging the host."""
    host_pid = os.getpid()
    worker = PluginWorker("fixture_hang_plugin:PLUGIN", timeout=1.0)
    assert worker.describe().event_types == ("remote.hang",)

    started = time.monotonic()
    with pytest.raises(PluginWorkerTimeoutError) as excinfo:
        worker.extract("remote.hang", _make_event("remote.hang"))
    elapsed = time.monotonic() - started

    assert 1.0 <= elapsed < 10.0
    assert "deadline" in str(excinfo.value)
    assert worker.dead_reason is not None
    assert os.getpid() == host_pid


def test_hanging_extractor_degrades_to_empty_result_within_the_deadline() -> None:
    """Through the extract path, a hanging plugin returns empty within the timeout."""
    api = PluginAPI()
    load_out_of_process_plugins(_settings(["fixture_hang_plugin:PLUGIN"], timeout=1.0), api)

    started = time.monotonic()
    result = extract(_make_event("remote.hang", seq=5))
    elapsed = time.monotonic() - started

    assert result.entities == []
    assert result.source_event_seq == 5
    assert elapsed < 10.0


def test_hung_worker_process_is_actually_killed() -> None:
    """Timing out a worker reaps the child process rather than leaking it."""
    worker = PluginWorker("fixture_hang_plugin:PLUGIN", timeout=1.0)
    worker.describe()
    process = worker._process
    assert process is not None

    with pytest.raises(PluginWorkerTimeoutError):
        worker.extract("remote.hang", _make_event("remote.hang"))

    assert process.poll() is not None


# ------------------------------------------------------------------
# Loader behavior
# ------------------------------------------------------------------

def test_projection_backends_are_refused_across_the_process_boundary() -> None:
    """A remote plugin's projection backend is reported unsupported, not silently dropped."""
    worker = PluginWorker("fixture_backend_plugin:PLUGIN", timeout=15.0)
    description = worker.describe()
    assert description.unsupported_backends == ("remote-null",)
    assert description.event_types == ()
    worker.close()


def test_loader_reuses_a_live_worker_across_repeated_loads() -> None:
    """Loading the same out-of-process plugin twice reuses one child process."""
    api = PluginAPI()
    settings = _settings(["fixture_good_plugin:PLUGIN"])
    load_out_of_process_plugins(settings, api)
    first = active_workers()["fixture_good_plugin:PLUGIN"]

    results = load_out_of_process_plugins(settings, api)
    assert [r.status for r in results] == ["loaded"]
    assert active_workers()["fixture_good_plugin:PLUGIN"] is first


def test_blank_references_are_skipped() -> None:
    """Empty entries in the out-of-process plugin list are ignored."""
    api = PluginAPI()
    assert load_out_of_process_plugins(_settings(["", "   "]), api) == []


def test_unknown_op_is_reported_without_killing_the_worker() -> None:
    """An unrecognized protocol op returns an error frame and leaves the worker alive."""
    worker = PluginWorker("fixture_good_plugin:PLUGIN", timeout=15.0)
    worker.describe()
    response = worker.request({"op": "nonsense"})
    assert response["ok"] is False
    assert "nonsense" in response["error"]
    assert worker.describe().name == "remote-good"
    worker.close()


def test_extracting_an_unregistered_event_type_reports_an_error() -> None:
    """Asking a worker for an event type it does not handle is a handled error."""
    worker = PluginWorker("fixture_good_plugin:PLUGIN", timeout=15.0)
    worker.describe()
    with pytest.raises(PluginWorkerError):
        worker.extract("not.registered", _make_event("not.registered"))
    assert worker.dead_reason is None
    worker.close()


def test_shutdown_workers_closes_and_clears_the_registry() -> None:
    """shutdown_workers reaps every supervised child and empties the registry."""
    api = PluginAPI()
    load_out_of_process_plugins(_settings(["fixture_good_plugin:PLUGIN"]), api)
    assert active_workers()

    shutdown_workers()
    assert active_workers() == {}


def test_close_on_an_unstarted_worker_is_safe() -> None:
    """Closing a worker that never spawned marks it closed without error."""
    worker = PluginWorker("fixture_good_plugin:PLUGIN", timeout=5.0)
    worker.close()
    assert worker.dead_reason == "closed"


# ------------------------------------------------------------------
# Integration with load_plugins / default behavior
# ------------------------------------------------------------------

def test_load_plugins_includes_out_of_process_results() -> None:
    """load_plugins reports subprocess plugins alongside in-process ones."""
    report = load_plugins(_settings(["fixture_good_plugin:PLUGIN"]))
    subprocess_results = [r for r in report.results if r.source == "subprocess"]
    assert [r.name for r in subprocess_results] == ["remote-good"]
    assert subprocess_results[0].status == "loaded"


def test_out_of_process_is_opt_in_and_spawns_nothing_by_default() -> None:
    """With no out-of-process plugins configured, no worker is ever spawned."""
    report = load_plugins(Settings())
    assert [r for r in report.results if r.source == "subprocess"] == []
    assert active_workers() == {}


def test_settings_parse_out_of_process_plugins_from_a_comma_separated_string() -> None:
    """ZAXY_PLUGINS_OUT_OF_PROCESS accepts a comma-separated list of import specs."""
    settings = Settings(plugins_out_of_process="a:PLUGIN, b:PLUGIN")
    assert settings.plugins_out_of_process == ["a:PLUGIN", "b:PLUGIN"]


def test_plugin_timeout_seconds_defaults_and_must_be_positive() -> None:
    """The worker deadline defaults to 10s and rejects non-positive values."""
    assert Settings().plugin_timeout_seconds == 10.0
    with pytest.raises(ValueError):
        Settings(plugin_timeout_seconds=0)


# ------------------------------------------------------------------
# CLI surface
# ------------------------------------------------------------------

def test_plugin_list_cli_reports_out_of_process_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`zaxy plugin list --json` surfaces subprocess plugins with source='subprocess'."""
    import json as _json

    from typer.testing import CliRunner

    from zaxy.cli.runtime import app

    monkeypatch.setenv("ZAXY_PLUGINS_OUT_OF_PROCESS", "fixture_good_plugin:PLUGIN")
    result = CliRunner().invoke(app, ["plugin", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = _json.loads(result.stdout)
    remote = [p for p in payload["plugins"] if p["source"] == "subprocess"]
    assert [p["name"] for p in remote] == ["remote-good"]
    assert remote[0]["status"] == "loaded"


def test_plugin_list_cli_reports_a_failing_out_of_process_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing subprocess plugin is listed as failed with its error, not omitted."""
    import json as _json

    from typer.testing import CliRunner

    from zaxy.cli.runtime import app

    monkeypatch.setenv("ZAXY_PLUGINS_OUT_OF_PROCESS", "fixture_import_crash_plugin:PLUGIN")
    result = CliRunner().invoke(app, ["plugin", "list", "--json"])

    assert result.exit_code == 0, result.output
    payload = _json.loads(result.stdout)
    remote = [p for p in payload["plugins"] if p["source"] == "subprocess"]
    assert remote and remote[0]["status"] == "failed"
    assert "exploded at import time" in remote[0]["error"]


# ------------------------------------------------------------------
# The reporting half of the guarantee
# ------------------------------------------------------------------

def test_crashing_plugin_records_a_degraded_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segfaulting plugin is reported to metrics as a degraded operation."""
    recorded: list[tuple[str, str]] = []

    class _Recorder:
        def record_degraded_operation(self, operation: str, reason: str) -> None:
            recorded.append((operation, reason))

    import zaxy.metrics

    monkeypatch.setattr(zaxy.metrics, "get_metrics", lambda: _Recorder())

    api = PluginAPI()
    load_out_of_process_plugins(_settings(["fixture_segfault_plugin:PLUGIN"]), api)
    extract(_make_event("remote.segfault"))

    assert recorded == [("plugin_out_of_process", "fixture_segfault_plugin:PLUGIN:PluginWorkerCrashedError")]


def test_crashing_plugin_is_logged_with_its_reference(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing out-of-process plugin names itself and the event type in the log."""
    api = PluginAPI()
    load_out_of_process_plugins(_settings(["fixture_segfault_plugin:PLUGIN"]), api)

    with caplog.at_level("WARNING"):
        extract(_make_event("remote.segfault"))

    assert any(
        "fixture_segfault_plugin:PLUGIN" in record.getMessage()
        and "remote.segfault" in record.getMessage()
        for record in caplog.records
    )
