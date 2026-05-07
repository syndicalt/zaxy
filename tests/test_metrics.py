"""Tests for zaxy.metrics — Prometheus metrics collector.

Covers the MetricsCollector and its no-op fallback when prometheus_client
is unavailable."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from zaxy.metrics import MetricsCollector, get_metrics

# ------------------------------------------------------------------
# MetricsCollector — with prometheus_client mocked
# ------------------------------------------------------------------

class TestMetricsCollectorEnabled:
    @pytest.fixture
    def mock_prometheus(self) -> MagicMock:
        """Patch prometheus_client classes so no real server starts."""
        with (
            patch("zaxy.metrics.Counter") as mock_counter,
            patch("zaxy.metrics.Histogram") as mock_hist,
            patch("zaxy.metrics.start_http_server") as mock_start,
            patch("zaxy.metrics._HAS_PROMETHEUS", True),
        ):
            mock_counter.return_value = MagicMock()
            mock_hist.return_value = MagicMock()
            yield {
                "Counter": mock_counter,
                "Histogram": mock_hist,
                "start_http_server": mock_start,
            }

    def test_initializes_metrics(self, mock_prometheus: MagicMock) -> None:
        """When enabled, all metric objects should be created."""
        mc = MetricsCollector(enabled=True)
        assert mc.enabled is True
        mock_prometheus["Counter"].assert_any_call(
            "zaxy_events_appended_total",
            "Total events appended to Eventloom",
            ["event_type"],
        )
        mock_prometheus["Counter"].assert_any_call(
            "zaxy_degraded_operations_total",
            "Total degraded operations and fallback paths",
            ["operation", "reason"],
        )
        mock_prometheus["Histogram"].assert_any_call(
            "zaxy_query_duration_seconds",
            "Query execution time",
            ["source"],
        )

    def test_start_server(self, mock_prometheus: MagicMock) -> None:
        """start_server should call start_http_server once."""
        mc = MetricsCollector(enabled=True)
        mc.start_server(port=9999)
        mock_prometheus["start_http_server"].assert_called_once_with(9999)
        # Second call should be a no-op
        mc.start_server(port=9999)
        mock_prometheus["start_http_server"].assert_called_once()

    def test_record_event_append(self, mock_prometheus: MagicMock) -> None:
        """record_event_append should increment the counter with labels."""
        mc = MetricsCollector(enabled=True)
        mc.record_event_append("goal.created")
        mc.events_appended.labels.assert_called_once_with(event_type="goal.created")
        mc.events_appended.labels.return_value.inc.assert_called_once()

    def test_record_query(self, mock_prometheus: MagicMock) -> None:
        """record_query should increment counter and observe histogram."""
        mc = MetricsCollector(enabled=True)
        mc.record_query(duration_s=0.123, source="exact")
        mc.queries_executed.labels.assert_called_once_with(source="exact")
        mc.queries_executed.labels.return_value.inc.assert_called_once()
        mc.query_duration.labels.assert_called_once_with(source="exact")
        mc.query_duration.labels.return_value.observe.assert_called_once_with(0.123)

    def test_record_upsert(self, mock_prometheus: MagicMock) -> None:
        """record_upsert should increment the upsert counter."""
        mc = MetricsCollector(enabled=True)
        mc.record_upsert("Goal")
        mc.graph_upserts.labels.assert_called_once_with(entity_type="Goal")
        mc.graph_upserts.labels.return_value.inc.assert_called_once()

    def test_record_invalidation(self, mock_prometheus: MagicMock) -> None:
        """record_invalidation should increment the invalidation counter."""
        mc = MetricsCollector(enabled=True)
        mc.record_invalidation()
        mc.invalidations.inc.assert_called_once()

    def test_record_degraded_operation(self, mock_prometheus: MagicMock) -> None:
        """record_degraded_operation should increment the fallback counter."""
        mc = MetricsCollector(enabled=True)
        mc.record_degraded_operation("query", "graph_unavailable")
        mc.degraded_operations.labels.assert_called_once_with(
            operation="query",
            reason="graph_unavailable",
        )
        mc.degraded_operations.labels.return_value.inc.assert_called_once()


# ------------------------------------------------------------------
# MetricsCollector — disabled / no prometheus_client
# ------------------------------------------------------------------

class TestMetricsCollectorDisabled:
    def test_noop_when_disabled(self) -> None:
        """When enabled=False, all methods should be no-ops."""
        mc = MetricsCollector(enabled=False)
        assert mc.enabled is False
        # These should not raise
        mc.start_server(port=8080)
        mc.record_event_append("x")
        mc.record_query(0.1)
        mc.record_upsert("x")
        mc.record_invalidation()
        mc.record_degraded_operation("query", "graph_unavailable")

    @patch("zaxy.metrics._HAS_PROMETHEUS", False)
    def test_disabled_when_import_missing(self) -> None:
        """When prometheus_client is not installed, enabled should be False."""
        mc = MetricsCollector(enabled=True)
        assert mc.enabled is False

    @patch("zaxy.metrics._HAS_PROMETHEUS", False)
    def test_noop_when_import_missing(self) -> None:
        """All operations should be safe when prometheus_client is absent."""
        mc = MetricsCollector(enabled=True)
        mc.start_server(port=8080)
        mc.record_event_append("x")
        mc.record_query(0.1)
        mc.record_upsert("x")
        mc.record_invalidation()
        mc.record_degraded_operation("query", "graph_unavailable")


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------

class TestGetMetrics:
    def test_returns_same_instance(self) -> None:
        """get_metrics should return a singleton."""
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_returns_metrics_collector(self) -> None:
        """get_metrics should return a MetricsCollector instance."""
        m = get_metrics()
        assert isinstance(m, MetricsCollector)
