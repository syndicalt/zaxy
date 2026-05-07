"""Prometheus metrics for Zaxy.

Exposes counters and histograms for key operations.
Useful for SLO monitoring and alerting.
"""

from __future__ import annotations

# Lazy import so Prometheus is optional
try:
    from prometheus_client import Counter, Histogram, start_http_server
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class MetricsCollector:
    """Prometheus metrics collector with graceful fallback.

    If prometheus_client is not installed, all operations are no-ops.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and _HAS_PROMETHEUS
        self._server_started = False

        if self.enabled:
            self.events_appended = Counter(
                "zaxy_events_appended_total",
                "Total events appended to Eventloom",
                ["event_type"],
            )
            self.queries_executed = Counter(
                "zaxy_queries_executed_total",
                "Total queries executed",
                ["source"],
            )
            self.query_duration = Histogram(
                "zaxy_query_duration_seconds",
                "Query execution time",
                ["source"],
            )
            self.graph_upserts = Counter(
                "zaxy_graph_upserts_total",
                "Total graph upserts",
                ["entity_type"],
            )
            self.invalidations = Counter(
                "zaxy_invalidations_total",
                "Total entity invalidations",
            )
            self.degraded_operations = Counter(
                "zaxy_degraded_operations_total",
                "Total degraded operations and fallback paths",
                ["operation", "reason"],
            )

    def start_server(self, port: int = 8080) -> None:
        """Start the Prometheus metrics HTTP server."""
        if self.enabled and not self._server_started:
            start_http_server(port)
            self._server_started = True

    def record_event_append(self, event_type: str) -> None:
        """Record an event append operation."""
        if self.enabled:
            self.events_appended.labels(event_type=event_type).inc()

    def record_query(self, duration_s: float, source: str = "hybrid") -> None:
        """Record a query execution."""
        if self.enabled:
            self.queries_executed.labels(source=source).inc()
            self.query_duration.labels(source=source).observe(duration_s)

    def record_upsert(self, entity_type: str) -> None:
        """Record a graph upsert operation."""
        if self.enabled:
            self.graph_upserts.labels(entity_type=entity_type).inc()

    def record_invalidation(self) -> None:
        """Record an invalidation operation."""
        if self.enabled:
            self.invalidations.inc()

    def record_degraded_operation(self, operation: str, reason: str) -> None:
        """Record a graceful degradation or fallback path."""
        if self.enabled:
            self.degraded_operations.labels(operation=operation, reason=reason).inc()


# Global singleton
_metrics: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    """Return the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector(enabled=_HAS_PROMETHEUS)
    return _metrics
