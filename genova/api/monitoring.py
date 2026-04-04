"""Prometheus metrics and observability for the Genova REST API.

Exposes request counters, latency histograms, active request gauges,
model inference metrics, and custom genomic metrics. All metrics are
served at the ``/metrics`` endpoint in Prometheus text format.

Environment variables:
    GENOVA_METRICS_ENABLED: Set to "1" or "true" to enable metrics.
    GENOVA_METRICS_PREFIX: Metric name prefix (default: "genova").

Gracefully degrades when ``prometheus_client`` is not installed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Optional prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


# ---------------------------------------------------------------------------
# No-op fallbacks when prometheus_client is missing
# ---------------------------------------------------------------------------


class _NoOpMetric:
    """Fallback metric that silently discards all operations."""

    def labels(self, *args: Any, **kwargs: Any) -> "_NoOpMetric":
        return self

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------------------


class GenovaMetrics:
    """Container for all Genova Prometheus metrics.

    Attributes:
        request_count: Counter of HTTP requests by endpoint, method, status.
        request_latency: Histogram of HTTP request duration.
        active_requests: Gauge of currently in-flight requests.
        inference_latency: Histogram of model inference duration.
        gpu_memory_usage: Gauge of GPU memory usage in bytes.
        predictions_per_second: Gauge tracking prediction throughput.
        variants_processed: Counter of total variants processed.
    """

    def __init__(self, prefix: str = "genova") -> None:
        self.prefix = prefix
        self.enabled = _HAS_PROMETHEUS

        if self.enabled:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self._init_noop_metrics()

    def _init_prometheus_metrics(self) -> None:
        p = self.prefix

        self.request_count = Counter(
            f"{p}_http_requests_total",
            "Total HTTP requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )

        self.request_latency = Histogram(
            f"{p}_http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )

        self.active_requests = Gauge(
            f"{p}_http_active_requests",
            "Number of active HTTP requests",
            registry=self.registry,
        )

        self.inference_latency = Histogram(
            f"{p}_model_inference_duration_seconds",
            "Model inference latency in seconds",
            ["task"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )

        self.gpu_memory_usage = Gauge(
            f"{p}_gpu_memory_bytes",
            "GPU memory usage in bytes",
            ["device", "type"],
            registry=self.registry,
        )

        self.predictions_per_second = Gauge(
            f"{p}_predictions_per_second",
            "Current prediction throughput",
            ["task"],
            registry=self.registry,
        )

        self.variants_processed = Counter(
            f"{p}_variants_processed_total",
            "Total number of variants processed",
            registry=self.registry,
        )

        self.sequences_processed = Counter(
            f"{p}_sequences_processed_total",
            "Total number of sequences processed",
            ["task"],
            registry=self.registry,
        )

    def _init_noop_metrics(self) -> None:
        self.registry = None
        noop = _NoOpMetric()
        self.request_count = noop  # type: ignore[assignment]
        self.request_latency = noop  # type: ignore[assignment]
        self.active_requests = noop  # type: ignore[assignment]
        self.inference_latency = noop  # type: ignore[assignment]
        self.gpu_memory_usage = noop  # type: ignore[assignment]
        self.predictions_per_second = noop  # type: ignore[assignment]
        self.variants_processed = noop  # type: ignore[assignment]
        self.sequences_processed = noop  # type: ignore[assignment]

    def generate_latest(self) -> bytes:
        """Generate the latest metrics in Prometheus text format."""
        if self.enabled and self.registry is not None:
            return generate_latest(self.registry)
        return b"# prometheus_client not installed\n"

    def content_type(self) -> str:
        """Return the appropriate content type for the /metrics response."""
        if self.enabled:
            return CONTENT_TYPE_LATEST
        return "text/plain; charset=utf-8"

    def update_gpu_metrics(self) -> None:
        """Update GPU memory metrics if CUDA is available."""
        if not self.enabled:
            return
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device = f"cuda:{i}"
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    self.gpu_memory_usage.labels(device=device, type="allocated").set(
                        allocated
                    )
                    self.gpu_memory_usage.labels(device=device, type="reserved").set(
                        reserved
                    )
        except ImportError:
            pass

    def record_inference(
        self, task: str, duration: float, num_items: int
    ) -> None:
        """Record an inference event with latency and throughput."""
        self.inference_latency.labels(task=task).observe(duration)
        self.sequences_processed.labels(task=task).inc(num_items)
        if duration > 0:
            self.predictions_per_second.labels(task=task).set(
                num_items / duration
            )


# ---------------------------------------------------------------------------
# Singleton metrics instance
# ---------------------------------------------------------------------------

_metrics: Optional[GenovaMetrics] = None


def get_metrics() -> GenovaMetrics:
    """Return the global metrics instance, creating it if needed."""
    global _metrics
    if _metrics is None:
        prefix = os.environ.get("GENOVA_METRICS_PREFIX", "genova")
        _metrics = GenovaMetrics(prefix=prefix)
    return _metrics


# ---------------------------------------------------------------------------
# Metrics middleware
# ---------------------------------------------------------------------------


class MetricsMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that records request metrics.

    Tracks request count, latency, and active request gauge for every
    HTTP request. Skips the /metrics endpoint itself to avoid recursion.
    """

    def __init__(self, app: Any, metrics: Optional[GenovaMetrics] = None) -> None:
        super().__init__(app)
        self.metrics = metrics or get_metrics()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Don't instrument the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        endpoint = request.url.path
        method = request.method

        self.metrics.active_requests.inc()
        start = time.monotonic()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            duration = time.monotonic() - start
            self.metrics.active_requests.dec()
            self.metrics.request_count.labels(
                endpoint=endpoint, method=method, status=status
            ).inc()
            self.metrics.request_latency.labels(endpoint=endpoint).observe(duration)

        return response


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


def setup_monitoring(app: Any) -> Optional[GenovaMetrics]:
    """Configure Prometheus metrics and /metrics endpoint on the FastAPI app.

    Returns the metrics instance, or None if metrics are disabled.

    This is a no-op if GENOVA_METRICS_ENABLED is not set to a truthy value.
    """
    enabled = os.environ.get("GENOVA_METRICS_ENABLED", "").lower()
    if enabled not in ("1", "true", "yes", "on"):
        logger.info("Metrics disabled (set GENOVA_METRICS_ENABLED=1 to enable).")
        return None

    if not _HAS_PROMETHEUS:
        logger.warning(
            "prometheus_client not installed. Metrics will be no-op. "
            "Install with: pip install prometheus_client"
        )

    metrics = get_metrics()

    # Add middleware
    app.add_middleware(MetricsMiddleware, metrics=metrics)

    # Register /metrics endpoint
    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus metrics endpoint."""
        metrics.update_gpu_metrics()
        body = metrics.generate_latest()
        return Response(content=body, media_type=metrics.content_type())

    logger.info("Prometheus metrics enabled at /metrics")
    return metrics
