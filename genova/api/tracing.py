"""OpenTelemetry tracing for the Genova REST API.

Provides request tracing with span creation and trace propagation through
model inference. Exports traces to an OTLP collector.

Environment variables:
    GENOVA_TRACING_ENABLED: Set to "1" or "true" to enable tracing.
    GENOVA_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317).
    GENOVA_SERVICE_NAME: Service name for traces (default: genova-api).

Gracefully degrades when ``opentelemetry`` packages are not installed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Optional OpenTelemetry imports
# ---------------------------------------------------------------------------

_HAS_OTEL = False

try:
    from opentelemetry import trace
    from opentelemetry.context import attach, detach
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# No-op tracer fallback
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Minimal no-op span for use when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    """No-op tracer that returns no-op spans."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Global tracer reference
# ---------------------------------------------------------------------------

_tracer: Any = None


def get_tracer() -> Any:
    """Return the global tracer instance.

    Returns either an OpenTelemetry tracer or a no-op fallback.
    """
    global _tracer
    if _tracer is None:
        if _HAS_OTEL:
            _tracer = trace.get_tracer("genova.api")
        else:
            _tracer = _NoOpTracer()
    return _tracer


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[dict] = None,
) -> Generator[Any, None, None]:
    """Create a traced span as a context manager.

    Works with or without OpenTelemetry installed.

    Args:
        name: Span name.
        attributes: Optional dict of span attributes.

    Yields:
        The span object (real or no-op).
    """
    tracer = get_tracer()

    if _HAS_OTEL and hasattr(tracer, "start_as_current_span"):
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, v)
            yield span
    else:
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


def setup_tracing(
    app: Any,
    service_name: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """Configure OpenTelemetry tracing on the FastAPI application.

    Args:
        app: FastAPI application instance.
        service_name: Service name for traces.
        endpoint: OTLP collector endpoint.

    Returns:
        True if tracing was successfully configured, False otherwise.
    """
    global _tracer

    enabled = os.environ.get("GENOVA_TRACING_ENABLED", "").lower()
    if enabled not in ("1", "true", "yes", "on"):
        logger.info("Tracing disabled (set GENOVA_TRACING_ENABLED=1 to enable).")
        return False

    if not _HAS_OTEL:
        logger.warning(
            "OpenTelemetry packages not installed. Tracing will be no-op. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi"
        )
        return False

    service_name = service_name or os.environ.get(
        "GENOVA_SERVICE_NAME", "genova-api"
    )
    endpoint = endpoint or os.environ.get(
        "GENOVA_OTLP_ENDPOINT", "http://localhost:4317"
    )

    try:
        # Create resource
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": _get_version(),
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set as global provider
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("genova.api")

        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)

        logger.info(
            "OpenTelemetry tracing enabled: service={}, endpoint={}",
            service_name,
            endpoint,
        )
        return True

    except Exception as e:
        logger.error("Failed to configure tracing: {}", e)
        return False


def _get_version() -> str:
    """Get the Genova version string."""
    try:
        from genova import __version__

        return __version__
    except ImportError:
        return "unknown"
