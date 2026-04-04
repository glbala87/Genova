"""Structured request/response logging middleware for the Genova REST API.

Logs every HTTP request with method, path, status code, latency, and a
unique request ID. Supports structured JSON output and correlation ID
propagation.

Environment variables:
    GENOVA_REQUEST_LOGGING_ENABLED: Set to "1" or "true" to enable.
    GENOVA_LOG_FORMAT: "json" for structured JSON, "text" for human-readable (default: "json").
    GENOVA_LOG_LEVEL: Default log level (default: "INFO").
    GENOVA_LOG_LEVEL_HEALTH: Log level for /health endpoint (default: "DEBUG").
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Callable, Dict, Optional

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _get_log_level(endpoint: str, default: str = "INFO") -> str:
    """Return the configured log level for an endpoint."""
    # Allow per-endpoint overrides via env vars
    # e.g., GENOVA_LOG_LEVEL_HEALTH=DEBUG for /health
    suffix = endpoint.strip("/").upper().replace("/", "_")
    env_key = f"GENOVA_LOG_LEVEL_{suffix}" if suffix else "GENOVA_LOG_LEVEL"
    level = os.environ.get(env_key, "").upper()
    if level in _LOG_LEVELS:
        return level
    return default


def _should_log(level: str, min_level: str) -> bool:
    """Check if a log level meets the minimum threshold."""
    order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    try:
        return order.index(level) >= order.index(min_level)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for structured request/response logging.

    For every request, logs:
    - HTTP method
    - Request path
    - Response status code
    - Latency in milliseconds
    - Unique request ID (X-Request-ID header or auto-generated UUID)
    - Correlation ID (X-Correlation-ID header, propagated if present)
    - Client IP address

    The request ID and correlation ID are added to response headers for
    downstream traceability.

    Args:
        app: The ASGI application.
        log_format: "json" for structured JSON logging, "text" for
            human-readable format.
        default_level: Default log level for endpoints without specific config.
        exclude_paths: Set of paths to exclude from logging entirely.
    """

    def __init__(
        self,
        app: Any,
        log_format: Optional[str] = None,
        default_level: Optional[str] = None,
        exclude_paths: Optional[set] = None,
    ) -> None:
        super().__init__(app)
        self.log_format = log_format or os.environ.get(
            "GENOVA_LOG_FORMAT", "json"
        )
        self.default_level = (
            default_level
            or os.environ.get("GENOVA_LOG_LEVEL", "INFO")
        ).upper()
        self.exclude_paths = exclude_paths or set()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path

        # Skip excluded paths
        if path in self.exclude_paths:
            return await call_next(request)

        # Generate or extract request ID
        request_id = request.headers.get(
            "X-Request-ID", str(uuid.uuid4())
        )
        # Propagate correlation ID
        correlation_id = request.headers.get(
            "X-Correlation-ID", request_id
        )

        # Store on request state for downstream use
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id

        method = request.method
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("User-Agent", "")

        start_time = time.monotonic()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._log_request(
                method=method,
                path=path,
                status=status_code,
                latency_ms=elapsed_ms,
                request_id=request_id,
                correlation_id=correlation_id,
                client_ip=client_ip,
                user_agent=user_agent,
                error=str(exc),
            )
            raise

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Add IDs to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Correlation-ID"] = correlation_id

        self._log_request(
            method=method,
            path=path,
            status=status_code,
            latency_ms=elapsed_ms,
            request_id=request_id,
            correlation_id=correlation_id,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        return response

    def _log_request(
        self,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
        request_id: str,
        correlation_id: str,
        client_ip: str,
        user_agent: str = "",
        error: Optional[str] = None,
    ) -> None:
        """Emit a log entry for a request."""
        level = _get_log_level(path, self.default_level)

        if self.log_format == "json":
            log_data: Dict[str, Any] = {
                "event": "http_request",
                "method": method,
                "path": path,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "request_id": request_id,
                "correlation_id": correlation_id,
                "client_ip": client_ip,
            }
            if user_agent:
                log_data["user_agent"] = user_agent
            if error:
                log_data["error"] = error
            message = json.dumps(log_data)
        else:
            message = (
                f"{method} {path} -> {status} "
                f"({latency_ms:.1f}ms) "
                f"[{request_id}]"
            )
            if error:
                message += f" ERROR: {error}"

        # Log at the appropriate level
        if level == "DEBUG":
            logger.debug(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR" or error:
            logger.error(message)
        else:
            logger.info(message)


# ---------------------------------------------------------------------------
# Setup function
# ---------------------------------------------------------------------------


def setup_request_logging(
    app: Any,
    log_format: Optional[str] = None,
    default_level: Optional[str] = None,
    exclude_paths: Optional[set] = None,
) -> bool:
    """Configure request logging middleware on the FastAPI application.

    Returns True if logging was enabled, False otherwise.
    """
    enabled = os.environ.get("GENOVA_REQUEST_LOGGING_ENABLED", "").lower()
    if enabled not in ("1", "true", "yes", "on"):
        logger.info(
            "Request logging disabled "
            "(set GENOVA_REQUEST_LOGGING_ENABLED=1 to enable)."
        )
        return False

    app.add_middleware(
        RequestLoggingMiddleware,
        log_format=log_format,
        default_level=default_level,
        exclude_paths=exclude_paths,
    )
    logger.info("Request logging middleware enabled.")
    return True
