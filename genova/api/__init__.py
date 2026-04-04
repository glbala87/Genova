"""REST API for serving Genova model predictions.

Public API
----------
Server:
    create_app

Inference:
    InferenceEngine

Security:
    APIKeyAuth, RateLimiter

Monitoring:
    GenovaMetrics, MetricsMiddleware, setup_monitoring, get_metrics

Tracing:
    setup_tracing, trace_span, get_tracer

Logging:
    RequestLoggingMiddleware, setup_request_logging

Schemas:
    VariantPredictionRequest, VariantPredictionResponse,
    ExpressionRequest, ExpressionResponse,
    MethylationRequest, MethylationResponse,
    EmbeddingRequest, EmbeddingResponse,
    HealthResponse, ModelInfoResponse
"""

from genova.api.inference import InferenceEngine
from genova.api.logging_middleware import RequestLoggingMiddleware, setup_request_logging
from genova.api.monitoring import (
    GenovaMetrics,
    MetricsMiddleware,
    get_metrics,
    setup_monitoring,
)
from genova.api.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResult,
    ExpressionRequest,
    ExpressionResponse,
    ExpressionResult,
    HealthResponse,
    MethylationRequest,
    MethylationResponse,
    MethylationResult,
    ModelInfoResponse,
    VariantPredictionRequest,
    VariantPredictionResponse,
    VariantPredictionResult,
)
from genova.api.security import APIKeyAuth, RateLimiter
from genova.api.server import create_app
from genova.api.tracing import get_tracer, setup_tracing, trace_span
from genova.api.websocket import WebSocketManager, register_websocket_routes
from genova.api.grpc_server import GenovaServicer, serve_grpc
from genova.api.batch_queue import BatchQueue, DynamicBatcher, Priority

__all__ = [
    # Server
    "create_app",
    # Inference
    "InferenceEngine",
    # Security
    "APIKeyAuth",
    "RateLimiter",
    # Monitoring
    "GenovaMetrics",
    "MetricsMiddleware",
    "get_metrics",
    "setup_monitoring",
    # Tracing
    "setup_tracing",
    "trace_span",
    "get_tracer",
    # Logging
    "RequestLoggingMiddleware",
    "setup_request_logging",
    # WebSocket
    "WebSocketManager",
    "register_websocket_routes",
    # gRPC
    "GenovaServicer",
    "serve_grpc",
    # Batch queue
    "BatchQueue",
    "DynamicBatcher",
    "Priority",
    # Schemas
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingResult",
    "ExpressionRequest",
    "ExpressionResponse",
    "ExpressionResult",
    "HealthResponse",
    "MethylationRequest",
    "MethylationResponse",
    "MethylationResult",
    "ModelInfoResponse",
    "VariantPredictionRequest",
    "VariantPredictionResponse",
    "VariantPredictionResult",
]
