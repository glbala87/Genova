"""gRPC interface for Genova model inference.

Provides a gRPC service with endpoints mirroring the REST API:
PredictVariant, PredictExpression, Embed, Health, plus a streaming
RPC for batch predictions.

Usage::

    from genova.api.grpc_server import serve_grpc

    serve_grpc(port=50051, model_path="./checkpoints/best")

The implementation works without a ``.proto`` file by defining service
descriptors programmatically.  If ``grpcio`` is not installed the module
degrades gracefully -- all public symbols remain importable but calling
``serve_grpc`` will raise an informative error.
"""

from __future__ import annotations

import json
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from loguru import logger

from genova import __version__
from genova.api.inference import InferenceEngine

# ---------------------------------------------------------------------------
# Optional gRPC dependency
# ---------------------------------------------------------------------------

try:
    import grpc
    from grpc import StatusCode

    _HAS_GRPC = True
except ImportError:
    _HAS_GRPC = False
    grpc = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# JSON-based message helpers (no .proto needed)
# ---------------------------------------------------------------------------

def _encode(data: Any) -> bytes:
    """Serialise a Python object to JSON bytes."""
    return json.dumps(data, default=str).encode("utf-8")


def _decode(raw: bytes) -> Any:
    """Deserialise JSON bytes to a Python object."""
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Generic request / response wrappers
# ---------------------------------------------------------------------------

class _JsonRequest:
    """Minimal wrapper emulating a protobuf message with a JSON payload."""

    def __init__(self, raw: bytes = b"{}") -> None:
        self._data = _decode(raw) if raw else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def data(self) -> Dict[str, Any]:
        return self._data


class _JsonResponse:
    """Minimal wrapper emulating a protobuf response."""

    def __init__(self, data: Any = None) -> None:
        self._data = data or {}

    def SerializeToString(self) -> bytes:  # noqa: N802
        return _encode(self._data)


# ---------------------------------------------------------------------------
# Generic serializer/deserializer for gRPC
# ---------------------------------------------------------------------------

class _JsonSerializer:
    """Custom (de)serializer so gRPC can exchange plain JSON bytes."""

    @staticmethod
    def request_deserializer(raw: bytes) -> _JsonRequest:
        return _JsonRequest(raw)

    @staticmethod
    def response_serializer(resp: _JsonResponse) -> bytes:
        return resp.SerializeToString()

    @staticmethod
    def request_serializer(req: _JsonRequest) -> bytes:
        return _encode(req.data)

    @staticmethod
    def response_deserializer(raw: bytes) -> _JsonResponse:
        return _JsonResponse(_decode(raw))


# ---------------------------------------------------------------------------
# GenovaServicer
# ---------------------------------------------------------------------------

class GenovaServicer:
    """gRPC servicer implementing Genova inference endpoints.

    Parameters
    ----------
    engine : InferenceEngine
        A loaded inference engine.
    """

    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    # -- Health --------------------------------------------------------------

    def Health(  # noqa: N802
        self,
        request: _JsonRequest,
        context: Any,
    ) -> _JsonResponse:
        """Return server health status."""
        loaded = self.engine.is_loaded()
        device = str(self.engine.device)
        return _JsonResponse({
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "device": device,
            "version": __version__,
        })

    # -- PredictVariant ------------------------------------------------------

    def PredictVariant(  # noqa: N802
        self,
        request: _JsonRequest,
        context: Any,
    ) -> _JsonResponse:
        """Predict variant effects."""
        if not self.engine.is_loaded():
            if _HAS_GRPC:
                context.abort(StatusCode.UNAVAILABLE, "Model not loaded")
            return _JsonResponse({"error": "Model not loaded"})

        ref_sequences = request.get("ref_sequences", [])
        alt_sequences = request.get("alt_sequences", [])

        if not ref_sequences or not alt_sequences:
            if _HAS_GRPC:
                context.abort(
                    StatusCode.INVALID_ARGUMENT,
                    "ref_sequences and alt_sequences are required",
                )
            return _JsonResponse({"error": "Missing sequences"})

        start = time.monotonic()
        predictions = self.engine.predict_variant(ref_sequences, alt_sequences)
        elapsed = time.monotonic() - start

        return _JsonResponse({
            "predictions": predictions,
            "model_version": __version__,
            "num_variants": len(predictions),
            "inference_time_s": round(elapsed, 4),
        })

    # -- PredictExpression ---------------------------------------------------

    def PredictExpression(  # noqa: N802
        self,
        request: _JsonRequest,
        context: Any,
    ) -> _JsonResponse:
        """Predict gene expression levels."""
        if not self.engine.is_loaded():
            if _HAS_GRPC:
                context.abort(StatusCode.UNAVAILABLE, "Model not loaded")
            return _JsonResponse({"error": "Model not loaded"})

        sequences = request.get("sequences", [])
        num_targets = request.get("num_targets", 1)

        start = time.monotonic()
        predictions = self.engine.predict_expression(sequences, num_targets=num_targets)
        elapsed = time.monotonic() - start

        return _JsonResponse({
            "predictions": [p.tolist() for p in predictions],
            "model_version": __version__,
            "num_sequences": len(predictions),
            "inference_time_s": round(elapsed, 4),
        })

    # -- Embed ---------------------------------------------------------------

    def Embed(  # noqa: N802
        self,
        request: _JsonRequest,
        context: Any,
    ) -> _JsonResponse:
        """Extract embeddings."""
        if not self.engine.is_loaded():
            if _HAS_GRPC:
                context.abort(StatusCode.UNAVAILABLE, "Model not loaded")
            return _JsonResponse({"error": "Model not loaded"})

        sequences = request.get("sequences", [])
        pooling = request.get("pooling", "mean")

        start = time.monotonic()
        embeddings = self.engine.embed(sequences, pooling=pooling)
        elapsed = time.monotonic() - start

        return _JsonResponse({
            "embeddings": [e.tolist() for e in embeddings],
            "model_version": __version__,
            "num_sequences": len(embeddings),
            "inference_time_s": round(elapsed, 4),
        })

    # -- Streaming batch predictions -----------------------------------------

    def StreamPredict(  # noqa: N802
        self,
        request_iterator: Any,
        context: Any,
    ) -> Iterator[_JsonResponse]:
        """Streaming RPC: yield a response for each request in the stream."""
        if not self.engine.is_loaded():
            if _HAS_GRPC:
                context.abort(StatusCode.UNAVAILABLE, "Model not loaded")
            return

        for request in request_iterator:
            task = request.get("task", "embed")
            sequences = request.get("sequences", [])

            try:
                if task == "variant":
                    mid = len(sequences) // 2
                    refs = sequences[:mid]
                    alts = sequences[mid:]
                    result = self.engine.predict_variant(refs, alts)
                elif task == "expression":
                    result = [p.tolist() for p in self.engine.predict_expression(sequences)]
                elif task == "methylation":
                    result = [p.tolist() for p in self.engine.predict_methylation(sequences)]
                else:
                    result = [e.tolist() for e in self.engine.embed(sequences)]

                yield _JsonResponse({
                    "predictions": result,
                    "task": task,
                    "num_sequences": len(sequences),
                })
            except Exception as exc:
                yield _JsonResponse({
                    "error": str(exc),
                    "task": task,
                })


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve_grpc(
    port: int = 50051,
    model_path: Optional[Union[str, Path]] = None,
    engine: Optional[InferenceEngine] = None,
    device: str = "auto",
    max_workers: int = 4,
    block: bool = True,
) -> Any:
    """Start a gRPC server exposing Genova inference endpoints.

    Parameters
    ----------
    port : int
        Port number (default 50051).
    model_path : str or Path, optional
        Path to model checkpoint. Ignored if *engine* is provided.
    engine : InferenceEngine, optional
        Pre-loaded inference engine.
    device : str
        Device string (``"auto"``, ``"cuda"``, ``"cpu"``).
    max_workers : int
        Thread-pool size for gRPC server.
    block : bool
        If *True*, block the calling thread until the server stops.

    Returns
    -------
    grpc.Server or None
        The running server instance (useful when ``block=False``).

    Raises
    ------
    ImportError
        If ``grpcio`` is not installed.
    """
    if not _HAS_GRPC:
        raise ImportError(
            "grpcio is required for the gRPC server. "
            "Install it with: pip install grpcio grpcio-tools"
        )

    # Build or reuse engine
    if engine is None:
        engine = InferenceEngine(model_path=str(model_path) if model_path else None, device=device)
        engine.load()

    servicer = GenovaServicer(engine)

    # Build method handlers using the generic JSON (de)serializers
    ser = _JsonSerializer()

    method_handlers: Dict[str, Any] = {
        "Health": grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=ser.request_deserializer,
            response_serializer=ser.response_serializer,
        ),
        "PredictVariant": grpc.unary_unary_rpc_method_handler(
            servicer.PredictVariant,
            request_deserializer=ser.request_deserializer,
            response_serializer=ser.response_serializer,
        ),
        "PredictExpression": grpc.unary_unary_rpc_method_handler(
            servicer.PredictExpression,
            request_deserializer=ser.request_deserializer,
            response_serializer=ser.response_serializer,
        ),
        "Embed": grpc.unary_unary_rpc_method_handler(
            servicer.Embed,
            request_deserializer=ser.request_deserializer,
            response_serializer=ser.response_serializer,
        ),
        "StreamPredict": grpc.stream_stream_rpc_method_handler(
            servicer.StreamPredict,
            request_deserializer=ser.request_deserializer,
            response_serializer=ser.response_serializer,
        ),
    }

    generic_handler = grpc.method_service_handler(
        "genova.GenovaService",
        method_handlers,
    ) if hasattr(grpc, "method_service_handler") else None

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

    # Register via generic handler or service-name-based approach
    if generic_handler is not None:
        server.add_generic_rpc_handlers([generic_handler])
    else:
        # Fallback: create a GenericRpcHandler manually
        class _GenovaHandler(grpc.GenericRpcHandler):
            def service(self, handler_call_details):
                method = handler_call_details.method
                # Method format: /genova.GenovaService/MethodName
                parts = method.split("/")
                method_name = parts[-1] if parts else method
                return method_handlers.get(method_name)

        server.add_generic_rpc_handlers([_GenovaHandler()])

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server started on port {}", port)

    if block:
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("gRPC server shutting down...")
            server.stop(grace=5)
    return server
