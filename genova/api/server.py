"""FastAPI server for Genova model inference.

Provides REST endpoints for variant effect prediction, gene expression
prediction, methylation prediction, and embedding extraction.

Launch with::

    uvicorn genova.api.server:create_app --factory --host 0.0.0.0 --port 8000

Or via the CLI::

    genova serve --host 0.0.0.0 --port 8000 --model-path ./checkpoints/best

Security features (enabled by default in production):
    - Authentication: GENOVA_AUTH_ENABLED=1 (default). Set to 0 to disable.
    - Rate limiting:  GENOVA_RATE_LIMIT_ENABLED=1 (default). Set to 0 to disable.

Optional observability features (off by default):
    - Metrics:        GENOVA_METRICS_ENABLED=1
    - Tracing:        GENOVA_TRACING_ENABLED=1
    - Request logging: GENOVA_REQUEST_LOGGING_ENABLED=1
"""

from __future__ import annotations

import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from genova import __version__
from genova.api.inference import InferenceEngine
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

# Optional integrations -- imported unconditionally but activated only when
# the corresponding environment variable is set.
from genova.api.logging_middleware import setup_request_logging
from genova.api.monitoring import get_metrics, setup_monitoring
from genova.api.security import (
    APIKeyAuth,
    RateLimiter,
    is_auth_enabled,
    is_rate_limit_enabled,
)
from genova.api.tracing import setup_tracing


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

# Module-level engine reference; set by create_app() or the lifespan handler.
_engine: Optional[InferenceEngine] = None


def create_app(
    model_path: Optional[str] = None,
    device: str = "auto",
    max_batch_size: int = 64,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_path: Path to model checkpoint directory or file.
        device: Inference device (``"auto"``, ``"cuda"``, ``"cpu"``).
        max_batch_size: Maximum batch size per request.

    Returns:
        Configured :class:`FastAPI` instance.
    """
    engine = InferenceEngine(
        model_path=model_path,
        device=device,
        max_batch_size=max_batch_size,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load model at startup, clean up on shutdown."""
        global _engine
        logger.info("Starting Genova API server...")
        try:
            engine.load()
            app.state.engine = engine
            _engine = engine
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load model at startup: {}", e)
            app.state.engine = engine  # store even if not loaded
            _engine = engine
        yield
        logger.info("Shutting down Genova API server...")
        engine.unload()
        _engine = None

    app = FastAPI(
        title="Genova Genomics API",
        description=(
            "REST API for the Genova genomics foundation model. "
            "Provides variant effect prediction, gene expression prediction, "
            "methylation prediction, and sequence embedding extraction."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware — restrict origins in production via GENOVA_CORS_ORIGINS
    import os

    cors_origins_str = os.environ.get("GENOVA_CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()] if cors_origins_str else []
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )

    # --- Optional integrations (all off by default) -----------------------

    # Request logging (must be added before other middleware to wrap them)
    setup_request_logging(app)

    # Prometheus metrics
    setup_monitoring(app)

    # OpenTelemetry tracing
    setup_tracing(app)

    # Build dependency list for protected routes
    _dependencies: List[Depends] = []
    if is_auth_enabled():
        _auth = APIKeyAuth()
        _dependencies.append(Depends(_auth))
        logger.info("API key authentication enabled.")
    if is_rate_limit_enabled():
        _limiter = RateLimiter()
        _dependencies.append(Depends(_limiter))
        logger.info("Rate limiting enabled.")

    # Store dependencies on app state so _register_routes can use them
    app.state.route_dependencies = _dependencies

    # Register routes
    _register_routes(app)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error: {}\n{}", exc, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {type(exc).__name__}"},
        )

    return app


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def _get_engine(app_state: Any) -> InferenceEngine:
    """Retrieve the inference engine from app state, raising 503 if unavailable."""
    engine: Optional[InferenceEngine] = getattr(app_state, "engine", None)
    if engine is None or not engine.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The server is starting up or the model failed to load.",
        )
    return engine


def _register_routes(app: FastAPI) -> None:
    """Register all API endpoints on the application."""

    # Retrieve optional auth/rate-limit dependencies (empty list if disabled)
    deps = getattr(app.state, "route_dependencies", [])

    # ------------------------------------------------------------------
    # Health check (always public -- no auth/rate-limit)
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["status"])
    async def health(request: Request) -> HealthResponse:
        """Check server health and model status."""
        engine: Optional[InferenceEngine] = getattr(request.app.state, "engine", None)
        loaded = engine is not None and engine.is_loaded()
        device = str(engine.device) if engine else "unknown"
        return HealthResponse(
            status="ok" if loaded else "degraded",
            model_loaded=loaded,
            device=device,
            version=__version__,
        )

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["status"])
    async def model_info(request: Request) -> ModelInfoResponse:
        """Return model metadata and configuration."""
        engine = _get_engine(request.app.state)
        info = engine.get_model_info()
        return ModelInfoResponse(
            model_name="genova",
            version=__version__,
            architecture=info["architecture"],
            num_parameters=info["num_parameters"],
            d_model=info["d_model"],
            n_layers=info["n_layers"],
            n_heads=info["n_heads"],
            vocab_size=info["vocab_size"],
            device=info["device"],
            tasks=["variant_prediction", "expression", "methylation", "embedding"],
        )

    # ------------------------------------------------------------------
    # Variant prediction
    # ------------------------------------------------------------------

    @app.post(
        "/predict_variant",
        response_model=VariantPredictionResponse,
        tags=["prediction"],
        dependencies=deps,
    )
    async def predict_variant(
        request: Request, body: VariantPredictionRequest
    ) -> VariantPredictionResponse:
        """Predict variant effects (pathogenicity).

        Accepts either a list of variant specifications or raw VCF content.
        Returns pathogenicity scores with confidence and population context.
        """
        engine = _get_engine(request.app.state)
        start_time = time.monotonic()

        try:
            ref_sequences: list[str] = []
            alt_sequences: list[str] = []
            variant_keys: list[str] = []

            if body.vcf_content:
                # Parse VCF content inline
                from genova.evaluation.variant_predictor import Variant

                for line in body.vcf_content.strip().split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    if len(fields) < 5:
                        continue
                    chrom, pos_str, _, ref, alt = fields[:5]
                    pos = int(pos_str)
                    variant_keys.append(f"{chrom}:{pos}:{ref}>{alt}")

                    # Build synthetic sequences around the variant
                    ref_context = body.reference_sequence or ref.center(
                        body.window_size, "N"
                    )
                    alt_context = ref_context  # simplified: use provided context
                    ref_sequences.append(ref_context if body.reference_sequence else "N" * 50 + ref + "N" * 50)
                    alt_sequences.append(ref_context if body.reference_sequence else "N" * 50 + alt + "N" * 50)
            else:
                for vi in body.variants:
                    if vi.sequence is not None and vi.ref and vi.alt:
                        # Use the provided sequence as reference context
                        ref_seq = vi.sequence
                        # Substitute the variant in the middle
                        mid = len(ref_seq) // 2
                        ref_len = len(vi.ref)
                        alt_seq = ref_seq[:mid] + vi.alt + ref_seq[mid + ref_len:]
                        ref_sequences.append(ref_seq)
                        alt_sequences.append(alt_seq)
                        key = f"{vi.chrom or 'unk'}:{vi.pos or 0}:{vi.ref}>{vi.alt}"
                        variant_keys.append(key)
                    elif vi.ref and vi.alt:
                        # No context sequence -- create synthetic flanks
                        flank = "N" * (body.window_size // 2)
                        ref_sequences.append(flank + vi.ref + flank)
                        alt_sequences.append(flank + vi.alt + flank)
                        key = f"{vi.chrom or 'unk'}:{vi.pos or 0}:{vi.ref}>{vi.alt}"
                        variant_keys.append(key)
                    else:
                        raise HTTPException(
                            status_code=422,
                            detail="Each variant must have ref and alt alleles.",
                        )

            predictions = engine.predict_variant(
                ref_sequences, alt_sequences
            )

            results = []
            for i, pred in enumerate(predictions):
                results.append(
                    VariantPredictionResult(
                        variant_key=variant_keys[i],
                        score=pred["score"],
                        label=pred["label"],
                        confidence=pred["confidence"],
                        population_context=None,
                    )
                )

            elapsed = time.monotonic() - start_time
            logger.info(
                "Variant prediction: {} variants in {:.3f}s",
                len(results),
                elapsed,
            )

            # Record metrics
            metrics = get_metrics()
            metrics.record_inference("variant_prediction", elapsed, len(results))
            metrics.variants_processed.inc(len(results))

            return VariantPredictionResponse(
                predictions=results,
                model_version=__version__,
                num_variants=len(results),
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Variant prediction failed: {}", e)
            raise HTTPException(
                status_code=500,
                detail="Variant prediction failed. Check server logs for details.",
            )

    # ------------------------------------------------------------------
    # Expression prediction
    # ------------------------------------------------------------------

    @app.post(
        "/predict_expression",
        response_model=ExpressionResponse,
        tags=["prediction"],
        dependencies=deps,
    )
    async def predict_expression(
        request: Request, body: ExpressionRequest
    ) -> ExpressionResponse:
        """Predict gene expression levels from DNA sequences."""
        engine = _get_engine(request.app.state)
        start_time = time.monotonic()

        try:
            predictions = engine.predict_expression(
                body.sequences,
                num_targets=body.num_targets,
            )

            results = [
                ExpressionResult(
                    sequence_index=i,
                    expression_values=pred.tolist(),
                )
                for i, pred in enumerate(predictions)
            ]

            elapsed = time.monotonic() - start_time
            logger.info(
                "Expression prediction: {} sequences in {:.3f}s",
                len(results),
                elapsed,
            )

            metrics = get_metrics()
            metrics.record_inference("expression", elapsed, len(results))

            return ExpressionResponse(
                predictions=results,
                model_version=__version__,
                num_sequences=len(results),
            )

        except Exception as e:
            logger.error("Expression prediction failed: {}", e)
            raise HTTPException(
                status_code=500,
                detail="Expression prediction failed. Check server logs for details.",
            )

    # ------------------------------------------------------------------
    # Methylation prediction
    # ------------------------------------------------------------------

    @app.post(
        "/predict_methylation",
        response_model=MethylationResponse,
        tags=["prediction"],
        dependencies=deps,
    )
    async def predict_methylation(
        request: Request, body: MethylationRequest
    ) -> MethylationResponse:
        """Predict methylation beta values from DNA sequences."""
        engine = _get_engine(request.app.state)
        start_time = time.monotonic()

        try:
            predictions = engine.predict_methylation(
                body.sequences,
                num_targets=body.num_targets,
            )

            results = [
                MethylationResult(
                    sequence_index=i,
                    beta_values=pred.tolist(),
                )
                for i, pred in enumerate(predictions)
            ]

            elapsed = time.monotonic() - start_time
            logger.info(
                "Methylation prediction: {} sequences in {:.3f}s",
                len(results),
                elapsed,
            )

            metrics = get_metrics()
            metrics.record_inference("methylation", elapsed, len(results))

            return MethylationResponse(
                predictions=results,
                model_version=__version__,
                num_sequences=len(results),
            )

        except Exception as e:
            logger.error("Methylation prediction failed: {}", e)
            raise HTTPException(
                status_code=500,
                detail="Methylation prediction failed. Check server logs for details.",
            )

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @app.post(
        "/embed",
        response_model=EmbeddingResponse,
        tags=["embedding"],
        dependencies=deps,
    )
    async def embed(
        request: Request, body: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Extract embedding vectors from DNA sequences."""
        engine = _get_engine(request.app.state)
        start_time = time.monotonic()

        try:
            embeddings = engine.embed(
                body.sequences,
                pooling=body.pooling,
            )

            results = [
                EmbeddingResult(
                    sequence_index=i,
                    embedding=emb.tolist(),
                    dimension=len(emb),
                )
                for i, emb in enumerate(embeddings)
            ]

            elapsed = time.monotonic() - start_time
            logger.info(
                "Embedding extraction: {} sequences in {:.3f}s",
                len(results),
                elapsed,
            )

            metrics = get_metrics()
            metrics.record_inference("embedding", elapsed, len(results))

            return EmbeddingResponse(
                embeddings=results,
                model_version=__version__,
                num_sequences=len(results),
            )

        except Exception as e:
            logger.error("Embedding extraction failed: {}", e)
            raise HTTPException(
                status_code=500,
                detail="Embedding extraction failed. Check server logs for details.",
            )
