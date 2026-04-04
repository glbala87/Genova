"""Integration test: API schema validation and endpoint testing.

Uses Pydantic validation on request/response schemas. Actual model
inference endpoints are not tested here since they require a loaded
model, but schema validation and health endpoints are covered.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from genova.api.schemas import (
    EmbeddingRequest,
    ExpressionRequest,
    HealthResponse,
    MethylationRequest,
    ModelInfoResponse,
    VariantInput,
    VariantPredictionRequest,
    VariantPredictionResponse,
    VariantPredictionResult,
)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSchemaValidation:
    """Test Pydantic request/response schema validation."""

    def test_variant_input_valid(self):
        """Valid VariantInput is accepted."""
        vi = VariantInput(
            sequence="ACGTACGT",
            chrom="chr1",
            pos=100,
            ref="A",
            alt="G",
        )
        assert vi.sequence == "ACGTACGT"
        assert vi.chrom == "chr1"

    def test_variant_input_invalid_sequence(self):
        """Sequence with invalid characters is rejected."""
        with pytest.raises(ValidationError):
            VariantInput(sequence="ACGTXYZ", ref="A", alt="G")

    def test_variant_input_empty_sequence_rejected(self):
        """Empty sequence string should be rejected (min_length=1)."""
        with pytest.raises(ValidationError):
            VariantInput(sequence="", ref="A", alt="G")

    def test_variant_prediction_request_valid(self):
        """Valid VariantPredictionRequest is accepted."""
        req = VariantPredictionRequest(
            variants=[
                VariantInput(sequence="ACGTACGT", ref="A", alt="G"),
            ],
            window_size=512,
        )
        assert len(req.variants) == 1
        assert req.window_size == 512

    def test_variant_prediction_request_empty_variants(self):
        """Empty variants list is rejected."""
        with pytest.raises(ValidationError):
            VariantPredictionRequest(variants=[])

    def test_variant_prediction_request_window_size_bounds(self):
        """Window size outside bounds is rejected."""
        with pytest.raises(ValidationError):
            VariantPredictionRequest(
                variants=[VariantInput(ref="A", alt="G")],
                window_size=10,  # below minimum of 64
            )

    def test_expression_request_valid(self):
        """Valid ExpressionRequest is accepted."""
        req = ExpressionRequest(
            sequences=["ACGTACGT", "TTTTAAAA"],
            num_targets=5,
        )
        assert len(req.sequences) == 2
        assert req.num_targets == 5

    def test_expression_request_invalid_chars(self):
        """Expression request with invalid DNA chars is rejected."""
        with pytest.raises(ValidationError):
            ExpressionRequest(sequences=["ACGTXYZ"])

    def test_expression_request_empty_sequence(self):
        """Empty sequence in list is rejected."""
        with pytest.raises(ValidationError):
            ExpressionRequest(sequences=[""])

    def test_methylation_request_valid(self):
        """Valid MethylationRequest is accepted."""
        req = MethylationRequest(
            sequences=["ACGTACGT"],
            num_targets=3,
        )
        assert req.num_targets == 3

    def test_methylation_request_invalid_chars(self):
        """Methylation request with invalid chars rejected."""
        with pytest.raises(ValidationError):
            MethylationRequest(sequences=["HELLO"])

    def test_embedding_request_valid(self):
        """Valid EmbeddingRequest is accepted."""
        req = EmbeddingRequest(
            sequences=["ACGTACGT"],
            pooling="mean",
        )
        assert req.pooling == "mean"

    def test_embedding_request_invalid_pooling(self):
        """Invalid pooling strategy is rejected."""
        with pytest.raises(ValidationError):
            EmbeddingRequest(
                sequences=["ACGTACGT"],
                pooling="invalid",
            )


# ---------------------------------------------------------------------------
# Response schema tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestResponseSchemas:
    """Test response model construction."""

    def test_health_response(self):
        resp = HealthResponse(
            status="ok",
            model_loaded=True,
            device="cpu",
            version="0.1.0",
        )
        assert resp.status == "ok"
        assert resp.model_loaded is True

    def test_model_info_response(self):
        resp = ModelInfoResponse(
            model_name="genova",
            version="0.1.0",
            architecture="transformer",
            num_parameters=1000000,
            d_model=256,
            n_layers=6,
            n_heads=4,
            vocab_size=4096,
            device="cpu",
            tasks=["variant_prediction", "expression"],
        )
        assert resp.architecture == "transformer"
        assert resp.num_parameters == 1000000

    def test_variant_prediction_response(self):
        resp = VariantPredictionResponse(
            predictions=[
                VariantPredictionResult(
                    variant_key="chr1:100:A>G",
                    score=0.85,
                    label="pathogenic",
                    confidence=0.9,
                ),
            ],
            model_version="0.1.0",
            num_variants=1,
        )
        assert len(resp.predictions) == 1
        assert resp.predictions[0].score == 0.85
