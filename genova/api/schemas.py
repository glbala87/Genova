"""Pydantic request/response schemas for the Genova REST API.

Defines validated data models for all API endpoints including variant
prediction, expression prediction, methylation prediction, and embedding
extraction.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VariantType(str, Enum):
    """Supported variant representation formats."""

    SNV = "snv"
    INSERTION = "insertion"
    DELETION = "deletion"
    MNV = "mnv"


# ---------------------------------------------------------------------------
# Variant prediction
# ---------------------------------------------------------------------------


class VariantInput(BaseModel):
    """A single variant specified by sequence context or VCF-style fields."""

    sequence: Optional[str] = Field(
        None,
        description="Reference DNA sequence surrounding the variant.",
        min_length=1,
    )
    chrom: Optional[str] = Field(None, description="Chromosome name.")
    pos: Optional[int] = Field(None, ge=1, description="1-based variant position.")
    ref: Optional[str] = Field(None, description="Reference allele.")
    alt: Optional[str] = Field(None, description="Alternate allele.")
    vcf_line: Optional[str] = Field(
        None,
        description="Raw VCF record line. If provided, chrom/pos/ref/alt are parsed from it.",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence_chars(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid = set("ACGTNacgtn")
            if not set(v).issubset(valid):
                raise ValueError(
                    f"Sequence contains invalid characters. Allowed: {sorted(valid)}"
                )
        return v


class VariantPredictionResult(BaseModel):
    """Prediction result for a single variant."""

    variant_key: str = Field(..., description="Variant identifier (chrom:pos:ref>alt).")
    score: float = Field(..., description="Pathogenicity score (0=benign, 1=pathogenic).")
    label: str = Field(..., description="Predicted label: 'benign' or 'pathogenic'.")
    confidence: float = Field(..., description="Prediction confidence (0-1).")
    population_context: Optional[Dict[str, Any]] = Field(
        None, description="Population frequency context if available."
    )


class VariantPredictionRequest(BaseModel):
    """Request body for /predict_variant endpoint."""

    variants: List[VariantInput] = Field(
        ..., min_length=1, description="One or more variants to predict."
    )
    vcf_content: Optional[str] = Field(
        None, description="Raw VCF file content. If provided, variants field is ignored."
    )
    reference_sequence: Optional[str] = Field(
        None, description="Reference genome sequence for context extraction."
    )
    window_size: int = Field(512, ge=64, le=4096, description="Context window size.")
    include_embeddings: bool = Field(
        False, description="Whether to include raw embeddings in response."
    )


class VariantPredictionResponse(BaseModel):
    """Response body for /predict_variant endpoint."""

    predictions: List[VariantPredictionResult]
    model_version: str
    num_variants: int


# ---------------------------------------------------------------------------
# Expression prediction
# ---------------------------------------------------------------------------


class ExpressionRequest(BaseModel):
    """Request body for /predict_expression endpoint."""

    sequences: List[str] = Field(
        ..., min_length=1, description="DNA sequences for expression prediction."
    )
    num_targets: int = Field(
        1, ge=1, description="Number of expression targets to predict."
    )

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        valid = set("ACGTNacgtn")
        for i, seq in enumerate(v):
            if not seq:
                raise ValueError(f"Sequence at index {i} is empty.")
            if not set(seq).issubset(valid):
                raise ValueError(
                    f"Sequence at index {i} contains invalid characters."
                )
        return v


class ExpressionResult(BaseModel):
    """Expression prediction for a single sequence."""

    sequence_index: int
    expression_values: List[float] = Field(
        ..., description="Predicted expression values per target."
    )


class ExpressionResponse(BaseModel):
    """Response body for /predict_expression endpoint."""

    predictions: List[ExpressionResult]
    model_version: str
    num_sequences: int


# ---------------------------------------------------------------------------
# Methylation prediction
# ---------------------------------------------------------------------------


class MethylationRequest(BaseModel):
    """Request body for /predict_methylation endpoint."""

    sequences: List[str] = Field(
        ..., min_length=1, description="DNA sequences for methylation prediction."
    )
    num_targets: int = Field(
        1, ge=1, description="Number of CpG site targets to predict."
    )

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        valid = set("ACGTNacgtn")
        for i, seq in enumerate(v):
            if not seq:
                raise ValueError(f"Sequence at index {i} is empty.")
            if not set(seq).issubset(valid):
                raise ValueError(
                    f"Sequence at index {i} contains invalid characters."
                )
        return v


class MethylationResult(BaseModel):
    """Methylation prediction for a single sequence."""

    sequence_index: int
    beta_values: List[float] = Field(
        ..., description="Predicted methylation beta values (0-1) per CpG target."
    )


class MethylationResponse(BaseModel):
    """Response body for /predict_methylation endpoint."""

    predictions: List[MethylationResult]
    model_version: str
    num_sequences: int


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    """Request body for /embed endpoint."""

    sequences: List[str] = Field(
        ..., min_length=1, description="DNA sequences to embed."
    )
    pooling: str = Field(
        "mean",
        description="Pooling strategy: 'mean', 'cls', or 'max'.",
    )

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        valid = set("ACGTNacgtn")
        for i, seq in enumerate(v):
            if not seq:
                raise ValueError(f"Sequence at index {i} is empty.")
            if not set(seq).issubset(valid):
                raise ValueError(
                    f"Sequence at index {i} contains invalid characters."
                )
        return v

    @field_validator("pooling")
    @classmethod
    def validate_pooling(cls, v: str) -> str:
        allowed = {"mean", "cls", "max"}
        if v not in allowed:
            raise ValueError(f"Pooling must be one of {allowed}, got {v!r}.")
        return v


class EmbeddingResult(BaseModel):
    """Embedding for a single sequence."""

    sequence_index: int
    embedding: List[float] = Field(..., description="Embedding vector.")
    dimension: int


class EmbeddingResponse(BaseModel):
    """Response body for /embed endpoint."""

    embeddings: List[EmbeddingResult]
    model_version: str
    num_sequences: int


# ---------------------------------------------------------------------------
# Health & model info
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str = "ok"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = ""


class ModelInfoResponse(BaseModel):
    """Response body for /model/info endpoint."""

    model_name: str = "genova"
    version: str
    architecture: str
    num_parameters: int
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int
    device: str
    tasks: List[str] = Field(default_factory=list)
