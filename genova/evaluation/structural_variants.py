"""Structural variant and copy number variation prediction for Genova.

Supports detection and classification of large structural variants (SVs)
including deletions (DEL), duplications (DUP), inversions (INV), and
breakends (BND).  Also estimates copy number from read-depth-like
embedding features.

Parses SV-aware VCF records with SVTYPE, SVLEN, and END INFO fields.

Example::

    from genova.evaluation.structural_variants import StructuralVariantPredictor

    predictor = StructuralVariantPredictor(model, tokenizer, device="cuda")
    sv_result = predictor.predict_sv(sequence, sv_type="DEL", sv_start=100, sv_end=500)
    cnv = predictor.predict_cnv(sequence, window_size=200)
    records = predictor.parse_sv_vcf("structural_variants.vcf")
"""

from __future__ import annotations

import gzip
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

SV_TYPES = ("DEL", "DUP", "INV", "BND")


@dataclass
class StructuralVariant:
    """A structural variant record.

    Attributes:
        chrom: Chromosome name.
        pos: 1-based start position.
        sv_type: Structural variant type (DEL, DUP, INV, BND).
        sv_end: 1-based end position of the SV.
        sv_len: Length of the SV (negative for deletions).
        variant_id: Variant identifier.
        qual: Quality score string.
        filter_field: FILTER field from VCF.
        info: Raw INFO field string.
        alt: ALT allele string.
        ref: REF allele string.
    """

    chrom: str
    pos: int
    sv_type: str
    sv_end: int = 0
    sv_len: int = 0
    variant_id: str = "."
    qual: str = "."
    filter_field: str = "."
    info: str = "."
    alt: str = "."
    ref: str = "."

    @property
    def key(self) -> str:
        """Unique string key for this SV."""
        return f"{self.chrom}:{self.pos}-{self.sv_end}:{self.sv_type}"

    @property
    def size(self) -> int:
        """Absolute size of the structural variant."""
        if self.sv_len != 0:
            return abs(self.sv_len)
        return max(0, self.sv_end - self.pos)


@dataclass
class SVPrediction:
    """Prediction result for a structural variant.

    Attributes:
        sv: The structural variant record.
        confidence: Prediction confidence score in [0, 1].
        predicted_type: Predicted SV type.
        breakpoint_scores: Per-position breakpoint likelihood scores.
        embedding_discontinuity: Discontinuity score at SV boundaries.
    """

    sv: StructuralVariant
    confidence: float
    predicted_type: str
    breakpoint_scores: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_discontinuity: float = 0.0


@dataclass
class CNVPrediction:
    """Copy number prediction for a genomic window.

    Attributes:
        start: 0-based start position.
        end: 0-based end position.
        copy_number: Estimated copy number (2 = normal diploid).
        confidence: Prediction confidence.
        log2_ratio: log2 ratio relative to expected baseline.
    """

    start: int
    end: int
    copy_number: float
    confidence: float
    log2_ratio: float


# ---------------------------------------------------------------------------
# SV-aware VCF parser
# ---------------------------------------------------------------------------


def _parse_info_field(info: str) -> Dict[str, str]:
    """Parse a VCF INFO field into a key-value dictionary.

    Args:
        info: Semicolon-delimited INFO string.

    Returns:
        Dict of key-value pairs. Flag fields have value ``"True"``.
    """
    result: Dict[str, str] = {}
    if info in (".", ""):
        return result
    for item in info.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        else:
            result[item] = "True"
    return result


def parse_sv_vcf(vcf_path: Union[str, Path]) -> List[StructuralVariant]:
    """Parse a structural-variant-aware VCF file.

    Extracts SVTYPE, SVLEN, and END fields from the INFO column to
    construct :class:`StructuralVariant` records.

    Args:
        vcf_path: Path to the VCF file (plain or gzip-compressed).

    Returns:
        List of :class:`StructuralVariant` records.
    """
    vcf_path = Path(vcf_path)
    opener = gzip.open if vcf_path.suffix == ".gz" else open
    mode = "rt" if vcf_path.suffix == ".gz" else "r"

    variants: List[StructuralVariant] = []

    with opener(vcf_path, mode) as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 8:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            variant_id = fields[2] if len(fields) > 2 else "."
            ref = fields[3].upper()
            alt = fields[4].upper()
            qual = fields[5] if len(fields) > 5 else "."
            filt = fields[6] if len(fields) > 6 else "."
            info_str = fields[7] if len(fields) > 7 else "."

            info = _parse_info_field(info_str)

            sv_type = info.get("SVTYPE", "")
            if sv_type not in SV_TYPES:
                # Try to infer from ALT field
                for st in SV_TYPES:
                    if f"<{st}>" in alt:
                        sv_type = st
                        break
                if sv_type not in SV_TYPES:
                    # Check if this is a large indel (>50bp)
                    ref_len = len(ref)
                    alt_len = len(alt)
                    if abs(ref_len - alt_len) > 50:
                        sv_type = "DEL" if ref_len > alt_len else "DUP"
                    else:
                        continue  # Skip non-SV records

            # Parse END
            sv_end = int(info.get("END", str(pos)))

            # Parse SVLEN
            sv_len_str = info.get("SVLEN", "0")
            try:
                sv_len = int(sv_len_str.split(",")[0])
            except ValueError:
                sv_len = 0

            # If END not set, compute from SVLEN
            if sv_end == pos and sv_len != 0:
                sv_end = pos + abs(sv_len)

            variants.append(
                StructuralVariant(
                    chrom=chrom,
                    pos=pos,
                    sv_type=sv_type,
                    sv_end=sv_end,
                    sv_len=sv_len,
                    variant_id=variant_id,
                    qual=qual,
                    filter_field=filt,
                    info=info_str,
                    alt=alt,
                    ref=ref,
                )
            )

    logger.info("Parsed {} structural variants from {}", len(variants), vcf_path)
    return variants


# ---------------------------------------------------------------------------
# StructuralVariantPredictor
# ---------------------------------------------------------------------------


class StructuralVariantPredictor:
    """Structural variant and copy number prediction using Genova embeddings.

    Detects structural variants from embedding discontinuities and classifies
    SV types using learned embedding features.  Also estimates copy number
    from window-level embedding norms.

    Args:
        model: Pretrained Genova encoder model.
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        window_size: Default window size for embedding extraction.
        sv_threshold: Confidence threshold for SV calls.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        window_size: int = 512,
        sv_threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.window_size = window_size
        self.sv_threshold = sv_threshold

        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_embeddings(self, sequence: str) -> np.ndarray:
        """Extract per-position embeddings for a DNA sequence.

        Args:
            sequence: DNA sequence string.

        Returns:
            2-D array of shape ``(L, D)`` with token-level embeddings.
        """
        token_ids = self.tokenizer.encode(sequence)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
        elif isinstance(outputs, Tensor):
            hidden = outputs
        else:
            hidden = getattr(outputs, "last_hidden_state", outputs)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        # Return per-position embeddings: (L, D)
        return hidden.squeeze(0).cpu().float().numpy()

    @torch.no_grad()
    def _get_pooled_embedding(self, sequence: str) -> np.ndarray:
        """Extract mean-pooled embedding for a DNA sequence.

        Args:
            sequence: DNA sequence string.

        Returns:
            1-D array of shape ``(D,)``.
        """
        embeddings = self._get_embeddings(sequence)
        return embeddings.mean(axis=0)

    # ------------------------------------------------------------------
    # Breakpoint detection
    # ------------------------------------------------------------------

    def _detect_breakpoints(
        self,
        embeddings: np.ndarray,
        sv_start: int,
        sv_end: int,
    ) -> Tuple[np.ndarray, float]:
        """Detect embedding discontinuities near SV breakpoints.

        Computes the cosine distance between consecutive embedding vectors
        and identifies peaks near the specified SV boundaries.

        Args:
            embeddings: ``(L, D)`` per-position embeddings.
            sv_start: Expected start of the SV (0-based token position).
            sv_end: Expected end of the SV (0-based token position).

        Returns:
            Tuple of:
                - ``breakpoint_scores``: per-position discontinuity scores ``(L,)``.
                - ``discontinuity``: aggregate discontinuity score at the
                  SV boundaries.
        """
        L = embeddings.shape[0]
        if L < 2:
            return np.zeros(L), 0.0

        # Cosine distance between consecutive positions
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normed = embeddings / norms

        distances = np.zeros(L)
        for i in range(1, L):
            cos_sim = np.dot(normed[i], normed[i - 1])
            distances[i] = 1.0 - cos_sim

        # Score near the breakpoints
        window = max(3, L // 50)
        start_idx = max(0, min(sv_start, L - 1))
        end_idx = max(0, min(sv_end, L - 1))

        start_region = distances[
            max(0, start_idx - window): min(L, start_idx + window + 1)
        ]
        end_region = distances[
            max(0, end_idx - window): min(L, end_idx + window + 1)
        ]

        start_disc = float(start_region.max()) if len(start_region) > 0 else 0.0
        end_disc = float(end_region.max()) if len(end_region) > 0 else 0.0

        discontinuity = (start_disc + end_disc) / 2.0

        return distances, discontinuity

    # ------------------------------------------------------------------
    # SV type classification
    # ------------------------------------------------------------------

    def _classify_sv_type(
        self,
        ref_embeddings: np.ndarray,
        sv_start: int,
        sv_end: int,
    ) -> Tuple[str, float]:
        """Classify SV type from embedding features.

        Uses heuristic features derived from the embedding patterns at and
        around the SV region:
        - DEL: embedding norm drops in the SV region
        - DUP: embedding norm increases / repeating pattern
        - INV: reversed correlation pattern
        - BND: sharp discontinuity at one point

        Args:
            ref_embeddings: ``(L, D)`` per-position embeddings.
            sv_start: 0-based start position.
            sv_end: 0-based end position.

        Returns:
            Tuple of ``(predicted_type, confidence)``.
        """
        L = ref_embeddings.shape[0]
        sv_start = max(0, min(sv_start, L - 1))
        sv_end = max(sv_start + 1, min(sv_end, L))

        # Compute features
        norms = np.linalg.norm(ref_embeddings, axis=-1)
        global_mean_norm = norms.mean()

        sv_region_norms = norms[sv_start:sv_end]
        sv_mean_norm = sv_region_norms.mean() if len(sv_region_norms) > 0 else 0.0

        # Norm ratio: low = deletion-like, high = duplication-like
        norm_ratio = sv_mean_norm / max(global_mean_norm, 1e-8)

        # Breakpoint sharpness
        _, discontinuity = self._detect_breakpoints(
            ref_embeddings, sv_start, sv_end
        )

        # Correlation pattern within SV region
        sv_emb = ref_embeddings[sv_start:sv_end]
        if sv_emb.shape[0] >= 4:
            half = sv_emb.shape[0] // 2
            first_half = sv_emb[:half].mean(axis=0)
            second_half = sv_emb[half:].mean(axis=0)
            correlation = float(
                np.dot(first_half, second_half)
                / (np.linalg.norm(first_half) * np.linalg.norm(second_half) + 1e-8)
            )
        else:
            correlation = 0.0

        # Heuristic classification
        scores: Dict[str, float] = {
            "DEL": 0.0,
            "DUP": 0.0,
            "INV": 0.0,
            "BND": 0.0,
        }

        # DEL: norm drops
        if norm_ratio < 0.8:
            scores["DEL"] = min(1.0, (0.8 - norm_ratio) / 0.3 + 0.5)
        elif norm_ratio < 1.0:
            scores["DEL"] = 0.3

        # DUP: norm increases or high self-correlation
        if norm_ratio > 1.2:
            scores["DUP"] = min(1.0, (norm_ratio - 1.2) / 0.3 + 0.5)
        if correlation > 0.9:
            scores["DUP"] = max(scores["DUP"], 0.6)

        # INV: negative or low correlation within region
        if correlation < 0.0:
            scores["INV"] = min(1.0, abs(correlation) + 0.4)
        elif correlation < 0.3:
            scores["INV"] = 0.3

        # BND: very high discontinuity at a single point
        if discontinuity > 0.5:
            scores["BND"] = min(1.0, discontinuity)

        # Select best
        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_type]

        return best_type, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_sv(
        self,
        sequence: str,
        sv_type: str,
        sv_start: int,
        sv_end: int,
    ) -> SVPrediction:
        """Predict a structural variant from sequence context.

        Extracts embeddings and analyzes discontinuities at the specified
        SV boundaries to assess confidence and classify the SV type.

        Args:
            sequence: DNA sequence encompassing the SV region.
            sv_type: Expected SV type (DEL, DUP, INV, BND).
            sv_start: 0-based start position of the SV in the sequence.
            sv_end: 0-based end position of the SV in the sequence.

        Returns:
            :class:`SVPrediction` with confidence and classification.
        """
        if sv_type not in SV_TYPES:
            raise ValueError(
                f"sv_type must be one of {SV_TYPES}, got {sv_type!r}"
            )

        embeddings = self._get_embeddings(sequence)
        breakpoint_scores, discontinuity = self._detect_breakpoints(
            embeddings, sv_start, sv_end
        )
        predicted_type, confidence = self._classify_sv_type(
            embeddings, sv_start, sv_end
        )

        sv = StructuralVariant(
            chrom="unknown",
            pos=sv_start + 1,  # convert to 1-based
            sv_type=sv_type,
            sv_end=sv_end + 1,
            sv_len=sv_end - sv_start if sv_type == "DEL" else (sv_end - sv_start),
        )

        return SVPrediction(
            sv=sv,
            confidence=confidence,
            predicted_type=predicted_type,
            breakpoint_scores=breakpoint_scores,
            embedding_discontinuity=discontinuity,
        )

    def predict_cnv(
        self,
        sequence: str,
        window_size: Optional[int] = None,
    ) -> List[CNVPrediction]:
        """Estimate copy number across a sequence using windowed embeddings.

        Divides the sequence into windows and computes per-window embedding
        norms.  Compares each window's norm to the global median to estimate
        relative copy number (2 = diploid baseline).

        Args:
            sequence: DNA sequence to analyze.
            window_size: Size of each analysis window in characters.
                Defaults to ``self.window_size``.

        Returns:
            List of :class:`CNVPrediction` objects, one per window.
        """
        window_size = window_size or self.window_size
        seq_len = len(sequence)

        if seq_len < window_size:
            window_size = seq_len

        predictions: List[CNVPrediction] = []
        window_norms: List[float] = []
        window_positions: List[Tuple[int, int]] = []

        # Compute per-window embedding norms
        for start in range(0, seq_len, window_size):
            end = min(start + window_size, seq_len)
            window_seq = sequence[start:end]
            if len(window_seq) < 10:
                continue

            embedding = self._get_pooled_embedding(window_seq)
            norm = float(np.linalg.norm(embedding))
            window_norms.append(norm)
            window_positions.append((start, end))

        if not window_norms:
            return predictions

        norms_array = np.array(window_norms)
        median_norm = np.median(norms_array)
        std_norm = np.std(norms_array) + 1e-8

        # Estimate copy number relative to median
        for i, (start, end) in enumerate(window_positions):
            norm = norms_array[i]
            # log2 ratio: deviation from median
            log2_ratio = float(np.log2(max(norm, 1e-8) / max(median_norm, 1e-8)))

            # Estimate CN (baseline = 2 for diploid)
            copy_number = 2.0 * (2.0 ** log2_ratio)
            copy_number = max(0.0, copy_number)  # floor at 0

            # Confidence from how close the norm is to the median
            z_score = abs(norm - median_norm) / std_norm
            # Higher z-score = more confident the CN differs from 2
            confidence = float(min(1.0, 1.0 - np.exp(-0.5 * z_score)))

            predictions.append(
                CNVPrediction(
                    start=start,
                    end=end,
                    copy_number=round(copy_number, 2),
                    confidence=confidence,
                    log2_ratio=round(log2_ratio, 4),
                )
            )

        return predictions

    def parse_sv_vcf(self, vcf_path: Union[str, Path]) -> List[StructuralVariant]:
        """Parse a structural variant VCF file.

        Convenience wrapper around the module-level :func:`parse_sv_vcf`.

        Args:
            vcf_path: Path to the SV-aware VCF file.

        Returns:
            List of :class:`StructuralVariant` records.
        """
        return parse_sv_vcf(vcf_path)

    def batch_predict(
        self,
        sv_records: List[StructuralVariant],
        reference_sequences: Optional[Dict[str, str]] = None,
        context_size: int = 1000,
    ) -> List[SVPrediction]:
        """Batch predict structural variants with confidence scores.

        For each SV record, extracts a context window from the reference
        sequence (if provided) or generates a synthetic context, then
        runs the SV predictor.

        Args:
            sv_records: List of :class:`StructuralVariant` records.
            reference_sequences: Optional dict mapping chromosome names to
                full chromosome sequences.  If ``None``, synthetic context
                is used.
            context_size: Number of bases of context around each SV.

        Returns:
            List of :class:`SVPrediction` objects.
        """
        predictions: List[SVPrediction] = []

        for i, sv in enumerate(sv_records):
            try:
                if reference_sequences and sv.chrom in reference_sequences:
                    chrom_seq = reference_sequences[sv.chrom]
                    # Extract context around the SV
                    start = max(0, sv.pos - 1 - context_size // 2)
                    end = min(
                        len(chrom_seq),
                        sv.sv_end + context_size // 2,
                    )
                    context_seq = chrom_seq[start:end]
                    sv_start_in_context = sv.pos - 1 - start
                    sv_end_in_context = sv.sv_end - start
                else:
                    # Generate synthetic context for testing
                    rng = np.random.RandomState(42 + i)
                    context_len = min(context_size, sv.size + context_size)
                    context_seq = "".join(
                        rng.choice(list("ACGT"), size=context_len)
                    )
                    sv_start_in_context = context_size // 4
                    sv_end_in_context = sv_start_in_context + sv.size

                pred = self.predict_sv(
                    sequence=context_seq,
                    sv_type=sv.sv_type,
                    sv_start=sv_start_in_context,
                    sv_end=sv_end_in_context,
                )
                # Update the SV reference in the prediction
                pred.sv = sv
                predictions.append(pred)

            except Exception as e:
                logger.warning(
                    "Failed to predict SV {} ({}): {}",
                    sv.key,
                    sv.sv_type,
                    e,
                )
                predictions.append(
                    SVPrediction(
                        sv=sv,
                        confidence=0.0,
                        predicted_type="UNKNOWN",
                        embedding_discontinuity=0.0,
                    )
                )

            if (i + 1) % 50 == 0:
                logger.info(
                    "Processed {}/{} structural variants.",
                    i + 1,
                    len(sv_records),
                )

        logger.info(
            "Batch prediction complete: {} SVs processed.", len(predictions)
        )
        return predictions
