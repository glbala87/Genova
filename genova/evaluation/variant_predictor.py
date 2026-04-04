"""Variant effect prediction for Genova.

Parses VCF files, extracts reference and alternate allele sequence
windows, computes embeddings via a pretrained Genova model, and
classifies variants as benign or pathogenic using the embedding
difference as features.

Example::

    from genova.evaluation.variant_predictor import VariantEffectPredictor

    predictor = VariantEffectPredictor(model, tokenizer, device="cuda")
    results = predictor.predict_vcf("variants.vcf", "reference.fa")
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
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer
from genova.utils.config import EvaluationConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """A single genomic variant parsed from a VCF record."""

    chrom: str
    pos: int  # 1-based, as in VCF
    ref: str
    alt: str
    variant_id: str = "."
    qual: str = "."
    filter_field: str = "."
    info: str = "."

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"


@dataclass
class VariantPrediction:
    """Prediction result for a single variant."""

    variant: Variant
    score: float  # pathogenicity score (0 = benign, 1 = pathogenic)
    label: str  # "benign" or "pathogenic"
    ref_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    alt_embedding: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# VCF parser (lightweight, no pysam dependency required)
# ---------------------------------------------------------------------------


def parse_vcf(vcf_path: Union[str, Path]) -> Iterator[Variant]:
    """Parse a VCF file and yield :class:`Variant` objects.

    Supports both plain-text and gzip-compressed VCF files.  Header
    lines (starting with ``#``) are skipped.

    Args:
        vcf_path: Path to the VCF file.

    Yields:
        :class:`Variant` instances, one per VCF record.  Multi-allelic
        ALT fields are split into separate variants.
    """
    vcf_path = Path(vcf_path)
    opener = gzip.open if vcf_path.suffix == ".gz" else open
    mode = "rt" if vcf_path.suffix == ".gz" else "r"

    with opener(vcf_path, mode) as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 5:
                continue

            chrom = fields[0]
            pos = int(fields[1])
            variant_id = fields[2] if len(fields) > 2 else "."
            ref = fields[3].upper()
            alt_field = fields[4].upper()
            qual = fields[5] if len(fields) > 5 else "."
            filt = fields[6] if len(fields) > 6 else "."
            info = fields[7] if len(fields) > 7 else "."

            # Split multi-allelic ALTs
            for alt in alt_field.split(","):
                alt = alt.strip()
                if alt in (".", "*"):
                    continue
                yield Variant(
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    variant_id=variant_id,
                    qual=qual,
                    filter_field=filt,
                    info=info,
                )


# ---------------------------------------------------------------------------
# Simple FASTA reader
# ---------------------------------------------------------------------------


class FastaReader:
    """Indexed access to a FASTA reference genome.

    Loads sequences lazily per chromosome on first access and caches them
    in memory.  Suitable for moderate-size genomes; for very large
    references consider using pysam.FastaFile instead.

    Args:
        fasta_path: Path to a ``.fa`` or ``.fasta`` file (optionally
            gzip-compressed).
    """

    def __init__(self, fasta_path: Union[str, Path]) -> None:
        self.fasta_path = Path(fasta_path)
        self._sequences: Dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        logger.info("Loading reference FASTA from {}", self.fasta_path)
        opener = gzip.open if self.fasta_path.suffix == ".gz" else open
        mode = "rt" if self.fasta_path.suffix == ".gz" else "r"

        current_chrom: Optional[str] = None
        parts: List[str] = []

        with opener(self.fasta_path, mode) as fh:  # type: ignore[arg-type]
            for line in fh:
                line = line.strip()
                if line.startswith(">"):
                    if current_chrom is not None:
                        self._sequences[current_chrom] = "".join(parts).upper()
                    current_chrom = line[1:].split()[0]
                    parts = []
                else:
                    parts.append(line)
            if current_chrom is not None:
                self._sequences[current_chrom] = "".join(parts).upper()

        self._loaded = True
        logger.info("Loaded {} chromosomes from reference.", len(self._sequences))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Fetch a subsequence from the reference.

        Args:
            chrom: Chromosome name (must match FASTA headers).
            start: 0-based start position (inclusive).
            end: 0-based end position (exclusive).

        Returns:
            The reference sequence string.

        Raises:
            KeyError: If *chrom* is not found in the FASTA.
        """
        self._load()
        if chrom not in self._sequences:
            raise KeyError(
                f"Chromosome {chrom!r} not found in reference. "
                f"Available: {list(self._sequences.keys())[:10]}..."
            )
        seq = self._sequences[chrom]
        start = max(0, start)
        end = min(len(seq), end)
        return seq[start:end]

    @property
    def chromosomes(self) -> List[str]:
        """Return list of available chromosome names."""
        self._load()
        return list(self._sequences.keys())


# ---------------------------------------------------------------------------
# Pathogenicity classifier head
# ---------------------------------------------------------------------------


class VariantClassifierHead(nn.Module):
    """Binary classifier that predicts pathogenicity from embedding differences.

    Takes the element-wise difference between reference and alternate
    embeddings and passes it through a small MLP.

    Args:
        d_model: Dimensionality of the input embeddings.
        hidden_dim: Hidden layer size. Defaults to ``d_model // 2``.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or max(64, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            diff: Embedding difference tensor, shape ``(B, d_model)``.

        Returns:
            Logits tensor, shape ``(B, 1)``.
        """
        return self.net(diff)


# ---------------------------------------------------------------------------
# Variant Effect Predictor
# ---------------------------------------------------------------------------


class VariantEffectPredictor:
    """End-to-end variant effect prediction pipeline.

    Given a pretrained Genova model, extracts sequence embeddings for
    reference and alternate alleles around each variant, computes the
    embedding difference, and optionally applies a classification head
    to predict pathogenicity.

    Args:
        model: A pretrained Genova encoder (backbone, not MLM head).
            Must return hidden states when called with ``input_ids``.
        tokenizer: :class:`GenomicTokenizer` with a built vocabulary.
        device: Device to run inference on.
        window_size: Number of bases to extract around each variant
            (centered on the variant position).
        classifier_head: Optional pretrained :class:`VariantClassifierHead`.
            If ``None``, raw embedding differences are returned instead
            of pathogenicity scores.
        threshold: Decision threshold for the pathogenicity score.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        window_size: int = 512,
        classifier_head: Optional[VariantClassifierHead] = None,
        threshold: float = 0.5,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.window_size = window_size
        self.classifier_head = classifier_head
        self.threshold = threshold

        self.model.to(self.device).eval()
        if self.classifier_head is not None:
            self.classifier_head.to(self.device).eval()

    # ------------------------------------------------------------------
    # Sequence extraction
    # ------------------------------------------------------------------

    def _extract_windows(
        self,
        variant: Variant,
        reference: FastaReader,
    ) -> Tuple[str, str]:
        """Extract reference and alternate sequence windows.

        The window is centred on the variant position.  For indels the
        window is adjusted so the full alternate allele fits.

        Args:
            variant: The variant to process.
            reference: Indexed reference genome.

        Returns:
            Tuple of ``(ref_seq, alt_seq)``.
        """
        # VCF is 1-based; convert to 0-based
        var_pos_0 = variant.pos - 1
        half = self.window_size // 2

        start = max(0, var_pos_0 - half)
        end = start + self.window_size

        ref_window = reference.fetch(variant.chrom, start, end)

        # Construct alternate window by substituting the allele
        var_offset = var_pos_0 - start
        ref_len = len(variant.ref)
        alt_window = (
            ref_window[:var_offset]
            + variant.alt
            + ref_window[var_offset + ref_len:]
        )

        # Truncate / pad to window_size
        ref_window = ref_window[: self.window_size]
        alt_window = alt_window[: self.window_size]

        return ref_window, alt_window

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_embedding(self, sequence: str) -> np.ndarray:
        """Tokenize a sequence, run through the model, and return the
        mean-pooled hidden state.

        Args:
            sequence: DNA sequence string.

        Returns:
            1-D numpy array of shape ``(d_model,)``.
        """
        token_ids = self.tokenizer.encode(sequence)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Handle different return types
        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
        elif isinstance(outputs, torch.Tensor):
            hidden = outputs
        else:
            hidden = getattr(outputs, "last_hidden_state", outputs)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        # Mean pool over sequence length (dim=1)
        if hidden.dim() == 3:
            embedding = hidden.mean(dim=1).squeeze(0)
        else:
            embedding = hidden.squeeze(0)

        return embedding.cpu().float().numpy()

    @torch.no_grad()
    def _get_embeddings_batch(
        self, sequences: List[str]
    ) -> List[np.ndarray]:
        """Compute embeddings for a batch of sequences.

        Args:
            sequences: List of DNA sequence strings.

        Returns:
            List of 1-D numpy arrays, each of shape ``(d_model,)``.
        """
        all_ids: List[List[int]] = [
            self.tokenizer.encode(seq) for seq in sequences
        ]
        max_len = max(len(ids) for ids in all_ids)

        # Pad
        padded = [
            ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            for ids in all_ids
        ]
        masks = [
            [1] * len(ids) + [0] * (max_len - len(ids))
            for ids in all_ids
        ]

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state", outputs.get("hidden_states"))
        elif isinstance(outputs, torch.Tensor):
            hidden = outputs
        else:
            hidden = getattr(outputs, "last_hidden_state", outputs)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        # Mean pool respecting attention mask
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        embeddings = (summed / counts).cpu().float().numpy()

        return [embeddings[i] for i in range(len(sequences))]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_variant(
        self,
        variant: Variant,
        reference: FastaReader,
    ) -> VariantPrediction:
        """Predict the effect of a single variant.

        Args:
            variant: The variant to score.
            reference: Indexed reference genome.

        Returns:
            A :class:`VariantPrediction` with the pathogenicity score.
        """
        ref_seq, alt_seq = self._extract_windows(variant, reference)
        ref_emb = self._get_embedding(ref_seq)
        alt_emb = self._get_embedding(alt_seq)

        diff = alt_emb - ref_emb

        score = 0.0
        if self.classifier_head is not None:
            diff_tensor = torch.tensor(
                diff, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            logit = self.classifier_head(diff_tensor)
            score = float(torch.sigmoid(logit).item())
        else:
            # Fallback: use L2 norm of difference as a proxy score
            score = float(np.linalg.norm(diff))

        label = "pathogenic" if score >= self.threshold else "benign"

        return VariantPrediction(
            variant=variant,
            score=score,
            label=label,
            ref_embedding=ref_emb,
            alt_embedding=alt_emb,
        )

    def predict_variants(
        self,
        variants: Sequence[Variant],
        reference: FastaReader,
        batch_size: int = 32,
    ) -> List[VariantPrediction]:
        """Predict effects for a list of variants with batched inference.

        Args:
            variants: Sequence of :class:`Variant` objects.
            reference: Indexed reference genome.
            batch_size: Number of variants processed per batch.

        Returns:
            List of :class:`VariantPrediction` objects.
        """
        predictions: List[VariantPrediction] = []
        n = len(variants)

        for start in range(0, n, batch_size):
            batch_variants = variants[start : start + batch_size]

            ref_seqs: List[str] = []
            alt_seqs: List[str] = []
            for v in batch_variants:
                ref_s, alt_s = self._extract_windows(v, reference)
                ref_seqs.append(ref_s)
                alt_seqs.append(alt_s)

            ref_embs = self._get_embeddings_batch(ref_seqs)
            alt_embs = self._get_embeddings_batch(alt_seqs)

            for i, v in enumerate(batch_variants):
                diff = alt_embs[i] - ref_embs[i]

                if self.classifier_head is not None:
                    diff_tensor = torch.tensor(
                        diff, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    logit = self.classifier_head(diff_tensor)
                    score = float(torch.sigmoid(logit).item())
                else:
                    score = float(np.linalg.norm(diff))

                label = "pathogenic" if score >= self.threshold else "benign"
                predictions.append(
                    VariantPrediction(
                        variant=v,
                        score=score,
                        label=label,
                        ref_embedding=ref_embs[i],
                        alt_embedding=alt_embs[i],
                    )
                )

            logger.info(
                "Processed {}/{} variants.",
                min(start + batch_size, n),
                n,
            )

        return predictions

    def predict_vcf(
        self,
        vcf_path: Union[str, Path],
        reference_path: Union[str, Path],
        batch_size: int = 32,
    ) -> List[VariantPrediction]:
        """Predict effects for all variants in a VCF file.

        Args:
            vcf_path: Path to the VCF file.
            reference_path: Path to the reference FASTA file.
            batch_size: Batch size for inference.

        Returns:
            List of :class:`VariantPrediction` objects.
        """
        variants = list(parse_vcf(vcf_path))
        logger.info("Parsed {} variants from {}", len(variants), vcf_path)

        reference = FastaReader(reference_path)
        return self.predict_variants(variants, reference, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def predict_variants(
    vcf_path: Union[str, Path],
    reference_path: Union[str, Path],
    model: nn.Module,
    config: EvaluationConfig,
    *,
    tokenizer: Optional[GenomicTokenizer] = None,
    classifier_head: Optional[VariantClassifierHead] = None,
    device: Union[str, torch.device] = "cpu",
    window_size: int = 512,
) -> List[VariantPrediction]:
    """High-level function to predict variant effects from a VCF file.

    Args:
        vcf_path: Path to the VCF file.
        reference_path: Path to the reference FASTA file.
        model: Pretrained Genova encoder.
        config: Evaluation configuration.
        tokenizer: Optional tokenizer.  If ``None``, a default k-mer
            tokenizer is created.
        classifier_head: Optional pathogenicity classifier head.
        device: Device for inference.
        window_size: Bases to extract around each variant.

    Returns:
        List of :class:`VariantPrediction` objects.
    """
    if tokenizer is None:
        tokenizer = GenomicTokenizer(mode="kmer", k=6)
        tokenizer.build_vocab()

    batch_size = config.eval_batch_size

    predictor = VariantEffectPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        window_size=window_size,
        classifier_head=classifier_head,
    )

    return predictor.predict_vcf(vcf_path, reference_path, batch_size=batch_size)
