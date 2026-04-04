"""Systematic in-silico mutagenesis framework for Genova models.

Introduces SNPs (saturation mutagenesis) and indels at every position of a
genomic sequence, then measures embedding and prediction changes to identify
functionally important positions.

Example::

    sim = VariantSimulator(model, tokenizer, device="cuda")
    snp_effects = sim.saturate_snps("ACGTACGT" * 100)
    indel_effects = sim.simulate_indels("ACGTACGT" * 100, indel_lengths=[1, 3])
    landscape = sim.compute_effect_landscape("ACGTACGT" * 100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

NUCLEOTIDES = ["A", "C", "G", "T"]


@dataclass
class VariantEffect:
    """Effect of a single in-silico variant.

    Attributes:
        position: 0-based position in the original sequence.
        ref_allele: Reference allele.
        alt_allele: Alternate allele.
        variant_type: ``"SNP"``, ``"insertion"``, or ``"deletion"``.
        l2_distance: L2 distance between reference and variant embeddings.
        cosine_similarity: Cosine similarity between embeddings.
        prediction_change: Change in predicted output (if applicable).
        effect_size: Combined effect score (normalised).
    """

    position: int
    ref_allele: str
    alt_allele: str
    variant_type: str = "SNP"
    l2_distance: float = 0.0
    cosine_similarity: float = 1.0
    prediction_change: float = 0.0
    effect_size: float = 0.0


# ---------------------------------------------------------------------------
# VariantSimulator
# ---------------------------------------------------------------------------


class VariantSimulator:
    """Systematic in-silico mutation framework.

    Generates every possible single-nucleotide substitution (saturation
    mutagenesis) or insertion/deletion at each position, runs batched
    inference, and measures embedding changes.

    Args:
        model: Trained Genova model.
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        batch_size: Number of variants per inference batch.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.model.to(self.device).eval()
        logger.info(
            "VariantSimulator initialised (device={}, batch_size={}).",
            self.device,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_embedding(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Get the pooled embedding for a sequence.

        Returns a 1-D tensor of shape ``(D,)``.
        """
        token_ids = self.tokenizer.encode(sequence, max_length=max_length)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract hidden states
        if isinstance(output, dict):
            hidden = output.get(
                "last_hidden_state",
                output.get("logits", output.get("embeddings")),
            )
            if hidden is None:
                hidden = next(iter(output.values()))
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Pool: mean over sequence length
        if hidden.dim() == 3:
            # (B, L, D) -> (D,)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return pooled.squeeze(0)
        elif hidden.dim() == 2:
            return hidden.squeeze(0)
        return hidden.flatten()

    @torch.no_grad()
    def _get_embeddings_batch(
        self,
        sequences: List[str],
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Get pooled embeddings for a batch of sequences.

        Returns a tensor of shape ``(N, D)``.
        """
        all_token_ids = [
            self.tokenizer.encode(seq, max_length=max_length)
            for seq in sequences
        ]

        # Pad to same length
        max_len = max(len(ids) for ids in all_token_ids)
        padded = [
            ids + [self.tokenizer.pad_token_id] * (max_len - len(ids))
            for ids in all_token_ids
        ]

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(output, dict):
            hidden = output.get(
                "last_hidden_state",
                output.get("logits", output.get("embeddings")),
            )
            if hidden is None:
                hidden = next(iter(output.values()))
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if hidden.dim() == 3:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return pooled
        return hidden

    # ------------------------------------------------------------------
    # Saturation mutagenesis (SNPs)
    # ------------------------------------------------------------------

    def saturate_snps(
        self,
        sequence: str,
        *,
        max_length: Optional[int] = None,
        region: Optional[Tuple[int, int]] = None,
    ) -> List[VariantEffect]:
        """Perform saturation mutagenesis: introduce every possible SNP.

        For each position in the sequence (or a specified region),
        substitutes all three alternative nucleotides and measures the
        effect on the model embedding.

        Args:
            sequence: Reference DNA sequence.
            max_length: Maximum token length for model.
            region: Optional ``(start, end)`` to restrict mutagenesis.

        Returns:
            List of :class:`VariantEffect` objects for every SNP.
        """
        seq_upper = sequence.upper()
        start, end = 0, len(seq_upper)
        if region is not None:
            start, end = region
            start = max(0, start)
            end = min(len(seq_upper), end)

        # Reference embedding
        ref_emb = self._get_embedding(seq_upper, max_length=max_length)

        # Generate all variants
        variant_seqs: List[str] = []
        variant_info: List[Tuple[int, str, str]] = []

        for pos in range(start, end):
            ref_nt = seq_upper[pos]
            for alt_nt in NUCLEOTIDES:
                if alt_nt == ref_nt:
                    continue
                mutant = seq_upper[:pos] + alt_nt + seq_upper[pos + 1 :]
                variant_seqs.append(mutant)
                variant_info.append((pos, ref_nt, alt_nt))

        # Batched inference
        effects = self._evaluate_variants(
            ref_emb, variant_seqs, variant_info, "SNP", max_length
        )

        logger.info(
            "Saturation mutagenesis: {} SNPs across positions [{}, {}).",
            len(effects),
            start,
            end,
        )
        return effects

    # ------------------------------------------------------------------
    # Indel simulation
    # ------------------------------------------------------------------

    def simulate_indels(
        self,
        sequence: str,
        *,
        indel_lengths: Optional[List[int]] = None,
        max_length: Optional[int] = None,
        region: Optional[Tuple[int, int]] = None,
    ) -> List[VariantEffect]:
        """Simulate insertions and deletions of configurable length.

        Args:
            sequence: Reference DNA sequence.
            indel_lengths: Lengths of indels to simulate.
                Defaults to ``[1, 2, 3]``.
            max_length: Maximum token length for model.
            region: Optional ``(start, end)`` to restrict simulation.

        Returns:
            List of :class:`VariantEffect` objects for all indels.
        """
        if indel_lengths is None:
            indel_lengths = [1, 2, 3]

        seq_upper = sequence.upper()
        start, end = 0, len(seq_upper)
        if region is not None:
            start, end = region
            start = max(0, start)
            end = min(len(seq_upper), end)

        ref_emb = self._get_embedding(seq_upper, max_length=max_length)

        variant_seqs: List[str] = []
        variant_info: List[Tuple[int, str, str]] = []
        variant_types: List[str] = []

        for length in indel_lengths:
            for pos in range(start, end):
                # Deletion
                if pos + length <= len(seq_upper):
                    deleted = seq_upper[pos : pos + length]
                    mutant = seq_upper[:pos] + seq_upper[pos + length :]
                    variant_seqs.append(mutant)
                    variant_info.append((pos, deleted, "-"))
                    variant_types.append("deletion")

                # Insertion (insert random nucleotide repeat at position)
                insert_seq = "A" * length  # canonical insertion
                mutant = seq_upper[:pos] + insert_seq + seq_upper[pos:]
                variant_seqs.append(mutant)
                variant_info.append((pos, "-", insert_seq))
                variant_types.append("insertion")

        # Batched inference
        all_effects: List[VariantEffect] = []
        for batch_start in range(0, len(variant_seqs), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(variant_seqs))
            batch_seqs = variant_seqs[batch_start:batch_end]
            batch_info = variant_info[batch_start:batch_end]
            batch_types = variant_types[batch_start:batch_end]

            batch_embs = self._get_embeddings_batch(batch_seqs, max_length)

            for i, (emb, info, vtype) in enumerate(
                zip(batch_embs, batch_info, batch_types)
            ):
                pos, ref_allele, alt_allele = info
                l2 = float(torch.norm(emb - ref_emb, p=2))
                cos = float(F.cosine_similarity(
                    ref_emb.unsqueeze(0), emb.unsqueeze(0)
                ))

                all_effects.append(
                    VariantEffect(
                        position=pos,
                        ref_allele=ref_allele,
                        alt_allele=alt_allele,
                        variant_type=vtype,
                        l2_distance=l2,
                        cosine_similarity=cos,
                        effect_size=l2 * (1.0 - cos),
                    )
                )

        logger.info(
            "Simulated {} indels (lengths={}) across positions [{}, {}).",
            len(all_effects),
            indel_lengths,
            start,
            end,
        )
        return all_effects

    # ------------------------------------------------------------------
    # Effect landscape
    # ------------------------------------------------------------------

    def compute_effect_landscape(
        self,
        sequence: str,
        *,
        max_length: Optional[int] = None,
        include_indels: bool = True,
        indel_lengths: Optional[List[int]] = None,
        region: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Compute a comprehensive effect landscape for a sequence.

        Combines saturation mutagenesis and indel simulation into a
        unified per-position effect summary.

        Args:
            sequence: Reference DNA sequence.
            max_length: Maximum token length.
            include_indels: Whether to include indel effects.
            indel_lengths: Indel lengths if *include_indels* is ``True``.
            region: Optional ``(start, end)`` region restriction.

        Returns:
            Dict with:
                - ``snp_effects``: List of SNP :class:`VariantEffect` objects.
                - ``indel_effects``: List of indel :class:`VariantEffect` objects.
                - ``position_scores``: ``(L,)`` array of max effect per position.
                - ``effect_matrix``: ``(L, 3)`` SNP effect matrix
                  (positions x alt alleles, L2 distance).
                - ``sensitive_positions``: Indices of top 10% positions.
        """
        snp_effects = self.saturate_snps(
            sequence, max_length=max_length, region=region
        )

        indel_effects: List[VariantEffect] = []
        if include_indels:
            indel_effects = self.simulate_indels(
                sequence,
                indel_lengths=indel_lengths,
                max_length=max_length,
                region=region,
            )

        seq_upper = sequence.upper()
        seq_len = len(seq_upper)
        start = region[0] if region else 0
        end = region[1] if region else seq_len

        # Build per-position score array
        position_scores = np.zeros(seq_len, dtype=np.float64)
        effect_matrix = np.zeros((seq_len, 3), dtype=np.float64)

        for eff in snp_effects:
            pos = eff.position
            if 0 <= pos < seq_len:
                position_scores[pos] = max(position_scores[pos], eff.l2_distance)
                # Map alt allele to column
                ref_nt = seq_upper[pos]
                alt_nts = [n for n in NUCLEOTIDES if n != ref_nt]
                try:
                    col = alt_nts.index(eff.alt_allele)
                    effect_matrix[pos, col] = eff.l2_distance
                except ValueError:
                    pass

        for eff in indel_effects:
            pos = eff.position
            if 0 <= pos < seq_len:
                position_scores[pos] = max(position_scores[pos], eff.l2_distance)

        # Identify sensitive positions (top 10%)
        n_sensitive = max(1, (end - start) // 10)
        region_scores = position_scores[start:end]
        sensitive_idx = np.argsort(region_scores)[-n_sensitive:][::-1]
        sensitive_positions = (sensitive_idx + start).tolist()

        logger.info(
            "Effect landscape: {} SNPs, {} indels, {} sensitive positions.",
            len(snp_effects),
            len(indel_effects),
            len(sensitive_positions),
        )

        return {
            "snp_effects": snp_effects,
            "indel_effects": indel_effects,
            "position_scores": position_scores,
            "effect_matrix": effect_matrix,
            "sensitive_positions": sensitive_positions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_variants(
        self,
        ref_emb: torch.Tensor,
        variant_seqs: List[str],
        variant_info: List[Tuple[int, str, str]],
        variant_type: str,
        max_length: Optional[int],
    ) -> List[VariantEffect]:
        """Run batched inference on variants and compute effects."""
        effects: List[VariantEffect] = []

        for batch_start in range(0, len(variant_seqs), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(variant_seqs))
            batch_seqs = variant_seqs[batch_start:batch_end]
            batch_info = variant_info[batch_start:batch_end]

            batch_embs = self._get_embeddings_batch(batch_seqs, max_length)

            for emb, (pos, ref_allele, alt_allele) in zip(batch_embs, batch_info):
                l2 = float(torch.norm(emb - ref_emb, p=2))
                cos = float(F.cosine_similarity(
                    ref_emb.unsqueeze(0), emb.unsqueeze(0)
                ))

                effects.append(
                    VariantEffect(
                        position=pos,
                        ref_allele=ref_allele,
                        alt_allele=alt_allele,
                        variant_type=variant_type,
                        l2_distance=l2,
                        cosine_similarity=cos,
                        effect_size=l2 * (1.0 - cos),
                    )
                )

        return effects
