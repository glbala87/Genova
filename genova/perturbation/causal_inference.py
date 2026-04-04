"""Causal analysis of genomic perturbations.

Approximates interventional do-calculus effects, detects epistatic
(pairwise non-additive) interactions between positions, and builds
interaction landscape matrices.

Example::

    analyzer = CausalAnalyzer(model, tokenizer, device="cuda")
    single = analyzer.single_perturbation("ACGT" * 200, position=50)
    epistasis = analyzer.epistatic_scan("ACGT" * 200, positions=[10, 20, 50])
    landscape = analyzer.interaction_landscape("ACGT" * 200, region=(40, 60))
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
from genova.perturbation.variant_simulator import NUCLEOTIDES, VariantSimulator

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PerturbationEffect:
    """Effect of a single-position perturbation (do-calculus approximation).

    Attributes:
        position: 0-based position that was perturbed.
        ref_allele: Original nucleotide.
        alt_allele: Substituted nucleotide.
        embedding_change: L2 distance in embedding space.
        cosine_change: 1 - cosine similarity.
        causal_score: Combined causal effect score.
        downstream_effects: Per-position change in downstream embeddings.
    """

    position: int
    ref_allele: str
    alt_allele: str
    embedding_change: float = 0.0
    cosine_change: float = 0.0
    causal_score: float = 0.0
    downstream_effects: Optional[np.ndarray] = None


@dataclass
class EpistaticInteraction:
    """Pairwise epistatic interaction between two positions.

    Attributes:
        pos_a: First position.
        pos_b: Second position.
        alleles_a: ``(ref, alt)`` alleles at position A.
        alleles_b: ``(ref, alt)`` alleles at position B.
        effect_a: Effect of mutating A alone.
        effect_b: Effect of mutating B alone.
        effect_ab: Effect of mutating both A and B.
        expected_additive: Sum of individual effects.
        epistatic_score: ``effect_ab - expected_additive``.
            Positive = synergistic, negative = antagonistic.
        interaction_type: ``"synergistic"``, ``"antagonistic"``, or ``"additive"``.
    """

    pos_a: int
    pos_b: int
    alleles_a: Tuple[str, str] = ("", "")
    alleles_b: Tuple[str, str] = ("", "")
    effect_a: float = 0.0
    effect_b: float = 0.0
    effect_ab: float = 0.0
    expected_additive: float = 0.0
    epistatic_score: float = 0.0
    interaction_type: str = "additive"


# ---------------------------------------------------------------------------
# CausalAnalyzer
# ---------------------------------------------------------------------------


class CausalAnalyzer:
    """Causal inference on genomic sequences via in-silico perturbation.

    Approximates interventional effects using the ``do(X=x)`` framework:
    perturb a position, observe downstream changes, and decompose
    additive vs non-additive (epistatic) contributions.

    Args:
        model: Trained Genova model.
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        batch_size: Batch size for inference.
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
            "CausalAnalyzer initialised (device={}, batch_size={}).",
            self.device,
            batch_size,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _get_hidden_states(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Get per-position hidden states ``(L, D)``."""
        token_ids = self.tokenizer.encode(sequence, max_length=max_length)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
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
            return hidden.squeeze(0)  # (L, D)
        return hidden

    @torch.no_grad()
    def _get_pooled_embedding(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Get mean-pooled embedding ``(D,)``."""
        hidden = self._get_hidden_states(sequence, max_length)
        return hidden.mean(dim=0)

    # ------------------------------------------------------------------
    # Single perturbation (do-calculus approximation)
    # ------------------------------------------------------------------

    def single_perturbation(
        self,
        sequence: str,
        position: int,
        *,
        alt_allele: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> List[PerturbationEffect]:
        """Compute the causal effect of perturbing a single position.

        Approximates ``do(X_pos = alt)`` by substituting the nucleotide
        at *position* and measuring the resulting change in the full
        hidden-state representation.

        Args:
            sequence: Reference DNA sequence.
            position: 0-based position to perturb.
            alt_allele: Specific alternative allele.  If ``None``, tests
                all three alternatives.
            max_length: Maximum token length.

        Returns:
            List of :class:`PerturbationEffect` objects (one per alt allele).
        """
        seq_upper = sequence.upper()
        if position < 0 or position >= len(seq_upper):
            raise ValueError(
                f"Position {position} out of range for sequence of length {len(seq_upper)}."
            )

        ref_nt = seq_upper[position]
        ref_hidden = self._get_hidden_states(seq_upper, max_length)  # (L, D)
        ref_pooled = ref_hidden.mean(dim=0)

        alt_alleles = [alt_allele] if alt_allele else [n for n in NUCLEOTIDES if n != ref_nt]

        effects: List[PerturbationEffect] = []

        for alt_nt in alt_alleles:
            mutant = seq_upper[:position] + alt_nt + seq_upper[position + 1 :]
            mut_hidden = self._get_hidden_states(mutant, max_length)
            mut_pooled = mut_hidden.mean(dim=0)

            # Global embedding change
            l2 = float(torch.norm(mut_pooled - ref_pooled, p=2))
            cos = float(F.cosine_similarity(
                ref_pooled.unsqueeze(0), mut_pooled.unsqueeze(0)
            ))
            cosine_change = 1.0 - cos

            # Per-position downstream effects
            min_len = min(ref_hidden.shape[0], mut_hidden.shape[0])
            position_changes = torch.norm(
                mut_hidden[:min_len] - ref_hidden[:min_len], p=2, dim=-1
            )
            downstream = position_changes.cpu().numpy()

            # Causal score: combine L2 and cosine change
            causal_score = l2 * cosine_change

            effects.append(
                PerturbationEffect(
                    position=position,
                    ref_allele=ref_nt,
                    alt_allele=alt_nt,
                    embedding_change=l2,
                    cosine_change=cosine_change,
                    causal_score=causal_score,
                    downstream_effects=downstream,
                )
            )

        logger.debug(
            "Single perturbation at position {}: {} effects computed.",
            position,
            len(effects),
        )
        return effects

    # ------------------------------------------------------------------
    # Epistatic scan
    # ------------------------------------------------------------------

    def epistatic_scan(
        self,
        sequence: str,
        positions: List[int],
        *,
        max_length: Optional[int] = None,
        epistasis_threshold: float = 0.1,
    ) -> List[EpistaticInteraction]:
        """Scan all pairs of positions for epistatic interactions.

        For each pair ``(A, B)``, computes:
        - Effect of mutating A alone
        - Effect of mutating B alone
        - Effect of mutating both A and B simultaneously
        - Epistatic score = double_effect - (effect_A + effect_B)

        Non-zero epistatic scores indicate non-additive interactions.

        Args:
            sequence: Reference DNA sequence.
            positions: List of positions to test (all pairs scanned).
            max_length: Maximum token length.
            epistasis_threshold: Minimum absolute epistatic score to
                report (filters noise).

        Returns:
            List of :class:`EpistaticInteraction` objects sorted by
            absolute epistatic score.
        """
        seq_upper = sequence.upper()
        ref_pooled = self._get_pooled_embedding(seq_upper, max_length)

        # Cache single-mutant effects
        single_effects: Dict[Tuple[int, str], float] = {}
        single_mutants: Dict[Tuple[int, str], str] = {}

        for pos in positions:
            if pos < 0 or pos >= len(seq_upper):
                logger.warning("Position {} out of range -- skipping.", pos)
                continue
            ref_nt = seq_upper[pos]
            # Use the alt allele with maximum effect
            best_alt = None
            best_effect = -1.0
            for alt_nt in NUCLEOTIDES:
                if alt_nt == ref_nt:
                    continue
                mutant = seq_upper[:pos] + alt_nt + seq_upper[pos + 1 :]
                mut_pooled = self._get_pooled_embedding(mutant, max_length)
                eff = float(torch.norm(mut_pooled - ref_pooled, p=2))
                single_effects[(pos, alt_nt)] = eff
                single_mutants[(pos, alt_nt)] = mutant
                if eff > best_effect:
                    best_effect = eff
                    best_alt = alt_nt

            if best_alt is not None:
                single_effects[(pos, "best")] = best_effect
                single_effects[(pos, "best_alt")] = best_alt  # type: ignore[assignment]

        # Test all pairs
        interactions: List[EpistaticInteraction] = []
        valid_positions = [p for p in positions if 0 <= p < len(seq_upper)]

        for i, pos_a in enumerate(valid_positions):
            ref_a = seq_upper[pos_a]
            best_alt_a = single_effects.get((pos_a, "best_alt"))
            if best_alt_a is None:
                continue
            best_alt_a = str(best_alt_a)
            effect_a = single_effects.get((pos_a, best_alt_a), 0.0)

            for pos_b in valid_positions[i + 1 :]:
                ref_b = seq_upper[pos_b]
                best_alt_b = single_effects.get((pos_b, "best_alt"))
                if best_alt_b is None:
                    continue
                best_alt_b = str(best_alt_b)
                effect_b = single_effects.get((pos_b, best_alt_b), 0.0)

                # Double mutant
                double = list(seq_upper)
                double[pos_a] = best_alt_a
                double[pos_b] = best_alt_b
                double_seq = "".join(double)

                double_pooled = self._get_pooled_embedding(double_seq, max_length)
                effect_ab = float(torch.norm(double_pooled - ref_pooled, p=2))

                expected = effect_a + effect_b
                epistatic_score = effect_ab - expected

                if abs(epistatic_score) < epistasis_threshold:
                    continue

                if epistatic_score > 0:
                    interaction_type = "synergistic"
                elif epistatic_score < 0:
                    interaction_type = "antagonistic"
                else:
                    interaction_type = "additive"

                interactions.append(
                    EpistaticInteraction(
                        pos_a=pos_a,
                        pos_b=pos_b,
                        alleles_a=(ref_a, best_alt_a),
                        alleles_b=(ref_b, best_alt_b),
                        effect_a=effect_a,
                        effect_b=effect_b,
                        effect_ab=effect_ab,
                        expected_additive=expected,
                        epistatic_score=epistatic_score,
                        interaction_type=interaction_type,
                    )
                )

        interactions.sort(key=lambda x: abs(x.epistatic_score), reverse=True)

        logger.info(
            "Epistatic scan: {} significant interactions from {} position pairs "
            "(threshold={}).",
            len(interactions),
            len(valid_positions) * (len(valid_positions) - 1) // 2,
            epistasis_threshold,
        )
        return interactions

    # ------------------------------------------------------------------
    # Interaction landscape
    # ------------------------------------------------------------------

    def interaction_landscape(
        self,
        sequence: str,
        *,
        region: Optional[Tuple[int, int]] = None,
        max_length: Optional[int] = None,
        epistasis_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Build a pairwise interaction landscape matrix.

        Computes epistatic scores for all pairs within a region and
        returns them as a symmetric matrix.

        Args:
            sequence: Reference DNA sequence.
            region: ``(start, end)`` region to analyse.  Defaults to
                the full sequence (expensive for long sequences).
            max_length: Maximum token length.
            epistasis_threshold: Passed to :meth:`epistatic_scan`.

        Returns:
            Dict with:
                - ``interaction_matrix``: ``(N, N)`` symmetric matrix
                  of epistatic scores.
                - ``positions``: List of position indices.
                - ``interactions``: List of :class:`EpistaticInteraction`.
                - ``synergistic_count``: Number of synergistic pairs.
                - ``antagonistic_count``: Number of antagonistic pairs.
        """
        seq_upper = sequence.upper()
        start, end = 0, len(seq_upper)
        if region is not None:
            start, end = max(0, region[0]), min(len(seq_upper), region[1])

        positions = list(range(start, end))

        if len(positions) > 100:
            logger.warning(
                "Region has {} positions -- interaction landscape will be "
                "computationally expensive. Consider narrowing the region.",
                len(positions),
            )

        interactions = self.epistatic_scan(
            sequence,
            positions,
            max_length=max_length,
            epistasis_threshold=epistasis_threshold,
        )

        # Build matrix
        n = len(positions)
        pos_to_idx = {p: i for i, p in enumerate(positions)}
        matrix = np.zeros((n, n), dtype=np.float64)

        synergistic = 0
        antagonistic = 0

        for inter in interactions:
            i = pos_to_idx.get(inter.pos_a)
            j = pos_to_idx.get(inter.pos_b)
            if i is not None and j is not None:
                matrix[i, j] = inter.epistatic_score
                matrix[j, i] = inter.epistatic_score

                if inter.interaction_type == "synergistic":
                    synergistic += 1
                elif inter.interaction_type == "antagonistic":
                    antagonistic += 1

        # Diagonal: single-mutant effects
        ref_pooled = self._get_pooled_embedding(seq_upper, max_length)
        for pos in positions:
            idx = pos_to_idx[pos]
            ref_nt = seq_upper[pos]
            max_effect = 0.0
            for alt_nt in NUCLEOTIDES:
                if alt_nt == ref_nt:
                    continue
                mutant = seq_upper[:pos] + alt_nt + seq_upper[pos + 1 :]
                mut_pooled = self._get_pooled_embedding(mutant, max_length)
                eff = float(torch.norm(mut_pooled - ref_pooled, p=2))
                max_effect = max(max_effect, eff)
            matrix[idx, idx] = max_effect

        logger.info(
            "Interaction landscape: {}x{} matrix, {} synergistic, "
            "{} antagonistic interactions.",
            n,
            n,
            synergistic,
            antagonistic,
        )

        return {
            "interaction_matrix": matrix,
            "positions": positions,
            "interactions": interactions,
            "synergistic_count": synergistic,
            "antagonistic_count": antagonistic,
        }
