"""Regulatory motif extraction from trained Genova models.

Extracts high-importance sequence regions using attention rollout or
gradient-based saliency, then builds Position Weight Matrices (PWMs) from
aligned candidate motifs.  Supports genome-wide scanning with a sliding
window approach.

Example::

    discoverer = MotifDiscovery(model, tokenizer, device="cuda")
    motifs = discoverer.extract_motifs(sequences, top_k=500)
    pwm = discoverer.build_pwm(motifs)
    hits = discoverer.scan_genome("ACGT" * 1000, pwm)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from genova.data.tokenizer import GenomicTokenizer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

NUCLEOTIDE_INDEX: Dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}
INDEX_NUCLEOTIDE: Dict[int, str] = {v: k for k, v in NUCLEOTIDE_INDEX.items()}


@dataclass
class Motif:
    """A discovered candidate motif.

    Attributes:
        sequence: Consensus DNA string.
        positions: List of (seq_idx, start, end) tuples where this motif
            was found in the input corpus.
        score: Aggregate importance score.
        pwm: Optional ``(L, 4)`` Position Weight Matrix (columns A, C, G, T).
    """

    sequence: str
    positions: List[Tuple[int, int, int]] = field(default_factory=list)
    score: float = 0.0
    pwm: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return len(self.sequence)


# ---------------------------------------------------------------------------
# PWM utilities
# ---------------------------------------------------------------------------


def sequences_to_pwm(
    sequences: Sequence[str],
    pseudocount: float = 0.01,
) -> np.ndarray:
    """Build a Position Weight Matrix from a collection of equal-length sequences.

    Args:
        sequences: Aligned DNA strings of identical length.
        pseudocount: Small value added to avoid log-zero issues.

    Returns:
        Frequency matrix of shape ``(L, 4)`` with columns ordered A, C, G, T.
        Each row sums to 1.

    Raises:
        ValueError: If sequences have inconsistent lengths or are empty.
    """
    if not sequences:
        raise ValueError("At least one sequence is required to build a PWM.")

    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("All sequences must have the same length for PWM construction.")

    counts = np.full((length, 4), pseudocount, dtype=np.float64)
    for seq in sequences:
        for i, nt in enumerate(seq.upper()):
            idx = NUCLEOTIDE_INDEX.get(nt)
            if idx is not None:
                counts[i, idx] += 1.0

    # Normalise rows to frequencies
    row_sums = counts.sum(axis=1, keepdims=True)
    pwm = counts / np.maximum(row_sums, 1e-12)
    return pwm


def pwm_to_consensus(pwm: np.ndarray) -> str:
    """Return the consensus sequence from a PWM.

    Args:
        pwm: ``(L, 4)`` frequency matrix.

    Returns:
        Consensus DNA string.
    """
    return "".join(INDEX_NUCLEOTIDE[int(idx)] for idx in np.argmax(pwm, axis=1))


def score_sequence_with_pwm(
    sequence: str,
    pwm: np.ndarray,
    background: Optional[np.ndarray] = None,
) -> float:
    """Score a sequence against a PWM using log-likelihood ratio.

    Args:
        sequence: DNA string (same length as PWM).
        pwm: ``(L, 4)`` Position Weight Matrix.
        background: ``(4,)`` background nucleotide frequencies.
            Defaults to uniform ``[0.25, 0.25, 0.25, 0.25]``.

    Returns:
        Sum of log-likelihood ratios across positions.
    """
    if background is None:
        background = np.full(4, 0.25)
    score = 0.0
    for i, nt in enumerate(sequence.upper()):
        idx = NUCLEOTIDE_INDEX.get(nt)
        if idx is not None:
            freq = max(pwm[i, idx], 1e-12)
            bg = max(background[idx], 1e-12)
            score += np.log2(freq / bg)
    return float(score)


# ---------------------------------------------------------------------------
# MotifDiscovery
# ---------------------------------------------------------------------------


class MotifDiscovery:
    """Extract regulatory motifs from a trained Genova model.

    Supports two extraction strategies:

    * **attention** -- Uses attention rollout to identify high-importance
      positions, then extracts surrounding subsequences.
    * **gradient** -- Computes input-gradient saliency maps and extracts
      regions with large gradient magnitude.

    Args:
        model: Trained Genova model (backbone or MLM wrapper).
        tokenizer: Tokenizer with built vocabulary.
        device: Inference device.
        motif_length: Default motif window size (nucleotides).
        method: ``"attention"`` or ``"gradient"``.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: GenomicTokenizer,
        *,
        device: Union[str, torch.device] = "cpu",
        motif_length: int = 12,
        method: str = "attention",
    ) -> None:
        if method not in ("attention", "gradient"):
            raise ValueError(f"method must be 'attention' or 'gradient', got '{method}'")

        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.motif_length = motif_length
        self.method = method

        self.model.to(self.device).eval()
        logger.info(
            "MotifDiscovery initialised (method={}, motif_length={}, device={}).",
            method,
            motif_length,
            self.device,
        )

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _attention_importance(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-position importance via attention rollout.

        Returns a 1-D array of length ``min(len(sequence), max_length)``
        with importance scores.
        """
        from genova.explainability.attention_analysis import AttentionAnalyzer

        analyzer = AttentionAnalyzer(
            self.model, self.tokenizer, device=str(self.device)
        )
        result = analyzer.analyze(sequence, max_length=max_length)
        importance = result.get("token_importance", np.array([]))
        return importance

    def _gradient_importance(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-position importance via input-gradient saliency.

        Computes ``|dL/dx * x|`` summed over the embedding dimension,
        where ``x`` is the token embedding and ``L`` is the model output
        norm (embedding-space proxy for output magnitude).
        """
        token_ids = self.tokenizer.encode(sequence, max_length=max_length)
        input_ids = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Enable gradient on embeddings
        self.model.eval()
        embedding_layer = self._find_embedding_layer()
        if embedding_layer is None:
            logger.warning("Could not locate embedding layer; falling back to attention.")
            return self._attention_importance(sequence, max_length)

        # Hook to capture and enable grad on embeddings
        embeddings: List[torch.Tensor] = []

        def hook_fn(module: nn.Module, input: Any, output: torch.Tensor) -> torch.Tensor:
            output.requires_grad_(True)
            output.retain_grad()
            embeddings.append(output)
            return output

        handle = embedding_layer.register_forward_hook(hook_fn)
        try:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # Use output norm as scalar target
            if isinstance(output, dict):
                logits = output.get("logits", output.get("last_hidden_state"))
                if logits is None:
                    logits = next(iter(output.values()))
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            target = logits.norm()
            target.backward()

            if embeddings and embeddings[0].grad is not None:
                grad = embeddings[0].grad  # (1, L, D)
                emb = embeddings[0]  # (1, L, D)
                saliency = (grad * emb).abs().sum(dim=-1).squeeze(0)  # (L,)
                importance = saliency.detach().cpu().numpy()
                # Normalise
                total = importance.sum()
                if total > 0:
                    importance = importance / total
                return importance

        finally:
            handle.remove()

        return np.array([])

    def _find_embedding_layer(self) -> Optional[nn.Module]:
        """Locate the token embedding layer in the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and "token" in name.lower():
                return module
        # Fallback: first large Embedding
        for _name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and module.num_embeddings > 10:
                return module
        return None

    def compute_importance(
        self,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-position importance using the configured method.

        Args:
            sequence: DNA sequence to analyze.
            max_length: Maximum token length for the model.

        Returns:
            1-D importance array.
        """
        if self.method == "gradient":
            return self._gradient_importance(sequence, max_length)
        return self._attention_importance(sequence, max_length)

    # ------------------------------------------------------------------
    # Motif extraction
    # ------------------------------------------------------------------

    def extract_motifs(
        self,
        sequences: Sequence[str],
        *,
        top_k: int = 100,
        motif_length: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        max_length: Optional[int] = None,
    ) -> List[Motif]:
        """Extract candidate motifs from a collection of sequences.

        For each sequence, computes importance scores, identifies peaks,
        and extracts surrounding windows as candidate motifs.

        Args:
            sequences: Iterable of DNA sequences.
            top_k: Maximum number of motifs to return (globally ranked).
            motif_length: Window size around peaks.  Defaults to
                ``self.motif_length``.
            importance_threshold: Minimum importance to consider a
                position a peak.  If ``None``, uses
                ``mean + 2 * std`` of the importance distribution.
            max_length: Maximum token length for the model.

        Returns:
            Sorted list of :class:`Motif` objects (highest score first).
        """
        motif_length = motif_length or self.motif_length
        half_w = motif_length // 2

        candidates: List[Motif] = []

        for seq_idx, seq in enumerate(sequences):
            importance = self.compute_importance(seq, max_length=max_length)
            if importance.size == 0:
                logger.debug("Empty importance for sequence {} -- skipping.", seq_idx)
                continue

            # Determine threshold
            if importance_threshold is not None:
                threshold = importance_threshold
            else:
                threshold = float(importance.mean() + 2.0 * importance.std())

            # Find peak positions (local maxima above threshold)
            peaks = self._find_peaks(importance, threshold)

            for peak_pos in peaks:
                # Map token position back to sequence position (approximate)
                start = max(0, peak_pos - half_w)
                end = min(len(seq), start + motif_length)
                start = max(0, end - motif_length)

                subseq = seq[start:end].upper()
                if len(subseq) < motif_length:
                    continue
                if "N" * (motif_length // 2) in subseq:
                    continue  # skip N-rich

                score = float(importance[peak_pos])
                candidates.append(
                    Motif(
                        sequence=subseq,
                        positions=[(seq_idx, start, end)],
                        score=score,
                    )
                )

        # Sort by score descending and take top_k
        candidates.sort(key=lambda m: m.score, reverse=True)
        candidates = candidates[:top_k]

        logger.info(
            "Extracted {} candidate motifs from {} sequences.",
            len(candidates),
            len(sequences),
        )
        return candidates

    @staticmethod
    def _find_peaks(
        importance: np.ndarray,
        threshold: float,
        min_distance: int = 3,
    ) -> List[int]:
        """Find local maxima in the importance array above a threshold.

        Args:
            importance: 1-D importance scores.
            threshold: Minimum value for a peak.
            min_distance: Minimum distance between peaks.

        Returns:
            Sorted list of peak indices.
        """
        peaks: List[int] = []
        n = len(importance)

        for i in range(1, n - 1):
            if importance[i] < threshold:
                continue
            if importance[i] >= importance[i - 1] and importance[i] >= importance[i + 1]:
                # Check min distance from existing peaks
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)

        return peaks

    # ------------------------------------------------------------------
    # PWM construction
    # ------------------------------------------------------------------

    def build_pwm(
        self,
        motifs: Sequence[Motif],
        pseudocount: float = 0.01,
    ) -> np.ndarray:
        """Build a Position Weight Matrix from discovered motifs.

        All motif sequences must have the same length.  Motifs with
        differing lengths are trimmed or padded to the mode length.

        Args:
            motifs: Collection of :class:`Motif` objects.
            pseudocount: Pseudocount for frequency estimation.

        Returns:
            ``(L, 4)`` frequency matrix (A, C, G, T columns).
        """
        if not motifs:
            raise ValueError("At least one motif is required to build a PWM.")

        # Determine the target length (mode of motif lengths)
        lengths = [m.length for m in motifs]
        target_len = max(set(lengths), key=lengths.count)

        # Collect sequences trimmed/padded to target length
        seqs: List[str] = []
        for m in motifs:
            s = m.sequence
            if len(s) > target_len:
                s = s[:target_len]
            elif len(s) < target_len:
                continue  # skip mismatched
            seqs.append(s)

        if not seqs:
            raise ValueError("No motif sequences with target length available.")

        pwm = sequences_to_pwm(seqs, pseudocount=pseudocount)

        # Assign PWM to each motif
        for m in motifs:
            if m.length == target_len:
                m.pwm = pwm.copy()

        logger.info(
            "Built PWM of length {} from {} sequences.", target_len, len(seqs)
        )
        return pwm

    # ------------------------------------------------------------------
    # Genome scanning
    # ------------------------------------------------------------------

    def scan_genome(
        self,
        genome: str,
        pwm: np.ndarray,
        *,
        threshold: Optional[float] = None,
        background: Optional[np.ndarray] = None,
        stride: int = 1,
    ) -> List[Dict[str, Any]]:
        """Scan a genome sequence for PWM matches.

        Slides the PWM across the genome and reports positions whose
        log-likelihood ratio score exceeds the threshold.

        Args:
            genome: Full genome or chromosome sequence.
            pwm: ``(L, 4)`` Position Weight Matrix.
            threshold: Minimum score.  Defaults to 50 % of the max
                possible score.
            background: ``(4,)`` background frequencies.
            stride: Sliding window step size.

        Returns:
            List of hit dicts with keys ``start``, ``end``, ``score``,
            ``sequence``, ``strand``.
        """
        motif_len = pwm.shape[0]
        genome_upper = genome.upper()

        if threshold is None:
            # 50% of maximum possible score
            max_score = float(
                np.sum(np.log2(np.maximum(pwm.max(axis=1), 1e-12) / 0.25))
            )
            threshold = max_score * 0.5

        # Also scan reverse complement
        from genova.data.tokenizer import reverse_complement

        rc_genome = reverse_complement(genome_upper)

        hits: List[Dict[str, Any]] = []

        for strand, seq in [("+", genome_upper), ("-", rc_genome)]:
            for start in range(0, len(seq) - motif_len + 1, stride):
                subseq = seq[start : start + motif_len]
                sc = score_sequence_with_pwm(subseq, pwm, background)
                if sc >= threshold:
                    # For reverse strand, map position back
                    if strand == "-":
                        mapped_start = len(seq) - start - motif_len
                    else:
                        mapped_start = start
                    hits.append(
                        {
                            "start": mapped_start,
                            "end": mapped_start + motif_len,
                            "score": sc,
                            "sequence": subseq,
                            "strand": strand,
                        }
                    )

        # Sort by score descending
        hits.sort(key=lambda h: h["score"], reverse=True)
        logger.info(
            "Genome scan found {} hits above threshold {:.2f}.",
            len(hits),
            threshold,
        )
        return hits
