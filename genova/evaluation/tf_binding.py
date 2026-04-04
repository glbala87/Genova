"""Transcription factor binding site prediction for Genova.

Predicts TF binding sites from DNA sequence using encoder embeddings
with a multi-label classification head.  Supports comparison against
known position weight matrices (PWMs) in JASPAR format.

Example::

    from genova.evaluation.tf_binding import TFBindingPredictor

    predictor = TFBindingPredictor(encoder, num_tfs=600, d_model=768)
    prediction = predictor.predict(sequence_tensor)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TFBindingPrediction:
    """Prediction result for TF binding at a genomic region.

    Attributes:
        tf_name: Name of the transcription factor.
        positions: List of 0-based positions where binding is predicted.
        scores: Binding probability at each predicted position.
        motif: Consensus motif string (if available from PWM comparison).
    """

    tf_name: str
    positions: List[int]
    scores: List[float]
    motif: Optional[str] = None


@dataclass
class JASPARMotif:
    """A single motif entry loaded from a JASPAR-format file.

    Attributes:
        matrix_id: JASPAR matrix accession (e.g. ``MA0001.1``).
        tf_name: Transcription factor name.
        pwm: Position weight matrix of shape ``(4, motif_length)``.
            Row order is A, C, G, T.
        consensus: Consensus sequence derived from the PWM.
    """

    matrix_id: str
    tf_name: str
    pwm: np.ndarray  # (4, L)
    consensus: str = ""

    def __post_init__(self) -> None:
        if not self.consensus and self.pwm.size > 0:
            bases = "ACGT"
            self.consensus = "".join(
                bases[i] for i in np.argmax(self.pwm, axis=0)
            )


# ---------------------------------------------------------------------------
# JASPAR file parser
# ---------------------------------------------------------------------------


def load_jaspar_motifs(path: Union[str, Path]) -> List[JASPARMotif]:
    """Load motifs from a JASPAR-format PFM/PWM file.

    The expected format is the standard JASPAR plain-text format::

        >MA0001.1  AGL3
        A  [ 0  3 79 40 ...]
        C  [94 75  4  3 ...]
        G  [ 1  0  3  4 ...]
        T  [ 2 19 11 50 ...]

    Args:
        path: Path to the JASPAR file.

    Returns:
        List of :class:`JASPARMotif` objects.
    """
    path = Path(path)
    motifs: List[JASPARMotif] = []

    with open(path, "r") as fh:
        lines = fh.readlines()

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith(">"):
            # Parse header: >MA0001.1  TF_NAME
            parts = line[1:].split(None, 1)
            matrix_id = parts[0] if parts else "unknown"
            tf_name = parts[1].strip() if len(parts) > 1 else matrix_id

            rows: Dict[str, List[float]] = {}
            for base in "ACGT":
                idx += 1
                if idx >= len(lines):
                    break
                row_line = lines[idx].strip()
                # Extract numbers from lines like "A  [ 1 2 3 ]"
                # or "A 1 2 3"
                numbers = re.findall(r"[-+]?\d*\.?\d+", row_line.split(base, 1)[-1] if base in row_line else row_line)
                rows[base] = [float(n) for n in numbers]

            if all(b in rows for b in "ACGT") and len(rows["A"]) > 0:
                pwm = np.array([rows[b] for b in "ACGT"], dtype=np.float64)
                # Normalise columns to probabilities
                col_sums = pwm.sum(axis=0, keepdims=True)
                col_sums = np.where(col_sums == 0, 1.0, col_sums)
                pwm = pwm / col_sums
                motifs.append(
                    JASPARMotif(matrix_id=matrix_id, tf_name=tf_name, pwm=pwm)
                )
        idx += 1

    return motifs


def _pwm_to_log_odds(pwm: np.ndarray, pseudocount: float = 0.01) -> np.ndarray:
    """Convert a PWM (probability matrix) to log-odds scores.

    Args:
        pwm: Position weight matrix of shape ``(4, L)``, rows sum to 1.
        pseudocount: Small value to avoid log(0).

    Returns:
        Log-odds matrix of the same shape.
    """
    bg = 0.25  # uniform background
    pwm_safe = pwm + pseudocount
    pwm_safe = pwm_safe / pwm_safe.sum(axis=0, keepdims=True)
    return np.log2(pwm_safe / bg)


def _sequence_to_onehot(sequence: str) -> np.ndarray:
    """Encode a DNA string as a one-hot matrix of shape ``(4, L)``.

    Row order: A=0, C=1, G=2, T=3.  Non-ACGT characters map to zeros.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    length = len(sequence)
    onehot = np.zeros((4, length), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in mapping:
            onehot[mapping[base], i] = 1.0
    return onehot


def scan_sequence_with_pwm(
    sequence: str,
    pwm: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[List[int], List[float]]:
    """Scan a DNA sequence with a PWM and return matching positions.

    Args:
        sequence: DNA string.
        pwm: Position weight matrix, shape ``(4, motif_len)``.
        threshold: Minimum log-odds score to report a hit.

    Returns:
        Tuple of (positions, scores) for all hits above threshold.
    """
    log_odds = _pwm_to_log_odds(pwm)
    onehot = _sequence_to_onehot(sequence)
    motif_len = log_odds.shape[1]
    seq_len = onehot.shape[1]

    if seq_len < motif_len:
        return [], []

    positions: List[int] = []
    scores: List[float] = []
    for i in range(seq_len - motif_len + 1):
        window = onehot[:, i : i + motif_len]
        score = float((window * log_odds).sum())
        if score >= threshold:
            positions.append(i)
            scores.append(score)

    return positions, scores


# ---------------------------------------------------------------------------
# TF Binding Predictor
# ---------------------------------------------------------------------------


class TFBindingPredictor(nn.Module):
    """Predict transcription factor binding sites from DNA sequence.

    Uses encoder embeddings followed by a per-position multi-label
    classification head to predict which TFs bind at each position.

    Args:
        encoder: Pretrained Genova encoder that accepts ``input_ids``
            and returns a dict with ``"last_hidden_state"`` of shape
            ``(B, L, D)``.
        num_tfs: Number of transcription factors to predict.
        d_model: Hidden dimension of the encoder output.
        dropout: Dropout probability for the classification head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_tfs: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_tfs = num_tfs
        self.d_model = d_model

        # Per-position multi-label classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_tfs),
        )

        # Optional TF name registry
        self._tf_names: List[str] = [f"TF_{i}" for i in range(num_tfs)]

    @property
    def tf_names(self) -> List[str]:
        """List of transcription factor names."""
        return self._tf_names

    @tf_names.setter
    def tf_names(self, names: List[str]) -> None:
        if len(names) != self.num_tfs:
            raise ValueError(
                f"Expected {self.num_tfs} TF names, got {len(names)}"
            )
        self._tf_names = list(names)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass returning raw logits.

        Args:
            input_ids: Token IDs of shape ``(B, L)``.
            attention_mask: Optional mask of shape ``(B, L)``.

        Returns:
            Logits tensor of shape ``(B, L, num_tfs)``.
        """
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(enc_out, dict):
            hidden = enc_out.get("last_hidden_state", enc_out.get("hidden_states"))
        elif isinstance(enc_out, Tensor):
            hidden = enc_out
        else:
            hidden = getattr(enc_out, "last_hidden_state", enc_out)

        if isinstance(hidden, (list, tuple)):
            hidden = hidden[-1]

        return self.head(hidden)  # (B, L, num_tfs)

    @torch.no_grad()
    def predict(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        threshold: float = 0.5,
    ) -> List[TFBindingPrediction]:
        """Predict TF binding sites for a single sequence.

        Args:
            input_ids: Token IDs of shape ``(1, L)`` or ``(L,)``.
            attention_mask: Optional mask.
            threshold: Probability threshold for calling a binding site.

        Returns:
            List of :class:`TFBindingPrediction`, one per TF that has
            at least one binding site above the threshold.
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        logits = self.forward(input_ids, attention_mask)  # (1, L, num_tfs)
        probs = torch.sigmoid(logits[0])  # (L, num_tfs)

        predictions: List[TFBindingPrediction] = []
        for tf_idx in range(self.num_tfs):
            tf_probs = probs[:, tf_idx]
            mask = tf_probs >= threshold
            if mask.any():
                positions = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(positions, int):
                    positions = [positions]
                scores = tf_probs[mask].tolist()
                predictions.append(
                    TFBindingPrediction(
                        tf_name=self._tf_names[tf_idx],
                        positions=positions,
                        scores=scores,
                    )
                )

        return predictions

    @torch.no_grad()
    def predict_batch(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        threshold: float = 0.5,
    ) -> List[List[TFBindingPrediction]]:
        """Predict TF binding sites for a batch of sequences.

        Args:
            input_ids: Token IDs of shape ``(B, L)``.
            attention_mask: Optional mask of shape ``(B, L)``.
            threshold: Probability threshold.

        Returns:
            List of length B, each containing a list of
            :class:`TFBindingPrediction` for that sequence.
        """
        self.eval()
        logits = self.forward(input_ids, attention_mask)  # (B, L, num_tfs)
        probs = torch.sigmoid(logits)  # (B, L, num_tfs)

        batch_results: List[List[TFBindingPrediction]] = []
        for b in range(probs.size(0)):
            sample_preds: List[TFBindingPrediction] = []
            for tf_idx in range(self.num_tfs):
                tf_probs = probs[b, :, tf_idx]
                mask = tf_probs >= threshold
                if mask.any():
                    positions = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                    if isinstance(positions, int):
                        positions = [positions]
                    scores = tf_probs[mask].tolist()
                    sample_preds.append(
                        TFBindingPrediction(
                            tf_name=self._tf_names[tf_idx],
                            positions=positions,
                            scores=scores,
                        )
                    )
            batch_results.append(sample_preds)

        return batch_results

    @staticmethod
    def compare_with_pwm(
        predicted_sites: List[TFBindingPrediction],
        jaspar_motifs: List[JASPARMotif],
        sequence: Optional[str] = None,
        pwm_threshold: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare predicted binding sites with JASPAR PWM motifs.

        For each predicted TF, finds the best-matching JASPAR motif
        (by name) and computes concordance between predicted positions
        and PWM-scanned positions.

        Args:
            predicted_sites: List of TF binding predictions.
            jaspar_motifs: List of JASPAR motifs to compare against.
            sequence: DNA sequence string (required if PWM scanning is
                desired; if ``None`` only name matching is performed).
            pwm_threshold: Log-odds threshold for PWM scanning.

        Returns:
            Dictionary mapping TF name to concordance metrics:
            ``{"matched_motif", "overlap_count", "predicted_count",
            "pwm_count", "jaccard"}``.
        """
        motif_by_name: Dict[str, JASPARMotif] = {}
        for m in jaspar_motifs:
            motif_by_name[m.tf_name.upper()] = m
            motif_by_name[m.matrix_id.upper()] = m

        results: Dict[str, Dict[str, Any]] = {}
        for pred in predicted_sites:
            tf_key = pred.tf_name.upper()
            motif = motif_by_name.get(tf_key)

            if motif is None:
                results[pred.tf_name] = {
                    "matched_motif": None,
                    "overlap_count": 0,
                    "predicted_count": len(pred.positions),
                    "pwm_count": 0,
                    "jaccard": 0.0,
                }
                continue

            # Update motif string on prediction
            pred.motif = motif.consensus

            if sequence is not None:
                pwm_positions, _ = scan_sequence_with_pwm(
                    sequence, motif.pwm, threshold=pwm_threshold
                )
                pred_set = set(pred.positions)
                pwm_set = set(pwm_positions)
                overlap = len(pred_set & pwm_set)
                union = len(pred_set | pwm_set)
                jaccard = overlap / max(union, 1)

                results[pred.tf_name] = {
                    "matched_motif": motif.matrix_id,
                    "overlap_count": overlap,
                    "predicted_count": len(pred.positions),
                    "pwm_count": len(pwm_positions),
                    "jaccard": jaccard,
                }
            else:
                results[pred.tf_name] = {
                    "matched_motif": motif.matrix_id,
                    "overlap_count": 0,
                    "predicted_count": len(pred.positions),
                    "pwm_count": 0,
                    "jaccard": 0.0,
                }

        return results

    def compute_loss(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-label binary cross entropy loss.

        Args:
            input_ids: Token IDs, shape ``(B, L)``.
            labels: Binary labels, shape ``(B, L, num_tfs)``.
            attention_mask: Optional mask, shape ``(B, L)``.
            pos_weight: Optional per-TF positive class weight, shape
                ``(num_tfs,)``.

        Returns:
            Scalar loss tensor.
        """
        logits = self.forward(input_ids, attention_mask)  # (B, L, num_tfs)
        loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight
        )

        # Mask out padding positions if attention_mask provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            per_pos_loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight, reduction="none"
            )
            loss = (per_pos_loss * mask).sum() / mask.sum().clamp(min=1.0) / self.num_tfs

        return loss
