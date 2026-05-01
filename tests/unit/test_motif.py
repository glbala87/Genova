"""Tests for motif discovery and clustering modules."""

from __future__ import annotations

import numpy as np
import pytest

from genova.motif.motif_discovery import (
    sequences_to_pwm,
    pwm_to_consensus,
    score_sequence_with_pwm,
)
from genova.motif.motif_clustering import MotifClusterer
from genova.motif.motif_discovery import Motif


# ---------------------------------------------------------------------------
# PWM construction edge cases
# ---------------------------------------------------------------------------


class TestPWMEdgeCases:

    def test_single_sequence(self):
        """PWM from a single sequence should have ~1.0 at correct positions."""
        pwm = sequences_to_pwm(["ACGT"], pseudocount=0.0001)
        assert pwm[0, 0] > 0.99  # A
        assert pwm[1, 1] > 0.99  # C
        assert pwm[2, 2] > 0.99  # G
        assert pwm[3, 3] > 0.99  # T

    def test_mixed_sequences(self):
        """PWM from mixed sequences should have intermediate values."""
        pwm = sequences_to_pwm(["AAAA", "CCCC"], pseudocount=0.01)
        # Position 0: half A, half C
        assert pwm[0, 0] > 0.4  # A fraction
        assert pwm[0, 1] > 0.4  # C fraction

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sequences_to_pwm([])

    def test_inconsistent_lengths_raises(self):
        with pytest.raises(ValueError):
            sequences_to_pwm(["ATCG", "AT"])

    def test_columns_sum_to_one(self):
        seqs = ["ATCG", "GCTA", "AAAA", "CCCC", "TTTT"]
        pwm = sequences_to_pwm(seqs, pseudocount=0.01)
        for i in range(pwm.shape[0]):
            assert abs(pwm[i].sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


class TestPWMConsensus:

    def test_consensus_of_identical(self):
        pwm = sequences_to_pwm(["ATCG"] * 10, pseudocount=0.0001)
        assert pwm_to_consensus(pwm) == "ATCG"

    def test_consensus_of_single(self):
        pwm = sequences_to_pwm(["GCTA"], pseudocount=0.0001)
        assert pwm_to_consensus(pwm) == "GCTA"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestPWMScoring:

    def test_consensus_score_positive(self):
        seqs = ["ATCG"] * 20
        pwm = sequences_to_pwm(seqs, pseudocount=0.01)
        consensus = pwm_to_consensus(pwm)
        score = score_sequence_with_pwm(consensus, pwm)
        assert score > 0

    def test_random_sequence_lower_score(self):
        seqs = ["ATCG"] * 20
        pwm = sequences_to_pwm(seqs, pseudocount=0.01)
        consensus_score = score_sequence_with_pwm("ATCG", pwm)
        random_score = score_sequence_with_pwm("GGGG", pwm)
        assert consensus_score > random_score


# ---------------------------------------------------------------------------
# MotifClusterer
# ---------------------------------------------------------------------------


class TestMotifClusterer:

    def _make_motif(self, seq: str, pwm=None) -> Motif:
        if pwm is None:
            pwm = sequences_to_pwm([seq] * 5, pseudocount=0.01)
        return Motif(sequence=seq, pwm=pwm, score=1.0)

    def test_creation(self):
        motifs = [self._make_motif("ATCG"), self._make_motif("GCTA")]
        clusterer = MotifClusterer(motifs)
        assert clusterer is not None

    def test_cluster_identical_motifs(self):
        """Identical motifs should cluster together."""
        motifs = [
            self._make_motif("ATCG"),
            self._make_motif("ATCG"),
            self._make_motif("GGCC"),
        ]
        clusterer = MotifClusterer(motifs)
        clusterer.cluster(threshold=0.8)
        assert len(clusterer.clusters) >= 1
        assert len(clusterer.clusters) <= 3
