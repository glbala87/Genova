"""Tests for generative evaluation (GC content, repeats, k-mer frequency)."""

from __future__ import annotations

import pytest
import numpy as np

from genova.generative.evaluation import (
    _gc_content,
    _count_nucleotides,
    _kmer_frequencies,
    _detect_trivial_repeats,
    _jensen_shannon_divergence,
    GenerationEvaluator,
)


# ---------------------------------------------------------------------------
# GC content
# ---------------------------------------------------------------------------

class TestGCContent:

    @pytest.mark.parametrize("seq,expected", [
        ("GGCC", 1.0),
        ("AATT", 0.0),
        ("ACGT", 0.5),
        ("GC", 1.0),
        ("AT", 0.0),
    ])
    def test_known_values(self, seq, expected):
        assert _gc_content(seq) == pytest.approx(expected)

    def test_case_insensitive(self):
        assert _gc_content("gcgc") == _gc_content("GCGC")

    def test_empty(self):
        assert _gc_content("") == pytest.approx(0.0)

    def test_with_n(self):
        # N counts in total length but not in G+C
        gc = _gc_content("GCNN")
        assert gc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Nucleotide counting
# ---------------------------------------------------------------------------

class TestCountNucleotides:

    def test_simple(self):
        counts = _count_nucleotides("AACCGGTT")
        assert counts["A"] == 2
        assert counts["C"] == 2
        assert counts["G"] == 2
        assert counts["T"] == 2
        assert counts["N"] == 0

    def test_with_n(self):
        counts = _count_nucleotides("ANN")
        assert counts["A"] == 1
        assert counts["N"] == 2


# ---------------------------------------------------------------------------
# K-mer frequencies
# ---------------------------------------------------------------------------

class TestKmerFrequencies:

    def test_simple(self):
        freqs = _kmer_frequencies("ACGT", k=2)
        assert "AC" in freqs
        assert "CG" in freqs
        assert "GT" in freqs
        # 3 k-mers total, each appears once -> 1/3
        assert freqs["AC"] == pytest.approx(1 / 3)

    def test_skips_n_kmers(self):
        freqs = _kmer_frequencies("ANGC", k=2)
        # AN and NG contain N, only GC is counted
        assert "GC" in freqs

    def test_empty_sequence(self):
        freqs = _kmer_frequencies("", k=3)
        assert freqs == {}

    def test_seq_shorter_than_k(self):
        freqs = _kmer_frequencies("AC", k=3)
        assert freqs == {}


# ---------------------------------------------------------------------------
# Trivial repeat detection
# ---------------------------------------------------------------------------

class TestTrivialRepeats:

    def test_homopolymer(self):
        result = _detect_trivial_repeats("AAAAAAAAAA")
        assert result["is_trivial"] is True
        assert result["fraction"] == pytest.approx(1.0)

    def test_dinucleotide_repeat(self):
        result = _detect_trivial_repeats("ACACACACAC")
        assert result["is_trivial"] is True

    def test_non_trivial(self):
        result = _detect_trivial_repeats("ACGTACGTNN", min_fraction=0.95)
        assert result["is_trivial"] is False

    def test_short_sequence(self):
        result = _detect_trivial_repeats("A")
        assert result["is_trivial"] is True
        assert result["fraction"] == 1.0


# ---------------------------------------------------------------------------
# Jensen-Shannon divergence
# ---------------------------------------------------------------------------

class TestJSD:

    def test_identical_distributions(self):
        p = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
        jsd = _jensen_shannon_divergence(p, p)
        assert jsd == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions(self):
        p = {"A": 1.0}
        q = {"T": 1.0}
        jsd = _jensen_shannon_divergence(p, q)
        assert jsd > 0

    def test_symmetric(self):
        p = {"A": 0.6, "C": 0.4}
        q = {"A": 0.3, "C": 0.7}
        assert _jensen_shannon_divergence(p, q) == pytest.approx(
            _jensen_shannon_divergence(q, p), abs=1e-10
        )


# ---------------------------------------------------------------------------
# GenerationEvaluator
# ---------------------------------------------------------------------------

class TestGenerationEvaluator:

    @pytest.fixture
    def evaluator(self):
        ref = ["ACGTACGTACGT", "GGCCAATTGGCC", "ACGTNNACGTNN"]
        return GenerationEvaluator(reference_sequences=ref, kmer_sizes=[3])

    @pytest.fixture
    def evaluator_no_ref(self):
        return GenerationEvaluator()

    def test_gc_content_analysis(self, evaluator):
        seqs = ["GGCC", "AATT", "ACGT"]
        result = evaluator.gc_content_analysis(seqs)
        assert "mean" in result
        assert "std" in result
        assert 0.0 <= result["mean"] <= 1.0

    def test_nucleotide_composition(self, evaluator):
        seqs = ["AACCGGTT"]
        result = evaluator.nucleotide_composition(seqs)
        assert result["fractions"]["A"] == pytest.approx(0.25)
        assert result["validity_rate"] == 1.0

    def test_nucleotide_composition_l1(self, evaluator):
        seqs = ["AACCGGTT"]
        result = evaluator.nucleotide_composition(seqs)
        assert "composition_l1_distance" in result

    def test_kmer_analysis(self, evaluator):
        seqs = ["ACGTACGTACGT"]
        result = evaluator.kmer_analysis(seqs, k=3)
        assert "3mer" in result
        assert result["3mer"]["num_unique_kmers"] > 0
        assert "jsd" in result["3mer"]

    def test_trivial_repeat_analysis(self, evaluator_no_ref):
        seqs = ["AAAAAAAAAA", "ACGTACGTNN"]
        result = evaluator_no_ref.trivial_repeat_analysis(seqs)
        assert result["trivial_count"] >= 1
        assert 0.0 <= result["trivial_rate"] <= 1.0

    def test_motif_enrichment(self, evaluator_no_ref):
        seqs = ["TATAAAT" * 10, "ACGTACGT"]
        result = evaluator_no_ref.motif_enrichment(seqs)
        assert "TATA_box" in result
        assert result["TATA_box"]["total_hits"] > 0

    def test_evaluate_summary(self, evaluator):
        seqs = ["ACGTACGT", "GGCCAATT"]
        summary = evaluator.evaluate(seqs)
        assert "num_sequences" in summary
        assert "gc_content_mean" in summary
        assert "trivial_repeat_rate" in summary
        assert summary["num_sequences"] == 2

    def test_compute_all_metrics(self, evaluator):
        seqs = ["ACGTACGTACGT", "GGCCAATTGGCC"]
        result = evaluator.compute_all_metrics(seqs)
        assert "summary" in result
        assert "nucleotide_composition" in result
        assert "gc_content" in result
        assert "kmer_analysis" in result
        assert "motif_enrichment" in result
        assert "trivial_repeats" in result

    def test_no_reference(self, evaluator_no_ref):
        seqs = ["ACGTACGT"]
        result = evaluator_no_ref.nucleotide_composition(seqs)
        assert "composition_l1_distance" not in result

    def test_invalid_characters(self, evaluator_no_ref):
        seqs = ["ACGTXYZ"]
        result = evaluator_no_ref.nucleotide_composition(seqs)
        assert result["invalid_sequences"] == 1
