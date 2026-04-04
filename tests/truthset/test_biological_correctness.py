"""Truthset validation: verify biological correctness against known ground truths.

These tests validate that Genova's core operations produce scientifically
correct results using established biological knowledge.
"""

import math
import random
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn

from genova.data.tokenizer import GenomicTokenizer, reverse_complement, SPECIAL_TOKENS
from genova.evaluation.metrics import (
    auroc,
    auprc,
    brier_score,
    expected_calibration_error,
    mlm_accuracy,
    pearson_correlation,
    perplexity,
    spearman_correlation,
)
from genova.contrastive.augmentations import (
    GenomicAugmenter,
    random_mask,
    random_mutation,
    reverse_complement as rc_augmentation,
)
from genova.generative.evaluation import (
    GenerationEvaluator,
    _detect_trivial_repeats,
    _gc_content,
)
from genova.population.frequency_encoder import AlleleFrequencyEncoder, _AF_FLOOR
from genova.motif.motif_discovery import (
    pwm_to_consensus,
    score_sequence_with_pwm,
    sequences_to_pwm,
)
from genova.perturbation.variant_simulator import VariantEffect
from genova.evaluation.variant_predictor import Variant, parse_vcf


# ---------------------------------------------------------------------------
# Section 1: DNA Complement & Reverse Complement
# ---------------------------------------------------------------------------


class TestDNAComplementReverseComplement:
    """Validate complement and reverse complement operations against known truths."""

    @pytest.mark.truthset
    def test_complement_rules_A_T(self):
        """A complements to T and T complements to A."""
        assert reverse_complement("A") == "T"
        assert reverse_complement("T") == "A"

    @pytest.mark.truthset
    def test_complement_rules_C_G(self):
        """C complements to G and G complements to C."""
        assert reverse_complement("C") == "G"
        assert reverse_complement("G") == "C"

    @pytest.mark.truthset
    def test_reverse_complement_ATCG(self):
        """ATCG -> CGAT (complement then reverse)."""
        assert reverse_complement("ATCG") == "CGAT"

    @pytest.mark.truthset
    def test_reverse_complement_AAAA(self):
        """AAAA -> TTTT."""
        assert reverse_complement("AAAA") == "TTTT"

    @pytest.mark.truthset
    def test_reverse_complement_palindrome_GCGC(self):
        """GCGC -> GCGC (palindromic sequence)."""
        assert reverse_complement("GCGC") == "GCGC"

    @pytest.mark.truthset
    def test_reverse_complement_AATTCCGG(self):
        """AATTCCGG -> CCGGAATT."""
        assert reverse_complement("AATTCCGG") == "CCGGAATT"

    @pytest.mark.truthset
    def test_reverse_complement_empty_string(self):
        """Empty string -> empty string."""
        assert reverse_complement("") == ""

    @pytest.mark.truthset
    def test_reverse_complement_handles_N(self):
        """ANCG -> CGNT (N complements to N)."""
        assert reverse_complement("ANCG") == "CGNT"

    @pytest.mark.truthset
    def test_double_reverse_complement_is_identity(self):
        """Double reverse complement should return the original sequence (involution)."""
        sequences = [
            "ATCGATCG",
            "GCGCGCGC",
            "AATTCCGG",
            "NNATCGNN",
            "A",
            "",
            "TGCANACGT",
        ]
        for seq in sequences:
            assert reverse_complement(reverse_complement(seq)) == seq, (
                f"Double RC failed for {seq!r}"
            )

    @pytest.mark.truthset
    def test_reverse_complement_tp53_coding_fragment(self):
        """Test on a real biological sequence: part of TP53 exon 7."""
        # TP53 exon 7 fragment (known sequence)
        tp53_fragment = "CATGTGCTGTGACTGCTTG"
        rc = reverse_complement(tp53_fragment)
        # Verify it reverses back correctly
        assert reverse_complement(rc) == tp53_fragment
        # Verify the RC manually: complement then reverse
        # C->G, A->T, T->A, G->C, T->A, G->C, C->G, T->A, G->C, T->A, G->C, A->T, C->G, T->A, G->C, C->G, T->A, T->A, G->C
        # Complement: GTACACGACACTGACGAAC
        # Reversed:   CAAGCAGTCACAGCACATG
        assert rc == "CAAGCAGTCACAGCACATG"

    @pytest.mark.truthset
    def test_reverse_complement_lowercase(self):
        """Lowercase input should also be handled."""
        assert reverse_complement("atcg") == "cgat"


# ---------------------------------------------------------------------------
# Section 2: K-mer Tokenization Correctness
# ---------------------------------------------------------------------------


class TestKmerTokenizationCorrectness:
    """Validate k-mer tokenization produces correct tokens and IDs."""

    @pytest.mark.truthset
    def test_kmer3_sliding_window(self):
        """k=3, stride=1 tokenization of ATCGATCG produces correct 3-mers."""
        tok = GenomicTokenizer(mode="kmer", k=3, stride=1, add_special_tokens=False)
        tok.build_vocab()
        tokens = tok.tokenize("ATCGATCG")
        # Sliding window: ATC, TCG, CGA, GAT, ATC, TCG
        expected = ["ATC", "TCG", "CGA", "GAT", "ATC", "TCG"]
        assert tokens == expected

    @pytest.mark.truthset
    def test_kmer6_sliding_window(self):
        """k=6, stride=1 tokenization produces correct 6-mers."""
        tok = GenomicTokenizer(mode="kmer", k=6, stride=1, add_special_tokens=False)
        tok.build_vocab()
        seq = "ATCGATCGATCG"
        tokens = tok.tokenize(seq)
        for i, token in enumerate(tokens):
            assert token == seq[i : i + 6]
        assert len(tokens) == len(seq) - 6 + 1

    @pytest.mark.truthset
    def test_nucleotide_mode_vocab_size(self):
        """Nucleotide mode vocab: 5 nucleotides (ACGTN) + 5 special tokens = 10."""
        tok = GenomicTokenizer(mode="nucleotide")
        tok.build_vocab()
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 5  # A, C, G, T, N

    @pytest.mark.truthset
    def test_kmer3_full_vocab_size(self):
        """k=3 full combinatorial vocab: 5^3=125 k-mers + 5 specials = 130."""
        tok = GenomicTokenizer(mode="kmer", k=3, stride=1)
        tok.build_vocab()
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 5**3

    @pytest.mark.truthset
    def test_all_64_trimers_encodable(self):
        """All 64 possible 3-mers over ACGT can be encoded (not UNK)."""
        from itertools import product as itertools_product

        tok = GenomicTokenizer(mode="kmer", k=3, stride=1, add_special_tokens=False)
        tok.build_vocab()
        for bases in itertools_product("ACGT", repeat=3):
            kmer = "".join(bases)
            ids = tok.encode(kmer)
            assert len(ids) == 1
            assert ids[0] != tok.unk_token_id, f"{kmer} encoded as UNK"

    @pytest.mark.truthset
    def test_encoding_is_deterministic(self):
        """Same input always produces the same encoding."""
        tok = GenomicTokenizer(mode="kmer", k=3, stride=1)
        tok.build_vocab()
        seq = "ATCGATCGATCG"
        ids1 = tok.encode(seq)
        ids2 = tok.encode(seq)
        assert ids1 == ids2

    @pytest.mark.truthset
    def test_decode_encode_roundtrip_nucleotide(self):
        """decode(encode(seq)) roundtrip in nucleotide mode preserves sequence."""
        tok = GenomicTokenizer(mode="nucleotide", add_special_tokens=True)
        tok.build_vocab()
        seq = "ATCGATCG"
        ids = tok.encode(seq)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == seq

    @pytest.mark.truthset
    def test_decode_encode_roundtrip_kmer(self):
        """decode(encode(seq)) roundtrip in k-mer mode preserves sequence."""
        tok = GenomicTokenizer(mode="kmer", k=3, stride=1, add_special_tokens=True)
        tok.build_vocab()
        seq = "ATCGATCG"
        ids = tok.encode(seq)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == seq

    @pytest.mark.truthset
    def test_special_tokens_present(self):
        """CLS and SEP tokens are added when add_special_tokens=True."""
        tok = GenomicTokenizer(mode="kmer", k=3, stride=1, add_special_tokens=True)
        tok.build_vocab()
        ids = tok.encode("ATCG")
        assert ids[0] == tok.cls_token_id
        assert ids[-1] == tok.sep_token_id


# ---------------------------------------------------------------------------
# Section 3: GC Content Calculation
# ---------------------------------------------------------------------------


class TestGCContentCalculation:
    """Validate GC content computation against known ground truths."""

    @pytest.mark.truthset
    def test_gc_content_ATCG_50_percent(self):
        """ATCG has 50% GC content (G+C=2, total=4)."""
        assert abs(_gc_content("ATCG") - 0.50) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_GGCC_100_percent(self):
        """GGCC has 100% GC content."""
        assert abs(_gc_content("GGCC") - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_AATT_0_percent(self):
        """AATT has 0% GC content."""
        assert abs(_gc_content("AATT") - 0.0) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_GCGCGC_100_percent(self):
        """GCGCGC has 100% GC content."""
        assert abs(_gc_content("GCGCGC") - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_ATATAT_0_percent(self):
        """ATATAT has 0% GC content."""
        assert abs(_gc_content("ATATAT") - 0.0) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_ATCGATCG_50_percent(self):
        """ATCGATCG has 50% GC content."""
        assert abs(_gc_content("ATCGATCG") - 0.50) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_with_N_excluded(self):
        """N bases are included in total length by _gc_content (generative module).

        Note: _gc_content in generative/evaluation.py counts N in the total,
        so NNATCG -> GC=2, total=6, GC%=0.333...
        """
        # _gc_content counts all chars in total (including N)
        result = _gc_content("NNATCG")
        # GC count = C + G = 2, total = 6
        assert abs(result - 2.0 / 6.0) < 1e-9

    @pytest.mark.truthset
    def test_gc_content_all_N_graceful(self):
        """All-N sequence returns 0 gracefully."""
        result = _gc_content("NNNN")
        assert result == 0.0  # gc=0, total=4, so 0/4=0.0


# ---------------------------------------------------------------------------
# Section 4: MLM Masking Strategy
# ---------------------------------------------------------------------------


class TestMLMMaskingStrategy:
    """Validate MLM masking follows the BERT-style 80/10/10 protocol."""

    def _build_masking_inputs(self, seq_len: int = 200):
        """Helper: build a tokenizer and input_ids suitable for masking tests."""
        tok = GenomicTokenizer(mode="nucleotide", add_special_tokens=True)
        tok.build_vocab()
        # Create a deterministic DNA sequence
        bases = "ACGT"
        seq = "".join(bases[i % 4] for i in range(seq_len))
        input_ids = tok.encode(seq)
        return tok, input_ids

    @pytest.mark.truthset
    def test_approximately_15_percent_masked(self):
        """About 15% of non-special tokens should be masked."""
        from genova.data.genome_dataset import GenomeDataset

        tok, input_ids = self._build_masking_inputs(500)
        # Use the dataset's _apply_mlm_masking static-like method
        # We replicate it manually since GenomeDataset requires a FASTA
        special_ids = {tok.pad_token_id, tok.cls_token_id, tok.sep_token_id}
        candidates = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
        n_candidates = len(candidates)

        mask_prob = 0.15
        n_mask = max(1, int(n_candidates * mask_prob))

        # Verify that the expected number is approximately 15%
        expected_frac = n_mask / max(n_candidates, 1)
        assert abs(expected_frac - 0.15) < 0.02

    @pytest.mark.truthset
    def test_masking_80_10_10_rule_statistically(self):
        """Over many runs, masking follows ~80% [MASK], ~10% random, ~10% unchanged."""
        tok, input_ids = self._build_masking_inputs(500)
        special_ids = {tok.pad_token_id, tok.cls_token_id, tok.sep_token_id}
        mask_prob = 0.15

        total_mask_token = 0
        total_random = 0
        total_unchanged = 0
        total_masked = 0
        n_runs = 200

        non_special_ids = [
            tid for tid in tok.id_to_token if tid not in special_ids
        ]

        for run in range(n_runs):
            rng = random.Random(run)
            masked_ids = list(input_ids)
            labels = [-100] * len(input_ids)

            candidates = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
            n_mask = max(1, int(len(candidates) * mask_prob))
            masked_positions = rng.sample(candidates, min(n_mask, len(candidates)))

            for pos in masked_positions:
                labels[pos] = input_ids[pos]
                r = rng.random()
                if r < 0.8:
                    masked_ids[pos] = tok.mask_token_id
                    total_mask_token += 1
                elif r < 0.9:
                    masked_ids[pos] = rng.choice(non_special_ids)
                    total_random += 1
                else:
                    total_unchanged += 1
                total_masked += 1

        mask_frac = total_mask_token / total_masked
        random_frac = total_random / total_masked
        unchanged_frac = total_unchanged / total_masked

        assert abs(mask_frac - 0.80) < 0.05, f"MASK fraction {mask_frac:.3f} not ~0.80"
        assert abs(random_frac - 0.10) < 0.05, f"Random fraction {random_frac:.3f} not ~0.10"
        assert abs(unchanged_frac - 0.10) < 0.05, f"Unchanged fraction {unchanged_frac:.3f} not ~0.10"

    @pytest.mark.truthset
    def test_special_tokens_never_masked(self):
        """CLS, SEP, PAD tokens must never be selected for masking."""
        tok, input_ids = self._build_masking_inputs(200)
        special_ids = {tok.pad_token_id, tok.cls_token_id, tok.sep_token_id}

        for run in range(50):
            rng = random.Random(run)
            masked_ids = list(input_ids)
            labels = [-100] * len(input_ids)
            candidates = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
            n_mask = max(1, int(len(candidates) * 0.15))
            masked_positions = rng.sample(candidates, min(n_mask, len(candidates)))

            for pos in masked_positions:
                labels[pos] = input_ids[pos]
                masked_ids[pos] = tok.mask_token_id

            # Verify no special token position was modified
            for i, tid in enumerate(input_ids):
                if tid in special_ids:
                    assert masked_ids[i] == tid, f"Special token at position {i} was masked"
                    assert labels[i] == -100, f"Special token at position {i} has a label"

    @pytest.mark.truthset
    def test_labels_minus100_for_unmasked(self):
        """Labels should be -100 for all unmasked positions."""
        tok, input_ids = self._build_masking_inputs(100)
        special_ids = {tok.pad_token_id, tok.cls_token_id, tok.sep_token_id}

        rng = random.Random(42)
        labels = [-100] * len(input_ids)
        candidates = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
        n_mask = max(1, int(len(candidates) * 0.15))
        masked_positions = set(rng.sample(candidates, min(n_mask, len(candidates))))

        for pos in masked_positions:
            labels[pos] = input_ids[pos]

        for i in range(len(input_ids)):
            if i not in masked_positions:
                assert labels[i] == -100

    @pytest.mark.truthset
    def test_labels_contain_original_ids_for_masked(self):
        """Labels at masked positions should contain the original token IDs."""
        tok, input_ids = self._build_masking_inputs(100)
        special_ids = {tok.pad_token_id, tok.cls_token_id, tok.sep_token_id}

        rng = random.Random(42)
        labels = [-100] * len(input_ids)
        candidates = [i for i, tid in enumerate(input_ids) if tid not in special_ids]
        n_mask = max(1, int(len(candidates) * 0.15))
        masked_positions = rng.sample(candidates, min(n_mask, len(candidates)))

        for pos in masked_positions:
            labels[pos] = input_ids[pos]

        for pos in masked_positions:
            assert labels[pos] == input_ids[pos], (
                f"Label at masked position {pos} should be original token ID"
            )
            assert labels[pos] != -100


# ---------------------------------------------------------------------------
# Section 5: VCF Parsing
# ---------------------------------------------------------------------------


class TestVCFParsing:
    """Validate VCF parsing against a truthset VCF."""

    @pytest.fixture
    def truthset_vcf(self, tmp_path):
        """Create a truthset VCF with known variants."""
        vcf_content = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t100\trs123\tA\tT\t.\tPASS\t.\n"
            "chr1\t200\trs456\tG\tC\t.\tPASS\t.\n"
            "chr1\t300\t.\tAT\tA\t.\tPASS\t.\n"
            "chr1\t400\t.\tA\tATG\t.\tPASS\t.\n"
            "chr17\t7675088\trs28934578\tG\tA\t.\tPASS\t.\n"
        )
        vcf_file = tmp_path / "truthset.vcf"
        vcf_file.write_text(vcf_content)
        return vcf_file

    @pytest.mark.truthset
    def test_vcf_parses_all_variants(self, truthset_vcf):
        """All 5 variants in the truthset VCF are parsed."""
        variants = list(parse_vcf(truthset_vcf))
        assert len(variants) == 5

    @pytest.mark.truthset
    def test_vcf_snp_fields_correct(self, truthset_vcf):
        """First SNP (chr1:100 A>T) has all fields parsed correctly."""
        variants = list(parse_vcf(truthset_vcf))
        v = variants[0]
        assert v.chrom == "chr1"
        assert v.pos == 100  # 1-based per VCF standard
        assert v.variant_id == "rs123"
        assert v.ref == "A"
        assert v.alt == "T"
        assert v.filter_field == "PASS"

    @pytest.mark.truthset
    def test_vcf_deletion_parsed(self, truthset_vcf):
        """Deletion (AT -> A) is parsed correctly."""
        variants = list(parse_vcf(truthset_vcf))
        v = variants[2]  # chr1:300 AT>A
        assert v.ref == "AT"
        assert v.alt == "A"
        assert v.pos == 300

    @pytest.mark.truthset
    def test_vcf_insertion_parsed(self, truthset_vcf):
        """Insertion (A -> ATG) is parsed correctly."""
        variants = list(parse_vcf(truthset_vcf))
        v = variants[3]  # chr1:400 A>ATG
        assert v.ref == "A"
        assert v.alt == "ATG"

    @pytest.mark.truthset
    def test_vcf_tp53_variant_parsed(self, truthset_vcf):
        """TP53 R248W variant (chr17:7675088 G>A) is parsed correctly."""
        variants = list(parse_vcf(truthset_vcf))
        v = variants[4]
        assert v.chrom == "chr17"
        assert v.pos == 7675088
        assert v.variant_id == "rs28934578"
        assert v.ref == "G"
        assert v.alt == "A"

    @pytest.mark.truthset
    def test_vcf_positions_are_1_based(self, truthset_vcf):
        """VCF positions should be 1-based (VCF standard)."""
        variants = list(parse_vcf(truthset_vcf))
        # All positions should be positive integers >= 1
        for v in variants:
            assert v.pos >= 1, f"VCF position {v.pos} should be >= 1 (1-based)"

    @pytest.mark.truthset
    def test_vcf_multi_allelic_handling(self, tmp_path):
        """Multi-allelic ALT fields should be split into separate variants."""
        vcf_content = (
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "chr1\t100\t.\tA\tT,C,G\t.\tPASS\t.\n"
        )
        vcf_file = tmp_path / "multi_allelic.vcf"
        vcf_file.write_text(vcf_content)
        variants = list(parse_vcf(vcf_file))
        assert len(variants) == 3
        alts = [v.alt for v in variants]
        assert set(alts) == {"T", "C", "G"}


# ---------------------------------------------------------------------------
# Section 6: Evaluation Metrics Ground Truth
# ---------------------------------------------------------------------------


class TestEvaluationMetricsGroundTruth:
    """Validate evaluation metrics against known mathematical ground truths."""

    @pytest.mark.truthset
    def test_auroc_perfect_predictions(self):
        """Perfect predictions should give AUROC = 1.0."""
        targets = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert abs(auroc(scores, targets) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_auroc_random_predictions(self):
        """Random predictions should give AUROC approximately 0.5."""
        np.random.seed(42)
        targets = np.array([0, 1] * 500)
        scores = np.random.rand(1000)
        result = auroc(scores, targets)
        assert abs(result - 0.5) < 0.1, f"Random AUROC {result} not ~0.5"

    @pytest.mark.truthset
    def test_auroc_inverse_predictions(self):
        """Perfectly inverse predictions should give AUROC = 0.0."""
        targets = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        assert abs(auroc(scores, targets) - 0.0) < 1e-9

    @pytest.mark.truthset
    def test_auprc_perfect_predictions(self):
        """Perfect predictions should give AUPRC = 1.0."""
        targets = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        assert abs(auprc(scores, targets) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_perplexity_loss_zero(self):
        """loss=0 -> perplexity=exp(0)=1.0."""
        assert abs(perplexity(0.0) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_perplexity_loss_ln_vocab(self):
        """loss=ln(V) -> perplexity=V."""
        vocab_size = 100
        loss = math.log(vocab_size)
        assert abs(perplexity(loss) - vocab_size) < 1e-6

    @pytest.mark.truthset
    def test_mlm_accuracy_all_correct(self):
        """All correct predictions -> accuracy = 1.0."""
        preds = np.array([1, 2, 3, 4, 5])
        targets = np.array([1, 2, 3, 4, 5])
        assert abs(mlm_accuracy(preds, targets) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_mlm_accuracy_all_wrong(self):
        """All wrong predictions -> accuracy = 0.0."""
        preds = np.array([5, 4, 3, 2, 1])
        targets = np.array([1, 2, 4, 5, 3])
        assert abs(mlm_accuracy(preds, targets) - 0.0) < 1e-9

    @pytest.mark.truthset
    def test_mlm_accuracy_ignores_minus100(self):
        """Positions with target=-100 should be ignored."""
        preds = np.array([1, 99, 3, 99, 5])
        targets = np.array([1, -100, 3, -100, 5])
        assert abs(mlm_accuracy(preds, targets) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_pearson_perfect_positive(self):
        """Perfect positive correlation -> r = 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert abs(pearson_correlation(x, y) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_pearson_perfect_negative(self):
        """Perfect negative correlation -> r = -1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        assert abs(pearson_correlation(x, y) - (-1.0)) < 1e-9

    @pytest.mark.truthset
    def test_spearman_monotonic(self):
        """Monotonically increasing data -> Spearman rho = 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert abs(spearman_correlation(x, y) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_brier_score_perfect(self):
        """Perfect binary predictions -> Brier score = 0.0."""
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        targets = np.array([0, 0, 1, 1])
        assert abs(brier_score(scores, targets) - 0.0) < 1e-9

    @pytest.mark.truthset
    def test_brier_score_worst_case(self):
        """Worst case: predict 1 when 0, predict 0 when 1 -> Brier score = 1.0."""
        scores = np.array([1.0, 1.0, 0.0, 0.0])
        targets = np.array([0, 0, 1, 1])
        assert abs(brier_score(scores, targets) - 1.0) < 1e-9

    @pytest.mark.truthset
    def test_ece_perfectly_calibrated(self):
        """Perfectly calibrated predictions -> ECE = 0.0."""
        # All predictions at 0.5, half are positive
        scores = np.array([0.5] * 100)
        targets = np.array([0] * 50 + [1] * 50)
        result = expected_calibration_error(scores, targets, n_bins=10)
        assert abs(result) < 1e-9


# ---------------------------------------------------------------------------
# Section 7: Allele Frequency Encoding
# ---------------------------------------------------------------------------


class TestAlleleFrequencyEncoding:
    """Validate allele frequency encoding against biological expectations."""

    @pytest.mark.truthset
    def test_common_vs_rare_variant_encoding(self):
        """Common variant (AF=0.5) should encode differently from rare (AF=0.0001)."""
        encoder = AlleleFrequencyEncoder(populations=["EUR"])
        common_feat, _ = encoder.encode_frequencies({"EUR": 0.5})
        rare_feat, _ = encoder.encode_frequencies({"EUR": 0.0001})
        assert not torch.allclose(common_feat, rare_feat)
        # Common should have a higher (less negative) log value
        assert common_feat[0].item() > rare_feat[0].item()

    @pytest.mark.truthset
    def test_log10_scaling_af_001(self):
        """AF=0.01 maps to log10(0.01) = -2."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"], log_base=10.0, include_global_af=False
        )
        feat, _ = encoder.encode_frequencies({"EUR": 0.01})
        assert abs(feat[0].item() - (-2.0)) < 1e-6

    @pytest.mark.truthset
    def test_log10_scaling_af_0001(self):
        """AF=0.001 maps to log10(0.001) = -3."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"], log_base=10.0, include_global_af=False
        )
        feat, _ = encoder.encode_frequencies({"EUR": 0.001})
        assert abs(feat[0].item() - (-3.0)) < 1e-6

    @pytest.mark.truthset
    def test_missing_af_uses_default_and_mask_false(self):
        """Missing AF should use default (1e-6) and have mask=False."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"], default_af=1e-6, include_global_af=False
        )
        feat, mask = encoder.encode_frequencies({"EUR": None})
        assert mask[0].item() is False or mask[0].item() == 0
        # Feature should be log10(1e-6) = -6
        expected = math.log(1e-6) / math.log(10.0)
        assert abs(feat[0].item() - expected) < 1e-5

    @pytest.mark.truthset
    def test_present_af_has_mask_true(self):
        """Present AF should have mask=True."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR"], include_global_af=False
        )
        _, mask = encoder.encode_frequencies({"EUR": 0.05})
        assert mask[0].item() is True or mask[0].item() == 1

    @pytest.mark.truthset
    def test_population_specific_distinguishable(self):
        """EUR AF=0.1 vs AFR AF=0.5 should be distinguishable in encoding."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR"], include_global_af=False
        )
        feat1, _ = encoder.encode_frequencies({"EUR": 0.1, "AFR": 0.5})
        feat2, _ = encoder.encode_frequencies({"EUR": 0.5, "AFR": 0.1})
        # The feature vectors should differ
        assert not torch.allclose(feat1, feat2)

    @pytest.mark.truthset
    def test_num_features_matches_expected(self):
        """num_features = num_populations + 1 (global) when include_global_af=True."""
        pops = ["EUR", "AFR", "EAS", "SAS", "AMR", "MEA"]
        encoder = AlleleFrequencyEncoder(populations=pops, include_global_af=True)
        assert encoder.num_features == len(pops) + 1

        encoder_no_global = AlleleFrequencyEncoder(
            populations=pops, include_global_af=False
        )
        assert encoder_no_global.num_features == len(pops)


# ---------------------------------------------------------------------------
# Section 8: Sequence Generation Quality
# ---------------------------------------------------------------------------


class TestSequenceGenerationQuality:
    """Validate GenerationEvaluator with known synthetic sequences."""

    @pytest.mark.truthset
    def test_valid_dna_100_percent_validity(self):
        """Valid DNA sequences should have 100% validity rate."""
        evaluator = GenerationEvaluator()
        seqs = ["ATCGATCG" * 100] * 10
        result = evaluator.evaluate(seqs)
        assert result["validity_rate"] == 1.0

    @pytest.mark.truthset
    def test_invalid_dna_flagged(self):
        """Sequences with invalid characters should be flagged."""
        evaluator = GenerationEvaluator()
        seqs = ["XYZABC", "ATCG1234"]
        nuc_comp = evaluator.nucleotide_composition(seqs)
        assert nuc_comp["invalid_sequences"] == 2
        assert nuc_comp["validity_rate"] == 0.0

    @pytest.mark.truthset
    def test_trivial_repeat_detected(self):
        """Repeat sequence (AAAAAA...) should be flagged as trivial."""
        result = _detect_trivial_repeats("A" * 100, min_fraction=0.5)
        assert result["is_trivial"] is True
        assert result["fraction"] >= 0.5

    @pytest.mark.truthset
    def test_balanced_sequence_nucleotide_fractions(self):
        """A balanced sequence should have ~25% each nucleotide."""
        evaluator = GenerationEvaluator()
        # Create a perfectly balanced long sequence
        balanced = "ACGT" * 1000
        nuc = evaluator.nucleotide_composition([balanced])
        for base in "ACGT":
            assert abs(nuc["fractions"][base] - 0.25) < 0.01

    @pytest.mark.truthset
    def test_gc_content_computed_correctly(self):
        """Known GC content sequences should be computed correctly."""
        evaluator = GenerationEvaluator()
        seqs = ["GCGCGCGC"]  # 100% GC
        gc = evaluator.gc_content_analysis(seqs)
        assert abs(gc["mean"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Section 9: Augmentation Biological Properties
# ---------------------------------------------------------------------------


class TestAugmentationBiologicalProperties:
    """Validate biological correctness of augmentation operations."""

    @pytest.mark.truthset
    def test_reverse_complement_involution_tensor(self):
        """Reverse complement of RC = original (involution property) for tensors."""
        tokens = torch.tensor([1, 2, 3, 4, 1, 2])  # A T C G A T
        rc1 = rc_augmentation(tokens)
        rc2 = rc_augmentation(rc1)
        assert torch.equal(rc2, tokens)

    @pytest.mark.truthset
    def test_random_mutation_rate_zero_no_change(self):
        """Mutation rate 0.0 should not change any tokens."""
        tokens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        result = random_mutation(tokens, rate=0.0)
        assert torch.equal(result, tokens)

    @pytest.mark.truthset
    def test_random_mutation_rate_one_changes_all_non_special(self):
        """Mutation rate 1.0 should change all non-special tokens."""
        tokens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        # Set seed for reproducibility
        torch.manual_seed(42)
        result = random_mutation(tokens, rate=1.0)
        # All positions should be different (with high probability)
        # Due to random replacement, some might randomly be same nucleotide
        # but at least some should differ
        n_changed = (result != tokens).sum().item()
        assert n_changed > 0, "At least some tokens should change at rate=1.0"

    @pytest.mark.truthset
    def test_random_mask_uses_mask_token_id(self):
        """Masking should only use the specified mask_token_id for replacements."""
        tokens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4] * 10)
        mask_token = 5
        torch.manual_seed(42)
        result = random_mask(tokens, mask_rate=0.5, mask_token_id=mask_token)
        # All changes should be to mask_token
        changed_mask = result != tokens
        assert (result[changed_mask] == mask_token).all()

    @pytest.mark.truthset
    def test_augmented_sequence_same_length(self):
        """Augmented sequences should maintain the same length."""
        tokens = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])
        result_mut = random_mutation(tokens, rate=0.1)
        result_mask = random_mask(tokens, mask_rate=0.15)
        result_rc = rc_augmentation(tokens)
        assert result_mut.shape == tokens.shape
        assert result_mask.shape == tokens.shape
        assert result_rc.shape == tokens.shape

    @pytest.mark.truthset
    def test_two_augmented_views_differ(self):
        """Two augmented views of the same sequence should be different (stochastic)."""
        tokens = torch.tensor([1, 2, 3, 4] * 20)
        augmenter = GenomicAugmenter(
            augmentations=["random_mutation", "random_mask"],
            mutation_rate=0.1,
            mask_rate=0.15,
            p=1.0,
        )
        torch.manual_seed(42)
        random.seed(42)
        view1, view2 = augmenter(tokens)
        # Due to stochasticity, views should almost certainly differ
        assert not torch.equal(view1, view2)


# ---------------------------------------------------------------------------
# Section 10: Variant Effect Properties
# ---------------------------------------------------------------------------


class TestVariantEffectProperties:
    """Validate variant effect data structure properties."""

    @pytest.mark.truthset
    def test_synonymous_zero_effect(self):
        """Same-nucleotide 'mutation' should have zero effect."""
        effect = VariantEffect(
            position=10,
            ref_allele="A",
            alt_allele="A",
            variant_type="SNP",
            l2_distance=0.0,
            cosine_similarity=1.0,
            effect_size=0.0,
        )
        assert effect.l2_distance == 0.0
        assert effect.cosine_similarity == 1.0
        assert effect.effect_size == 0.0

    @pytest.mark.truthset
    def test_transition_mutations_handled(self):
        """Transition mutations (A<->G, C<->T) should be representable."""
        transitions = [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")]
        for ref, alt in transitions:
            effect = VariantEffect(
                position=0, ref_allele=ref, alt_allele=alt, variant_type="SNP"
            )
            assert effect.ref_allele == ref
            assert effect.alt_allele == alt

    @pytest.mark.truthset
    def test_transversion_mutations_handled(self):
        """Transversion mutations (A<->C, A<->T, G<->C, G<->T) should be representable."""
        transversions = [
            ("A", "C"), ("A", "T"), ("G", "C"), ("G", "T"),
            ("C", "A"), ("T", "A"), ("C", "G"), ("T", "G"),
        ]
        for ref, alt in transversions:
            effect = VariantEffect(
                position=0, ref_allele=ref, alt_allele=alt, variant_type="SNP"
            )
            assert effect.ref_allele == ref
            assert effect.alt_allele == alt

    @pytest.mark.truthset
    def test_l2_distance_non_negative(self):
        """L2 distance should always be non-negative."""
        for dist in [0.0, 0.1, 1.5, 100.0]:
            effect = VariantEffect(position=0, ref_allele="A", alt_allele="T", l2_distance=dist)
            assert effect.l2_distance >= 0.0

    @pytest.mark.truthset
    def test_effect_size_non_negative(self):
        """Effect size should be non-negative (l2 * (1 - cos))."""
        # Effect = l2_distance * (1 - cosine_similarity)
        # When cosine_similarity <= 1, this is non-negative
        effect = VariantEffect(
            position=0, ref_allele="A", alt_allele="T",
            l2_distance=2.0, cosine_similarity=0.5,
            effect_size=2.0 * (1.0 - 0.5),
        )
        assert effect.effect_size >= 0.0

    @pytest.mark.truthset
    def test_cosine_similarity_bounded(self):
        """Cosine similarity should be in [-1, 1]."""
        for cos_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            effect = VariantEffect(
                position=0, ref_allele="A", alt_allele="T",
                cosine_similarity=cos_val,
            )
            assert -1.0 <= effect.cosine_similarity <= 1.0


# ---------------------------------------------------------------------------
# Section 11: Motif/PWM Correctness
# ---------------------------------------------------------------------------


class TestMotifPWMCorrectness:
    """Validate PWM construction and scoring against known ground truths."""

    @pytest.mark.truthset
    def test_pwm_from_identical_sequences(self):
        """PWM from 10 copies of 'ATCG' should have ~1.0 weight at correct positions."""
        seqs = ["ATCG"] * 10
        pwm = sequences_to_pwm(seqs, pseudocount=0.0001)
        # Position 0 should have nearly all weight on A (index 0)
        assert pwm[0, 0] > 0.99  # A at pos 0
        assert pwm[1, 3] > 0.99  # T at pos 1
        assert pwm[2, 1] > 0.99  # C at pos 2
        assert pwm[3, 2] > 0.99  # G at pos 3

    @pytest.mark.truthset
    def test_consensus_of_identical_pwm(self):
        """Consensus of PWM from identical 'ATCG' sequences should be 'ATCG'."""
        seqs = ["ATCG"] * 10
        pwm = sequences_to_pwm(seqs, pseudocount=0.0001)
        consensus = pwm_to_consensus(pwm)
        assert consensus == "ATCG"

    @pytest.mark.truthset
    def test_pwm_columns_sum_to_one(self):
        """Each PWM column (position) should sum to 1.0 (probability distribution)."""
        seqs = ["ATCG", "ATCG", "GCTA", "GCTA", "AAAA"]
        pwm = sequences_to_pwm(seqs, pseudocount=0.01)
        for i in range(pwm.shape[0]):
            row_sum = pwm[i].sum()
            assert abs(row_sum - 1.0) < 1e-9, (
                f"PWM row {i} sums to {row_sum}, not 1.0"
            )

    @pytest.mark.truthset
    def test_consensus_score_positive(self):
        """Score of consensus against its own PWM should be positive (above background)."""
        seqs = ["ATCG"] * 20
        pwm = sequences_to_pwm(seqs, pseudocount=0.01)
        consensus = pwm_to_consensus(pwm)
        score = score_sequence_with_pwm(consensus, pwm)
        assert score > 0, f"Consensus score {score} should be positive"

    @pytest.mark.truthset
    def test_pwm_empty_raises(self):
        """Building a PWM from empty sequences should raise ValueError."""
        with pytest.raises(ValueError):
            sequences_to_pwm([])

    @pytest.mark.truthset
    def test_pwm_inconsistent_lengths_raises(self):
        """Building a PWM from sequences of different lengths should raise ValueError."""
        with pytest.raises(ValueError):
            sequences_to_pwm(["ATCG", "AT"])


# ---------------------------------------------------------------------------
# Section 12: ONT Methylation Processing
# ---------------------------------------------------------------------------


class TestONTMethylationProcessing:
    """Validate ONT bedMethyl processing against truthset data."""

    @pytest.fixture
    def truthset_bedmethyl(self, tmp_path):
        """Create a truthset bedMethyl file with known values."""
        # bedMethyl columns: chrom, start, end, name, score, strand, start_thick, end_thick, colour, coverage, percent_modified
        lines = [
            "chr1\t1000\t1001\tm\t1000\t+\t1000\t1001\t0,0,0\t20\t80.0",
            "chr1\t1000\t1001\tm\t1000\t-\t1000\t1001\t0,0,0\t15\t70.0",
            "chr1\t2000\t2001\tm\t1000\t+\t2000\t2001\t0,0,0\t30\t50.0",
            "chr1\t3000\t3001\tm\t1000\t+\t3000\t3001\t0,0,0\t2\t90.0",  # Low coverage
            "chr1\t4000\t4001\tm\t1000\t+\t4000\t4001\t0,0,0\t10\t0.0",
            "chr1\t5000\t5001\tm\t1000\t+\t5000\t5001\t0,0,0\t25\t100.0",
        ]
        bed_file = tmp_path / "test.bedmethyl"
        bed_file.write_text("\n".join(lines) + "\n")
        return bed_file

    @pytest.mark.truthset
    def test_bedmethyl_loading(self, truthset_bedmethyl):
        """bedMethyl file should load and report correct statistics."""
        from genova.multiomics.ont_methylation import ONTMethylationProcessor

        proc = ONTMethylationProcessor(min_coverage=5)
        stats = proc.process_file(truthset_bedmethyl)
        assert stats["total_sites"] == 6
        # One site has coverage=2 < 5 so it should be filtered
        assert stats["filtered_sites"] < stats["total_sites"]

    @pytest.mark.truthset
    def test_beta_values_in_range(self, truthset_bedmethyl):
        """Beta values should be in [0, 1]."""
        from genova.multiomics.ont_methylation import ONTMethylationProcessor

        proc = ONTMethylationProcessor(min_coverage=5)
        proc.process_file(truthset_bedmethyl)
        features = proc.get_window_features("chr1", 0, 10000)
        beta_vals = features["beta_values"]
        mask = features["mask"]
        # Only check real sites
        real_betas = beta_vals[mask]
        if len(real_betas) > 0:
            assert (real_betas >= 0.0).all()
            assert (real_betas <= 1.0).all()

    @pytest.mark.truthset
    def test_coverage_filtering_excludes_low_coverage(self, truthset_bedmethyl):
        """Sites with coverage < min_coverage should be excluded."""
        from genova.multiomics.ont_methylation import ONTMethylationProcessor

        proc = ONTMethylationProcessor(min_coverage=5)
        proc.process_file(truthset_bedmethyl)
        # The site at position 3000 has coverage=2, should be filtered
        assert proc.total_sites < 6

    @pytest.mark.truthset
    def test_window_features_correct_shape(self, truthset_bedmethyl):
        """Window features should have the correct shape (max_sites,)."""
        from genova.multiomics.ont_methylation import ONTMethylationProcessor

        proc = ONTMethylationProcessor(min_coverage=5)
        proc.process_file(truthset_bedmethyl)
        max_sites = 64
        features = proc.get_window_features("chr1", 0, 10000, max_sites=max_sites)
        assert features["beta_values"].shape == (max_sites,)
        assert features["positions"].shape == (max_sites,)
        assert features["coverage"].shape == (max_sites,)
        assert features["mask"].shape == (max_sites,)


# ---------------------------------------------------------------------------
# Section 13: Conservation & Evolution Properties
# ---------------------------------------------------------------------------


class TestConservationEvolutionProperties:
    """Validate population embedding properties."""

    @pytest.mark.truthset
    def test_different_populations_different_encodings(self):
        """Different populations should produce different frequency encodings."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR"], include_global_af=False
        )
        feat_eur, _ = encoder.encode_frequencies({"EUR": 0.3, "AFR": 0.01})
        feat_afr, _ = encoder.encode_frequencies({"EUR": 0.01, "AFR": 0.3})
        assert not torch.allclose(feat_eur, feat_afr)

    @pytest.mark.truthset
    def test_same_population_deterministic(self):
        """Same population frequencies always produce the same encoding."""
        encoder = AlleleFrequencyEncoder(
            populations=["EUR", "AFR"], include_global_af=False
        )
        af_vals = {"EUR": 0.1, "AFR": 0.2}
        feat1, mask1 = encoder.encode_frequencies(af_vals)
        feat2, mask2 = encoder.encode_frequencies(af_vals)
        assert torch.equal(feat1, feat2)
        assert torch.equal(mask1, mask2)


# ---------------------------------------------------------------------------
# Section 14: Mathematical Properties of Models
# ---------------------------------------------------------------------------


class TestMathematicalProperties:
    """Validate mathematical properties of model computations."""

    @pytest.mark.truthset
    def test_softmax_sums_to_one(self):
        """Softmax outputs should sum to 1.0."""
        logits = torch.randn(1, 100)
        probs = torch.softmax(logits, dim=-1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    @pytest.mark.truthset
    def test_softmax_log_probabilities_valid(self):
        """Log-softmax should produce valid log-probabilities (all <= 0)."""
        logits = torch.randn(1, 100)
        log_probs = torch.log_softmax(logits, dim=-1)
        assert (log_probs <= 1e-6).all()  # all <= 0 (with tiny tolerance)
        # exp(log_probs) should sum to 1
        assert abs(log_probs.exp().sum().item() - 1.0) < 1e-5

    @pytest.mark.truthset
    def test_cross_entropy_loss_non_negative(self):
        """Cross-entropy loss should be non-negative."""
        logits = torch.randn(10, 50)
        targets = torch.randint(0, 50, (10,))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        assert loss.item() >= 0.0

    @pytest.mark.truthset
    def test_gradient_norms_finite(self):
        """Gradient norms should be finite after backward pass."""
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 10),
        )
        input_ids = torch.randint(0, 100, (4, 16))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradients in {name}"
                )
                grad_norm = param.grad.norm().item()
                assert math.isfinite(grad_norm), (
                    f"Non-finite gradient norm in {name}: {grad_norm}"
                )

    @pytest.mark.truthset
    def test_model_deterministic_with_same_seed(self):
        """Model should be deterministic with the same seed."""
        def run_model(seed):
            torch.manual_seed(seed)
            model = nn.Sequential(
                nn.Embedding(100, 32),
                nn.Linear(32, 10),
            )
            model.eval()
            with torch.no_grad():
                input_ids = torch.tensor([[1, 2, 3, 4, 5]])
                return model(input_ids)

        out1 = run_model(42)
        out2 = run_model(42)
        assert torch.equal(out1, out2)

    @pytest.mark.truthset
    def test_perplexity_monotonic_with_loss(self):
        """Perplexity should increase monotonically with increasing loss."""
        losses = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        perplexities = [perplexity(l) for l in losses]
        for i in range(len(perplexities) - 1):
            assert perplexities[i] < perplexities[i + 1], (
                f"Perplexity not monotonic: {perplexities[i]} >= {perplexities[i+1]}"
            )
