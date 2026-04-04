"""Integration test: variant effect prediction pipeline.

Creates synthetic VCF and FASTA files, runs the full variant prediction
pipeline with a small model (random weights), and checks output format.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from genova.utils.config import ModelConfig, EvaluationConfig
from genova.data.tokenizer import GenomicTokenizer
from genova.models.transformer import GenovaTransformer
from genova.evaluation.variant_predictor import (
    FastaReader,
    Variant,
    VariantClassifierHead,
    VariantEffectPredictor,
    parse_vcf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_fasta(path: Path) -> None:
    """Write a minimal synthetic reference FASTA."""
    import random

    rng = random.Random(42)
    seq = "".join(rng.choices("ACGT", k=1000))
    path.write_text(f">chr1\n{seq}\n>chr2\n{'ACGT' * 250}\n")


def _write_synthetic_vcf(path: Path) -> None:
    """Write a minimal synthetic VCF with a few SNPs."""
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
        "chr1\t100\trs1\tA\tG\t.\tPASS\t.",
        "chr1\t200\trs2\tC\tT\t.\tPASS\t.",
        "chr1\t500\trs3\tG\tA,C\t.\tPASS\t.",
        "chr2\t150\trs4\tT\tC\t.\tPASS\t.",
    ]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVariantPrediction:
    """End-to-end variant effect prediction."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_dir = tmp_path

        # Write synthetic files
        self.fasta_path = tmp_path / "reference.fa"
        _write_synthetic_fasta(self.fasta_path)

        self.vcf_path = tmp_path / "variants.vcf"
        _write_synthetic_vcf(self.vcf_path)

        # Tokenizer
        self.tokenizer = GenomicTokenizer(mode="kmer", k=3, stride=1)
        self.tokenizer.build_vocab()

        # Tiny model
        self.config = ModelConfig(
            arch="transformer",
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ff=128,
            dropout=0.0,
            attention_dropout=0.0,
            max_position_embeddings=512,
            rotary_emb=False,
            flash_attention=False,
            gradient_checkpointing=False,
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=0,
        )
        self.model = GenovaTransformer(self.config)
        self.model.eval()

    def test_parse_vcf(self):
        """VCF parser yields correct number of variants."""
        variants = list(parse_vcf(self.vcf_path))
        # 4 lines, but rs3 has 2 ALTs -> 5 total
        assert len(variants) == 5, f"Expected 5 variants, got {len(variants)}"
        assert variants[0].chrom == "chr1"
        assert variants[0].pos == 100
        assert variants[0].ref == "A"
        assert variants[0].alt == "G"

    def test_fasta_reader(self):
        """FastaReader can fetch subsequences."""
        reader = FastaReader(self.fasta_path)
        seq = reader.fetch("chr1", 0, 10)
        assert len(seq) == 10, f"Expected 10 bases, got {len(seq)}"
        assert set(seq).issubset(set("ACGTN")), "Should contain valid bases"

        chroms = reader.chromosomes
        assert "chr1" in chroms
        assert "chr2" in chroms

    def test_variant_effect_predictor_single(self):
        """VariantEffectPredictor processes a single variant."""
        predictor = VariantEffectPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            window_size=100,
        )

        reference = FastaReader(self.fasta_path)
        variant = Variant(chrom="chr1", pos=100, ref="A", alt="G")

        result = predictor.predict_variant(variant, reference)

        assert result.variant.chrom == "chr1"
        assert isinstance(result.score, float), "Score should be a float"
        assert result.score >= 0, "Score should be non-negative"
        assert result.label in ("benign", "pathogenic"), (
            f"Label should be benign or pathogenic, got {result.label}"
        )
        assert result.ref_embedding is not None
        assert result.alt_embedding is not None
        assert result.ref_embedding.shape == (self.config.d_model,)

    def test_variant_effect_predictor_batch(self):
        """VariantEffectPredictor processes multiple variants."""
        predictor = VariantEffectPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            window_size=100,
        )

        reference = FastaReader(self.fasta_path)
        variants = [
            Variant(chrom="chr1", pos=100, ref="A", alt="G"),
            Variant(chrom="chr1", pos=200, ref="C", alt="T"),
        ]

        results = predictor.predict_variants(variants, reference, batch_size=2)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for r in results:
            assert isinstance(r.score, float)
            assert np.isfinite(r.score), "Score should be finite"

    def test_predict_vcf_end_to_end(self):
        """Full VCF-based prediction pipeline."""
        predictor = VariantEffectPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            window_size=100,
        )

        results = predictor.predict_vcf(
            self.vcf_path,
            self.fasta_path,
            batch_size=4,
        )

        assert len(results) == 5, f"Expected 5 results from VCF, got {len(results)}"

    def test_with_classifier_head(self):
        """VariantEffectPredictor with a classifier head produces scores in [0, 1]."""
        classifier = VariantClassifierHead(
            d_model=self.config.d_model,
            hidden_dim=32,
        )

        predictor = VariantEffectPredictor(
            model=self.model,
            tokenizer=self.tokenizer,
            device="cpu",
            window_size=100,
            classifier_head=classifier,
        )

        reference = FastaReader(self.fasta_path)
        variant = Variant(chrom="chr1", pos=100, ref="A", alt="G")

        result = predictor.predict_variant(variant, reference)

        assert 0.0 <= result.score <= 1.0, (
            f"Classifier score should be in [0,1], got {result.score}"
        )
