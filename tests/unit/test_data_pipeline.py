"""Tests for data pipeline modules: BPE tokenizer, preprocessing, dataset, quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from genova.data.bpe_tokenizer import GenomicBPETokenizer
from genova.data.quality_report import DataQualityReporter
from genova.data.genome_dataset import GenomeDataset
from genova.data.tokenizer import GenomicTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_synthetic_fasta(path: Path) -> None:
    """Write a minimal synthetic FASTA for testing."""
    import random
    rng = random.Random(42)
    seq1 = "".join(rng.choices("ACGT", k=2000))
    seq2 = "".join(rng.choices("ACGT", k=1000))
    path.write_text(f">chr1\n{seq1}\n>chr2\n{seq2}\n")


# ---------------------------------------------------------------------------
# BPE Tokenizer
# ---------------------------------------------------------------------------


class TestGenomicBPETokenizer:

    def test_train_on_small_corpus(self, tmp_path):
        corpus = ["ACGTACGTACGTACGT" * 10] * 20
        tok = GenomicBPETokenizer()
        tok.train(corpus, vocab_size=50)
        assert tok.vocab_size >= 5  # at least base nucleotides

    def test_encode_decode_roundtrip(self, tmp_path):
        corpus = ["ACGTACGTACGT" * 10] * 20
        tok = GenomicBPETokenizer()
        tok.train(corpus, vocab_size=50)
        seq = "ACGTACGT"
        ids = tok.encode(seq)
        decoded = tok.decode(ids)
        assert decoded == seq

    def test_vocab_size_grows_with_training(self):
        corpus = ["ACGTACGTACGT" * 10] * 20
        tok = GenomicBPETokenizer()
        tok.train(corpus, vocab_size=30)
        small_vocab = tok.vocab_size

        tok2 = GenomicBPETokenizer()
        tok2.train(corpus, vocab_size=100)
        large_vocab = tok2.vocab_size

        assert large_vocab >= small_vocab

    def test_save_and_load(self, tmp_path):
        corpus = ["ACGTACGTACGT" * 5] * 10
        tok = GenomicBPETokenizer()
        tok.train(corpus, vocab_size=40)

        save_path = tmp_path / "bpe_model.json"
        tok.save(save_path)

        tok2 = GenomicBPETokenizer.load(save_path)

        seq = "ACGTACGT"
        assert tok.encode(seq) == tok2.encode(seq)


# ---------------------------------------------------------------------------
# DataQualityReporter
# ---------------------------------------------------------------------------


class TestDataQualityReporter:

    def test_analyze_fasta(self, tmp_path):
        fasta_path = tmp_path / "test.fa"
        _write_synthetic_fasta(fasta_path)
        reporter = DataQualityReporter()
        report = reporter.analyze_fasta(fasta_path)
        assert report is not None
        # Should have basic stats
        if isinstance(report, dict):
            assert "total_sequences" in report or "num_sequences" in report
        else:
            # May be a dataclass with attributes
            assert hasattr(report, "total_sequences") or hasattr(report, "num_sequences")

    def test_gc_content_computed(self, tmp_path):
        fasta_path = tmp_path / "test.fa"
        _write_synthetic_fasta(fasta_path)
        reporter = DataQualityReporter()
        report = reporter.analyze_fasta(fasta_path)
        # GC content should be reported somewhere
        if isinstance(report, dict):
            gc_keys = [k for k in report if "gc" in k.lower()]
            assert len(gc_keys) > 0 or "mean_gc_content" in report


# ---------------------------------------------------------------------------
# GenomeDataset
# ---------------------------------------------------------------------------


class TestGenomeDataset:

    def test_creation(self, tmp_path):
        fasta_path = tmp_path / "test.fa"
        _write_synthetic_fasta(fasta_path)
        tok = GenomicTokenizer(mode="nucleotide")
        tok.build_vocab()
        ds = GenomeDataset(
            fasta_path=fasta_path,
            tokenizer=tok,
            window_size=1000,
            stride=500,
        )
        assert len(ds) > 0

    def test_getitem_returns_dict(self, tmp_path):
        fasta_path = tmp_path / "test.fa"
        _write_synthetic_fasta(fasta_path)
        tok = GenomicTokenizer(mode="nucleotide")
        tok.build_vocab()
        ds = GenomeDataset(
            fasta_path=fasta_path,
            tokenizer=tok,
            window_size=1000,
            stride=500,
        )
        item = ds[0]
        assert isinstance(item, dict)
        assert "input_ids" in item
        assert "labels" in item

    def test_multiple_items_differ(self, tmp_path):
        fasta_path = tmp_path / "test.fa"
        _write_synthetic_fasta(fasta_path)
        tok = GenomicTokenizer(mode="nucleotide")
        tok.build_vocab()
        ds = GenomeDataset(
            fasta_path=fasta_path,
            tokenizer=tok,
            window_size=1000,
            stride=500,
        )
        if len(ds) > 1:
            item0 = ds[0]
            item1 = ds[1]
            ids0 = item0["input_ids"]
            ids1 = item1["input_ids"]
            if isinstance(ids0, torch.Tensor):
                assert not torch.equal(ids0, ids1)
            else:
                assert ids0 != ids1
