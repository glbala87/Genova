"""Tests for GenomicTokenizer."""

from __future__ import annotations

import pytest

from genova.data.tokenizer import GenomicTokenizer, SPECIAL_TOKENS, reverse_complement


# ---------------------------------------------------------------------------
# Nucleotide mode
# ---------------------------------------------------------------------------

class TestNucleotideMode:
    """Tests for mode='nucleotide'."""

    @pytest.fixture
    def tok(self):
        t = GenomicTokenizer(mode="nucleotide", add_special_tokens=False)
        t.build_vocab()
        return t

    @pytest.fixture
    def tok_special(self):
        t = GenomicTokenizer(mode="nucleotide", add_special_tokens=True)
        t.build_vocab()
        return t

    def test_vocab_size(self, tok):
        # 5 specials + 5 nucleotides (A, C, G, T, N)
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 5

    def test_encode_decode_roundtrip(self, tok):
        seq = "ACGTACGT"
        ids = tok.encode(seq)
        decoded = tok.decode(ids)
        assert decoded == seq

    @pytest.mark.parametrize("seq", ["A", "ACGT", "NNNN", "ACGTNACGT"])
    def test_roundtrip_various(self, tok, seq):
        ids = tok.encode(seq)
        assert tok.decode(ids) == seq

    def test_special_tokens_added(self, tok_special):
        ids = tok_special.encode("ACGT")
        assert ids[0] == tok_special.cls_token_id
        assert ids[-1] == tok_special.sep_token_id

    def test_special_tokens_skipped_in_decode(self, tok_special):
        ids = tok_special.encode("ACGT")
        decoded = tok_special.decode(ids, skip_special_tokens=True)
        assert decoded == "ACGT"

    def test_case_insensitive(self, tok):
        ids_upper = tok.encode("ACGT")
        ids_lower = tok.encode("acgt")
        assert ids_upper == ids_lower

    def test_encode_before_build_raises(self):
        t = GenomicTokenizer(mode="nucleotide")
        with pytest.raises(RuntimeError, match="Vocabulary has not been built"):
            t.encode("ACGT")


# ---------------------------------------------------------------------------
# K-mer mode
# ---------------------------------------------------------------------------

class TestKmerMode:
    """Tests for mode='kmer'."""

    @pytest.fixture
    def tok3(self):
        t = GenomicTokenizer(mode="kmer", k=3, stride=1, add_special_tokens=False)
        t.build_vocab()
        return t

    def test_vocab_size_combinatorial(self, tok3):
        # 5 specials + 5^3 = 125 kmers
        assert tok3.vocab_size == len(SPECIAL_TOKENS) + 5 ** 3

    def test_tokenize_length(self, tok3):
        seq = "ACGTACGT"  # 8 bases, k=3, stride=1 -> 6 tokens
        tokens = tok3.tokenize(seq)
        assert len(tokens) == 8 - 3 + 1

    def test_encode_decode_roundtrip(self, tok3):
        seq = "ACGTACGT"
        ids = tok3.encode(seq)
        decoded = tok3.decode(ids)
        assert decoded == seq

    @pytest.mark.parametrize("k", [3, 4, 5, 6])
    def test_roundtrip_various_k(self, k):
        t = GenomicTokenizer(mode="kmer", k=k, stride=1, add_special_tokens=False)
        t.build_vocab()
        seq = "ACGTACGTACGTACGT"
        ids = t.encode(seq)
        decoded = t.decode(ids)
        assert decoded == seq

    def test_build_vocab_from_sequences(self):
        t = GenomicTokenizer(mode="kmer", k=3, stride=1)
        t.build_vocab(sequences=["ACGTAC", "CGTACG"])
        # Should only contain k-mers present in the data
        assert t.vocab_size < len(SPECIAL_TOKENS) + 5 ** 3

    def test_build_vocab_min_freq(self):
        t = GenomicTokenizer(mode="kmer", k=3, stride=1)
        t.build_vocab(sequences=["ACGTAC"], min_freq=2)
        # Very few k-mers appear >= 2 times in a short seq
        assert t.vocab_size <= len(SPECIAL_TOKENS) + 5 ** 3

    def test_unk_for_missing_kmer(self):
        t = GenomicTokenizer(mode="kmer", k=3, stride=1, add_special_tokens=False)
        t.build_vocab(sequences=["AAAA"])  # only AAA in vocab
        ids = t.encode("GGG")
        # GGG not in vocab -> UNK
        assert t.unk_token_id in ids

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be between"):
            GenomicTokenizer(mode="kmer", k=2)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            GenomicTokenizer(mode="codon")


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatchEncode:

    @pytest.fixture
    def tok(self):
        t = GenomicTokenizer(mode="nucleotide", add_special_tokens=True)
        t.build_vocab()
        return t

    def test_batch_shapes(self, tok):
        seqs = ["ACGT", "ACGTACGT"]
        result = tok.batch_encode(seqs, padding=True)
        ids = result["input_ids"]
        masks = result["attention_mask"]
        assert len(ids) == 2
        assert len(ids[0]) == len(ids[1])  # padded to same length
        assert len(masks[0]) == len(masks[1])

    def test_attention_mask_values(self, tok):
        seqs = ["AC", "ACGTACGT"]
        result = tok.batch_encode(seqs, padding=True)
        short_mask = result["attention_mask"][0]
        long_mask = result["attention_mask"][1]
        # Short sequence has trailing zeros
        assert 0 in short_mask
        assert all(v == 1 for v in long_mask)

    def test_no_padding(self, tok):
        seqs = ["ACGT", "ACGTACGT"]
        result = tok.batch_encode(seqs, padding=False)
        assert len(result["input_ids"][0]) != len(result["input_ids"][1])

    def test_max_length(self, tok):
        result = tok.batch_encode(["ACGTACGTACGT"], max_length=6)
        assert len(result["input_ids"][0]) <= 6


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

class TestSaveLoad:

    def test_roundtrip(self, tmp_path):
        t = GenomicTokenizer(mode="kmer", k=4, stride=1, add_special_tokens=True)
        t.build_vocab()
        path = tmp_path / "tokenizer.json"
        t.save(path)

        loaded = GenomicTokenizer.load(path)
        assert loaded.mode == t.mode
        assert loaded.k == t.k
        assert loaded.stride == t.stride
        assert loaded.vocab_size == t.vocab_size

        seq = "ACGTACGTACGT"
        assert loaded.encode(seq) == t.encode(seq)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GenomicTokenizer.load(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# reverse_complement helper
# ---------------------------------------------------------------------------

class TestReverseComplement:

    @pytest.mark.parametrize("seq,expected", [
        ("ACGT", "ACGT"),
        ("AAAA", "TTTT"),
        ("CCCC", "GGGG"),
        ("ATCG", "CGAT"),
        ("N", "N"),
    ])
    def test_reverse_complement(self, seq, expected):
        assert reverse_complement(seq) == expected

    def test_double_rc_identity(self):
        seq = "ACGTACGT"
        assert reverse_complement(reverse_complement(seq)) == seq
