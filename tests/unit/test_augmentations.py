"""Tests for genomic augmentations."""

from __future__ import annotations

import pytest
import torch

from genova.contrastive.augmentations import (
    reverse_complement,
    random_mutation,
    random_mask,
    subsequence_crop,
    window_shuffle,
    GenomicAugmenter,
    _DEFAULT_COMPLEMENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tokens():
    """Batch of token ids: A=1, T=2, C=3, G=4, pad=0."""
    return torch.tensor([
        [1, 2, 3, 4, 1, 2, 3, 4],
        [4, 3, 2, 1, 4, 3, 2, 1],
    ])


@pytest.fixture
def single_tokens():
    return torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# reverse_complement
# ---------------------------------------------------------------------------

class TestReverseComplement:

    def test_double_rc_identity(self, sample_tokens):
        rc = reverse_complement(sample_tokens)
        rc2 = reverse_complement(rc)
        assert torch.equal(rc2, sample_tokens)

    def test_complement_mapping(self):
        tokens = torch.tensor([1, 2, 3, 4])  # A, T, C, G
        rc = reverse_complement(tokens)
        # reversed then complemented: G->C, C->G, T->A, A->T reversed
        # complement of [1,2,3,4] = [2,1,4,3], reversed = [3,4,1,2]
        expected = torch.tensor([3, 4, 1, 2])
        assert torch.equal(rc, expected)

    def test_1d_input(self, single_tokens):
        rc = reverse_complement(single_tokens)
        assert rc.shape == single_tokens.shape

    def test_2d_input(self, sample_tokens):
        rc = reverse_complement(sample_tokens)
        assert rc.shape == sample_tokens.shape

    def test_special_tokens_preserved(self):
        tokens = torch.tensor([6, 1, 2, 3, 4, 7])  # cls, A, T, C, G, sep
        rc = reverse_complement(tokens)
        # After complement + reverse, specials map to themselves
        # complement: [6,2,1,4,3,7], reversed: [7,3,4,1,2,6]
        assert rc[0].item() == 7  # sep moved to front
        assert rc[-1].item() == 6  # cls moved to end


# ---------------------------------------------------------------------------
# random_mutation
# ---------------------------------------------------------------------------

class TestRandomMutation:

    def test_shape_preserved(self, sample_tokens):
        mutated = random_mutation(sample_tokens, rate=0.5)
        assert mutated.shape == sample_tokens.shape

    def test_no_mutation_at_zero_rate(self, sample_tokens):
        mutated = random_mutation(sample_tokens, rate=0.0)
        assert torch.equal(mutated, sample_tokens)

    def test_mutations_occur_at_high_rate(self, sample_tokens):
        torch.manual_seed(42)
        mutated = random_mutation(sample_tokens, rate=0.99)
        # With 99% rate, at least some positions should differ
        assert not torch.equal(mutated, sample_tokens)

    def test_special_tokens_not_mutated(self):
        tokens = torch.tensor([0, 5, 6, 7, 1, 2, 3, 4])
        torch.manual_seed(42)
        mutated = random_mutation(tokens, rate=1.0)
        # Special token ids (0,5,6,7) should remain unchanged
        assert mutated[0].item() == 0
        assert mutated[1].item() == 5
        assert mutated[2].item() == 6
        assert mutated[3].item() == 7

    def test_1d(self, single_tokens):
        mutated = random_mutation(single_tokens, rate=0.5)
        assert mutated.shape == single_tokens.shape


# ---------------------------------------------------------------------------
# random_mask
# ---------------------------------------------------------------------------

class TestRandomMask:

    def test_shape_preserved(self, sample_tokens):
        masked = random_mask(sample_tokens, mask_rate=0.5)
        assert masked.shape == sample_tokens.shape

    def test_no_masking_at_zero_rate(self, sample_tokens):
        masked = random_mask(sample_tokens, mask_rate=0.0)
        assert torch.equal(masked, sample_tokens)

    def test_mask_token_used(self, sample_tokens):
        torch.manual_seed(42)
        masked = random_mask(sample_tokens, mask_rate=0.99, mask_token_id=5)
        assert (masked == 5).any()

    def test_special_not_masked(self):
        tokens = torch.tensor([0, 6, 7, 1, 2, 3, 4, 1])
        torch.manual_seed(0)
        masked = random_mask(tokens, mask_rate=1.0, mask_token_id=5)
        assert masked[0].item() == 0
        assert masked[1].item() == 6
        assert masked[2].item() == 7


# ---------------------------------------------------------------------------
# subsequence_crop
# ---------------------------------------------------------------------------

class TestSubsequenceCrop:

    def test_output_length(self, sample_tokens):
        cropped = subsequence_crop(sample_tokens, min_frac=0.5, max_frac=0.9)
        assert cropped.shape == sample_tokens.shape  # padded to original length

    def test_contains_pad(self, sample_tokens):
        cropped = subsequence_crop(sample_tokens, min_frac=0.5, max_frac=0.5)
        # With 50% kept, there should be padding
        assert (cropped == 0).any()

    def test_1d(self, single_tokens):
        cropped = subsequence_crop(single_tokens, min_frac=0.5, max_frac=0.9)
        assert cropped.dim() == 1


# ---------------------------------------------------------------------------
# window_shuffle
# ---------------------------------------------------------------------------

class TestWindowShuffle:

    def test_shape_preserved(self, sample_tokens):
        shuffled = window_shuffle(sample_tokens, window_size=4)
        assert shuffled.shape == sample_tokens.shape

    def test_same_elements(self, sample_tokens):
        shuffled = window_shuffle(sample_tokens, window_size=4)
        # Should contain the same multiset of elements per row
        for i in range(sample_tokens.shape[0]):
            assert sorted(sample_tokens[i].tolist()) == sorted(shuffled[i].tolist())


# ---------------------------------------------------------------------------
# GenomicAugmenter
# ---------------------------------------------------------------------------

class TestGenomicAugmenter:

    def test_call_returns_two_views(self, sample_tokens):
        aug = GenomicAugmenter(augmentations=["random_mutation"], mutation_rate=0.1, p=1.0)
        v1, v2 = aug(sample_tokens)
        assert v1.shape == sample_tokens.shape
        assert v2.shape == sample_tokens.shape

    def test_compose(self):
        aug = GenomicAugmenter(augmentations=["random_mutation", "random_mask"])
        ops = aug.compose()
        assert len(ops) == 2

    def test_unknown_augmentation_raises(self):
        aug = GenomicAugmenter(augmentations=["nonexistent_aug"])
        with pytest.raises(ValueError, match="Unknown augmentation"):
            aug.compose()

    def test_p_zero_no_change(self, sample_tokens):
        aug = GenomicAugmenter(
            augmentations=["random_mutation", "random_mask"],
            mutation_rate=1.0,
            mask_rate=1.0,
            p=0.0,
        )
        result = aug.apply_pipeline(sample_tokens)
        assert torch.equal(result, sample_tokens)
