"""Genomic sequence augmentations for contrastive learning.

Provides biologically-informed data augmentation strategies for DNA sequences
that preserve or controllably perturb sequence properties to generate
positive pairs for self-supervised contrastive objectives.

Example::

    augmenter = GenomicAugmenter(
        augmentations=["reverse_complement", "random_mutation"],
        mutation_rate=0.01,
        mask_rate=0.15,
        mask_token_id=4,
    )
    view1, view2 = augmenter(input_ids)  # two augmented views
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Complement mapping for nucleotide token IDs
# Default mapping assumes: 0=pad, 1=A, 2=T, 3=C, 4=G, 5=mask, 6=cls, 7=sep
# ---------------------------------------------------------------------------

_DEFAULT_COMPLEMENT: Dict[int, int] = {
    0: 0,   # pad -> pad
    1: 2,   # A -> T
    2: 1,   # T -> A
    3: 4,   # C -> G
    4: 3,   # G -> C
    5: 5,   # mask -> mask
    6: 6,   # cls -> cls
    7: 7,   # sep -> sep
}


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------


def reverse_complement(
    tokens: Tensor,
    complement_map: Optional[Dict[int, int]] = None,
) -> Tensor:
    """Return the reverse complement of a tokenised DNA sequence.

    Args:
        tokens: Integer tensor of shape ``(L,)`` or ``(B, L)``.
        complement_map: Mapping from token id to its complement.
            Defaults to the standard A<->T, C<->G mapping.

    Returns:
        Tensor of same shape with bases complemented and order reversed.
    """
    cmap = complement_map or _DEFAULT_COMPLEMENT
    # Build a lookup tensor covering all possible token ids
    max_id = max(
        max(cmap.keys()), max(cmap.values()), int(tokens.max().item())
    ) + 1
    lookup = torch.arange(max_id, device=tokens.device)
    for src, dst in cmap.items():
        lookup[src] = dst
    complemented = lookup[tokens.long()]
    return complemented.flip(-1)


def random_mutation(
    tokens: Tensor,
    rate: float = 0.01,
    vocab_nucleotides: Sequence[int] = (1, 2, 3, 4),
    special_token_ids: Sequence[int] = (0, 5, 6, 7),
) -> Tensor:
    """Introduce random point mutations at a configurable rate.

    Args:
        tokens: ``(L,)`` or ``(B, L)`` integer tensor.
        rate: Probability of mutating each position.
        vocab_nucleotides: Token ids considered valid nucleotides.
        special_token_ids: Token ids that must never be mutated.

    Returns:
        Mutated copy of *tokens*.
    """
    mutated = tokens.clone()
    mask = torch.rand_like(tokens, dtype=torch.float) < rate

    # Never mutate special tokens
    for sid in special_token_ids:
        mask = mask & (tokens != sid)

    n_mutations = int(mask.sum().item())
    if n_mutations > 0:
        nucs = torch.tensor(vocab_nucleotides, device=tokens.device)
        replacements = nucs[torch.randint(len(nucs), (n_mutations,), device=tokens.device)]
        mutated[mask] = replacements
    return mutated


def random_mask(
    tokens: Tensor,
    mask_rate: float = 0.15,
    mask_token_id: int = 5,
    special_token_ids: Sequence[int] = (0, 6, 7),
) -> Tensor:
    """Randomly mask positions with a mask token.

    Args:
        tokens: ``(L,)`` or ``(B, L)`` integer tensor.
        mask_rate: Fraction of positions to mask.
        mask_token_id: Token id used for masking.
        special_token_ids: Token ids that must not be masked.

    Returns:
        Masked copy of *tokens*.
    """
    masked = tokens.clone()
    probs = torch.rand_like(tokens, dtype=torch.float)
    mask = probs < mask_rate

    for sid in special_token_ids:
        mask = mask & (tokens != sid)

    masked[mask] = mask_token_id
    return masked


def subsequence_crop(
    tokens: Tensor,
    min_frac: float = 0.5,
    max_frac: float = 0.9,
    pad_token_id: int = 0,
) -> Tensor:
    """Crop a random contiguous sub-sequence from the input.

    The returned tensor is padded to the original length.

    Args:
        tokens: ``(L,)`` or ``(B, L)`` integer tensor.
        min_frac: Minimum fraction of the sequence length to keep.
        max_frac: Maximum fraction of the sequence length to keep.
        pad_token_id: Token id used for padding.

    Returns:
        Cropped and right-padded copy of *tokens*.
    """
    single = tokens.dim() == 1
    if single:
        tokens = tokens.unsqueeze(0)

    B, L = tokens.shape
    frac = random.uniform(min_frac, max_frac)
    crop_len = max(1, int(L * frac))
    max_start = L - crop_len

    result = torch.full_like(tokens, pad_token_id)
    for i in range(B):
        start = random.randint(0, max_start)
        result[i, :crop_len] = tokens[i, start : start + crop_len]

    return result.squeeze(0) if single else result


def window_shuffle(
    tokens: Tensor,
    window_size: int = 8,
    special_token_ids: Sequence[int] = (0, 5, 6, 7),
) -> Tensor:
    """Shuffle nucleotide tokens within non-overlapping windows.

    Special tokens are preserved in their original positions.

    Args:
        tokens: ``(L,)`` or ``(B, L)`` integer tensor.
        window_size: Size of each shuffle window.
        special_token_ids: Token ids that must not be moved.

    Returns:
        Window-shuffled copy of *tokens*.
    """
    single = tokens.dim() == 1
    if single:
        tokens = tokens.unsqueeze(0)

    result = tokens.clone()
    B, L = result.shape
    special_set = set(special_token_ids)

    for b in range(B):
        for start in range(0, L, window_size):
            end = min(start + window_size, L)
            window = result[b, start:end].clone()

            # Identify positions eligible for shuffling
            eligible_mask = torch.tensor(
                [w.item() not in special_set for w in window],
                dtype=torch.bool,
                device=tokens.device,
            )
            eligible_vals = window[eligible_mask]
            if len(eligible_vals) > 1:
                perm = torch.randperm(len(eligible_vals), device=tokens.device)
                window[eligible_mask] = eligible_vals[perm]
            result[b, start:end] = window

    return result.squeeze(0) if single else result


# ---------------------------------------------------------------------------
# Augmentation registry
# ---------------------------------------------------------------------------

_AUGMENTATION_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "reverse_complement": reverse_complement,
    "random_mutation": random_mutation,
    "random_mask": random_mask,
    "subsequence_crop": subsequence_crop,
    "window_shuffle": window_shuffle,
}


# ---------------------------------------------------------------------------
# Composer class
# ---------------------------------------------------------------------------


@dataclass
class GenomicAugmenter:
    """Compose and apply genomic sequence augmentations.

    Provides a callable interface that produces two independently-augmented
    views of the same input batch, suitable for contrastive learning.

    Args:
        augmentations: Ordered list of augmentation names to apply for each
            view.  Available: ``reverse_complement``, ``random_mutation``,
            ``random_mask``, ``subsequence_crop``, ``window_shuffle``.
        mutation_rate: Point-mutation probability for ``random_mutation``.
        mask_rate: Masking probability for ``random_mask``.
        mask_token_id: Token id used by ``random_mask``.
        crop_min_frac: Minimum fraction kept by ``subsequence_crop``.
        crop_max_frac: Maximum fraction kept by ``subsequence_crop``.
        shuffle_window: Window size for ``window_shuffle``.
        complement_map: Custom complement mapping for ``reverse_complement``.
        p: Per-augmentation application probability (stochastic augmentation).
    """

    augmentations: List[str] = field(
        default_factory=lambda: ["random_mutation", "random_mask"]
    )
    mutation_rate: float = 0.01
    mask_rate: float = 0.15
    mask_token_id: int = 5
    crop_min_frac: float = 0.5
    crop_max_frac: float = 0.9
    shuffle_window: int = 8
    complement_map: Optional[Dict[int, int]] = None
    p: float = 1.0

    def _build_kwargs(self, name: str) -> Dict:
        """Build keyword arguments for a named augmentation."""
        if name == "reverse_complement":
            return {"complement_map": self.complement_map}
        if name == "random_mutation":
            return {"rate": self.mutation_rate}
        if name == "random_mask":
            return {"mask_rate": self.mask_rate, "mask_token_id": self.mask_token_id}
        if name == "subsequence_crop":
            return {"min_frac": self.crop_min_frac, "max_frac": self.crop_max_frac}
        if name == "window_shuffle":
            return {"window_size": self.shuffle_window}
        return {}

    def compose(self) -> List[Callable[[Tensor], Tensor]]:
        """Return the list of augmentation callables with bound parameters.

        Returns:
            List of callables, each mapping ``Tensor -> Tensor``.
        """
        ops: List[Callable[[Tensor], Tensor]] = []
        for name in self.augmentations:
            if name not in _AUGMENTATION_REGISTRY:
                raise ValueError(
                    f"Unknown augmentation '{name}'. "
                    f"Available: {list(_AUGMENTATION_REGISTRY.keys())}"
                )
            fn = _AUGMENTATION_REGISTRY[name]
            kwargs = self._build_kwargs(name)

            # Capture fn and kwargs in closure
            def _make_op(
                _fn: Callable[..., Tensor] = fn, _kw: Dict = kwargs
            ) -> Callable[[Tensor], Tensor]:
                def _op(tokens: Tensor) -> Tensor:
                    return _fn(tokens, **_kw)
                return _op

            ops.append(_make_op())
        return ops

    def apply_pipeline(self, tokens: Tensor) -> Tensor:
        """Apply the full augmentation pipeline to a single input.

        Each augmentation in the pipeline is applied with probability *p*.

        Args:
            tokens: ``(L,)`` or ``(B, L)`` integer tensor.

        Returns:
            Augmented copy of *tokens*.
        """
        result = tokens.clone()
        for op in self.compose():
            if random.random() < self.p:
                result = op(result)
        return result

    def __call__(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate two independently-augmented views of the input.

        Args:
            tokens: ``(B, L)`` integer tensor of token ids.

        Returns:
            Tuple ``(view1, view2)`` of augmented tensors with the same shape.
        """
        view1 = self.apply_pipeline(tokens)
        view2 = self.apply_pipeline(tokens)
        return view1, view2
