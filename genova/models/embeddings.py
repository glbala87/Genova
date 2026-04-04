"""Genomic embedding layer for Genova.

Combines token embeddings, positional encoding (sinusoidal, learnable,
rotary, or ALiBi), optional segment embeddings, LayerNorm, and dropout
into a single module suitable for both Transformer and Mamba backbones.

Supports:
- Sinusoidal positional encoding (Vaswani et al., 2017).
- Learned positional embeddings.
- Rotary Positional Embeddings (RoPE; Su et al., 2021).
- ALiBi -- Attention with Linear Biases (Press et al., 2022).
  When ``embedding_type="alibi"``, no additive position information is
  injected into the embedding; instead the :class:`ALiBiPositionalBias`
  module provides a per-head bias added directly to the attention scores
  inside the attention layer.

Example::

    from genova.utils.config import ModelConfig
    cfg = ModelConfig(d_model=256, vocab_size=4096, max_position_embeddings=2048)
    emb = GenomicEmbedding(cfg)
    x = emb(input_ids)  # (B, L) -> (B, L, d_model)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from genova.utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Produces a buffer of shape ``(1, max_len, d_model)`` that is added to
    token embeddings.  The buffer is not a parameter and will not be
    updated during training.
    """

    def __init__(self, d_model: int, max_len: int = 8192) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, seq_len: int) -> Tensor:
        """Return positional encoding for the first *seq_len* positions."""
        return self.pe[:, :seq_len]


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (Su et al., 2021 -- RoFormer).

    Applies rotation matrices to query and key tensors so that the dot
    product between Q and K naturally encodes relative position.  Only
    the first ``dim`` dimensions of each head are rotated; the rest are
    left unchanged if ``dim < head_dim``.

    Args:
        dim: Dimension to apply rotary embeddings to.  Typically equal to
            ``head_dim`` (``d_model // n_heads``).
        max_seq_len: Maximum sequence length to pre-compute frequencies for.
        base: Base frequency for the sinusoidal schedule (default 10000).
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies: shape (dim // 2,)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build and register cos/sin buffers for positions ``0 .. seq_len-1``."""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` each of shape ``(1, 1, seq_len, dim)``.

        Args:
            seq_len: Current sequence length.

        Returns:
            Tuple of ``(cos, sin)`` tensors for positions ``0 .. seq_len-1``.
        """
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :],
            self.sin_cached[:, :, :seq_len, :],
        )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the second half of the last dimension by swapping and negating.

    Given ``x = [x1, x2]`` along dim=-1, returns ``[-x2, x1]``.

    Args:
        x: Tensor of shape ``(..., dim)`` where ``dim`` is even.

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Apply rotary positional embeddings to Q and K tensors.

    This applies the rotation only to Q and K (not V), as specified by
    the RoPE formulation.

    Args:
        q: Query tensor of shape ``(B, H, L, head_dim)``.
        k: Key tensor of shape ``(B, H, L, head_dim)``.
        cos: Cosine component ``(1, 1, L, head_dim)``.
        sin: Sine component ``(1, 1, L, head_dim)``.

    Returns:
        Tuple of rotated ``(q, k)`` tensors with the same shapes.
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# ALiBi -- Attention with Linear Biases
# ---------------------------------------------------------------------------


class ALiBiPositionalBias(nn.Module):
    """Attention with Linear Biases (Press et al., 2022).

    Adds a fixed, position-dependent bias to attention scores so that no
    learned or sinusoidal positional embeddings are needed.  Each head
    receives a different geometric slope so it can attend to a different
    effective context length.

    The slopes follow a geometric series::

        slope_i = 2^{-8 * (i+1) / n_heads}

    for head index *i* in ``0 .. n_heads - 1``.

    The bias for each pair of positions ``(q, k)`` is::

        bias[h, q, k] = -slope_h * |q - k|

    No parameters are learned; the bias buffer is recomputed when the
    sequence length grows.

    Args:
        n_heads: Number of attention heads.
        max_seq_len: Initial maximum sequence length for the cached bias
            tensor.  Automatically extended if a longer sequence is seen.
    """

    def __init__(self, n_heads: int, max_seq_len: int = 8192) -> None:
        super().__init__()
        self.n_heads = n_heads

        # Compute slopes: geometric series 2^(-8 * (i+1) / n_heads)
        slopes = torch.tensor(
            [2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)],
            dtype=torch.float32,
        )
        self.register_buffer("slopes", slopes, persistent=False)  # (n_heads,)

        # Pre-build bias cache
        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build the bias tensor for positions ``0 .. seq_len - 1``.

        Args:
            seq_len: Sequence length to cache.
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.slopes.device)
        # (seq_len, seq_len) pairwise absolute distances
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # (n_heads, seq_len, seq_len)
        bias = -self.slopes.unsqueeze(1).unsqueeze(2) * distance.unsqueeze(0)
        self.register_buffer("_bias_cache", bias.unsqueeze(0), persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, seq_len: int) -> Tensor:
        """Return the ALiBi bias tensor for a given sequence length.

        Args:
            seq_len: Current sequence length.

        Returns:
            Bias tensor of shape ``(1, n_heads, seq_len, seq_len)``.
        """
        if seq_len > self._cached_seq_len:
            self._build_cache(seq_len)
        return self._bias_cache[:, :, :seq_len, :seq_len]


# ---------------------------------------------------------------------------
# Genomic Embedding
# ---------------------------------------------------------------------------


class GenomicEmbedding(nn.Module):
    """Embedding layer for genomic sequences.

    Supports:
    - Token embeddings for k-mer or nucleotide tokens.
    - Positional encoding: ``"sinusoidal"``, ``"learned"``,
      ``"sinusoidal+learned"``, ``"rope"`` (rotary), or ``"alibi"``.
    - Optional segment (token-type) embeddings controlled by
      *num_segment_types*.
    - Post-embedding LayerNorm and dropout.

    When ``embedding_type="rope"``, no additive positional embeddings are
    applied here.  Instead, the :class:`RotaryPositionalEmbedding` is
    exposed via :attr:`rotary_emb` for the attention layer to consume.

    When ``embedding_type="alibi"``, no positional information is added to
    the embeddings.  Position information is injected via
    :class:`ALiBiPositionalBias` directly in the attention layer.

    Args:
        config: A :class:`ModelConfig` instance.
        embedding_type: One of ``"sinusoidal"``, ``"learned"``,
            ``"sinusoidal+learned"``, ``"rope"``, or ``"alibi"``.
            Defaults to ``"learned"``.
        num_segment_types: Number of segment types.  Set to ``0`` or
            ``None`` to disable segment embeddings.
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type: str = "learned",
        num_segment_types: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.embedding_type = embedding_type
        self.pad_token_id = config.pad_token_id

        # --- token embeddings ---
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )

        # --- positional encoding ---
        # ALiBi injects position info in the attention layer, not here.
        use_sinusoidal = embedding_type in ("sinusoidal", "sinusoidal+learned")
        use_learned = embedding_type in ("learned", "sinusoidal+learned")
        use_rope = embedding_type == "rope"
        use_alibi = embedding_type == "alibi"

        self.sinusoidal_pe: Optional[SinusoidalPositionalEncoding] = None
        self.learned_pe: Optional[nn.Embedding] = None
        self.rotary_emb: Optional[RotaryPositionalEmbedding] = None

        if use_sinusoidal:
            self.sinusoidal_pe = SinusoidalPositionalEncoding(
                config.d_model, config.max_position_embeddings
            )
        if use_learned:
            self.learned_pe = nn.Embedding(
                config.max_position_embeddings, config.d_model
            )
        if use_rope:
            head_dim = config.d_model // config.n_heads
            self.rotary_emb = RotaryPositionalEmbedding(
                dim=head_dim,
                max_seq_len=config.max_position_embeddings,
                base=getattr(config, "rope_base_freq", 10000.0),
            )

        # --- segment embeddings ---
        self.segment_embeddings: Optional[nn.Embedding] = None
        if num_segment_types and num_segment_types > 0:
            self.segment_embeddings = nn.Embedding(num_segment_types, config.d_model)

        # --- post-embedding layer ---
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        segment_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Embed input token ids.

        Args:
            input_ids: Integer tensor of shape ``(B, L)``.
            segment_ids: Optional integer tensor of shape ``(B, L)``
                identifying segment membership.
            position_ids: Optional integer tensor of shape ``(B, L)``
                with explicit position indices.  If ``None``, positions
                ``0 .. L-1`` are used.

        Returns:
            Float tensor of shape ``(B, L, d_model)``.
        """
        batch_size, seq_len = input_ids.shape

        x = self.token_embeddings(input_ids)

        # positional encoding (RoPE is applied in the attention layer, not here)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if self.sinusoidal_pe is not None:
            x = x + self.sinusoidal_pe(seq_len)
        if self.learned_pe is not None:
            x = x + self.learned_pe(position_ids)

        # segment embeddings
        if self.segment_embeddings is not None and segment_ids is not None:
            x = x + self.segment_embeddings(segment_ids)

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x
