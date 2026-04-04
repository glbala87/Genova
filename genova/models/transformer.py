"""Transformer encoder for the Genova genomic foundation model.

Implements a BERT-style encoder with configurable pre-norm / post-norm,
multi-head self-attention, GELU feed-forward blocks, and a masked-language-
modelling (MLM) head.  Fully compatible with PyTorch AMP (mixed precision).

Supports optional enhancements (all off by default, opt-in via config):
- **Flash Attention v2**: auto-detected ``flash_attn`` package for O(N)
  memory attention.
- **RoPE**: Rotary positional embeddings applied to Q and K in attention.
- **RMSNorm**: Faster alternative to LayerNorm when ``config.norm_type``
  is ``"rmsnorm"``.
- **Grouped-Query Attention (GQA)** / **Multi-Query Attention (MQA)**:
  reduce KV heads via ``config.n_kv_heads``.
- **SwiGLU Activation**: gated FFN alternative when ``config.activation``
  is ``"swiglu"``.
- **Sliding Window Attention**: local attention via ``config.sliding_window_size``.
- **ALiBi**: Attention with Linear Biases via ``config.pos_encoding = "alibi"``.

Example::

    from genova.utils.config import ModelConfig
    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    model = GenovaForMLM(cfg)
    out = model(input_ids, labels=labels)
    loss = out["loss"]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from genova.utils.config import ModelConfig
from genova.models.embeddings import GenomicEmbedding, apply_rotary_pos_emb

# ---------------------------------------------------------------------------
# Flash Attention v2 auto-detection
# ---------------------------------------------------------------------------

_HAS_FLASH_ATTN = False
_flash_attn_func = None
_flash_attn_qkvpacked_func = None

try:
    from flash_attn import flash_attn_func as _fa_func  # type: ignore[import-untyped]
    from flash_attn import flash_attn_qkvpacked_func as _fa_qkvpacked  # type: ignore[import-untyped]

    _HAS_FLASH_ATTN = True
    _flash_attn_func = _fa_func
    _flash_attn_qkvpacked_func = _fa_qkvpacked
except ImportError:
    pass


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Faster than LayerNorm because it skips the mean-centering step.

    Formula::

        x_norm = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        d_model: Feature dimension to normalise over.
        eps: Small constant for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm.

        Args:
            x: Input tensor of shape ``(..., d_model)``.

        Returns:
            Normalised tensor of the same shape.
        """
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Norm factory
# ---------------------------------------------------------------------------


def _build_norm(d_model: int, norm_type: str, eps: float = 1e-12) -> nn.Module:
    """Create a normalisation layer based on *norm_type*.

    Args:
        d_model: Feature dimension.
        norm_type: ``"layernorm"``, ``"rmsnorm"``, or ``"prenorm"``
            (treated as LayerNorm; the pre/post distinction is handled
            by the encoder layer).
        eps: Epsilon for numerical stability.

    Returns:
        An ``nn.Module`` implementing the requested normalisation.

    Raises:
        ValueError: If *norm_type* is not recognised.
    """
    lower = norm_type.lower()
    if lower == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    if lower in ("layernorm", "prenorm", "postnorm"):
        return nn.LayerNorm(d_model, eps=eps)
    raise ValueError(
        f"Unknown norm_type {norm_type!r}. Choose 'layernorm' or 'rmsnorm'."
    )


# ---------------------------------------------------------------------------
# Sliding window attention mask helper
# ---------------------------------------------------------------------------


def _build_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a sliding window attention mask.

    Each token attends only to tokens within ``window_size // 2`` positions
    on either side (inclusive of itself).

    Args:
        seq_len: Sequence length.
        window_size: Total window size (tokens attend to ``window_size``
            nearest neighbours including themselves).
        device: Target device.
        dtype: Target dtype.

    Returns:
        Mask tensor of shape ``(1, 1, seq_len, seq_len)`` with 0 for
        allowed positions and ``-inf`` for blocked positions.
    """
    # positions matrix: (seq_len, seq_len)
    positions = torch.arange(seq_len, device=device)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    half_window = window_size // 2
    mask = torch.where(
        distance <= half_window,
        torch.zeros(1, device=device, dtype=dtype),
        torch.full((1,), float("-inf"), device=device, dtype=dtype),
    )
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)


# ---------------------------------------------------------------------------
# SwiGLU activation
# ---------------------------------------------------------------------------


class SwiGLU(nn.Module):
    """SwiGLU gated feed-forward network (Shazeer, 2020).

    Computes::

        gate = W_gate(x)
        up   = W_up(x)
        out  = W_down(swish(gate) * up)

    Uses 3 linear projections (gate, up, down) instead of the standard 2.

    Args:
        d_model: Input / output feature dimension.
        d_ff: Hidden (intermediate) dimension for the gate and up projections.
        dropout: Dropout probability applied after the down projection.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU: ``W_down(swish(W_gate(x)) * W_up(x))``.

        Args:
            x: ``(B, L, D)``

        Returns:
            ``(B, L, D)``
        """
        gate = F.silu(self.w_gate(x))  # swish == silu
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional Flash Attention, RoPE, GQA/MQA,
    ALiBi, and sliding window attention.

    **Grouped-Query Attention (GQA) / Multi-Query Attention (MQA)**:

    - When ``num_kv_heads < num_heads``: GQA -- KV heads are shared across
      groups of query heads.
    - When ``num_kv_heads == 1``: MQA -- a single KV head serves all queries.
    - When ``num_kv_heads == num_heads``: standard MHA (default).

    **Sliding Window Attention**: When ``config.sliding_window_size`` is set,
    each token only attends to a local window of neighbours.

    **ALiBi**: When the model uses ``pos_encoding="alibi"``, an
    :class:`ALiBiPositionalBias` module is passed in and its bias tensor is
    added to the raw attention scores.

    Args:
        config: A :class:`ModelConfig` instance.
        rotary_emb: Optional pre-built :class:`RotaryPositionalEmbedding`
            shared from the embedding layer.
        alibi_bias: Optional :class:`ALiBiPositionalBias` module.
    """

    def __init__(
        self,
        config: ModelConfig,
        rotary_emb: Optional[nn.Module] = None,
        alibi_bias: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})"
        )
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = math.sqrt(self.head_dim)

        # --- GQA / MQA ---
        # n_kv_heads=None means standard MHA (same as n_heads)
        n_kv_heads = getattr(config, "n_kv_heads", None)
        self.n_kv_heads: int = n_kv_heads if n_kv_heads is not None else config.n_heads
        assert config.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.n_groups: int = config.n_heads // self.n_kv_heads

        kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, kv_dim)
        self.v_proj = nn.Linear(config.d_model, kv_dim)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.attn_dropout_p = config.attention_dropout
        self.out_dropout = nn.Dropout(config.dropout)

        # Flash Attention
        self.use_flash = (
            getattr(config, "flash_attention", False) and _HAS_FLASH_ATTN
        )

        # Rotary embeddings (reference to shared module; may be None)
        self.rotary_emb = rotary_emb

        # ALiBi bias (reference to shared module; may be None)
        self.alibi_bias = alibi_bias

        # Sliding window attention
        self.window_size: Optional[int] = getattr(config, "sliding_window_size", None)

    # ------------------------------------------------------------------
    # KV head expansion
    # ------------------------------------------------------------------

    @staticmethod
    def _repeat_kv(x: Tensor, n_groups: int) -> Tensor:
        """Repeat KV heads to match the number of query heads.

        Args:
            x: ``(B, n_kv_heads, L, head_dim)``
            n_groups: Number of query heads per KV head.

        Returns:
            ``(B, n_kv_heads * n_groups, L, head_dim)``
        """
        if n_groups == 1:
            return x
        B, H_kv, L, D = x.shape
        # (B, H_kv, 1, L, D) -> (B, H_kv, n_groups, L, D) -> (B, H_kv*n_groups, L, D)
        return (
            x.unsqueeze(2)
            .expand(B, H_kv, n_groups, L, D)
            .reshape(B, H_kv * n_groups, L, D)
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: ``(B, L, D)``
            attention_mask: ``(B, L)`` with 1 for real tokens, 0 for padding.

        Returns:
            ``(B, L, D)``
        """
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(L)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads for GQA/MQA
        k = self._repeat_kv(k, self.n_groups)
        v = self._repeat_kv(v, self.n_groups)

        if self.use_flash:
            context = self._flash_attention(q, k, v, attention_mask)
        else:
            context = self._standard_attention(q, k, v, attention_mask)

        context = context.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_dropout(self.out_proj(context))

    def _standard_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """Standard scaled dot-product attention with optional sliding
        window mask and ALiBi bias.

        Args:
            q: ``(B, H, L, head_dim)``
            k: ``(B, H, L, head_dim)``
            v: ``(B, H, L, head_dim)``
            attention_mask: ``(B, L)`` or ``None``

        Returns:
            ``(B, H, L, head_dim)``
        """
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, L, L)

        # Sliding window mask
        if self.window_size is not None:
            L = q.shape[2]
            sw_mask = _build_sliding_window_mask(
                L, self.window_size, device=q.device, dtype=attn_weights.dtype
            )
            attn_weights = attn_weights + sw_mask

        # ALiBi bias
        if self.alibi_bias is not None:
            L = q.shape[2]
            alibi = self.alibi_bias(L)  # (1, H, L, L)
            attn_weights = attn_weights + alibi

        if attention_mask is not None:
            # (B, 1, 1, L) -- broadcast over heads and query positions
            mask = attention_mask[:, None, None, :].to(dtype=attn_weights.dtype)
            attn_weights = attn_weights + (1.0 - mask) * torch.finfo(attn_weights.dtype).min

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, v)  # (B, H, L, head_dim)

    def _flash_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """Flash Attention v2 forward pass.

        Flash Attention expects ``(B, L, H, head_dim)`` layout and returns
        the same.  Causal masking is handled natively via the ``causal``
        flag.  Padding masks are not directly supported by flash_attn, so
        we rely on the caller to handle padding externally or pass
        unpadded sequences.

        Args:
            q: ``(B, H, L, head_dim)``
            k: ``(B, H, L, head_dim)``
            v: ``(B, H, L, head_dim)``
            attention_mask: ``(B, L)`` or ``None`` (unused with flash attn)

        Returns:
            ``(B, H, L, head_dim)``
        """
        # flash_attn expects (B, L, H, D)
        q = q.transpose(1, 2)  # (B, L, H, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        dropout_p = self.attn_dropout_p if self.training else 0.0

        flash_kwargs: Dict[str, object] = {
            "dropout_p": dropout_p,
            "causal": False,  # encoder -- no causal mask
        }
        if self.window_size is not None:
            flash_kwargs["window_size"] = (
                self.window_size // 2,
                self.window_size // 2,
            )

        out = _flash_attn_func(q, k, v, **flash_kwargs)
        # back to (B, H, L, D)
        return out.transpose(1, 2)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU or SwiGLU activation.

    When ``config.activation == "swiglu"``, a :class:`SwiGLU` gated FFN is
    used instead of the standard two-layer MLP with GELU.  GELU remains the
    default for backward compatibility.

    Args:
        config: A :class:`ModelConfig` instance.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        activation = getattr(config, "activation", "gelu").lower()

        if activation == "swiglu":
            self.net = SwiGLU(config.d_model, config.d_ff, dropout=config.dropout)
        else:
            # Standard GELU FFN (default)
            self.fc1 = nn.Linear(config.d_model, config.d_ff)
            self.fc2 = nn.Linear(config.d_ff, config.d_model)
            self.dropout = nn.Dropout(config.dropout)
            self.net = None  # type: ignore[assignment]

    def forward(self, x: Tensor) -> Tensor:
        """Apply feed-forward: ``Linear -> GELU -> Linear -> Dropout``
        or ``SwiGLU``.

        Args:
            x: ``(B, L, D)``

        Returns:
            ``(B, L, D)``
        """
        if self.net is not None:
            return self.net(x)
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer.

    Supports pre-norm (``norm_type="prenorm"``) or post-norm
    (``norm_type="layernorm"`` or ``"postnorm"``).  The normalisation
    module itself is configurable: ``"layernorm"`` uses standard
    :class:`nn.LayerNorm` and ``"rmsnorm"`` uses the faster
    :class:`RMSNorm`.

    Args:
        config: A :class:`ModelConfig` instance.
        rotary_emb: Optional rotary embedding module shared across layers.
        alibi_bias: Optional ALiBi bias module shared across layers.
    """

    def __init__(
        self,
        config: ModelConfig,
        rotary_emb: Optional[nn.Module] = None,
        alibi_bias: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.pre_norm = config.norm_type.lower() == "prenorm"

        self.attn = MultiHeadSelfAttention(
            config, rotary_emb=rotary_emb, alibi_bias=alibi_bias,
        )
        self.ff = FeedForward(config)
        self.norm1 = _build_norm(config.d_model, config.norm_type)
        self.norm2 = _build_norm(config.d_model, config.norm_type)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the encoder layer.

        Args:
            x: ``(B, L, D)``
            attention_mask: ``(B, L)`` with 1 for real tokens.

        Returns:
            ``(B, L, D)``
        """
        if self.pre_norm:
            x = x + self.attn(self.norm1(x), attention_mask)
            x = x + self.ff(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, attention_mask))
            x = self.norm2(x + self.ff(x))
        return x


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------


class GenovaTransformer(nn.Module):
    """Stack of transformer encoder layers with genomic embeddings.

    Args:
        config: A :class:`ModelConfig` instance.
        embedding_type: Positional encoding style forwarded to
            :class:`GenomicEmbedding`.  One of ``"learned"``,
            ``"sinusoidal"``, ``"sinusoidal+learned"``, ``"rope"``,
            or ``"alibi"``.
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type: str = "learned",
    ) -> None:
        super().__init__()
        self.config = config

        self.embeddings = GenomicEmbedding(config, embedding_type=embedding_type)

        # Share rotary embedding across all layers (if used)
        rotary_emb = self.embeddings.rotary_emb  # None when not using RoPE

        # ALiBi bias (shared across layers; None when not using ALiBi)
        alibi_bias: Optional[nn.Module] = None
        if embedding_type == "alibi":
            from genova.models.embeddings import ALiBiPositionalBias
            alibi_bias = ALiBiPositionalBias(n_heads=config.n_heads)
        self.alibi_bias = alibi_bias

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config, rotary_emb=rotary_emb, alibi_bias=alibi_bias,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = _build_norm(config.d_model, config.norm_type)
        self.gradient_checkpointing = config.gradient_checkpointing

        self.apply(self._init_weights)

    # ------------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following BERT-style truncated normal."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Run the encoder.

        Args:
            input_ids: ``(B, L)`` integer tensor.
            attention_mask: ``(B, L)`` with 1s for real tokens.
            segment_ids: Optional ``(B, L)`` segment ids.
            output_hidden_states: If ``True``, return all layer outputs.

        Returns:
            Dict with keys:
                - ``last_hidden_state``: ``(B, L, D)``
                - ``hidden_states`` (optional): list of ``(B, L, D)``
        """
        x = self.embeddings(input_ids, segment_ids=segment_ids)

        all_hidden: List[Tensor] = []
        if output_hidden_states:
            all_hidden.append(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_reentrant=False
                )
            else:
                x = layer(x, attention_mask)
            if output_hidden_states:
                all_hidden.append(x)

        x = self.final_norm(x)

        out: Dict[str, Tensor] = {"last_hidden_state": x}
        if output_hidden_states:
            out["hidden_states"] = all_hidden
        return out


# ---------------------------------------------------------------------------
# MLM wrapper
# ---------------------------------------------------------------------------


class MLMHead(nn.Module):
    """Masked-language-model projection head.

    ``hidden -> dense -> GELU -> LayerNorm -> projection``

    Args:
        config: A :class:`ModelConfig` instance.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: ``(B, L, D)``

        Returns:
            ``(B, L, V)``
        """
        x = F.gelu(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class GenovaForMLM(nn.Module):
    """Genova transformer with a masked-language-modelling head.

    Provides a HuggingFace-style ``forward()`` that returns a dictionary
    with ``logits``, optional ``loss``, and optional ``hidden_states``.

    Args:
        config: A :class:`ModelConfig` instance.
        embedding_type: Positional encoding style.
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type: str = "learned",
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = GenovaTransformer(config, embedding_type=embedding_type)
        self.mlm_head = MLMHead(config)

        # Optionally tie embeddings
        if config.tie_word_embeddings:
            self.mlm_head.decoder.weight = self.transformer.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass with optional MLM loss computation.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` mask (1 = real, 0 = pad).
            labels: ``(B, L)`` ground-truth token ids for MLM.
                Positions set to ``-100`` are ignored in the loss.
            segment_ids: Optional segment ids.
            output_hidden_states: Whether to return intermediate states.

        Returns:
            Dict with keys ``logits`` (always), ``loss`` (when *labels*
            given), ``hidden_states`` (when requested).
        """
        enc_out = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )

        logits = self.mlm_head(enc_out["last_hidden_state"])

        result: Dict[str, Tensor] = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        if output_hidden_states and "hidden_states" in enc_out:
            result["hidden_states"] = enc_out["hidden_states"]

        return result
