"""Mamba / Selective State Space model for Genova.

Provides a pure-PyTorch fallback implementation of Mamba-style selective
state-space layers so the model can run without the ``mamba_ssm`` package.
When ``mamba_ssm`` *is* available the native CUDA kernels are used
automatically for better throughput on long sequences (100k-1M tokens).

The forward interface mirrors :class:`GenovaForMLM` -- returns a dict with
``logits``, optional ``loss``, and optional ``hidden_states``.

Example::

    from genova.utils.config import ModelConfig
    cfg = ModelConfig(arch="mamba", d_model=256, n_layers=8, vocab_size=4096)
    model = GenovaMambaForMLM(cfg)
    out = model(input_ids, labels=labels)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from genova.utils.config import ModelConfig
from genova.models.embeddings import GenomicEmbedding

# ---------------------------------------------------------------------------
# Try to import the native Mamba package; fall back to pure PyTorch.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm import Mamba as NativeMamba  # type: ignore[import-untyped]
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


# ---------------------------------------------------------------------------
# Pure-PyTorch selective state-space block
# ---------------------------------------------------------------------------


class SelectiveSSM(nn.Module):
    """Pure-PyTorch selective state-space layer (S6-style).

    This is a simplified but functional implementation of the selective
    scan mechanism from Mamba (Gu & Dao, 2023).  It uses a 1-D depthwise
    convolution followed by input-dependent discretisation of a diagonal
    state-space model.

    Args:
        d_model: Model / channel dimension.
        d_state: SSM state expansion factor.
        d_conv: Local convolution width.
        expand: Inner dimension expansion factor.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection: maps d_model -> 2 * d_inner (split into x and z)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1-D depthwise conv over the sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # Learnable log(A) initialised to HiPPO-like values
        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # Learnable D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # dt (delta) bias
        self.dt_bias = nn.Parameter(torch.randn(self.d_inner) * 0.1)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, L, D)``
        Returns:
            ``(B, L, D)``
        """
        B, L, D = x.shape

        # --- in projection ---
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # --- conv ---
        x_conv = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal trim
        x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, d_inner)

        # --- SSM parameter projection ---
        ssm_params = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        B_param = ssm_params[:, :, : self.d_state]           # (B, L, d_state)
        C_param = ssm_params[:, :, self.d_state : 2 * self.d_state]  # (B, L, d_state)
        dt_raw  = ssm_params[:, :, -1]                        # (B, L)

        # --- discretise ---
        A = -torch.exp(self.A_log)  # (d_inner, d_state) -- negative for stability
        dt = F.softplus(dt_raw.unsqueeze(-1) + self.dt_bias.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner)

        # --- selective scan (sequential, for correctness) ---
        y = self._selective_scan(x_conv, dt, A, B_param, C_param)

        # --- skip connection + gate ---
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        return self.out_proj(y)

    # ------------------------------------------------------------------

    def _selective_scan(
        self,
        x: Tensor,
        dt: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
    ) -> Tensor:
        """Run the discretised SSM scan over the sequence.

        Uses a chunked parallel scan when possible for efficiency on
        long sequences, but falls back to a sequential scan for
        correctness and simplicity.

        Args:
            x:  (B, L, d_inner)
            dt: (B, L, d_inner)
            A:  (d_inner, d_state)
            B:  (B, L, d_state)
            C:  (B, L, d_state)

        Returns:
            y: (B, L, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Precompute dA and dB for all timesteps
        # dA: (B, L, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB: (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # broadcast x over d_inner

        # Scale input
        x_db = x.unsqueeze(-1) * dB  # (B, L, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys: List[Tensor] = []
        for t in range(seq_len):
            h = h * dA[:, t] + x_db[:, t]  # (B, d_inner, d_state)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, d_inner)


# ---------------------------------------------------------------------------
# Mamba block (wraps SSM + norm + residual)
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Single Mamba encoder block: LayerNorm -> SSM -> residual.

    Uses the native ``mamba_ssm.Mamba`` kernel when available and falls
    back to :class:`SelectiveSSM` otherwise.

    Args:
        config: Model configuration.
        d_state: SSM latent state dimension.
        d_conv: Convolution width.
        expand: Inner expansion factor.
    """

    def __init__(
        self,
        config: ModelConfig,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, eps=1e-12)

        if _HAS_MAMBA_SSM:
            self.ssm = NativeMamba(
                d_model=config.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.ssm = SelectiveSSM(
                d_model=config.d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """``x`` has shape ``(B, L, D)``."""
        return x + self.dropout(self.ssm(self.norm(x)))


# ---------------------------------------------------------------------------
# Full Mamba encoder
# ---------------------------------------------------------------------------


class GenovaMamba(nn.Module):
    """Stack of Mamba blocks with genomic embeddings.

    Designed for long-range genomic sequences (100k - 1M tokens) where
    quadratic attention is prohibitive.

    Args:
        config: A :class:`ModelConfig` instance.
        embedding_type: Positional encoding style.
        d_state: SSM state dimension.
        d_conv: Convolution width.
        expand: Inner expansion factor.
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type: str = "learned",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.config = config

        self.embeddings = GenomicEmbedding(config, embedding_type=embedding_type)
        self.layers = nn.ModuleList(
            [
                MambaBlock(config, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.gradient_checkpointing = config.gradient_checkpointing

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Run the Mamba encoder.

        Args:
            input_ids: ``(B, L)`` token ids.
            attention_mask: ``(B, L)`` -- accepted for API compatibility
                but not used by SSM layers (they process all positions).
            segment_ids: Optional segment ids forwarded to embeddings.
            output_hidden_states: If ``True``, collect all layer outputs.

        Returns:
            Dict with ``last_hidden_state`` and optional ``hidden_states``.
        """
        x = self.embeddings(input_ids, segment_ids=segment_ids)

        all_hidden: List[Tensor] = []
        if output_hidden_states:
            all_hidden.append(x)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
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


class MambaMLMHead(nn.Module):
    """MLM head identical to the transformer variant."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.decoder = nn.Linear(config.d_model, config.vocab_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        x = F.gelu(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class GenovaMambaForMLM(nn.Module):
    """Genova Mamba model with a masked-language-modelling head.

    API-compatible with :class:`GenovaForMLM`.

    Args:
        config: A :class:`ModelConfig` instance.
        embedding_type: Positional encoding style.
        d_state: SSM state dimension.
        d_conv: Convolution width.
        expand: Inner expansion factor.
    """

    def __init__(
        self,
        config: ModelConfig,
        embedding_type: str = "learned",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = GenovaMamba(
            config,
            embedding_type=embedding_type,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mlm_head = MambaMLMHead(config)

        if config.tie_word_embeddings:
            self.mlm_head.decoder.weight = self.backbone.embeddings.token_embeddings.weight

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass with optional MLM loss.

        Returns:
            Dict with ``logits``, optional ``loss``, optional
            ``hidden_states``.
        """
        enc_out = self.backbone(
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
