"""Multi-task prediction heads for Genova.

Provides a config-driven wrapper that attaches one or more downstream
task heads to a shared encoder backbone:

- **MLM** -- masked language modelling (classification over vocab).
- **gene_expression** -- gene expression prediction (regression).
- **methylation** -- methylation beta-value prediction (regression, 0-1).

Loss balancing supports fixed weights or learned uncertainty weighting
(Kendall et al., 2018).

Example::

    from genova.utils.config import ModelConfig
    tasks = {
        "mlm": {"enabled": True, "weight": 1.0},
        "gene_expression": {"enabled": True, "weight": 0.5, "num_targets": 200},
        "methylation": {"enabled": True, "weight": 0.5, "num_targets": 1},
    }
    cfg = ModelConfig(d_model=256, vocab_size=4096, n_layers=6, n_heads=4, d_ff=1024)
    model = GenovaMultiTask(cfg, task_configs=tasks, backbone="transformer")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from genova.utils.config import ModelConfig


# ---------------------------------------------------------------------------
# Individual task heads
# ---------------------------------------------------------------------------


class MLMTaskHead(nn.Module):
    """Masked-language-modelling head (dense -> GELU -> LN -> projection)."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """``(B, L, D) -> (B, L, V)``"""
        x = F.gelu(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class GeneExpressionHead(nn.Module):
    """Regression head for gene expression prediction.

    Operates on the ``[CLS]`` token (position 0) by default.

    Args:
        d_model: Hidden size.
        num_targets: Number of regression targets (e.g. number of genes).
        pool: Pooling strategy -- ``"cls"`` or ``"mean"``.
    """

    def __init__(
        self,
        d_model: int,
        num_targets: int = 1,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Linear(d_model, num_targets),
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            hidden_states: ``(B, L, D)``
            attention_mask: ``(B, L)`` used for mean pooling.

        Returns:
            ``(B, num_targets)``
        """
        if self.pool == "cls":
            pooled = hidden_states[:, 0]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        return self.head(pooled)


class MethylationHead(nn.Module):
    """Regression head for methylation beta values (output in [0, 1]).

    Uses a sigmoid activation on the final output.

    Args:
        d_model: Hidden size.
        num_targets: Number of CpG sites to predict.
        pool: Pooling strategy.
    """

    def __init__(
        self,
        d_model: int,
        num_targets: int = 1,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Linear(d_model, num_targets),
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns values in ``[0, 1]`` via sigmoid."""
        if self.pool == "cls":
            pooled = hidden_states[:, 0]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                pooled = hidden_states.mean(dim=1)
        return torch.sigmoid(self.head(pooled))


# ---------------------------------------------------------------------------
# Multi-task head container
# ---------------------------------------------------------------------------


class MultiTaskHead(nn.Module):
    """Container for multiple prediction heads with loss balancing.

    Args:
        config: Model configuration.
        task_configs: Dictionary mapping task name to its settings.
            Recognised keys per task:

            - ``enabled`` (bool): whether the head is active.
            - ``weight`` (float): fixed loss weight (default 1.0).
            - ``num_targets`` (int): for regression heads.
            - ``pool`` (str): pooling strategy for regression heads.

        uncertainty_weighting: If ``True``, learn per-task log-variance
            parameters and weight losses automatically (Kendall et al.,
            2018).  Overrides fixed weights.
    """

    def __init__(
        self,
        config: ModelConfig,
        task_configs: Dict[str, Dict[str, Any]],
        uncertainty_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.task_configs = task_configs
        self.uncertainty_weighting = uncertainty_weighting
        self.heads = nn.ModuleDict()
        self._fixed_weights: Dict[str, float] = {}

        for name, tcfg in task_configs.items():
            if not tcfg.get("enabled", True):
                continue

            self._fixed_weights[name] = float(tcfg.get("weight", 1.0))

            if name == "mlm":
                self.heads[name] = MLMTaskHead(config.d_model, config.vocab_size)
            elif name == "gene_expression":
                self.heads[name] = GeneExpressionHead(
                    d_model=config.d_model,
                    num_targets=int(tcfg.get("num_targets", 1)),
                    pool=tcfg.get("pool", "cls"),
                )
            elif name == "methylation":
                self.heads[name] = MethylationHead(
                    d_model=config.d_model,
                    num_targets=int(tcfg.get("num_targets", 1)),
                    pool=tcfg.get("pool", "cls"),
                )
            else:
                raise ValueError(f"Unknown task head: {name!r}")

        # Learnable log-variance for uncertainty weighting
        if uncertainty_weighting and len(self.heads) > 0:
            self.log_vars = nn.ParameterDict(
                {name: nn.Parameter(torch.zeros(1)) for name in self.heads}
            )

    @property
    def active_tasks(self) -> List[str]:
        """Names of currently enabled task heads."""
        return list(self.heads.keys())

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """Run all active heads and optionally compute losses.

        Args:
            hidden_states: ``(B, L, D)`` from the backbone encoder.
            attention_mask: ``(B, L)`` mask.
            labels: Dict mapping task name to label tensor:
                - ``"mlm"``: ``(B, L)`` token ids (``-100`` ignored).
                - ``"gene_expression"``: ``(B, num_targets)`` float.
                - ``"methylation"``: ``(B, num_targets)`` float in [0,1].

        Returns:
            Dict with per-task ``{name}_logits``, per-task ``{name}_loss``
            (when labels given), and ``total_loss`` (scalar).
        """
        result: Dict[str, Tensor] = {}
        losses: Dict[str, Tensor] = {}

        for name, head in self.heads.items():
            if name == "mlm":
                logits = head(hidden_states)
                result[f"{name}_logits"] = logits
                if labels is not None and name in labels:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        labels[name].view(-1),
                        ignore_index=-100,
                    )
                    losses[name] = loss
                    result[f"{name}_loss"] = loss
            else:
                preds = head(hidden_states, attention_mask)
                result[f"{name}_logits"] = preds
                if labels is not None and name in labels:
                    if name == "methylation":
                        loss = F.mse_loss(preds, labels[name])
                    else:
                        loss = F.mse_loss(preds, labels[name])
                    losses[name] = loss
                    result[f"{name}_loss"] = loss

        # --- loss balancing ---
        if losses:
            if self.uncertainty_weighting and hasattr(self, "log_vars"):
                total = torch.tensor(0.0, device=hidden_states.device)
                for name, loss in losses.items():
                    precision = torch.exp(-self.log_vars[name])
                    total = total + precision * loss + self.log_vars[name]
                result["total_loss"] = total.squeeze()
            else:
                total = torch.tensor(0.0, device=hidden_states.device)
                for name, loss in losses.items():
                    total = total + self._fixed_weights.get(name, 1.0) * loss
                result["total_loss"] = total

        return result


# ---------------------------------------------------------------------------
# End-to-end multi-task model
# ---------------------------------------------------------------------------


class GenovaMultiTask(nn.Module):
    """Multi-task Genova model: shared backbone + task-specific heads.

    Args:
        config: Model configuration.
        task_configs: Task head specifications (see :class:`MultiTaskHead`).
        backbone: ``"transformer"`` or ``"mamba"``.
        uncertainty_weighting: Use learned loss weights.
        embedding_type: Forwarded to the backbone.
    """

    def __init__(
        self,
        config: ModelConfig,
        task_configs: Dict[str, Dict[str, Any]],
        backbone: str = "transformer",
        uncertainty_weighting: bool = False,
        embedding_type: str = "learned",
    ) -> None:
        super().__init__()
        self.config = config

        # Build backbone
        if backbone == "transformer":
            from genova.models.transformer import GenovaTransformer
            self.backbone = GenovaTransformer(config, embedding_type=embedding_type)
        elif backbone == "mamba":
            from genova.models.mamba_model import GenovaMamba
            self.backbone = GenovaMamba(config, embedding_type=embedding_type)
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}")

        self.task_heads = MultiTaskHead(
            config,
            task_configs,
            uncertainty_weighting=uncertainty_weighting,
        )

        # Optionally tie MLM embeddings
        if (
            config.tie_word_embeddings
            and "mlm" in self.task_heads.heads
            and hasattr(self.task_heads.heads["mlm"], "decoder")
        ):
            self.task_heads.heads["mlm"].decoder.weight = (
                self.backbone.embeddings.token_embeddings.weight
            )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        segment_ids: Optional[Tensor] = None,
        labels: Optional[Dict[str, Tensor]] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward pass through backbone and all active task heads.

        Args:
            input_ids: ``(B, L)``
            attention_mask: ``(B, L)``
            segment_ids: Optional ``(B, L)``
            labels: Dict of task-name -> label tensor.
            output_hidden_states: Return intermediate hidden states.

        Returns:
            Dict with per-task logits/losses, ``total_loss``, and
            optional ``hidden_states``.
        """
        enc_out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            output_hidden_states=output_hidden_states,
        )

        hidden = enc_out["last_hidden_state"]
        result = self.task_heads(hidden, attention_mask=attention_mask, labels=labels)

        if output_hidden_states and "hidden_states" in enc_out:
            result["hidden_states"] = enc_out["hidden_states"]

        return result
