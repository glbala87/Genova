"""Cross-species transfer learning for genomic models.

Implements strategies for pre-training on multi-species data and fine-tuning
on a target species (typically human).  Includes domain adaptation via
gradient reversal to learn species-invariant features.

Example::

    from genova.utils.config import ModelConfig
    from genova.evolutionary.multi_species import SpeciesConfig, MultiSpeciesEncoder
    from genova.evolutionary.transfer_learning import CrossSpeciesTransferLearner

    species = [
        SpeciesConfig(name="human", genome_path="/data/hg38.fa"),
        SpeciesConfig(name="mouse", genome_path="/data/mm10.fa"),
    ]
    cfg = ModelConfig(d_model=256, n_heads=4, n_layers=6, d_ff=1024,
                      vocab_size=4096)
    encoder = MultiSpeciesEncoder(cfg, species_configs=species)
    learner = CrossSpeciesTransferLearner(encoder, target_species="human")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.autograd import Function

from genova.evolutionary.multi_species import MultiSpeciesEncoder


# ---------------------------------------------------------------------------
# Gradient reversal layer (for domain-adversarial training)
# ---------------------------------------------------------------------------


class _GradientReversalFunction(Function):
    """Autograd function that reverses the gradient during backprop."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:  # type: ignore[override]
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper for gradient reversal.

    During the forward pass, the input is passed through unchanged.
    During backpropagation, the gradient is multiplied by ``-lambda_``.

    This is the core component of domain-adversarial training
    (Ganin et al., 2016) and is used here to encourage the backbone to
    learn species-invariant representations.

    Args:
        lambda_: Scaling factor for the reversed gradient.  A schedule
            that ramps from 0 to 1 during training often works best.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: Tensor) -> Tensor:
        """Pass *x* through, reversing gradient on backward."""
        return _GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Update the reversal strength."""
        self.lambda_ = lambda_

    def extra_repr(self) -> str:
        return f"lambda_={self.lambda_}"


# ---------------------------------------------------------------------------
# Species discriminator head
# ---------------------------------------------------------------------------


class SpeciesDiscriminator(nn.Module):
    """Small classifier that predicts species from pooled representations.

    Used in adversarial training: when paired with gradient reversal, it
    encourages the backbone to produce species-invariant features.

    Args:
        input_dim: Dimensionality of the pooled representation.
        num_species: Number of species classes.
        hidden_dim: Width of the hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_species: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_species),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict species logits from pooled embeddings.

        Args:
            x: ``(B, D)`` pooled representation.

        Returns:
            ``(B, num_species)`` logits.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Cross-species transfer learner
# ---------------------------------------------------------------------------


class CrossSpeciesTransferLearner(nn.Module):
    """Orchestrates cross-species pre-training and target-species fine-tuning.

    Supports three training modes:

    1. **Multi-species pre-training** (``mode="pretrain"``):
       Standard MLM-style pre-training on all species with optional
       adversarial species classification to encourage invariant features.

    2. **Domain-adversarial training** (``mode="adversarial"``):
       Uses gradient reversal on a species discriminator so that the
       backbone learns features that are informative for the main task
       but uninformative about species identity.

    3. **Fine-tuning** (``mode="finetune"``):
       Freezes or reduces LR for most backbone parameters and trains a
       task-specific head on target-species data only.

    Args:
        encoder: A :class:`MultiSpeciesEncoder` with backbone and species
            embeddings.
        target_species: Name of the target species for fine-tuning
            (e.g. ``"human"``).
        adversarial_weight: Weight of the adversarial loss relative to
            the main task loss.
        grl_lambda: Initial gradient-reversal lambda.
        num_finetune_layers: Number of backbone layers to unfreeze during
            fine-tuning (counted from the output end).  ``-1`` unfreezes all.
    """

    def __init__(
        self,
        encoder: MultiSpeciesEncoder,
        target_species: str = "human",
        adversarial_weight: float = 0.1,
        grl_lambda: float = 1.0,
        num_finetune_layers: int = -1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_species = target_species
        self.adversarial_weight = adversarial_weight
        self.num_finetune_layers = num_finetune_layers

        d_model = encoder.config.d_model
        num_species = len(encoder.species_configs)

        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_=grl_lambda)

        # Species discriminator
        self.species_discriminator = SpeciesDiscriminator(
            input_dim=d_model,
            num_species=num_species,
        )

        # Task head (placeholder linear; users typically replace this)
        self.task_head: Optional[nn.Module] = None

        self._mode: str = "pretrain"
        logger.info(
            "CrossSpeciesTransferLearner initialised: target={}, "
            "adversarial_weight={}, num_species={}",
            target_species,
            adversarial_weight,
            num_species,
        )

    # -- mode management -----------------------------------------------------

    @property
    def mode(self) -> str:
        """Current training mode."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch training mode.

        Args:
            mode: One of ``"pretrain"``, ``"adversarial"``, ``"finetune"``.
        """
        if mode not in ("pretrain", "adversarial", "finetune"):
            raise ValueError(
                f"mode must be 'pretrain', 'adversarial', or 'finetune'; got '{mode}'"
            )
        self._mode = mode
        if mode == "finetune":
            self._freeze_for_finetuning()
        else:
            self._unfreeze_all()
        logger.info("CrossSpeciesTransferLearner mode set to '{}'", mode)

    def set_task_head(self, head: nn.Module) -> None:
        """Attach a task-specific head for downstream fine-tuning.

        Args:
            head: Module that takes ``(B, L, D)`` or ``(B, D)`` and
                returns task predictions.
        """
        self.task_head = head
        logger.info("Task head attached: {}", type(head).__name__)

    # -- freezing strategies -------------------------------------------------

    def _freeze_for_finetuning(self) -> None:
        """Freeze backbone parameters except the last *num_finetune_layers*."""
        # Freeze everything first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze species embedding (always trainable)
        for param in self.encoder.species_embedding.parameters():
            param.requires_grad = True

        # Unfreeze shared projection if present
        if self.encoder.shared_projection is not None:
            for param in self.encoder.shared_projection.parameters():
                param.requires_grad = True

        # Unfreeze last N layers of the backbone
        if self.num_finetune_layers != 0:
            backbone_layers = self._get_backbone_layers()
            if self.num_finetune_layers == -1:
                layers_to_unfreeze = backbone_layers
            else:
                layers_to_unfreeze = backbone_layers[-self.num_finetune_layers :]
            for layer in layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True

        # Task head and discriminator always trainable
        if self.task_head is not None:
            for param in self.task_head.parameters():
                param.requires_grad = True
        for param in self.species_discriminator.parameters():
            param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            "Fine-tuning: {}/{} parameters trainable ({:.1f}%)",
            n_trainable,
            n_total,
            100.0 * n_trainable / max(n_total, 1),
        )

    def _unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def _get_backbone_layers(self) -> List[nn.Module]:
        """Extract individual layers from the backbone for selective freezing."""
        backbone = self.encoder.backbone
        # Try common attribute names for the layer list
        for attr in ("layers", "blocks", "encoder_layers", "layer"):
            if hasattr(backbone, attr):
                layers = getattr(backbone, attr)
                if isinstance(layers, nn.ModuleList):
                    return list(layers)
        # Fallback: treat the whole backbone as a single "layer"
        logger.warning(
            "Could not identify individual backbone layers; "
            "treating entire backbone as one unit"
        )
        return [backbone]

    # -- GRL schedule --------------------------------------------------------

    def update_grl_lambda(self, progress: float) -> None:
        """Update gradient reversal lambda based on training progress.

        Uses the schedule from Ganin et al. (2016)::

            lambda = 2 / (1 + exp(-10 * p)) - 1

        where *p* is the training progress in ``[0, 1]``.

        Args:
            progress: Fraction of training completed (0.0 to 1.0).
        """
        import math

        new_lambda = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        self.grl.set_lambda(new_lambda)

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        species_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass with mode-dependent behaviour.

        Args:
            input_ids: ``(B, L)`` token ids.
            species_ids: ``(B,)`` species identifiers.
            attention_mask: ``(B, L)`` attention mask.
            labels: Optional targets for the task head.

        Returns:
            Dict with keys depending on the mode:
                - ``"last_hidden_state"``: ``(B, L, D)`` hidden states.
                - ``"pooled"``: ``(B, D)`` pooled representations.
                - ``"adversarial_loss"``: scalar (adversarial / pretrain mode).
                - ``"species_logits"``: ``(B, num_species)`` logits.
                - ``"task_loss"``: scalar (if task_head and labels provided).
                - ``"total_loss"``: combined loss.
        """
        # Encode with species embeddings
        enc_out = self.encoder(
            input_ids,
            species_ids=species_ids,
            attention_mask=attention_mask,
            return_pooled=True,
        )
        hidden = enc_out["last_hidden_state"]  # (B, L, D)
        pooled = enc_out["pooled"]  # (B, D)

        result: Dict[str, Tensor] = {
            "last_hidden_state": hidden,
            "pooled": pooled,
        }

        # -- adversarial branch (pretrain / adversarial modes) ---------------
        if self._mode in ("pretrain", "adversarial"):
            reversed_pooled = self.grl(pooled)
            species_logits = self.species_discriminator(reversed_pooled)
            adv_loss = F.cross_entropy(species_logits, species_ids)
            result["species_logits"] = species_logits
            result["adversarial_loss"] = adv_loss

        # -- task branch (finetune mode or if task_head is set) ---------------
        total_loss = torch.tensor(0.0, device=input_ids.device)

        if self.task_head is not None and labels is not None:
            task_out = self.task_head(hidden)
            if task_out.dim() == 3 and labels.dim() == 1:
                # sequence-level classification: pool first
                task_out = task_out.mean(dim=1)
            if task_out.shape[-1] > 1 and labels.dtype in (torch.long, torch.int):
                task_loss = F.cross_entropy(
                    task_out.reshape(-1, task_out.size(-1)),
                    labels.reshape(-1),
                )
            else:
                task_loss = F.mse_loss(task_out.squeeze(-1), labels.float())
            result["task_loss"] = task_loss
            total_loss = total_loss + task_loss

        if "adversarial_loss" in result:
            total_loss = total_loss + self.adversarial_weight * result["adversarial_loss"]

        result["total_loss"] = total_loss
        return result

    # -- convenience methods -------------------------------------------------

    def get_target_species_id(self) -> int:
        """Return the integer id of the target species."""
        return self.encoder.species_to_id[self.target_species]

    def extract_invariant_features(
        self,
        input_ids: Tensor,
        species_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract species-invariant pooled features (no gradient reversal).

        Useful for downstream evaluation where species-invariant
        representations are desired.

        Args:
            input_ids: ``(B, L)`` token ids.
            species_ids: ``(B,)`` species identifiers.
            attention_mask: ``(B, L)`` attention mask.

        Returns:
            ``(B, D)`` species-invariant pooled embeddings.
        """
        with torch.no_grad():
            enc_out = self.encoder(
                input_ids,
                species_ids=species_ids,
                attention_mask=attention_mask,
                return_pooled=True,
            )
        return enc_out["pooled"]
