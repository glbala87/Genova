"""Contrastive learning objectives for genomic representations.

Public API
----------
Augmentations:
    GenomicAugmenter, reverse_complement, random_mutation,
    random_mask, subsequence_crop, window_shuffle

Model:
    ContrastiveGenovaModel, ProjectionHead, NTXentLoss

Trainer:
    ContrastiveTrainer, TrainerConfig
"""

from genova.contrastive.augmentations import (
    GenomicAugmenter,
    reverse_complement,
    random_mutation,
    random_mask,
    subsequence_crop,
    window_shuffle,
)
from genova.contrastive.contrastive_model import (
    ContrastiveGenovaModel,
    ProjectionHead,
    NTXentLoss,
)
from genova.contrastive.contrastive_trainer import (
    ContrastiveTrainer,
    TrainerConfig,
)

__all__ = [
    # Augmentations
    "GenomicAugmenter",
    "reverse_complement",
    "random_mutation",
    "random_mask",
    "subsequence_crop",
    "window_shuffle",
    # Model
    "ContrastiveGenovaModel",
    "ProjectionHead",
    "NTXentLoss",
    # Trainer
    "ContrastiveTrainer",
    "TrainerConfig",
]
