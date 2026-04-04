"""Training loops, schedulers, and optimization utilities.

Public API
----------
Trainer:
    GenovaTrainer

Schedulers:
    CosineWithWarmup, LinearWithWarmup, create_scheduler

Distributed:
    setup_distributed, cleanup_distributed, is_main_process, reduce_metrics

FSDP:
    FSDPConfig, setup_fsdp, save_fsdp_checkpoint, load_fsdp_checkpoint

Active learning:
    ActiveLearner, ActiveLearningState, SamplingStrategy

Semi-supervised learning:
    SemiSupervisedTrainer, SemiSupervisedMetrics, pseudo_label,
    consistency_loss

Curriculum learning:
    CurriculumScheduler, CurriculumSampler

Entry point:
    run_training
"""

from genova.training.trainer import GenovaTrainer
from genova.training.scheduler import (
    CosineWithWarmup,
    LinearWithWarmup,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialDecay,
    create_scheduler,
)
from genova.training.ema import EMAModel
from genova.training.distillation import (
    DistillationLoss,
    FeatureDistillationLoss,
    DistillationTrainer,
)
from genova.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    reduce_metrics,
    broadcast_object,
)
from genova.training.fsdp import (
    FSDPConfig,
    setup_fsdp,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
)
from genova.training.active_learning import (
    ActiveLearner,
    ActiveLearningState,
    SamplingStrategy,
)
from genova.training.semi_supervised import (
    SemiSupervisedTrainer,
    SemiSupervisedMetrics,
    pseudo_label,
    consistency_loss,
)
from genova.training.train import run_training
from genova.training.differential_privacy import DPTrainer, SimpleRDPAccountant
from genova.training.curriculum import CurriculumScheduler, CurriculumSampler

__all__ = [
    # Trainer
    "GenovaTrainer",
    # Schedulers
    "CosineWithWarmup",
    "LinearWithWarmup",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialDecay",
    "create_scheduler",
    # EMA
    "EMAModel",
    # Knowledge Distillation
    "DistillationLoss",
    "FeatureDistillationLoss",
    "DistillationTrainer",
    # Distributed
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "reduce_metrics",
    "broadcast_object",
    # FSDP
    "FSDPConfig",
    "setup_fsdp",
    "save_fsdp_checkpoint",
    "load_fsdp_checkpoint",
    # Active learning
    "ActiveLearner",
    "ActiveLearningState",
    "SamplingStrategy",
    # Semi-supervised learning
    "SemiSupervisedTrainer",
    "SemiSupervisedMetrics",
    "pseudo_label",
    "consistency_loss",
    # Entry point
    "run_training",
    # Differential Privacy
    "DPTrainer",
    "SimpleRDPAccountant",
    # Curriculum learning
    "CurriculumScheduler",
    "CurriculumSampler",
]
