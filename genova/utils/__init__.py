"""Utility modules for the Genova framework.

Provides configuration management, logging, device handling,
reproducibility utilities, and publication figure/table generation.
"""

from genova.utils.config import GenovaConfig
from genova.utils.device import get_device, DeviceManager
from genova.utils.logging import setup_logging
from genova.utils.reproducibility import set_seed, enable_deterministic_mode
from genova.utils.paper_figures import (
    create_all_figures,
    create_architecture_figure,
    create_attention_figure,
    create_benchmark_figure,
    create_embedding_figure,
    create_variant_landscape_figure,
)
from genova.utils.paper_tables import (
    export_latex,
    generate_ablation_table,
    generate_comparison_table,
    generate_compute_table,
)
from genova.utils.model_card import ModelCard

__all__ = [
    "GenovaConfig",
    "get_device",
    "DeviceManager",
    "setup_logging",
    "set_seed",
    "enable_deterministic_mode",
    "create_all_figures",
    "create_architecture_figure",
    "create_attention_figure",
    "create_benchmark_figure",
    "create_embedding_figure",
    "create_variant_landscape_figure",
    "export_latex",
    "generate_ablation_table",
    "generate_comparison_table",
    "generate_compute_table",
    "ModelCard",
]
