"""Genova: A genomics foundation model framework.

Genova provides tools for training, evaluating, and deploying large-scale
genomic language models. It supports DNA/RNA sequence modeling with
transformer-based architectures, multi-omics integration, motif discovery,
evolutionary analysis, and interpretability.

Example usage::

    from genova.utils.config import GenovaConfig

    config = GenovaConfig.from_yaml("configs/default.yaml")
"""

__version__ = "1.0.0"
__author__ = "Genova Team"
__all__ = ["__version__"]
