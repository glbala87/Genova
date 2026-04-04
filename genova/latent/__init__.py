"""Latent space exploration and representation analysis.

Modules:
    embedding_analyzer: Extract, reduce, cluster, and annotate embeddings.
    visualization: Publication-quality latent space visualisations.
"""

from genova.latent.embedding_analyzer import (
    EmbeddingAnalyzer,
    RegionMetadata,
)
from genova.latent.visualization import (
    plot_umap,
    plot_clusters,
    plot_embedding_trajectory,
    plot_feature_space,
    plot_latent_summary,
)

__all__ = [
    "EmbeddingAnalyzer",
    "RegionMetadata",
    "plot_umap",
    "plot_clusters",
    "plot_embedding_trajectory",
    "plot_feature_space",
    "plot_latent_summary",
]
