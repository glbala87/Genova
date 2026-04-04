"""Explainability and interpretability methods for genomic predictions.

Public API
----------
SHAP:
    GenomicSHAPExplainer

Attention:
    AttentionAnalyzer

Integrated Gradients & SmoothGrad:
    IntegratedGradientsExplainer, SmoothGradExplainer

Visualization:
    plot_sequence_importance, plot_attribution, plot_variant_effect,
    plot_motif_contributions, plot_attention_heatmap
"""

from genova.explainability.shap_explainer import GenomicSHAPExplainer
from genova.explainability.attention_analysis import AttentionAnalyzer
from genova.explainability.integrated_gradients import (
    IntegratedGradientsExplainer,
    SmoothGradExplainer,
)
from genova.explainability.visualization import (
    plot_sequence_importance,
    plot_attribution,
    plot_variant_effect,
    plot_motif_contributions,
    plot_attention_heatmap,
)

__all__ = [
    "GenomicSHAPExplainer",
    "AttentionAnalyzer",
    "IntegratedGradientsExplainer",
    "SmoothGradExplainer",
    "plot_sequence_importance",
    "plot_attribution",
    "plot_variant_effect",
    "plot_motif_contributions",
    "plot_attention_heatmap",
]
