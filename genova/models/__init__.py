"""Model architectures for the Genova genomic foundation model.

Public API
----------
Embeddings:
    GenomicEmbedding, SinusoidalPositionalEncoding, ALiBiPositionalBias

Transformer:
    GenovaTransformer, GenovaForMLM, SwiGLU

Mamba / SSM:
    MambaBlock, GenovaMamba, GenovaMambaForMLM

Multi-task:
    MultiTaskHead, GenovaMultiTask

Pruning:
    compute_head_importance, prune_heads, prune_ffn, PruningSchedule

Factory:
    create_model, count_parameters, model_summary
"""

from genova.models.embeddings import (
    GenomicEmbedding,
    SinusoidalPositionalEncoding,
    ALiBiPositionalBias,
)
from genova.models.transformer import (
    GenovaTransformer,
    GenovaForMLM,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerEncoderLayer,
    MLMHead,
    SwiGLU,
)
from genova.models.mamba_model import (
    MambaBlock,
    SelectiveSSM,
    GenovaMamba,
    GenovaMambaForMLM,
)
from genova.models.multi_task import (
    MultiTaskHead,
    GenovaMultiTask,
    MLMTaskHead,
    GeneExpressionHead,
    MethylationHead,
)
from genova.models.model_factory import create_model, count_parameters, model_summary
from genova.models.export import export_onnx, verify_onnx, export_torchscript
from genova.models.pruning import (
    compute_head_importance,
    prune_heads,
    prune_ffn,
    PruningSchedule,
)

__all__ = [
    # Embeddings
    "GenomicEmbedding",
    "SinusoidalPositionalEncoding",
    "ALiBiPositionalBias",
    # Transformer
    "GenovaTransformer",
    "GenovaForMLM",
    "MultiHeadSelfAttention",
    "FeedForward",
    "TransformerEncoderLayer",
    "MLMHead",
    "SwiGLU",
    # Mamba / SSM
    "MambaBlock",
    "SelectiveSSM",
    "GenovaMamba",
    "GenovaMambaForMLM",
    # Multi-task
    "MultiTaskHead",
    "GenovaMultiTask",
    "MLMTaskHead",
    "GeneExpressionHead",
    "MethylationHead",
    # Pruning
    "compute_head_importance",
    "prune_heads",
    "prune_ffn",
    "PruningSchedule",
    # Factory
    "create_model",
    "count_parameters",
    "model_summary",
    # Export
    "export_onnx",
    "verify_onnx",
    "export_torchscript",
]
