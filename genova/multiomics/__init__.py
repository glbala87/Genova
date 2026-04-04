"""Multi-omics data integration (epigenomics, transcriptomics, etc.).

Provides modality-specific encoders, attention-based data fusion, and
end-to-end multi-omics models that integrate DNA sequence, methylation,
and RNA-seq data for genomic analysis.
"""

from genova.multiomics.data_fusion import MultiOmicsAligner, OmicsDataFusion
from genova.multiomics.multiomics_model import (
    MODALITY_DNA,
    MODALITY_METHYLATION,
    MODALITY_RNASEQ,
    MethylationEncoder,
    ModalityProjection,
    MultiOmicsEncoder,
    MultiOmicsGenovaModel,
    RNASeqEncoder,
)
from genova.multiomics.ont_methylation import ONTMethylationProcessor

__all__ = [
    "MODALITY_DNA",
    "MODALITY_METHYLATION",
    "MODALITY_RNASEQ",
    "MethylationEncoder",
    "ModalityProjection",
    "MultiOmicsAligner",
    "MultiOmicsEncoder",
    "MultiOmicsGenovaModel",
    "OmicsDataFusion",
    "ONTMethylationProcessor",
    "RNASeqEncoder",
]
