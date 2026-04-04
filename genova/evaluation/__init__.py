"""Evaluation metrics, benchmarking, and downstream task harnesses.

Public API
----------
Metrics:
    compute_metrics, mlm_accuracy, perplexity, auroc, auprc,
    expected_calibration_error, brier_score, pearson_correlation,
    spearman_correlation, mse

Variant effect prediction:
    VariantEffectPredictor, VariantClassifierHead, Variant,
    VariantPrediction, parse_vcf, predict_variants, FastaReader

Structural variant prediction:
    StructuralVariantPredictor, StructuralVariant, SVPrediction,
    CNVPrediction, parse_sv_vcf

TF binding site prediction:
    TFBindingPredictor, TFBindingPrediction, JASPARMotif,
    load_jaspar_motifs, scan_sequence_with_pwm

Chromatin state prediction:
    ChromatinStatePredictor, ChromatinPrediction

3D genome structure prediction:
    Genome3DPredictor, TADBoundaryPrediction, ContactMapPrediction,
    CompartmentPrediction

Enhancer-promoter interaction prediction:
    EPInteractionPredictor, EPInteractionResult, EPPair

Cross-validation:
    CrossValidator, CVResult, FoldResult

Statistical testing:
    bootstrap_ci, mcnemar_test, paired_ttest, wilcoxon_test,
    delong_test, cohens_d, cliffs_delta, bonferroni_correction,
    fdr_correction
"""

from genova.evaluation.metrics import (
    compute_metrics,
    mlm_accuracy,
    perplexity,
    auroc,
    auprc,
    expected_calibration_error,
    brier_score,
    pearson_correlation,
    spearman_correlation,
    mse,
)
from genova.evaluation.variant_predictor import (
    VariantEffectPredictor,
    VariantClassifierHead,
    Variant,
    VariantPrediction,
    parse_vcf,
    predict_variants,
    FastaReader,
)
from genova.evaluation.structural_variants import (
    StructuralVariantPredictor,
    StructuralVariant,
    SVPrediction,
    CNVPrediction,
    parse_sv_vcf,
)
from genova.evaluation.tf_binding import (
    TFBindingPredictor,
    TFBindingPrediction,
    JASPARMotif,
    load_jaspar_motifs,
    scan_sequence_with_pwm,
)
from genova.evaluation.chromatin import (
    ChromatinStatePredictor,
    ChromatinPrediction,
)
from genova.evaluation.genome_3d import (
    Genome3DPredictor,
    TADBoundaryPrediction,
    ContactMapPrediction,
    CompartmentPrediction,
)
from genova.evaluation.epi_interaction import (
    EPInteractionPredictor,
    EPInteractionResult,
    EPPair,
)
from genova.evaluation.cross_validation import (
    CrossValidator,
    CVResult,
    FoldResult,
)
from genova.evaluation.statistical_tests import (
    bootstrap_ci,
    mcnemar_test,
    paired_ttest,
    wilcoxon_test,
    delong_test,
    cohens_d,
    cliffs_delta,
    bonferroni_correction,
    fdr_correction,
)

from genova.evaluation.bias_audit import BiasAuditor, BiasReport

__all__ = [
    # Metrics
    "compute_metrics",
    "mlm_accuracy",
    "perplexity",
    "auroc",
    "auprc",
    "expected_calibration_error",
    "brier_score",
    "pearson_correlation",
    "spearman_correlation",
    "mse",
    # Variant effect prediction
    "VariantEffectPredictor",
    "VariantClassifierHead",
    "Variant",
    "VariantPrediction",
    "parse_vcf",
    "predict_variants",
    "FastaReader",
    # Structural variant prediction
    "StructuralVariantPredictor",
    "StructuralVariant",
    "SVPrediction",
    "CNVPrediction",
    "parse_sv_vcf",
    # TF binding site prediction
    "TFBindingPredictor",
    "TFBindingPrediction",
    "JASPARMotif",
    "load_jaspar_motifs",
    "scan_sequence_with_pwm",
    # Chromatin state prediction
    "ChromatinStatePredictor",
    "ChromatinPrediction",
    # 3D genome structure prediction
    "Genome3DPredictor",
    "TADBoundaryPrediction",
    "ContactMapPrediction",
    "CompartmentPrediction",
    # Enhancer-promoter interaction prediction
    "EPInteractionPredictor",
    "EPInteractionResult",
    "EPPair",
    # Cross-validation
    "CrossValidator",
    "CVResult",
    "FoldResult",
    # Statistical testing
    "bootstrap_ci",
    "mcnemar_test",
    "paired_ttest",
    "wilcoxon_test",
    "delong_test",
    "cohens_d",
    "cliffs_delta",
    "bonferroni_correction",
    "fdr_correction",
    # Bias audit
    "BiasAuditor",
    "BiasReport",
]
