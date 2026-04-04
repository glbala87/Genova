"""Standardised benchmark suites for genomic model evaluation."""

from genova.benchmark.benchmark_suite import BenchmarkSuite
from genova.benchmark.comparison import ModelComparator, ModelResult
from genova.benchmark.tasks import (
    BenchmarkDataset,
    BenchmarkTask,
    EnhancerClassificationTask,
    PromoterPredictionTask,
    SpliceSiteTask,
    VariantEffectTask,
    TASK_REGISTRY,
    get_task,
)
from genova.benchmark.standard_benchmarks import (
    StandardBenchmark,
    LinearProbe,
    GeneFindingBenchmark,
    PromoterDetectionBenchmark,
    SpliceSiteBenchmark,
    EnhancerPromoterBenchmark,
    ChromatinAccessibilityBenchmark,
    HistoneModificationBenchmark,
    RegulatoryElementBenchmark,
    STANDARD_BENCHMARK_REGISTRY,
    run_standard_benchmarks,
)

__all__ = [
    "BenchmarkSuite",
    "ModelComparator",
    "ModelResult",
    "BenchmarkDataset",
    "BenchmarkTask",
    "EnhancerClassificationTask",
    "PromoterPredictionTask",
    "SpliceSiteTask",
    "VariantEffectTask",
    "TASK_REGISTRY",
    "get_task",
    # Standard benchmarks
    "StandardBenchmark",
    "LinearProbe",
    "GeneFindingBenchmark",
    "PromoterDetectionBenchmark",
    "SpliceSiteBenchmark",
    "EnhancerPromoterBenchmark",
    "ChromatinAccessibilityBenchmark",
    "HistoneModificationBenchmark",
    "RegulatoryElementBenchmark",
    "STANDARD_BENCHMARK_REGISTRY",
    "run_standard_benchmarks",
]
