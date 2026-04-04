"""Population genetics and variant-level analysis.

Provides population-aware genomic models that integrate ancestry context
and population-specific allele frequencies for improved variant
classification and clinical interpretation.
"""

from genova.population.frequency_encoder import AlleleFrequencyEncoder
from genova.population.population_model import (
    DEFAULT_POPULATION_LABELS,
    PopulationAwareEncoder,
    PopulationAwareVariantPredictor,
    PopulationEmbedding,
    VariantFrequencyEncoder,
)

__all__ = [
    "AlleleFrequencyEncoder",
    "DEFAULT_POPULATION_LABELS",
    "PopulationAwareEncoder",
    "PopulationAwareVariantPredictor",
    "PopulationEmbedding",
    "VariantFrequencyEncoder",
]
